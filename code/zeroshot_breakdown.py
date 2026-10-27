"""Zero-shot TopiOCQA baselines on the rebuilt full index, with the same subgroup breakdowns and article accuracy
(Art@1) as the trained models in runs/ldm_stats.json. Unlike prepare_aux.py zeroshot, this writes no training files.
Usage: python code/zeroshot_breakdown.py  -> runs/zeroshot_topiocqa_breakdown.json"""
import os, glob, json, numpy as np, torch
from retrieval import Index, metrics, summarize, encode_queries, ROOT
from prepare_aux import key, query_variants

DD = os.path.join(ROOT, "data")
convs = json.load(open(os.path.join(DD, "topiocqa_dev.json")))
rw = json.load(open(os.path.join(DD, "topiocqa_dev.llmrw.json")))
items = [(c, i) for c in convs for i, t in enumerate(c["turns"]) if t["gold"]]
keys = [key(c, i) for c, i in items]
tmap = {key(c, i): c["turns"][i] for c, i in items}
names = ["raw", "concat", "llm_rewrite"]
texts = {n: [query_variants(c, i, rw)[n] for c, i in items] for n in names}

idx = Index("topiocqa")
d = os.path.join(DD, "index_topiocqa")
ids_ = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(d, "ids_*.npy")))])
doc = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(d, "docid_*.npy")))])
p2d = np.full(int(ids_.max()) + 1, -1, dtype=np.int64); p2d[ids_] = doc

Q = torch.cat([encode_queries(texts[n]) for n in names])
print("searching", tuple(Q.shape), flush=True)
_, I = idx.search(Q, k=100)
turn_no = np.array([i + 1 for _, i in items]); sh = np.array([tmap[k]["shift"] for k in keys])
golds = [idx.pid(tmap[k]["gold"]) for k in keys]
gd = np.array([p2d[tmap[k]["gold"][0]] for k in keys])
res = {}
for j, n in enumerate(names):
    R = I[j * len(keys):(j + 1) * len(keys)]
    m = metrics(R, golds)
    r = {"all": summarize(m), "article_acc@1": round(float((p2d[R[:, 0]] == gd).mean() * 100), 2)}
    for lo, hi in [(1, 3), (4, 6), (7, 10), (11, 99)]:
        sel = (turn_no >= lo) & (turn_no <= hi)
        r[f"turn{lo}-{hi}"] = summarize({k: v[sel] for k, v in m.items()})
    for v, nm in [(0, "stay"), (1, "new_topic"), (2, "return")]:
        sel = (sh == v) & (turn_no > 1)
        r[nm] = summarize({k: x[sel] for k, x in m.items()})
    res[n] = r
    print(n, r["all"], r["article_acc@1"], flush=True)
json.dump(res, open(os.path.join(ROOT, "runs", "zeroshot_topiocqa_breakdown.json"), "w"), indent=1)
