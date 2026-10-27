"""Document store for training the latent-document model: every passage of every document that contains a
gold passage or one of the top-50 hard negatives (train + eval split), grouped by document.
Also writes the teacher embeddings (frozen-encoder embedding of the stand-alone rewrite of each turn).
Usage: python build_dstore.py topiocqa|orquac"""
import os, sys, json, numpy as np, torch
from retrieval import Index, ROOT, encode_queries

data = sys.argv[1]
DD = os.path.join(ROOT, "data")
SPLITS = {"topiocqa": ["train", "dev"], "orquac": ["train", "test"]}[data]
idx = Index(data)
dsh = [np.load(os.path.join(DD, f"index_{data}", f"docid_{os.path.basename(f)[4:9]}.npy")) for f in idx.embs]
pid_all = np.concatenate(idx.ids); doc_all = np.concatenate(dsh)
p2row = np.full(int(pid_all.max()) + 1, -1, dtype=np.int64); p2row[pid_all] = np.arange(len(pid_all))

need = set()
for s in SPLITS:
    for c in json.load(open(os.path.join(DD, f"{data}_{s}.json"))):
        for t in c["turns"]:
            need.update(idx.pid(t["gold"]))
    for r in json.load(open(os.path.join(DD, f"hn_{data}_{s}.json"))).values():
        need.update(r[:50])
need = np.array(sorted(need)); need = need[need < len(p2row)]
docs = np.unique(doc_all[p2row[need][p2row[need] >= 0]])
rows = np.nonzero(np.isin(doc_all, docs))[0]
print("passages", len(need), "docs", len(docs), "rows", len(rows), flush=True)
E = np.zeros((len(rows), 768), dtype=np.float16)
off = 0
for f, ids in zip(idx.embs, idx.ids):
    lo, hi = off, off + len(ids); off = hi
    sel = rows[(rows >= lo) & (rows < hi)]
    if len(sel):
        E[np.searchsorted(rows, sel)] = np.load(f, mmap_mode="r")[sel - lo]
np.save(os.path.join(DD, f"dstore_{data}.npy"), E)
np.savez(os.path.join(DD, f"dstore_{data}_meta.npz"), pids=pid_all[rows], doc=doc_all[rows])

# teacher: stand-alone rewrite (human for OR-QuAC, offline LLM for TopiOCQA) through the frozen query tower
for s in SPLITS:
    convs = json.load(open(os.path.join(DD, f"{data}_{s}.json")))
    p = os.path.join(DD, f"{data}_{s}.llmrw.json"); rw = json.load(open(p)) if os.path.exists(p) else {}
    keys, texts = [], []
    for c in convs:
        for i, t in enumerate(c["turns"]):
            r = t.get("rewrite") or rw.get(f"{c['conv_id']}|{i}")
            if r:
                keys.append(f"{c['conv_id']}|{i}"); texts.append(r)
    T = encode_queries(texts).numpy().astype(np.float16)
    np.save(os.path.join(DD, f"teacher_{data}_{s}.npy"), dict(zip(keys, T)))
    print("teacher", s, len(keys), flush=True)
