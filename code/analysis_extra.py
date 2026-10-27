"""Extra analyses from saved rank files (top-10 ids + per-query metrics). Writes runs/extra_analysis.json."""
import json, os, numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda *p: os.path.join(R, *p)
titles = json.load(open(J("data/docs_topiocqa/titles.json"), encoding="utf8"))
cs = np.cumsum(np.load(J("data/docs_topiocqa/count.npy")))
doc = lambda p: int(np.searchsorted(cs, int(p), "right"))
convs = json.load(open(J("data/topiocqa_dev.json"), encoding="utf8"))
turn = {f"{c['conv_id']}|{i}": (c, i) for c in convs for i in range(len(c["turns"]))}

def load(run):
    d = np.load(J("runs", run, "rank_topiocqa_dev.npy"), allow_pickle=True).item()
    return d["keys"], d["ids"], d["per_q"]

MODELS = {"concat": ["F_concat_s42", "F_concat_s43", "F_concat_s44"], "prism": ["F_prism_s42", "F_prism_s43", "F_prism_s44"],
          "shared": ["H_shared_s42", "H_shared_s43", "H_shared_s44"], "hint": ["H_hier_s42", "H_hier_s43", "H_hier_s44"],
          "nodoc": ["H_nodocloss_s42", "H_nodocloss_s43", "H_nodocloss_s44"]}
out = {}

# 1) Hit@k (k=1..10), mean over seeds
out["hit_at_k"] = {}
for m, runs in MODELS.items():
    acc = []
    for r in runs:
        keys, ids, _ = load(r)
        hits = np.array([[ids[j][:k].tolist().count(turn[kk][0]["turns"][turn[kk][1]]["gold"][0]) > 0 or
                          any(g in ids[j][:k].tolist() for g in turn[kk][0]["turns"][turn[kk][1]]["gold"]) for k in range(1, 11)]
                         for j, kk in enumerate(keys)])
        acc.append(hits.mean(0) * 100)
    out["hit_at_k"][m] = np.round(np.mean(acc, 0), 2).tolist()

# 2) error types of the top-1 passage (seed 42)
def err_types(run):
    keys, ids, _ = load(run)
    c = dict(correct=0, same_article=0, earlier_topic=0, unrelated=0)
    for j, kk in enumerate(keys):
        cv, i = turn[kk]; t = cv["turns"][i]; top = int(ids[j][0])
        if top in t["gold"]: c["correct"] += 1; continue
        gd = {doc(g) for g in t["gold"]}; td = doc(top)
        earlier = {x["topic"] for x in cv["turns"][:i]}
        if td in gd: c["same_article"] += 1
        elif titles[td] in earlier: c["earlier_topic"] += 1
        else: c["unrelated"] += 1
    c["n"] = len(keys); return c
out["errors"] = {m: err_types(MODELS[m][0]) for m in ["concat", "prism", "shared", "hint"]}

# 3) gold article in top-10 (article-level hit) vs passage top-1, seed 42
def art_hit(run):
    keys, ids, _ = load(run)
    a = np.mean([any(doc(p) in {doc(g) for g in turn[kk][0]["turns"][turn[kk][1]]["gold"]} for p in ids[j][:10]) for j, kk in enumerate(keys)])
    return round(100 * a, 2)
out["article_in_top10"] = {m: art_hit(MODELS[m][0]) for m in ["concat", "shared", "hint"]}

# 4) per-conversation win/tie/loss of HINT vs concat (MRR averaged per conversation, seed-averaged)
def conv_mrr(runs):
    acc = {}
    for r in runs:
        keys, _, pq = load(r)
        for kk, v in zip(keys, pq["MRR"]):
            acc.setdefault(kk.split("|")[0], []).append(v)
    return {c: np.mean(v) for c, v in acc.items()}
cm_h, cm_c = conv_mrr(MODELS["hint"]), conv_mrr(MODELS["concat"])
d = np.array([cm_h[c] - cm_c[c] for c in cm_h])
out["conv_wtl"] = dict(win=int((d > 0.01).sum()), tie=int((np.abs(d) <= 0.01).sum()), loss=int((d < -0.01).sum()), n=len(d),
                       diffs=np.round(np.sort(d) * 100, 2).tolist())

# 5) MRR by number of turns since the last topic switch, and by number of distinct topics so far
def bucket_mrr(runs, fn, labels):
    acc = {l: [] for l in labels}
    for r in runs:
        keys, _, pq = load(r)
        for kk, v in zip(keys, pq["MRR"]):
            cv, i = turn[kk]; b = fn(cv, i)
            if b in acc: acc[b].append(v)
    return {l: round(100 * float(np.mean(v)), 2) for l, v in acc.items()}, {l: len(v) // len(runs) for l, v in acc.items()}
def since_switch(cv, i):
    j = i
    while j > 0 and cv["turns"][j]["topic"] == cv["turns"][j - 1]["topic"]: j -= 1
    s = i - j
    return "0" if s == 0 else "1" if s == 1 else "2" if s == 2 else "3+"
def n_topics(cv, i):
    n = len({t["topic"] for t in cv["turns"][:i + 1]})
    return "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4+"
out["since_switch"] = {m: bucket_mrr(MODELS[m], since_switch, ["0", "1", "2", "3+"]) for m in ["concat", "shared", "hint"]}
out["n_topics"] = {m: bucket_mrr(MODELS[m], n_topics, ["1", "2", "3", "4+"]) for m in ["concat", "shared", "hint"]}

# 6) case studies (seed 42): HINT top-1 correct, concat not in top-10
kh, ih, _ = load("H_hier_s42"); kc, ic, _ = load("F_concat_s42"); pc = {k: j for j, k in enumerate(kc)}
cases = []
for j, kk in enumerate(kh):
    cv, i = turn[kk]; t = cv["turns"][i]
    if int(ih[j][0]) in t["gold"] and not any(int(p) in t["gold"] for p in ic[pc[kk]][:10]) and i >= 3:
        cases.append(dict(key=kk, q=t["q"], shift=t["shift"], gold=t["topic"], section=t["section"],
                          prev=[x["q"] for x in cv["turns"][max(0, i - 2):i]], prev_topic=cv["turns"][i - 1]["topic"],
                          concat_top=[titles[doc(p)] for p in ic[pc[kk]][:3]], hint_top=[titles[doc(p)] for p in ih[j][:3]]))
out["n_cases"] = len(cases); out["cases"] = cases[:40]
json.dump(out, open(J("runs/extra_analysis.json"), "w", encoding="utf8"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k not in ("cases", "conv_wtl")}, ensure_ascii=False))
print({k: v for k, v in out["conv_wtl"].items() if k != "diffs"}, out["n_cases"])

