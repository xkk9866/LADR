"""Mechanism analysis for HINT (TopiOCQA dev):
(1) cross-turn stability of q_G vs q_F by switch type,
(2) document-level discrimination (gold article retrieval over all articles) with q_G / q_F / concat q,
(3) within-document discrimination (rank of the gold section among the article's sections) with q_F / q_G / concat q.
usage: python analyze_hint.py <hint_run> <concat_run>"""
import sys, json, numpy as np, torch
hint, conc = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], "--kind", "concat", "--two_level", "--name", hint, "--eval_only"]
import train as T
from retrieval import DocIndex, _q8, _intmm

T.DOCS = DocIndex("topiocqa")
convs, side = T.load_side("topiocqa", "dev")


def encode(run, two):
    T.args.two_level = two
    m = T.P.build("concat", two_level=two).cuda()
    m.load_state_dict(torch.load(f"../runs/{run}/model.pt", map_location="cuda"))
    Q, _ = T.encode_split(m, "topiocqa", convs, side)
    G = dict(T.QG)
    del m; torch.cuda.empty_cache()
    return Q, G


QF, QG = encode(hint, True)
QC, _ = encode(conc, False)
keys = [T.key(c, i) for c in convs for i, t in enumerate(c["turns"]) if t["gold"]]
tm = {T.key(c, i): (c, i) for c in convs for i, _ in enumerate(c["turns"])}
res = {}

# (1) cross-turn stability
stab = {s: {"G": [], "F": [], "C": []} for s in ("stay", "new", "return")}
names = {0: "stay", 1: "new", 2: "return"}
for k in keys:
    c, i = tm[k]
    if i == 0: continue
    kp = T.key(c, i - 1)
    s = names[c["turns"][i]["shift"]]
    stab[s]["G"].append(float(QG[k] @ QG[kp])); stab[s]["F"].append(float(QF[k] @ QF[kp])); stab[s]["C"].append(float(QC[k] @ QC[kp]))
res["stability"] = {s: {v: round(float(np.mean(x)), 4) for v, x in d.items()} | {"n": len(d["G"])} for s, d in stab.items()}
# separation of stay vs switch by each vector: AUROC of (1 - cos) for switch detection
from sklearn.metrics import roc_auc_score
y = [0] * len(stab["stay"]["G"]) + [1] * (len(stab["new"]["G"]) + len(stab["return"]["G"]))
for v in "GFC":
    sc = [-x for x in stab["stay"][v] + stab["new"][v] + stab["return"][v]]
    res.setdefault("switch_auroc_from_cos", {})[v] = round(100 * roc_auc_score(y, sc), 2)

# (2) document-level discrimination over all 5.9M articles
gold_doc = np.array([T.DOCS.p2d[tm[k][0]["turns"][tm[k][1]]["gold"][0]] for k in keys])
D8 = torch.from_numpy(np.ascontiguousarray(T.DOCS.E8)).cuda()


def doc_rank(Q):
    Qm = torch.stack([Q[k] for k in keys]).cuda()
    n = Qm.shape[0]; pad = (-n) % 32
    Q8 = _q8(torch.cat([Qm, Qm.new_zeros(pad, Qm.shape[1])]))
    ranks = np.zeros(n, dtype=np.int64)
    for s in range(0, Q8.shape[0], 256):
        sc = torch.cat([_intmm(Q8[s:s + 256], D8[j:j + 1_000_000].T) for j in range(0, D8.shape[0], 1_000_000)], 1).float()
        rows = range(s, min(s + 256, n))
        g = torch.tensor(gold_doc[list(rows)], device="cuda")
        gs = sc[torch.arange(len(rows), device="cuda"), g]
        ranks[list(rows)] = ((sc[: len(rows)] > gs[:, None]).sum(1) + 1).cpu().numpy()
    return {"R@1": round(100 * float((ranks <= 1).mean()), 2), "R@10": round(100 * float((ranks <= 10).mean()), 2),
            "MRR": round(100 * float((1 / ranks).mean()), 2)}


res["doc_level"] = {"q_G (HINT)": doc_rank(QG), "q_F (HINT)": doc_rank(QF), "q (concat)": doc_rank(QC)}

# (3) within-document discrimination: rank gold section among its article's sections
idx = T._index("topiocqa")
p2row = {}
want_docs = set(gold_doc.tolist())
sec = {}
for f16, ids, ds in zip(idx.embs, idx.ids, T.DOCS.shard_doc):
    m = np.isin(ds, list(want_docs))
    if m.any():
        E = np.load(f16, mmap_mode="r")
        for r in np.nonzero(m)[0]:
            sec.setdefault(int(ds[r]), []).append((int(ids[r]), np.asarray(E[r], dtype=np.float32)))


def within(Q):
    mrr, n = [], 0
    for k, gd in zip(keys, gold_doc):
        c, i = tm[k]; g = c["turns"][i]["gold"][0]
        S = sec.get(int(gd), [])
        if len(S) < 2: continue
        P = np.stack([e for _, e in S]); sc = P @ Q[k].numpy()
        gi = [j for j, (pid, _) in enumerate(S) if pid == g]
        if not gi: continue
        mrr.append(1 / (1 + int((sc > sc[gi[0]]).sum())))
    return {"MRR": round(100 * float(np.mean(mrr)), 2), "n": len(mrr), "mean_sections": round(float(np.mean([len(sec[d]) for d in set(gold_doc.tolist()) if d in sec])), 2)}


res["within_doc"] = {"q_F (HINT)": within(QF), "q_G (HINT)": within(QG), "q (concat)": within(QC)}
json.dump(res, open(f"../runs/{hint}/mechanism.json", "w"), indent=1)
print(json.dumps(res, indent=1))
