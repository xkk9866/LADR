"""OR-QuAC results: in-domain (O_*), transfer TopiOCQA->OR-QuAC and OR-QuAC->TopiOCQA, mean/std over three seeds and
conversation-level paired bootstrap of LADR against each baseline. -> runs/orq_stats.json"""
import os, json, numpy as np
from retrieval import ROOT

R = os.path.join(ROOT, "runs")
SETS = {"in": ("orquac_test", {"concat": "O_concat_s", "hint": "O_hint_s", "ladr": "O_ladr_s"}),
        "t2o": ("orquac_test", {"concat": "T_concat_s", "hint": "T_hint_s", "ladr": "T_ladr_s"}),
        "o2t": ("topiocqa_dev", {"concat": "O_concat_s", "hint": "O_hint_s", "ladr": "O_ladr_s"})}
out = {}


def boot(za, zb, metric="MRR", n=10000):
    keys = za[0]["keys"]
    x = np.mean([z["per_q"][metric] for z in za], 0) - np.mean([z["per_q"][metric] for z in zb], 0)
    conv = np.array([k.rsplit("|", 1)[0] for k in keys]); u, inv = np.unique(conv, return_inverse=True)
    s = np.bincount(inv, x, len(u)); c = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(0); d = []
    for _ in range(n):
        j = rng.integers(0, len(u), len(u)); d.append(s[j].sum() / c[j].sum())
    d = np.array(d) * 100
    return dict(diff=round(float(x.mean() * 100), 2), ci=[round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
                p=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


for st, (tag, groups) in SETS.items():
    out[st] = {}; ranks = {}
    for g, pre in groups.items():
        runs = [f"{pre}{s}" for s in (42, 43, 44) if os.path.exists(os.path.join(R, f"{pre}{s}", f"eval_{tag}.json"))]
        ev = [json.load(open(os.path.join(R, r, f"eval_{tag}.json")))["all"] for r in runs]
        out[st][g] = {m: [round(float(np.mean([e[m] for e in ev])), 2), round(float(np.std([e[m] for e in ev])), 2)] for m in ev[0]}
        out[st][g]["n"] = len(runs)
        ranks[g] = [np.load(os.path.join(R, r, f"rank_{tag}.npy"), allow_pickle=True).item() for r in runs]
    out[st]["boot"] = {f"ladr-{b}": {m: boot(ranks["ladr"], ranks[b], m) for m in ("MRR", "NDCG@3", "R@100")} for b in ("concat", "hint")}
p = os.path.join(R, "zeroshot_orquac_rebuilt.json")
if os.path.exists(p):
    out["zeroshot"] = json.load(open(p))
json.dump(out, open(os.path.join(R, "orq_stats.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
