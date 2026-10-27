"""Final tables for the hierarchical intent model: means/stds, conversation-level bootstrap vs concat."""
import json, numpy as np, os
R = "runs"; MET = ["MRR", "NDCG@3", "R@10", "R@100"]
ev = lambda n, t="topiocqa_dev": json.load(open(f"{R}/{n}/eval_{t}.json"))
def perq(n, t="topiocqa_dev"):
    d = np.load(f"{R}/{n}/rank_{t}.npy", allow_pickle=True).item()
    return d["keys"], np.stack([d["per_q"][m] for m in MET], 1)
groups = {"concat": ["F_concat_s42", "F_concat_s43", "F_concat_s44"], "prism": ["F_prism_s42", "F_prism_s43", "F_prism_s44"],
          "hier": ["H_hier_s42", "H_hier_s43", "H_hier_s44"]}
out = {}
for g, rs in groups.items():
    v = np.array([[ev(r)["all"][m] for m in MET] for r in rs])
    out[g] = dict(mean=v.mean(0).round(2).tolist(), std=v.std(0).round(2).tolist())
    secs = ["turn1-3", "turn4-6", "turn7-10", "turn11-99", "stay", "new_topic", "return"]
    out[g]["break"] = {s: round(float(np.mean([ev(r)[s]["MRR"] for r in rs])), 2) for s in secs}
    out[g]["break_ndcg"] = {s: round(float(np.mean([ev(r)[s]["NDCG@3"] for r in rs])), 2) for s in secs}
# conversation-level paired bootstrap (seed-averaged per-query scores)
def avg(rs):
    ks, _ = perq(rs[0]); A = np.mean([perq(r)[1] for r in rs], 0); return ks, A
ks, H = avg(groups["hier"]); _, C = avg(groups["concat"]); _, P = avg(groups["prism"])
conv = np.array([k.rsplit("|", 1)[0] for k in ks]); uc = np.unique(conv)
rows = {c: np.where(conv == c)[0] for c in uc}
rng = np.random.default_rng(0)
def boot(X, Y, B=10000):
    d = X - Y; obs = d.mean(0); cnt = np.zeros(len(MET)); lo = []
    for _ in range(B):
        s = np.concatenate([rows[c] for c in rng.choice(uc, len(uc))])
        m = d[s].mean(0); lo.append(m); cnt += (m <= 0)
    lo = np.array(lo)
    return dict(diff=(100 * obs).round(2).tolist(), ci95=[(100 * np.percentile(lo, 2.5, 0)).round(2).tolist(),
                (100 * np.percentile(lo, 97.5, 0)).round(2).tolist()], p_one_sided=(cnt / B).round(5).tolist())
out["boot_hier_vs_concat"] = boot(H, C); out["boot_hier_vs_prism"] = boot(H, P); out["boot_prism_vs_concat"] = boot(P, C)
for n in ["H_shared_s42", "H_nodocloss_s42", "F_concat_s42", "H_hier_s42"]:
    e = ev(n); out[n] = {**e["all"], "stay": e["stay"]["MRR"], "new": e["new_topic"]["MRR"], "return": e["return"]["MRR"]}
for n in ["V_concat", "V_prism", "V_hier", "V_hier_shared"]:
    e = ev(n, "topiocqa_val"); out[n] = {**e["all"], "return": e["return"]["MRR"]}
json.dump(out, open(f"{R}/hier_stats.json", "w"), indent=1)
print(json.dumps(out, indent=1))
