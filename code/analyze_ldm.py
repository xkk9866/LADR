"""Aggregate the latent-document experiments: seed means, conversation-level paired bootstrap, breakdowns,
document-posterior probes (article accuracy, entropy, unsupervised topic-switch detection). -> runs/ldm_stats.json"""
import os, json, glob, numpy as np
from sklearn.metrics import roc_auc_score
from retrieval import ROOT

R = os.path.join(ROOT, "runs"); DD = os.path.join(ROOT, "data")
TAG = "topiocqa_dev"
convs = json.load(open(os.path.join(DD, "topiocqa_dev.json")))
tmap = {f"{c['conv_id']}|{i}": t for c in convs for i, t in enumerate(c["turns"])}
idx = os.path.join(DD, "index_topiocqa")
ids = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(idx, "ids_*.npy")))])
doc = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(idx, "docid_*.npy")))])
p2d = np.full(int(ids.max()) + 1, -1, dtype=np.int64); p2d[ids] = doc
out = {}


def load(run):
    p = os.path.join(R, run, f"rank_{TAG}.npy")
    return np.load(p, allow_pickle=True).item() if os.path.exists(p) else None


def seeds(prefix):
    return sorted(d for d in os.listdir(R) if d.startswith(prefix) and os.path.exists(os.path.join(R, d, f"eval_{TAG}.json")))


groups = {"concat": "C_concat_s", "hint": "H_hint_s", "ours": "L_centroid_s", "lse": "L_ours_s", "max": "L_max_s",
          "lse_joint": "L_joint_s", "lse_shared": "L_shared_s", "lse_fixtau": "L_fixtau_s",
          "shared": "LC_shared_s", "fixtau": "LC_fixtau_s", "nokd": "LC_nokd_s", "joint": "L_lsejoint_s"}
for g, pre in groups.items():
    runs = seeds(pre)
    if not runs: continue
    ev = [json.load(open(os.path.join(R, r, f"eval_{TAG}.json"))) for r in runs]
    agg = {}
    for part in ev[0]:
        if part == "tau":
            agg["tau"] = {k: [e["tau"][k] for e in ev] for k in ("g", "f")}; continue
        if not isinstance(ev[0][part], dict): continue
        agg[part] = {m: [round(float(np.mean([e[part][m] for e in ev])), 2), round(float(np.std([e[part][m] for e in ev])), 2)]
                     for m in ev[0][part]}
    # article accuracy of the top-1 passage
    acc = []
    for r in runs:
        z = load(r)
        gd = np.array([p2d[tmap[k]["gold"][0]] for k in z["keys"]]); acc.append(float((p2d[z["ids"][:, 0]] == gd).mean() * 100))
    agg["article_acc@1"] = [round(np.mean(acc), 2), round(np.std(acc), 2)]
    agg["runs"] = runs
    out[g] = agg


def boot(a, b, metric="MRR", n=10000, seed=0):
    """paired bootstrap over conversations; a, b: lists of run names (per-query metrics averaged over seeds)."""
    za = [load(r) for r in a]; zb = [load(r) for r in b]
    keys = za[0]["keys"]
    x = np.mean([z["per_q"][metric] for z in za], 0) - np.mean([z["per_q"][metric] for z in zb], 0)
    conv = np.array([k.rsplit("|", 1)[0] for k in keys]); u, inv = np.unique(conv, return_inverse=True)
    s = np.bincount(inv, x, len(u)); c = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        j = rng.integers(0, len(u), len(u)); d.append(s[j].sum() / c[j].sum())
    d = np.array(d) * 100
    return dict(diff=round(float(x.mean() * 100), 2), ci=[round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
                p=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


out["boot"] = {}
pairs = [("ours", "concat"), ("ours", "hint"), ("hint", "concat")] + [("ours", g) for g in ("joint", "lse", "max", "shared", "fixtau", "nokd")] + [("joint", "hint")]
for a, b in pairs:
    if a in out and b in out:
        k = min(len(out[a]["runs"]), len(out[b]["runs"]))
        out["boot"][f"{a}-{b}"] = {m: boot(out[a]["runs"][:k], out[b]["runs"][:k], m) for m in ("MRR", "NDCG@3", "R@100")}

# document posterior probes (ours, every seed)
probe = []
for r in out.get("ours", {}).get("runs", []):
    z = np.load(os.path.join(R, r, f"docpost_{TAG}.npy"), allow_pickle=True).item()
    keys, D, S, lZ = z["keys"], z["doc_ids"], z["doc_logit"], z["logZ"]
    P = np.exp(S - lZ[:, None])                                   # top-20 posterior mass
    gd = np.array([p2d[tmap[k]["gold"][0]] for k in keys])
    r1 = float((D[:, 0] == gd).mean() * 100); r10 = float((D[:, :10] == gd[:, None]).any(1).mean() * 100)
    mass_gold = float(np.mean([P[i][D[i] == gd[i]].sum() for i in range(len(keys))]))
    ent = -(P * np.log(P + 1e-12)).sum(1) - (1 - P.sum(1)).clip(1e-12) * np.log((1 - P.sum(1)).clip(1e-12))
    # unsupervised switch detection: Jensen-Shannon divergence between consecutive document posteriors
    pos = {k: i for i, k in enumerate(keys)}; ys, js = [], []
    for k, i in pos.items():
        c, t = k.rsplit("|", 1); t = int(t)
        if t == 0 or f"{c}|{t-1}" not in pos: continue
        j = pos[f"{c}|{t-1}"]
        U = np.union1d(D[i], D[j])
        p = np.array([P[i][D[i] == u].sum() for u in U]); q = np.array([P[j][D[j] == u].sum() for u in U])
        p = np.append(p, max(1e-9, 1 - p.sum())); q = np.append(q, max(1e-9, 1 - q.sum())); m = (p + q) / 2
        kl = lambda a, b: float((a * np.log((a + 1e-12) / (b + 1e-12))).sum())
        js.append(0.5 * kl(p, m) + 0.5 * kl(q, m)); ys.append(int(tmap[k]["shift"] > 0))
    sh = np.array([tmap[k]["shift"] for k in keys]); tn = np.array([int(k.rsplit("|", 1)[1]) for k in keys])
    probe.append(dict(run=r, article_R1=r1, article_R10=r10, gold_mass=mass_gold, switch_AUROC=100 * roc_auc_score(ys, js),
                      entropy={nm: float(ent[(sh == v) & (tn > 0)].mean()) for v, nm in [(0, "stay"), (1, "new"), (2, "return")]}))
out["probe"] = probe
json.dump(out, open(os.path.join(R, "ldm_stats.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k in ("boot", "probe")}, indent=1))
for g in groups:
    if g in out: print(g, out[g]["all"], out[g]["article_acc@1"], out[g].get("tau"))
