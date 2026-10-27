"""Inference-time sensitivity of LADR, one streaming pass over the full collection:
     s_{w,b}(p) = w * A_d(p) + a_F(p) - b * B_d(p)
   w scales the topic log-prior (w = 1: trained model); b scales the within-document normaliser
   (b = 1: exact latent-document likelihood, b = 0: product of experts with the same vectors).
Usage: python code/sweep_ladr.py <run>  -> runs/<run>/sweep_ladr.json (other outputs untouched)."""
import sys, os, json, numpy as np, torch
name = sys.argv[1]
sys.argv = ["train.py", "--kind", "concat", "--latent", "--expert", "centroid", "--name", name, "--eval_only"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from retrieval import DocIndex, _q8, _intmm, seg_lse, metrics, summarize

GRID = [(w, 1.0) for w in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)] + [(1.0, b) for b in (0.0, 0.25, 0.5, 0.75)]
model = T.P.build("concat", two_level=True, shared=False, tau=T.args.tau).cuda()
model.load_state_dict(torch.load(os.path.join(T.OUT, "model.pt"), map_location="cuda"), strict=False)
tau_f, tau_g = model.log_tau_f.exp().item(), model.log_tau_g.exp().item()
convs, side = T.load_side("topiocqa", "dev")
Q, _ = T.encode_split(model, "topiocqa", convs, side)
keys = [T.key(c, i) for c in convs for i, t in enumerate(c["turns"]) if t["gold"]]
QF = torch.stack([Q[k] for k in keys]); QG = torch.stack([T.QG[k] for k in keys])
index = T._index("topiocqa"); docs = DocIndex("topiocqa")
n = QF.shape[0]; padn = (-n) % 32 + (32 if n < 32 else 0); dev = "cuda"; k = 100; bs = 96; S = 127 * 127
QF8 = _q8(torch.cat([QF, QF.new_zeros(padn, QF.shape[1])]).to(dev)); QG8 = _q8(torch.cat([QG, QG.new_zeros(padn, QG.shape[1])]).to(dev))
rows = QF8.shape[0]
best = {g: (torch.full((rows, k), -1e30, device=dev), torch.zeros((rows, k), device=dev, dtype=torch.int64)) for g in GRID}
files, ids_l, dsh_l = index.search_files, index.ids, docs.shard_doc
with torch.no_grad():
    for si in range(len(files)):
        E = np.load(files[si], mmap_mode="r"); I = ids_l[si]; D = dsh_l[si]
        a = int(np.searchsorted(D, D[0], "right")) if si > 0 and dsh_l[si - 1][-1] == D[0] else 0
        E, I, D = E[a:], I[a:], D[a:]
        if si + 1 < len(files) and dsh_l[si + 1][0] == D[-1]:
            b = int(np.searchsorted(dsh_l[si + 1], D[-1], "right"))
            E = np.concatenate([E, np.load(files[si + 1], mmap_mode="r")[:b]]); I = np.concatenate([I, ids_l[si + 1][:b]])
            D = np.concatenate([D, dsh_l[si + 1][:b]])
        Et = torch.from_numpy(np.ascontiguousarray(E)).to(dev).T.contiguous(); It = torch.from_numpy(np.asarray(I)).to(dev)
        lo = int(D[0]); nd = int(D[-1]) - lo + 1; inv = torch.from_numpy(D.astype(np.int64) - lo).to(dev)
        Dt = torch.from_numpy(np.ascontiguousarray(docs.E8[lo:lo + nd])).to(dev).T.contiguous()
        for s in range(0, rows, bs):
            aF = _intmm(QF8[s:s + bs], Et).float() / (S * tau_f)
            A = (_intmm(QG8[s:s + bs], Dt).float() / (S * tau_g))[:, inv]
            B = seg_lse(aF, inv, nd)[:, inv]
            for (w, bb) in GRID:
                ts, ti = (w * A + aF - bb * B).topk(k, dim=1)
                bs_, bi_ = best[(w, bb)]
                cs = torch.cat([bs_[s:s + bs], ts], 1); ci = torch.cat([bi_[s:s + bs], It[ti]], 1)
                b_, j_ = cs.topk(k, dim=1); bs_[s:s + bs] = b_; bi_[s:s + bs] = ci.gather(1, j_)
            del aF, A, B
        del Et, Dt
        print("shard", si, flush=True)
tmap = {T.key(c, i): t for c in convs for i, t in enumerate(c["turns"])}
golds = [index.pid(tmap[kk]["gold"]) for kk in keys]
out = {"tau_f": tau_f, "tau_g": tau_g}
for g in GRID:
    out[f"w={g[0]},b={g[1]}"] = summarize(metrics(best[g][1][:n].cpu().numpy(), golds)); print(g, out[f"w={g[0]},b={g[1]}"], flush=True)
json.dump(out, open(os.path.join(T.OUT, "sweep_ladr.json"), "w"), indent=1)
