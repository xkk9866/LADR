"""Inference-time sensitivity: score(p) = qF.p + w * qG.d(p) for several w, one streaming pass.
Usage: python code/sweep_weight.py H_hier_s42   -> runs/<name>/sweep_w.json (does not touch other outputs)."""
import sys, os, json, numpy as np, torch
name = sys.argv[1]
sys.argv = ["train.py", "--kind", "concat", "--two_level", "--name", name, "--eval_only"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from retrieval import DocIndex, _q8, _intmm, metrics, summarize

W = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
model = T.P.build("concat", two_level=True, shared=False, tau=T.args.tau).cuda()
model.load_state_dict(torch.load(os.path.join(T.OUT, "model.pt"), map_location="cuda"), strict=False)
convs, side = T.load_side("topiocqa", "dev")
Q, _ = T.encode_split(model, "topiocqa", convs, side)
keys = [T.key(c, i) for c in convs for i, t in enumerate(c["turns"]) if t["gold"]]
QF = torch.stack([Q[k] for k in keys]); QG = torch.stack([T.QG[k] for k in keys])
idx = T._index("topiocqa"); docs = DocIndex("topiocqa")

n = QF.shape[0]; padn = (-n) % 32 + (32 if n < 32 else 0); dev = "cuda"; k = 100; bs = 512
QF8 = _q8(torch.cat([QF, QF.new_zeros(padn, QF.shape[1])]).to(dev))
QG8 = _q8(torch.cat([QG, QG.new_zeros(padn, QG.shape[1])]).to(dev))
rows = QF8.shape[0]
best = {w: (torch.full((rows, k), -1e4, device=dev), torch.zeros((rows, k), device=dev, dtype=torch.int64)) for w in W}
with torch.no_grad():
    for f8, ids, dshard in zip(idx.search_files, idx.ids, docs.shard_doc):
        E = torch.from_numpy(np.ascontiguousarray(np.load(f8, mmap_mode="r"))).to(dev); Et = E.T.contiguous()
        I = torch.from_numpy(ids).to(dev)
        lo, hi = int(dshard.min()), int(dshard.max()) + 1
        Dt = torch.from_numpy(np.ascontiguousarray(docs.E8[lo:hi])).to(dev).T.contiguous()
        inv = torch.from_numpy(dshard.astype(np.int64) - lo).to(dev)
        for s in range(0, rows, bs):
            sp = _intmm(QF8[s:s + bs], Et).float() / (127 * 127)
            sd = (_intmm(QG8[s:s + bs], Dt).float() / (127 * 127))[:, inv]
            for w in W:
                ts, ti = (sp + w * sd).topk(k, dim=1)
                bs_, bi_ = best[w]
                cs = torch.cat([bs_[s:s + bs], ts], 1); ci = torch.cat([bi_[s:s + bs], I[ti]], 1)
                b_, j_ = cs.topk(k, dim=1); bs_[s:s + bs] = b_; bi_[s:s + bs] = ci.gather(1, j_)
            del sp, sd
        del E, Et, Dt
        print("shard", f8[-14:], flush=True)

tmap = {T.key(c, i): t for c in convs for i, t in enumerate(c["turns"])}
golds = [idx.pid(tmap[kk]["gold"]) for kk in keys]
out = {}
for w in W:
    out[str(w)] = summarize(metrics(best[w][1][:n].cpu().numpy(), golds))
    print(w, out[str(w)], flush=True)
json.dump(out, open(os.path.join(T.OUT, "sweep_w.json"), "w"), indent=1)
