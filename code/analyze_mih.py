"""Diagnose a MIH checkpoint on the held-out validation conversations:
per-hypothesis MRR, oracle (best hypothesis per query), argmax-posterior and mixture."""
import sys, json, numpy as np, torch
sys.argv = [sys.argv[0], "--kind", "prism", "--mih", sys.argv[2] if len(sys.argv) > 2 else "5", "--name", sys.argv[1],
            "--holdout", "200", "--eval_only"]
import train as T
from retrieval import metrics

model = T.P.build("prism", k_split=4, n_scales=3, mih=T.args.mih).cuda()
model.load_state_dict(torch.load(f"{T.OUT}/model.pt", map_location="cuda"))
convs, side = T.load_side("topiocqa", "train"); convs = convs[-200:]
Q, meta = T.encode_split(model, "topiocqa", convs, side)
keys = [T.key(c, i) for c in convs for i, t in enumerate(c["turns"]) if t["gold"]]
tm = {T.key(c, i): t for c in convs for i, t in enumerate(c["turns"])}
golds = [tm[k]["gold"] for k in keys]
QQ = torch.stack([Q[k] for k in keys]); n, V, _ = QQ.shape
LPq = torch.stack([T.LP[k] for k in keys]).numpy()
idx = T._index("topiocqa")
s_, i_ = idx.search(QQ.reshape(n * V, -1), k=100)
i_ = i_.reshape(n, V, 100)
per = []
for v in range(V):
    m = metrics(i_[:, v], golds)["MRR"]; per.append(m)
    print(f"hyp {v}: MRR {100*m.mean():.2f}  mean pi {np.exp(LPq[:, v]).mean():.3f}")
P = np.stack(per, 1)
print(f"oracle best hypothesis: {100*P.max(1).mean():.2f}")
am = LPq.argmax(1)
print(f"argmax-pi hypothesis: {100*P[np.arange(n), am].mean():.2f}")
sh = np.array([tm[k]["shift"] for k in keys])
for s, nm in [(0, "stay"), (1, "new"), (2, "return")]:
    sel = sh == s
    print(nm, "per-hyp", (100 * P[sel].mean(0)).round(2), "oracle", round(100 * P[sel].max(1).mean(), 2),
          "best-hyp share", np.bincount(P[sel].argmax(1), minlength=V) / sel.sum())
