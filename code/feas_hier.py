"""Zero-training feasibility: rescore with an article-level term using an existing single-vector model."""
import sys, json, numpy as np, torch
run = sys.argv[1]; kind = sys.argv[2]
sys.argv = [sys.argv[0], "--kind", kind, "--name", run, "--eval_only"]
import train as T
from retrieval import DocIndex, search_hier, metrics, summarize
model = T.P.build(kind, **({"k_split": 4} if kind == "prism" else {})).cuda()
model.load_state_dict(torch.load(f"{T.OUT}/model.pt", map_location="cuda"))
convs, side = T.load_side("topiocqa", "dev")
Q, _ = T.encode_split(model, "topiocqa", convs, side)
keys = [T.key(c, i) for c in convs for i, t in enumerate(c["turns"]) if t["gold"]]
tm = {T.key(c, i): t for c in convs for i, t in enumerate(c["turns"])}
golds = [tm[k]["gold"] for k in keys]
QQ = torch.stack([Q[k] for k in keys])
idx = T._index("topiocqa"); docs = DocIndex()
for w in [0.5, 1.0]:
    _, ids = search_hier(idx, docs, QQ, QQ * w)
    print(run, "w", w, summarize(metrics(ids, golds)), flush=True)
