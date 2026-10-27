"""Wall time and peak GPU memory of exact search over the full TopiOCQA collection (1,024 queries):
single-vector, additive two-level (HINT) and latent-document (LADR). Writes runs/bench_ladr.json."""
import os, sys, json, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import Index, DocIndex, search_hier, search_latent, ROOT
torch.manual_seed(0)
idx, docs = Index("topiocqa"), DocIndex("topiocqa")
Q = torch.nn.functional.normalize(torch.randn(1024, 768), dim=-1); G = torch.nn.functional.normalize(torch.randn(1024, 768), dim=-1)
out = {}
for name, fn in [("single", lambda: idx.search(Q, k=100)), ("hint", lambda: search_hier(idx, docs, Q, G)),
                 ("ladr", lambda: search_latent(idx, docs, Q, G, 0.0227, 0.0117, expert="centroid"))]:
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time(); fn(); torch.cuda.synchronize()
    out[name] = dict(seconds=round(time.time() - t, 1), peak_GB=round(torch.cuda.max_memory_allocated() / 2**30, 2)); print(name, out[name], flush=True)
json.dump(out, open(os.path.join(ROOT, "runs", "bench_ladr.json"), "w"), indent=1)
