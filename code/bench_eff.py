"""Real system measurements on one RTX 4090: encoder latency/throughput, exact search latency
(single-level vs two-level int8), GPU memory, index storage."""
import os, sys, time, json, glob, numpy as np, torch
sys.argv = [sys.argv[0], "--kind", "concat", "--name", "bench", "--eval_only"]
import train as T
from retrieval import Index, DocIndex, search_hier
ROOT = T.ROOT
out = {}
convs, side = T.load_side("topiocqa", "dev")
items = [(c, i) for c in convs for i in range(len(c["turns"]))]


def enc_bench(two):
    T.args.two_level = two
    m = T.P.build("concat", two_level=two).cuda().eval()
    r = {}
    for bs in (1, 64):
        sel = items[:256 if bs > 1 else 64]
        def run():
            for s in range(0, len(sel), bs):
                b = [dict(conv_id=c["conv_id"], turns=c["turns"][:i + 1], _only_last=True) for c, i in sel[s:s + bs]]
                batch, _ = T.make_batch(b, "topiocqa", side, False)
                T.run_model(m, batch)
            torch.cuda.synchronize()
        run()
        torch.cuda.reset_peak_memory_stats(); t0 = time.time(); run(); dt = time.time() - t0
        r[f"bs{bs}"] = {"ms_per_query": round(1000 * dt / len(sel), 2), "qps": round(len(sel) / dt, 1),
                        "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    del m; torch.cuda.empty_cache()
    return r


out["encoder"] = {"concat": enc_bench(False), "hint": enc_bench(True)}
idx = Index("topiocqa"); docs = DocIndex("topiocqa")
Q = torch.nn.functional.normalize(torch.randn(1024, 768), dim=-1)
srch = {}
for n in (1, 64, 1024):
    q = Q[:n]
    idx.search(q, k=100); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); idx.search(q, k=100); torch.cuda.synchronize(); t1 = time.time() - t0
    m1 = torch.cuda.max_memory_allocated() / 1e9
    search_hier(idx, docs, q, q, k=100); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); search_hier(idx, docs, q, q, k=100); torch.cuda.synchronize(); t2 = time.time() - t0
    m2 = torch.cuda.max_memory_allocated() / 1e9
    srch[n] = {"single_s": round(t1, 2), "two_level_s": round(t2, 2), "overhead_pct": round(100 * (t2 / t1 - 1), 1),
               "single_peak_GB": round(m1, 2), "two_level_peak_GB": round(m2, 2)}
    print(n, srch[n], flush=True)
out["search_full_collection_streamed"] = srch
gb = lambda pat: round(sum(os.path.getsize(f) for f in glob.glob(pat)) / 1e9, 2)
out["storage_GB"] = {"passage_int8": gb(os.path.join(ROOT, "data/index_topiocqa/*.i8.npy")),
                     "document_int8": gb(os.path.join(ROOT, "data/docs_topiocqa/emb.i8.npy")),
                     "doc_map": gb(os.path.join(ROOT, "data/index_topiocqa/docid_*.npy"))}
json.dump(out, open(os.path.join(ROOT, "runs", "efficiency.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
