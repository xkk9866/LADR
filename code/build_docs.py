"""Article-level (document) index for the hierarchical intent model.
For every passage shard: doc id per row (article title). Document embedding = normalised mean of the
(normalised) passage embeddings of its sections. Outputs data/index_topiocqa/docid_XXXXX.npy and
data/docs_topiocqa/{emb.npy (fp16), titles.json}."""
import os, glob, json, numpy as np, torch, pyarrow.parquet as pq

import sys
DATA = sys.argv[1] if len(sys.argv) > 1 else "topiocqa"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "data", f"index_{DATA}")
OUT = os.path.join(ROOT, "data", f"docs_{DATA}"); os.makedirs(OUT, exist_ok=True)

tpath = os.path.join(OUT, "titles.json")
if DATA == "orquac":   # passage ids are "<article id>@<block>"; blocks of an article are contiguous
    files = sorted(glob.glob(os.path.join(IDX, "ids_*.json")))
    if not os.path.exists(tpath):
        key2doc, titles = {}, []
        for f in files:
            name = os.path.basename(f)[4:-5]
            d = np.empty(0, dtype=np.int32); ids = json.load(open(f)); d = np.empty(len(ids), dtype=np.int32)
            for j, x in enumerate(ids):
                a = x.rsplit("@", 1)[0]
                k = key2doc.get(a)
                if k is None:
                    k = key2doc[a] = len(titles); titles.append(a)
                d[j] = k
            np.save(os.path.join(IDX, f"docid_{name}.npy"), d); print(name, len(titles), flush=True)
        json.dump(titles, open(tpath, "w"))
    titles = json.load(open(tpath))
    files = [f.replace("ids_", "emb_").replace(".json", ".npy") for f in files]
else:
    files = sorted(glob.glob(os.path.join(ROOT, "data", "wiki", "*.parquet")))
if DATA == "orquac":
    pass
elif os.path.exists(tpath) and all(os.path.exists(os.path.join(IDX, f"docid_{os.path.basename(f).split('-')[1]}.npy")) for f in files):
    titles = json.load(open(tpath))
else:
    title2doc, titles = {}, []
    for f in files:
        name = os.path.basename(f).split("-")[1]
        t = pq.read_table(f, columns=["id", "title"]).to_pydict()
        ids = np.load(os.path.join(IDX, f"ids_{name}.npy"))
        assert len(ids) == len(t["id"]) and int(t["id"][0].split(":")[1]) == ids[0]
        d = np.empty(len(ids), dtype=np.int32)
        for j, ti in enumerate(t["title"]):
            k = title2doc.get(ti)
            if k is None:
                k = title2doc[ti] = len(titles); titles.append(ti)
            d[j] = k
        np.save(os.path.join(IDX, f"docid_{name}.npy"), d)
        print(name, len(titles), flush=True)
    json.dump(titles, open(tpath, "w"))
n = len(titles)
# streaming per shard on the GPU: articles are contiguous in the corpus; the article straddling two shards is carried
out = np.lib.format.open_memmap(os.path.join(OUT, "emb.tmp.npy"), mode="w+", dtype=np.float16, shape=(n, 768))
cnt = np.zeros(n, dtype=np.int32)
cid, csum, ccnt = -1, None, 0
for f in files:
    name = os.path.basename(f)[4:-4] if DATA == "orquac" else os.path.basename(f).split("-")[1]
    E = torch.from_numpy(np.load(os.path.join(IDX, f"emb_{name}.npy"))).cuda().float()
    d = np.load(os.path.join(IDX, f"docid_{name}.npy")).astype(np.int64)
    assert np.all(np.diff(d) >= 0)
    lo, hi = int(d[0]), int(d[-1]) + 1
    S = torch.zeros(hi - lo, 768, device="cuda").index_add_(0, torch.from_numpy(d - lo).cuda(), E)
    c = np.bincount(d - lo, minlength=hi - lo)
    if cid >= 0:
        if cid == lo:
            S[0] += csum; c[0] += ccnt
        else:
            out[cid] = torch.nn.functional.normalize(csum / ccnt, dim=0).half().cpu().numpy(); cnt[cid] = ccnt
    cid, csum, ccnt = hi - 1, S[-1].clone(), int(c[-1])
    if hi - lo > 1:
        M = torch.nn.functional.normalize(S[:-1] / torch.from_numpy(np.maximum(c[:-1], 1)).cuda().float()[:, None], dim=-1)
        out[lo:hi - 1] = M.half().cpu().numpy(); cnt[lo:hi - 1] = c[:-1]
    del E, S
    print("doc", name, hi, flush=True)
out[cid] = torch.nn.functional.normalize(csum / ccnt, dim=0).half().cpu().numpy(); cnt[cid] = ccnt
out.flush(); del out
os.replace(os.path.join(OUT, "emb.tmp.npy"), os.path.join(OUT, "emb.npy"))
np.save(os.path.join(OUT, "count.npy"), cnt)
print("docs", n, "mean sections", float(cnt.mean()), "empty", int((cnt == 0).sum()))
