"""Encode a passage corpus with a frozen bge-base passage tower into fp16 shards.
Usage: python encode_corpus.py topiocqa|orquac
Shards are written per input file so encoding can run while the download is in progress."""
import sys, os, glob, time, gzip, json
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "BAAI/bge-base-en-v1.5"
MAXLEN, BS = 256, 512


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    return tok, m


@torch.no_grad()
def encode(texts, tok, m):
    # sort by length for throughput, restore order afterwards
    order = np.argsort([len(t) for t in texts])[::-1]
    out = np.zeros((len(texts), 768), dtype=np.float16)
    for i in range(0, len(texts), BS):
        idx = order[i:i + BS]
        b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to("cuda")
        h = m(**b).last_hidden_state[:, 0]
        out[idx] = torch.nn.functional.normalize(h.float(), dim=-1).half().cpu().numpy()
    return out


def topiocqa():
    import pyarrow.parquet as pq
    src = os.path.join(ROOT, "data", "wiki")
    dst = os.path.join(ROOT, "data", "index_topiocqa"); os.makedirs(dst, exist_ok=True)
    tok, m = load_model()
    done = set()
    while True:
        files = sorted(glob.glob(os.path.join(src, "*.parquet")))
        log = os.path.join(src, "dl.log")
        complete = os.path.exists(log) and "DONE" in open(log).read()
        # a file is ready if a later file exists or the whole download finished
        part, nparts = int(os.environ.get("PART", 0)), int(os.environ.get("NPARTS", 1))
        files = [f for f in files if int(os.path.basename(f).split("-")[1]) % nparts == part]
        ready = [f for f in files if f not in done and os.path.exists(f.replace("-of-00027.parquet", ".ok"))]
        if not ready:
            if complete and len(done) == len(files) and files:
                break
            time.sleep(30); continue
        for f in ready:
            name = os.path.basename(f).split("-")[1]
            if os.path.exists(os.path.join(dst, f"emb_{name}.npy")):
                done.add(f); continue
            t0 = time.time()
            t = pq.read_table(f).to_pydict()
            texts = [f"{a} [SEP] {b} {c}" for a, b, c in zip(t["title"], t["sub_title"], t["contents"])]
            e = encode(texts, tok, m)
            ids = np.array([int(x.split(":")[1]) for x in t["id"]], dtype=np.int64)
            np.save(os.path.join(dst, f"ids_{name}.npy"), ids)
            np.save(os.path.join(dst, f"emb_{name}.npy"), e)
            done.add(f)
            print(name, len(texts), f"{len(texts)/(time.time()-t0):.0f}/s", flush=True)


def orquac():
    src = os.path.join(ROOT, "data", "orquac", "all_blocks.txt.gz")
    dst = os.path.join(ROOT, "data", "index_orquac"); os.makedirs(dst, exist_ok=True)
    tok, m = load_model()
    shard, buf_t, buf_id, k = 1_000_000, [], [], 0

    def flush():
        nonlocal buf_t, buf_id, k
        if os.path.exists(os.path.join(dst, f"emb_{k:05d}.npy")):
            pass
        else:
            t0 = time.time()
            e = encode(buf_t, tok, m)
            np.save(os.path.join(dst, f"emb_{k:05d}.npy"), e)
            with open(os.path.join(dst, f"ids_{k:05d}.json"), "w") as fo:
                json.dump(buf_id, fo)
            print(k, len(buf_t), f"{len(buf_t)/(time.time()-t0):.0f}/s", flush=True)
        buf_t, buf_id, k = [], [], k + 1

    with gzip.open(src, "rt", encoding="utf8") as f:
        for line in f:
            d = json.loads(line)
            buf_t.append(f"{d['title']} [SEP] {d['text']}"); buf_id.append(d["id"])
            if len(buf_t) == shard:
                flush()
    if buf_t:
        flush()


if __name__ == "__main__":
    {"topiocqa": topiocqa, "orquac": orquac}[sys.argv[1]]()
