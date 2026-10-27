"""Encode a passage corpus with the frozen bge-base passage tower (server pipeline).
Writes per shard: emb_XXXXX.npy (fp16), emb_XXXXX.i8.npy (int8, scale 127), ids_XXXXX.npy|json.
Tokenisation runs in DataLoader workers so the GPU is never idle; shards are processed as downloads finish.
Usage: python encode_fast.py topiocqa|orquac"""
import sys, os, glob, time, gzip, json
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "BAAI/bge-base-en-v1.5"
MAXLEN, BS = 256, 512
tok = AutoTokenizer.from_pretrained(MODEL)


class Batches(Dataset):
    """Length-sorted batches; each item is one tokenised batch (padding done in the worker)."""
    def __init__(self, texts):
        self.texts = texts
        order = np.argsort([len(t) for t in texts])[::-1]
        self.chunks = [order[i:i + BS] for i in range(0, len(order), BS)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        idx = self.chunks[i]
        b = tok([self.texts[j] for j in idx], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        return torch.from_numpy(idx.copy()), b["input_ids"], b["attention_mask"]


@torch.no_grad()
def encode(texts, m):
    out = torch.zeros(len(texts), 768, dtype=torch.float16)
    dl = DataLoader(Batches(texts), batch_size=None, num_workers=6, prefetch_factor=4)
    for idx, ii, aa in dl:
        h = m(input_ids=ii.cuda(non_blocking=True), attention_mask=aa.cuda(non_blocking=True)).last_hidden_state[:, 0]
        out[idx] = torch.nn.functional.normalize(h.float(), dim=-1).half().cpu()
    return out.numpy()


def save(dst, name, e):
    np.save(os.path.join(dst, f"emb_{name}.tmp.npy"), e)
    np.save(os.path.join(dst, f"emb_{name}.i8.tmp.npy"), np.clip(np.round(e.astype(np.float32) * 127), -127, 127).astype(np.int8))
    os.replace(os.path.join(dst, f"emb_{name}.i8.tmp.npy"), os.path.join(dst, f"emb_{name}.i8.npy"))
    os.replace(os.path.join(dst, f"emb_{name}.tmp.npy"), os.path.join(dst, f"emb_{name}.npy"))


def topiocqa(m):
    import pyarrow.parquet as pq
    src = os.path.join(ROOT, "data", "wiki")
    dst = os.path.join(ROOT, "data", "index_topiocqa"); os.makedirs(dst, exist_ok=True)
    done = set()
    while len(done) < 27:
        ready = sorted(f for f in glob.glob(os.path.join(src, "*.parquet"))
                       if f not in done and os.path.exists(f.replace("-of-00027.parquet", ".ok")))
        if not ready:
            time.sleep(30); continue
        f = ready[0]; name = os.path.basename(f).split("-")[1]
        if not os.path.exists(os.path.join(dst, f"emb_{name}.npy")):
            t0 = time.time()
            t = pq.read_table(f, columns=["id", "title", "sub_title", "contents"]).to_pydict()
            texts = [f"{a} [SEP] {b} {c}" for a, b, c in zip(t["title"], t["sub_title"], t["contents"])]
            e = encode(texts, m)
            np.save(os.path.join(dst, f"ids_{name}.npy"), np.array([int(x.split(":")[1]) for x in t["id"]], dtype=np.int64))
            save(dst, name, e)
            print(name, len(texts), f"{len(texts)/(time.time()-t0):.0f}/s", flush=True)
        done.add(f)
    open(os.path.join(dst, "DONE"), "w").write("ok")


def orquac(m):
    src = os.path.join(ROOT, "data", "orquac")
    while not os.path.exists(os.path.join(src, "dl.log")):
        time.sleep(60)
    dst = os.path.join(ROOT, "data", "index_orquac"); os.makedirs(dst, exist_ok=True)
    shard, buf_t, buf_id, k = 1_000_000, [], [], 0

    def flush():
        nonlocal buf_t, buf_id, k
        name = f"{k:05d}"
        if not os.path.exists(os.path.join(dst, f"emb_{name}.npy")):
            t0 = time.time()
            save(dst, name, encode(buf_t, m))
            json.dump(buf_id, open(os.path.join(dst, f"ids_{name}.json"), "w"))
            print(k, len(buf_t), f"{len(buf_t)/(time.time()-t0):.0f}/s", flush=True)
        buf_t, buf_id, k = [], [], k + 1

    with gzip.open(os.path.join(src, "all_blocks.txt.gz"), "rt", encoding="utf8") as f:
        for line in f:
            d = json.loads(line)
            buf_t.append(f"{d['title']} [SEP] {d['text']}"); buf_id.append(d["id"])
            if len(buf_t) == shard:
                flush()
    if buf_t:
        flush()
    open(os.path.join(dst, "DONE"), "w").write("ok")


if __name__ == "__main__":
    m = AutoModel.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    for d in sys.argv[1:]:
        {"topiocqa": topiocqa, "orquac": orquac}[d](m)
