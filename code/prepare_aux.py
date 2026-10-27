"""Auxiliary artefacts.
  gold    : encode gold passages (frozen passage tower) -> pstore_<data>.npz
  teacher : encode LLM / human rewrites (frozen query tower) -> teacher_<data>_<split>.npy
  zeroshot: zero-shot baselines on the full index (raw / last-3 / full concat / LLM rewrite / human rewrite),
            and from the full-concat run on train+eval splits: hard negatives (hn_*.json) + result feedback (fb_*.json),
            then extends the passage store with those passages."""
import os, sys, json, numpy as np, torch
from retrieval import Index, metrics, summarize, encode_queries, ROOT
DD = os.path.join(ROOT, "data")
SPLITS = {"topiocqa": ["train", "dev"], "orquac": ["train", "test"]}


def key(c, i):
    return f"{c['conv_id']}|{i}"


def add_pstore(data, ids, embs):
    p = os.path.join(DD, f"pstore_{data}.npz")
    if os.path.exists(p):
        z = np.load(p); old = dict(zip(z["ids"].tolist(), z["embs"]))
    else:
        old = {}
    for i, e in zip(ids, embs):
        old[int(i)] = e
    k = np.array(list(old.keys()), dtype=np.int64)
    np.savez(p, ids=k, embs=np.stack([old[x] for x in k]).astype(np.float16))
    print("pstore", data, len(k))


def gold_topiocqa():
    from encode_corpus import load_model, encode
    seen = {}
    for s in ["train", "valid"]:
        for l in open(os.path.join(DD, f"topiocqa_{s}.jsonl"), encoding="utf8"):
            g = json.loads(l)["Gold_passage"]
            seen[int(g["id"].split(":")[1])] = f"{g['title']} {g['text']}"
    tok, m = load_model()
    ids = list(seen); e = encode([seen[i] for i in ids], tok, m)
    add_pstore("topiocqa", ids, e)


def teacher(data):
    for s in SPLITS[data]:
        convs = json.load(open(os.path.join(DD, f"{data}_{s}.json")))
        rw = {}
        p = os.path.join(DD, f"{data}_{s}.llmrw.json")
        if os.path.exists(p):
            rw = json.load(open(p))
        keys, texts = [], []
        for c in convs:
            for i, t in enumerate(c["turns"]):
                r = t.get("rewrite") or rw.get(key(c, i))
                if r:
                    keys.append(key(c, i)); texts.append(r)
        E = encode_queries(texts).numpy().astype(np.float16)
        np.save(os.path.join(DD, f"teacher_{data}_{s}.npy"), dict(zip(keys, E)))
        print("teacher", data, s, len(keys))


def query_variants(c, i, rw):
    t = c["turns"]
    hist = []
    for j in range(i - 1, -1, -1):
        hist += [t[j]["q"], t[j]["a"]]
    v = {"raw": t[i]["q"],
         "last3": " [SEP] ".join([t[i]["q"]] + hist[:6]),
         "concat": " [SEP] ".join([t[i]["q"]] + hist)}
    if rw.get(key(c, i)): v["llm_rewrite"] = rw[key(c, i)]
    if t[i].get("rewrite"): v["human_rewrite"] = t[i]["rewrite"]
    return v


def zeroshot(data):
    idx = Index(data)
    Qs, tags = [], []
    plan = {}
    for s in SPLITS[data]:
        convs = json.load(open(os.path.join(DD, f"{data}_{s}.json")))
        p = os.path.join(DD, f"{data}_{s}.llmrw.json")
        rw = json.load(open(p)) if os.path.exists(p) else {}
        eval_split = s != "train"
        per = {}
        for c in convs:
            for i, t in enumerate(c["turns"]):
                for name, txt in query_variants(c, i, rw).items():
                    if eval_split or name == "concat":
                        per.setdefault(name, ([], []))
                        per[name][0].append(key(c, i)); per[name][1].append(txt)
        for name, (keys, texts) in per.items():
            Qs.append(encode_queries(texts)); tags.append((s, name, keys))
        plan[s] = convs
    Q = torch.cat(Qs); print("searching", Q.shape, flush=True)
    sc, ids = idx.search(Q, k=100)
    off, res = 0, {}
    for s, name, keys in tags:
        I = ids[off:off + len(keys)]; off += len(keys)
        tmap = {key(c, i): t for c in plan[s] for i, t in enumerate(c["turns"])}
        if name == "concat":
            json.dump({k: [int(x) for x in r[:100]] for k, r in zip(keys, I)}, open(os.path.join(DD, f"hn_{data}_{s}.json"), "w"))
            json.dump({k: [int(x) for x in r[:3]] for k, r in zip(keys, I)}, open(os.path.join(DD, f"fb_{data}_{s}.json"), "w"))
        if s != "train":
            kk = [j for j, k in enumerate(keys) if tmap[k]["gold"]]
            m = metrics(I[kk], [idx.pid(tmap[keys[j]]["gold"]) for j in kk])
            res[f"{s}/{name}"] = summarize(m)
            np.save(os.path.join(ROOT, "runs", f"zs_{data}_{s}_{name}.npy"),
                    {"keys": [keys[j] for j in kk], "ids": I[kk][:, :10], "per_q": m})
            print(s, name, res[f"{s}/{name}"], flush=True)
    json.dump(res, open(os.path.join(ROOT, "runs", f"zeroshot_{data}.json"), "w"), indent=1)
    # passage store: top-50 hard negatives + feedback + golds
    need = set()
    for s in SPLITS[data]:
        for r in json.load(open(os.path.join(DD, f"hn_{data}_{s}.json"))).values():
            need.update(r[:50])
        if data == "orquac":
            for c in plan[s]:
                for t in c["turns"]:
                    need.update(idx.pid(t["gold"]))
    want, E = idx.lookup(sorted(need))
    add_pstore(data, want, E)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    cmd, data = sys.argv[1], sys.argv[2]
    {"gold": lambda: gold_topiocqa(), "teacher": lambda: teacher(data), "zeroshot": lambda: zeroshot(data)}[cmd]()
