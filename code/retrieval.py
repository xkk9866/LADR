"""Exact inner-product search over the sharded fp16 index (GPU, streamed shard by shard),
plus standard metrics (MRR, NDCG@3, Recall@10/100), passage-embedding lookup and query encoding."""
import os, glob, json, math, numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QPREFIX = "Represent this sentence for searching relevant passages: "


class Index:
    def __init__(self, name):
        self.dir = os.path.join(ROOT, "data", f"index_{name}")
        self.embs = sorted(f for f in glob.glob(os.path.join(self.dir, "emb_*.npy")) if not f.endswith((".i8.npy", ".tmp.npy")))
        i8 = [f[:-4] + ".i8.npy" for f in self.embs]
        self.int8 = bool(i8) and all(os.path.exists(f) for f in i8)
        self.search_files = i8 if self.int8 else self.embs
        self.ids = []
        for e in self.embs:
            p = os.path.join(os.path.dirname(e), os.path.basename(e).replace("emb_", "ids_"))
            if os.path.exists(p):
                self.ids.append(np.load(p))
            else:  # OR-QuAC string ids, mapped to ints through a global table
                self.ids.append(np.array(self._str2int(json.load(open(p[:-4] + ".json"))), dtype=np.int64))
        self._lookup = None

    def _str2int(self, xs):
        if not hasattr(self, "_s2i"):
            self._s2i, self.i2s = {}, []
        out = []
        for x in xs:
            if x not in self._s2i:
                self._s2i[x] = len(self.i2s); self.i2s.append(x)
            out.append(self._s2i[x])
        return out

    def pid(self, raw):
        """Map dataset passage ids to index ids (identity for TopiOCQA)."""
        if hasattr(self, "_s2i"):
            return [r if isinstance(r, (int, np.integer)) else self._s2i[r] for r in raw
                    if isinstance(r, (int, np.integer)) or r in self._s2i]
        return list(raw)

    @torch.no_grad()
    def search(self, Q, k=100, dev="cuda", bs=1024):
        """Q: (n, d) float tensor (normalised). Returns scores (n,k), ids (n,k) numpy."""
        Q = torch.as_tensor(Q, dtype=torch.float32, device=dev)
        n = Q.shape[0]
        if self.int8:  # _int_mm needs >16 rows and multiples of 8
            Q = torch.cat([Q, Q.new_zeros((-n) % 32 + (32 if n < 32 else 0), Q.shape[1])])
            Q = (Q * 127).round().clamp(-127, 127).to(torch.int8)
        else:
            Q = Q.half()
        rows = Q.shape[0]
        best_s = torch.full((rows, k), -1e4, device=dev, dtype=torch.float32)
        best_i = torch.zeros((rows, k), device=dev, dtype=torch.int64)
        for e, ids in zip(self.search_files, self.ids):
            E = torch.from_numpy(np.ascontiguousarray(np.load(e, mmap_mode="r"))).to(dev)
            if self.int8:
                pad = (-E.shape[0]) % 8
                Et = torch.cat([E, E.new_zeros(pad, E.shape[1])]).T.contiguous() if pad else E.T.contiguous()
            I = torch.from_numpy(ids).to(dev)
            for s in range(0, Q.shape[0] if self.int8 else n, bs):
                sc = torch._int_mm(Q[s:s + bs], Et)[:, :E.shape[0]] if self.int8 else Q[s:s + bs] @ E.T
                ts, ti = sc.topk(k, dim=1); ts = ts.float() / (127 * 127 if self.int8 else 1); del sc
                cs = torch.cat([best_s[s:s + bs], ts], 1); ci = torch.cat([best_i[s:s + bs], I[ti]], 1)
                bs_, bi_ = cs.topk(k, dim=1)
                best_s[s:s + bs] = bs_; best_i[s:s + bs] = ci.gather(1, bi_)
            del E
            if self.int8: del Et
        return best_s[:n].cpu().numpy(), best_i[:n].cpu().numpy()

    def lookup(self, pids):
        """Fetch passage embeddings for a set of index ids (returns dict id->row in matrix)."""
        want = np.unique(np.asarray(pids, dtype=np.int64))
        out = np.zeros((len(want), 768), dtype=np.float16)
        pos = {p: i for i, p in enumerate(want)}
        for e, ids in zip(self.embs, self.ids):
            m = np.isin(ids, want)
            if m.any():
                rows = np.nonzero(m)[0]
                E = np.load(e, mmap_mode="r")
                out[[pos[x] for x in ids[rows]]] = E[rows]
        return want, out


def metrics(ranked, golds):
    """ranked: (n,k) ids; golds: list of lists. Returns per-query dict arrays."""
    R = {"MRR": [], "NDCG@3": [], "R@10": [], "R@100": []}
    for r, g in zip(ranked, golds):
        g = set(g)
        hits = [1 if x in g else 0 for x in r]
        first = next((i for i, h in enumerate(hits) if h), None)
        R["MRR"].append(0 if first is None else 1 / (first + 1))
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits[:3]))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(3, len(g))))
        R["NDCG@3"].append(dcg / idcg if idcg else 0)
        R["R@10"].append(sum(hits[:10]) / max(1, len(g)))
        R["R@100"].append(sum(hits[:100]) / max(1, len(g)))
    return {k: np.array(v) for k, v in R.items()}


def summarize(m):
    return {k: round(100 * float(v.mean()), 2) for k, v in m.items()}


_tok = _enc = None


@torch.no_grad()
def encode_queries(texts, prefix=True, bs=256, maxlen=512):
    global _tok, _enc
    from transformers import AutoTokenizer, AutoModel
    if _enc is None:
        _tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
        _enc = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5", dtype=torch.float16).cuda().eval()
    out = []
    for i in range(0, len(texts), bs):
        b = _tok([(QPREFIX if prefix else "") + t for t in texts[i:i + bs]], padding=True, truncation=True,
                 max_length=maxlen, return_tensors="pt").to("cuda")
        h = _enc(**b).last_hidden_state[:, 0].float()
        out.append(torch.nn.functional.normalize(h, dim=-1).cpu())
    return torch.cat(out)


def to_int8(name):
    """Quantise an fp16 index to int8 (symmetric, scale 127; embeddings are unit-norm)."""
    d = os.path.join(ROOT, "data", f"index_{name}")
    for e in sorted(glob.glob(os.path.join(d, "emb_*.npy"))):
        if e.endswith(".i8.npy") or os.path.exists(e[:-4] + ".i8.npy"):
            continue
        x = np.load(e).astype(np.float32)
        np.save(e[:-4] + ".tmp.npy", np.clip(np.round(x * 127), -127, 127).astype(np.int8))
        os.replace(e[:-4] + ".tmp.npy", e[:-4] + ".i8.npy")
        print(e, flush=True)


if __name__ == "__main__":
    import sys
    to_int8(sys.argv[1])


def _q8(Q):
    return (Q.float() * 127).round().clamp(-127, 127).to(torch.int8)


def _intmm(A8, B8t):
    """A8 (n,d) int8 with n % 32 == 0 and n >= 32; B8t (d,m) int8 -> (n,m) int32 (pads m to a multiple of 8)."""
    m = B8t.shape[1]; pad = (-m) % 8
    if pad:
        B8t = torch.cat([B8t, B8t.new_zeros(B8t.shape[0], pad)], 1)
    return torch._int_mm(A8, B8t.contiguous())[:, :m]


class DocIndex:
    """Article-level embeddings (normalised mean of section embeddings) + passage -> article map."""
    def __init__(self, name="topiocqa"):
        d = os.path.join(ROOT, "data", f"docs_{name}")
        p8 = os.path.join(d, "emb.i8.npy")
        if not os.path.exists(p8):
            np.save(p8, np.clip(np.round(np.load(os.path.join(d, "emb.npy")).astype(np.float32) * 127), -127, 127).astype(np.int8))
        self.E8 = np.load(p8, mmap_mode="r")
        self.E16 = np.load(os.path.join(d, "emb.npy"), mmap_mode="r")
        idx = os.path.join(ROOT, "data", f"index_{name}")
        self.shard_doc = [np.load(f) for f in sorted(glob.glob(os.path.join(idx, "docid_*.npy")))]
        idf = sorted(glob.glob(os.path.join(idx, "ids_*.npy")))
        ids = (np.concatenate([np.load(f) for f in idf]) if idf
               else np.arange(sum(len(x) for x in self.shard_doc)))   # OR-QuAC: index ids are global row numbers
        doc = np.concatenate(self.shard_doc)
        self.p2d = np.full(int(ids.max()) + 1, -1, dtype=np.int32); self.p2d[ids] = doc

    def emb(self, docs):
        return np.asarray(self.E16[np.asarray(docs)])


@torch.no_grad()
def search_hier(index, docs, QF, QG, k=100, dev="cuda", bs=512):
    """Exact hierarchical search: score(p) = qF.p + qG.d(p) with int8 GEMMs for both levels."""
    n = QF.shape[0]
    padn = (-n) % 32 + (32 if n < 32 else 0)
    QF8 = _q8(torch.cat([QF, QF.new_zeros(padn, QF.shape[1])]).to(dev))
    QG8 = _q8(torch.cat([QG, QG.new_zeros(padn, QG.shape[1])]).to(dev))
    rows = QF8.shape[0]
    best_s = torch.full((rows, k), -1e4, device=dev); best_i = torch.zeros((rows, k), device=dev, dtype=torch.int64)
    for f8, ids, dshard in zip(index.search_files, index.ids, docs.shard_doc):
        E = torch.from_numpy(np.ascontiguousarray(np.load(f8, mmap_mode="r"))).to(dev); Et = E.T.contiguous()
        I = torch.from_numpy(ids).to(dev)
        lo, hi = int(dshard.min()), int(dshard.max()) + 1
        Dt = torch.from_numpy(np.ascontiguousarray(docs.E8[lo:hi])).to(dev).T.contiguous()
        inv = torch.from_numpy(dshard.astype(np.int64) - lo).to(dev)
        for s in range(0, rows, bs):
            sp = _intmm(QF8[s:s + bs], Et).float()
            sd = _intmm(QG8[s:s + bs], Dt).float()
            sc = (sp + sd[:, inv]) / (127 * 127)
            ts, ti = sc.topk(k, dim=1); del sp, sd, sc
            cs = torch.cat([best_s[s:s + bs], ts], 1); ci = torch.cat([best_i[s:s + bs], I[ti]], 1)
            b_, bi_ = cs.topk(k, dim=1); best_s[s:s + bs] = b_; best_i[s:s + bs] = ci.gather(1, bi_)
        del E, Et, Dt
    return best_s[:n].cpu().numpy(), best_i[:n].cpu().numpy()


def seg_lse(X, seg, nseg, op="lse"):
    """Row-wise log-sum-exp (or max) of X (n, m) over column segments seg (m,) -> (n, nseg)."""
    idx = seg.long().unsqueeze(0).expand(X.shape[0], -1)   # int64 index (numpy default is int32 on Windows)
    m = X.new_full((X.shape[0], nseg), -1e30).scatter_reduce(1, idx, X.detach(), "amax", include_self=True)
    if op == "max":
        return X.new_full((X.shape[0], nseg), -1e30).scatter_reduce(1, idx, X, "amax", include_self=True)
    e = torch.exp(X - m.gather(1, idx))
    return m + torch.log(X.new_zeros(X.shape[0], nseg).scatter_add(1, idx, e))


class DocStore:
    """All passages of every document that a training / evaluation candidate belongs to, grouped by document.
    rows are sorted by document, so a document is the contiguous block [start[d], end[d])."""
    def __init__(self, name):
        z = np.load(os.path.join(ROOT, "data", f"dstore_{name}_meta.npz"))
        self.pids, self.doc = z["pids"], z["doc"]
        self.E = torch.from_numpy(np.load(os.path.join(ROOT, "data", f"dstore_{name}.npy")))
        self.p2row = np.full(int(self.pids.max()) + 1, -1, dtype=np.int64); self.p2row[self.pids] = np.arange(len(self.pids))
        self.udoc, self.start = np.unique(self.doc, return_index=True)
        self.end = np.append(self.start[1:], len(self.doc))

    def blocks(self, docs):
        """rows and local segment ids of the given documents (in the given order)."""
        j = np.searchsorted(self.udoc, docs)
        s, e = self.start[j], self.end[j]
        rows = np.concatenate([np.arange(a, b) for a, b in zip(s, e)])
        seg = np.repeat(np.arange(len(docs)), e - s)
        return rows, seg


@torch.no_grad()
def search_latent(index, docs, QF, QG, tau_f, tau_g, expert="lse", cond=True, k=100, dk=20, dev="cuda", bs=96):
    """Exact search under the latent-document model
         log P(p|x) = A_d(p) + a_F(p) - B_d(p),   a_F = <qF,p>/tau_f,  B_d = LSE_{p' in d} a_F(p'),
         A_d = LSE_{p' in d} <qG,p'>/tau_g  (expert="lse"; "max": max-pooling; "centroid": <qG,centroid(d)>/tau_g).
    Documents that straddle two shards are completed with the head of the next shard, so every document
    statistic is computed over all of its passages. Also returns the top-dk documents by A and log Z = LSE_d A_d."""
    n = QF.shape[0]
    padn = (-n) % 32 + (32 if n < 32 else 0)
    QF8 = _q8(torch.cat([QF, QF.new_zeros(padn, QF.shape[1])]).to(dev))
    QG8 = _q8(torch.cat([QG, QG.new_zeros(padn, QG.shape[1])]).to(dev))
    rows = QF8.shape[0]; S = 127 * 127
    best_s = torch.full((rows, k), -1e30, device=dev); best_i = torch.zeros((rows, k), device=dev, dtype=torch.int64)
    bd_s = torch.full((rows, dk), -1e30, device=dev); bd_i = torch.zeros((rows, dk), device=dev, dtype=torch.int64)
    logZ = torch.full((rows,), -1e30, device=dev)
    files, ids_l, dsh_l = index.search_files, index.ids, docs.shard_doc
    for si in range(len(files)):
        E = np.load(files[si], mmap_mode="r"); I = ids_l[si]; D = dsh_l[si]
        a = int(np.searchsorted(D, D[0], "right")) if si > 0 and dsh_l[si - 1][-1] == D[0] else 0
        E, I, D = E[a:], I[a:], D[a:]
        if si + 1 < len(files) and dsh_l[si + 1][0] == D[-1]:
            b = int(np.searchsorted(dsh_l[si + 1], D[-1], "right"))
            E = np.concatenate([E, np.load(files[si + 1], mmap_mode="r")[:b]]); I = np.concatenate([I, ids_l[si + 1][:b]])
            D = np.concatenate([D, dsh_l[si + 1][:b]])
        Et = torch.from_numpy(np.ascontiguousarray(E)).to(dev).T.contiguous()
        It = torch.from_numpy(np.asarray(I)).to(dev)
        lo = int(D[0]); nd = int(D[-1]) - lo + 1
        inv = torch.from_numpy(D.astype(np.int64) - lo).to(dev)
        if expert == "centroid":
            Dt = torch.from_numpy(np.ascontiguousarray(docs.E8[lo:lo + nd])).to(dev).T.contiguous()
        for s in range(0, rows, bs):
            aF = _intmm(QF8[s:s + bs], Et).float() / (S * tau_f)
            if expert == "centroid":
                A = _intmm(QG8[s:s + bs], Dt).float() / (S * tau_g)
            else:
                A = seg_lse(_intmm(QG8[s:s + bs], Et).float() / (S * tau_g), inv, nd, "max" if expert == "max" else "lse")
            sc = aF + A[:, inv]
            if cond:
                sc -= seg_lse(aF, inv, nd)[:, inv]
            del aF
            ts, ti = sc.topk(k, dim=1); del sc
            cs = torch.cat([best_s[s:s + bs], ts], 1); ci = torch.cat([best_i[s:s + bs], It[ti]], 1)
            b_, bi_ = cs.topk(k, dim=1); best_s[s:s + bs] = b_; best_i[s:s + bs] = ci.gather(1, bi_)
            logZ[s:s + bs] = torch.logaddexp(logZ[s:s + bs], torch.logsumexp(A, 1))
            ds_, di_ = A.topk(min(dk, nd), dim=1)
            cs = torch.cat([bd_s[s:s + bs], ds_], 1); ci = torch.cat([bd_i[s:s + bs], di_ + lo], 1)
            b_, bi_ = cs.topk(dk, dim=1); bd_s[s:s + bs] = b_; bd_i[s:s + bs] = ci.gather(1, bi_)
            del A
        del Et
    return (best_s[:n].cpu().numpy(), best_i[:n].cpu().numpy(),
            bd_s[:n].cpu().numpy(), bd_i[:n].cpu().numpy(), logZ[:n].cpu().numpy())
