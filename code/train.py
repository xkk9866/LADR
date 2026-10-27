"""Train / evaluate PRISM and the trained query-side baselines on a frozen passage index.
python train.py --kind prism --data topiocqa --name prism_full [--no_surprise ...]"""
import os, json, time, random, argparse, numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score
import prism as P
from retrieval import Index, metrics, summarize, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="prism"); ap.add_argument("--data", default="topiocqa")
ap.add_argument("--name", required=True); ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--lr_new", type=float, default=1e-3); ap.add_argument("--n_hn", type=int, default=4)
ap.add_argument("--tau", type=float, default=0.02); ap.add_argument("--lam_pred", type=float, default=0.2)
ap.add_argument("--lam_kd", type=float, default=1.0); ap.add_argument("--lam_shift", type=float, default=0.2)
ap.add_argument("--n_scales", type=int, default=3); ap.add_argument("--m", type=int, default=4)
ap.add_argument("--k_split", type=int, default=4)
ap.add_argument("--text_budget", type=int, default=256); ap.add_argument("--lr_gate", type=float, default=2e-2)
ap.add_argument("--lam_ant", type=float, default=0.3)
ap.add_argument("--mih", type=int, default=0, help="number of intent hypotheses (0 = single query)")
ap.add_argument("--lam_doc", type=float, default=1.0)
ap.add_argument("--lam_mix", type=float, default=1.0); ap.add_argument("--concat_len", type=int, default=384)
ap.add_argument("--holdout", type=int, default=0, help="hold out the last N training conversations as validation")
ap.add_argument("--expert", default="lse", help="latent model document expert: lse | max | centroid")
ap.add_argument("--cond", type=int, default=1, help="latent model: normalise the focus distribution within each document")
ap.add_argument("--learn_tau", type=int, default=1)
for f in ["latent", "two_level", "shared_level", "views", "kalman", "use_rw", "no_ant", "no_mem", "single_stream", "no_return", "no_block", "no_surprise", "no_gate_attn", "no_gate_write", "no_order", "mlp_pred", "no_feedback", "no_kd",
          "no_shift", "no_pred", "no_hn", "eval_only"]:
    ap.add_argument("--" + f, action="store_true")
ap.add_argument("--eval_data", default=None, help="evaluate on another dataset (cross-dataset)")
ap.add_argument("--eval_split", default=None); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max_turns", type=int, default=25)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--log_every", type=int, default=50); ap.add_argument("--no_eval", action="store_true")
args = ap.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
DD = os.path.join(ROOT, "data"); OUT = os.path.join(ROOT, "runs", args.name); os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(P.BGE)
SPLITS = {"topiocqa": ("train", "dev"), "orquac": ("train", "test")}


def key(c, i):
    return f"{c['conv_id']}|{i}"


_IDX = {}


def _index(data):
    if data not in _IDX:
        _IDX[data] = Index(data)
    return _IDX[data]


def load_side(data, split):
    convs = json.load(open(os.path.join(DD, f"{data}_{split}.json")))
    if data == "orquac":  # string passage ids -> integer index ids
        idx = _index(data)
        for c in convs:
            for t in c["turns"]:
                t["gold"] = idx.pid(t["gold"])
    side = {}
    for nm in ["hn", "fb"]:
        p = os.path.join(DD, f"{nm}_{data}_{split}.json")
        side[nm] = json.load(open(p)) if os.path.exists(p) else {}
    p = os.path.join(DD, f"{data}_{split}.llmrw.json")
    rw = json.load(open(p)) if os.path.exists(p) else {}
    if args.use_rw:   # attach the LLM intent hypothesis to every turn
        for c in convs:
            for i, t in enumerate(c["turns"]):
                t["rw"] = rw.get(key(c, i), "")
    p = os.path.join(DD, f"teacher_{data}_{split}.npy")
    side["teacher"] = (np.load(p, allow_pickle=True).item() if os.path.exists(p) else {})
    return convs, side


class PStore:
    """Passage embeddings as one fp16 matrix + (data, id) -> row map (low memory)."""
    def __init__(self):
        self.mats, self.rows = [], {}

    def add(self, d, ids, embs):
        k = len(self.mats); self.mats.append(embs)
        self.rows.update({(d, int(i)): (k, j) for j, i in enumerate(ids)})

    def __contains__(self, key):
        return key in self.rows

    def __getitem__(self, key):
        k, j = self.rows[key]; return self.mats[k][j]

    def __len__(self):
        return len(self.rows)


pstore = PStore()
DSTORE = None
if args.latent and not args.eval_only:
    from retrieval import DocStore
    DSTORE = DocStore(args.data)
    pstore.add(args.data, DSTORE.pids, DSTORE.E.numpy())
for d in ({args.data, args.eval_data or args.data} if DSTORE is None else set()):
    p = os.path.join(DD, f"pstore_{d}.npz")
    if os.path.exists(p):
        z = np.load(p); pstore.add(d, z["ids"], z["embs"])
    elif os.path.exists(os.path.join(DD, f"dstore_{d}.npy")):   # server pipeline: the document store is the passage store
        z = np.load(os.path.join(DD, f"dstore_{d}_meta.npz")); pstore.add(d, z["pids"], np.load(os.path.join(DD, f"dstore_{d}.npy")))
print("pstore", len(pstore), flush=True)


def turn_text(c, i):
    t = c["turns"]
    if args.kind == "concat":
        parts = ([t[i]["rw"]] if args.use_rw and t[i].get("rw") else []) + [t[i]["q"]]
        for j in range(i - 1, -1, -1):
            parts += [t[j]["q"], t[j]["a"]]
        return " [SEP] ".join(parts), args.concat_len
    if i == 0:
        return t[i]["q"], 128
    return f"{t[i]['q']} [SEP] {t[i-1]['q']} [SEP] {t[i-1]['a']}", 128   # constant-size window; the rest lives in the state


_tokcache = {}


def _toks(text, n):
    k = (text, n)
    if k not in _tokcache:
        _tokcache[k] = tok(text, add_special_tokens=False, truncation=True, max_length=n)["input_ids"]
    return _tokcache[k]


def aged_tokens(c, i):
    """[CLS] q_t [SEP] q_{t-1} [SEP] a_{t-1} [SEP] ... with the turn age of every token.
    Turns that do not fit the text budget become memory turns (compressed to one intent token each),
    or are dropped when the hierarchical memory is disabled."""
    t = c["turns"]
    budget = args.text_budget
    x = [tok.cls_token_id] + _toks(t[i]["q"], 64) + [tok.sep_token_id]
    if args.use_rw and t[i].get("rw"):   # LLM intent hypothesis is part of the current-turn observation
        x = [tok.cls_token_id] + _toks(t[i]["rw"], 64) + [tok.sep_token_id] + x[1:]
    a = [0] * len(x); mem = []
    full = True
    for j in range(i - 1, -1, -1):
        seg = _toks(t[j]["q"], 64) + [tok.sep_token_id] + _toks(t[j]["a"], 96) + [tok.sep_token_id]
        if full and len(x) + len(seg) <= budget:
            x += seg; a += [i - j] * len(seg)
        else:
            full = False
            if args.no_mem: break
            mem.append(([tok.cls_token_id] + _toks(t[j]["q"], 32) + [tok.sep_token_id] + _toks(t[j]["a"], 48) + [tok.sep_token_id], i - j))
    return x, a, mem


def make_batch(convs, data, side, train):
    pos = [[len(c["turns"]) - 1] if c.get("_only_last") else list(range(min(args.max_turns, len(c["turns"]))))
           for c in convs]
    T = max(len(p) for p in pos); B = len(convs)
    texts, valid, items = [], torch.zeros(T, B, dtype=torch.bool), []
    maxlen = 96
    for t in range(T):
        for b, c in enumerate(convs):
            if t < len(pos[b]):
                i = pos[b][t]
                s, maxlen = turn_text(c, i); texts.append(s); valid[t, b] = True; items.append((t, b, c, i))
            else:
                texts.append("")
    if args.kind == "prism":
        seqs = [aged_tokens(c, i) for (t, b, c, i) in items]
        L = max(len(x[0]) for x in seqs)
        ids = torch.zeros(T, B, L, dtype=torch.long); att = torch.zeros(T, B, L, dtype=torch.long)
        age = torch.full((T, B, L), -1, dtype=torch.long)
        M = max(len(x[2]) for x in seqs); Lm = max([len(m[0]) for x in seqs for m in x[2]] + [1])
        mids = torch.zeros(T, B, M, Lm, dtype=torch.long); matt = torch.zeros(T, B, M, Lm, dtype=torch.long)
        mage = torch.full((T, B, M), -1, dtype=torch.long)
        for (t, b, c, i), (x, a, mem) in zip(items, seqs):
            ids[t, b, :len(x)] = torch.tensor(x); att[t, b, :len(x)] = 1; age[t, b, :len(x)] = torch.tensor(a)
            for k, (mx, ma) in enumerate(mem):
                mids[t, b, k, :len(mx)] = torch.tensor(mx); matt[t, b, k, :len(mx)] = 1; mage[t, b, k] = ma
        out = dict(ids=ids, att=att, age=age, valid=valid, mem_ids=mids, mem_att=matt, mem_age=mage)
        if not args.no_feedback:
            K = 3; fb = torch.zeros(T, B, K, 768); fbm = torch.zeros(T, B, K, dtype=torch.bool)
            for (t, b, c, i) in items:
                if i == 0: continue
                for j, p in enumerate(side["fb"].get(key(c, i - 1), [])[:K]):
                    if (data, p) in pstore:
                        fb[t, b, j] = torch.from_numpy(pstore[(data, p)].astype(np.float32)); fbm[t, b, j] = True
            out["fb"], out["fb_mask"] = fb, fbm
        return out, items
    enc = tok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
    if args.two_level and args.kind == "concat":   # topic token right after [CLS]
        TOPIC = tok.convert_tokens_to_ids("[unused0]")
        ii, aa = enc["input_ids"], enc["attention_mask"]
        enc = {"input_ids": torch.cat([ii[:, :1], torch.full_like(ii[:, :1], TOPIC), ii[:, 1:]], 1),
               "attention_mask": torch.cat([aa[:, :1], aa[:, :1], aa[:, 1:]], 1)}
    L = enc["input_ids"].shape[1]
    batch = dict(ids=enc["input_ids"].view(T, B, L), att=enc["attention_mask"].view(T, B, L), valid=valid)
    if args.kind == "prism_rec" and not args.no_feedback:
        K = 3; fb = torch.zeros(T, B, K, 768); fbm = torch.zeros(T, B, K, dtype=torch.bool)
        for (t, b, c, i) in items:
            if i == 0: continue
            pids = side["fb"].get(key(c, i - 1), [])[:K]
            for j, p in enumerate(pids):
                if (data, p) in pstore:
                    fb[t, b, j] = torch.from_numpy(pstore[(data, p)].astype(np.float32)); fbm[t, b, j] = True
        batch["fb"], batch["fb_mask"] = fb, fbm
    return batch, items


def run_model(model, batch):
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if args.kind == "concat":  # every turn is independent: flatten to (1, T*B)
            T, B, L = batch["ids"].shape
            o = model(dict(ids=batch["ids"].reshape(1, T * B, L), att=batch["att"].reshape(1, T * B, L)))
            o = {k: v.reshape(T, B, -1) for k, v in o.items()}
        elif args.kind == "prism":
            T, B, L = batch["ids"].shape
            inp = {k: batch[k].reshape(T * B, L) for k in ("ids", "att", "age")}
            M = batch["mem_ids"].shape[2]
            Lm = batch["mem_ids"].shape[3]; inp["mem_ids"] = batch["mem_ids"].reshape(T * B, M, Lm); inp["mem_att"] = batch["mem_att"].reshape(T * B, M, Lm)
            inp["mem_age"] = batch["mem_age"].reshape(T * B, M)
            if "fb" in batch:
                inp["fb"] = batch["fb"].reshape(T * B, *batch["fb"].shape[2:]); inp["fb_mask"] = batch["fb_mask"].reshape(T * B, -1)
            o = model(inp)
            o = {k: v.reshape(T, B, *v.shape[1:]) for k, v in o.items() if k in ("q", "u", "u_hat", "sigma", "q_prior", "gain", "logpi", "q_h")}
        else:
            o = model(batch)
    return o


LOGS_EXTRA = {}
DOCS = None


def train_step(model, convs, side, opt, sched):
    _SIDE["teacher"] = side["teacher"]
    batch, items = make_batch(convs, args.data, side, True)
    o = run_model(model, batch)
    items = [x for x in items if x[2]["turns"][x[3]]["gold"]]
    q = torch.stack([o["q"][t, b] for (t, b, c, i) in items]).float()
    golds = [c["turns"][i]["gold"] for (t, b, c, i) in items]
    pos_ids = [g[0] for g in golds]
    neg_ids = []
    if not args.no_hn:
        for (t, b, c, i), g in zip(items, golds):
            hn = [p for p in side["hn"].get(key(c, i), []) if p not in g and (args.data, p) in pstore]
            neg_ids += random.sample(hn[:50], min(args.n_hn, len(hn[:50])))
    cand = list(dict.fromkeys(pos_ids + neg_ids))
    pos = {p: j for j, p in enumerate(cand)}
    E = torch.from_numpy(np.stack([pstore[(args.data, p)] for p in cand]).astype(np.float32)).cuda()
    if args.latent:
        return latent_step(model, o, items, golds, pos_ids, neg_ids, cand, q, opt, sched)
    if "qg" in o:
        # hierarchical intent: score(p) = qF.p + qG.d(p); plus an article-level contrast for the topic intent
        qg = torch.stack([o["qg"][t, b] for (t, b, c, i) in items]).float()
        cd = DOCS.p2d[np.array(cand)]
        ud, inv = np.unique(cd, return_inverse=True)
        Dm = torch.from_numpy(DOCS.emb(ud).astype(np.float32)).cuda()
        sdoc = qg @ Dm.T                                                     # (N, n_docs)
        logits = (q @ E.T + sdoc[:, torch.from_numpy(inv).cuda()]) / args.tau
        gd = torch.from_numpy(np.searchsorted(ud, DOCS.p2d[np.array(pos_ids)])).cuda()
        ld_logits = sdoc / args.tau
        LOGS_EXTRA["doc"] = F.cross_entropy(ld_logits, gd)
    elif q.dim() == 3:
        logits = (q @ E.T).amax(1) / args.tau
    else:
        logits = q @ E.T / args.tau
    gold_mask = torch.zeros_like(logits, dtype=torch.bool)
    for r, g in enumerate(golds):
        for p in g:
            if p in pos: gold_mask[r, pos[p]] = True
    tgt = torch.tensor([pos[p] for p in pos_ids], device="cuda")
    own = F.one_hot(tgt, len(cand)).bool()
    logits = logits.masked_fill(gold_mask & ~own, -1e4)   # other positives of the same query are not negatives
    loss = F.cross_entropy(logits, tgt); logs = {"rank": loss.item()}
    if "doc" in LOGS_EXTRA:
        ld = LOGS_EXTRA.pop("doc"); loss = loss + args.lam_doc * ld; logs["doc"] = ld.item()
    if "logpi" in o:
        # mixture of intent hypotheses: p(gold) = sum_k pi_k softmax_k(gold); its gradient w.r.t. pi is the EM
        # responsibility of each hypothesis, and each hypothesis is trained in proportion to its responsibility
        lp = torch.stack([o["logpi"][t, b] for (t, b, c, i) in items]).float()
        qh = torch.stack([o["q_h"][t, b] for (t, b, c, i) in items]).float()
        Lk = ((qh @ E.T) / args.tau).masked_fill((gold_mask & ~own)[:, None, :], -1e4)     # (N,K,C)
        logp_k = F.log_softmax(Lk, -1).gather(2, tgt[:, None, None].expand(-1, Lk.shape[1], 1)).squeeze(-1)
        nll = -torch.logsumexp(lp + logp_k, 1).mean()
        loss = loss + args.lam_mix * nll; logs["mix"] = nll.item()
        logs["H(pi)"] = float(-(lp.exp() * lp).sum(1).mean())
    if not args.no_kd and side["teacher"]:
        tk = [side["teacher"].get(key(c, i)) for (t, b, c, i) in items]
        idx = [j for j, x in enumerate(tk) if x is not None]
        if idx:
            te = torch.from_numpy(np.stack([tk[j] for j in idx]).astype(np.float32)).cuda()
            qk = q[:, 0] if q.dim() == 3 else q
            kd = F.mse_loss(qk[idx], te) * 768 / 10
            loss = loss + args.lam_kd * kd; logs["kd"] = kd.item()
    if args.kind in ("prism", "prism_rec"):
        sel = [(t, b) for (t, b, c, i) in items if i > 0]
        if sel and not args.no_pred and not args.no_surprise:
            ui = torch.stack([o["u"][t, b] for t, b in sel]).float(); uh = torch.stack([o["u_hat"][t, b] for t, b in sel]).float()
            lp = F.cross_entropy(F.normalize(uh, dim=-1) @ F.normalize(ui.detach(), dim=-1).T / 0.1,
                                 torch.arange(len(sel), device="cuda"))
            loss = loss + args.lam_pred * lp; logs["pred"] = lp.item()
        if sel and not args.no_shift and not args.no_surprise and args.data == "topiocqa":
            y = torch.tensor([float(c["turns"][i]["shift"] > 0) for (t, b, c, i) in items if i > 0], device="cuda")
            s = torch.stack([o["sigma"][t, b] for t, b in sel]).float()
            ls = F.binary_cross_entropy_with_logits(s, y)
            loss = loss + args.lam_shift * ls; logs["shift"] = ls.item()
        if sel and "q_prior" in o and args.kalman:
            # anticipatory retrieval: the prior (history trajectory only) must already find the current gold passage
            qp = torch.stack([o["q_prior"][t, b] for t, b in sel]).float()
            rows = [r for r, (t, b, c, i) in enumerate(items) if i > 0]
            la = F.cross_entropy((qp @ E.T / args.tau).masked_fill((gold_mask & ~own)[rows], -1e4), tgt[rows])
            loss = loss + args.lam_ant * la; logs["ant"] = la.item()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return logs


def latent_step(model, o, items, golds, pos_ids, neg_ids, cand, q, opt, sched):
    """Latent-document likelihood. The topic intent defines a coarse distribution over passages whose mass,
    aggregated per document, is the document posterior P(d|x); the focus intent defines a fine distribution
    normalised inside each document, P(p|d,x). The loss is the exact negative log-likelihood
    -log P(d+|x) - log P(p+|d+,x), with the document posterior normalised over the in-batch candidate documents."""
    from retrieval import seg_lse
    qg = torch.stack([o["qg"][t, b] for (t, b, c, i) in items]).float()
    m = model.module if hasattr(model, "module") else model
    tg, tf = m.log_tau_g.exp(), m.log_tau_f.exp()
    p2d = lambda ps: DSTORE.doc[DSTORE.p2row[np.asarray(ps)]]
    U = np.unique(p2d(cand))
    rows, seg = DSTORE.blocks(U)
    E = DSTORE.E[torch.from_numpy(rows)].cuda().float(); seg_t = torch.from_numpy(seg).cuda()
    aF = q @ E.T / tf                                                       # (N, Np) fine scores
    if args.expert == "centroid":
        A = qg @ torch.from_numpy(DOCS.emb(U).astype(np.float32)).cuda().T / tg
    else:
        A = seg_lse(qg @ E.T / tg, seg_t, len(U), "max" if args.expert == "max" else "lse")   # (N, |U|) log P(d|x) + c
    rpos = {r: j for j, r in enumerate(rows)}
    N = len(items)
    tdoc = torch.from_numpy(np.searchsorted(U, p2d(pos_ids))).cuda()
    gmask = torch.zeros(N, len(U), dtype=torch.bool, device="cuda")       # documents holding another gold passage
    pmask = torch.zeros(N, len(rows), dtype=torch.bool, device="cuda")    # gold passages inside the target document
    for r, g in enumerate(golds):
        for x in [x for x in g if x < len(DSTORE.p2row) and DSTORE.p2row[x] >= 0]:
            j = int(np.searchsorted(U, p2d([x])[0])); gmask[r, j] = True
            if j == int(tdoc[r]): pmask[r, rpos[int(DSTORE.p2row[x])]] = True
    gmask[torch.arange(N), tdoc] = False
    logs = {}
    if args.cond:
        l_doc = F.cross_entropy(A.masked_fill(gmask, -1e4), tdoc)
        B = seg_lse(aF, seg_t, len(U))
        l_in = (B.gather(1, tdoc[:, None]).squeeze(1) - torch.logsumexp(aF.masked_fill(~pmask, -1e30), 1)).mean()
        loss = l_doc + l_in; logs.update(doc=l_doc.item(), within=l_in.item())
    else:   # joint product of experts over the candidate passages (no within-document normalisation)
        cidx = torch.tensor([rpos[int(DSTORE.p2row[p])] for p in cand], device="cuda")
        cdoc = torch.from_numpy(np.searchsorted(U, p2d(cand))).cuda()
        logits = aF[:, cidx] + A[:, cdoc]
        pos = {p: j for j, p in enumerate(cand)}
        gold_mask = torch.zeros_like(logits, dtype=torch.bool)
        for r, g in enumerate(golds):
            for p in g:
                if p in pos: gold_mask[r, pos[p]] = True
        tgt = torch.tensor([pos[p] for p in pos_ids], device="cuda")
        logits = logits.masked_fill(gold_mask & ~F.one_hot(tgt, len(cand)).bool(), -1e4)
        loss = F.cross_entropy(logits, tgt); logs["rank"] = loss.item()
    logs.update(tau_g=tg.item(), tau_f=tf.item())
    if not args.no_kd and side_teacher():
        tk = [side_teacher().get(key(c, i)) for (t, b, c, i) in items]
        idx = [j for j, x in enumerate(tk) if x is not None]
        if idx:
            te = torch.from_numpy(np.stack([tk[j] for j in idx]).astype(np.float32)).cuda()
            kd = F.mse_loss(q[idx], te) * 768 / 10
            loss = loss + args.lam_kd * kd; logs["kd"] = kd.item()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return logs


_SIDE = {}


def side_teacher():
    return _SIDE.get("teacher", {})


@torch.no_grad()
def encode_split(model, data, convs, side):
    model.eval(); Q, meta = {}, {}
    global LP, QG
    LP, QG = {}, {}
    bs = args.bs * 4 if args.kind not in ("concat", "prism") else 8
    for s in range(0, len(convs), bs):
        chunk = convs[s:s + bs]
        batch, items = make_batch(chunk, data, side, False)
        o = run_model(model, batch)
        for (t, b, c, i) in items:
            Q[key(c, i)] = o["q"][t, b].float().cpu()
            if "qg" in o:
                QG[key(c, i)] = o["qg"][t, b].float().cpu()
            if "logpi" in o:
                LP[key(c, i)] = o["logpi"][t, b].float().cpu()
            if "sigma" in o:
                meta[key(c, i)] = dict(sigma=o["sigma"][t, b].item())
                if "gain" in o: meta[key(c, i)]["gain"] = o["gain"][t, b].item()
                if "g" in o:
                    meta[key(c, i)].update(g=o["g"][t, b].float().tolist(), alpha=o["alpha"][t, b].float().tolist())
    model.train()
    return Q, meta


def evaluate(model, data, split, tag, index=None):
    if split == "val":   # held-out tail of the training conversations
        convs, side = load_side(data, SPLITS[data][0]); convs = convs[-args.holdout:]
    else:
        convs, side = load_side(data, split)
    Q, meta = encode_split(model, data, convs, side)
    keys = [key(c, i) for c in convs for i, t in enumerate(c["turns"]) if t["gold"]]
    idx = index or _index(data)
    QQ = torch.stack([Q[k] for k in keys])
    if QG and args.latent:
        from retrieval import search_latent
        sc, ids, dsc, dids, logZ = search_latent(idx, DOCS, QQ, torch.stack([QG[k] for k in keys]),
                                                 model.log_tau_f.exp().item(), model.log_tau_g.exp().item(),
                                                 expert=args.expert, cond=bool(args.cond))
        np.save(os.path.join(OUT, f"docpost_{data}_{split}.npy"), {"keys": keys, "doc_ids": dids, "doc_logit": dsc, "logZ": logZ})
    elif QG:
        from retrieval import search_hier
        sc, ids = search_hier(idx, DOCS, QQ, torch.stack([QG[k] for k in keys]))
    if LP:
        for k in keys:
            meta.setdefault(k, {})["pi"] = np.exp(LP[k].numpy()).round(4).tolist()
    if QQ.dim() == 3 and LP:   # mixture of hypotheses: per-hypothesis top-k, union, exact mixture score
        n, V, _ = QQ.shape
        LPq = torch.stack([LP[k] for k in keys]).numpy()
        s_, i_ = idx.search(QQ.reshape(n * V, -1), k=100)
        s_, i_ = s_.reshape(n, V, 100), i_.reshape(n, V, 100)
        ids = np.zeros((n, 100), dtype=np.int64); sc = np.zeros((n, 100), dtype=np.float32)
        for r in range(n):
            floor = s_[r, :, -1]                               # unseen candidate: bounded by the k-th score
            cand = np.unique(i_[r].reshape(-1))
            S = np.tile(floor[:, None], (1, len(cand)))
            pos = {p: j for j, p in enumerate(cand)}
            for v in range(V):
                for p, val in zip(i_[r, v], s_[r, v]):
                    S[v, pos[p]] = val
            mix = np.logaddexp.reduce(LPq[r][:, None] + S / args.tau, axis=0)
            o_ = np.argsort(-mix)[:100]
            ids[r, :len(o_)] = cand[o_]; sc[r, :len(o_)] = mix[o_]
        for k in keys:
            meta.setdefault(k, {})["pi"] = np.exp(LP[k].numpy()).round(4).tolist()
    elif QQ.dim() == 3:   # multi-view query: exact max-over-views scoring = per-view top-k merged by max
        n, V, _ = QQ.shape
        s_, i_ = idx.search(QQ.reshape(n * V, -1), k=100)
        s_, i_ = s_.reshape(n, V * 100), i_.reshape(n, V * 100)
        ids = np.zeros((n, 100), dtype=np.int64); sc = np.zeros((n, 100), dtype=np.float32)
        for r in range(n):
            best = {}
            for p, v in zip(i_[r], s_[r]):
                if v > best.get(p, -1e9): best[p] = v
            top = sorted(best.items(), key=lambda x: -x[1])[:100]
            ids[r, :len(top)] = [p for p, _ in top]; sc[r, :len(top)] = [v for _, v in top]
    elif not QG:
        sc, ids = idx.search(QQ, k=100)
    tmap = {key(c, i): t for c in convs for i, t in enumerate(c["turns"])}
    golds = [idx.pid(tmap[k]["gold"]) for k in keys]
    m = metrics(ids, golds); res = {"all": summarize(m)}
    turn_no = np.array([int(k.rsplit("|", 1)[1]) + 1 for k in keys])
    for lo, hi in [(1, 3), (4, 6), (7, 10), (11, 99)]:
        sel = (turn_no >= lo) & (turn_no <= hi)
        if sel.any(): res[f"turn{lo}-{hi}"] = summarize({k: v[sel] for k, v in m.items()})
    if data == "topiocqa":
        sh = np.array([tmap[k]["shift"] for k in keys])
        for v, nm in [(0, "stay"), (1, "new_topic"), (2, "return")]:
            sel = (sh == v) & (turn_no > 1)
            res[nm] = summarize({k: x[sel] for k, x in m.items()})
        if meta:
            sel = turn_no > 1
            y = (sh[sel] > 0).astype(int); s = np.array([meta[k]["sigma"] for k in keys])[sel]
            pr = [((s > th) & (y == 1)).sum() / max(1, (s > th).sum()) for th in np.quantile(s, np.linspace(0.05, 0.95, 91))]
            rc = [((s > th) & (y == 1)).sum() / max(1, y.sum()) for th in np.quantile(s, np.linspace(0.05, 0.95, 91))]
            f1 = max(2 * p * r / max(1e-9, p + r) for p, r in zip(pr, rc))
            res["shift_detect"] = dict(AUROC=round(100 * roc_auc_score(y, s), 2),
                                       AP=round(100 * average_precision_score(y, s), 2), F1=round(100 * f1, 2))
    if args.latent:
        res["tau"] = dict(g=model.log_tau_g.exp().item(), f=model.log_tau_f.exp().item())
    json.dump(res, open(os.path.join(OUT, f"eval_{tag}.json"), "w"), indent=1)
    np.save(os.path.join(OUT, f"rank_{tag}.npy"), {"keys": keys, "ids": ids[:, :10], "per_q": {k: v for k, v in m.items()}, "meta": meta})
    print(tag, json.dumps(res), flush=True)
    return res


def main():
    kw = {}
    if args.kind == "prism":
        kw = dict(k_split=args.k_split, n_scales=args.n_scales, use_surprise=not args.no_surprise,
                  ordered_scales=not args.no_order, predictor="mlp" if args.mlp_pred else "tcn",
                  use_return=not args.no_return, block_phase1=not args.no_block, feedback=not args.no_feedback, dual_stream=not args.single_stream, kalman=args.kalman, views=args.views, mih=args.mih)
    if args.kind == "prism_rec":
        kw = dict(n_scales=args.n_scales, m=args.m, k_split=args.k_split, use_surprise=not args.no_surprise,
                  gate_attn=not args.no_gate_attn, gate_write=not args.no_gate_write, ordered_scales=not args.no_order,
                  predictor="mlp" if args.mlp_pred else "tcn", feedback=not args.no_feedback)
    if args.kind == "concat":
        kw = dict(two_level=args.two_level or args.latent, shared=args.shared_level, tau=args.tau)
    global DOCS
    if args.latent:
        args.two_level = True
    if args.two_level:
        from retrieval import DocIndex
        DOCS = DocIndex((args.eval_data or args.data) if args.eval_only else args.data)
    model = P.build(args.kind, **kw).cuda()
    if args.kind == "concat":
        model.bert.gradient_checkpointing_enable()
    ck = os.path.join(OUT, "model.pt")
    if args.eval_only:
        model.load_state_dict(torch.load(ck, map_location="cuda"))
    else:
        tr_split, _ = SPLITS[args.data]
        convs, side = load_side(args.data, tr_split)
        if args.holdout:
            convs = convs[:-args.holdout]
        if args.limit: convs = convs[:args.limit]
        if args.kind in ("concat", "prism"):   # flatten into single-turn "conversations" (each carries its own history text)
            convs = [dict(conv_id=c["conv_id"], turns=c["turns"], _i=i) for c in convs for i in range(len(c["turns"]))]
        convs = [c for c in convs if all(t["gold"] for t in c["turns"])] if args.data == "topiocqa" else convs
        GATE = ("lam_raw", "rho", "age_bias", "sur_w", "sur_b") + (("log_tau_g", "log_tau_f") if args.learn_tau else ())
        if not args.learn_tau:
            model.log_tau_g.requires_grad_(False); model.log_tau_f.requires_grad_(False)   # scalar gate parameters: own (larger) learning rate
        bert_p = [p for n, p in model.named_parameters() if n.startswith("bert.")]
        gate_p = [p for n, p in model.named_parameters() if n in GATE]
        new_p = [p for n, p in model.named_parameters() if not n.startswith("bert.") and n not in GATE and p.requires_grad]
        groups = [{"params": bert_p, "lr": args.lr}, {"params": new_p, "lr": args.lr_new}]
        if gate_p:
            groups.append({"params": gate_p, "lr": args.lr_gate, "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
        bs = args.bs if args.kind not in ("concat", "prism") else 96
        steps = args.epochs * (len(convs) // bs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / (0.05 * steps)) * max(0.0, 1 - s / steps))
        st, t0 = 0, time.time()
        for ep in range(args.epochs):
            random.shuffle(convs)
            for s in range(0, len(convs) - bs + 1, bs):
                chunk = convs[s:s + bs]
                if args.kind in ("concat", "prism"):
                    chunk = [_single(c) for c in chunk]
                else:
                    chunk = [_drop_goldless(c) for c in chunk]
                logs = train_step(model, chunk, side, opt, sched); st += 1
                if st % args.log_every == 0:
                    print(f"ep{ep} step{st}/{steps} {time.time()-t0:.0f}s", {k: round(v, 4) for k, v in logs.items()}, flush=True)
        torch.save(model.state_dict(), ck)
    if args.no_eval: return
    data = args.eval_data or args.data
    split = args.eval_split or SPLITS[data][1]
    evaluate(model, data, split, f"{data}_{split}")


def _single(c):
    i = c["_i"]
    t = dict(c["turns"][i]); return dict(conv_id=c["conv_id"], turns=c["turns"][:i] + [t], _only_last=True)


def _drop_goldless(c):
    return c


if __name__ == "__main__":
    main()
