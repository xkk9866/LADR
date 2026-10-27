"""PRISM: Predictive Recurrent Intent-State Memory for conversational dense retrieval,
together with the trained query-side baselines that share its training pipeline.

model kinds
  prism   : recurrent multi-timescale memory tokens living inside the query Transformer;
            surprise (prediction error of a causal TCN prior) gates memory attention and memory writes
  concat  : ConvDR-style encoder over the reversed concatenated history (history concatenation)
  gru     : per-turn encoder + GRU over turns (sequential encoding)
  hier    : per-turn encoder + causal Transformer over turns (sequential encoding, attention)
"""
import math, torch, torch.utils.checkpoint, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BGE = "BAAI/bge-base-en-v1.5"
NEG = -1e4


def ordered(raw):
    """Monotone non-decreasing vector from unconstrained params (cumsum of softplus)."""
    return raw[:1] + torch.cat([raw.new_zeros(1), F.softplus(raw[1:])]).cumsum(0)


class CausalTCN(nn.Module):
    """Dilated causal 1-D CNN that predicts the next intent observation from past observations."""
    def __init__(self, d, layers=3, k=2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, dilation=2 ** i) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.k = k
        self.start = nn.Parameter(torch.zeros(d))

    def forward(self, U):  # U: (B, t, d) past observations -> (B, d) prediction for step t+1
        if U.shape[1] == 0:
            return self.start.expand(U.shape[0], -1)
        x = U.transpose(1, 2)
        for i, (c, n) in enumerate(zip(self.convs, self.norms)):
            pad = (self.k - 1) * 2 ** i
            y = c(F.pad(x, (pad, 0)))
            x = n((x + F.gelu(y)).transpose(1, 2)).transpose(1, 2)
        return x[:, :, -1]


class MLPPred(nn.Module):  # ablation: prior from the last observation only
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.start = nn.Parameter(torch.zeros(d))

    def forward(self, U):
        return self.start.expand(U.shape[0], -1) if U.shape[1] == 0 else self.f(U[:, -1])


class PRISM(nn.Module):
    def __init__(self, n_scales=3, m=4, k_split=4, use_surprise=True, gate_attn=True, gate_write=True,
                 ordered_scales=True, predictor="tcn", feedback=True, dp=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BGE, attn_implementation="sdpa")
        d = self.bert.config.hidden_size
        self.d, self.ns, self.m, self.k = d, n_scales, m, k_split
        self.use_surprise, self.gate_attn, self.gate_write = use_surprise, gate_attn, gate_write
        self.ordered_scales, self.feedback = ordered_scales, feedback
        self.S0 = nn.Parameter(torch.randn(n_scales, m, d) * 0.02)
        self.scale_emb = nn.Parameter(torch.randn(n_scales, 1, d) * 0.02)
        self.fb_proj = nn.Linear(d, d); self.fb_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mem_ln = nn.LayerNorm(d)
        self.u_proj = nn.Linear(d, dp)
        self.pred = CausalTCN(dp) if predictor == "tcn" else MLPPred(dp)
        self.sur_w = nn.Parameter(torch.tensor(8.0)); self.sur_b = nn.Parameter(torch.tensor(-3.0))
        # attention gate g = sigmoid(a - b * sigma), write rate alpha = sigmoid(c + e * sigma)
        # scales ordered long(0) -> task(1) -> focus(2): b, c, e non-decreasing
        self.a = nn.Parameter(torch.full((n_scales,), 3.0))
        self.b_raw = nn.Parameter(torch.tensor([0.5] + [0.5] * (n_scales - 1)))
        self.c_raw = nn.Parameter(torch.tensor([-3.0] + [1.0] * (n_scales - 1)))
        self.e_raw = nn.Parameter(torch.tensor([0.0] + [0.5] * (n_scales - 1)))
        self.router = nn.Linear(d, n_scales)
        self.out = nn.Linear(2 * d, d)
        nn.init.zeros_(self.out.weight[:, d:]); self.out.weight.data[:, :d] = torch.eye(d); nn.init.zeros_(self.out.bias)

    def _vec(self, raw, positive=False):
        v = ordered(raw) if self.ordered_scales else raw
        return F.softplus(v) if positive else v

    def gates(self, sigma):  # sigma: (B,) surprise logit
        b = self._vec(self.b_raw, positive=True)
        c = self._vec(self.c_raw)
        e = self._vec(self.e_raw, positive=True)
        g = torch.sigmoid(self.a[None] - b[None] * sigma[:, None])        # (B, ns) attention gate
        alpha = torch.sigmoid(c[None] + e[None] * sigma[:, None])         # (B, ns) write rate
        return g, alpha

    def step(self, ids, att, S, U_hist, fb, fb_mask):
        """One turn for a batch of conversations.
        ids/att: (B,L); S: (B,ns*m,d); U_hist: (B,t,dp); fb: (B,K,d) or None."""
        B, L = ids.shape
        x = self.bert.embeddings(input_ids=ids)
        mem = S + self.scale_emb.repeat_interleave(self.m, 0).reshape(1, -1, self.d)
        parts, key_ok = [x, mem], [att.bool(), torch.ones(B, mem.shape[1], dtype=torch.bool, device=x.device)]
        if self.feedback and fb is not None:
            parts.append(self.fb_proj(fb) + self.fb_type); key_ok.append(fb_mask)
        h = torch.cat(parts, 1)
        N, M = h.shape[1], mem.shape[1]
        key_ok = torch.cat(key_ok, 1)
        base = torch.zeros(B, 1, N, N, device=x.device, dtype=h.dtype).masked_fill(~key_ok[:, None, None, :], NEG)
        # phase 1: text queries see text only -> history-free observation u_t
        p1 = base.clone(); p1[:, :, :L, L:] = NEG
        layers = self.bert.encoder.layer
        for li in range(self.k):
            h = layers[li](h, attention_mask=p1)[0]
        tm = att.unsqueeze(-1).to(h.dtype)
        u = self.u_proj((h[:, :L] * tm).sum(1) / tm.sum(1))                  # (B, dp)
        u_hat = self.pred(U_hist)
        cos = F.cosine_similarity(u_hat, u, dim=-1)
        sigma = self.sur_w * (1 - cos) + self.sur_b if self.use_surprise else torch.zeros_like(cos)
        g, alpha = self.gates(sigma)
        # phase 2: surprise-gated attention from text to memory scales (and to result feedback)
        p2 = base.clone()
        if self.gate_attn:
            lg = torch.log(g.clamp_min(1e-4)).to(h.dtype)                    # (B, ns)
            p2[:, :, :L, L:L + M] += lg.repeat_interleave(self.m, 1)[:, None, None, :]
            if N > L + M:
                p2[:, :, :L, L + M:] += lg[:, -1:][:, None, None, :]
        for li in range(self.k, len(layers)):
            h = layers[li](h, attention_mask=p2)[0]
        cls = h[:, 0]
        S_hat = self.mem_ln(h[:, L:L + M])
        a = alpha if self.gate_write else torch.full_like(alpha, 0.5)
        a = a.repeat_interleave(self.m, 1).unsqueeze(-1)
        S_new = (1 - a) * S + a * S_hat
        Sm = S_new.reshape(B, self.ns, self.m, self.d).mean(2)                # (B, ns, d)
        pi = torch.softmax(self.router(cls), -1)
        q = self.out(torch.cat([cls, (pi.unsqueeze(-1) * Sm).sum(1)], -1))
        return dict(q=F.normalize(q, dim=-1), S=S_new, u=u, u_hat=u_hat, sigma=sigma, g=g, alpha=alpha, pi=pi)

    def forward(self, batch):
        """batch: dict with ids (T,B,L), att (T,B,L), valid (T,B), fb (T,B,K,d) optional, fb_mask (T,B,K)."""
        T, B, _ = batch["ids"].shape
        S = self.S0.reshape(1, -1, self.d).expand(B, -1, -1)
        U = torch.zeros(B, 0, self.u_proj.out_features, device=batch["ids"].device)
        outs = []
        for t in range(T):
            fb = batch["fb"][t] if "fb" in batch else None
            fbm = batch["fb_mask"][t] if "fb_mask" in batch else None
            o = self.step(batch["ids"][t], batch["att"][t], S, U, fb, fbm)
            v = batch["valid"][t].view(B, 1, 1).to(S.dtype)
            S = v * o["S"] + (1 - v) * S
            U = torch.cat([U, o["u"].unsqueeze(1)], 1)
            outs.append(o)
        return {k: torch.stack([o[k] for o in outs]) for k in outs[0] if k != "S"}


class TurnSeq(nn.Module):
    """Per-turn encoder + sequential model over turns ("gru" or "hier")."""
    def __init__(self, kind):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BGE, attn_implementation="sdpa")
        d = self.bert.config.hidden_size
        self.kind = kind
        if kind == "gru":
            self.seq = nn.GRU(d, d, batch_first=True)
        else:
            self.pos = nn.Parameter(torch.randn(32, d) * 0.02)
            self.seq = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True,
                                                                        norm_first=True), 2)
        self.out = nn.Linear(2 * d, d)
        nn.init.zeros_(self.out.weight[:, d:]); self.out.weight.data[:, :d] = torch.eye(d); nn.init.zeros_(self.out.bias)

    def forward(self, batch):
        T, B, L = batch["ids"].shape
        e = self.bert(input_ids=batch["ids"].reshape(T * B, L), attention_mask=batch["att"].reshape(T * B, L)
                      ).last_hidden_state[:, 0].reshape(T, B, -1).transpose(0, 1)  # (B,T,d)
        if self.kind == "gru":
            h, _ = self.seq(e)
        else:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=e.device), 1)
            h = self.seq(e + self.pos[:T], mask=mask)
        q = self.out(torch.cat([e, h], -1)).transpose(0, 1)
        return dict(q=F.normalize(q, dim=-1))


class Concat(nn.Module):
    """History concatenation (ConvDR / Conv-ANCE style) with the full reversed history as input.
    two_level=True gives the hierarchical intent encoder: position 0 ([CLS]) carries the fast focus intent
    (matched against passages) and position 1 (a topic token) the slow topic intent (matched against articles)."""
    def __init__(self, two_level=False, shared=False, tau=0.02):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BGE, attn_implementation="sdpa")
        self.two_level, self.shared = two_level, shared
        # latent-document model: temperatures of the topic (coarse) and focus (fine) passage distributions
        self.log_tau_g = nn.Parameter(torch.tensor(math.log(tau)))
        self.log_tau_f = nn.Parameter(torch.tensor(math.log(tau)))

    def forward(self, batch):
        T, B, L = batch["ids"].shape
        h = self.bert(input_ids=batch["ids"].reshape(T * B, L), attention_mask=batch["att"].reshape(T * B, L)
                      ).last_hidden_state
        out = dict(q=F.normalize(h[:, 0], dim=-1).reshape(T, B, -1))
        if self.two_level:
            g = h[:, 0] if self.shared else h[:, 1]
            out["qg"] = F.normalize(g, dim=-1).reshape(T, B, -1)
        return out


def build(kind, **kw):
    if kind == "prism":
        return PRISM2(**kw)
    if kind == "prism_rec":
        return PRISM(**kw)
    if kind in ("gru", "hier"):
        return TurnSeq(kind)
    if kind == "concat":
        return Concat(**kw)
    raise ValueError(kind)


class PRISM2(nn.Module):
    """PRISM (final): predictive multi-timescale intent tracking *inside* one Transformer over the history.

    tokens carry their turn age a (0 = current turn, a>=1 history, -1 pad).
    layers 1..k  : block-diagonal attention -> per-turn, history-free intent observations u_a
    causal TCN   : predicts every turn's intent from the turns before it -> surprise sigma_a (event boundaries)
    layers k+1.. : full attention; heads are split into long / task / focus time scales and the key turn a gets
                   the per-head bias  -lambda_tau * C_a + rho_tau * relu(cos(u_0, u_a))
                   C_a = accumulated surprise between turn a and now (soft number of intent shifts in between),
                   lambda_long <= lambda_task <= lambda_focus (ordered by construction)."""

    def __init__(self, k_split=4, n_scales=3, use_surprise=True, ordered_scales=True, predictor="tcn",
                 use_return=True, block_phase1=True, feedback=True, dual_stream=True, kalman=False, views=False, mih=0, dp=256, max_age=32):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BGE, attn_implementation="sdpa")
        d = self.bert.config.hidden_size; H = self.bert.config.num_attention_heads
        self.k, self.H, self.ns = k_split, H, n_scales
        self.use_surprise, self.ordered_scales, self.use_return, self.block_phase1 = use_surprise, ordered_scales, use_return, block_phase1
        self.head_scale = torch.arange(H) * n_scales // H           # heads -> time scale id
        self.u_proj = nn.Linear(d, dp)
        self.pred = CausalTCN(dp) if predictor == "tcn" else MLPPred(dp)
        self.sur_w = nn.Parameter(torch.tensor(4.0)); self.sur_b = nn.Parameter(torch.tensor(-2.0))
        # neutral init: the model starts as plain history attention and learns how strongly to gate
        self.lam_raw = nn.Parameter(torch.tensor([-4.0] + [-6.0] * (n_scales - 1)))
        self.rho = nn.Parameter(torch.zeros(n_scales))
        self.age_bias = nn.Parameter(torch.zeros(n_scales, max_age))  # content-free recency prior per scale
        self.feedback, self.dual_stream, self.kalman, self.views = feedback, dual_stream, kalman, views
        # multi-timescale intent views: one query token per time scale (long / task / focus)
        self.view_emb = nn.Parameter(torch.randn(n_scales, d) * 0.02)
        # mixture of intent hypotheses: segment-conditioned query tokens (current / previous / earlier segment,
        # new topic, whole history) and a posterior over them; segments come from accumulated surprise
        self.mih = mih
        if mih:
            assert mih >= 3
            self.n_seg = mih - 2
            self.hyp_emb = nn.Parameter(torch.randn(mih, d) * 0.02)
            self.seg_logw = nn.Parameter(torch.tensor(0.0))          # log segment width
            self.pi_head = nn.Linear(d, mih)
            nn.init.zeros_(self.pi_head.weight); nn.init.zeros_(self.pi_head.bias)
            # posterior scorer: judges each hypothesis from (session state, hypothesis state)
            self.pi_mlp = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, 1))
            nn.init.zeros_(self.pi_mlp[2].weight); nn.init.zeros_(self.pi_mlp[2].bias)
        self.fb_proj = nn.Linear(d, d); self.fb_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        # hierarchical intent memory: turns beyond the text budget are compressed by the observation stream
        # (the same first-k encoder layers) into one intent token each
        self.mem_proj = nn.Linear(d, d); self.mem_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        # Kalman-style intent update: posterior query = observation + gain(surprise) * prior
        self.prior_out = nn.Linear(dp, d)
        nn.init.zeros_(self.prior_out.weight); nn.init.zeros_(self.prior_out.bias)
        self.gain_a = nn.Parameter(torch.tensor(1.0)); self.gain_b = nn.Parameter(torch.tensor(1.0))

    def _layer(self, layer, h, mask):
        if self.training:
            return torch.utils.checkpoint.checkpoint(lambda x, m: layer(x, attention_mask=m)[0], h, mask, use_reentrant=False)
        return layer(h, attention_mask=mask)[0]

    def lambdas(self):
        v = ordered(self.lam_raw) if self.ordered_scales else self.lam_raw
        return F.softplus(v)

    def forward(self, batch):
        ids, att, age = batch["ids"], batch["att"], batch["age"]        # (B,L)
        B, L = ids.shape
        A = int(age.max().item()) + 1
        h = self.bert.embeddings(input_ids=ids)
        text_att = att
        layers = self.bert.encoder.layer
        mem_u = None
        if "mem_ids" in batch and batch["mem_ids"].shape[1] > 0:
            mi, ma, mage = batch["mem_ids"], batch["mem_att"], batch["mem_age"]       # (B,M,Lm), (B,M,Lm), (B,M)
            M, Lm = mi.shape[1], mi.shape[2]
            flat_ok = ma.reshape(B * M, Lm).sum(1) > 0
            m = torch.zeros(B * M, h.shape[-1], dtype=h.dtype, device=h.device)
            if flat_ok.any():
                x = self.bert.embeddings(input_ids=mi.reshape(B * M, Lm)[flat_ok])
                am = ma.reshape(B * M, Lm)[flat_ok]
                km = torch.zeros(x.shape[0], 1, 1, Lm, dtype=h.dtype, device=h.device).masked_fill(~am.bool()[:, None, None, :], NEG)
                for li in range(self.k):
                    x = self._layer(layers[li], x, km)
                am_ = am.unsqueeze(-1).to(x.dtype)
                m[flat_ok] = ((x * am_).sum(1) / am_.sum(1).clamp_min(1)).to(m.dtype)
            m = m.reshape(B, M, -1)
            mok = mage >= 0
            h = torch.cat([h, self.mem_proj(m) + self.mem_type.to(h.dtype)], 1)
            att = torch.cat([att, mok.to(att.dtype)], 1)
            age = torch.cat([age, mage], 1)
            text_att = torch.cat([text_att, torch.zeros_like(mok, dtype=att.dtype)], 1)
            mem_u = (self.u_proj(m), mage, mok)
            B, L = att.shape
            A = int(age.max().item()) + 1
        if self.feedback and "fb" in batch:   # result tokens: passages returned at the previous turn (age 1)
            fb, fbm = batch["fb"], batch["fb_mask"]
            h = torch.cat([h, self.fb_proj(fb.to(h.dtype)) + self.fb_type.to(h.dtype)], 1)
            att = torch.cat([att, fbm.to(att.dtype)], 1)
            age = torch.cat([age, torch.where(fbm, torch.ones_like(fbm, dtype=age.dtype), torch.full_like(fbm, -1, dtype=age.dtype))], 1)
            text_att = torch.cat([text_att, torch.zeros_like(fbm, dtype=att.dtype)], 1)
            B, L = att.shape
        if self.mih:
            NV = self.mih
            vt = h[:, :1] + self.hyp_emb.to(h.dtype)[None]
        elif self.views:
            NV = self.ns
            vt = h[:, :1] + self.view_emb.to(h.dtype)[None]
        if self.views or self.mih:
            V0 = h.shape[1]
            h = torch.cat([h, vt], 1)
            att = torch.cat([att, torch.ones(B, NV, dtype=att.dtype, device=att.device)], 1)
            age = torch.cat([age, torch.zeros(B, NV, dtype=age.dtype, device=age.device)], 1)
            text_att = torch.cat([text_att, torch.zeros(B, NV, dtype=att.dtype, device=att.device)], 1)
            B, L = att.shape
        pad = ~att.bool()
        neg = torch.tensor(NEG, dtype=h.dtype, device=h.device)
        keymask = torch.zeros(B, 1, 1, L, dtype=h.dtype, device=h.device).masked_fill(pad[:, None, None, :], NEG)
        if self.block_phase1:
            same = (age[:, :, None] == age[:, None, :])
            p1 = torch.where(same, torch.zeros((), dtype=h.dtype, device=h.device), neg)[:, None] + keymask
        else:
            p1 = keymask
        main1 = keymask
        hypiso = None
        if self.mih:
            # isolation: a hypothesis token is visible only to itself, so hypotheses cannot leak into each other
            # through text tokens; before the segmentation exists (layers 1..k) hypotheses read the current turn only
            iso = torch.zeros(1, 1, L, L, dtype=h.dtype, device=h.device)
            iso[:, :, :, V0:V0 + NV] = NEG
            iso[:, :, V0:V0 + NV, V0:V0 + NV] = torch.where(torch.eye(NV, dtype=torch.bool, device=h.device),
                                                          torch.zeros((), dtype=h.dtype, device=h.device), neg)
            hypiso = iso
            cur = torch.where(age[:, None, :] == 0, torch.zeros((), dtype=h.dtype, device=h.device), neg)  # (B,1,L)
            m1 = torch.zeros(B, 1, L, L, dtype=h.dtype, device=h.device)
            m1[:, :, V0:V0 + NV, :] = cur[:, :, None, :].expand(B, 1, NV, L)
            main1 = keymask + iso + m1
            p1 = p1 + iso
        if self.dual_stream and self.block_phase1:
            # observation stream (shared weights, block-diagonal) gives history-free u; main stream keeps full attention
            ho = h
            for li in range(self.k):
                ho = self._layer(layers[li], ho, p1)
                h = self._layer(layers[li], h, main1)
        else:
            for li in range(self.k):
                h = self._layer(layers[li], h, p1)
            ho = h
        # per-turn observations (age 0..A-1)
        oh = F.one_hot(age.clamp_min(0), A).to(h.dtype) * text_att.unsqueeze(-1).to(h.dtype)   # (B,L,A) text only
        cnt = oh.sum(1)                                                                    # (B,A)
        U = self.u_proj((oh.transpose(1, 2) @ ho) / cnt.clamp_min(1).unsqueeze(-1))         # (B,A,dp)
        present = cnt > 0
        if mem_u is not None:   # observations of compressed (memory) turns
            mu, mage, mok = mem_u
            ohm = F.one_hot(mage.clamp_min(0), A).to(U.dtype) * mok.unsqueeze(-1).to(U.dtype)  # (B,M,A)
            U = U + ohm.transpose(1, 2) @ mu.to(U.dtype)
            present = present | (ohm.sum(1) > 0)
        # chronological order: oldest first. turn with age a sits at position A-1-a
        Uc = U.flip(1)
        preds = torch.stack([self.pred(Uc[:, :i]) for i in range(A)], 1)                   # prediction for each position
        cos_pred = F.cosine_similarity(preds, Uc, dim=-1)
        sig = (self.sur_w * (1 - cos_pred) + self.sur_b).flip(1)                           # (B,A) surprise of turn age a
        presc = present.flip(1)
        first = presc & (presc.cumsum(1) == 1)                                             # oldest present turn: no prior
        p_shift = torch.sigmoid(sig) * present.to(sig.dtype) * (~first.flip(1)).to(sig.dtype)
        if not self.use_surprise:
            p_shift = torch.zeros_like(p_shift)
        # accumulated shift mass between turn a and now: sum over ages 0..a-1
        C = torch.cumsum(p_shift, 1) - p_shift                                            # (B,A)
        r = F.relu(F.cosine_similarity(U[:, :1], U, dim=-1))                              # similarity to current intent
        lam = self.lambdas()                                                               # (ns,)
        hs = self.head_scale.to(h.device)
        ageb = self.age_bias[:, :A]                                                        # (ns,A)
        bias = -lam[hs][None, :, None] * C[:, None, :]                                    # (B,H,A)
        if self.use_return:
            bias = bias + self.rho[hs][None, :, None] * r[:, None, :]
        bias = bias + ageb[hs][None]
        bias[:, :, 0] = 0                                                                  # current turn never gated
        tok_bias = bias.gather(2, age.clamp_min(0)[:, None, :].expand(B, self.H, L)).to(h.dtype)   # (B,H,L)
        if self.mih:
            # hypothesis j <= n_seg-1 reads the j-th most recent segment: soft membership from accumulated surprise
            w = torch.exp(self.seg_logw).clamp_min(0.1)
            js = torch.arange(self.n_seg, device=h.device, dtype=C.dtype)
            hb = -((C[:, None, :] - js[None, :, None]) ** 2) / (2 * w * w)                 # (B,n_seg,A)
            hb = torch.cat([hb, torch.full_like(hb[:, :1], -30.0), torch.zeros_like(hb[:, :1])], 1)  # new topic, whole history
            hb = hb.clamp_min(-30.0)
            hb[:, :, 0] = 0                                                                # every hypothesis reads the current turn
            vb = hb.gather(2, age.clamp_min(0)[:, None, :].expand(B, NV, L)).to(h.dtype)
            qb = torch.zeros(B, L, L, dtype=h.dtype, device=h.device)
            qb[:, V0:V0 + NV, :] = vb
            p2 = qb[:, None] + keymask + hypiso
        elif self.views:
            # head-wise scale bias is replaced by query-wise scale bias: view token tau sees turn a with bias_tau(a)
            sb = -lam[None, :, None] * C[:, None, :] + ageb[None]                         # (B,ns,A)
            if self.use_return:
                sb = sb + self.rho[None, :, None] * r[:, None, :]
            sb[:, :, 0] = 0
            vb = sb.gather(2, age.clamp_min(0)[:, None, :].expand(B, self.ns, L)).to(h.dtype)  # (B,ns,L)
            qb = torch.zeros(B, L, L, dtype=h.dtype, device=h.device)
            qb[:, V0:V0 + self.ns, :] = vb
            p2 = qb[:, None] + keymask
        else:
            p2 = tok_bias[:, :, None, :] + keymask
        for li in range(self.k, len(layers)):
            h = self._layer(layers[li], h, p2)
        idx = torch.full((B,), A - 1, device=h.device)
        prior = preds[torch.arange(B), idx]                                                 # TCN prior for the current turn
        has_prev = present[:, 1] if A > 1 else torch.zeros(B, dtype=torch.bool, device=h.device)
        gain = torch.sigmoid(self.gain_a - self.gain_b * sig[:, 0]) * has_prev.to(sig.dtype)  # trust the prior when unsurprised
        prior_vec = self.prior_out(prior)
        logpi = None
        if self.mih:
            qh = F.normalize(h[:, V0:V0 + NV].float(), dim=-1)                               # (B,K,d) hypothesis queries
            hk = h[:, V0:V0 + NV]
            pl = self.pi_head(h[:, 0]) + self.pi_mlp(torch.cat([h[:, :1].expand_as(hk), hk], -1)).squeeze(-1)
            logpi = F.log_softmax(pl.float(), -1)                                          # posterior over hypotheses
            q = F.normalize((logpi.exp().unsqueeze(-1) * qh).sum(1), dim=-1)               # posterior-mean intent query
        elif self.views:
            q = F.normalize(torch.cat([h[:, :1], h[:, V0:V0 + self.ns]], 1).float(), dim=-1)   # (B,1+ns,d): CLS + views
        elif self.kalman:
            q = F.normalize(h[:, 0].float() + (gain.unsqueeze(-1) * prior_vec).float(), dim=-1)
        else:
            q = F.normalize(h[:, 0].float(), dim=-1)
        q_prior = F.normalize(prior_vec.float(), dim=-1)
        out = dict(q=q, u=Uc[torch.arange(B), idx], u_hat=prior, sigma=sig[:, 0], q_prior=q_prior, gain=gain,
                    has_prev=has_prev,
                    lam=lam.detach(), C=C.detach(), r=r.detach())
        if logpi is not None:
            out["logpi"] = logpi; out["q_h"] = qh
        return out
