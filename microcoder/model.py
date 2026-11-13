"""MicroCoder: a tiny decoder-only Transformer sized for microcontrollers.

Deliberately simple so the exact same maths can be re-implemented in ~300
lines of portable C (esp32/engine): RMSNorm, rotary position embeddings,
multi-head causal attention, GELU MLP, tied input/output embeddings, no biases.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 2048
    d: int = 128
    layers: int = 4
    heads: int = 4
    ff_mult: int = 4
    ctx: int = 256
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)


def param_breakdown(c: Config):
    emb = c.vocab * c.d
    attn = 4 * c.d * c.d
    mlp = 2 * c.ff_mult * c.d * c.d
    norms = 2 * c.d
    per_layer = attn + mlp + norms
    total = emb + c.layers * per_layer + c.d
    return dict(embedding=emb, per_layer=per_layer, attention_per_layer=attn, mlp_per_layer=mlp,
                layers_total=c.layers * per_layer, final_norm=c.d, total=total)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_tables(ctx, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    t = torch.arange(ctx).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x: B,H,T,hd ; interleaved pairs (0,1),(2,3)...
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.n1 = RMSNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.o = nn.Linear(c.d, c.d, bias=False)
        self.n2 = RMSNorm(c.d)
        self.up = nn.Linear(c.d, c.ff_mult * c.d, bias=False)
        self.down = nn.Linear(c.ff_mult * c.d, c.d, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        H = self.c.heads
        q, k, v = self.qkv(self.n1(x)).split(D, -1)
        q, k, v = [z.view(B, T, H, D // H).transpose(1, 2) for z in (q, k, v)]
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, T, D)))
        x = x + self.drop(self.down(F.gelu(self.up(self.n2(x)), approximate="tanh")))
        return x


class MicroCoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.nf = RMSNorm(c.d)
        cos, sin = rope_tables(c.ctx, c.d // c.heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * c.layers))
            nn.init.normal_(b.down.weight, std=0.02 / math.sqrt(2 * c.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        return self.nf(x) @ self.emb.weight.T  # tied output projection

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def _step(self, idx, start, cache):
        """Incremental forward for positions start..start+len(idx)-1 with a KV cache."""
        c = self.c
        x = self.emb(idx)  # 1,T,D
        B, T, D = x.shape
        H, hd = c.heads, D // c.heads
        cos, sin = self.cos[start:start + T], self.sin[start:start + T]
        for li, b in enumerate(self.blocks):
            q, k, v = b.qkv(b.n1(x)).split(D, -1)
            q, k, v = [z.view(B, T, H, hd).transpose(1, 2) for z in (q, k, v)]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            if cache[li] is not None:
                k = torch.cat([cache[li][0], k], 2)
                v = torch.cat([cache[li][1], v], 2)
            cache[li] = (k, v)
            mask = torch.ones(T, k.shape[2], dtype=torch.bool).tril(k.shape[2] - T)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = x + b.o(a.transpose(1, 2).reshape(B, T, D))
            x = x + b.down(F.gelu(b.up(b.n2(x)), approximate="tanh"))
        return (self.nf(x) @ self.emb.weight.T)[0, -1]

    @torch.no_grad()
    def generate(self, ids, max_new, stop_id, temperature=0.0, top_k=0, gen=None):
        """KV-cached decoding. Returns the list of new token ids (stop token excluded)."""
        out = []
        ids = ids[-self.c.ctx:]
        cache = [None] * self.c.layers
        logits = self._step(torch.tensor([ids]), 0, cache)
        pos = len(ids)
        for _ in range(max_new):
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = logits.topk(top_k)
                    logits[logits < v[-1]] = -float("inf")
                nxt = int(torch.multinomial(F.softmax(logits, -1), 1, generator=gen))
            if nxt == stop_id:
                break
            out.append(nxt)
            if pos >= self.c.ctx:
                break
            logits = self._step(torch.tensor([[nxt]]), pos, cache)
            pos += 1
        return out
