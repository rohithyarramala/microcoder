"""Train a MicroCoder model on CPU (or GPU if present).

python -m microcoder.train --name m1_s0 --d 128 --layers 4 --heads 4 --mix gen+repair --seed 0
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from .dataset import gen_text, repair_text
from .model import Config, MicroCoder
from .tok import Tok


def load(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def encode_rows(tok, texts, ctx):
    out = []
    c_id = tok.sp["<c>"]
    for t in texts:
        ids = tok.encode(t)
        if len(ids) > ctx:
            continue
        start = ids.index(c_id) + 1  # loss only on the answer (code + <e>)
        out.append((ids, start))
    return out


def batchify(rows, pad):
    T = max(len(r[0]) for r in rows)
    x = torch.full((len(rows), T), pad, dtype=torch.long)
    y = torch.full((len(rows), T), -100, dtype=torch.long)
    for i, (ids, s) in enumerate(rows):
        x[i, :len(ids)] = torch.tensor(ids)
        tgt = torch.tensor(ids[1:])
        y[i, :len(ids) - 1] = tgt
        y[i, :s - 1] = -100
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--tok", default="data/tok2048.json")
    ap.add_argument("--mix", default="gen+repair", help="gen | gen+repair | gen+repair+self")
    ap.add_argument("--repair_frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(max(1, os.cpu_count()))
    tok = Tok(a.tok)
    gen = load(f"{a.data}/train_gen.jsonl")
    rep = load(f"{a.data}/train_repair.jsonl") if "repair" in a.mix else []
    if "name" in a.mix:
        rep += load(f"{a.data}/train_repair_name.jsonl")
    if "self" in a.mix:
        rep += load(f"{a.data}/train_selfrepair.jsonl") * 3  # upweight the model's own error distribution
    G = encode_rows(tok, [gen_text(r["instr"], r["code"]) for r in gen], a.ctx)
    R = encode_rows(tok, [repair_text(r["instr"], r["broken"], r["err"], r["code"]) for r in rep], a.ctx)
    print(f"gen={len(G)} repair={len(R)}", flush=True)

    cfg = Config(vocab=tok.vocab, d=a.d, layers=a.layers, heads=a.heads, ctx=a.ctx, dropout=a.dropout)
    model = MicroCoder(cfg)
    if a.init:
        model.load_state_dict(torch.load(a.init)["model"])
    print("params", model.n_params(), flush=True)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    warm = min(200, a.steps // 10)
    sched = lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm)))
    rng = random.Random(a.seed)
    t0 = time.time()
    log = []
    model.train()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        nr = int(a.bs * a.repair_frac) if R else 0
        rows = [rng.choice(G) for _ in range(a.bs - nr)] + [rng.choice(R) for _ in range(nr)]
        x, y = batchify(rows, tok.sp["<pad>"])
        loss = F.cross_entropy(model(x).view(-1, cfg.vocab), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == a.steps - 1:
            log.append((step, round(loss.item(), 4)))
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs("ckpt", exist_ok=True)
    torch.save(dict(model=model.state_dict(), cfg=cfg.to_dict(), args=vars(a), log=log,
                    train_seconds=time.time() - t0, n_params=model.n_params()), f"ckpt/{a.name}.pt")
    print("saved", a.name, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
