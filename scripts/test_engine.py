"""Parity tests: C tokenizer == HF tokenizer; C INT8 engine logits == PyTorch with the same INT8 weights.

python scripts/test_engine.py ckpt/x.pt export/x
"""
import json, sys
sys.path.insert(0, ".")
import numpy as np, torch
from microcoder.model import Config, MicroCoder
from microcoder.tok import Tok
from microcoder.cengine import CEngine

ck = torch.load(sys.argv[1]); d = sys.argv[2]
cfg = Config(**ck["cfg"]); tok = Tok(ck["args"]["tok"])
eng = CEngine(f"{d}/model.bin", cfg, f"{d}/tok.bin")
# --- tokenizer parity on every prompt/code text we use
texts = []
for f in ["data/test_id.jsonl", "data/test_ood.jsonl", "data/train_gen.jsonl"]:
    for x in open(f):
        r = json.loads(x); texts += [r["instr"], r["code"]]
for x in open("data/mbpp.jsonl"):
    r = json.loads(x); texts += [r["text"], r["code"]]
bad = [t for t in texts if eng.encode(t) != tok.encode(t)]
print(f"tokenizer parity: {len(texts) - len(bad)}/{len(texts)} identical")
for t in bad[:3]: print("  MISMATCH:", repr(t[:80]))
# --- logits parity: torch model with fake-quantised weights vs C engine
m = MicroCoder(cfg); m.load_state_dict(ck["model"]); m.eval()
mq = MicroCoder(cfg); sd = {k: v.clone() for k, v in ck["model"].items()}
for k, v in sd.items():
    if v.dim() == 2:
        s = v.abs().amax(1, keepdim=True) / 127; s[s == 0] = 1e-12
        sd[k] = (v / s).round().clamp(-127, 127) * s
mq.load_state_dict(sd); mq.eval()
ids = tok.encode("<q>Write a function c_to_f(c) that converts a temperature c in Celsius to Fahrenheit.<c>def c_to_f(c):")
with torch.no_grad():
    ref_q = mq(torch.tensor([ids]))[0].numpy(); ref_f = m(torch.tensor([ids]))[0].numpy()
ce = np.stack([eng.logits(t, p) for p, t in enumerate(ids)])
print("max |C - torch(int8 weights)| =", float(np.abs(ce - ref_q).max()))
print("max |C - torch(fp32)|         =", float(np.abs(ce - ref_f).max()),
      " argmax agreement:", float((ce.argmax(1) == ref_f.argmax(1)).mean()))
