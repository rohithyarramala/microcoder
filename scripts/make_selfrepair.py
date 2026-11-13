"""Stage 2 data: collect the model's OWN syntax errors on training prompts.

Sample drafts at temperature 1.0 for every training instruction; keep drafts that
fail compile() and pair them with the reference code. Only training data is used.
python scripts/make_selfrepair.py ckpt/m1_s0.pt [samples_per_prompt]
"""
import json, sys
sys.path.insert(0, ".")
import torch
from microcoder.evaluate import load_model
from microcoder.tok import Tok
from microcoder.dataset import prompt_gen
from microcoder.sandbox import syntax_error

m, ck = load_model(sys.argv[1]); tok = Tok(ck["args"]["tok"])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
torch.set_num_threads(1)
g = torch.Generator().manual_seed(7)
rows = [json.loads(x) for x in open("data/train_gen.jsonl")]
out, seen, n = [], set(), 0
for r in rows:
    ids = tok.encode(prompt_gen(r["instr"]))
    for _ in range(k):
        n += 1
        code = tok.decode(m.generate(ids, 120, tok.sp["<e>"], temperature=1.0, gen=g))
        err = syntax_error(code)
        if err and (r["instr"], code) not in seen:
            seen.add((r["instr"], code))
            out.append(dict(instr=r["instr"], broken=code, err=err, code=r["code"], family=r["family"]))
with open("data/train_selfrepair.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
print(f"{len(out)} self-repair pairs from {n} samples ({100*len(out)/n:.1f}% of samples had syntax errors)")
