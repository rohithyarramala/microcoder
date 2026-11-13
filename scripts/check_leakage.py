"""Prove there is no train/test leakage. Writes results/leakage.json."""
import json, re, sys
sys.path.insert(0, ".")
from microcoder.dataset import norm
L = lambda p: [json.loads(x) for x in open(p)]
gen, rep = L("data/train_gen.jsonl"), L("data/train_repair.jsonl")
meta = json.load(open("data/meta.json"))
train_instr = {norm(r["instr"]) for r in gen} | {norm(r["instr"]) for r in rep}
train_code = {r["code"] for r in gen}
train_fams = {r["family"] for r in gen} | {r["family"] for r in rep}
def ngrams(s, n=8):
    w = norm(s).split(); return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
train_ng = set().union(*(ngrams(r["instr"]) for r in gen))
out = {}
for split in ["test_id", "test_ood"]:
    T = L(f"data/{split}.jsonl")
    out[split] = dict(
        n=len(T),
        unique_instructions=len({norm(t["instr"]) for t in T}),
        exact_instruction_overlap=sum(norm(t["instr"]) in train_instr for t in T),
        tasks_sharing_any_8gram_with_train=sum(bool(ngrams(t["instr"]) & train_ng) for t in T),
        reference_code_identical_to_a_train_example=sum(t["code"] in train_code for t in T),
        family_seen_in_training=sum(t["family"] in train_fams for t in T))
assert out["test_id"]["exact_instruction_overlap"] == 0 and out["test_ood"]["exact_instruction_overlap"] == 0
assert out["test_ood"]["family_seen_in_training"] == 0
assert not set(meta["heldout_families"]) & train_fams
json.dump(out, open("results/leakage.json", "w"), indent=1)
print(json.dumps(out, indent=1))
