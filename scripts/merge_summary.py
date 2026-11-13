"""Merge methods evaluated in a separate run (results/<extra>) into results/<base>.

python scripts/merge_summary.py m1_s0 m1_s0_lint
"""
import json, sys
base, extra = sys.argv[1], sys.argv[2]
B = json.load(open(f"results/{base}/summary.json")); E = json.load(open(f"results/{extra}/summary.json"))
for sp, S in E["splits"].items():
    for m, v in S["methods"].items():
        B["splits"][sp]["methods"].setdefault(m, v)
    rb = [json.loads(x) for x in open(f"results/{base}/{sp}.jsonl")]
    re_ = [json.loads(x) for x in open(f"results/{extra}/{sp}.jsonl")]
    for a, b in zip(rb, re_):
        for m in S["methods"]:
            a.setdefault(m, b[m])
    with open(f"results/{base}/{sp}.jsonl", "w") as f:
        for a in rb:
            f.write(json.dumps(a) + "\n")
json.dump(B, open(f"results/{base}/summary.json", "w"), indent=1)
print("merged", extra, "into", base)
