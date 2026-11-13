"""Classify failures of one method: compile error / runtime exception / wrong value / wrong function.

python scripts/failure_analysis.py m1_s0 a_ref -> results/failures_<name>_<method>.json
"""
import collections, json, sys
sys.path.insert(0, ".")
from microcoder.sandbox import run_tests, syntax_error

name, meth = sys.argv[1], sys.argv[2]
out = {}
for sp in ("test_id", "test_ood"):
    T = [json.loads(x) for x in open(f"data/{sp}.jsonl")]
    R = [json.loads(x) for x in open(f"results/{name}/{sp}.jsonl")]
    c = collections.Counter()
    ex = collections.defaultdict(list)
    for t, r in zip(T, R):
        code = r[meth]["code"] or ""
        if r[meth]["passed"]:
            c["pass"] += 1; continue
        if syntax_error(code):
            k = "syntax error"
        else:
            ok, n, err = run_tests(code, t["fn"], t["tests"], t["expected"])
            if err in ("KeyError",):
                k = "function not defined"
            elif err:
                k = "runtime exception"
                c["runtime:" + err] += 1
            elif n > 0:
                k = "partly correct values"
            else:
                k = "wrong values"
        c[k] += 1
        if len(ex[k]) < 3:
            ex[k].append(dict(family=t["family"], instr=t["instr"], code=code, ref=t["code"]))
    out[sp] = dict(counts=dict(c), examples=ex)
    print(sp, dict(c))
json.dump(out, open(f"results/failures_{name}_{meth}.json", "w"), indent=1)
