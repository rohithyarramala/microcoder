"""Evaluate a checkpoint with every decoding strategy on every test split.

python -m microcoder.evaluate ckpt/m1_s0.pt [--splits test_id,test_ood] [--methods greedy,resample,ref,ref_noerr]
Writes results/<name>/<split>.jsonl (per task) and results/<name>/summary.json
"""
import argparse, json, math, os, time
import torch
from .model import Config, MicroCoder
from .tok import Tok
from .infer import Generator, solve, METHODS
import re
from .sandbox import run_tests, run_asserts


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def load_model(path):
    ck = torch.load(path)
    m = MicroCoder(Config(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--splits", default="test_id,test_ood")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--tok", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--engine", default=None, help="path to C engine .bin to evaluate the INT8 deployable model instead")
    a = ap.parse_args()
    torch.set_num_threads(1)
    name = a.out or os.path.basename(a.ckpt).replace(".pt", "")
    m, ck = load_model(a.ckpt)
    tok = Tok(a.tok or ck["args"]["tok"])
    if a.engine:
        from .cengine import CEngine
        m = CEngine(a.engine, m.c)
    gen = Generator(m, tok)
    os.makedirs(f"results/{name}", exist_ok=True)
    summary = dict(ckpt=a.ckpt, n_params=ck["n_params"], engine=bool(a.engine), splits={})
    for split in a.splits.split(","):
        if split == "mbpp":
            tasks = []
            for x in open("data/mbpp.jsonl"):
                r = json.loads(x)
                if 11 <= r["task_id"] <= 510:  # standard MBPP test split
                    defs = re.findall(r"def (\w+)\(", r["code"])
                    fn = next((d for d in defs if d + "(" in r["test_list"][0]), defs[0] if defs else "f")
                    tasks.append(dict(family="mbpp", cat="mbpp", instr=r["text"] + " The function is called " + fn + ".",
                                      fn=fn, setup=r["test_setup_code"], asserts=r["test_list"]))
        else:
            tasks = [json.loads(x) for x in open(f"{a.data}/{split}.jsonl")]
        rows = []
        t0 = time.time()
        for i, t in enumerate(tasks):
            r = dict(i=i, family=t["family"], cat=t["cat"])
            for meth in a.methods.split(","):
                s = solve(gen, t["instr"], meth, seed=i)
                ok = False
                if s["valid"] and "asserts" in t:
                    ok = run_asserts(s["code"], t["setup"], t["asserts"])
                elif s["valid"]:
                    ok, _, _ = run_tests(s["code"], t["fn"], t["tests"], t["expected"])
                r[meth] = dict(valid=s["valid"], passed=ok, n_gens=s["n_gens"], code=s["code"])
            rows.append(r)
        with open(f"results/{name}/{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        S = dict(n=len(rows), seconds=round(time.time() - t0, 1), methods={})
        for meth in a.methods.split(","):
            nv = sum(r[meth]["valid"] for r in rows)
            npass = sum(r[meth]["passed"] for r in rows)
            S["methods"][meth] = dict(
                valid=nv, valid_pct=round(100 * nv / len(rows), 1), valid_ci=wilson(nv, len(rows)),
                passed=npass, pass_pct=round(100 * npass / len(rows), 1), pass_ci=wilson(npass, len(rows)),
                mean_gens=round(sum(r[meth]["n_gens"] for r in rows) / len(rows), 3),
                by_cat={c: [sum(r[meth]["passed"] for r in rows if r["cat"] == c), sum(r["cat"] == c for r in rows)]
                        for c in sorted({r["cat"] for r in rows})})
        summary["splits"][split] = S
        print(split, json.dumps({k: (v["valid_pct"], v["pass_pct"], v["mean_gens"]) for k, v in S["methods"].items()}), f"{S['seconds']}s", flush=True)
    json.dump(summary, open(f"results/{name}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
