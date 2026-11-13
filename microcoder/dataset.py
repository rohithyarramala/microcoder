"""Build the training corpus and the leak-free benchmark.

Splits
------
* 16 of the 64 task families (25%, stratified by category, fixed seed) are
  HELD OUT: no example from them is ever used for training. They form the
  out-of-family test set (test_ood).
* For the 48 training families, phrasing #4 is never used in training. The
  in-family test set (test_id) uses only that held-out phrasing, with freshly
  sampled names and constants.
* After generation, scripts/check_leakage.py proves there is no exact or
  normalised instruction overlap between train and either test split.

Prompt format (special tokens):
    <q> instruction <c> code <e>                                  generation
    <q> instruction <b> broken code <r> error message <c> code <e>   repair
"""
import json
import os
import random
import re

from .families import FAMILIES
from .sandbox import reference_outputs, syntax_error

CATS = ["sensor", "math", "list", "string", "recursion"]
HOLDOUT_PER_CAT = {"sensor": 6, "math": 3, "list": 3, "string": 2, "recursion": 2}
SPLIT_SEED = 1234
TEST_PHRASING = 3


def split_families():
    rng = random.Random(SPLIT_SEED)
    held = []
    for c in CATS:
        ids = sorted(f["id"] for f in FAMILIES if f["cat"] == c)
        held += rng.sample(ids, HOLDOUT_PER_CAT[c])
    train = [f for f in FAMILIES if f["id"] not in held]
    test = [f for f in FAMILIES if f["id"] in held]
    return train, test


# --- training-only augmentation (never applied to test splits) -------------
_SYL = ["read", "calc", "get", "temp", "volt", "adc", "sig", "val", "proc", "chk", "flt", "data", "buf",
        "acc", "gyro", "lux", "hum", "press", "node", "pin", "raw", "out", "res", "comp", "eval", "est",
        "x", "y", "z", "my", "dev", "ctl", "pwm", "imu", "rx", "tx", "lvl", "cnt", "avg", "fix", "do", "run"]
_ARGS = ["a", "b", "x", "y", "n", "s", "v", "w", "k", "p", "q", "t", "d", "u", "arr", "lst", "seq", "xs",
         "data", "buf", "inp", "val", "num", "txt", "msg", "raw", "sig", "r", "m", "z", "c", "h", "e", "items2"]
OPENERS = [("Write a function ", ["Write a Python function ", "Write a function ", "Make a function ", "Code a function ",
                                  "Please write a function ", "Build a function ", "I need a function ", "Give me a function "]),
           ("Define ", ["Define ", "Please define ", "Write ", "Code ", "Make "]),
           ("Create ", ["Create ", "Make ", "Build ", "Write ", "Please create "])]


def _rand_ident(rng, k):
    return "_".join(rng.choice(_SYL) for _ in range(k))


def augment_instr(instr, rng):
    for old, news in OPENERS:
        if instr.startswith(old):
            instr = rng.choice(news) + instr[len(old):]
            break
    if rng.random() < 0.2:
        instr = instr.rstrip(".")
    if rng.random() < 0.15:
        instr = instr[0].lower() + instr[1:]
    return instr


def instantiate(fam, rng, phrasing, aug=False, name_p=0.5):
    p = dict(fam["params"](rng))
    if "derived" in fam:
        p.update(fam["derived"](p))
    names = {}
    for key, pool in zip("abc", fam["args"]):
        names[key] = rng.choice(_ARGS) if aug and rng.random() < name_p else rng.choice(pool)
    if len(set(names.values())) < len(names):  # avoid duplicate arg names
        return None
    fn = _rand_ident(rng, rng.randint(1, 3)) if aug and rng.random() < name_p else rng.choice(fam["fn"])
    if fn in names.values():
        return None
    fields = dict(p, fn=fn, **names)
    instr = fam["phr"][phrasing].format(**fields)
    if aug:
        instr = augment_instr(instr, rng)
    code = fam["code"].format(**fields)
    assert syntax_error(code) is None, (fam["id"], code)
    tests = fam["tests"](rng, p)
    try:
        expected = reference_outputs(code, fn, tests)
    except Exception:
        return None
    return dict(family=fam["id"], cat=fam["cat"], instr=instr, code=code, fn=fn,
                tests=tests, expected=expected, phrasing=phrasing)


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


# ----------------------------------------------------------------- corruption
def _corrupt_once(code, rng):
    lines = code.split("\n")
    ops = ["colon", "close", "open", "indent", "dedent", "keyword", "dropop", "dupe"]
    op = rng.choice(ops)
    idx = [i for i, l in enumerate(lines) if l.strip()]
    i = rng.choice(idx)
    l = lines[i]
    if op == "colon":
        cands = [k for k in idx if lines[k].rstrip().endswith(":")]
        if not cands:
            return None
        k = rng.choice(cands)
        lines[k] = lines[k].rstrip()[:-1]
    elif op in ("close", "open"):
        chars = ")]" if op == "close" else "(["
        pos = [(k, j) for k in idx for j, ch in enumerate(lines[k]) if ch in chars]
        if not pos:
            return None
        k, j = rng.choice(pos)
        lines[k] = lines[k][:j] + lines[k][j + 1:]
    elif op == "indent":
        body = [k for k in idx if k > 0]
        if not body:
            return None
        k = rng.choice(body)
        lines[k] = "  " + lines[k]
    elif op == "dedent":
        body = [k for k in idx if lines[k].startswith("    ")]
        if not body:
            return None
        k = rng.choice(body)
        lines[k] = lines[k][4:] if rng.random() < 0.5 else lines[k][2:]
    elif op == "keyword":
        subs = [("return", "retrun"), ("def", "df"), ("for", "fro"), ("while", "whlie"), ("if", "fi"), (" in ", " ni ")]
        cands = [(k, a, b) for k in idx for a, b in subs if re.search(r"\b%s\b" % a.strip(), lines[k])]
        if not cands:
            return None
        k, a, b = rng.choice(cands)
        lines[k] = lines[k].replace(a, b, 1)
    elif op == "dropop":
        toks = [(k, m.start(), m.end()) for k in idx for m in re.finditer(r"==|<=|>=|\+=|=|\+|\*|,|/|%", lines[k])]
        if not toks:
            return None
        k, s, e = rng.choice(toks)
        lines[k] = lines[k][:s] + lines[k][e:]
    elif op == "dupe":
        toks = [(k, m.start(), m.end()) for k in idx for m in re.finditer(r"\w+|[^\w\s]", lines[k])]
        k, s, e = rng.choice(toks)
        lines[k] = lines[k][:e] + " " + lines[k][s:e] + lines[k][e:]
    return "\n".join(lines)


def corrupt(code, rng):
    """Return (broken_code, error_message) with a genuine SyntaxError, or None."""
    for _ in range(10):
        b = _corrupt_once(code, rng)
        if rng.random() < 0.2 and b is not None:  # sometimes two faults
            b2 = _corrupt_once(b, rng)
            b = b2 if b2 is not None else b
        if b is None or b == code:
            continue
        err = syntax_error(b)
        if err is not None:
            return b, err
    return None


def gen_text(instr, code):
    return "<q>" + instr + "<c>" + code + "<e>"


def repair_text(instr, broken, err, code):
    return "<q>" + instr + "<b>" + broken + "<r>" + err + "<c>" + code + "<e>"


def prompt_gen(instr):
    return "<q>" + instr + "<c>"


def prompt_repair(instr, broken, err):
    return "<q>" + instr + "<b>" + broken + "<r>" + err + "<c>"


# ----------------------------------------------------------------- build
def build(out_dir, draws_per_family=500, n_id_per_family=6, n_ood_per_family=12, seed=0, aug=True, name_p=0.5,
          train_phrasings=(0, 1, 2), n_dev_per_family=0, deploy=False):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    train_f, held_f = split_families()
    if deploy:  # application model: every family and every phrasing (NOT for evaluation)
        train_f, held_f = list(FAMILIES), []
        train_phrasings = (0, 1, 2, 3)

    # test sets first, so we can exclude their exact instructions from training
    test_id, seen = [], set()
    for fam in ([] if deploy else train_f):
        k = 0
        while k < n_id_per_family:
            t = instantiate(fam, rng, TEST_PHRASING)
            if t and norm(t["instr"]) not in seen:
                seen.add(norm(t["instr"]))
                test_id.append(t)
                k += 1
    test_ood = []
    for fam in held_f:
        k, tries = 0, 0
        while k < n_ood_per_family and tries < 500:
            tries += 1
            t = instantiate(fam, rng, rng.randrange(4))
            if t and norm(t["instr"]) not in seen:
                seen.add(norm(t["instr"]))
                test_ood.append(t)
                k += 1

    # optional development split (model selection): train families, a phrasing
    # that is excluded from training in that configuration
    dev_ph = [k for k in range(3) if k not in train_phrasings]
    test_dev = []
    if n_dev_per_family and dev_ph:
        for fam in train_f:
            k = 0
            while k < n_dev_per_family:
                t = instantiate(fam, rng, dev_ph[0])
                if t and norm(t["instr"]) not in seen:
                    seen.add(norm(t["instr"]))
                    test_dev.append(t)
                    k += 1
    gen, rep = {}, {}
    for fam in train_f:
        for _ in range(draws_per_family):
            t = instantiate(fam, rng, rng.choice(train_phrasings), aug=aug, name_p=name_p)  # never phrasing 3
            if not t or norm(t["instr"]) in seen:
                continue
            gen.setdefault(norm(t["instr"]), (t["instr"], t["code"], fam["id"]))
            c = corrupt(t["code"], rng)
            if c:
                rep.setdefault((norm(t["instr"]), c[0]), (t["instr"], c[0], c[1], t["code"], fam["id"]))

    def dump(name, rows):
        with open(os.path.join(out_dir, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    dump("train_gen.jsonl", [dict(instr=i, code=c, family=f) for i, c, f in gen.values()])
    dump("train_repair.jsonl", [dict(instr=i, broken=b, err=e, code=c, family=f) for i, b, e, c, f in rep.values()])
    dump("test_id.jsonl", test_id)
    dump("test_ood.jsonl", test_ood)
    if test_dev:
        dump("test_dev.jsonl", test_dev)
    meta = dict(train_families=[f["id"] for f in train_f], heldout_families=[f["id"] for f in held_f],
                n_train_gen=len(gen), n_train_repair=len(rep), n_test_id=len(test_id), n_test_ood=len(test_ood))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta


# ----------------------------------------------------------------- undefined-name corruptions
_UNDEF = ["s", "t", "x", "k", "n", "out", "res", "val", "total", "count", "best", "i", "c", "lo", "hi", "y"]


def corrupt_name(code, rng):
    """Replace one read of a name or one numeric constant by a name that is never defined.
    Returns (broken, lint_message) or None."""
    import re as _re
    from .lint import undefined_name
    lines = code.split("\n")
    cands = []
    for li, l in enumerate(lines):
        if li == 0 or not l.strip():
            continue
        for m in _re.finditer(r"(?<![\w.'\"])([A-Za-z_]\w*|\d+(?:\.\d+)?)(?![\w'\"(])", l):
            tok = m.group(1)
            if tok in ("return", "if", "for", "in", "while", "and", "or", "not", "True", "False", "None", "else", "def"):
                continue
            cands.append((li, m.start(1), m.end(1)))
    rng.shuffle(cands)
    for li, s, e in cands[:6]:
        new = rng.choice(_UNDEF)
        b = list(lines)
        b[li] = b[li][:s] + new + b[li][e:]
        broken = "\n".join(b)
        if syntax_error(broken):
            continue
        msg = undefined_name(broken)
        if msg:
            return broken, msg
    return None


def build_name_repairs(data_dir, seed=0):
    rng = random.Random(seed)
    rows = [json.loads(x) for x in open(os.path.join(data_dir, "train_gen.jsonl"))]
    out = []
    for r in rows:
        for _ in range(2):
            c = corrupt_name(r["code"], rng)
            if c:
                out.append(dict(instr=r["instr"], broken=c[0], err=c[1], code=r["code"], family=r["family"]))
    with open(os.path.join(data_dir, "train_repair_name.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    return len(out)
