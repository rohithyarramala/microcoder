"""Decoding strategies compared in the paper.

Every strategy has the same budget: at most 1 + N_MAX = 4 generations.

greedy        one-shot greedy decoding
anchor        greedy, but the answer is forced to start with the signature
              `def name(args):` copied verbatim from the instruction
              (signature anchoring: zero parameters, zero extra compute)
ref           Recursive Error-Feedback on top of plain greedy
a_resample    anchor + sample-then-filter: while compile() fails, draw a new
              sample (T=0.7) from the original prompt (no feedback)
a_ref         anchor + REF with the full CPython error message
a_ref_line    anchor + REF with a MicroPython-style message that only keeps the
              line number ("SyntaxError: invalid syntax (line N)")
a_ref_noerr   anchor + REF with a constant message ("SyntaxError")
a_resample_lint, a_ref_lint
              as a_resample / a_ref, but the gate is compile() AND the
              undefined-name check of lint.py (both run on the device)
"""
import re

import torch

from .dataset import prompt_gen, prompt_repair
from .sandbox import syntax_error
from .lint import undefined_name

N_MAX = 3
MAX_NEW = 120
TEMP = 0.7
METHODS = ["greedy", "anchor", "ref", "a_resample", "a_ref", "a_ref_line", "a_ref_noerr", "a_resample_lint", "a_ref_lint"]
SIG = re.compile(r"([A-Za-z_]\w*)\(([^()]*)\)")


class Generator:
    """Wraps a torch MicroCoder or the C engine (anything with .generate and .c)."""

    def __init__(self, model, tok):
        self.m, self.tok = model, tok
        self.stop = tok.sp["<e>"]
        self.memo = {}

    def __call__(self, prompt, temperature=0.0, g=None):
        if temperature <= 0 and prompt in self.memo:
            return self.memo[prompt]
        ids = self.tok.encode(prompt)
        budget = min(MAX_NEW, self.m.c.ctx - len(ids))
        if budget < 8:
            return None  # prompt does not fit the context window
        out = self.m.generate(ids, budget, self.stop, temperature=temperature, gen=g)
        text = self.tok.decode(out)
        if temperature <= 0:
            self.memo[prompt] = text
        return text


def signature_prefix(instr):
    m = SIG.search(instr)
    return "def %s(%s):" % (m.group(1), m.group(2)) if m else ""  # no newline: "\n" + indent is one token


def _msg(err, mode):
    if mode == "full":
        return err
    if mode == "line":
        m = re.search(r"\(line (\d+)\)", err)
        return "SyntaxError: invalid syntax (line %s)" % (m.group(1) if m else "?")
    return "SyntaxError"


def solve(gen, instr, method, seed=0):
    """Return dict(code, valid, n_gens)."""
    g = torch.Generator().manual_seed(seed)
    pre = signature_prefix(instr) if method.startswith("a") else ""

    def draft(prompt, temp=0.0):
        out = gen(prompt + pre, temp, g)
        return None if out is None else pre + out

    lint = method.endswith("_lint")
    gate = (lambda c: syntax_error(c) or undefined_name(c)) if lint else syntax_error
    code = draft(prompt_gen(instr))
    err = gate(code) if code is not None else "too long"
    n = 1
    if method in ("greedy", "anchor"):
        return dict(code=code, valid=err is None, n_gens=n)
    mode = {"ref": "full", "a_ref": "full", "a_ref_line": "line", "a_ref_noerr": "none", "a_ref_lint": "full"}.get(method)
    while err is not None and n <= N_MAX:
        if method.startswith("a_resample") or code is None:
            new = draft(prompt_gen(instr), TEMP)
        else:
            p = prompt_repair(instr, code, _msg(err, mode))
            new = draft(p)
            if new is None or new == code:  # greedy repair repeated the draft: sample instead
                new = draft(p, TEMP)
        n += 1
        if new is None:
            break
        code = new
        err = gate(code)
    return dict(code=code, valid=syntax_error(code) is None if code is not None else False, n_gens=n)
