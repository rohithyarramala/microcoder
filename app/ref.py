# MicroPython: on-device Recursive Error-Feedback (REF) loop.
# Works on an ESP32 running MicroPython built with the `microcoder` native
# module (esp32/mp_module). The verification gate is MicroPython's own compile().
import io
import sys
import microcoder
from lint import undefined_name  # same file as microcoder/lint.py

N_MAX = 3


def _syntax_error(code):
    try:
        compile(code, "<gen>", "exec")
        return None
    except SyntaxError as e:
        buf = io.StringIO()
        sys.print_exception(e, buf)
        txt = buf.getvalue()
        line = "?"
        i = txt.find('line ')
        if i >= 0:
            j = i + 5
            while j < len(txt) and txt[j].isdigit():
                j += 1
            line = txt[i + 5:j]
        msg = e.args[0] if e.args else "invalid syntax"
        return "SyntaxError: %s (line %s)" % (msg, line)


def _gate(code):
    """compile() first, then the undefined-name check."""
    return _syntax_error(code) or undefined_name(code)


def _isw(c):
    return c.isalpha() or c.isdigit() or c == "_"


def signature_prefix(instr):
    """Signature anchoring: 'def name(args):' copied from the instruction."""
    s = instr
    for i in range(1, len(s)):
        if s[i] != "(" or not _isw(s[i - 1]):
            continue
        j = i
        while j > 0 and _isw(s[j - 1]):
            j -= 1
        while j < i and not (s[j].isalpha() or s[j] == "_"):
            j += 1
        if j == i:
            continue
        k = i + 1
        while k < len(s) and s[k] not in "()":
            k += 1
        if k < len(s) and s[k] == ")":
            return "def " + s[j:k + 1] + ":"
    return ""


def _gen(prompt, pre, temperature=None):
    out = microcoder.generate(prompt + pre, temperature=temperature)
    return None if out is None else pre + out


def solve(instr, n_max=N_MAX):
    """Signature anchoring + REF with the compile()+name gate. Return (code_or_None, n_generations, trace)."""
    pre = signature_prefix(instr)
    code = _gen("<q>" + instr + "<c>", pre)
    trace = [code]
    err = _gate(code) if code is not None else "too long"
    n = 1
    while err is not None and n <= n_max:
        p = "<q>" + instr + "<b>" + code + "<r>" + err + "<c>"
        new = _gen(p, pre)
        if new is None or new == code:
            new = _gen(p, pre, 0.7)
        n += 1
        if new is None:
            break
        code = new
        trace.append(code)
        err = _gate(code)
    return (code if err is None else None), n, trace
