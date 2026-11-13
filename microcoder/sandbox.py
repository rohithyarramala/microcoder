"""Compile gate and functional-correctness checker.

`syntax_error(code)` is the verification oracle used by REF: it calls the
built-in compile() exactly as an on-device MicroPython `compile()` would, and
returns a short, line-numbered error message or None.

`run_tests(code, fn, tests, expected)` executes a candidate in a forked child
process with a CPU timeout and compares outputs with the reference outputs.
"""
import math
import os
import pickle
import resource
import signal


def syntax_error(code):
    try:
        compile(code, "<gen>", "exec")
        return None
    except SyntaxError as e:
        msg = e.msg if e.msg else "invalid syntax"
        return "SyntaxError: %s (line %s)" % (msg, e.lineno)
    except Exception as e:  # e.g. ValueError for null bytes
        return "%s: %s" % (type(e).__name__, str(e)[:60])


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or (isinstance(a, bool) and isinstance(b, bool) and a == b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) or isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


def _child(code, fn, tests, w):
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    signal.alarm(2)
    out = []
    try:
        g = {"__name__": "__gen__"}
        exec(compile(code, "<gen>", "exec"), g)
        f = g[fn]
        for args in tests:
            try:
                out.append(("ok", f(*[pickle.loads(pickle.dumps(a)) for a in args])))
            except BaseException as e:  # noqa
                out.append(("err", type(e).__name__))
    except BaseException as e:  # noqa
        out = [("err", type(e).__name__)] * len(tests)
    try:
        data = pickle.dumps(out)
    except Exception:
        data = pickle.dumps([("err", "Unpicklable")] * len(tests))
    os.write(w, data)
    os._exit(0)


def run_tests(code, fn, tests, expected):
    """Return (passed_all, n_passed, first_error_name_or_None)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            _child(code, fn, tests, w)
        finally:
            os._exit(1)
    os.close(w)
    chunks = []
    while True:
        b = os.read(r, 65536)
        if not b:
            break
        chunks.append(b)
    os.close(r)
    os.waitpid(pid, 0)
    try:
        out = pickle.loads(b"".join(chunks))
    except Exception:
        return False, 0, "Timeout"
    n = 0
    err = None
    for (kind, val), exp in zip(out, expected):
        if kind == "ok" and _same(val, exp):
            n += 1
        elif kind == "err" and err is None:
            err = val
    return n == len(expected), n, err


def reference_outputs(code, fn, tests):
    g = {}
    exec(code, g)
    return [g[fn](*[pickle.loads(pickle.dumps(a)) for a in t]) for t in tests]


def _child_asserts(code, setup, asserts, w):
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    signal.alarm(3)
    ok = 0
    try:
        g = {"__name__": "__gen__"}
        exec(compile(code, "<gen>", "exec"), g)
        if setup:
            exec(setup, g)
        for a in asserts:
            try:
                exec(a, g)
                ok += 1
            except BaseException:  # noqa
                pass
    except BaseException:  # noqa
        pass
    os.write(w, pickle.dumps(ok))
    os._exit(0)


def run_asserts(code, setup, asserts):
    """MBPP-style evaluation. Returns True if every assert passes."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            _child_asserts(code, setup, asserts, w)
        finally:
            os._exit(1)
    os.close(w)
    data = os.read(r, 1024)
    os.close(r)
    os.waitpid(pid, 0)
    try:
        return pickle.loads(data) == len(asserts)
    except Exception:
        return False
