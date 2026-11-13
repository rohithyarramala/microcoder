"""Undefined-name gate: a tiny static check that also runs in MicroPython.

The commonest functional failure of the tiny model is using a name that was
never defined (e.g. returning `s` instead of a constant). compile() accepts
such code, so we add a second, equally cheap gate: every identifier that is
read must be a parameter, a local assignment/loop variable, the function
itself, a keyword or a builtin. Written without `re`/`ast` so the identical
code can run on the device (app/ref.py imports the same logic).
"""

KEYWORDS = set("False None True and as assert async await break class continue def del elif else except finally for "
               "from global if import in is lambda nonlocal not or pass raise return try while with yield".split())
BUILTINS = set("abs all any bin bool bytes chr dict divmod enumerate filter float format hex int isinstance len list map "
               "max min next oct ord pow print range repr reversed round set slice sorted str sum tuple type zip "
               "math ValueError TypeError Exception".split())


def _strip(code):
    out, i, n = [], 0, len(code)
    while i < n:
        ch = code[i]
        if ch in "'\"":
            q = ch
            j = i + 1
            while j < n and code[j] != q and code[j] != "\n":
                j += 2 if code[j] == "\\" else 1
            out.append(" " * (min(j, n - 1) - i + 1))
            i = j + 1
        elif ch == "#":
            while i < n and code[i] != "\n":
                out.append(" ")
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _isid0(c):
    return c.isalpha() or c == "_"


def _isid(c):
    return c.isalpha() or c.isdigit() or c == "_"


def _tokens(line):
    toks, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if _isid0(c):
            j = i
            while j < n and _isid(line[j]):
                j += 1
            toks.append(("id", line[i:j], i))
            i = j
        elif c.isdigit():
            j = i
            while j < n and (_isid(line[j]) or line[j] == "."):
                j += 1
            toks.append(("num", line[i:j], i))
            i = j
        elif c in " \t":
            i += 1
        else:
            two = line[i:i + 2]
            if two in ("==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "//", "**", "^=", "|=", "&="):
                toks.append(("op", two, i))
                i += 2
            else:
                toks.append(("op", c, i))
                i += 1
    return toks


def undefined_name(code):
    """Return 'NameError: name 'x' is not defined (line n)' or None."""
    lines = _strip(code).split("\n")
    defined = set()
    tl = [_tokens(l) for l in lines]
    for toks in tl:  # pass 1: collect definitions
        ids = [t[1] for t in toks]
        if ids[:1] == ["def"] and len(toks) > 1:
            defined.add(toks[1][1])
            depth = 0
            for t in toks[2:]:
                if t[1] == "(":
                    depth += 1
                elif t[1] == ")":
                    depth -= 1
                elif t[0] == "id" and depth == 1:
                    defined.add(t[1])
            continue
        for k, t in enumerate(toks):
            if t[1] == "for":  # for a, b in ...
                m = k + 1
                while m < len(toks) and toks[m][1] != "in":
                    if toks[m][0] == "id":
                        defined.add(toks[m][1])
                    m += 1
            if t[1] in ("=", "+=", "-=", "*=", "/=", "%=", "^=", "|=", "&=") and k > 0:
                m = k - 1  # targets left of the first assignment operator
                while m >= 0 and toks[m][1] not in ("(", "[", ":"):
                    if toks[m][0] == "id" and (m == 0 or toks[m - 1][1] != "."):
                        defined.add(toks[m][1])
                    m -= 1
            if t[1] == "as" and k + 1 < len(toks):
                defined.add(toks[k + 1][1])
    for ln, toks in enumerate(tl, 1):  # pass 2: every read must be known
        for k, t in enumerate(toks):
            if t[0] != "id" or t[1] in KEYWORDS or t[1] in BUILTINS or t[1] in defined:
                continue
            if k > 0 and toks[k - 1][1] == ".":
                continue  # attribute
            if k + 1 < len(toks) and toks[k + 1][1] == "=" and k > 0 and toks[k - 1][1] in ("(", ","):
                continue  # keyword argument
            return "NameError: name '%s' is not defined (line %d)" % (t[1], ln)
    return None
