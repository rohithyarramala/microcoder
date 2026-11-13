"""Generate every number and table in the paper from results/ — nothing is typed by hand.

python scripts/make_tables.py   ->  paper/generated/{numbers.tex, tab_*.tex}, results/all_numbers.json
"""
import re
import glob, json, math, os, statistics, sys
sys.path.insert(0, ".")
from microcoder.model import Config, param_breakdown

OUT = "paper/generated"
os.makedirs(OUT, exist_ok=True)
macros, allnum = {}, {}

LABEL = {"greedy": "Greedy (one-shot)", "anchor": "Signature anchoring (SA)", "ref": "REF (no SA)",
         "a_resample": "SA + resample-until-compiles", "a_ref": "SA + REF (full message)",
         "a_ref_line": "SA + REF (line-only message)", "a_ref_noerr": "SA + REF (no message)",
         "a_resample_lint": "SA + resample, compile+name gate", "a_ref_lint": "SA + REF, compile+name gate"}
METHODS = list(LABEL)


def S(name):
    p = f"results/{name}/summary.json"
    return json.load(open(p)) if os.path.exists(p) else None


def mac(name, val):
    name = "".join(ch for ch in name if ch.isalpha())
    macros[name] = val


def fmt_ci(m):
    return f"{m['pass_pct']:.1f} [{m['pass_ci'][0]:.0f}, {m['pass_ci'][1]:.0f}]"


# ---------------------------------------------------------------- Table: params
ck_cfg = {"S": Config(vocab=2048, d=96, layers=3, heads=4), "M": Config(vocab=2048, d=128, layers=4, heads=4),
          "L": Config(vocab=2048, d=192, layers=6, heads=6)}
rows = []
for k, c in ck_cfg.items():
    b = param_breakdown(c)
    rows.append(f"{k} & {c.layers} & {c.d} & {c.heads} & {b['embedding']:,} & {b['layers_total']:,} & {b['total']:,} & {b['total'] / 2**20:.2f} \\\\")
    allnum[f"params_{k}"] = b
open(f"{OUT}/tab_params.tex", "w").write("\n".join(rows) + "\n")
bM = param_breakdown(ck_cfg["M"])
mac("paramsM", f"{bM['total']:,}")
mac("embM", f"{bM['embedding']:,}")

# ---------------------------------------------------------------- main results (3 seeds of M)
seeds = [s for s in (S("m1_s0"), S("m1_s1"), S("m1_s2")) if s]
allnum["n_seeds"] = len(seeds)
mac("nseeds", str(len(seeds)))
lines = []
for m in METHODS:
    cells = []
    for sp in ("test_id", "test_ood"):
        vs = [s["splits"][sp]["methods"][m]["valid_pct"] for s in seeds if m in s["splits"][sp]["methods"]]
        ps = [s["splits"][sp]["methods"][m]["pass_pct"] for s in seeds if m in s["splits"][sp]["methods"]]
        if not ps:
            cells += ["--", "--"]; continue
        sd = lambda x: (statistics.stdev(x) if len(x) > 1 else 0.0)
        cells += [f"{statistics.mean(vs):.1f}$\\pm${sd(vs):.1f}", f"{statistics.mean(ps):.1f}$\\pm${sd(ps):.1f}"]
        allnum[f"main_{sp}_{m}"] = dict(valid_mean=statistics.mean(vs), valid_sd=sd(vs), pass_mean=statistics.mean(ps), pass_sd=sd(ps), seeds=ps)
        mac(f"{sp}{m}pass", f"{statistics.mean(ps):.1f}")
        mac(f"{sp}{m}valid", f"{statistics.mean(vs):.1f}")
        mac(f"{sp}{m}passsd", f"{sd(ps):.1f}")
        mac(f"{sp}{m}passmin", f"{min(ps):.1f}")
        mac(f"{sp}{m}passmax", f"{max(ps):.1f}")
    g = [s["splits"]["test_id"]["methods"][m]["mean_gens"] for s in seeds if m in s["splits"]["test_id"]["methods"]]
    if g:
        mac(f"testid{m}gens", f"{statistics.mean(g):.2f}")
    cells.append(f"{statistics.mean(g):.2f}" if g else "--")
    lines.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
open(f"{OUT}/tab_main.tex", "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- size comparison over seeds (S vs M)
for tag, names in (("Small", ["s_s0", "s_s1", "s_s2"]), ("Medium", ["m1_s0", "m1_s1", "m1_s2"])):
    ss = [S(n) for n in names if S(n)]
    mac(f"seeds{tag}n", str(len(ss)))
    for sp in ("test_id", "test_ood"):
        for m in ("greedy", "anchor", "a_ref", "a_ref_lint"):
            ps = [x["splits"][sp]["methods"][m]["pass_pct"] for x in ss if m in x["splits"][sp]["methods"]]
            if ps:
                mac(f"seeds{tag}{sp}{m}", f"{statistics.mean(ps):.1f}")
                mac(f"seeds{tag}{sp}{m}sd", f"{(statistics.stdev(ps) if len(ps) > 1 else 0):.1f}")
                allnum[f"seeds_{tag}_{sp}_{m}"] = ps

# ---------------------------------------------------------------- per-seed CI table for seed 0
s0 = S("m1_s0")
if s0:
    lines = []
    for m in METHODS:
        a, b = s0["splits"]["test_id"]["methods"][m], s0["splits"]["test_ood"]["methods"][m]
        lines.append(f"{LABEL[m]} & {a['passed']}/{s0['splits']['test_id']['n']} & {fmt_ci(a)} & {b['passed']}/{s0['splits']['test_ood']['n']} & {fmt_ci(b)} \\\\")
    open(f"{OUT}/tab_seed0_ci.tex", "w").write("\n".join(lines) + "\n")
    # per category (in-family split) for the best practical method
    cats = s0["splits"]["test_id"]["methods"]["a_ref"]["by_cat"]
    lines = []
    for c in sorted(cats):
        g0 = s0["splits"]["test_id"]["methods"]["greedy"]["by_cat"][c]
        g1 = s0["splits"]["test_id"]["methods"]["anchor"]["by_cat"][c]
        g2 = s0["splits"]["test_id"]["methods"]["a_ref"]["by_cat"][c]
        lines.append(f"{c} & {g0[1]} & {g0[0]} & {g1[0]} & {g2[0]} \\\\")
    open(f"{OUT}/tab_cat.tex", "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- ablations (seed 0 rows)
abl = [("v0_m1_s0", "First attempt: 500 draws, no dropout"), ("v1_m1_s0", "Second attempt: random names 50\\%"),
       ("m1g_s0", "No repair data (generation only)"),
       ("m1_s0", "Final recipe (M, 1.05M params)"), ("m1n_s0", "+ undefined-name repair data"),
       ("s_s0", "Smaller model (S)"), ("l_s0", "Larger model (L)"), ("m1_s0_int8", "Final recipe M, INT8 C engine"), ("s_s0_int8", "Smaller model S, INT8 C engine")]
lines = []
for key, lab in abl:
    s = S(key)
    if not s:
        continue
    mm = s["splits"]["test_id"]["methods"]
    oo = s["splits"]["test_ood"]["methods"] if "test_ood" in s["splits"] else {}
    get = lambda d, m, f: f"{d[m][f]:.1f}" if m in d else "--"
    lines.append(f"{lab} & {s['n_params']/1e6:.2f}M & {get(mm,'greedy','pass_pct')} & {get(mm,'anchor','pass_pct')} & "
                 f"{get(mm,'a_ref','pass_pct')} & {get(mm,'a_ref_lint','pass_pct')} & {get(oo,'a_ref_lint','pass_pct')} \\\\")
    allnum[f"abl_{key}"] = {m: dict(valid=mm[m]["valid_pct"], pass_=mm[m]["pass_pct"]) for m in mm}
    alias = {"v0_m1_s0": "First", "v1_m1_s0": "Second", "m1g_s0": "Genonly", "m1_s0": "Final", "m1n_s0": "Namerep",
             "m1sr_s0": "Selfrep", "s_s0": "Small", "l_s0": "Large", "m1_s0_int8": "Int", "s_s0_int8": "Smallint"}[key]
    for m in mm:
        mac(f"abl{alias}{m}pass", f"{mm[m]['pass_pct']:.1f}")
        mac(f"abl{alias}{m}valid", f"{mm[m]['valid_pct']:.1f}")
    for m in oo:
        mac(f"ablood{alias}{m}pass", f"{oo[m]['pass_pct']:.1f}")
    mac(f"abl{alias}params", f"{s['n_params']/1e6:.2f}")
open(f"{OUT}/tab_ablation.tex", "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- dev-split model selection
lines = []
for key, lab in [("dev_noaug", "No augmentation"), ("dev_p0", "Varied openings"),
                 ("dev_p0.15", "Varied openings + random names (15\\%)"), ("dev_p0.5", "Varied openings + random names (50\\%)")]:
    s_ = S(key)
    if not s_:
        continue
    d = s_["splits"]["test_dev"]["methods"]
    lines.append(f"{lab} & {d['greedy']['pass_pct']:.1f} & {d['anchor']['pass_pct']:.1f} & {d['a_ref']['pass_pct']:.1f} \\\\")
    al = {"dev_noaug": "Noaug", "dev_p0": "Openers", "dev_p0.15": "Namesfifteen", "dev_p0.5": "Namesfifty"}[key]
    for m in d:
        mac(f"dev{al}{m}pass", f"{d[m]['pass_pct']:.1f}")
    allnum[key] = {m: d[m]["pass_pct"] for m in d}
open(f"{OUT}/tab_dev.tex", "w").write("\n".join(lines) + "\n")
ndev = S("dev_p0")["splits"]["test_dev"]["n"] if S("dev_p0") else 0
mac("ndev", str(ndev))

# ---------------------------------------------------------------- failure analysis (seed 0, SA+REF)
fp = "results/failures_m1_s0_a_ref.json"
if os.path.exists(fp):
    fc = json.load(open(fp))["test_id"]["counts"]
    mac("failidtotal", str(sum(v for k, v in fc.items() if k != "pass" and not k.startswith("runtime:"))))
    mac("failidruntime", str(fc.get("runtime exception", 0)))
    mac("failidwrong", str(fc.get("wrong values", 0)))
    mac("failidpartly", str(fc.get("partly correct values", 0)))
    mac("failidsyntax", str(fc.get("syntax error", 0)))
    mac("failidnameerr", str(fc.get("runtime:NameError", 0)))
    allnum["failures"] = fc

# ---------------------------------------------------------------- self-repair data yield
if os.path.exists("logs_selfrepair.txt"):
    import re as _re
    m_ = _re.search(r"(\d+) self-repair pairs from (\d+) samples", open("logs_selfrepair.txt").read())
    if m_:
        mac("selfrepairpairs", m_.group(1)); mac("selfrepairsamples", f"{int(m_.group(2)):,}")

# ---------------------------------------------------------------- MBPP
mb = S("m1_s0_mbpp")
if mb:
    d = mb["splits"]["mbpp"]["methods"]
    for m in d:
        mac(f"mbpp{m}valid", f"{d[m]['valid_pct']:.1f}")
        mac(f"mbpp{m}pass", f"{d[m]['pass_pct']:.1f}")
        mac(f"mbpp{m}passed", str(d[m]["passed"]))
    allnum["mbpp"] = d

# ---------------------------------------------------------------- leakage + data
lk = json.load(open("results/leakage.json"))
meta = json.load(open("data/meta.json"))
for sp in ("test_id", "test_ood"):
    mac(f"n{sp}", str(lk[sp]["n"]))
mac("codeoverlapid", str(lk["test_id"]["reference_code_identical_to_a_train_example"]))
mac("ngramood", str(lk["test_ood"]["tasks_sharing_any_8gram_with_train"]))
mac("ntraingen", f"{meta['n_train_gen']:,}")
mac("ntrainrepair", f"{meta['n_train_repair']:,}")
mac("nfamtrain", str(len(meta["train_families"])))
mac("nfamheld", str(len(meta["heldout_families"])))
allnum["leakage"] = lk

# ---------------------------------------------------------------- deployment
dp = "results/deploy.json"
if os.path.exists(dp):
    d = json.load(open(dp))
    allnum["deploy"] = d
    for k, v in d.items():
        if isinstance(v, (int, float)):
            mac("dep" + k, f"{v:,}" if isinstance(v, int) else f"{v:g}")
if os.path.exists("results/deploy_small.json"):
    d = json.load(open("results/deploy_small.json"))
    allnum["deploy_small"] = d
    for k, v in d.items():
        if isinstance(v, (int, float)):
            mac("dsm" + k, f"{v:,}" if isinstance(v, int) else f"{v:g}")
    # Analytical ESP32-S3 estimate (used only if no board run exists); assumptions are stated in the paper.
    CYC_PER_MAC, F_CPU, FLASH_BPS = 4, 240e6, 40e6
    mid = d["macs_middle"]
    lo = mid * CYC_PER_MAC / F_CPU * 1e3                      # compute-bound
    hi = lo + d["int8_model_bytes"] / FLASH_BPS * 1e3          # + streaming all weights from flash each token
    mac("estlo", f"{lo:.0f}"); mac("esthi", f"{hi:.0f}")
    allnum["esp32_estimate_ms_per_token"] = dict(lo=lo, hi=hi, cyc_per_mac=CYC_PER_MAC, f_cpu=F_CPU, flash_Bps=FLASH_BPS)
# ESP-IDF build footprint (board config) from `idf.py size`, saved by the esp32-build CI workflow
idf = "results/idf_size_esp32s3_v5.4.2.txt"
if os.path.exists(idf):
    t = open(idf, encoding="utf-8").read()
    def row(name):
        m = re.search(r"\u2502\s*" + re.escape(name) + r"\s*\u2502\s*(\d+)\s*\u2502\s*([\d.]*)\s*\u2502\s*(\d*)\s*\u2502\s*(\d*)\s*\u2502", t)
        return m.groups() if m else None
    app = int(re.search(r"binary size (0x[0-9a-f]+) bytes", t).group(1), 16)
    part = int(re.search(r"Smallest app partition is (0x[0-9a-f]+) bytes", t).group(1), 16)
    fd, fc, dr, ir = row("Flash Data"), row("Flash Code"), row("DIRAM"), row("IRAM")
    sz = dict(app_bytes=app, app_partition_bytes=part, flash_data_bytes=int(fd[0]), flash_code_bytes=int(fc[0]),
              diram_used_bytes=int(dr[0]), diram_total_bytes=int(dr[3]), iram_used_bytes=int(ir[0]), iram_total_bytes=int(ir[3]))
    allnum["idf_size_esp32s3"] = sz
    for k, v in sz.items():
        mac("idf" + k.replace("_", ""), f"{v:,}")
    mac("idfapppct", f"{100 * app / part:.0f}")
    mac("idfdirampct", f"{100 * sz['diram_used_bytes'] / sz['diram_total_bytes']:.1f}")
dev = sorted(glob.glob("results/device_*.json"))
for p in dev:
    allnum[os.path.basename(p)] = json.load(open(p))
# the ESP32-S3 board run (esp32/run_lab.sh board) fills the device macros
board = [p for p in dev if "board" in p]
if board:
    d = json.load(open(board[-1]))
    mac("devdecode", f"{d['decode_ms_per_token_median']:g}")
    mac("devprefill", f"{d['prefill_ms_per_token_median']:g}")
    mac("devendtoend", f"{d['end_to_end_ms_median'] / 1000:.2f}")
    mac("devpass", f"{d['pass_pct']:.1f}")
    mac("devn", str(d["n_scored"]))

with open(f"{OUT}/numbers.tex", "w") as f:
    f.write("% generated by scripts/make_tables.py — do not edit\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
json.dump(allnum, open("results/all_numbers.json", "w"), indent=1, default=str)
print(f"{len(macros)} macros; tables:", sorted(os.listdir(OUT)))
