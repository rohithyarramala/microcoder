"""Score a UART log from the ESP32 firmware (or the host simulation).

python scripts/score_device_log.py device_log.txt [split]  -> prints accuracy + latency stats, writes results/device_<name>.json
"""
import json, re, statistics, sys
sys.path.insert(0, ".")
from microcoder.sandbox import run_tests, syntax_error

# ESP-IDF consoles print CRLF by default (CONFIG_NEWLIB_STDOUT_LINE_ENDING_CRLF); normalise before parsing
log = open(sys.argv[1], errors="replace").read().replace("\r\n", "\n")
split = sys.argv[2] if len(sys.argv) > 2 else json.load(open("esp32/firmware/main/prompts.json"))["split"]
tasks = []
for sp in split.split(","):
    tasks += [json.loads(x) for x in open(f"data/{sp}.jsonl")]
info = dict(re.findall(r"(\w+)=(\S+)", (re.search(r"MC_INFO (.*)", log) or [""])[0] if re.search(r"MC_INFO (.*)", log) else ""))
blocks = re.findall(r"MC_BEGIN (\d+)\n(.*?)\nMC_END \d+ prompt_tokens=(\d+) new_tokens=(\d+) prefill_us=(\d+) decode_us=(\d+)", log, re.S)
valid = passed = 0
per_tok, prefill_per_tok, totals = [], [], []
for i, code, pt, nt, pre, dec in blocks:
    t = tasks[int(i)]
    ok_syn = syntax_error(code) is None
    valid += ok_syn
    if ok_syn:
        passed += run_tests(code, t["fn"], t["tests"], t["expected"])[0]
    pt, nt, pre, dec = int(pt), int(nt), int(pre), int(dec)
    if nt: per_tok.append(dec / nt / 1000)
    prefill_per_tok.append(pre / pt / 1000)
    totals.append((pre + dec) / 1000)
n = len(blocks)
res = dict(log=sys.argv[1], split=split, n_scored=n, valid=valid, passed=passed,
           valid_pct=round(100 * valid / max(1, n), 1), pass_pct=round(100 * passed / max(1, n), 1),
           decode_ms_per_token_median=round(statistics.median(per_tok), 3) if per_tok else None,
           prefill_ms_per_token_median=round(statistics.median(prefill_per_tok), 3) if prefill_per_tok else None,
           end_to_end_ms_median=round(statistics.median(totals), 1) if totals else None,
           end_to_end_ms_p95=round(sorted(totals)[int(0.95 * (len(totals) - 1))], 1) if totals else None, info=info)
print(json.dumps(res, indent=1))
name = re.sub(r"\W+", "_", sys.argv[1].rsplit("/", 1)[-1])
json.dump(res, open(f"results/device_{name}.json", "w"), indent=1)
