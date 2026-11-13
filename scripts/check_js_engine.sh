#!/usr/bin/env bash
# Check that the browser engine (app/web/engine.js, a JavaScript port of the C engine) reproduces the
# firmware host simulation exactly: same model (ckpt/s_s0.pt), same 624 prompts, SA + greedy decoding.
# Needs node; compares against lab_sim_log.txt (bash esp32/run_lab.sh sim). Writes results/js_engine_check.json.
set -euo pipefail
D=$(mktemp -d)
python - "$D" <<'PY'
import sys, json, torch
sys.path.insert(0, ".")
from microcoder.export import export_model, export_tok
d = sys.argv[1]; ck = torch.load("ckpt/s_s0.pt", weights_only=False)
open(f"{d}/model.bin", "wb").write(export_model(ck)); open(f"{d}/tok.bin", "wb").write(export_tok(ck["args"]["tok"]))
json.dump([json.loads(x)["instr"] for f in ("test_id", "test_ood") for x in open(f"data/{f}.jsonl")], open(f"{d}/prompts.json", "w"))
PY
node scripts/check_js_engine.js app/web/engine.js "$D" 624
python - "$D" <<'PY'
import sys, json, re
log = open("lab_sim_log.txt").read()
sim = {int(i): c.strip() for i, c in re.findall(r"MC_BEGIN (\d+)\n(.*?)\nMC_END \d+ ", log, re.S)}
js = json.load(open(sys.argv[1] + "/js_out.json"))
same = sum(1 for i, c in enumerate(js) if c is not None and c.strip() == sim.get(i))
r = dict(n=len(js), identical_to_host_sim=same)
print(r); json.dump(r, open("results/js_engine_check.json", "w"))
PY
