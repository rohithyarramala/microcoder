"""Deployment numbers that can be computed or measured without the chip.

python scripts/deploy_stats.py ckpt/m1_s0.pt  -> results/deploy.json

* exact flash / RAM footprint of the INT8 engine (from the blob and mc_state_bytes)
* multiply-accumulates per generated token (exact count from the engine's loops)
* host latency of the same C engine, measured by the host simulation of the firmware
"""
import json, os, subprocess, sys
sys.path.insert(0, ".")
import torch
from microcoder.export import export_model, export_tok
from microcoder.model import Config, param_breakdown

ck = torch.load(sys.argv[1])
c = Config(**ck["cfg"])
mb, tb = export_model(ck), export_tok(ck["args"]["tok"])
d, f, hd = c.d, c.ff_mult * c.d, c.d // c.heads
state_floats = 3 * d + f + d + c.ctx + c.vocab + 2 * c.layers * c.ctx * d + 2 * c.ctx * (hd // 2)
kv_bytes = 2 * c.layers * c.ctx * d * 4


def macs(pos):
    return c.layers * (3 * d * d + d * d + 2 * d * f + 2 * (pos + 1) * d) + c.vocab * d


out = dict(
    params=sum(v.numel() for k, v in ck["model"].items()),
    fp32_bytes=4 * sum(v.numel() for v in ck["model"].values()),
    int8_model_bytes=len(mb), tokenizer_bytes=len(tb), flash_total_bytes=len(mb) + len(tb),
    state_bytes=4 * state_floats, kv_cache_bytes=kv_bytes, activation_bytes=4 * state_floats - kv_bytes,
    macs_first=macs(0), macs_middle=macs(127), macs_last=macs(255),
)
# host simulation timing (same firmware main.c + engine), on both test splits
subprocess.run([sys.executable, "scripts/make_firmware_assets.py", sys.argv[1], "test_id,test_ood"], check=True)
subprocess.run(["make", "-s", "-B", "-C", "esp32/host_sim"], check=True)
log = subprocess.run(["./esp32/host_sim/sim"], capture_output=True, text=True, check=True).stdout
open("results/host_sim_log.txt", "w").write(log)
sc = json.loads(subprocess.run([sys.executable, "scripts/score_device_log.py", "results/host_sim_log.txt", "test_id,test_ood"],
                               capture_output=True, text=True, check=True).stdout)
out.update(host_decode_ms_per_token=sc["decode_ms_per_token_median"], host_prefill_ms_per_token=sc["prefill_ms_per_token_median"],
           host_end_to_end_ms_median=sc["end_to_end_ms_median"], host_sim_greedy_valid_pct=sc["valid_pct"],
           host_sim_greedy_pass_pct=sc["pass_pct"], host_sim_n=sc["n_scored"],
           host_cpu=subprocess.run("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2", shell=True, capture_output=True, text=True).stdout.strip())
json.dump(out, open("results/deploy.json", "w"), indent=1)
print(json.dumps(out, indent=1))
