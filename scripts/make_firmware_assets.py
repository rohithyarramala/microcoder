"""Embed the exported model, tokenizer and benchmark prompts into the firmware sources.

python scripts/make_firmware_assets.py ckpt/<model>.pt [split]
TARGET=mp python scripts/make_firmware_assets.py ckpt/deploy.pt   # model for the MicroPython field assistant
"""
import json, os, sys
sys.path.insert(0, ".")
import torch
from microcoder.export import export_model, export_tok, c_array

ck = torch.load(sys.argv[1])
split = sys.argv[2] if len(sys.argv) > 2 else "test_ood"
mb, tb = export_model(ck), export_tok(ck["args"]["tok"])
src = c_array("mc_model_blob", mb) + c_array("mc_tok_blob", tb)
# TARGET=firmware (default), mp (MicroPython app module) or both
target = os.environ.get("TARGET", "firmware")
dirs = {"firmware": ["esp32/firmware/main"], "mp": ["esp32/mp_module/microcoder"],
        "both": ["esp32/firmware/main", "esp32/mp_module/microcoder"]}[target]
for d in dirs:
    open(os.path.join(d, "model_data.c"), "w").write(src)
tasks = []
for sp in split.split(","):
    tasks += [json.loads(x) for x in open(f"data/{sp}.jsonl")]
if target == "mp":
    print(f"embedded model {len(mb)} B, tokenizer {len(tb)} B into the MicroPython module"); sys.exit(0)
# MC_SHARD=k/N embeds only slice k (0-based) of N; the firmware prints global indices via mc_prompt_offset
off, total = 0, len(tasks)
if os.environ.get("MC_SHARD"):
    k, N = map(int, os.environ["MC_SHARD"].split("/"))
    off, end = k * total // N, (k + 1) * total // N
    tasks = tasks[off:end]
lines = ["/* benchmark prompts generated from data/%s.jsonl */" % split, "const char *const mc_prompts[] = {"]
lines += ["    " + json.dumps(t["instr"]) + "," for t in tasks]
lines += ["};", "const int mc_n_prompts = %d;" % len(tasks), "const int mc_prompt_offset = %d;" % off]
open("esp32/firmware/main/prompts.c", "w").write("\n".join(lines) + "\n")
json.dump(dict(split=split, n=len(tasks), offset=off, total=total), open("esp32/firmware/main/prompts.json", "w"))
print(f"embedded model {len(mb)} B, tokenizer {len(tb)} B, {len(tasks)} prompts from {split} (offset {off} of {total})")
