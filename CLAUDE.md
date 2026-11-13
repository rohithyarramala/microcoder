# MicroCoder — context for Claude Code

This repository is the rebuilt research project behind the paper in `paper/` (an honest, reproducible
study of 0.5–1M-parameter Python code models on microcontrollers). The previous version of this work was
rejected by IEEE IoT-J for inconsistent numbers, train/test leakage, a non-reproducible artifact and
unmeasured hardware claims. The rules below exist so that never happens again.

## Hard rules
1. **Never type a number into the paper.** Every number comes from `results/` via `python scripts/make_tables.py`
   (writes `paper/generated/numbers.tex` and `paper/generated/tab_*.tex`). If a number is missing, run the
   experiment; do not estimate it.
2. **Never tune on the test splits** (`data/test_id.jsonl`, `data/test_ood.jsonl`). Model selection uses
   `data_dev_*/test_dev.jsonl` (build with `python scripts/build_dev.py`).
3. **Run `python scripts/check_leakage.py` after any data change.** It must pass.
4. **Simulator timing is not hardware timing.** QEMU/Wokwi/host-sim runs check function and accuracy only.
   Only a log from a physical ESP32-S3 (`esp32/run_lab.sh board`) may fill the measured-latency rows; the
   paper switches automatically between "measured" and "estimate" via `\ifdefined\devdecode`.
5. Report negative results as they are (out-of-family ≈1–2 %, MBPP 0/500, error message does not help).

## Layout
- `microcoder/` model, data, training, evaluation, decoding strategies (`infer.py`), lint gate
- `esp32/engine/` INT8 C engine + tokenizer; `esp32/firmware/` ESP-IDF app; `esp32/host_sim/`; `esp32/mp_module/`
- `app/` on-device REF loop, field assistant, browser playground
- `scripts/` everything that produces data, results, tables; `results/` per-task outputs
- `ckpt/` trained models: `m1_s{0,1,2}` (1.05M), `s_s{0,1,2}` (0.53M, deployed), `m1g_s0`, `m1n_s0`, `v0_m1_s0`, `v1_m1_s0`, `deploy` (application model, all families — never evaluate it on the test splits)

## Common commands
```bash
python -m microcoder.evaluate ckpt/s_s0.pt                      # all strategies, both test splits
python scripts/deploy_stats.py ckpt/s_s0.pt                     # footprint, MACs, host-sim timing
python scripts/make_tables.py && (cd paper && latexmk -pdf main) # regenerate the paper
bash esp32/run_lab.sh sim                                       # host simulation of the firmware
```
