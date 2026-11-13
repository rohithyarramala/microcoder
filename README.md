# MicroCoder

A 1.05M-parameter (and 0.53M) Python code model that fits in about 1 MB of ESP32-S3 flash, with a
leak-free benchmark, a compiler-in-the-loop repair study, an INT8 C engine, ESP32 firmware and an
offline "field assistant" application. Every number in `paper/` is produced by the scripts here.

## Layout

| Path | What |
|---|---|
| `microcoder/families.py` | 64 task families (sensor/IoT, math, list, string, recursion) with reference solutions and test generators |
| `microcoder/dataset.py` | train/test/dev split construction, syntax and undefined-name corruptions |
| `microcoder/model.py` | the Transformer (RMSNorm, RoPE, tied embeddings, no biases) |
| `microcoder/train.py`, `evaluate.py`, `infer.py` | training, evaluation, decoding strategies (SA, REF, resampling) |
| `microcoder/lint.py` | undefined-name gate (also runs in MicroPython) |
| `microcoder/export.py`, `cengine.py` | INT8 export and ctypes wrapper for the C engine |
| `esp32/engine/` | portable C99 INT8 engine + BPE tokenizer |
| `esp32/firmware/` | ESP-IDF benchmark firmware for ESP32-S3 (+ Wokwi config) |
| `esp32/host_sim/` | runs the unmodified firmware on a PC under ESP32-S3 memory budgets |
| `esp32/mp_module/` | MicroPython native module `microcoder` |
| `app/` | on-device REF loop (`ref.py`), field assistant (`assistant.py`), browser playground (`web/`) |
| `scripts/` | data build, leakage audit, experiment queues, tables, device-log scoring |
| `results/` | per-task outputs and summaries for every run |
| `paper/` | LaTeX source; `generated/` is written by `scripts/make_tables.py` |

## Quick start

```bash
pip install torch tokenizers numpy
python scripts/build_data.py && python scripts/check_leakage.py
python -m microcoder.train --name m1_s0 --tok data/tok2048_final.json --dropout 0.1 --steps 4000
python -m microcoder.evaluate ckpt/m1_s0.pt
python scripts/deploy_stats.py ckpt/m1_s0.pt
python scripts/make_tables.py
```

## Measure on a real ESP32-S3

```bash
bash esp32/run_lab.sh sim                    # no hardware: host simulation of the firmware
pip install pyserial && bash esp32/run_lab.sh board /dev/ttyACM0   # ESP32-S3 board (8 MB flash + PSRAM)
python scripts/make_tables.py                # device numbers land in results/device_*.json
```

See `esp32/README.md` for Wokwi, QEMU and the MicroPython build.
