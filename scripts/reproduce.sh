#!/usr/bin/env bash
# Rebuild every result in the paper from scratch (CPU only, ~8 h on 2 cores; much faster with a GPU).
set -euo pipefail
pip install torch tokenizers numpy
python3 scripts/build_data.py                 # data/ : train corpus + test splits (deterministic)
python3 scripts/check_leakage.py              # fails loudly if any test instruction leaks into training
python3 scripts/build_dev.py                  # data_dev_*/ : model-selection splits
bash scripts/dev_select.sh                    # selection runs, evaluated on the dev split only
bash scripts/final_runs.sh                    # final models (3 seeds + ablations) on test splits
python3 scripts/deploy_stats.py ckpt/m1_s0.pt # INT8 export, memory, MACs, host simulation of the firmware
python3 scripts/make_tables.py                # paper/generated/*.tex
