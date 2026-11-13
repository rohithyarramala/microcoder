# Extra variant after the main queue: repair data that also covers undefined names (lint gate).
set -x
while ! grep -q POST_DONE logs_post.txt 2>/dev/null; do sleep 30; done
python3 -W ignore -m microcoder.evaluate ckpt/m1g_s0.pt --methods a_resample_lint,a_ref_lint --out m1g_s0_lint > logs_eval_m1g_lint.txt 2>&1
python3 scripts/merge_summary.py m1g_s0 m1g_s0_lint
python3 -m microcoder.train --name m1n_s0 --tok data/tok2048_final.json --dropout 0.1 --steps 4000 --seed 0 --mix gen+repair+name > logs_m1n_s0.txt 2>&1
python3 -W ignore -m microcoder.evaluate ckpt/m1n_s0.pt > logs_eval_m1n_s0.txt 2>&1
python3 scripts/make_tables.py > logs_tables.txt 2>&1
echo EXTRA_DONE
