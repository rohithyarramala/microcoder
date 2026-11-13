set -x
while pgrep -f final_runs.sh >/dev/null; do sleep 30; done
python3 -W ignore -m microcoder.evaluate ckpt/m1_s0.pt --splits mbpp --methods greedy,anchor,a_resample,a_ref --out m1_s0_mbpp > logs_eval_mbpp.txt 2>&1
python3 -m microcoder.export ckpt/m1_s0.pt export/m1_s0 > logs_export.txt 2>&1
python3 scripts/test_engine.py ckpt/m1_s0.pt export/m1_s0 > logs_test_engine.txt 2>&1
python3 -W ignore -m microcoder.evaluate ckpt/m1_s0.pt --engine export/m1_s0/model.bin --out m1_s0_int8 > logs_eval_int8.txt 2>&1
python3 scripts/deploy_stats.py ckpt/m1_s0.pt > logs_deploy.txt 2>&1
python3 scripts/make_tables.py > logs_tables.txt 2>&1
echo POST_DONE
