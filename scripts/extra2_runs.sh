# Two more seeds of the small (0.53M) model, which matched the 1.05M model on seed 0.
set -x
while ! grep -q EXTRA_DONE logs_extra.txt 2>/dev/null; do sleep 30; done
T="--tok data/tok2048_final.json --dropout 0.1 --d 96 --layers 3 --heads 4 --steps 4000"
for s in 1 2; do
  python3 -m microcoder.train --name s_s$s $T --seed $s > logs_s_s$s.txt 2>&1
  python3 -W ignore -m microcoder.evaluate ckpt/s_s$s.pt > logs_eval_s_s$s.txt 2>&1
done
python3 -W ignore -m microcoder.evaluate ckpt/v0_m1_s0.pt --methods a_resample_lint,a_ref_lint --out v0_lint > /dev/null 2>&1 && python3 scripts/merge_summary.py v0_m1_s0 v0_lint
python3 -W ignore -m microcoder.evaluate ckpt/v1_m1_s0.pt --methods a_resample_lint,a_ref_lint --out v1_lint > /dev/null 2>&1 && python3 scripts/merge_summary.py v1_m1_s0 v1_lint
python3 scripts/make_tables.py > logs_tables.txt 2>&1
echo EXTRA2_DONE
