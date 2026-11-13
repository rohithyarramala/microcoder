# Resume after the sandbox restart. Order = importance for the paper.
set -x
T="--tok data/tok2048_final.json --dropout 0.1"
ev() { python3 -W ignore -m microcoder.evaluate ckpt/$1.pt > logs_eval_$1.txt 2>&1; }
gcc -O2 -shared -fPIC -o esp32/engine/libmicrocoder.so esp32/engine/microcoder.c esp32/engine/host_api.c -lm
# 1. deployment + INT8 + MBPP
python3 -m microcoder.export ckpt/m1_s0.pt export/m1_s0 > logs_export.txt 2>&1
python3 scripts/test_engine.py ckpt/m1_s0.pt export/m1_s0 > logs_test_engine.txt 2>&1
python3 scripts/deploy_stats.py ckpt/m1_s0.pt > logs_deploy.txt 2>&1
python3 -W ignore -m microcoder.evaluate ckpt/m1_s0.pt --splits mbpp --methods greedy,anchor,a_resample,a_ref,a_ref_lint --out m1_s0_mbpp > logs_eval_mbpp.txt 2>&1
python3 -W ignore -m microcoder.evaluate ckpt/m1_s0.pt --engine export/m1_s0/model.bin --out m1_s0_int8 > logs_eval_int8.txt 2>&1
python3 scripts/make_tables.py > logs_tables.txt 2>&1
echo STAGE1_DONE
# 2. small model seeds + lint evals of the first two attempts
for s in 1 2; do python3 -m microcoder.train --name s_s$s $T --d 96 --layers 3 --heads 4 --steps 4000 --seed $s > logs_s_s$s.txt 2>&1; ev s_s$s; done
python3 -W ignore -m microcoder.evaluate ckpt/v0_m1_s0.pt --methods a_resample_lint,a_ref_lint --out v0_lint > /dev/null 2>&1 && python3 scripts/merge_summary.py v0_m1_s0 v0_lint
python3 -W ignore -m microcoder.evaluate ckpt/v1_m1_s0.pt --methods a_resample_lint,a_ref_lint --out v1_lint > /dev/null 2>&1 && python3 scripts/merge_summary.py v1_m1_s0 v1_lint
python3 scripts/make_tables.py > logs_tables.txt 2>&1
echo STAGE2_DONE
# 3. undefined-name repair variant
python3 -m microcoder.train --name m1n_s0 $T --steps 4000 --seed 0 --mix gen+repair+name > logs_m1n_s0.txt 2>&1; ev m1n_s0
python3 scripts/make_tables.py > logs_tables.txt 2>&1
echo STAGE3_DONE
# 4. application model (all families) + web playground
python3 -c "
import sys; sys.path.insert(0,'.')
from microcoder.dataset import build
print(build('data_deploy', draws_per_family=1200, seed=0, aug=False, deploy=True))"
python3 -m microcoder.train --name deploy --data data_deploy $T --steps 4000 --seed 0 > logs_deploy_train.txt 2>&1
python3 scripts/build_web_demo.py ckpt/deploy.pt
echo STAGE4_DONE
# 5. large model
python3 -m microcoder.train --name l_s0 $T --d 192 --layers 6 --heads 6 --steps 3000 --seed 0 > logs_l_s0.txt 2>&1; ev l_s0
python3 scripts/make_tables.py > logs_tables.txt 2>&1
echo ALL_DONE
