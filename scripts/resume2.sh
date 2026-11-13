set -x
T="--tok data/tok2048_final.json --dropout 0.1"
ev() { python3 -W ignore -m microcoder.evaluate ckpt/$1.pt > logs_eval_$1.txt 2>&1; }
# Deployment numbers for the small model too (it matched/beat M over 3 seeds)
python3 -m microcoder.export ckpt/s_s0.pt export/s_s0 > /dev/null 2>&1
python3 -W ignore -m microcoder.evaluate ckpt/s_s0.pt --engine export/s_s0/model.bin --out s_s0_int8 > logs_eval_s_int8.txt 2>&1
mkdir -p results/deploy_small && python3 scripts/deploy_stats.py ckpt/s_s0.pt > logs_deploy_small.txt 2>&1 && cp results/deploy.json results/deploy_small.json && cp results/host_sim_log.txt results/host_sim_log_small.txt
python3 scripts/deploy_stats.py ckpt/m1_s0.pt > logs_deploy.txt 2>&1
# Application model: small size, all families
python3 -c "
import sys; sys.path.insert(0,'.')
from microcoder.dataset import build
print(build('data_deploy', draws_per_family=1200, seed=0, aug=False, deploy=True))"
python3 -m microcoder.train --name deploy --data data_deploy $T --d 96 --layers 3 --heads 4 --steps 4000 --seed 0 > logs_deploy_train.txt 2>&1
python3 scripts/build_web_demo.py ckpt/deploy.pt
python3 scripts/make_tables.py > logs_tables.txt 2>&1
echo STAGE_A_DONE
python3 -m microcoder.train --name m1n_s0 $T --steps 4000 --seed 0 --mix gen+repair+name > logs_m1n_s0.txt 2>&1; ev m1n_s0
python3 scripts/make_tables.py > logs_tables.txt 2>&1
echo STAGE_B_DONE
python3 -m microcoder.train --name l_s0 $T --d 192 --layers 6 --heads 6 --steps 3000 --seed 0 > logs_l_s0.txt 2>&1; ev l_s0
python3 scripts/make_tables.py > logs_tables.txt 2>&1
echo ALL_DONE
