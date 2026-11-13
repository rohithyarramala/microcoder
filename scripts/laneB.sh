set -x
T="--tok data/tok2048_v1.json --dropout 0.1"
python3 -m microcoder.train --name m1_s1 $T --steps 4000 --seed 1 > logs_m1_s1.txt 2>&1
python3 -m microcoder.train --name m1g_s0 $T --steps 4000 --seed 0 --mix gen > logs_m1g_s0.txt 2>&1
python3 -m microcoder.train --name s_s0 $T --d 96 --layers 3 --heads 4 --steps 4000 --seed 0 > logs_s_s0.txt 2>&1
for m in m1_s1 m1g_s0 s_s0; do python3 -W ignore -m microcoder.evaluate ckpt/$m.pt > logs_eval_$m.txt 2>&1; done
echo LANE_B_DONE
