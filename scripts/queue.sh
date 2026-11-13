# Sequential experiment queue (one job at a time uses both cores best).
T="--tok data/tok2048_v1.json --dropout 0.1"
while pgrep -f "name m1_s0 " >/dev/null; do sleep 20; done
run() { echo "== $*"; "$@"; }
run python3 -W ignore -m microcoder.evaluate ckpt/m1_s0.pt > logs_eval_m1_s0.txt 2>&1
run python3 -m microcoder.train --name m1g_s0 $T --steps 4000 --seed 0 --mix gen > logs_m1g_s0.txt 2>&1
run python3 -W ignore -m microcoder.evaluate ckpt/m1g_s0.pt > logs_eval_m1g_s0.txt 2>&1
run python3 -m microcoder.train --name m1_s1 $T --steps 4000 --seed 1 > logs_m1_s1.txt 2>&1
run python3 -W ignore -m microcoder.evaluate ckpt/m1_s1.pt > logs_eval_m1_s1.txt 2>&1
run python3 -m microcoder.train --name m1_s2 $T --steps 4000 --seed 2 > logs_m1_s2.txt 2>&1
run python3 -W ignore -m microcoder.evaluate ckpt/m1_s2.pt > logs_eval_m1_s2.txt 2>&1
run python3 -m microcoder.train --name s_s0 $T --d 96 --layers 3 --heads 4 --steps 4000 --seed 0 > logs_s_s0.txt 2>&1
run python3 -W ignore -m microcoder.evaluate ckpt/s_s0.pt > logs_eval_s_s0.txt 2>&1
echo QUEUE_DONE
