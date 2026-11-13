# Final experiments with the recipe chosen on the dev split: NO instruction augmentation, dropout 0.1
# (dev SA+REF pass: noaug 79.5 > openers-only 70.8 > names p=0.15 47.6 > names p=0.5 37.2).
set -x
python3 -c "
import sys; sys.path.insert(0,'.')
from microcoder.dataset import build
print(build('data', draws_per_family=1200, n_id_per_family=8, n_ood_per_family=15, seed=0, aug=False))"
python3 scripts/check_leakage.py > logs_leak.txt 2>&1
python3 -c "
import sys,json; sys.path.insert(0,'.')
from microcoder.tok import train_tokenizer, corpus_texts
L=lambda p:[json.loads(x) for x in open(p)]
train_tokenizer(corpus_texts(L('data/train_gen.jsonl'),L('data/train_repair.jsonl')),2048,'data/tok2048_final.json')"
T="--tok data/tok2048_final.json --dropout 0.1"
ev() { python3 -W ignore -m microcoder.evaluate ckpt/$1.pt > logs_eval_$1.txt 2>&1; }
python3 -m microcoder.train --name m1_s0 $T --steps 4000 --seed 0 > logs_m1_s0.txt 2>&1; ev m1_s0
python3 -m microcoder.train --name m1g_s0 $T --steps 4000 --seed 0 --mix gen > logs_m1g_s0.txt 2>&1; ev m1g_s0
python3 -m microcoder.train --name m1_s1 $T --steps 4000 --seed 1 > logs_m1_s1.txt 2>&1; ev m1_s1
python3 -m microcoder.train --name m1_s2 $T --steps 4000 --seed 2 > logs_m1_s2.txt 2>&1; ev m1_s2
python3 -m microcoder.train --name s_s0 $T --d 96 --layers 3 --heads 4 --steps 4000 --seed 0 > logs_s_s0.txt 2>&1; ev s_s0
python3 scripts/make_selfrepair.py ckpt/m1_s0.pt 2 > logs_selfrepair.txt 2>&1
python3 -m microcoder.train --name m1sr_s0 $T --mix gen+repair+self --init ckpt/m1_s0.pt --steps 800 --lr 1e-3 --seed 0 > logs_m1sr_s0.txt 2>&1; ev m1sr_s0
python3 -m microcoder.train --name l_s0 $T --d 192 --layers 6 --heads 6 --steps 3000 --seed 0 > logs_l_s0.txt 2>&1; ev l_s0
echo FINAL_DONE
