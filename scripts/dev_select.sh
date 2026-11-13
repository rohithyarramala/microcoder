# Model selection on the DEV split only (train phrasings {0,1}, dev = phrasing 2 of training families).
T="--tok data/tok2048_v1.json --dropout 0.1 --steps 2500 --seed 0"
for P in 0 0.15 0.5; do
  python3 -m microcoder.train --name dev_p$P --data data_dev_p$P $T > logs_dev_p$P.txt 2>&1
  python3 -W ignore -m microcoder.evaluate ckpt/dev_p$P.pt --data data_dev_p$P --splits test_dev --methods greedy,anchor,a_ref > logs_eval_dev_p$P.txt 2>&1
done
echo DEV_DONE
