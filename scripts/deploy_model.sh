# Application model: trained on all 64 families and all 4 phrasings. It is NOT evaluated in the paper's
# tables (it has seen every test family and phrasing); the paper's numbers come from the held-out models.
set -x
while ! grep -q EXTRA2_DONE logs_extra2.txt 2>/dev/null; do sleep 30; done
python3 -c "
import sys; sys.path.insert(0,'.')
from microcoder.dataset import build
print(build('data_deploy', draws_per_family=1200, seed=0, aug=False, deploy=True))"
SIZE=${SIZE:-"--d 128 --layers 4 --heads 4"}
python3 -m microcoder.train --name deploy --data data_deploy --tok data/tok2048_final.json --dropout 0.1 --steps 4000 --seed 0 $SIZE > logs_deploy_train.txt 2>&1
python3 scripts/build_web_demo.py ckpt/deploy.pt
echo DEPLOY_DONE
