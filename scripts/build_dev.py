"""Model-selection datasets: train phrasings {0,1}; dev = phrasing 2 of training families (never test data)."""
import sys
sys.path.insert(0, ".")
from microcoder.dataset import build
for name, kw in [("p0", dict(name_p=0.0)), ("p0.15", dict(name_p=0.15)), ("p0.5", dict(name_p=0.5)), ("noaug", dict(aug=False))]:
    m = build(f"data_dev_{name}", draws_per_family=1200, n_id_per_family=8, n_ood_per_family=15, seed=0,
              train_phrasings=(0, 1), n_dev_per_family=6, **kw)
    print(name, m["n_train_gen"], m["n_train_repair"])
