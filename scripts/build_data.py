"""Generate data/ (train corpus + leak-free test splits) with the final, dev-selected recipe."""
import json, sys
sys.path.insert(0, ".")
from microcoder.dataset import build, build_name_repairs
meta = build("data", draws_per_family=1200, n_id_per_family=8, n_ood_per_family=15, seed=0, aug=False)
print(json.dumps({k: v for k, v in meta.items() if not k.endswith("families")}, indent=1))
print("held-out families:", meta["heldout_families"])
print("undefined-name repair triples:", build_name_repairs("data"))
