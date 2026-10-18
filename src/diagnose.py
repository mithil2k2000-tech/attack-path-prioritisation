"""Why do strategies differ? Which rule types each fixes first, and the marginal value of each rule type."""
import json, random
from collections import Counter
import numpy as np, pandas as pd
from study import *

corpus = json.loads((RESULTS / "corpus.json").read_text())
m = AttackerModel()
first = {k: Counter() for k in ["severity", "risk", "path", "oracle"]}
value = {}
for i, d in enumerate(corpus):
    arch = Architecture(d); fs = run_engine(arch)
    L0 = expected_loss(d, m)
    full = d
    for f in fs: full = remediate(full, f)
    span = L0 - expected_loss(full, m)
    if span <= 1e-9: continue
    for name, fn in STRATEGIES.items():
        first[name][fn(arch, fs, d)[0].rule_id] += 1
    best = min(fs, key=lambda f: expected_loss(remediate(d, f), m))
    first["oracle"][best.rule_id] += 1
    for f in fs:
        value.setdefault(f.rule_id, []).append((L0 - expected_loss(remediate(d, f), m)) / span)
rows = []
for r, v in sorted(value.items(), key=lambda kv: -np.mean(kv[1])):
    rows.append({"rule": r, "n": len(v), "mean_single_fix_share": round(float(np.mean(v)), 3),
                 **{f"first_{k}": first[k][r] for k in first}})
df = pd.DataFrame(rows); print(df.to_string(index=False)); df.to_csv(RESULTS / "rule_value.csv", index=False)
