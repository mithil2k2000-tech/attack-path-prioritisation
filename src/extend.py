"""Extensions: (1) exposure-corrected risk model, (2) model-guided greedy ordering under a misspecified attacker model."""
import json, random
from dataclasses import fields, replace
import numpy as np, pandas as pd
from study import *  # noqa: F403  (also puts SentinelCloud's agent/ on sys.path)
from risk_model import impact_score

corpus = json.loads((RESULTS / "corpus.json").read_text())
base = AttackerModel()


def exposure_fixed(asset_id, arch):
    """Original _exposure plus the two cases it misses: data stores reachable
    straight from the internet, and assets sharing a flat zone with an exposed workload."""
    a = arch.assets.get(asset_id, {})
    if a.get("internet_facing") or a.get("public_network_access") or a.get("allow_public_blob"):
        return 3
    for f in arch.flows_into(asset_id):
        if arch.assets.get(f["from"], {}).get("internet_facing"):
            return 3 if not f.get("authenticated") else 2
    return 1


def order_risk_fixed(arch, findings, data):
    def score(f):
        lik = max((exposure_fixed(a, arch) for a in f.asset_ids), default=1)
        return impact_score(f, arch) * lik
    return sorted(findings, key=lambda f: -score(f))


def greedy_with(model):
    def order(arch, findings, data):
        cur, left, out = data, list(findings), []
        while left:
            best = min(left, key=lambda f: expected_loss(remediate(cur, f), model))
            out.append(best); left.remove(best); cur = remediate(cur, best)
            if len(out) >= 5:
                out += left; break
        return out
    return order


def perturbed(rng):
    kw = {}
    for f in fields(AttackerModel):
        v = getattr(base, f.name) * rng.uniform(0.5, 1.5)
        kw[f.name] = v if f.name in ("unencrypted_multiplier",) else min(v, 0.95)
    return AttackerModel(**kw)


rows = []
for i, d in enumerate(corpus):
    arch = Architecture(d); fs = run_engine(arch)
    L0 = expected_loss(d, base)
    full = d
    for f in fs: full = remediate(full, f)
    Lall = expected_loss(full, base)
    if L0 - Lall < 1e-9: continue
    k = min(5, len(fs))
    rec = {"arch": i, "size": d["name"].split("-")[1]}
    strategies = {"risk_fixed": order_risk_fixed(arch, fs, d)}
    rng = random.Random(1000 + i)
    for j in range(3):
        strategies[f"greedy_mis{j}"] = greedy_with(perturbed(rng))(arch, fs, d)
    for name, order in strategies.items():
        c = remediation_curve(d, order, base, k)
        red = [(L0 - x) / (L0 - Lall) for x in c]
        rec[f"{name}_at1"], rec[f"{name}_at3"], rec[f"{name}_auc"] = red[1], red[min(3, k)], float(np.mean(red[1:]))
    rows.append(rec)
df = pd.DataFrame(rows)
for m in ("at1", "at3", "auc"):
    df[f"greedy_mis_{m}"] = df[[f"greedy_mis{j}_{m}" for j in range(3)]].mean(axis=1)
df.to_csv(RESULTS / "extensions.csv", index=False)
print(df[[c for c in df.columns if c.startswith(("risk_fixed", "greedy_mis_"))]].mean().round(3).to_string())
