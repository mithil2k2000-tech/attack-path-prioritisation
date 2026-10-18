"""Paper 2: does attack-path awareness help prioritise cloud security findings?

We generate a corpus of synthetic cloud architectures in SentinelCloud's
architecture.json schema, run SentinelCloud's own unmodified rule engine,
risk model and attack-path simulator on each, and compare remediation
orderings against an independent probabilistic attacker model.

Strategies compared (all rank the same findings):
  severity   - the rule engine's fixed severity (CRITICAL > HIGH > MEDIUM), as a CI gate would
  risk       - SentinelCloud's risk model: impact (asset sensitivity) x likelihood (edge exposure)
  path       - attack-path aware: findings ranked by the sensitivity-weighted number of
               feasible attack chains (from attack_sim.py) they sit on, then by severity
  random     - mean of 20 random orderings (lower reference)
  oracle     - greedy: at each step fix whatever lowers expected loss most (upper reference)

Ground truth is an expected-loss model that none of the strategies use
directly (see `expected_loss`). Its parameters are assumptions, so the study
reports a sensitivity analysis over them.
"""

from __future__ import annotations

import copy
import json
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

AGENT = Path(__file__).resolve().parents[2] / "sentinelcloud" / "agent"
sys.path.insert(0, str(AGENT))

from attack_sim import find_attack_chains  # noqa: E402
from risk_model import score_findings  # noqa: E402
from threat_engine import SENSITIVE_TYPES, Architecture, run_engine  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
SENS_W = {"low": 1, "medium": 2, "high": 3, "critical": 4}


# ------------------------------------------------------------------ attacker model

@dataclass(frozen=True)
class AttackerModel:
    p_gateway_waf: float = 0.3        # initial compromise of an internet-facing gateway with a WAF
    p_gateway_nowaf: float = 0.6      # ... without a WAF
    p_public_data: float = 0.7        # direct internet access to a data store with public network access
    p_public_blob: float = 0.8        # anonymous public blob access
    p_flow_auth: float = 0.1          # moving along an authenticated flow
    p_flow_unauth: float = 0.6        # moving along an unauthenticated flow
    p_flat: float = 0.5               # lateral movement inside a flat trust boundary
    p_iam_subscription: float = 0.8   # stolen workload identity with subscription-wide scope
    p_iam_group: float = 0.5          # ... scoped to the resource group (approximated by trust boundary)
    p_iam_missing: float = 0.2        # no identity: ambient/shared credentials assumed
    unencrypted_multiplier: float = 1.5
    logging_dampening: float = 0.85   # detection shortens dwell time when logging is on


def expected_loss(data: dict, m: AttackerModel) -> float:
    """Sensitivity-weighted expected compromise of data assets.

    Compromise probabilities are propagated with a noisy-OR fixed point over
    every attack edge (flows, flat boundaries, identities). This treats paths
    as independent, a standard approximation for probabilistic attack graphs.
    """
    assets = {a["id"]: a for a in data["assets"]}
    ids = list(assets)
    idx = {a: i for i, a in enumerate(ids)}
    n = len(ids)
    damp = m.logging_dampening if data.get("logging", {}).get("enabled") else 1.0
    direct = np.zeros(n)
    for a, d in assets.items():
        if d.get("type") == "gateway" and d.get("internet_facing"):
            direct[idx[a]] = m.p_gateway_waf if d.get("waf_enabled") else m.p_gateway_nowaf
        if d.get("type") in SENSITIVE_TYPES:
            p = 0.0
            if d.get("public_network_access"):
                p = 1 - (1 - p) * (1 - m.p_public_data)
            if d.get("allow_public_blob"):
                p = 1 - (1 - p) * (1 - m.p_public_blob)
            direct[idx[a]] = max(direct[idx[a]], p)
    edges: list[tuple[int, int, float]] = []
    for f in data["flows"]:
        if assets.get(f["from"], {}).get("type") == "client":
            continue  # the client is the attacker's side, handled by the gateway's direct exposure
        p = m.p_flow_auth if f.get("authenticated") else m.p_flow_unauth
        edges.append((idx[f["from"]], idx[f["to"]], p * damp))
    sensitive = [a for a, d in assets.items() if d.get("type") in SENSITIVE_TYPES]
    for tb in data.get("trust_boundaries", []):
        members = tb.get("contains", [])
        computes = [x for x in members if assets.get(x, {}).get("type") == "compute"]
        datas = [x for x in members if x in sensitive]
        for c in computes:
            for s in datas:
                edges.append((idx[c], idx[s], m.p_flat * damp))
    boundary = {x: tb["id"] for tb in data.get("trust_boundaries", []) for x in tb.get("contains", [])}
    bindings = {b["principal"]: b for b in data.get("identity_bindings", [])}
    for c, d in assets.items():
        if d.get("type") != "compute":
            continue
        b = bindings.get(c)
        role, scope = (b or {}).get("role"), (b or {}).get("scope")
        for s in sensitive:
            if not role:
                p = m.p_iam_missing
            elif scope == "subscription":
                p = m.p_iam_subscription
            elif scope == "resource-group" and boundary.get(s) == boundary.get(c):
                p = m.p_iam_group
            else:
                continue
            edges.append((idx[c], idx[s], p * damp))
    P = direct.copy()
    for _ in range(50):
        miss = 1 - direct
        for u, v, p in edges:
            miss[v] *= 1 - P[u] * p
        new = 1 - miss
        if np.max(np.abs(new - P)) < 1e-9:
            P = new
            break
        P = new
    loss = 0.0
    for s in sensitive:
        d = assets[s]
        w = SENS_W.get(d.get("sensitivity", "low"), 1)
        if d.get("encrypted_at_rest") is False:
            w *= m.unencrypted_multiplier
        loss += w * P[idx[s]]
    return float(loss)


# ------------------------------------------------------------------ remediation

def finding_key(f) -> str:
    return f"{f.rule_id}|{'|'.join(f.asset_ids)}"


def remediate(data: dict, f) -> dict:
    """Apply the fix each rule's own mitigation text describes."""
    d = copy.deepcopy(data)
    assets = {a["id"]: a for a in d["assets"]}
    r, ids = f.rule_id, f.asset_ids
    if r in ("IAM-BROAD-01", "IAM-BROAD-02"):
        for b in d["identity_bindings"]:
            if b["principal"] == ids[0]:
                b["scope"] = "resource"
    elif r == "IAM-MISSING-01":
        d["identity_bindings"] = [b for b in d["identity_bindings"] if b["principal"] != ids[0]]
        d["identity_bindings"].append({"principal": ids[0], "role": f"{ids[0]}-scoped", "scope": "resource"})
    elif r == "NET-PUBLIC-01":
        assets[ids[0]]["public_network_access"] = False
    elif r == "NET-PUBLIC-02":
        assets[ids[0]]["allow_public_blob"] = False
    elif r == "DATA-ENC-01":
        assets[ids[0]]["encrypted_at_rest"] = True
    elif r == "NET-SEGMENT-01":
        members = set(ids)
        for tb in d["trust_boundaries"]:
            if set(tb["contains"]) == members:
                data_members = [x for x in tb["contains"] if assets[x].get("type") in SENSITIVE_TYPES]
                tb["contains"] = [x for x in tb["contains"] if x not in data_members]
                d["trust_boundaries"].append({"id": tb["id"] + "-data", "contains": data_members})
                break
    elif r == "AUTHN-FLOW-01":
        for fl in d["flows"]:
            if fl["from"] == ids[0] and fl["to"] == ids[1]:
                fl["authenticated"] = True
    elif r == "NET-WAF-01":
        assets[ids[0]]["waf_enabled"] = True
    elif r == "LOG-001":
        d["logging"] = {"enabled": True}
    return d


# ------------------------------------------------------------------ strategies

def order_severity(arch, findings, data):
    return sorted(findings, key=lambda f: -f.severity.value)  # stable: engine order breaks ties


def order_risk(arch, findings, data):
    return [s.finding for s in score_findings(findings, arch)]


def order_path(arch, findings, data):
    chains = [c for c in find_attack_chains(arch) if c.feasibility == "HIGH"]
    def chain_score(f):
        total = 0
        for c in chains:
            nodes = {c.entry} | {h["to"] for h in c.hops}
            broken = {(h["from"], h["to"]) for h in c.hops if not h["authenticated"]}
            hit = (f.rule_id == "AUTHN-FLOW-01" and tuple(f.asset_ids[:2]) in broken) or \
                  (f.rule_id != "AUTHN-FLOW-01" and bool(set(f.asset_ids) & nodes))
            if hit:
                total += SENS_W.get(c.target_sensitivity, 1)
        return total
    return sorted(findings, key=lambda f: (-chain_score(f), -f.severity.value))


def remediation_curve(data, order, m, K):
    losses = [expected_loss(data, m)]
    cur = data
    for f in order[:K]:
        cur = remediate(cur, f)
        losses.append(expected_loss(cur, m))
    return losses


def oracle_curve(data, findings, m, K):
    losses = [expected_loss(data, m)]
    cur, left = data, list(findings)
    for _ in range(min(K, len(left))):
        best = min(left, key=lambda f: expected_loss(remediate(cur, f), m))
        cur = remediate(cur, best)
        left.remove(best)
        losses.append(expected_loss(cur, m))
    return losses


# ------------------------------------------------------------------ generator

def generate(rng: random.Random, size: str) -> dict:
    n_compute = {"small": rng.randint(2, 4), "medium": rng.randint(5, 8), "large": rng.randint(9, 14)}[size]
    n_data = {"small": rng.randint(1, 3), "medium": rng.randint(3, 5), "large": rng.randint(5, 9)}[size]
    hardening = rng.random()  # 0 = careless team, 1 = disciplined team
    p = lambda base: base * (1 - hardening) + 0.02  # noqa: E731  probability of a weakness
    assets = [{"id": "mobile-app", "type": "client", "subnet": "internet", "sensitivity": "low"}]
    gateways = [f"gw{i}" for i in range(1 if size != "large" else 2)]
    for g in gateways:
        assets.append({"id": g, "type": "gateway", "subnet": "edge", "sensitivity": "medium", "internet_facing": True,
                       "waf_enabled": rng.random() > p(0.7)})
    computes = [f"svc{i}" for i in range(n_compute)]
    for c in computes:
        assets.append({"id": c, "type": "compute", "subnet": "app", "sensitivity": rng.choice(["medium", "high"])})
    datas = []
    for i in range(n_data):
        t = rng.choices(["database", "storage", "secret-store"], weights=[5, 3, 2])[0]
        a = {"id": f"{t}{i}", "type": t, "subnet": "data",
             "sensitivity": rng.choices(["medium", "high", "critical"], weights=[2, 4, 3])[0],
             "public_network_access": rng.random() < p(0.35), "encrypted_at_rest": rng.random() > p(0.4)}
        if t == "storage":
            a["allow_public_blob"] = rng.random() < p(0.3)
        assets.append(a)
        datas.append(a["id"])
    flows = [{"from": "mobile-app", "to": g, "protocol": "https", "port": 443, "authenticated": True} for g in gateways]
    for c in computes:  # every service is behind some gateway or another service
        src = rng.choice(gateways + [x for x in computes if x < c]) if rng.random() < 0.5 else rng.choice(gateways)
        if src != c:
            flows.append({"from": src, "to": c, "protocol": "https", "port": 443, "authenticated": rng.random() > p(0.6)})
    for dd in datas:
        for c in rng.sample(computes, k=min(len(computes), rng.randint(1, 3))):
            flows.append({"from": c, "to": dd, "protocol": "tcp", "port": 5432, "authenticated": rng.random() > p(0.6)})
    if rng.random() < p(0.6):  # one flat zone holding compute and data
        tbs = [{"id": "edge", "contains": gateways}, {"id": "app-data", "contains": computes + datas}]
    else:
        tbs = [{"id": "edge", "contains": gateways}, {"id": "app", "contains": computes}, {"id": "data", "contains": datas}]
    bindings = []
    for c in computes:
        r = rng.random()
        if r < p(0.25):
            continue  # no identity
        scope = "subscription" if rng.random() < p(0.3) else ("resource-group" if rng.random() < p(0.5) else "resource")
        bindings.append({"principal": c, "role": "Contributor" if scope == "subscription" else f"{c}-role", "scope": scope})
    return {"name": f"synthetic-{size}", "assets": assets, "trust_boundaries": tbs, "flows": flows,
            "identity_bindings": bindings, "logging": {"enabled": rng.random() > p(0.4)}, "_hardening": hardening}


# ------------------------------------------------------------------ run

STRATEGIES = {"severity": order_severity, "risk": order_risk, "path": order_path}


def evaluate(data: dict, m: AttackerModel, K: int = 5, rng: random.Random | None = None) -> dict | None:
    arch = Architecture(data)
    findings = run_engine(arch)
    if len(findings) < 3:
        return None
    L0 = expected_loss(data, m)
    full = data
    for f in findings:
        full = remediate(full, f)
    Lall = expected_loss(full, m)
    if L0 - Lall < 1e-9:
        return None
    k = min(K, len(findings))
    out = {"n_findings": len(findings), "L0": L0, "Lall": Lall, "K": k}
    curves = {name: remediation_curve(data, fn(arch, findings, data), m, k) for name, fn in STRATEGIES.items()}
    curves["oracle"] = oracle_curve(data, findings, m, k)
    rnd = []
    rng = rng or random.Random(0)
    for _ in range(20):
        perm = findings[:]
        rng.shuffle(perm)
        rnd.append(remediation_curve(data, perm, m, k))
    curves["random"] = list(np.mean(rnd, axis=0))
    for name, c in curves.items():
        red = [(L0 - x) / (L0 - Lall) for x in c]
        out[f"{name}_at1"] = red[1]
        out[f"{name}_at3"] = red[min(3, k)]
        out[f"{name}_auc"] = float(np.mean(red[1:]))
    return out


def main():
    rng = random.Random(2026)
    corpus = []
    for size in ("small", "medium", "large"):
        while sum(1 for c in corpus if c["name"].endswith(size)) < 200:
            d = generate(rng, size)
            if len(run_engine(Architecture(d))) >= 3:
                corpus.append(d)
    (RESULTS / "corpus.json").write_text(json.dumps(corpus))
    base = AttackerModel()
    rows = []
    for i, d in enumerate(corpus):
        r = evaluate(d, base, rng=random.Random(i))
        if r:
            rows.append({"arch": i, "size": d["name"].split("-")[1], "hardening": d["_hardening"], **r})
    pd.DataFrame(rows).to_csv(RESULTS / "main.csv", index=False)
    print("main done", len(rows))
    # Sensitivity analysis: each parameter halved and raised by 50% (capped at 0.95), on a 150-architecture subsample.
    sub = random.Random(1).sample(range(len(corpus)), 150)
    srows = []
    for field in ["p_flow_auth", "p_flow_unauth", "p_flat", "p_iam_subscription", "p_iam_missing", "p_public_data",
                  "p_gateway_nowaf", "unencrypted_multiplier"]:
        for factor in (0.5, 1.5):
            val = getattr(base, field) * factor
            if field != "unencrypted_multiplier":
                val = min(val, 0.95)
            m = replace(base, **{field: val})
            for i in sub:
                r = evaluate(corpus[i], m, rng=random.Random(i))
                if r:
                    srows.append({"param": field, "factor": factor, "arch": i, **r})
        print("sens", field)
    pd.DataFrame(srows).to_csv(RESULTS / "sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
