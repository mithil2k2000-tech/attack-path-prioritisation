"""Paper 2 tables and figures from results/*.csv."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R, FIG, TAB = ROOT / "results", ROOT / "paper" / "figures", ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})

main = pd.read_csv(R / "main.csv")
ext = pd.read_csv(R / "extensions.csv")
df = main.merge(ext.drop(columns=["size"]), on="arch")
LABEL = {"random": "Random order", "risk": "Risk model (original)", "path": "Attack-path aware", "severity": "Rule severity",
         "risk_fixed": "Risk model (exposure-corrected)", "greedy_mis": "Model-guided, $\\pm$50\\% misspecified",
         "oracle": "Model-guided, true parameters"}
ORDER = list(LABEL)
facts = {}


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


rows = []
for s in ORDER:
    r = {"Strategy": LABEL[s]}
    for m, lab in (("at1", "After 1 fix"), ("at3", "After 3 fixes"), ("auc", "Mean over 1--5")):
        mu, h = ci(df[f"{s}_{m}"])
        r[lab] = f"{mu:.3f} $\\pm$ {h:.3f}"
    if s != "severity":
        diff = df[f"{s}_auc"] - df["severity_auc"]
        w = stats.wilcoxon(df[f"{s}_auc"], df["severity_auc"])
        r["vs.~severity"] = f"{diff.mean():+.3f} ({'$p<0.001$' if w.pvalue < 0.001 else f'$p={w.pvalue:.3f}$'})"
        facts[f"{s}_vs_severity"] = {"diff": diff.mean(), "p": w.pvalue, "better_share": float((diff > 0).mean())}
    else:
        r["vs.~severity"] = "--"
    rows.append(r)
t = pd.DataFrame(rows)
(TAB / "main.tex").write_text(
    "\\begin{table*}[t]\n\\centering\n\\caption{Share of the achievable loss reduction realised by each prioritisation "
    f"strategy across {len(df)} synthetic architectures (mean $\\pm$ 95\\% CI). The last column is the mean paired "
    "difference in the 1--5 mean against rule severity, with a two-sided Wilcoxon signed-rank test.}\n"
    "\\label{tab:main}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n" + t.to_latex(index=False, escape=False, column_format="lrrrr") + "\\end{table*}\n")
facts["n"] = len(df)
facts["means"] = {s: {m: float(df[f"{s}_{m}"].mean()) for m in ("at1", "at3", "auc")} for s in ORDER}

# Figure 1: mean over 1-5 fixes by architecture size.
fig, ax = plt.subplots(figsize=(6.8, 2.4))
sizes = ["small", "medium", "large"]
w = 0.11
colors = ["#bdbdbd", "#e6a5a1", "#9ecae1", "#3182bd", "#c0504d", "#31a354", "#006d2c"]
for i, s in enumerate(ORDER):
    vals = [df[df["size"] == z][f"{s}_auc"].mean() for z in sizes]
    errs = [ci(df[df["size"] == z][f"{s}_auc"])[1] for z in sizes]
    ax.bar(np.arange(3) + (i - 3) * w, vals, w, yerr=errs, color=colors[i], label=LABEL[s].replace("$\\pm$", "±").replace("\\%", "%"),
           capsize=1.5, error_kw={"lw": 0.6})
ax.set_xticks(range(3))
ax.set_xticklabels([f"{z} (n={int((df['size'] == z).sum())})" for z in sizes])
ax.set_ylabel("Loss reduction share\n(mean over 1–5 fixes)")
ax.set_ylim(0, 1)
ax.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False)
fig.tight_layout()
fig.savefig(FIG / "by_size.pdf", bbox_inches="tight")
plt.close(fig)

# Rule value table.
rv = pd.read_csv(R / "rule_value.csv")
rv_t = pd.DataFrame({"Rule": rv.rule, "Findings": rv.n, "Mean value of fixing alone": rv.mean_single_fix_share.map(lambda v: f"{v:.3f}"),
                     "Sev.": rv.first_severity, "Risk": rv.first_risk, "Path": rv.first_path, "Oracle": rv.first_oracle})
(TAB / "rules.tex").write_text(
    "\\begin{table}[t]\n\\centering\n\\caption{Value of fixing each rule's findings on their own (share of the achievable "
    "loss reduction, averaged over findings), and how often each strategy chooses that rule as its first fix.}\n"
    "\\label{tab:rules}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n" + rv_t.to_latex(index=False, escape=False, column_format="lrrrrrr") + "\\end{table}\n")
facts["rules"] = rv.to_dict("records")

# Sensitivity: does the ranking of the three practical strategies survive parameter changes?
sens = pd.read_csv(R / "sensitivity.csv")
srows = []
for (p, f), g in sens.groupby(["param", "factor"]):
    srows.append({"param": p, "factor": f, **{s: g[f"{s}_auc"].mean() for s in ("severity", "risk", "path", "oracle", "random")}})
s = pd.DataFrame(srows)
s["severity_beats_path"] = s.severity > s.path
s["severity_beats_risk"] = s.severity > s.risk
facts["sensitivity"] = {"settings": len(s), "severity_beats_path": int(s.severity_beats_path.sum()),
                        "severity_beats_risk": int(s.severity_beats_risk.sum()),
                        "oracle_min": float(s.oracle.min()), "severity_range": [float(s.severity.min()), float(s.severity.max())],
                        "path_range": [float(s.path.min()), float(s.path.max())], "risk_range": [float(s.risk.min()), float(s.risk.max())]}
nice = {"p_flow_auth": "authenticated hop", "p_flow_unauth": "unauthenticated hop", "p_flat": "flat-zone lateral move",
        "p_iam_subscription": "subscription-wide identity", "p_iam_missing": "missing identity",
        "p_public_data": "public data store", "p_gateway_nowaf": "gateway without WAF", "unencrypted_multiplier": "unencrypted impact"}
st = s.assign(Parameter=s.param.map(nice), Change=s.factor.map({0.5: "$\\times$0.5", 1.5: "$\\times$1.5"}))
st = st[["Parameter", "Change", "severity", "risk", "path", "oracle"]].rename(
    columns={"severity": "Severity", "risk": "Risk", "path": "Path", "oracle": "Oracle"})
for c in ("Severity", "Risk", "Path", "Oracle"):
    st[c] = st[c].map(lambda v: f"{v:.3f}")
(TAB / "sensitivity.tex").write_text(
    "\\begin{table}[t]\n\\centering\n\\caption{Sensitivity analysis: mean loss-reduction share over 1--5 fixes when one "
    "attacker-model parameter is halved or raised by 50\\% (150 architectures each).}\n\\label{tab:sens}\n\\small\n"
    "\\setlength{\\tabcolsep}{4pt}\n" + st.to_latex(index=False, escape=False, column_format="llrrrr") + "\\end{table}\n")

# Corpus description.
corpus = json.loads((R / "corpus.json").read_text())
facts["corpus"] = {"n": len(corpus), "findings_mean": float(main.n_findings.mean()), "findings_min": int(main.n_findings.min()),
                   "findings_max": int(main.n_findings.max()),
                   "assets_mean": float(np.mean([len(c["assets"]) for c in corpus])),
                   "flows_mean": float(np.mean([len(c["flows"]) for c in corpus]))}
(R / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
print(t.to_string(index=False))
print(json.dumps(facts["sensitivity"], indent=1), json.dumps(facts["corpus"], indent=1))
