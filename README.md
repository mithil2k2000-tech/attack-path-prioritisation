# Does attack-path awareness help prioritise cloud security findings?

📄 **Paper (PDF): [attack-path-prioritisation.pdf](attack-path-prioritisation.pdf)** · [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23593004.svg)](https://doi.org/10.5281/zenodo.23593004)

Code, data and paper for an empirical study of how to order design-level security findings from a
threat-modelling-as-code tool.

## The question

A threat-modelling tool flags ten findings on your architecture. You can fix three this sprint.
Which three? We compare the orderings tools actually offer — fixed rule severity, a risk score and
attack-path counting — on the same findings from the unmodified [SentinelCloud](https://github.com/mithil2k2000-tech/sentinelcloud)
engine, across 600 synthetic architectures, against an independent noisy-OR attacker model.

## Results

Share of the achievable expected-loss reduction, averaged over the first five fixes (600 architectures):

| Strategy | Mean over 1–5 fixes | vs. severity |
|---|---|---|
| Random order | 0.437 | −0.188 |
| Risk model (original) | 0.475 | −0.151 |
| Attack-path aware | 0.575 | −0.050 |
| **Rule severity** | **0.625** | — |
| Risk model (exposure-corrected) | 0.726 | +0.101 |
| Model-guided, ±50% misspecified | 0.795 | +0.170 |
| Model-guided, true parameters (oracle) | 0.811 | +0.186 |

All differences against severity: paired Wilcoxon p < 0.001.

* Plain severity beat both the risk score and attack-path ordering, in all 16 sensitivity settings.
* The risk model never closed public network access on a data store first, because its likelihood term
  only counts exposure through flows from internet-facing assets. Fixing that one blind spot lifted it above severity.
* A greedy ordering driven by an attacker model with every parameter off by up to ±50% was within 0.02 of the oracle.

These results come from a synthetic corpus and a model-based ground truth; see the paper's limitations section.

## Layout

```
src/study.py      generator, attacker model, remediation operators, strategies, main run, sensitivity analysis
src/diagnose.py   value of fixing each rule alone, and which rule each strategy picks first
src/extend.py     exposure-corrected risk model; greedy ordering under a misspecified model
src/analyse.py    statistics, tables and figure for the paper
results/          corpus.json (600 architectures) and every result file used in the paper
paper/            LaTeX source and compiled PDF
```

## Reproduce

```bash
pip install -r requirements.txt
./reproduce.sh
```

`reproduce.sh` clones SentinelCloud next to this repository and pins it to commit `61c82cf`, the version used in the paper.

## AI assistance

An AI assistant (Claude, Anthropic) helped write experiment code and draft text. The author designed the
study, reviewed the code and results, and is responsible for the content.

## Licence

MIT
