#!/usr/bin/env bash
# Rebuilds every number, table and figure in the paper (about 40 minutes on a laptop).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d ../sentinelcloud ]; then
  git clone https://github.com/mithil2k2000-tech/sentinelcloud.git ../sentinelcloud
fi
git -C ../sentinelcloud checkout --quiet 61c82cf   # the exact engine version used in the paper
python3 src/study.py       # corpus, main comparison, sensitivity analysis
python3 src/diagnose.py    # per-rule value
python3 src/extend.py      # exposure-corrected risk model, misspecified greedy
python3 src/analyse.py     # tables, figure, facts.json
(cd paper && latexmk -pdf -quiet main.tex)
