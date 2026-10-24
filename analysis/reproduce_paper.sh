#!/usr/bin/env bash
# Regenerate the paper's Figs. 3-5 and Tab. 2 from results/rspdbench_aggregate.csv.
# Run from the repository root: bash analysis/reproduce_paper.sh
set -euo pipefail
mkdir -p figures
python analysis/figures/plot_degradation_fingerprint_heatmap.py --severity 3 --group-compounds \
  --output figures/fig3_fingerprint_s3.png --normalized-output figures/fig3_fingerprint_s3_normalized.png
python analysis/figures/plot_compound_vs_primitive_excess.py --mode max --plot-style main --color-by mechanism \
  --max-rows-per-family 10 --output figures/fig4_compound_excess.png --values-output figures/fig4_compound_excess.csv
python analysis/figures/plot_key_severity_curves.py \
  --output figures/fig5_c1_amplify_saturate.png --values-output figures/fig5_c1_amplify_saturate.csv
python analysis/table2.py
