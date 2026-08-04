#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NGSIM_DATA_PATH="${1:-${PROJECT_DIR}/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv}"
NGSIM_OUTPUT_DIR="${2:-${PROJECT_DIR}/result/ngsim}"
python_bin="${PYTHON_BIN:-python}"
workers="${WORKERS:-10}"

cd "${PROJECT_DIR}"

"${python_bin}" ngsim_experiment.py \
  --data-path "${NGSIM_DATA_PATH}" \
  --split-manifest manifests/ngsim_split.json \
  --methods fa_sal salnx tebbe_abm \
  --rounds 150 \
  --trials 10 \
  --parallel-workers "${workers}" \
  --horizon 4 \
  --fa-sal-horizon 4 \
  --salnx-horizon 4 \
  --tebbe-horizon 4 \
  --action-values -4 -2 0 1 2 \
  --min-headway 8 \
  --safety-headway-offset 0.5 \
  --dt 0.1 \
  --frame-stride 5 \
  --n-init 80 \
  --n-eval 100 \
  --safe-start-headway-max 14.5 \
  --safe-start-rel-speed-max 0 \
  --safe-start-min-count 20 \
  --rmse-eval-headway-max 14.5 \
  --rmse-eval-rel-speed-max 0.1 \
  --rmse-eval-min-count 20 \
  --collision-absorbing \
  --collision-headway 0 \
  --collision-safety-penalty -10 \
  --fa-beam-width 1 \
  --fa-sal-l-ell-quantile-sweep 0.9 \
  --fa-sal-l-ell-scale-sweep 1e-10 \
  --salnx-alpha-sweep 0.2 \
  --salnx-mc-samples 128 \
  --salnx-uncertainty-criterion logdet \
  --tebbe-alpha 0.2 \
  --tebbe-confidence-delta 0.01 \
  --tebbe-sample-start 32 \
  --tebbe-sample-stages 6 \
  --tebbe-uncertainty-criterion logdet \
  --save-dir "${NGSIM_OUTPUT_DIR}"
