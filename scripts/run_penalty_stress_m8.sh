#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

python_bin="${PYTHON_BIN:-python}"
workers="${WORKERS:-12}"
output_dir="${OUTPUT_DIR:-result/penalty_stress_m8}"
log_path="${output_dir}/experiment.log"
table_path="${output_dir}/table.md"

mkdir -p "${output_dir}"

OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
"${python_bin}" main.py \
  --environment-preset penalty_stress \
  --result-dir "${output_dir}" \
  --methods fa_sal salnx tebbe_abm \
  --rounds 150 \
  --trials 10 \
  --seed-offset 0 \
  --e_p 250 \
  --r_i 5 \
  --evaluation-horizon 8 \
  --fa-horizon 8 \
  --salnx-horizon 8 \
  --tebbe-horizon 8 \
  --fa-continuation-policy-sweep uncertainty-max random-safe greedy-margin \
  --fa-beta-multiplier 1.0 \
  --fa-lf-estimator legacy_axis_quantile \
  --fa-lf-quantile 0.5 \
  --fa-lf-scale-sweep 1.0 \
  --fa-l-ell-quantile 0.9 \
  --fa-l-ell-scale-sweep 0.001 \
  --salnx-alpha-sweep 0.01 \
  --tebbe-alpha-sweep 0.01 \
  --tebbe-confidence-delta-sweep 0.01 \
  --tebbe-endpoint-candidates 11 \
  --no-tebbe-local-refinement \
  --tebbe-sample-start 100 \
  --tebbe-sample-stages 10 \
  --paired-seeds \
  --parallel-workers "${workers}" \
  --max-tasks-per-child 1 \
  2>&1 | tee "${log_path}"

"${python_bin}" scripts/build_penalty_stress_m8_table.py \
  --result-dir "${output_dir}" \
  --output "${table_path}"
