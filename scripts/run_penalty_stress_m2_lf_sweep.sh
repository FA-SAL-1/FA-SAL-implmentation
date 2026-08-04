#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

python_bin="${PYTHON_BIN:-python}"
workers="${WORKERS:-12}"
output_dir="${OUTPUT_DIR:-result/penalty_stress_m2_lf_sweep}"
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
  --methods fa_sal \
  --rounds 150 \
  --trials 10 \
  --seed-offset 0 \
  --e_p 250 \
  --r_i 5 \
  --evaluation-horizon 2 \
  --fa-horizon 2 \
  --fa-continuation-policy uncertainty-max \
  --fa-beta-multiplier 1.0 \
  --fa-lf-estimator legacy_axis_quantile \
  --fa-lf-quantile 0.5 \
  --fa-lf-scale-sweep 0.5 1.0 2.0 \
  --fa-l-ell-quantile 0.9 \
  --fa-l-ell-scale-sweep 0.001 \
  --paired-seeds \
  --parallel-workers "${workers}" \
  --max-tasks-per-child 1 \
  2>&1 | tee "${log_path}"

"${python_bin}" scripts/build_penalty_stress_m2_lf_table.py \
  --result-dir "${output_dir}" \
  --output "${table_path}"
