#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

python_bin="${PYTHON_BIN:-python}"
workers="${WORKERS:-10}"
output_root="${OUTPUT_DIR:-result/fixed_adaptive_scale}"
input_file="${output_root}/fixed/results.json"
scale_results="${output_root}/scale_conditions/results.json"
nominal_results="${output_root}/nominal_adaptive/results.json"

mkdir -p "$(dirname "${input_file}")" "$(dirname "${scale_results}")" "$(dirname "${nominal_results}")"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
"${python_bin}" scripts/run_synthetic_gp_initial_scale_joint_noise50_lf_update_m4.py \
  --trials 10 \
  --workers "${workers}" \
  --output "${input_file}" \
  2>&1 | tee "${output_root}/fixed/run.log"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
"${python_bin}" scripts/run_synthetic_gp_initial_scale_joint_noise50_mean_gradient_median_m4.py \
  --trials 10 \
  --workers "${workers}" \
  --output "${scale_results}" \
  2>&1 | tee "${output_root}/scale_conditions/run.log"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
"${python_bin}" scripts/run_synthetic_gp_nominal_adaptive_mean_gradient_median_m4.py \
  --trials 10 \
  --workers "${workers}" \
  --output "${nominal_results}" \
  2>&1 | tee "${output_root}/nominal_adaptive/run.log"

"${python_bin}" scripts/build_fixed_adaptive_scale_table.py \
  --input "${input_file}" \
  --scale-results "${scale_results}" \
  --nominal-adaptive-results "${nominal_results}" \
  --output "${output_root}/table.md"
