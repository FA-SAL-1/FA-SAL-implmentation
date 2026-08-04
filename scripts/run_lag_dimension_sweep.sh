#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

python_bin="${PYTHON_BIN:-python}"
workers="${WORKERS:-10}"
output_root="${OUTPUT_DIR:-result/lag_dimension_sweep}"

mkdir -p "${output_root}"

for pair in "2 1" "4 1" "8 1" "2 2" "2 4" "2 8" "4 4" "8 8"; do
  read -r dy du <<<"${pair}"
  output_dir="${output_root}/dy${dy}_du${du}"
  mkdir -p "${output_dir}/checkpoints"
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  "${python_bin}" lag_dimension_run.py \
    --output-lags "${dy}" \
    --input-lags "${du}" \
    --environment-preset penalty_stress \
    --methods fa_sal \
    --rounds 150 \
    --trials 10 \
    --e_p 250 \
    --r_i 5 \
    --evaluation-horizon 4 \
    --fa-horizon 4 \
    --fa-beam-width 1 \
    --fa-continuation-policy-sweep random-safe \
    --fa-lf-estimator legacy_axis_quantile \
    --fa-lf-quantile 0.5 \
    --fa-lf-scale-sweep 1 \
    --fa-l-ell-quantile 0.9 \
    --fa-l-ell-scale-sweep 0.001 \
    --snapshot-iters \
    --paired-seeds \
    --parallel-workers "${workers}" \
    --max-tasks-per-child 1 \
    --checkpoint-dir "${output_dir}/checkpoints" \
    --result-dir "${output_dir}" \
    2>&1 | tee "${output_dir}/run.log"
done

"${python_bin}" scripts/build_lag_dimension_sweep_table.py \
  --results-root "${output_root}" \
  --output "${output_root}/table.md"
