#!/usr/bin/env bash
set -euo pipefail


repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

python_bin="${PYTHON_BIN:-python}"
workers="${WORKERS:-12}"
output_dir="${OUTPUT_DIR:-result/base}"
log_path="${output_dir}/run.log"

mkdir -p "${output_dir}"

echo "Running base"
echo "  Python:  ${python_bin}"
echo "  Workers: ${workers}"
echo "  Output:  ${output_dir}"

OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
"${python_bin}" main.py \
  --environment-preset base \
  --fail-y 5.2 \
  --result-dir "${output_dir}" \
  --methods fa_sal salnx tebbe_abm \
  --rounds 150 \
  --trials 10 \
  --seed-offset 0 \
  --e_p 250 \
  --r_i 5 \
  --evaluation-horizon 4 \
  --fa-horizon 4 \
  --salnx-horizon 4 \
  --tebbe-horizon 4 \
  --fa-lf-estimator legacy_axis_quantile \
  --fa-lf-quantile 0.5 \
  --fa-lf-scale-sweep 1.0 \
  --fa-l-ell-quantile 0.9 \
  --fa-l-ell-scale-sweep 0.00001 \
  --salnx-alpha-sweep 0.2 \
  --tebbe-alpha 0.1 \
  --tebbe-confidence-delta 0.01 \
  --tebbe-sample-start 32 \
  --tebbe-sample-stages 6 \
  --tebbe-endpoint-candidates 81 \
  --tebbe-legacy-mc-prefix \
  --tebbe-local-refinement \
  --no-paired-seeds \
  --parallel-workers "${workers}" \
  2>&1 | tee "${log_path}"

"${python_bin}" scripts/build_result_table.py \
  --result-dir "${output_dir}" \
  --output "${output_dir}/table.md"

echo "Base complete: ${output_dir}"
