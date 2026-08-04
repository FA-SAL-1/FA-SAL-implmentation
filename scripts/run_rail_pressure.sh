#!/usr/bin/env bash
set -euo pipefail


repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

python_bin="${PYTHON_BIN:-python}"
workers="${WORKERS:-12}"
output_dir="${OUTPUT_DIR:-result/rail_pressure}"
log_path="${output_dir}/run.log"

mkdir -p "${output_dir}"

echo "Running rail pressure"
echo "  Python:  ${python_bin}"
echo "  Workers: ${workers}"
echo "  Output:  ${output_dir}"

OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
"${python_bin}" rail_pressure_experiment.py \
  --source-dir "${repo_dir}/High-Pressure-Fluid-System" \
  --save-dir "${output_dir}" \
  --methods fa_sal salnx tebbe_abm \
  --rounds 150 \
  --trials 10 \
  --parallel-workers "${workers}" \
  --m 5 \
  --n-init-trajectories 25 \
  --n-eval 150 \
  --speed-grid-points 21 \
  --actuation-grid-points 21 \
  --psi-max 18 \
  --lambda-p 10 \
  --delta-f 0.05 \
  --delta-g 0.05 \
  --fa-sal-beta-schedule fixed \
  --fa-sal-beta-f 1 \
  --fa-sal-beta-g 1 \
  --fa-sal-l-ell-quantile 0.9 \
  --fa-sal-l-ell-scale-sweep 0.0001 \
  --salnx-alpha-sweep 0.2 \
  --salnx-mc-samples 128 \
  --tebbe-alpha 0.2 \
  --tebbe-confidence-delta 0.01 \
  --tebbe-sample-start 32 \
  --tebbe-sample-stages 6 \
  --kernel se \
  --length-scale 1 \
  --pressure-noise-std 1 \
  --safety-noise-std 0.01 \
  --seed 0 \
  2>&1 | tee "${log_path}"

"${python_bin}" scripts/build_result_table.py \
  --result-dir "${output_dir}" \
  --output "${output_dir}/table.md"

echo "Rail pressure complete: ${output_dir}"
