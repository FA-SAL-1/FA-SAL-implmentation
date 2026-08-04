#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

python_bin="${PYTHON_BIN:-python}"
workers="${WORKERS:-2}"
output_root="${OUTPUT_DIR:-result/rail_pressure_lf_sweep}"

mkdir -p "${output_root}"

for scale in 0.5 1 2 4 8; do
  output_dir="${output_root}/lf_${scale}"
  mkdir -p "${output_dir}"
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  "${python_bin}" rail_pressure_experiment.py \
    --source-dir "${repo_dir}/High-Pressure-Fluid-System" \
    --save-dir "${output_dir}" \
    --methods fa_sal \
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
    --fa-sal-beta-schedule finite_domain_time_uniform \
    --fa-sal-l-ell-quantile 0.9 \
    --fa-sal-l-ell-scale-sweep 0.0001 \
    --fa-sal-lf-multiplier "${scale}" \
    --kernel se \
    --length-scale 1 \
    --pressure-noise-std 1 \
    --safety-noise-std 0.01 \
    --seed 0 \
    2>&1 | tee "${output_dir}/run.log"
done

"${python_bin}" scripts/build_rail_pressure_lf_sweep_table.py \
  --results-root "${output_root}" \
  --output "${output_root}/table.md"
