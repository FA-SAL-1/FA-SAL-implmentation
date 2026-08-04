#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
python_bin="${PYTHON_BIN:-python}"
workers="${WORKERS:-10}"
output_root="${OUTPUT_DIR:-result/gp_scale_grid}"
mkdir -p "${output_root}/grid"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "${python_bin}" scripts/run_synthetic_gp_scale_grid.py --trials 10 --workers "${workers}" --output "${output_root}/grid/results.json"
"${python_bin}" scripts/build_gp_scale_grid_table.py --grid "${output_root}/grid/results.json" --output "${output_root}/table.md"
