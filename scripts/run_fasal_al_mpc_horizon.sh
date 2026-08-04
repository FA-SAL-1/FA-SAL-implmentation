#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

python_bin="${PYTHON_BIN:-python}"
workers="${WORKERS:-10}"
output_root="${OUTPUT_ROOT:-result/fasal_al_mpc_horizons}"

mkdir -p "${output_root}"

run_fa() {
  local m="$1"
  local estimator="$2"
  local l_ell="$3"
  local beta="$4"
  local out_dir="${output_root}/m${m}/fa_sal"
  mkdir -p "${out_dir}"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  "${python_bin}" main.py \
    --environment-preset penalty_stress --result-dir "${out_dir}" --methods fa_sal \
    --rounds 150 --trials 10 --seed-offset 0 --e_p 250 --r_i 5 \
    --evaluation-horizon "${m}" --fa-horizon "${m}" \
    --fa-continuation-policy uncertainty-max --fa-beta-multiplier "${beta}" \
    --fa-lf-estimator "${estimator}" --fa-lf-quantile 0.5 --fa-lf-scale-sweep 1.0 \
    --fa-l-ell-quantile 0.9 --fa-l-ell-scale-sweep "${l_ell}" \
    --paired-seeds --parallel-workers "${workers}" --max-tasks-per-child 1 \
    2>&1 | tee "${out_dir}/experiment.log"
}

run_al_mpc() {
  local m="$1"
  local rounds=$((150 * m))
  local out_dir="${output_root}/m${m}/al_mpc"
  mkdir -p "${out_dir}"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  "${python_bin}" main.py \
    --environment-preset penalty_stress --result-dir "${out_dir}" --methods al_mpc \
    --rounds "${rounds}" --trials 10 --seed-offset 0 --e_p 250 --r_i 5 \
    --evaluation-horizon "${m}" --nominal-mpc-horizon "${m}" \
    --control-beam-width 1 --control-target 4.45 --nominal-mpc-safety-margin 0.0 \
    --active-mpc-information-weight 5.0 --fa-lf-estimator jacobian \
    --paired-seeds --parallel-workers "${workers}" --max-tasks-per-child 1 \
    2>&1 | tee "${out_dir}/experiment.log"
}

run_fa 1 jacobian 0.14 0.25
run_fa 2 legacy_axis_quantile 0.14 1.0
run_fa 4 jacobian 0.14 1.0
run_fa 8 legacy_axis_quantile 0.001 1.0

for m in 1 2 4 8; do
  run_al_mpc "${m}"
done

"${python_bin}" scripts/build_fasal_al_mpc_horizon_table.py \
  --result-root "${output_root}" \
  --output "${output_root}/table.md"
