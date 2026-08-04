

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import gp_models
import run_synthetic_gp_hyperparameter_update as base

START_ROUNDS = (10, 30, 50)
MODES = ("fixed_noise", "joint_noise")
CONDITIONS = tuple(f"{mode}_r{start}" for mode in MODES for start in START_ROUNDS)
for mode in MODES:
    for start in START_ROUNDS:
        base.CONDITIONS[f"{mode}_r{start}"] = {
            "length_scale": 1.0,
            "variance": 1.0,
            "retrain": True,
            "retrain_start": start,
        }


def _nll(log_params: np.ndarray, d2: np.ndarray, centered_y: np.ndarray, fixed_noise: float | None) -> float:
    values = np.exp(log_params)
    ell, variance = float(values[0]), float(values[1])
    noise = float(fixed_noise if fixed_noise is not None else values[2])
    covariance = variance * np.exp(-0.5 * d2 / ell**2)
    diagonal = covariance.diagonal().copy()
    np.fill_diagonal(covariance, diagonal + noise + 1e-8)
    try:
        factor = np.linalg.cholesky(covariance)
        alpha = cho_solve((factor, True), centered_y, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e100
    return float(0.5 * centered_y @ alpha + np.log(np.diag(factor)).sum() + 0.5 * len(centered_y) * math.log(2 * math.pi))


def _optimize(gp: gp_models.GaussianProcess1D, X: np.ndarray, y: np.ndarray, *, joint_noise: bool) -> dict[str, Any]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    y_scale = max(float(np.std(y)), 1e-8)
    centered_y = y - float(np.mean(y))
    d2 = gp_models._sqeuclidean(X, X)
    previous_variance = float(getattr(gp, "_physical_kernel_variance", gp.kernel_cfg.variance * y_scale**2))
    previous_noise = float(getattr(gp, "_physical_noise_var", gp.noise_var))
    starts_2d = [(gp.kernel_cfg.length_scale, previous_variance), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]
    if joint_noise:
        starts = [np.log([ell, variance, previous_noise]) for ell, variance in starts_2d]
        bounds = [(math.log(0.1), math.log(10.0)), (math.log(0.1), math.log(10.0)), (math.log(1e-8), math.log(1.0))]
        fixed_noise = None
    else:
        starts = [np.log([ell, variance]) for ell, variance in starts_2d]
        bounds = [(math.log(0.1), math.log(10.0))] * 2
        fixed_noise = previous_noise
    results = [minimize(_nll, start, args=(d2, centered_y, fixed_noise), method="L-BFGS-B", bounds=bounds, options={"maxiter": 200, "ftol": 1e-9}) for start in starts]
    result = min((item for item in results if np.isfinite(item.fun)), key=lambda item: float(item.fun))
    values = np.exp(result.x)
    ell, physical_variance = float(values[0]), float(values[1])
    physical_noise = float(values[2]) if joint_noise else previous_noise
    gp.kernel_cfg.length_scale = ell
    gp.kernel_cfg.variance = physical_variance / y_scale**2
    gp.noise_var = physical_noise / y_scale**2
    gp._physical_kernel_variance = physical_variance
    gp._physical_noise_var = physical_noise
    return {"length_scale": ell, "variance": physical_variance, "noise_variance": physical_noise, "normalized_variance": float(gp.kernel_cfg.variance), "normalized_noise_variance": float(gp.noise_var), "success": bool(result.success), "nll": float(result.fun), "nit": int(result.nit), "multi_start": len(starts)}


def _run(task: tuple[str, int]) -> dict[str, Any]:
    condition, seed = task
    joint_noise = condition.startswith("joint_noise")
    base._optimize_kernel = lambda gp, X, y: _optimize(gp, X, y, joint_noise=joint_noise)
    result = base._run_task((condition, seed))
    for name, history in result["hyperparameter_history"].items():
        if history:
            suffix = "f" if name == "dynamics" else "g"
            result["final_hyperparameters"][f"variance_{suffix}"] = float(history[-1]["variance"])
            result["final_hyperparameters"][f"noise_variance_{suffix}"] = float(history[-1]["noise_variance"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--input", type=Path, default=ROOT / "result/synthetic_gp_fixed/results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "result/synthetic_gp_noise_delay/results.json")
    args = parser.parse_args()
    fixed = json.loads(args.input.read_text(encoding="utf-8"))
    nominal = [item for item in fixed["trials"] if item["condition"] == "nominal_fixed" and int(item["seed"]) < args.trials]
    jobs = [(condition, seed) for condition in CONDITIONS for seed in range(args.trials)]
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            item = future.result()
            records.append(item)
            print(f"[{index}/{len(jobs)}] {item['condition']} seed={item['seed']} runtime={item['runtime_seconds']:.1f}s", flush=True)
    records.sort(key=lambda item: (CONDITIONS.index(item["condition"]), int(item["seed"])))
    groups = {"nominal_fixed": nominal, **{condition: [item for item in records if item["condition"] == condition] for condition in CONDITIONS}}
    summary = {name: base._summarize(items) for name, items in groups.items()}
    for name, items in groups.items():
        if name == "nominal_fixed":
            continue
        for key in ("noise_variance_f", "noise_variance_g"):
            values = np.asarray([item["final_hyperparameters"][key] for item in items], dtype=float)
            summary[name][key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    payload = {"experiment": {"benchmark": "penalty_stress", "method": "FA-SAL uncertainty-max", "horizon": 4, "rounds": 150, "paired_seeds": list(range(args.trials)), "conditions": list(CONDITIONS), "retrain_interval": 10, "retrain_start_rounds": list(START_ROUNDS), "physical_mle": True, "multi_start": 4, "joint_noise_bounds": [1e-8, 1.0], "lf_rule": "fixed"}, "trials": groups, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
