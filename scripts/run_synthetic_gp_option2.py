

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import experiment
import gp_models
import run_synthetic_gp_hyperparameter_update as base


CONDITIONS = ("joint_low_retrained", "joint_high_retrained")
GRADIENT_SAMPLES = 32
DOMAIN_ANCHORS = 64
LF_AGGREGATION = "maximum posterior-gradient norm"

_LATEST_DYNAMICS: Any = None
_PLANNER_CONFIGS: list[Any] = []


def _physical_nll(
    log_params: np.ndarray, X: np.ndarray, y: np.ndarray, noise_var: float
) -> float:
    ell, variance = np.exp(log_params)
    cfg = gp_models.KernelConfig("se", float(variance), float(ell))
    centered = np.asarray(y, dtype=float).reshape(-1) - float(np.mean(y))
    K = gp_models.kernel_matrix(np.asarray(X, dtype=float), np.asarray(X, dtype=float), cfg)
    C = K + (float(noise_var) + 1e-8) * np.eye(len(centered))
    try:
        L = np.linalg.cholesky(C)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, centered))
    except np.linalg.LinAlgError:
        return 1e100
    return float(
        0.5 * centered @ alpha
        + np.sum(np.log(np.diag(L)))
        + 0.5 * len(centered) * math.log(2.0 * math.pi)
    )


def _optimize_physical(
    gp: gp_models.GaussianProcess1D, X: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    y_scale = max(float(np.std(np.asarray(y, dtype=float))), 1e-8)
    physical_noise = float(getattr(gp, "_physical_noise_var", gp.noise_var))
    previous_physical = float(
        getattr(gp, "_physical_kernel_variance", gp.kernel_cfg.variance * y_scale**2)
    )
    starts = [
        np.log([gp.kernel_cfg.length_scale, previous_physical]),
        np.log([0.5, 0.5]),
        np.log([1.0, 1.0]),
        np.log([2.0, 2.0]),
    ]
    results = [
        minimize(
            _physical_nll,
            start,
            args=(np.asarray(X, float), np.asarray(y, float), physical_noise),
            method="L-BFGS-B",
            bounds=[(math.log(0.1), math.log(10.0))] * 2,
            options={"maxiter": 200, "ftol": 1e-9},
        )
        for start in starts
    ]
    valid = [result for result in results if np.isfinite(result.fun)]
    result = min(valid, key=lambda item: float(item.fun))
    ell, physical_variance = (float(value) for value in np.exp(result.x))
    gp.kernel_cfg.length_scale = ell
    gp._physical_kernel_variance = physical_variance
    gp._physical_noise_var = physical_noise
    gp.kernel_cfg.variance = physical_variance / y_scale**2
    gp.noise_var = physical_noise / y_scale**2
    return {
        "length_scale": ell,
        "variance": physical_variance,
        "normalized_variance": float(gp.kernel_cfg.variance),
        "success": bool(result.success),
        "nll": float(result.fun),
        "nit": int(result.nit),
        "multi_start": len(starts),
    }


def _anchors(env: Any) -> np.ndarray:
    rng = np.random.default_rng(86173)
    y1 = rng.uniform(-2.0, 6.0, DOMAIN_ANCHORS)
    y2 = rng.uniform(-2.0, 6.0, DOMAIN_ANCHORS)
    actions = np.asarray(env.candidate_actions(), dtype=float)
    u = actions[np.arange(DOMAIN_ANCHORS) % len(actions)]
    return np.column_stack([y1, y2, u])


def _sample_gradient_bound(
    gp: gp_models.GaussianProcess1D,
    points: np.ndarray,
    *,
    update_round: int,
) -> float:
    X = np.asarray(gp.X_train, dtype=float)
    points = np.asarray(points, dtype=float)
    ell = float(gp.kernel_cfg.length_scale)
    variance = float(gp.kernel_cfg.variance)
    K = gp_models.kernel_matrix(X, points, gp.kernel_cfg)
    derivatives = K[:, :, None] * (X[:, None, :] - points[None, :, :]) / ell**2
    solved = np.linalg.solve(gp.L, derivatives.reshape(X.shape[0], -1))
    solved = solved.reshape(X.shape[0], points.shape[0], X.shape[1])
    means = gp.y_scale * np.einsum("nmd,n->md", derivatives, gp.alpha)
    prior = (variance / ell**2) * np.eye(X.shape[1])
    rng = np.random.default_rng(91000 + int(update_round))
    maximum = 0.0
    for index in range(points.shape[0]):
        covariance = gp.y_scale**2 * (
            prior - solved[:, index, :].T @ solved[:, index, :]
        )
        covariance = 0.5 * (covariance + covariance.T)
        values, vectors = np.linalg.eigh(covariance)
        covariance = (vectors * np.clip(values, 0.0, None)) @ vectors.T
        gradients = rng.multivariate_normal(
            means[index], covariance, size=GRADIENT_SAMPLES
        )
        maximum = max(maximum, float(np.max(np.linalg.norm(gradients, axis=1))))
    return maximum


def _install_hooks() -> None:
    global _LATEST_DYNAMICS, _PLANNER_CONFIGS
    _LATEST_DYNAMICS = None
    _PLANNER_CONFIGS = []
    original_dyn = experiment.DynamicsGP
    original_planner = experiment.FutureAwarePlanner
    original_l_ell = experiment.estimate_local_lcb_lipschitz_constant

    class TrackingDynamics(original_dyn):
        def __init__(self, *args: Any, **kwargs: Any):
            global _LATEST_DYNAMICS
            super().__init__(*args, **kwargs)
            _LATEST_DYNAMICS = self

    class TrackingPlanner(original_planner):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            _PLANNER_CONFIGS.append(self.cfg)

    def updating_l_ell(*args: Any, **kwargs: Any) -> float:
        value = float(original_l_ell(*args, **kwargs))
        dyn = _LATEST_DYNAMICS
        history = getattr(dyn, "hyperparameter_history", []) if dyn is not None else []
        if history:
            update = history[-1]
            if "estimated_lf" not in update:
                env = kwargs.get("env", args[2] if len(args) > 2 else None)
                estimate = _sample_gradient_bound(
                    dyn.gp, _anchors(env), update_round=int(update["after_round"])
                )
                update["estimated_lf"] = estimate
                update["gradient_samples"] = GRADIENT_SAMPLES
                update["domain_anchors"] = DOMAIN_ANCHORS
                for cfg in _PLANNER_CONFIGS:
                    cfg.Lf = estimate
        return value

    experiment.DynamicsGP = TrackingDynamics
    experiment.FutureAwarePlanner = TrackingPlanner
    experiment.estimate_local_lcb_lipschitz_constant = updating_l_ell
    base.synthetic_experiment = experiment
    base.gp_models = gp_models
    base._optimize_kernel = _optimize_physical


def _run(task: tuple[str, int]) -> dict[str, Any]:
    _install_hooks()
    result = base._run_task(task)
    dyn = _LATEST_DYNAMICS
    if dyn is not None and hasattr(dyn.gp, "_physical_kernel_variance"):
        result["final_hyperparameters"]["variance_f"] = float(
            dyn.gp._physical_kernel_variance
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "result/synthetic_gp_option2/results.json",
    )
    args = parser.parse_args()
    tasks = [(condition, seed) for condition in CONDITIONS for seed in range(args.trials)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(tasks)}] {result['condition']} seed={result['seed']} "
                f"runtime={result['runtime_seconds']:.1f}s",
                flush=True,
            )
    ordered = sorted(results, key=lambda item: (CONDITIONS.index(item["condition"]), item["seed"]))
    payload = {
        "experiment": {
            "option": 2,
            "conditions": list(CONDITIONS),
            "paired_seeds": list(range(args.trials)),
            "retrain_rounds": list(range(10, 151, 10)),
            "gradient_samples": GRADIENT_SAMPLES,
            "domain_anchors": DOMAIN_ANCHORS,
            "lf_aggregation": LF_AGGREGATION,
            "l_ell_rule": "local_gradient",
        },
        "trials": ordered,
        "summary": {
            condition: base._summarize(
                [item for item in ordered if item["condition"] == condition]
            )
            for condition in CONDITIONS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
