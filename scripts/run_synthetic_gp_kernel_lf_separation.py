

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import cho_solve


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import gp_models
import run_synthetic_gp_hyperparameter_update as base
import run_synthetic_gp_option2 as option2
import run_synthetic_gp_option2_full_grid_median as grid
import run_synthetic_gp_option2_full_grid_median_optimized as optimized


ADAPTIVE_CONDITION = "nominal_retrained"
MODES = ("kernel_only", "kernel_plus_lf")
base.CONDITIONS[ADAPTIVE_CONDITION] = {
    "length_scale": 1.0,
    "variance": 1.0,
    "retrain": True,
}


def _samplewise_maximum_median(
    gp: gp_models.GaussianProcess1D,
    points: np.ndarray,
    *,
    update_round: int,
) -> float:

    X = np.asarray(gp.X_train, dtype=float)
    points = np.asarray(points, dtype=float)
    sample_count = int(option2.GRADIENT_SAMPLES)
    dimension = X.shape[1]
    rng = np.random.default_rng(92000 + int(update_round))

    base_frequency = rng.standard_normal((grid.RFF_FEATURES, dimension))
    frequency = base_frequency / float(gp.kernel_cfg.length_scale)
    phase = rng.uniform(0.0, 2.0 * np.pi, grid.RFF_FEATURES)
    scale = np.sqrt(
        2.0 * float(gp.kernel_cfg.variance) / grid.RFF_FEATURES
    )
    phi_train = scale * np.cos(X @ frequency.T + phase)
    phi_grid = scale * np.cos(points @ frequency.T + phase)
    weights = rng.standard_normal((grid.RFF_FEATURES, sample_count))
    prior_train = phi_train @ weights
    prior_grid = phi_grid @ weights

    normalized_y = (
        np.asarray(gp.y_train, dtype=float).reshape(-1, 1)
        - float(gp.y_mean)
    ) / float(gp.y_scale)
    noise = np.sqrt(float(gp.noise_var)) * rng.standard_normal(
        (X.shape[0], sample_count)
    )
    residual = normalized_y - prior_train - noise
    correction_weights = cho_solve(
        (gp.L, True), residual, check_finite=False
    )
    cross_kernel = gp_models.kernel_matrix(points, X, gp.kernel_cfg)
    posterior = prior_grid + cross_kernel @ correction_weights
    posterior = float(gp.y_mean) + float(gp.y_scale) * posterior

    action_count = len(np.unique(points[:, 2]))
    values = posterior.T.reshape(
        sample_count,
        grid.GRID_POINTS,
        grid.GRID_POINTS,
        action_count,
    )
    dy = 8.0 / (grid.GRID_POINTS - 1)
    actions = np.sort(np.unique(points[:, 2]))
    du = float(actions[1] - actions[0])
    partials = np.gradient(
        values,
        dy,
        dy,
        du,
        axis=(1, 2, 3),
        edge_order=2,
    )
    norms = np.sqrt(sum(np.square(partial) for partial in partials))
    samplewise_maxima = norms.reshape(sample_count, -1).max(axis=1)
    return float(np.median(samplewise_maxima))


def _run(job: tuple[str, int]) -> dict[str, Any]:
    mode, seed = job
    base._optimize_kernel = optimized._optimize_physical_cached
    if mode == "kernel_only":
        result = base._run_task((ADAPTIVE_CONDITION, seed))
        dynamics_history = result["hyperparameter_history"]["dynamics"]
        safety_history = result["hyperparameter_history"]["safety"]
        if dynamics_history:
            result["final_hyperparameters"]["variance_f"] = float(
                dynamics_history[-1]["variance"]
            )
        if safety_history:
            result["final_hyperparameters"]["variance_g"] = float(
                safety_history[-1]["variance"]
            )
    elif mode == "kernel_plus_lf":
        option2._optimize_physical = optimized._optimize_physical_cached
        option2._anchors = grid._full_grid
        option2._sample_gradient_bound = _samplewise_maximum_median
        option2.LF_AGGREGATION = (
            "median of sample-wise maximum posterior-function gradient norms"
        )
        result = option2._run((ADAPTIVE_CONDITION, seed))
    else:
        raise ValueError(mode)
    result["mode"] = mode
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "result/synthetic_gp_fixed/results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "result/"
        "synthetic_gp_kernel_lf/results.json",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    nominal_trials = [
        item
        for item in payload["trials"]
        if item["condition"] == "nominal_fixed"
        and int(item["seed"]) < int(args.trials)
    ]
    if len(nominal_trials) != int(args.trials):
        raise ValueError("The input file lacks the requested paired seeds.")

    jobs = [
        (mode, seed)
        for mode in MODES
        for seed in range(int(args.trials))
    ]
    adaptive_results = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {pool.submit(_run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            adaptive_results.append(result)
            print(
                f"[{index}/{len(jobs)}] {result['mode']} "
                f"seed={result['seed']} "
                f"runtime={result['runtime_seconds']:.1f}s",
                flush=True,
            )

    adaptive_results.sort(
        key=lambda item: (MODES.index(item["mode"]), int(item["seed"]))
    )
    groups = {
        "nominal_fixed": nominal_trials,
        **{
            mode: [
                item for item in adaptive_results if item["mode"] == mode
            ]
            for mode in MODES
        },
    }
    payload = {
        "experiment": {
            "benchmark": "penalty_stress",
            "method": "FA-SAL (uncertainty-max)",
            "horizon": 4,
            "rounds": 150,
            "paired_seeds": list(range(int(args.trials))),
            "initial_kernel": {
                "length_scale_f": 1.0,
                "length_scale_g": 1.0,
                "variance_f": 1.0,
                "variance_g": 1.0,
            },
            "retrain_rounds": list(range(10, 151, 10)),
            "mle_bounds": {
                "length_scale": [0.1, 10.0],
                "physical_kernel_variance": [0.1, 10.0],
            },
            "noise_variance_retrained": False,
            "lf_estimator": {
                "posterior_function_samples": int(option2.GRADIENT_SAMPLES),
                "domain": "fixed 31x31 state grid x all 11 actions",
                "aggregation": (
                    "median_j max_x ||gradient f_sample_j(x)||_2"
                ),
                "cumulative_maximum": False,
                "cap": None,
            },
            "l_ell_rule": "local_gradient",
        },
        "trials": groups,
        "summary": {
            name: base._summarize(records)
            for name, records in groups.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
