#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
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
import run_synthetic_gp_option2 as option2
import run_synthetic_gp_option2_full_grid_median as median


def _physical_nll_cached(
    log_params: np.ndarray,
    squared_distances: np.ndarray,
    centered_y: np.ndarray,
    noise_var: float,
) -> float:
    ell, variance = np.exp(log_params)
    covariance = float(variance) * np.exp(
        -0.5 * squared_distances / float(ell) ** 2
    )
    diagonal = covariance.diagonal().copy()
    np.fill_diagonal(covariance, diagonal + float(noise_var) + 1e-8)
    try:
        factor = np.linalg.cholesky(covariance)
        alpha = cho_solve(
            (factor, True),
            centered_y,
            check_finite=False,
        )
    except np.linalg.LinAlgError:
        return 1e100
    return float(
        0.5 * centered_y @ alpha
        + np.sum(np.log(np.diag(factor)))
        + 0.5 * len(centered_y) * math.log(2.0 * math.pi)
    )


def _optimize_physical_cached(
    gp: gp_models.GaussianProcess1D,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    y_scale = max(float(np.std(y)), 1e-8)
    centered_y = y - float(np.mean(y))
    squared_distances = gp_models._sqeuclidean(X, X)
    physical_noise = float(getattr(gp, "_physical_noise_var", gp.noise_var))
    previous_physical = float(
        getattr(
            gp,
            "_physical_kernel_variance",
            gp.kernel_cfg.variance * y_scale**2,
        )
    )
    starts = [
        np.log([gp.kernel_cfg.length_scale, previous_physical]),
        np.log([0.5, 0.5]),
        np.log([1.0, 1.0]),
        np.log([2.0, 2.0]),
    ]
    results = [
        minimize(
            _physical_nll_cached,
            start,
            args=(squared_distances, centered_y, physical_noise),
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


def _coherent_function_gradient_median_optimized(
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

    base_frequency = rng.standard_normal((median.RFF_FEATURES, dimension))
    frequency = base_frequency / float(gp.kernel_cfg.length_scale)
    phase = rng.uniform(0.0, 2.0 * np.pi, median.RFF_FEATURES)
    scale = np.sqrt(
        2.0 * float(gp.kernel_cfg.variance) / median.RFF_FEATURES
    )
    phi_train = scale * np.cos(X @ frequency.T + phase)
    phi_grid = scale * np.cos(points @ frequency.T + phase)
    weights = rng.standard_normal((median.RFF_FEATURES, sample_count))
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
        (gp.L, True),
        residual,
        check_finite=False,
    )
    cross_kernel = gp_models.kernel_matrix(points, X, gp.kernel_cfg)
    posterior = prior_grid + cross_kernel @ correction_weights
    posterior = float(gp.y_mean) + float(gp.y_scale) * posterior

    action_count = len(np.unique(points[:, 2]))
    values = posterior.T.reshape(
        sample_count,
        median.GRID_POINTS,
        median.GRID_POINTS,
        action_count,
    )
    dy = 8.0 / (median.GRID_POINTS - 1)
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
    return float(np.quantile(norms, 0.5))


option2._optimize_physical = _optimize_physical_cached
option2._anchors = median._full_grid
option2._sample_gradient_bound = (
    _coherent_function_gradient_median_optimized
)


if __name__ == "__main__":
    option2.main()
