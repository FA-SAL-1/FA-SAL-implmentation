

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import gp_models
import run_synthetic_gp_option2 as option2


GRID_POINTS = 31
RFF_FEATURES = 512


def _full_grid(env: Any) -> np.ndarray:
    y = np.linspace(-2.0, 6.0, GRID_POINTS)
    actions = np.asarray(env.candidate_actions(), dtype=float)
    y_t, y_tm1, u = np.meshgrid(y, y, actions, indexing="ij")
    return np.column_stack([y_t.ravel(), y_tm1.ravel(), u.ravel()])


def _coherent_function_gradient_median(
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

    base_frequency = rng.standard_normal((RFF_FEATURES, dimension))
    frequency = base_frequency / float(gp.kernel_cfg.length_scale)
    phase = rng.uniform(0.0, 2.0 * np.pi, RFF_FEATURES)
    scale = np.sqrt(2.0 * float(gp.kernel_cfg.variance) / RFF_FEATURES)

    phi_train = scale * np.cos(X @ frequency.T + phase)
    phi_grid = scale * np.cos(points @ frequency.T + phase)
    weights = rng.standard_normal((RFF_FEATURES, sample_count))
    prior_train = phi_train @ weights
    prior_grid = phi_grid @ weights

    normalized_y = (
        np.asarray(gp.y_train, dtype=float).reshape(-1, 1) - float(gp.y_mean)
    ) / float(gp.y_scale)
    noise = np.sqrt(float(gp.noise_var)) * rng.standard_normal(
        (X.shape[0], sample_count)
    )
    residual = normalized_y - prior_train - noise
    correction_weights = np.linalg.solve(
        gp.L.T, np.linalg.solve(gp.L, residual)
    )
    cross_kernel = gp_models.kernel_matrix(points, X, gp.kernel_cfg)
    posterior = prior_grid + cross_kernel @ correction_weights
    posterior = float(gp.y_mean) + float(gp.y_scale) * posterior

    action_count = len(np.unique(points[:, 2]))
    values = posterior.T.reshape(
        sample_count, GRID_POINTS, GRID_POINTS, action_count
    )
    dy = 8.0 / (GRID_POINTS - 1)
    actions = np.sort(np.unique(points[:, 2]))
    du = float(actions[1] - actions[0])
    partials = np.gradient(values, dy, dy, du, axis=(1, 2, 3), edge_order=2)
    norms = np.sqrt(sum(np.square(partial) for partial in partials))
    return float(np.quantile(norms, 0.5))


option2.DOMAIN_ANCHORS = GRID_POINTS * GRID_POINTS * 11
option2.LF_AGGREGATION = (
    "50th percentile (median) of posterior-gradient norms "
    "over sampled functions and full-grid points"
)
option2._anchors = _full_grid
option2._sample_gradient_bound = _coherent_function_gradient_median


if __name__ == "__main__":
    option2.main()
