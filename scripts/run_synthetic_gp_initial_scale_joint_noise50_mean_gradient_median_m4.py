

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import gp_models
import run_synthetic_gp_initial_scale_joint_noise50_lf_update_m4 as driver

_BASE_COMMON_KWARGS = driver._ORIGINAL_COMMON_KWARGS


def _base_kwargs():
    kwargs = _BASE_COMMON_KWARGS()
    kwargs.update(n_eval=100, recovery_eval_interval=10)
    return kwargs


def _posterior_mean_coordinate_gradient_median(
    gp: gp_models.GaussianProcess1D,
    points: np.ndarray,
    *,
    update_round: int,
) -> float:

    del update_round
    X = np.asarray(gp.X_train, dtype=float)
    points = np.asarray(points, dtype=float)
    ell = float(gp.kernel_cfg.length_scale)
    kernel = gp_models.kernel_matrix(X, points, gp.kernel_cfg)
    derivatives = (
        kernel[:, :, None]
        * (X[:, None, :] - points[None, :, :])
        / ell**2
    )
    gradients = float(gp.y_scale) * np.einsum(
        "nmd,n->md", derivatives, gp.alpha, optimize=True
    )
    return float(np.median(np.abs(gradients)))


driver.CONDITIONS = (
    "joint_low_fixed",
    "joint_low_adaptive_lf",
    "joint_high_fixed",
    "joint_high_adaptive_lf",
)
driver._ORIGINAL_COMMON_KWARGS = _base_kwargs
driver.separation._samplewise_maximum_median = (
    _posterior_mean_coordinate_gradient_median
)
driver.option2.LF_AGGREGATION = (
    "median absolute coordinate derivative of the updated dynamics-GP "
    "posterior mean on the fixed full-domain grid"
)

if __name__ == "__main__":
    driver.main()
