from __future__ import annotations

import numpy as np

import experiment
import lag_dimension_experiment as lagged


def estimate_lagged_local_lcb_lipschitz_constant(
    safe_model,
    beta_g,
    env,
    points,
    y_min,
    y_max,
    y_step,
    u_step,
    quantile=0.9,
    max_points=64,
):
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.size == 0:
        return 0.0
    if len(points) > max_points:
        points = points[
            np.linspace(0, len(points) - 1, max_points).astype(int)
        ]
    specs = [
        (axis, max(float(y_step), 1e-6), float(y_min), float(y_max))
        for axis in range(env.output_lags)
    ]
    specs.extend(
        (
            axis,
            max(float(u_step), 1e-6),
            -float(env.u_max),
            float(env.u_max),
        )
        for axis in range(env.output_lags, env.input_dimension)
    )
    norms = []
    for point in points:
        center = float(safe_model.lcb(point, beta_g))
        slopes = []
        for axis, step, lower, upper in specs:
            plus, minus = point.copy(), point.copy()
            plus[axis] = min(float(point[axis]) + step, upper)
            minus[axis] = max(float(point[axis]) - step, lower)
            candidates = []
            if plus[axis] != point[axis]:
                candidates.append(
                    abs(float(safe_model.lcb(plus, beta_g)) - center)
                    / abs(float(plus[axis]) - float(point[axis]))
                )
            if minus[axis] != point[axis]:
                candidates.append(
                    abs(center - float(safe_model.lcb(minus, beta_g)))
                    / abs(float(point[axis]) - float(minus[axis]))
                )
            slopes.append(max(candidates) if candidates else 0.0)
        norms.append(float(np.linalg.norm(slopes)))
    return float(np.quantile(np.asarray(norms), quantile)) if norms else 0.0


experiment.estimate_local_lcb_lipschitz_constant = (
    estimate_lagged_local_lcb_lipschitz_constant
)


if __name__ == "__main__":
    import main

    main.main()
