import os
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import psutil
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class _TqdmFallback:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else ())

        def update(self, n=1):
            return None

        def set_postfix_str(self, text):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, **kwargs):
        return _TqdmFallback(iterable, **kwargs)

from environment import NARXDoubleIntegratorEnv
from gp_models import DynamicsGP, KernelConfig, SafetyGP, normal_cdf, normal_cdf_array
from planning import (
    CandidateInfo,
    ConfidenceSchedule,
    FARandomPlanner,
    FutureAwarePlanner,
    NominalMPCPlanner,
    ActiveLearningMPCPlanner,
    PlannerConfig,
    PointwiseSafePlanner,
    SafeExplorationPlanner,
    SALNXPlanner,
    TebbeABMPlanner,
    evaluate_planned_rollout,
)


class _BoundedOracleCache(OrderedDict):


    def __init__(self, max_entries: int):
        super().__init__()
        self.max_entries = max(1, int(max_entries))

    def __setitem__(self, key, value) -> None:
        if key in self:
            super().__setitem__(key, value)
            return
        if len(self) >= self.max_entries:
            self.popitem(last=False)
        super().__setitem__(key, value)


class _PeakRSSSampler:


    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = float(interval_seconds)
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = int(self.process.memory_info().rss)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.peak_bytes = max(
                    self.peak_bytes,
                    int(self.process.memory_info().rss),
                )
            except (psutil.Error, OSError):
                return

    def start(self) -> None:
        self._thread.start()

    def stop_mib(self) -> float:
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, 2.0 * self.interval_seconds))
        try:
            self.peak_bytes = max(
                self.peak_bytes,
                int(self.process.memory_info().rss),
            )
        except (psutil.Error, OSError):
            pass
        return float(self.peak_bytes / (1024.0**2))


def generate_initial_safe_data(
    env: NARXDoubleIntegratorEnv,
    rng: np.random.Generator,
    n_init: int,
    actions: np.ndarray,
    y_range: Tuple[float, float],
    velocity_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y_next: List[float] = []
    g_values: List[float] = []

    while len(X) < n_init:
        y_t = rng.uniform(*y_range)
        velocity = rng.uniform(*velocity_range)
        y_tm1 = y_t - velocity * env.dt
        u_t = float(rng.choice(actions))
        z = env.make_regressor(y_t, y_tm1, u_t)
        if not env.is_safe(z):
            continue
        X.append(z)
        y_next.append(env.step(z, noise=True))
        g_values.append(env.observe_safety(z, noise=True))

    return np.asarray(X, dtype=float), np.asarray(y_next, dtype=float), np.asarray(g_values, dtype=float)


def candidate_id(info: CandidateInfo) -> Tuple[float, float]:
    param = info.trajectory_param if info.trajectory_param is not None else info.u0
    return (round(float(info.u0), 12), round(float(param), 12))


def choose_fallback_candidate(infos: Sequence[CandidateInfo]) -> Tuple[Optional[float], Optional[CandidateInfo]]:
    if not infos:
        return None, None

    def fallback_key(info: CandidateInfo) -> Tuple[float, float, float, float]:
        return (
            float(info.pointwise_lcb),
            float(info.min_safety_slack),
            float(info.trajectory_safety_prob),
            float(info.acquisition),
        )

    best = max(infos, key=fallback_key)
    return float(best.u0), best


def oracle_key_for_chosen(
    chosen: CandidateInfo,
    metric_infos: Sequence[CandidateInfo],
) -> Optional[Tuple[float, float]]:
    chosen_key = candidate_id(chosen)
    metric_keys = {candidate_id(info) for info in metric_infos}
    if chosen_key in metric_keys:
        return chosen_key

    if chosen.trajectory_param is not None:
        target_eta = float(chosen.trajectory_param)
        eta_matches = [
            candidate_id(info)
            for info in metric_infos
            if info.trajectory_param is not None and abs(float(info.trajectory_param) - target_eta) <= 1e-12
        ]
        if eta_matches:
            return eta_matches[0]

    action_matches = [
        candidate_id(info)
        for info in metric_infos
        if abs(float(info.u0) - float(chosen.u0)) <= 1e-12
    ]
    if action_matches:
        return action_matches[0]
    return None


def build_evaluation_dataset(
    env: NARXDoubleIntegratorEnv,
    rng: np.random.Generator,
    actions: np.ndarray,
    n_eval: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_eval: List[np.ndarray] = []
    y_eval: List[float] = []
    safe_eval: List[float] = []
    while len(X_eval) < n_eval:
        if env.kind == "runaway_zone" and rng.random() < 0.75:
            y_t = rng.uniform(env.runaway_center - 1.0, env.runaway_center + 0.9)
            velocity = rng.uniform(max(0.2, env.runaway_velocity_threshold), 2.2)
        elif env.kind in {"diagnostic_deadend", "sparse_deadend"} and rng.random() < 0.75:
            y_t = rng.uniform(max(0.8, env.y_max - 2.2), env.y_max - 0.2)
            velocity = rng.uniform(0.4, 1.8)
        elif env.kind in {"diagnostic_deadend", "sparse_deadend"} and rng.random() < 0.95:
            y_t = rng.uniform(0.5, env.y_max - 0.1)
            velocity = rng.uniform(-0.2, 1.4)
        elif env.kind == "funnel_lattice" and rng.random() < 0.55:
            center = float(rng.choice(env.lattice_centers()))
            y_t = rng.uniform(center - 0.8, center + 0.8)
            velocity = rng.uniform(-0.1, 1.8)
        elif env.kind == "funnel_lattice" and rng.random() < 0.8:
            centers = env.lattice_centers()
            if len(centers) >= 2:
                gap_idx = int(rng.integers(len(centers) - 1))
                y_t = rng.uniform(centers[gap_idx] + 0.35, centers[gap_idx + 1] - 0.35)
            else:
                y_t = rng.uniform(0.0, env.y_max)
            velocity = rng.uniform(0.0, 1.2)
        elif env.kind in {"commitment_funnel", "goal_funnel"} and rng.random() < 0.85:
            y_t = rng.uniform(env.funnel_center - 0.9, env.funnel_center + 0.7)
            velocity = rng.uniform(max(0.25, env.funnel_velocity_threshold), 2.6)
        elif env.kind == "goal_funnel" and rng.random() < 0.95:
            y_t = rng.uniform(env.goal_y_min - 0.8, env.goal_y_max + 0.2)
            velocity = rng.uniform(0.0, 1.8)
        elif rng.random() < 0.85:
            y_t = rng.uniform(max(0.2, env.y_max - 1.5), env.y_max + 0.3)
            velocity = rng.uniform(0.5, 2.5)
        else:
            y_t = rng.uniform(-0.8, env.y_max + 0.8)
            velocity = rng.uniform(-1.2, 1.8)
        y_tm1 = y_t - velocity * env.dt
        u_t = float(rng.choice(actions))
        z = env.make_regressor(y_t, y_tm1, u_t)
        X_eval.append(z)
        y_eval.append(env.transition_mean(z))
        safe_eval.append(float(env.is_safe(z)))
    return np.asarray(X_eval, dtype=float), np.asarray(y_eval, dtype=float), np.asarray(safe_eval, dtype=float)


def build_safe_volume_dataset(
    env: NARXDoubleIntegratorEnv,
    y_min: float,
    y_max: float,
    y_grid_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y_values = np.linspace(y_min, y_max, y_grid_points)
    actions = env.candidate_actions()
    X_volume: List[np.ndarray] = []
    safe_volume: List[float] = []
    for y_t in y_values:
        for y_tm1 in y_values:
            for u_t in actions:
                z = env.make_regressor(float(y_t), float(y_tm1), float(u_t))
                X_volume.append(z)
                safe_volume.append(float(env.is_safe(z)))
    return np.asarray(X_volume, dtype=float), np.asarray(safe_volume, dtype=float)


def _axis_lipschitz_upper_bound(values: np.ndarray, spacings: Sequence[float], quantile: float = 1.0) -> float:
    grads_sq = 0.0
    for axis, spacing in enumerate(spacings):
        if values.shape[axis] < 2 or spacing <= 0.0:
            continue
        axis_diff = np.diff(values, axis=axis) / float(spacing)
        if axis_diff.size > 0:
            robust_grad = float(np.quantile(np.abs(axis_diff), quantile))
            grads_sq += robust_grad**2
    return float(np.sqrt(grads_sq))


def estimate_dynamics_lipschitz_constant(
    env: NARXDoubleIntegratorEnv,
    y_min: float,
    y_max: float,
    y_grid_points: int,
    quantile: float = 1.0,
    estimator: str = "legacy_axis_quantile",
) -> float:

    if estimator == "legacy_axis_quantile":
        grid_points = max(5, min(int(y_grid_points), 17))
        y_values = np.linspace(y_min, y_max, grid_points)
        actions = env.candidate_actions()
        values = np.empty((grid_points, grid_points, len(actions)), dtype=float)
        for i, y_t in enumerate(y_values):
            for j, y_tm1 in enumerate(y_values):
                for k, u_t in enumerate(actions):
                    z = env.make_regressor(float(y_t), float(y_tm1), float(u_t))
                    values[i, j, k] = env.transition_mean(z)
        dy = float(y_values[1] - y_values[0]) if grid_points > 1 else 1.0
        du = float(actions[1] - actions[0]) if len(actions) > 1 else 1.0
        return _axis_lipschitz_upper_bound(
            values,
            (dy, dy, du),
            quantile=float(np.clip(quantile, 0.0, 1.0)),
        )
    if estimator != "jacobian":
        raise ValueError(f"Unknown dynamics Lipschitz estimator: {estimator}")

    grid_points = max(5, int(y_grid_points))
    y_values = np.linspace(y_min, y_max, grid_points)
    actions = env.candidate_actions()
    values = np.empty((grid_points, grid_points, len(actions)), dtype=float)
    for i, y_t in enumerate(y_values):
        for j, y_tm1 in enumerate(y_values):
            for k, u_t in enumerate(actions):
                z = env.make_regressor(float(y_t), float(y_tm1), float(u_t))
                values[i, j, k] = env.transition_mean(z)
    dy = float(y_values[1] - y_values[0])
    du = float(actions[1] - actions[0]) if len(actions) > 1 else 1.0
    edge_order = 2 if min(values.shape) >= 3 else 1
    partials = np.gradient(values, dy, dy, du, edge_order=edge_order)
    jacobian_norms = np.sqrt(sum(np.square(partial) for partial in partials))
    return float(np.quantile(jacobian_norms, float(np.clip(quantile, 0.0, 1.0))))


def estimate_lcb_lipschitz_constant(
    safe_model: SafetyGP,
    beta_g: float,
    env: NARXDoubleIntegratorEnv,
    y_min: float,
    y_max: float,
    y_grid_points: int,
    quantile: float = 1.0,
) -> float:
    grid_points = max(5, min(int(y_grid_points), 17))
    y_values = np.linspace(y_min, y_max, grid_points)
    actions = env.candidate_actions()
    X_grid: List[np.ndarray] = []
    for y_t in y_values:
        for y_tm1 in y_values:
            for u_t in actions:
                X_grid.append(env.make_regressor(float(y_t), float(y_tm1), float(u_t)))
    X_grid_np = np.asarray(X_grid, dtype=float)
    lcb_values = safe_model.lcb_batch(X_grid_np, beta_g).reshape(grid_points, grid_points, len(actions))
    dy = float(y_values[1] - y_values[0]) if grid_points > 1 else 1.0
    du = float(actions[1] - actions[0]) if len(actions) > 1 else 1.0
    return _axis_lipschitz_upper_bound(lcb_values, (dy, dy, du), quantile=quantile)


def estimate_local_lcb_lipschitz_constant(
    safe_model: SafetyGP,
    beta_g: float,
    env: NARXDoubleIntegratorEnv,
    points: np.ndarray,
    y_min: float,
    y_max: float,
    y_step: float,
    u_step: float,
    quantile: float = 0.9,
    max_points: int = 64,
) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.size == 0:
        return 0.0

    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(int)
        points = points[idx]

    y_step = max(float(y_step), 1e-6)
    u_step = max(float(u_step), 1e-6)
    local_norms: List[float] = []

    for z in points:
        z = np.asarray(z, dtype=float)
        center = float(safe_model.lcb(z, beta_g))
        axis_slopes: List[float] = []
        for axis, step, lower, upper in (
            (0, y_step, y_min, y_max),
            (1, y_step, y_min, y_max),
            (2, u_step, -env.u_max, env.u_max),
        ):
            plus = z.copy()
            minus = z.copy()
            plus[axis] = min(float(z[axis]) + step, upper)
            minus[axis] = max(float(z[axis]) - step, lower)

            candidates = []
            if abs(float(plus[axis]) - float(z[axis])) > 1e-10:
                lcb_plus = float(safe_model.lcb(plus, beta_g))
                candidates.append(abs(lcb_plus - center) / abs(float(plus[axis]) - float(z[axis])))
            if abs(float(minus[axis]) - float(z[axis])) > 1e-10:
                lcb_minus = float(safe_model.lcb(minus, beta_g))
                candidates.append(abs(center - lcb_minus) / abs(float(z[axis]) - float(minus[axis])))
            axis_slopes.append(max(candidates) if candidates else 0.0)

        local_norms.append(float(np.linalg.norm(axis_slopes)))

    if not local_norms:
        return 0.0
    return float(np.quantile(np.asarray(local_norms, dtype=float), quantile))


def evaluate_model_metrics(
    dyn_model: DynamicsGP,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> Dict[str, float]:
    pred_y, _ = dyn_model.gp.mean_std(X_eval)
    dynamics_rmse = float(np.sqrt(np.mean((pred_y - y_eval) ** 2)))
    return {
        "dynamics_rmse": dynamics_rmse,
    }


def evaluate_safety_set_recovery_metrics(
    safe_model: SafetyGP,
    cfg: PlannerConfig,
    X_eval: np.ndarray,
    true_safe: np.ndarray,
) -> Dict[str, float]:
    mu_g, std_g = safe_model.predict_batch(X_eval)
    prob_safe = normal_cdf_array(np.divide(mu_g, np.maximum(std_g, 1e-8)))
    predicted_safe = np.asarray(prob_safe >= (1.0 - cfg.alpha_safety), dtype=bool)
    true_safe = np.asarray(true_safe, dtype=bool)

    intersection = float(np.sum(predicted_safe & true_safe))
    true_count = float(np.sum(true_safe))
    pred_count = float(np.sum(predicted_safe))
    union = float(np.sum(predicted_safe | true_safe))

    recall = intersection / true_count if true_count > 0 else 1.0
    precision = intersection / pred_count if pred_count > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return {
        "safety_set_recovery_recall": float(recall),
        "safety_set_recovery_precision": float(precision),
        "safety_set_recovery_iou": float(iou),
    }


def _dead_end_free_cache_key(z: np.ndarray, horizon: int) -> Tuple[float, float, float, int]:
    return (
        round(float(z[0]), 10),
        round(float(z[1]), 10),
        round(float(z[2]), 10),
        int(horizon),
    )


def _ordered_oracle_successors(
    env: NARXDoubleIntegratorEnv,
    z: np.ndarray,
    y_next: float,
    actions: np.ndarray,
) -> List[np.ndarray]:

    successors = [
        env.shift(z, y_next, float(next_action))
        for next_action in actions
    ]
    successors.sort(key=env.safety_value, reverse=True)
    return successors


def is_true_dead_end_free(
    env: NARXDoubleIntegratorEnv,
    z: np.ndarray,
    horizon: int,
    actions: np.ndarray,
    cache: Optional[Dict[Tuple[float, float, float, int], bool]] = None,
) -> bool:
    if cache is None:
        cache = {}
    key = _dead_end_free_cache_key(z, horizon)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if not env.is_safe(z):
        cache[key] = False
        return False
    if horizon <= 1:
        cache[key] = True
        return True

    y_next = env.step(z, noise=False)
    for z_next in _ordered_oracle_successors(env, z, y_next, actions):
        if is_true_dead_end_free(env, z_next, horizon - 1, actions, cache):
            cache[key] = True
            return True
    cache[key] = False
    return False





def oracle_future_viable_at_threshold(
    env: NARXDoubleIntegratorEnv,
    z: np.ndarray,
    horizon: int,
    actions: np.ndarray,
    threshold: float = 0.0,
    cache: Optional[Dict[Tuple[float, float, float, int, float], bool]] = None,
) -> bool:

    if cache is None:
        cache = {}

    threshold = float(threshold)
    key = (*_dead_end_free_cache_key(z, horizon), threshold)
    cached = cache.get(key)
    if cached is not None:
        return bool(cached)

    if float(env.safety_value(z)) < threshold:
        cache[key] = False
        return False

    if horizon <= 1:
        cache[key] = True
        return True

    y_next = env.step(z, noise=False)
    for z_next in _ordered_oracle_successors(env, z, y_next, actions):
        if oracle_future_viable_at_threshold(
            env=env,
            z=z_next,
            horizon=horizon - 1,
            actions=actions,
            threshold=threshold,
            cache=cache,
        ):
            cache[key] = True
            return True

    cache[key] = False
    return False


class BackwardViabilityDP:


    def __init__(
        self,
        env: NARXDoubleIntegratorEnv,
        actions: np.ndarray,
        horizon: int,
        y_min: float,
        y_max: float,
        y_grid_points: int,
        fallback_cache: Optional[
            Dict[Tuple[float, float, float, int, float], bool]
        ] = None,
    ) -> None:
        self.env = env
        self.actions = np.asarray(actions, dtype=float)
        self.horizon = max(1, int(horizon))
        self.y_values = np.linspace(
            min(float(y_min), -float(env.y_max)),
            max(float(y_max), float(env.y_max)),
            max(5, int(y_grid_points)),
        )
        self.y_min = float(self.y_values[0])
        self.y_max = float(self.y_values[-1])
        self.y_step = float(self.y_values[1] - self.y_values[0])
        self._action_indices = {
            round(float(action), 12): idx
            for idx, action in enumerate(self.actions)
        }
        self._fallback_cache = fallback_cache if fallback_cache is not None else {}
        self._tables: Dict[float, np.ndarray] = {}
        self._next_y_indices, self._next_inside = self._build_transition_map()

    def _build_transition_map(self) -> Tuple[np.ndarray, np.ndarray]:
        n_y = len(self.y_values)
        n_actions = len(self.actions)
        next_indices = np.empty((n_y, n_y, n_actions), dtype=np.int32)
        inside = np.empty((n_y, n_y, n_actions), dtype=bool)
        for i, y_t in enumerate(self.y_values):
            for j, y_tm1 in enumerate(self.y_values):
                for k, action in enumerate(self.actions):
                    z = self.env.make_regressor(y_t, y_tm1, float(action))
                    y_next = float(self.env.step(z, noise=False))
                    inside[i, j, k] = self.y_min <= y_next <= self.y_max
                    next_indices[i, j, k] = int(
                        np.clip(
                            np.rint((y_next - self.y_min) / self.y_step),
                            0,
                            n_y - 1,
                        )
                    )
        return next_indices, inside

    def _table(self, threshold: float) -> np.ndarray:
        threshold = float(threshold)
        cached = self._tables.get(threshold)
        if cached is not None:
            return cached

        n_y = len(self.y_values)
        n_actions = len(self.actions)
        table = np.empty(
            (self.horizon, n_y, n_y, n_actions),
            dtype=bool,
        )
        safe_y = np.minimum(
            float(self.env.y_max) - self.y_values,
            self.y_values + float(self.env.y_max),
        ) >= threshold
        table[0] = np.broadcast_to(
            safe_y[:, None, None],
            (n_y, n_y, n_actions),
        )
        current_y_indices = np.broadcast_to(
            np.arange(n_y, dtype=np.int32)[:, None],
            (n_y, n_y),
        )

        for depth_idx in range(1, self.horizon):
            previous = table[depth_idx - 1]
            for action_idx in range(n_actions):
                next_y_indices = self._next_y_indices[:, :, action_idx]
                continuation = np.zeros((n_y, n_y), dtype=bool)
                for next_action_idx in range(n_actions):
                    continuation |= (
                        self._next_inside[:, :, action_idx]
                        & previous[
                            next_y_indices,
                            current_y_indices,
                            next_action_idx,
                        ]
                    )
                table[depth_idx, :, :, action_idx] = (
                    safe_y[:, None] & continuation
                )

        self._tables[threshold] = table
        return table

    def query(
        self,
        z: np.ndarray,
        horizon: int,
        threshold: float,
    ) -> bool:
        threshold = float(threshold)
        if float(self.env.safety_value(z)) < threshold:
            return False

        horizon = max(1, int(horizon))
        action_idx = self._action_indices.get(round(float(z[2]), 12))
        if action_idx is None and horizon > 1:
            y_next = float(self.env.step(z, noise=False))
            for z_next in _ordered_oracle_successors(
                self.env,
                z,
                y_next,
                self.actions,
            ):
                if self.query(z_next, horizon - 1, threshold):
                    return True
            return False
        inside = (
            self.y_min <= float(z[0]) <= self.y_max
            and self.y_min <= float(z[1]) <= self.y_max
            and action_idx is not None
            and horizon <= self.horizon
        )
        if not inside:
            return oracle_future_viable_at_threshold(
                env=self.env,
                z=z,
                horizon=horizon,
                actions=self.actions,
                threshold=threshold,
                cache=self._fallback_cache,
            )

        y_t_idx = int(
            np.clip(
                np.rint((float(z[0]) - self.y_min) / self.y_step),
                0,
                len(self.y_values) - 1,
            )
        )
        y_tm1_idx = int(
            np.clip(
                np.rint((float(z[1]) - self.y_min) / self.y_step),
                0,
                len(self.y_values) - 1,
            )
        )
        return bool(
            self._table(threshold)[
                horizon - 1,
                y_t_idx,
                y_tm1_idx,
                int(action_idx),
            ]
        )

def evaluate_selected_action_dead_end_oracle(
    env: NARXDoubleIntegratorEnv,
    y_t: float,
    y_tm1: float,
    selected_action: float,
    horizon: int,
    actions: np.ndarray,
    epsilon: float,
    cache: Optional[Dict[Tuple[float, float, float, int, float], bool]] = None,
    viability_dp: Optional[BackwardViabilityDP] = None,
) -> Dict[str, float]:

    z1 = env.make_regressor(float(y_t), float(y_tm1), float(selected_action))
    immediate_margin = float(env.safety_value(z1))
    immediate_safe = bool(immediate_margin >= 0.0)

    if cache is None:
        cache = {}

    epsilon = float(epsilon)
    known_thresholds: List[Tuple[float, bool]] = []

    def threshold_query(threshold: float) -> bool:
        threshold = float(threshold)
        for known_threshold, result in known_thresholds:
            if result and known_threshold >= threshold:
                return True
            if not result and known_threshold <= threshold:
                return False
        if viability_dp is not None:
            result = viability_dp.query(z1, int(horizon), threshold)
        else:
            result = oracle_future_viable_at_threshold(
                env=env,
                z=z1,
                horizon=int(horizon),
                actions=actions,
                threshold=threshold,
                cache=cache,
            )
        known_thresholds.append((threshold, result))
        return result

    interior_future_safe = threshold_query(epsilon)
    true_future_safe = threshold_query(0.0)
    selected_dead_end = bool(immediate_safe and not true_future_safe)

    epsilon_dead_end = False
    if immediate_margin >= epsilon:
        above_negative_epsilon = threshold_query(
            float(np.nextafter(-epsilon, np.inf))
        )
        epsilon_dead_end = not above_negative_epsilon

    return {
        "immediate_margin": float(immediate_margin),
        "immediate_safe": float(immediate_safe),
        "future_safe": float(true_future_safe),
        "interior_future_safe": float(interior_future_safe),
        "dead_end": float(selected_dead_end),
        "epsilon_dead_end": float(epsilon_dead_end),
        "plan_complete": 1.0,
    }


def evaluate_candidate_action_dead_end_oracle(
    env: NARXDoubleIntegratorEnv,
    y_t: float,
    y_tm1: float,
    info: CandidateInfo,
    horizon: int,
    actions: np.ndarray,
    epsilon: float,
    cache: Optional[Dict[Tuple[float, float, float, int, float], bool]] = None,
    viability_dp: Optional[BackwardViabilityDP] = None,
) -> Dict[str, float]:

    return evaluate_selected_action_dead_end_oracle(
        env=env,
        y_t=y_t,
        y_tm1=y_tm1,
        selected_action=float(info.u0),
        horizon=horizon,
        actions=actions,
        epsilon=epsilon,
        cache=cache,
        viability_dp=viability_dp,
    )

def build_dead_end_free_dataset(
    env: NARXDoubleIntegratorEnv,
    X_eval: np.ndarray,
    horizon: int,
    actions: np.ndarray,
    viability_dp: Optional[BackwardViabilityDP] = None,
) -> np.ndarray:
    cache: Dict[Tuple[float, float, float, int], bool] = {}
    if viability_dp is not None:
        return np.asarray(
            [float(viability_dp.query(z, horizon, 0.0)) for z in X_eval],
            dtype=float,
        )
    return np.asarray(
        [float(is_true_dead_end_free(env, z, horizon, actions, cache)) for z in X_eval],
        dtype=float,
    )


def evaluate_dead_end_free_recovery_metrics(
    planner,
    X_eval: np.ndarray,
    true_dead_end_free: np.ndarray,
    actions: np.ndarray,
) -> Dict[str, float]:
    if hasattr(planner, "batch_any_feasible"):
        predicted_dead_end_free = np.asarray(planner.batch_any_feasible(X_eval, actions), dtype=bool)
    else:
        predicted_dead_end_free = []
        for z in X_eval:
            if hasattr(planner, "set_trajectory_anchor"):
                planner.set_trajectory_anchor(float(z[2]))
            infos = planner.candidate_infos(float(z[0]), float(z[1]), actions)
            predicted_dead_end_free.append(float(any(info.feasible_full_horizon for info in infos)))
        predicted_dead_end_free = np.asarray(predicted_dead_end_free, dtype=bool)
    true_dead_end_free = np.asarray(true_dead_end_free, dtype=bool)

    intersection = float(np.sum(predicted_dead_end_free & true_dead_end_free))
    true_count = float(np.sum(true_dead_end_free))
    pred_count = float(np.sum(predicted_dead_end_free))
    union = float(np.sum(predicted_dead_end_free | true_dead_end_free))

    recall = intersection / true_count if true_count > 0 else 1.0
    precision = intersection / pred_count if pred_count > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return {
        "dead_end_free_recovery_recall": float(recall),
        "dead_end_free_recovery_precision": float(precision),
        "dead_end_free_recovery_iou": float(iou),
    }


def evaluate_theorem_metrics(
    feasible_actions: Sequence[Tuple[float, float]],
    oracle_data: Dict[Tuple[float, float], Dict[str, float]],
) -> Dict[str, float]:
    feasible_set = set(feasible_actions)
    true_future_safe = {key for key, stats in oracle_data.items() if float(stats["future_safe"]) > 0.5}
    interior_future_safe = {key for key, stats in oracle_data.items() if float(stats["interior_future_safe"]) > 0.5}
    boundary_band = {
        key
        for key, stats in oracle_data.items()
        if float(stats["future_safe"]) > 0.5 and float(stats["interior_future_safe"]) <= 0.5
    }
    false_certified = feasible_set - true_future_safe
    false_certification_rate = len(false_certified) / len(feasible_set) if feasible_set else 0.0
    interior_recall = len(feasible_set & interior_future_safe) / len(interior_future_safe) if interior_future_safe else 1.0
    interior_union = feasible_set | interior_future_safe
    interior_iou = len(feasible_set & interior_future_safe) / len(interior_union) if interior_union else 1.0
    boundary_misses = (true_future_safe - feasible_set) - boundary_band
    boundary_miss_rate = len(boundary_misses) / len(true_future_safe) if true_future_safe else 0.0
    return {
        "false_certification_rate": float(false_certification_rate),
        "interior_recall_epsilon": float(interior_recall),
        "interior_iou_epsilon": float(interior_iou),
        "boundary_miss_rate": float(boundary_miss_rate),
        "true_future_safe_count": float(len(true_future_safe)),
        "feasible_future_safe_count": float(len(feasible_set)),
    }


def history_to_phase_points(history_y: Sequence[float]) -> np.ndarray:
    if len(history_y) < 2:
        return np.empty((0, 2), dtype=float)
    pts = [(float(history_y[i - 1]), float(history_y[i])) for i in range(1, len(history_y))]
    return np.asarray(pts, dtype=float)


def planned_states_to_phase_points(planned_states: Sequence[np.ndarray]) -> np.ndarray:
    if not planned_states:
        return np.empty((0, 2), dtype=float)
    pts = [(float(z[1]), float(z[0])) for z in planned_states]
    return np.asarray(pts, dtype=float)


def make_snapshot(
    env: NARXDoubleIntegratorEnv,
    safe_model: SafetyGP,
    cfg: PlannerConfig,
    chosen: Optional[CandidateInfo],
    history_y: Sequence[float],
    rmse_history: Sequence[float],
    round_idx: int,
    x_min: float,
    x_max: float,
    grid_size: int,
    u_slice: float,
) -> Dict[str, object]:
    x1 = np.linspace(x_min, x_max, grid_size)
    x2 = np.linspace(x_min, x_max, grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    Z_grid = np.column_stack([X2.reshape(-1), X1.reshape(-1), np.full(X1.size, float(u_slice))])

    true_safe = np.asarray([env.is_safe(z) for z in Z_grid], dtype=float).reshape(X1.shape)
    mu_g, std_g = safe_model.gp.mean_std(Z_grid)
    prob_safe = np.asarray(
        [normal_cdf(mu / max(std, 1e-8)) for mu, std in zip(mu_g, std_g)],
        dtype=float,
    ).reshape(X1.shape)
    predicted_safe = prob_safe >= (1.0 - cfg.alpha_safety)

    return {
        "round_idx": int(round_idx),
        "x1_grid": X1,
        "x2_grid": X2,
        "true_safe": true_safe,
        "predicted_safe": predicted_safe.astype(float),
        "prob_safe": prob_safe,
        "executed_path": history_to_phase_points(history_y),
        "planned_path": planned_states_to_phase_points(chosen.planned_states) if chosen is not None else np.empty((0, 2), dtype=float),
        "planned_actions": np.asarray(chosen.planned_actions, dtype=float) if chosen is not None else np.empty((0,), dtype=float),
        "trajectory_param": float(chosen.trajectory_param) if chosen is not None and chosen.trajectory_param is not None else float("nan"),
        "rmse_history": np.asarray(rmse_history, dtype=float),
    }


def append_round_logs(
    logs: Dict[str, object],
    certified_depth: float,
    feasible_size: float,
    true_future_safe_count: float,
    interior_recall: float,
    interior_iou: float,
    false_certification_rate: float,
    boundary_miss_rate: float,
    dynamics_rmse: float,
    safety_set_recovery_recall: float,
    safety_set_recovery_precision: float,
    safety_set_recovery_iou: float,
    dead_end_free_recovery_recall: float,
    dead_end_free_recovery_precision: float,
    dead_end_free_recovery_iou: float,
    dead_end_ratio: float,
    instantaneous_dead_end_rate: float,
    epsilon_dead_end_selected: float,
    selected_dead_end: float,
    selected_future_safe: float,
    unsafe_transition: float,
    dead_end_safety_violation_rate: float,
    rollout_uncertainty_radius: float,
    chosen_action: float,
    no_feasible_action: float,
    executed_transition_count: float,
) -> None:
    logs["certified_depths"].append(certified_depth)
    logs["feasible_sizes"].append(feasible_size)
    logs["true_future_safe_counts"].append(true_future_safe_count)
    logs["interior_recalls"].append(interior_recall)
    logs["false_certification_rate"].append(false_certification_rate)
    logs["boundary_miss_rate"].append(boundary_miss_rate)
    logs["interior_recall_epsilon"].append(interior_recall)
    logs["interior_iou_epsilon"].append(interior_iou)
    logs["dynamics_rmse"].append(dynamics_rmse)
    logs["safety_set_recovery_recall"].append(safety_set_recovery_recall)
    logs["safety_set_recovery_precision"].append(safety_set_recovery_precision)
    logs["safety_set_recovery_iou"].append(safety_set_recovery_iou)
    logs["dead_end_free_recovery_recall"].append(dead_end_free_recovery_recall)
    logs["dead_end_free_recovery_precision"].append(dead_end_free_recovery_precision)
    logs["dead_end_free_recovery_iou"].append(dead_end_free_recovery_iou)
    logs["dead_end_candidate_ratios"].append(dead_end_ratio)
    logs["instantaneous_dead_end_rate"].append(instantaneous_dead_end_rate)
    logs["epsilon_dead_end_selected"].append(epsilon_dead_end_selected)
    logs["selected_dead_end"].append(selected_dead_end)
    logs["selected_future_safe"].append(selected_future_safe)
    logs["unsafe_transitions"].append(unsafe_transition)
    logs["dead_end_safety_violation_rate"].append(dead_end_safety_violation_rate)
    logs["rollout_uncertainty_radius"].append(rollout_uncertainty_radius)
    logs["chosen_actions"].append(chosen_action)
    logs["no_feasible_action"].append(no_feasible_action)
    logs["executed_transition_count"].append(executed_transition_count)


def active_round_count(logs: Dict[str, object]) -> int:
    actions = np.asarray(logs["chosen_actions"], dtype=float)
    return int(np.sum(~np.isnan(actions)))


def pad_remaining_rounds(logs: Dict[str, object], total_rounds: int) -> None:
    while len(logs["certified_depths"]) < total_rounds:
        append_round_logs(
            logs=logs,
            certified_depth=0.0,
            feasible_size=0.0,
            true_future_safe_count=0.0,
            interior_recall=0.0,
            interior_iou=0.0,
            false_certification_rate=0.0,
            boundary_miss_rate=0.0,
            dynamics_rmse=0.0,
            safety_set_recovery_recall=0.0,
            safety_set_recovery_precision=0.0,
            safety_set_recovery_iou=0.0,
            dead_end_free_recovery_recall=0.0,
            dead_end_free_recovery_precision=0.0,
            dead_end_free_recovery_iou=0.0,
            dead_end_ratio=0.0,
            instantaneous_dead_end_rate=0.0,
            epsilon_dead_end_selected=0.0,
            selected_dead_end=0.0,
            selected_future_safe=0.0,
            unsafe_transition=0.0,
            dead_end_safety_violation_rate=0.0,
            rollout_uncertainty_radius=0.0,
            chosen_action=float("nan"),
            no_feasible_action=0.0,
            executed_transition_count=0.0,
        )


def run_trial(
    planner_type: str,
    seed: int,
    kernel_kind: str,
    T: int,
    fa_horizon: int,
    fa_beam_width: int,
    fa_continuation_policy: str,
    fa_random_horizon: int,
    pointwise_horizon: int,
    safe_exploration_horizon: int,
    salnx_horizon: int,
    tebbe_horizon: int,
    nominal_mpc_horizon: int,
    control_target: float,
    control_q_y: float,
    control_q_velocity: float,
    control_r_u: float,
    control_r_delta_u: float,
    control_terminal_weight: float,
    control_beam_width: int,
    active_mpc_information_weight: float,
    nominal_mpc_safety_margin: float,
    n_init: int,
    epsilon_margin: float,
    recovery_eval_interval: int,
    evaluation_horizon: int,
    confidence_schedule: ConfidenceSchedule,
    fa_lx: float,
    fa_ly: float,
    fa_lf: float,
    fa_lf_quantile: float,
    fa_lf_estimator: str,
    fa_lf_scale: float,
    fa_lf_cap: float,
    fa_l_ell: float,
    fa_l_ell_quantile: float,
    fa_l_ell_scale: float,
    fa_buffer_weight: float,
    alpha_safety: float,
    safe_exploration_alpha: float,
    salnx_criterion: str,
    salnx_joint_mc_samples: int,
    tebbe_alpha: float,
    tebbe_criterion: str,
    tebbe_confidence_delta: float,
    tebbe_sample_start: int,
    tebbe_sample_stages: int,
    tebbe_endpoint_candidates: int,
    tebbe_legacy_mc_prefix: bool,
    tebbe_local_refinement: bool,
    n_eval: int,
    snapshot_rounds: Sequence[int],
    snapshot_grid_size: int,
    snapshot_u_slice: float,
    env_kind: str,
    env_y_max: float,
    env_noise_std: float,
    safety_noise_std: float,
    runaway_center: float,
    runaway_halfwidth: float,
    runaway_strength: float,
    runaway_velocity_threshold: float,
    funnel_center: float,
    funnel_halfwidth: float,
    funnel_runaway_strength: float,
    funnel_velocity_threshold: float,
    funnel_brake_fade: float,
    lattice_start: float,
    lattice_period: float,
    lattice_count: int,
    lattice_halfwidth: float,
    lattice_tailwidth: float,
    lattice_runaway_strength: float,
    lattice_tail_runaway_multiplier: float,
    lattice_velocity_threshold: float,
    lattice_brake_fade: float,
    lattice_tail_brake_fade: float,
    smooth_switches: bool,
    switch_sharpness: float,
    goal_y_min: float,
    goal_y_max: float,
    fail_y: float,
    failure_absorbing: bool,
    episodic_reset_period: int,
    init_y_range: Tuple[float, float],
    init_velocity_range: Tuple[float, float],
    start_y_tm1: float,
    start_y_t: float,
    show_progress: bool = False,
    progress_desc: str = "",
    heartbeat_interval: int = 0,
    heartbeat_label: str = "",
    progress_position: int = 0,
    metric_eval_points: int = 0,
) -> Dict[str, object]:
    trial_start_perf = time.perf_counter()
    peak_rss_sampler = _PeakRSSSampler()
    peak_rss_sampler.start()
    rng = np.random.default_rng(seed)
    env = NARXDoubleIntegratorEnv(
        kind=env_kind,
        y_max=env_y_max,
        sigma_eps=env_noise_std,
        sigma_g=safety_noise_std,
        runaway_center=runaway_center,
        runaway_halfwidth=runaway_halfwidth,
        runaway_strength=runaway_strength,
        runaway_velocity_threshold=runaway_velocity_threshold,
        funnel_center=funnel_center,
        funnel_halfwidth=funnel_halfwidth,
        funnel_runaway_strength=funnel_runaway_strength,
        funnel_velocity_threshold=funnel_velocity_threshold,
        funnel_brake_fade=funnel_brake_fade,
        lattice_start=lattice_start,
        lattice_period=lattice_period,
        lattice_count=lattice_count,
        lattice_halfwidth=lattice_halfwidth,
        lattice_tailwidth=lattice_tailwidth,
        lattice_runaway_strength=lattice_runaway_strength,
        lattice_tail_runaway_multiplier=lattice_tail_runaway_multiplier,
        lattice_velocity_threshold=lattice_velocity_threshold,
        lattice_brake_fade=lattice_brake_fade,
        lattice_tail_brake_fade=lattice_tail_brake_fade,
        smooth_switches=bool(smooth_switches),
        switch_sharpness=float(switch_sharpness),
        goal_y_min=goal_y_min,
        goal_y_max=goal_y_max,
        fail_y=fail_y,
        failure_absorbing=failure_absorbing,
        rng=np.random.default_rng(seed),
    )
    actions = env.candidate_actions()

    X_train, y_train, g_train = generate_initial_safe_data(
        env=env,
        rng=rng,
        n_init=n_init,
        actions=actions,
        y_range=init_y_range,
        velocity_range=init_velocity_range,
    )

    kernel_cfg = KernelConfig(kind=kernel_kind, variance=1.0, length_scale=1.0)
    dyn_model = DynamicsGP(kernel_cfg=kernel_cfg, noise_std=max(env.sigma_eps, 1e-3))
    safe_model = SafetyGP(kernel_cfg=kernel_cfg, noise_std=max(env.sigma_g, 1e-4))
    horizon_by_planner = {
        "future": int(fa_horizon),
        "myopic_fa_sal": 1,
        "fa_random": int(fa_random_horizon),
        "pointwise": int(pointwise_horizon),
        "safe_exploration": int(safe_exploration_horizon),
        "salnx": int(salnx_horizon),
        "tebbe_abm": int(tebbe_horizon),
        "nominal_mpc": int(nominal_mpc_horizon),
        "al_mpc": int(nominal_mpc_horizon),
    }
    horizon = int(horizon_by_planner[planner_type])
    evaluation_horizon = int(evaluation_horizon)

    def make_cfg(h: int) -> PlannerConfig:
        return PlannerConfig(
            horizon=int(h),
            Lx=fa_lx,
            Ly=fa_ly,
            Lf=fa_lf,
            L_ell=fa_l_ell,
            fa_beam_width=fa_beam_width,
            fa_continuation_policy=str(fa_continuation_policy),
            fa_continuation_seed=70000 + int(seed),
            fa_safety_buffer_weight=fa_buffer_weight,
            alpha_safety=alpha_safety,
            safe_exploration_alpha=safe_exploration_alpha,
            salnx_criterion=salnx_criterion,
            salnx_joint_mc_samples=salnx_joint_mc_samples,
            tebbe_alpha=tebbe_alpha,
            tebbe_criterion=tebbe_criterion,
            tebbe_confidence_delta=tebbe_confidence_delta,
            tebbe_sample_start=tebbe_sample_start,
            tebbe_sample_stages=tebbe_sample_stages,
            tebbe_endpoint_candidates=tebbe_endpoint_candidates,
            tebbe_legacy_mc_prefix=tebbe_legacy_mc_prefix,
            tebbe_local_refinement=tebbe_local_refinement,
            control_target=control_target,
            control_q_y=control_q_y,
            control_q_velocity=control_q_velocity,
            control_r_u=control_r_u,
            control_r_delta_u=control_r_delta_u,
            control_terminal_weight=control_terminal_weight,
            control_beam_width=control_beam_width,
            active_mpc_information_weight=active_mpc_information_weight,
            nominal_mpc_safety_margin=nominal_mpc_safety_margin,
        )

    cfg = make_cfg(horizon)
    X_eval, y_eval, safe_eval = build_evaluation_dataset(
        env=env,
        rng=np.random.default_rng(50000 + seed),
        actions=actions,
        n_eval=n_eval,
    )
    if int(metric_eval_points) > 0:
        metric_count = min(int(metric_eval_points), int(len(X_eval)))
        X_eval = X_eval[:metric_count]
        y_eval = y_eval[:metric_count]
        safe_eval = safe_eval[:metric_count]
    estimated_lf = estimate_dynamics_lipschitz_constant(
        env=env,
        y_min=confidence_schedule.y_min,
        y_max=confidence_schedule.y_max,
        y_grid_points=confidence_schedule.y_grid_points,
        quantile=fa_lf_quantile,
        estimator=fa_lf_estimator,
    )
    scaled_lf = float(fa_lf_scale) * float(estimated_lf)
    cfg.Lf = min(max(float(fa_lf), scaled_lf), float(fa_lf_cap))

    def build_planner(planner_key: str, planner_cfg: PlannerConfig):
        if planner_key == "future":
            return FutureAwarePlanner(env, dyn_model, safe_model, planner_cfg), "FA-SAL"
        if planner_key == "myopic_fa_sal":
            myopic_cfg = PlannerConfig(**{**planner_cfg.__dict__, "horizon": 1})
            return FutureAwarePlanner(env, dyn_model, safe_model, myopic_cfg), "Myopic FA-SAL (m=1)"
        if planner_key == "fa_random":
            return FARandomPlanner(env, dyn_model, safe_model, planner_cfg, rng=np.random.default_rng(90000 + seed)), "FA-SAL-Random"
        if planner_key == "pointwise":
            return PointwiseSafePlanner(env, dyn_model, safe_model, planner_cfg), "Pointwise"
        if planner_key == "safe_exploration":
            return SafeExplorationPlanner(env, dyn_model, safe_model, planner_cfg), "Safe Exploration"
        if planner_key == "salnx":
            return SALNXPlanner(env, dyn_model, safe_model, planner_cfg), "SAL-NX"
        if planner_key == "tebbe_abm":
            return TebbeABMPlanner(env, dyn_model, safe_model, planner_cfg), "Tebbe-ABM"
        if planner_key == "nominal_mpc":
            return NominalMPCPlanner(env, dyn_model, safe_model, planner_cfg), "Nominal MPC"
        if planner_key == "al_mpc":
            return ActiveLearningMPCPlanner(env, dyn_model, safe_model, planner_cfg), "AL-MPC"
        raise ValueError("Unsupported planner_type.")

    planner, method_name = build_planner(planner_type, cfg)
    metric_cfg = make_cfg(evaluation_horizon)
    metric_planner, _ = build_planner(planner_type, metric_cfg)

    y_tm1, y_t = float(start_y_tm1), float(start_y_t)
    u_prev = 0.0
    if not env.state_is_safe(y_t, y_tm1):
        raise RuntimeError("Initial trajectory state must be safe.")

    logs: Dict[str, object] = {
        "method": method_name,
        "seed": int(seed),
        "estimated_lf": float(estimated_lf),
        "effective_lf": float(cfg.Lf),
        "beta_f": [],
        "beta_g": [],
        "effective_l_ell": [],
        "training_sizes": [],
        "candidate_counts": [],
        "gp_fit_seconds": [],
        "planning_seconds": [],
        "evaluation_seconds": [],
        "round_total_seconds": [],
        "certified_depths": [],
        "feasible_sizes": [],
        "feasible_ratios": [],
        "true_future_safe_counts": [],
        "interior_recalls": [],
        "false_certification_rate": [],
        "boundary_miss_rate": [],
        "interior_recall_epsilon": [],
        "interior_iou_epsilon": [],
        "dynamics_rmse": [],
        "safety_set_recovery_recall": [],
        "safety_set_recovery_precision": [],
        "safety_set_recovery_iou": [],
        "dead_end_free_recovery_recall": [],
        "dead_end_free_recovery_precision": [],
        "dead_end_free_recovery_iou": [],
        "dead_end_candidate_ratios": [],
        "instantaneous_dead_end_rate": [],
        "epsilon_dead_end_selected": [],
        "selected_dead_end": [],
        "selected_future_safe": [],
        "unsafe_transitions": [],
        "dead_end_safety_violation_rate": [],
        "rollout_uncertainty_radius": [],
        "chosen_actions": [],
        "no_feasible_action": [],
        "executed_transition_count": [],
        "history_y": [y_tm1, y_t],
        "snapshots": [],
    }
    dead_end_count = 0.0
    executed_count = 0.0
    candidate_oracle_cache: Dict[
        Tuple[float, float, float, int, float], bool
    ] = _BoundedOracleCache(max_entries=100_000)
    viability_dp = (
        BackwardViabilityDP(
            env=env,
            actions=actions,
            horizon=evaluation_horizon,
            y_min=confidence_schedule.y_min,
            y_max=confidence_schedule.y_max,
            y_grid_points=confidence_schedule.y_grid_points,
            fallback_cache=candidate_oracle_cache,
        )
        if evaluation_horizon == 8
        else None
    )
    true_dead_end_free_eval = build_dead_end_free_dataset(
        env=env,
        X_eval=X_eval,
        horizon=evaluation_horizon,
        actions=actions,
        viability_dp=viability_dp,
    )
    last_recovery_metrics = {
        "safety_set_recovery_recall": 0.0,
        "safety_set_recovery_precision": 0.0,
        "safety_set_recovery_iou": 0.0,
        "dead_end_free_recovery_recall": 0.0,
        "dead_end_free_recovery_precision": 0.0,
        "dead_end_free_recovery_iou": 0.0,
    }
    recovery_eval_interval = max(1, int(recovery_eval_interval))
    if evaluation_horizon == 8:
        recovery_eval_interval = 5
    theorem_eval_interval = 5 if evaluation_horizon == 8 else 1
    last_theorem_metrics: Optional[Dict[str, float]] = None
    last_dead_end_ratio = 0.0

    round_iterator = range(1, T + 1)
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            round_iterator,
            desc=progress_desc or f"{method_name} rounds",
            leave=False,
            dynamic_ncols=True,
            position=max(0, int(progress_position)),
            mininterval=1.0,
        )
        round_iterator = progress_bar

    for round_idx in round_iterator:
        round_start_perf = time.perf_counter()
        if episodic_reset_period > 0 and round_idx > 1 and (round_idx - 1) % episodic_reset_period == 0:
            y_tm1, y_t = float(start_y_tm1), float(start_y_t)
            u_prev = 0.0
            logs["history_y"].append(float(y_tm1))
            logs["history_y"].append(float(y_t))

        cfg.beta_f = confidence_schedule.beta_f(round_idx, env)
        cfg.beta_g = confidence_schedule.beta_g(round_idx, env)

        training_size_this_round = int(X_train.shape[0])
        fit_start_perf = time.perf_counter()
        dyn_model.fit(X_train, y_train)
        safe_model.fit(X_train, g_train)
        gp_fit_seconds = float(time.perf_counter() - fit_start_perf)
        evaluation_start_perf = time.perf_counter()
        z_current = env.make_regressor(y_t, y_tm1, u_prev)
        distance_to_current = np.linalg.norm(X_train - z_current.reshape(1, -1), axis=1)
        nearest_count = min(48, X_train.shape[0])
        nearest_idx = np.argsort(distance_to_current)[:nearest_count]
        points = np.vstack([z_current.reshape(1, -1), X_train[nearest_idx]])
        y_step = (
            float(confidence_schedule.y_max - confidence_schedule.y_min)
            / max(confidence_schedule.y_grid_points - 1, 1)
        )
        u_step = float(actions[1] - actions[0]) if len(actions) > 1 else 0.2
        estimated_l_ell = estimate_local_lcb_lipschitz_constant(
            safe_model=safe_model,
            beta_g=cfg.beta_g,
            env=env,
            points=points,
            y_min=confidence_schedule.y_min,
            y_max=confidence_schedule.y_max,
            y_step=y_step,
            u_step=u_step,
            quantile=fa_l_ell_quantile,
        )
        scaled_l_ell = float(fa_l_ell_scale) * float(estimated_l_ell)
        if np.isfinite(scaled_l_ell) and scaled_l_ell > 0.0:
            cfg.L_ell = float(scaled_l_ell)
        else:
            cfg.L_ell = float(fa_l_ell)
        metric_cfg.beta_f = cfg.beta_f
        metric_cfg.beta_g = cfg.beta_g
        metric_cfg.Lf = cfg.Lf
        metric_cfg.L_ell = cfg.L_ell
        model_metrics = evaluate_model_metrics(
            dyn_model=dyn_model,
            X_eval=X_eval,
            y_eval=y_eval,
        )
        if round_idx == 1 or ((round_idx - 1) % recovery_eval_interval == 0):
            if hasattr(metric_planner, "set_trajectory_anchor"):
                metric_planner.set_trajectory_anchor(float(u_prev))
            safety_recovery_metrics = evaluate_safety_set_recovery_metrics(
                safe_model=safe_model,
                cfg=metric_cfg,
                X_eval=X_eval,
                true_safe=safe_eval,
            )
            dead_end_recovery_metrics = evaluate_dead_end_free_recovery_metrics(
                planner=metric_planner,
                X_eval=X_eval,
                true_dead_end_free=true_dead_end_free_eval,
                actions=actions,
            )
            last_recovery_metrics = {
                **safety_recovery_metrics,
                **dead_end_recovery_metrics,
            }
        recovery_metrics = last_recovery_metrics
        preplanning_evaluation_seconds = float(
            time.perf_counter() - evaluation_start_perf
        )

        if hasattr(planner, "set_trajectory_anchor"):
            planner.set_trajectory_anchor(float(u_prev))
        planning_start_perf = time.perf_counter()
        selected_action, chosen, infos = planner.select_action(y_t, y_tm1, actions)
        planning_seconds = float(time.perf_counter() - planning_start_perf)
        evaluation_start_perf = time.perf_counter()
        if hasattr(metric_planner, "set_trajectory_anchor"):
            metric_planner.set_trajectory_anchor(float(u_prev))
        metric_infos = metric_planner.candidate_infos(y_t, y_tm1, actions)
        feasible_actions = {candidate_id(info) for info in metric_infos if info.feasible_full_horizon}
        evaluate_theorem_this_round = (
            last_theorem_metrics is None
            or (round_idx - 1) % theorem_eval_interval == 0
        )
        if evaluate_theorem_this_round:
            oracle_by_u0: Dict[float, Dict[str, float]] = {}
            for info in metric_infos:
                action_key = round(float(info.u0), 12)
                if action_key not in oracle_by_u0:
                    oracle_by_u0[action_key] = evaluate_selected_action_dead_end_oracle(
                        env=env,
                        y_t=y_t,
                        y_tm1=y_tm1,
                        selected_action=float(info.u0),
                        horizon=evaluation_horizon,
                        actions=actions,
                        epsilon=epsilon_margin,
                        cache=candidate_oracle_cache,
                        viability_dp=viability_dp,
                    )
            oracle_data = {
                candidate_id(info): oracle_by_u0[round(float(info.u0), 12)]
                for info in metric_infos
            }
            last_dead_end_ratio = float(
                np.mean([stats["dead_end"] for stats in oracle_data.values()])
            )
            last_theorem_metrics = evaluate_theorem_metrics(
                feasible_actions=sorted(feasible_actions),
                oracle_data=oracle_data,
            )
        assert last_theorem_metrics is not None
        theorem_metrics = last_theorem_metrics
        dead_end_ratio = last_dead_end_ratio
        interior_recall = float(theorem_metrics["interior_recall_epsilon"])
        interior_iou = float(theorem_metrics["interior_iou_epsilon"])
        rollout_uncertainty_radius = float(max((info.max_rollout_std for info in infos), default=0.0))
        evaluation_seconds = preplanning_evaluation_seconds + float(
            time.perf_counter() - evaluation_start_perf
        )

        fallback_used = selected_action is None
        if fallback_used:
            selected_action, chosen = choose_fallback_candidate(infos)

            if selected_action is None or chosen is None:
                selected_action = float(np.min(actions))
                z_fallback = env.make_regressor(y_t, y_tm1, selected_action)
                chosen = CandidateInfo(
                    u0=selected_action,
                    certified_depth=0,
                    feasible_full_horizon=False,
                    acquisition=float("-inf"),
                    pointwise_lcb=float("nan"),
                    planned_actions=[selected_action],
                    planned_states=[z_fallback.copy()],
                    planned_outputs=[],
                    rollout_stds=[],
                    max_rollout_std=0.0,
                    trajectory_safety_prob=0.0,
                    min_safety_slack=float("-inf"),
                    trajectory_param=None,
                )

        if chosen is None:
            raise RuntimeError("Planner returned an action without its chosen candidate info.")
        if round_idx in snapshot_rounds:
            logs["snapshots"].append(
                make_snapshot(
                    env=env,
                    safe_model=safe_model,
                    cfg=cfg,
                    chosen=chosen,
                    history_y=logs["history_y"],
                    rmse_history=logs["dynamics_rmse"] + [float(model_metrics["dynamics_rmse"])],
                    round_idx=round_idx,
                    x_min=confidence_schedule.y_min,
                    x_max=confidence_schedule.y_max,
                    grid_size=snapshot_grid_size,
                    u_slice=snapshot_u_slice,
                    )
                )

        execute_full_trajectory = (
            planner_type in {"future", "fa_random", "salnx", "tebbe_abm"}
        ) and (not fallback_used)

        action_sequence = list(chosen.planned_actions) if execute_full_trajectory else [float(selected_action)]
        current_y_tm1 = float(y_tm1)
        current_y_t = float(y_t)
        last_action = float(u_prev)
        trajectory_unsafe = False
        goal_reached = False
        executed_transition_count = 0.0

        trajectory_dead_end_hit = False
        trajectory_epsilon_dead_end_hit = False
        trajectory_future_safe_all = True
        trajectory_dead_end_point_count = 0.0

        for action in action_sequence:
            z_exec = env.make_regressor(current_y_t, current_y_tm1, float(action))

            immediate_margin = float(env.safety_value(z_exec))
            immediate_safe = bool(immediate_margin >= 0.0)
            point_future_safe = (
                viability_dp.query(z_exec, evaluation_horizon, 0.0)
                if viability_dp is not None
                else oracle_future_viable_at_threshold(
                    env=env,
                    z=z_exec,
                    horizon=evaluation_horizon,
                    actions=actions,
                    threshold=0.0,
                    cache=candidate_oracle_cache,
                )
            )
            point_dead_end = bool(immediate_safe and not point_future_safe)
            point_epsilon_dead_end = False
            epsilon_threshold = float(epsilon_margin)
            if immediate_margin >= epsilon_threshold:
                strict_negative_threshold = float(
                    np.nextafter(-epsilon_threshold, np.inf)
                )
                if point_future_safe and 0.0 >= strict_negative_threshold:
                    above_negative_epsilon = True
                elif not point_future_safe and 0.0 <= strict_negative_threshold:
                    above_negative_epsilon = False
                else:
                    above_negative_epsilon = (
                        viability_dp.query(
                            z_exec,
                            evaluation_horizon,
                            strict_negative_threshold,
                        )
                        if viability_dp is not None
                        else oracle_future_viable_at_threshold(
                            env=env,
                            z=z_exec,
                            horizon=evaluation_horizon,
                            actions=actions,
                            threshold=strict_negative_threshold,
                            cache=candidate_oracle_cache,
                        )
                    )
                point_epsilon_dead_end = not above_negative_epsilon

            trajectory_dead_end_hit = trajectory_dead_end_hit or point_dead_end
            trajectory_epsilon_dead_end_hit = trajectory_epsilon_dead_end_hit or point_epsilon_dead_end
            trajectory_future_safe_all = trajectory_future_safe_all and point_future_safe
            trajectory_dead_end_point_count += float(point_dead_end)

            unsafe_before_step = not env.is_safe(z_exec)
            y_next = env.step(z_exec, noise=True)
            z_next_probe = env.make_regressor(float(y_next), current_y_t, float(action))
            unsafe_after_step = not env.is_safe(z_next_probe)

            if unsafe_after_step and not unsafe_before_step:
                y_next = env.failure_observation(z_next_probe, noise=True)

            transition_unsafe_now = bool(unsafe_before_step or unsafe_after_step)
            trajectory_unsafe = trajectory_unsafe or transition_unsafe_now

            X_train = np.vstack([X_train, z_exec.reshape(1, -1)])
            y_train = np.append(y_train, y_next)
            g_train = np.append(g_train, env.observe_safety(z_exec, noise=True))
            logs["history_y"].append(float(y_next))

            last_action = float(action)
            goal_reached = goal_reached or env.goal_reached(y_next)
            executed_transition_count += 1.0

            if transition_unsafe_now:
                current_y_tm1, current_y_t = float(start_y_tm1), float(start_y_t)
                last_action = 0.0
                logs["history_y"].append(float(current_y_tm1))
                logs["history_y"].append(float(current_y_t))
                break

            current_y_tm1, current_y_t = current_y_t, y_next

        executed_count += 1.0
        dead_end_selected = float(trajectory_dead_end_hit)
        epsilon_dead_end_selected = float(trajectory_epsilon_dead_end_hit)
        selected_future_safe = float(trajectory_future_safe_all)
        dead_end_count += dead_end_selected

        next_state_safe = float(not trajectory_unsafe)

        append_round_logs(
            logs=logs,
            certified_depth=float(chosen.certified_depth),
            feasible_size=float(len(feasible_actions)),
            true_future_safe_count=float(theorem_metrics["true_future_safe_count"]),
            interior_recall=float(interior_recall),
            interior_iou=float(interior_iou),
            false_certification_rate=float(theorem_metrics["false_certification_rate"]),
            boundary_miss_rate=float(theorem_metrics["boundary_miss_rate"]),
            dynamics_rmse=float(model_metrics["dynamics_rmse"]),
            safety_set_recovery_recall=float(recovery_metrics["safety_set_recovery_recall"]),
            safety_set_recovery_precision=float(recovery_metrics["safety_set_recovery_precision"]),
            safety_set_recovery_iou=float(recovery_metrics["safety_set_recovery_iou"]),
            dead_end_free_recovery_recall=float(recovery_metrics["dead_end_free_recovery_recall"]),
            dead_end_free_recovery_precision=float(recovery_metrics["dead_end_free_recovery_precision"]),
            dead_end_free_recovery_iou=float(recovery_metrics["dead_end_free_recovery_iou"]),
            dead_end_ratio=dead_end_ratio,
            instantaneous_dead_end_rate=dead_end_selected,
            epsilon_dead_end_selected=epsilon_dead_end_selected,
            selected_dead_end=dead_end_selected,
            selected_future_safe=selected_future_safe,
            unsafe_transition=float(int(trajectory_unsafe)),
            dead_end_safety_violation_rate=float(dead_end_count / executed_count),
            rollout_uncertainty_radius=rollout_uncertainty_radius,
            chosen_action=float(selected_action),
            no_feasible_action=float(int(fallback_used)),
            executed_transition_count=executed_transition_count,
        )
        logs["feasible_ratios"].append(
            float(len(feasible_actions) / len(metric_infos)) if metric_infos else 0.0
        )
        logs["beta_f"].append(float(cfg.beta_f))
        logs["beta_g"].append(float(cfg.beta_g))
        logs["effective_l_ell"].append(float(cfg.L_ell))
        logs["training_sizes"].append(float(training_size_this_round))
        logs["candidate_counts"].append(float(len(metric_infos)))
        logs["gp_fit_seconds"].append(gp_fit_seconds)
        logs["planning_seconds"].append(planning_seconds)
        logs["evaluation_seconds"].append(evaluation_seconds)
        logs["round_total_seconds"].append(
            float(time.perf_counter() - round_start_perf)
        )
        y_tm1, y_t = current_y_tm1, current_y_t
        u_prev = last_action

        heartbeat_interval = max(0, int(heartbeat_interval))
        if heartbeat_interval and (
            round_idx == 1
            or round_idx == T
            or round_idx % heartbeat_interval == 0
        ):
            print(
                f"Progress {heartbeat_label or method_name} "
                f"round={round_idx}/{T} "
                f"elapsed={time.perf_counter() - trial_start_perf:.1f}s "
                f"last_round={logs['round_total_seconds'][-1]:.2f}s "
                f"gp={gp_fit_seconds:.2f}s "
                f"planning={planning_seconds:.2f}s "
                f"evaluation={evaluation_seconds:.2f}s",
                flush=True,
            )

        if env.kind == "goal_funnel" and (goal_reached or not next_state_safe):
            y_tm1, y_t = float(start_y_tm1), float(start_y_t)
            u_prev = 0.0
            logs["history_y"].append(float(y_tm1))
            logs["history_y"].append(float(y_t))
            continue


    if progress_bar is not None:
        progress_bar.close()
    pad_remaining_rounds(logs, T)
    extra_series = (
        "feasible_ratios",
        "beta_f",
        "beta_g",
        "effective_l_ell",
        "training_sizes",
        "candidate_counts",
        "gp_fit_seconds",
        "planning_seconds",
        "evaluation_seconds",
        "round_total_seconds",
    )
    for key in extra_series:
        while len(logs[key]) < T:
            logs[key].append(0.0)
    logs["runtime_seconds"] = float(time.perf_counter() - trial_start_perf)
    logs["peak_rss_mib"] = peak_rss_sampler.stop_mib()
    return logs


def summarize_trials(results: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    keys = [
        "certified_depths",
        "feasible_sizes",
        "feasible_ratios",
        "beta_f",
        "beta_g",
        "effective_l_ell",
        "training_sizes",
        "candidate_counts",
        "gp_fit_seconds",
        "planning_seconds",
        "evaluation_seconds",
        "round_total_seconds",
        "true_future_safe_counts",
        "interior_recalls",
        "false_certification_rate",
        "boundary_miss_rate",
        "interior_recall_epsilon",
        "interior_iou_epsilon",
        "dynamics_rmse",
        "safety_set_recovery_recall",
        "safety_set_recovery_precision",
        "safety_set_recovery_iou",
        "dead_end_free_recovery_recall",
        "dead_end_free_recovery_precision",
        "dead_end_free_recovery_iou",
        "dead_end_candidate_ratios",
        "instantaneous_dead_end_rate",
        "epsilon_dead_end_selected",
        "selected_dead_end",
        "selected_future_safe",
        "unsafe_transitions",
        "dead_end_safety_violation_rate",
        "rollout_uncertainty_radius",
        "no_feasible_action",
        "executed_transition_count",
    ]
    summary: Dict[str, np.ndarray] = {}
    for key in keys:
        stacked = np.asarray([trial[key] for trial in results], dtype=float)
        summary[f"{key}_mean"] = stacked.mean(axis=0)
        summary[f"{key}_std"] = stacked.std(axis=0)
        summary[f"{key}_sum_mean"] = np.array([stacked.sum(axis=1).mean()])
        summary[f"{key}_sum_std"] = np.array([stacked.sum(axis=1).std()])

    active_rounds = np.asarray([active_round_count(trial) for trial in results], dtype=float)
    summary["active_rounds_mean"] = np.array([active_rounds.mean()])
    summary["active_rounds_std"] = np.array([active_rounds.std()])
    runtime_seconds = np.asarray([trial["runtime_seconds"] for trial in results], dtype=float)
    no_feasible_rates = np.asarray(
        [
            np.sum(np.asarray(trial["no_feasible_action"], dtype=float))
            / max(1.0, float(active_rounds[trial_idx]))
            for trial_idx, trial in enumerate(results)
        ],
        dtype=float,
    )
    summary["runtime_seconds_mean"] = np.array([runtime_seconds.mean()])
    summary["runtime_seconds_std"] = np.array([runtime_seconds.std()])
    summary["no_feasible_rate_mean"] = np.array([no_feasible_rates.mean()])
    summary["no_feasible_rate_std"] = np.array([no_feasible_rates.std()])

    safety_iou_series = np.asarray([trial["safety_set_recovery_iou"] for trial in results], dtype=float)
    iou_series = np.asarray([trial["dead_end_free_recovery_iou"] for trial in results], dtype=float)
    safety_recall_series = np.asarray([trial["safety_set_recovery_recall"] for trial in results], dtype=float)
    dead_end_recall_series = np.asarray([trial["dead_end_free_recovery_recall"] for trial in results], dtype=float)
    chosen_actions = np.asarray([trial["chosen_actions"] for trial in results], dtype=float)
    active_mask = ~np.isnan(chosen_actions)
    active_safety_iou_mean = []
    active_safety_iou_std = []
    active_iou_mean = []
    active_iou_std = []
    active_safety_recall_mean = []
    active_safety_recall_std = []
    active_dead_end_recall_mean = []
    active_dead_end_recall_std = []
    for round_idx in range(iou_series.shape[1]):
        safety_iou_round_values = safety_iou_series[active_mask[:, round_idx], round_idx]
        round_values = iou_series[active_mask[:, round_idx], round_idx]
        safety_recall_round_values = safety_recall_series[active_mask[:, round_idx], round_idx]
        dead_end_recall_round_values = dead_end_recall_series[active_mask[:, round_idx], round_idx]
        if safety_iou_round_values.size == 0:
            active_safety_iou_mean.append(np.nan)
            active_safety_iou_std.append(np.nan)
        else:
            active_safety_iou_mean.append(float(np.mean(safety_iou_round_values)))
            active_safety_iou_std.append(float(np.std(safety_iou_round_values)))
        if round_values.size == 0:
            active_iou_mean.append(np.nan)
            active_iou_std.append(np.nan)
        else:
            active_iou_mean.append(float(np.mean(round_values)))
            active_iou_std.append(float(np.std(round_values)))
        if safety_recall_round_values.size == 0:
            active_safety_recall_mean.append(np.nan)
            active_safety_recall_std.append(np.nan)
        else:
            active_safety_recall_mean.append(float(np.mean(safety_recall_round_values)))
            active_safety_recall_std.append(float(np.std(safety_recall_round_values)))
        if dead_end_recall_round_values.size == 0:
            active_dead_end_recall_mean.append(np.nan)
            active_dead_end_recall_std.append(np.nan)
        else:
            active_dead_end_recall_mean.append(float(np.mean(dead_end_recall_round_values)))
            active_dead_end_recall_std.append(float(np.std(dead_end_recall_round_values)))
    summary["safety_set_recovery_iou_active_mean"] = np.asarray(active_safety_iou_mean, dtype=float)
    summary["safety_set_recovery_iou_active_std"] = np.asarray(active_safety_iou_std, dtype=float)
    summary["dead_end_free_recovery_iou_active_mean"] = np.asarray(active_iou_mean, dtype=float)
    summary["dead_end_free_recovery_iou_active_std"] = np.asarray(active_iou_std, dtype=float)
    summary["safety_set_recovery_recall_active_mean"] = np.asarray(active_safety_recall_mean, dtype=float)
    summary["safety_set_recovery_recall_active_std"] = np.asarray(active_safety_recall_std, dtype=float)
    summary["dead_end_free_recovery_recall_active_mean"] = np.asarray(active_dead_end_recall_mean, dtype=float)
    summary["dead_end_free_recovery_recall_active_std"] = np.asarray(active_dead_end_recall_std, dtype=float)

    penalized_safety_iou_series = np.where(active_mask, safety_iou_series, 0.0)
    penalized_dead_end_iou_series = np.where(active_mask, iou_series, 0.0)
    penalized_safety_recall_series = np.where(active_mask, safety_recall_series, 0.0)
    penalized_dead_end_recall_series = np.where(active_mask, dead_end_recall_series, 0.0)
    rmse_series = np.asarray([trial["dynamics_rmse"] for trial in results], dtype=float)
    finite_rmse = rmse_series[np.isfinite(rmse_series)]
    rmse_penalty = float(np.max(finite_rmse)) if finite_rmse.size > 0 else 0.0
    penalized_rmse_series = np.where(active_mask, rmse_series, rmse_penalty)

    summary["safety_set_recovery_iou_penalized_mean"] = penalized_safety_iou_series.mean(axis=0)
    summary["safety_set_recovery_iou_penalized_std"] = penalized_safety_iou_series.std(axis=0)
    summary["dead_end_free_recovery_iou_penalized_mean"] = penalized_dead_end_iou_series.mean(axis=0)
    summary["dead_end_free_recovery_iou_penalized_std"] = penalized_dead_end_iou_series.std(axis=0)
    summary["safety_set_recovery_recall_penalized_mean"] = penalized_safety_recall_series.mean(axis=0)
    summary["safety_set_recovery_recall_penalized_std"] = penalized_safety_recall_series.std(axis=0)
    summary["dead_end_free_recovery_recall_penalized_mean"] = penalized_dead_end_recall_series.mean(axis=0)
    summary["dead_end_free_recovery_recall_penalized_std"] = penalized_dead_end_recall_series.std(axis=0)
    summary["dynamics_rmse_penalized_mean"] = penalized_rmse_series.mean(axis=0)
    summary["dynamics_rmse_penalized_std"] = penalized_rmse_series.std(axis=0)

    unsafe_series = np.asarray([trial["unsafe_transitions"] for trial in results], dtype=float)
    cumulative_svr_series = []
    for trial_idx, trial in enumerate(results):
        actions = np.asarray(trial["chosen_actions"], dtype=float)
        unsafe = unsafe_series[trial_idx]
        cumulative = np.full_like(unsafe, np.nan, dtype=float)
        executed_count = 0.0
        unsafe_count = 0.0
        for round_idx, action in enumerate(actions):
            if np.isnan(action):
                continue
            executed_count += 1.0
            unsafe_count += float(unsafe[round_idx])
            cumulative[round_idx] = unsafe_count / executed_count
        cumulative_svr_series.append(cumulative)
    cumulative_svr_series = np.asarray(cumulative_svr_series, dtype=float)

    svr_mean = []
    svr_std = []
    for round_idx in range(cumulative_svr_series.shape[1]):
        round_values = cumulative_svr_series[:, round_idx]
        round_values = round_values[np.isfinite(round_values)]
        if round_values.size == 0:
            svr_mean.append(np.nan)
            svr_std.append(np.nan)
        else:
            svr_mean.append(float(np.mean(round_values)))
            svr_std.append(float(np.std(round_values)))
    summary["safety_violation_rate_mean"] = np.asarray(svr_mean, dtype=float)
    summary["safety_violation_rate_std"] = np.asarray(svr_std, dtype=float)

    active_final_cert_depth = []
    active_final_recall = []
    active_final_false_certification_rate = []
    active_final_boundary_miss_rate = []
    active_final_rmse = []
    active_final_safety_set_recovery_recall = []
    active_final_safety_set_recovery_iou = []
    active_final_dead_end_free_recovery_recall = []
    active_final_dead_end_free_recovery_iou = []
    active_final_dead_end_safety_violation_rate = []
    active_final_safety_violation_rate = []
    for trial in results:
        active_rounds_this_trial = active_round_count(trial)
        if active_rounds_this_trial > 0:
            active_final_cert_depth.append(float(trial["certified_depths"][active_rounds_this_trial - 1]))
            active_final_recall.append(float(trial["interior_recalls"][active_rounds_this_trial - 1]))
            active_final_false_certification_rate.append(float(trial["false_certification_rate"][active_rounds_this_trial - 1]))
            active_final_boundary_miss_rate.append(float(trial["boundary_miss_rate"][active_rounds_this_trial - 1]))
            active_final_rmse.append(float(trial["dynamics_rmse"][active_rounds_this_trial - 1]))
            active_final_safety_set_recovery_recall.append(
                float(trial["safety_set_recovery_recall"][active_rounds_this_trial - 1])
            )
            active_final_safety_set_recovery_iou.append(
                float(trial["safety_set_recovery_iou"][active_rounds_this_trial - 1])
            )
            active_final_dead_end_free_recovery_recall.append(
                float(trial["dead_end_free_recovery_recall"][active_rounds_this_trial - 1])
            )
            active_final_dead_end_free_recovery_iou.append(
                float(trial["dead_end_free_recovery_iou"][active_rounds_this_trial - 1])
            )
            active_final_dead_end_safety_violation_rate.append(
                float(trial["dead_end_safety_violation_rate"][active_rounds_this_trial - 1])
            )
            active_final_safety_violation_rate.append(float(cumulative_svr_series[len(active_final_safety_violation_rate), active_rounds_this_trial - 1]))
        else:
            active_final_cert_depth.append(0.0)
            active_final_recall.append(0.0)
            active_final_false_certification_rate.append(0.0)
            active_final_boundary_miss_rate.append(0.0)
            active_final_rmse.append(0.0)
            active_final_safety_set_recovery_recall.append(0.0)
            active_final_safety_set_recovery_iou.append(0.0)
            active_final_dead_end_free_recovery_recall.append(0.0)
            active_final_dead_end_free_recovery_iou.append(0.0)
            active_final_dead_end_safety_violation_rate.append(0.0)
            active_final_safety_violation_rate.append(0.0)

    summary["active_final_certified_depth_mean"] = np.array([np.mean(active_final_cert_depth)])
    summary["active_final_certified_depth_std"] = np.array([np.std(active_final_cert_depth)])
    summary["active_final_recall_mean"] = np.array([np.mean(active_final_recall)])
    summary["active_final_recall_std"] = np.array([np.std(active_final_recall)])
    summary["active_final_false_certification_rate_mean"] = np.array([np.mean(active_final_false_certification_rate)])
    summary["active_final_false_certification_rate_std"] = np.array([np.std(active_final_false_certification_rate)])
    summary["active_final_boundary_miss_rate_mean"] = np.array([np.mean(active_final_boundary_miss_rate)])
    summary["active_final_boundary_miss_rate_std"] = np.array([np.std(active_final_boundary_miss_rate)])
    summary["active_final_rmse_mean"] = np.array([np.mean(active_final_rmse)])
    summary["active_final_rmse_std"] = np.array([np.std(active_final_rmse)])
    summary["active_final_safety_set_recovery_recall_mean"] = np.array(
        [np.mean(active_final_safety_set_recovery_recall)]
    )
    summary["active_final_safety_set_recovery_recall_std"] = np.array(
        [np.std(active_final_safety_set_recovery_recall)]
    )
    summary["active_final_safety_set_recovery_iou_mean"] = np.array(
        [np.mean(active_final_safety_set_recovery_iou)]
    )
    summary["active_final_safety_set_recovery_iou_std"] = np.array(
        [np.std(active_final_safety_set_recovery_iou)]
    )
    summary["active_final_dead_end_free_recovery_recall_mean"] = np.array(
        [np.mean(active_final_dead_end_free_recovery_recall)]
    )
    summary["active_final_dead_end_free_recovery_recall_std"] = np.array(
        [np.std(active_final_dead_end_free_recovery_recall)]
    )
    summary["active_final_dead_end_free_recovery_iou_mean"] = np.array(
        [np.mean(active_final_dead_end_free_recovery_iou)]
    )
    summary["active_final_dead_end_free_recovery_iou_std"] = np.array(
        [np.std(active_final_dead_end_free_recovery_iou)]
    )
    summary["active_final_dead_end_safety_violation_rate_mean"] = np.array(
        [np.mean(active_final_dead_end_safety_violation_rate)]
    )
    summary["active_final_dead_end_safety_violation_rate_std"] = np.array(
        [np.std(active_final_dead_end_safety_violation_rate)]
    )
    summary["active_final_safety_violation_rate_mean"] = np.array([np.mean(active_final_safety_violation_rate)])
    summary["active_final_safety_violation_rate_std"] = np.array([np.std(active_final_safety_violation_rate)])
    return summary
