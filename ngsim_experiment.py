import argparse
import csv
import itertools
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from blas_threads import configure_blas_threads

BLAS_THREAD_LIMITS = configure_blas_threads()

import numpy as np

from gp_models import DynamicsGP, KernelConfig, SafetyGP, normal_cdf_array
from ngsim_planning import (
    FARandomTrajectoryPlanner,
    FutureAwareTrajectoryPlanner,
    PlannerConfig,
    SALNXTrajectoryPlanner,
    TebbeABMTrajectoryPlanner,
)

try:
    from config_ngsim import NGSIM_CONFIG
except Exception:
    NGSIM_CONFIG = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


NGSIM_DATA_URL = "https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data"


def _config_value(section: str, field: str, fallback):
    if NGSIM_CONFIG is None:
        return fallback
    config_section = getattr(NGSIM_CONFIG, section, None)
    return getattr(config_section, field, fallback)


def _iter_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _method_config_prefix(method: str) -> str:
    return {
        "fa_sal": "fa_sal",
        "fa_random": "fa_random",
        "myopic_fa_sal": "fa_sal",
        "salnx": "salnx",
        "tebbe_abm": "tebbe",
    }.get(str(method), str(method))


def _format_sweep_value(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def _format_label_value(value: float) -> str:
    return f"{float(value):g}"


def _safe_filename_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_").lower()


def _effective_min_headway(args) -> float:
    effective = float(args.min_headway) - float(args.safety_headway_offset)
    if effective <= 0.0:
        raise ValueError(
            "Effective d_min must be positive: "
            f"{args.min_headway:g} - {args.safety_headway_offset:g} = {effective:g}."
        )
    return effective


def _setting_stem(args) -> str:
    collision_token = "collabs" if bool(getattr(args, "collision_absorbing", False)) else "nocoll"
    return (
        f"headway{_format_sweep_value(float(args.min_headway))}_"
        f"hoff{_format_sweep_value(float(args.safety_headway_offset))}_"
        f"startH{_format_sweep_value(float(args.safe_start_headway_max))}_"
        f"rel{_format_sweep_value(float(args.safe_start_rel_speed_max))}_"
        f"evalH{_format_sweep_value(float(args.rmse_eval_headway_max))}_"
        f"evalRel{_format_sweep_value(float(args.rmse_eval_rel_speed_max))}_"
        f"{collision_token}_ch{_format_sweep_value(float(getattr(args, 'collision_headway', 0.0)))}_"
        f"cpen{_format_sweep_value(float(getattr(args, 'collision_safety_penalty', 0.0)))}"
    )


def _variant_result_path(args, variant: Dict[str, object]) -> Path:
    result_id = _safe_filename_part(str(variant["result_id"]))
    return args.save_dir / (
        f"ngsim_{result_id}_m{int(args.horizon)}_rounds{int(args.rounds)}_"
        f"trials{int(args.trials)}_seed{int(args.seed)}_{_setting_stem(args)}_results.json"
    )


def _build_method_variants(args) -> List[Dict[str, object]]:
    labels = {
        "fa_sal": "FA-SAL",
        "fa_random": "FA-SAL-Random",
        "salnx": "SAL-NX",
        "tebbe_abm": "Tebbe-ABM",
    }
    variants: List[Dict[str, object]] = []
    for method in args.methods:
        method_name = str(method)
        base_label = labels.get(method_name, method_name)
        method_horizon = int(
            getattr(args, f"{_method_config_prefix(method_name)}_horizon", getattr(args, "horizon"))
        )
        if method_name in {"fa_sal", "fa_random"}:
            prefix = _method_config_prefix(method_name)
            quantile_sweep = tuple(float(v) for v in getattr(args, f"{prefix}_l_ell_quantile_sweep", ()))
            quantiles = quantile_sweep or (float(getattr(args, f"{prefix}_l_ell_quantile")),)
            scale_sweep = tuple(float(v) for v in getattr(args, f"{prefix}_l_ell_scale_sweep", ()))
            scales = scale_sweep or (float(getattr(args, f"{prefix}_l_ell_scale")),)
            if scale_sweep or quantile_sweep:
                for scale in scales:
                    for quantile in quantiles:
                        quantile_id = _format_sweep_value(quantile)
                        variants.append(
                            {
                                "label": (
                                    f"{base_label} (l_ell_scale={_format_label_value(scale)}, "
                                    f"q={_format_label_value(quantile)})"
                                ),
                                "base_method": method_name,
                                "result_id": (
                                    f"{method_name}_l_ell_scale_{_format_sweep_value(scale)}"
                                    f"_q_{quantile_id}"
                                ),
                                "overrides": {
                                    "horizon": int(method_horizon),
                                    f"{prefix}_l_ell_scale": float(scale),
                                    f"{prefix}_l_ell_quantile": float(quantile),
                                },
                            }
                        )
                continue
            scale = scales[0]
            quantile = quantiles[0]
            quantile_id = _format_sweep_value(quantile)
            variants.append(
                {
                    "label": (
                        f"{base_label} (l_ell_scale={_format_label_value(scale)}, "
                        f"q={_format_label_value(quantile)})"
                    ),
                    "base_method": method_name,
                    "result_id": (
                        f"{method_name}_l_ell_scale_{_format_sweep_value(scale)}"
                        f"_q_{quantile_id}"
                    ),
                    "overrides": {"horizon": int(method_horizon)},
                }
            )
            continue
        if method_name == "salnx":
            sweep = tuple(float(v) for v in getattr(args, "salnx_alpha_sweep", ()))
            if sweep:
                for alpha in sweep:
                    variants.append(
                        {
                            "label": f"{base_label} (alpha={_format_label_value(alpha)})",
                            "base_method": method_name,
                            "result_id": f"salnx_alpha_{_format_sweep_value(alpha)}",
                            "overrides": {"horizon": int(method_horizon), "salnx_alpha": float(alpha)},
                        }
                    )
                continue
        variants.append(
            {
                "label": base_label,
                "base_method": method_name,
                "result_id": method_name,
                "overrides": {"horizon": int(method_horizon)},
            }
        )
    return variants


def _namespace_with_overrides(args, overrides: Dict[str, object]):
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def _run_trial_task(task):
    args, method, seed, data, action_sequences = task
    return run_trial(
        args,
        method=method,
        seed=seed,
        data=data,
        action_sequences=action_sequences,
    )


def _canonical_column(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _column_map(columns: Sequence[str]) -> Dict[str, str]:
    lookup = {_canonical_column(col): col for col in columns}
    aliases = {
        "vehicle_id": ("vehicleid", "vehicle"),
        "frame_id": ("frameid", "frame"),
        "local_y": ("localy", "y"),
        "v_vel": ("vvel", "velocity", "speed"),
        "v_acc": ("vacc", "acceleration", "accel"),
        "preceding": ("preceding", "leader", "precedingvehicle"),
        "v_length": ("vlength", "length", "vehiclelength"),
        "space_headway": ("spaceheadway", "headway", "spacing"),
    }
    resolved: Dict[str, str] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in lookup:
                resolved[target] = lookup[candidate]
                break
    missing = [key for key in ("vehicle_id", "frame_id", "local_y", "v_vel", "v_acc", "preceding") if key not in resolved]
    if missing:
        raise ValueError(f"NGSIM CSV is missing required columns: {missing}. Available columns: {list(columns)}")
    return resolved


@dataclass
class CarFollowingData:
    X: np.ndarray
    y_next: np.ndarray
    g: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    residuals: np.ndarray
    min_headway: float
    dt: float
    residual_neighbors: int
    collision_absorbing: bool = False
    collision_headway: float = 0.0
    collision_safety_penalty: float = -10.0

    def normalize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return (X - self.feature_mean.reshape(1, -1)) / self.feature_scale.reshape(1, -1)

    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        X_norm = np.asarray(X_norm, dtype=float)
        if X_norm.ndim == 1:
            X_norm = X_norm.reshape(1, -1)
        return X_norm * self.feature_scale.reshape(1, -1) + self.feature_mean.reshape(1, -1)

    def nominal_next_headway(self, state: np.ndarray, action: float) -> float:
        headway, rel_speed, _ego_speed = map(float, state)
        rel_next = rel_speed - float(action) * self.dt
        return max(0.0, headway + self.dt * rel_next)

    def residual_correction(self, row: np.ndarray) -> float:
        row = np.asarray(row, dtype=float).reshape(1, -1)
        Xn = self.normalize(self.X)
        rn = self.normalize(row)
        dists = np.sum((Xn - rn) ** 2, axis=1)
        k = max(1, min(int(self.residual_neighbors), dists.size))
        idx = np.argpartition(dists, k - 1)[:k]
        weights = 1.0 / np.maximum(dists[idx], 1e-8)
        weights = weights / np.sum(weights)
        correction = float(np.sum(weights * self.residuals[idx]))
        return float(np.clip(correction, -10.0, 10.0))

    def _apply_collision_absorbing(self, headway: float) -> float:
        headway = max(0.0, float(headway))
        if self.collision_absorbing and headway < self.min_headway:
            return max(0.0, float(self.collision_headway))
        return headway

    def safety_margin_from_headway(self, headway: float) -> float:
        headway = max(0.0, float(headway))
        if self.collision_absorbing and headway < self.min_headway:
            return float(self.collision_safety_penalty)
        return float(headway - self.min_headway)

    def deterministic_next_headway(self, state: np.ndarray, action: float) -> float:
        state = np.asarray(state, dtype=float).reshape(3)
        if self.collision_absorbing and float(state[0]) < self.min_headway:
            return max(0.0, float(self.collision_headway))
        row = self.row_from_state_action(state, action)
        value = self.nominal_next_headway(state, action) + self.residual_correction(row)
        return self._apply_collision_absorbing(value)

    def deterministic_step(self, state: np.ndarray, action: float) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(3)
        next_headway = self.deterministic_next_headway(state, action)
        if self.collision_absorbing and next_headway <= max(0.0, float(self.collision_headway)) + 1e-8:
            return np.array([next_headway, 0.0, 0.0], dtype=float)
        next_rel_speed = np.clip((next_headway - float(state[0])) / max(self.dt, 1e-8), -40.0, 40.0)
        next_ego_speed = max(0.0, float(state[2]) + float(action) * self.dt)
        return np.array([next_headway, next_rel_speed, next_ego_speed], dtype=float)

    def true_next_headway(self, state: np.ndarray, action: float, noise_std: float, rng: np.random.Generator) -> float:
        state = np.asarray(state, dtype=float).reshape(3)
        if self.collision_absorbing and float(state[0]) < self.min_headway:
            return max(0.0, float(self.collision_headway))
        row = np.array([float(state[0]), float(state[1]), float(state[2]), float(action)], dtype=float)
        value = self.nominal_next_headway(state, action) + self.residual_correction(row)
        if noise_std > 0.0:
            value += float(rng.normal(0.0, noise_std))
        return self._apply_collision_absorbing(value)

    def step(self, state: np.ndarray, action: float, noise_std: float, rng: np.random.Generator) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(3)
        next_headway = self.true_next_headway(state, action, noise_std=noise_std, rng=rng)
        if self.collision_absorbing and next_headway <= max(0.0, float(self.collision_headway)) + 1e-8:
            return np.array([next_headway, 0.0, 0.0], dtype=float)
        next_rel_speed = np.clip((next_headway - float(state[0])) / max(self.dt, 1e-8), -40.0, 40.0)
        next_ego_speed = max(0.0, float(state[2]) + float(action) * self.dt)
        return np.array([next_headway, next_rel_speed, next_ego_speed], dtype=float)

    def row_from_state_action(self, state: np.ndarray, action: float) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(3)
        return np.array([state[0], state[1], state[2], float(action)], dtype=float)

    def safety_value(self, row: np.ndarray) -> float:
        row = np.asarray(row, dtype=float).reshape(-1)
        state = row[:3]
        action = float(row[3])
        next_headway = self.deterministic_next_headway(state, action)
        return self.safety_margin_from_headway(next_headway)


def _load_single_ngsim_csv(
    path: Path,
    unit_scale: float,
    max_source_rows: int,
    dt: float,
    frame_stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return np.empty((0, 4)), np.empty((0,))
        cmap = _column_map(reader.fieldnames)
        records: Dict[Tuple[int, int], Dict[str, float]] = {}
        for row_idx, raw in enumerate(reader):
            if max_source_rows > 0 and row_idx >= int(max_source_rows):
                break
            try:
                vehicle_id = int(float(raw[cmap["vehicle_id"]]))
                frame_id = int(float(raw[cmap["frame_id"]]))
                preceding = int(float(raw[cmap["preceding"]]))
                local_y = float(raw[cmap["local_y"]])
                v_vel = float(raw[cmap["v_vel"]])
                v_acc = float(raw[cmap["v_acc"]])
                v_length = float(raw[cmap["v_length"]]) if "v_length" in cmap else 0.0
                space_headway = float(raw[cmap["space_headway"]]) if "space_headway" in cmap else float("nan")
            except (KeyError, TypeError, ValueError):
                continue
            records[(vehicle_id, frame_id)] = {
                "vehicle_id": float(vehicle_id),
                "frame_id": float(frame_id),
                "preceding": float(preceding),
                "local_y": local_y,
                "v_vel": v_vel,
                "v_acc": v_acc,
                "v_length": v_length,
                "space_headway": space_headway,
            }
    if not records:
        return np.empty((0, 4)), np.empty((0,))

    rows: List[Tuple[float, float, float, float]] = []
    targets: List[float] = []
    scale = float(unit_scale)
    frame_stride = max(1, int(frame_stride))
    for (vehicle_id, frame_id), ego in records.items():
        ego_next = records.get((vehicle_id, frame_id + frame_stride))
        preceding = int(ego["preceding"])
        leader = records.get((preceding, frame_id)) if preceding > 0 else None
        if ego_next is not None and np.isfinite(float(ego.get("space_headway", float("nan")))) and np.isfinite(float(ego_next.get("space_headway", float("nan")))):
            headway = float(ego["space_headway"]) * scale
            y_next = float(ego_next["space_headway"]) * scale
            action = float(ego["v_acc"]) * scale
            if leader is not None:
                rel_speed = (float(leader["v_vel"]) - float(ego["v_vel"])) * scale
            else:
                rel_speed = (y_next - headway) / max(float(dt), 1e-8) + float(dt) * action
            ego_speed = float(ego["v_vel"]) * scale
            rows.append((headway, rel_speed, ego_speed, action))
            targets.append(y_next)
            continue

        if preceding <= 0:
            continue
        leader_next = records.get((preceding, frame_id + frame_stride))
        if leader is None or ego_next is None or leader_next is None:
            continue

        leader_length = float(leader.get("v_length", 0.0))
        headway = (float(leader["local_y"]) - float(ego["local_y"]) - leader_length) * scale
        rel_speed = (float(leader["v_vel"]) - float(ego["v_vel"])) * scale
        ego_speed = float(ego["v_vel"]) * scale
        action = float(ego["v_acc"]) * scale
        y_next = (float(leader_next["local_y"]) - float(ego_next["local_y"]) - leader_length) * scale
        rows.append((headway, rel_speed, ego_speed, action))
        targets.append(y_next)

    if not rows:
        return np.empty((0, 4)), np.empty((0,))

    X_raw = np.asarray(rows, dtype=float)
    y_raw = np.asarray(targets, dtype=float)
    headway = X_raw[:, 0]
    rel_speed = X_raw[:, 1]
    ego_speed = X_raw[:, 2]
    action = X_raw[:, 3]
    y_next = y_raw

    mask = (
        np.isfinite(headway)
        & np.isfinite(rel_speed)
        & np.isfinite(ego_speed)
        & np.isfinite(action)
        & np.isfinite(y_next)
        & (headway > 0.5)
        & (headway < 150.0)
        & (y_next > 0.0)
        & (y_next < 150.0)
        & (ego_speed >= 0.0)
        & (np.abs(rel_speed) <= 40.0)
        & (np.abs(y_next - headway) <= 12.0)
        & (np.abs(action) < 12.0)
    )
    X = X_raw[mask]
    return X.astype(float), y_next[mask].astype(float)


def load_ngsim_data(
    data_path: Path,
    max_transitions: int,
    unit_scale: float,
    min_headway: float,
    dt: float,
    frame_stride: int,
    residual_neighbors: int,
    seed: int,
    collision_absorbing: bool,
    collision_headway: float,
    collision_safety_penalty: float,
) -> CarFollowingData:
    path = Path(data_path)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(
            f"NGSIM data path not found: {path}. Download trajectory CSV files from {NGSIM_DATA_URL} "
            "and set NGSIM_CONFIG.data.data_path in config_ngsim.py."
        )
    if not files:
        raise FileNotFoundError(f"No CSV files found under {path}. Download NGSIM CSV files from {NGSIM_DATA_URL}.")

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    effective_dt = float(dt) * max(1, int(frame_stride))
    max_source_rows = max(100000, int(max_transitions) * 200) if max_transitions > 0 else 0
    for file_path in files:
        X_file, y_file = _load_single_ngsim_csv(
            file_path,
            unit_scale=unit_scale,
            max_source_rows=max_source_rows,
            dt=effective_dt,
            frame_stride=frame_stride,
        )
        if X_file.size:
            X_parts.append(X_file)
            y_parts.append(y_file)
    if not X_parts:
        raise ValueError("No valid car-following transitions were extracted from the selected NGSIM CSV files.")

    X = np.vstack(X_parts)
    y_next = np.concatenate(y_parts)
    rng = np.random.default_rng(seed)
    if max_transitions > 0 and X.shape[0] > int(max_transitions):
        idx = rng.choice(X.shape[0], size=int(max_transitions), replace=False)
        X = X[idx]
        y_next = y_next[idx]

    if bool(collision_absorbing):
        collision_mask = y_next < float(min_headway)
        y_next = y_next.copy()
        y_next[collision_mask] = max(0.0, float(collision_headway))

    nominal = X[:, 0] + effective_dt * (X[:, 1] - effective_dt * X[:, 3])
    residuals = y_next - nominal
    feature_mean = np.mean(X, axis=0)
    feature_scale = np.std(X, axis=0)
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
    g = y_next - float(min_headway)
    if bool(collision_absorbing):
        collision_mask = y_next <= max(0.0, float(collision_headway)) + 1e-8
        g = g.copy()
        g[collision_mask] = float(collision_safety_penalty)
    return CarFollowingData(
        X=X,
        y_next=y_next,
        g=g,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        residuals=residuals,
        min_headway=float(min_headway),
        dt=effective_dt,
        residual_neighbors=int(residual_neighbors),
        collision_absorbing=bool(collision_absorbing),
        collision_headway=max(0.0, float(collision_headway)),
        collision_safety_penalty=float(collision_safety_penalty),
    )


PLANNER_ACTION_TO_ACCELERATION = {
    -4.0: -6.0,
    -2.0: -4.0,
    0.0: -2.0,
    1.0: 0.0,
    2.0: 1.0,
}


def map_planner_action_to_acceleration(action: float) -> float:
    value = float(action)
    for planner_action, acceleration in PLANNER_ACTION_TO_ACCELERATION.items():
        if np.isclose(value, planner_action, rtol=0.0, atol=1e-12):
            return acceleration
    raise ValueError(
        f"Unknown planner action {value:g}; expected one of "
        f"{tuple(PLANNER_ACTION_TO_ACCELERATION)}."
    )


def planner_action_for_acceleration(acceleration: float) -> float:
    value = float(acceleration)
    for planner_action, mapped_acceleration in PLANNER_ACTION_TO_ACCELERATION.items():
        if np.isclose(value, mapped_acceleration, rtol=0.0, atol=1e-12):
            return planner_action
    raise ValueError(
        f"Unknown physical acceleration {value:g}; expected one of "
        f"{tuple(PLANNER_ACTION_TO_ACCELERATION.values())}."
    )


def _make_action_sequences(actions: Sequence[float], horizon: int) -> List[Tuple[float, ...]]:
    return [tuple(float(v) for v in seq) for seq in itertools.product(tuple(actions), repeat=int(horizon))]


def _rollout_rows(data: CarFollowingData, state: np.ndarray, actions: Sequence[float]) -> np.ndarray:
    current = np.asarray(state, dtype=float).reshape(3)
    rows = []
    for action in actions:
        acceleration = map_planner_action_to_acceleration(action)
        row = data.row_from_state_action(current, acceleration)
        rows.append(row)
        current = data.deterministic_step(current, acceleration)
    return np.asarray(rows, dtype=float)


def _select_planner(method: str, dyn_model: DynamicsGP, safe_model: SafetyGP, cfg: PlannerConfig, rng: np.random.Generator):
    if method == "fa_sal":
        return FutureAwareTrajectoryPlanner(dyn_model, safe_model, cfg)
    if method == "fa_random":
        return FARandomTrajectoryPlanner(dyn_model, safe_model, cfg, rng=rng)
    if method == "salnx":
        return SALNXTrajectoryPlanner(dyn_model, safe_model, cfg)
    if method == "tebbe_abm":
        return TebbeABMTrajectoryPlanner(dyn_model, safe_model, cfg)
    raise ValueError(f"Unsupported method: {method}")


def _method_alpha(args, method: str) -> float:
    if method == "tebbe_abm":
        return float(args.tebbe_alpha)
    return float(getattr(args, f"{_method_config_prefix(method)}_alpha", args.salnx_alpha))


def _method_uncertainty_criterion(args, method: str) -> str:
    return str(getattr(args, f"{_method_config_prefix(method)}_uncertainty_criterion", "logdet"))


def _method_l_ell_scale(args, method: str) -> float:
    return float(getattr(args, f"{_method_config_prefix(method)}_l_ell_scale", 0.0))


def _method_l_ell_quantile(args, method: str) -> float:
    return float(getattr(args, f"{_method_config_prefix(method)}_l_ell_quantile", 0.9))


def _method_l_ell_fallback(args, method: str) -> float:
    return float(getattr(args, f"{_method_config_prefix(method)}_l_ell", 1.0))


def _confidence_beta_from_delta(delta: float) -> float:
    delta = min(max(float(delta), 1e-12), 1.0 - 1e-12)
    return float(max(1.0, 2.0 * np.log(1.0 / delta)))


def _method_effective_l_ell(args, method: str, estimated_l_ell: Optional[float]) -> float:
    fallback = _method_l_ell_fallback(args, method)
    scale = _method_l_ell_scale(args, method)
    if estimated_l_ell is not None and np.isfinite(estimated_l_ell) and float(estimated_l_ell) > 0.0:
        return max(0.0, float(scale) * float(estimated_l_ell))
    return max(0.0, float(fallback))


def estimate_ngsim_local_lcb_lipschitz_constant(
    safe_model: SafetyGP,
    points: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    steps: np.ndarray,
    quantile: float = 0.9,
    beta_g: float = 1.0,
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

    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    steps = np.asarray(steps, dtype=float)
    quantile = float(np.clip(quantile, 0.0, 1.0))
    local_norms: List[float] = []
    for z in points:
        z = np.asarray(z, dtype=float)
        center = float(safe_model.lcb(z, beta_g))
        axis_slopes: List[float] = []
        for axis in range(z.shape[0]):
            step = float(steps[axis])
            lower = float(lower_bounds[axis])
            upper = float(upper_bounds[axis])
            if step <= 0.0 or upper <= lower:
                axis_slopes.append(0.0)
                continue
            plus = z.copy()
            minus = z.copy()
            plus[axis] = min(float(z[axis]) + step, upper)
            minus[axis] = max(float(z[axis]) - step, lower)
            slopes = []
            if abs(float(plus[axis]) - float(z[axis])) > 1e-10:
                slopes.append(abs(float(safe_model.lcb(plus, beta_g)) - center) / abs(float(plus[axis]) - float(z[axis])))
            if abs(float(minus[axis]) - float(z[axis])) > 1e-10:
                slopes.append(abs(center - float(safe_model.lcb(minus, beta_g))) / abs(float(z[axis]) - float(minus[axis])))
            axis_slopes.append(max(slopes) if slopes else 0.0)
        local_norms.append(float(np.linalg.norm(axis_slopes)))
    if not local_norms:
        return 0.0
    return float(np.quantile(np.asarray(local_norms, dtype=float), quantile))


def _iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    union = float(np.sum(predicted | truth))
    return 1.0 if union <= 0.0 else float(np.sum(predicted & truth) / union)


def _current_safe_indices(data: CarFollowingData) -> np.ndarray:
    return np.flatnonzero(data.X[:, 0] >= data.min_headway)


def _safe_transition_indices(data: CarFollowingData) -> np.ndarray:
    return np.flatnonzero((data.X[:, 0] >= data.min_headway) & (data.g >= 0.0))


def _challenging_safe_start_indices(args, data: CarFollowingData) -> np.ndarray:
    safe_idx = _current_safe_indices(data)
    if safe_idx.size == 0:
        return np.arange(data.X.shape[0])

    headway_max = float(getattr(args, "safe_start_headway_max", float("inf")))
    rel_speed_max = float(getattr(args, "safe_start_rel_speed_max", float("inf")))
    min_count = int(getattr(args, "safe_start_min_count", 0))
    mask = np.ones(safe_idx.size, dtype=bool)
    if np.isfinite(headway_max):
        mask &= data.X[safe_idx, 0] <= headway_max
    if np.isfinite(rel_speed_max):
        mask &= data.X[safe_idx, 1] <= rel_speed_max
    challenging_idx = safe_idx[mask]
    if challenging_idx.size >= max(1, min_count):
        return challenging_idx
    return safe_idx


def _candidate_metrics(data: CarFollowingData, infos: Sequence[object]) -> Dict[str, float]:
    if not infos:
        return {
            "false_certification_rate": 0.0,
            "feasible_sizes": 0.0,
        }
    pred = np.array([bool(getattr(info, "feasible_full_horizon", False)) for info in infos], dtype=bool)
    true = []
    for info in infos:
        rows = data.denormalize(np.asarray(getattr(info, "trajectory"), dtype=float))
        state = rows[0, :3].copy()
        margins = []
        for row in rows:
            action = float(row[3])
            next_state = data.deterministic_step(state, action)
            margins.append(data.safety_margin_from_headway(float(next_state[0])))
            state = next_state
        true.append(bool(np.all(np.asarray(margins, dtype=float) >= 0.0)))
    true_arr = np.asarray(true, dtype=bool)
    false_cert = pred & ~true_arr
    return {
        "false_certification_rate": float(np.sum(false_cert) / np.sum(pred)) if np.any(pred) else 0.0,
        "feasible_sizes": float(np.sum(pred)),
    }


def _deterministic_trajectory_margins(
    data: CarFollowingData,
    state: np.ndarray,
    actions: Sequence[float],
) -> np.ndarray:
    current = np.asarray(state, dtype=float).reshape(3).copy()
    margins = []
    for action in actions:
        acceleration = map_planner_action_to_acceleration(action)
        next_state = data.deterministic_step(current, acceleration)
        margins.append(data.safety_margin_from_headway(float(next_state[0])))
        current = next_state
    return np.asarray(margins, dtype=float)


def _context_has_true_future_safe_sequence(
    data: CarFollowingData,
    state: np.ndarray,
    action_sequences: Sequence[Tuple[float, ...]],
) -> bool:
    state = np.asarray(state, dtype=float).reshape(3)
    if data.safety_margin_from_headway(float(state[0])) < 0.0:
        return False
    return any(
        bool(np.all(_deterministic_trajectory_margins(data, state, actions) >= 0.0))
        for actions in action_sequences
    )


def _evaluate_dead_end_free_contexts(
    planner,
    data: CarFollowingData,
    X_eval: np.ndarray,
    action_sequences: Sequence[Tuple[float, ...]],
) -> Dict[str, float]:
    predicted = []
    truth = []
    for row in np.asarray(X_eval, dtype=float):
        state = row[:3].copy()
        predicted.append(
            any(
                planner.evaluate_trajectory(
                    np.asarray(actions, dtype=float),
                    data.normalize(_rollout_rows(data, state, actions)),
                ).feasible_full_horizon
                for actions in action_sequences
            )
        )
        truth.append(_context_has_true_future_safe_sequence(data, state, action_sequences))

    predicted_arr = np.asarray(predicted, dtype=bool)
    truth_arr = np.asarray(truth, dtype=bool)
    return {"dead_end_free_recovery_iou": _iou(predicted_arr, truth_arr)}


def run_trial(args, method: str, seed: int, data: CarFollowingData, action_sequences: Sequence[Tuple[float, ...]]) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(int(seed) + 1_000_003)
    safe_idx = _safe_transition_indices(data)
    if safe_idx.size < args.n_init:
        safe_idx = _current_safe_indices(data)
    if safe_idx.size < args.n_init:
        safe_idx = np.arange(data.X.shape[0])
    context_idx_pool = _challenging_safe_start_indices(args, data)
    rmse_context_args = argparse.Namespace(
        safe_start_headway_max=float(args.rmse_eval_headway_max),
        safe_start_rel_speed_max=float(args.rmse_eval_rel_speed_max),
        safe_start_min_count=int(args.rmse_eval_min_count),
    )
    rmse_context_idx_pool = _challenging_safe_start_indices(rmse_context_args, data)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    if manifest.get("schema") != "ngsim_eval_split_v2":
        raise ValueError("Split manifest must use schema ngsim_eval_split_v2.")
    if int(manifest.get("pool_seed", -1)) != int(args.seed):
        raise ValueError(f"Split manifest pool_seed must match --seed={args.seed}.")
    if int(manifest.get("n_eval", -1)) != int(args.n_eval):
        raise ValueError(f"Split manifest n_eval must match --n-eval={args.n_eval}.")

    expected_rollout = {
        "headway_max": float(args.safe_start_headway_max),
        "rel_speed_max": float(args.safe_start_rel_speed_max),
        "min_count": int(args.safe_start_min_count),
    }
    if manifest.get("rollout_context") != expected_rollout:
        raise ValueError(
            f"Split manifest rollout context {manifest.get('rollout_context')} does not match requested {expected_rollout}."
        )
    expected_rmse = {
        "headway_max": float(args.rmse_eval_headway_max),
        "rel_speed_max": float(args.rmse_eval_rel_speed_max),
        "min_count": int(args.rmse_eval_min_count),
    }
    if manifest.get("rmse_eval_context") != expected_rmse:
        raise ValueError(
            f"Split manifest RMSE context {manifest.get('rmse_eval_context')} does not match requested {expected_rmse}."
        )

    trial_manifest = next(
        (item for item in manifest.get("trials", []) if int(item["trial_seed"]) == int(seed)),
        None,
    )
    if trial_manifest is None:
        raise ValueError(f"Split manifest has no entry for trial seed {seed}: {args.split_manifest}")

    def validated_indices(key: str, expected_size: int, allowed: Optional[np.ndarray] = None) -> np.ndarray:
        values = np.asarray(trial_manifest.get(key, []), dtype=int)
        if values.ndim != 1 or values.size != int(expected_size):
            raise ValueError(f"Split manifest seed {seed} field {key} must contain exactly {expected_size} indices.")
        if np.any(values < 0) or np.any(values >= data.X.shape[0]):
            raise ValueError(f"Split manifest seed {seed} field {key} contains an out-of-range index.")
        if np.unique(values).size != values.size:
            raise ValueError(f"Split manifest seed {seed} field {key} contains duplicate indices.")
        if allowed is not None and not np.all(np.isin(values, allowed)):
            raise ValueError(f"Split manifest seed {seed} field {key} violates its required pool condition.")
        return values

    init_idx = validated_indices("initial_train_pool_indices", int(args.n_init), safe_idx)
    eval_idx = validated_indices("global_eval_pool_indices", int(args.n_eval))
    rmse_eval_idx = validated_indices("rmse_eval_pool_indices", int(args.n_eval), rmse_context_idx_pool)
    if np.intersect1d(init_idx, eval_idx).size or np.intersect1d(init_idx, rmse_eval_idx).size:
        raise ValueError(f"Split manifest seed {seed} has train/evaluation index overlap.")

    post_split_rng_state = trial_manifest.get("post_split_rng_state")
    if not isinstance(post_split_rng_state, dict):
        raise ValueError(f"Split manifest seed {seed} is missing post_split_rng_state.")
    if post_split_rng_state.get("bit_generator") != rng.bit_generator.__class__.__name__:
        raise ValueError(f"Split manifest seed {seed} uses an incompatible RNG bit generator.")
    try:
        rng.bit_generator.state = post_split_rng_state
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Split manifest seed {seed} has an invalid post_split_rng_state.") from exc

    X_train = data.X[init_idx].copy()
    y_train = data.y_next[init_idx].copy()
    g_train = data.g[init_idx].copy()
    X_eval = data.X[eval_idx].copy()
    y_eval = data.y_next[eval_idx].copy()
    g_eval = data.g[eval_idx].copy()
    X_rmse_eval = data.X[rmse_eval_idx].copy()
    y_rmse_eval = data.y_next[rmse_eval_idx].copy()
    state = X_train[-1, :3].copy()
    dyn_model = DynamicsGP(KernelConfig(kind=args.kernel, variance=1.0, length_scale=args.length_scale), noise_std=args.dynamics_noise_std)
    safe_model = SafetyGP(KernelConfig(kind=args.kernel, variance=1.0, length_scale=args.length_scale), noise_std=args.safety_noise_std)

    logs: Dict[str, List[float]] = {
        "dynamics_rmse": [],
        "evaluation_dynamics_rmse": [],
        "safety_violation_rate": [],
        "dead_end_safety_violation_rate": [],
        "safety_set_recovery_iou": [],
        "safety_set_recovery_iou_active": [],
        "dead_end_free_recovery_iou": [],
        "dead_end_free_recovery_iou_active": [],
        "false_certification_rate": [],
        "feasible_sizes": [],
        "selected_safety_probability": [],
        "selected_score": [],
        "estimated_l_ell": [],
        "effective_l_ell": [],
    }
    violations = 0.0
    dead_end_violations = 0.0

    for round_idx in _iter_progress(
        range(int(args.rounds)),
        desc=f"{method} seed {seed}",
        leave=False,
        dynamic_ncols=True,
        disable=bool(getattr(args, "quiet", False)),
    ):
        if bool(getattr(args, "reset_context_each_round", True)):
            context_idx = int(rng.choice(context_idx_pool))
            state = data.X[context_idx, :3].copy()

        dyn_model.fit(data.normalize(X_train), y_train)
        safe_model.fit(data.normalize(X_train), g_train)

        is_final_round = round_idx == int(args.rounds) - 1
        alpha = _method_alpha(args, method)
        if is_final_round:
            rmse_y_pred, _ = dyn_model.predict_batch(data.normalize(X_rmse_eval))
            evaluation_rmse = float(np.sqrt(np.mean((rmse_y_pred - y_rmse_eval) ** 2)))
            mean_g, std_g = safe_model.predict_batch(data.normalize(X_eval))
            pred_safe = normal_cdf_array(mean_g / np.maximum(std_g, 1e-8)) >= 1.0 - alpha
            true_safe = g_eval >= 0.0
            point_iou = _iou(pred_safe, true_safe)

        distance_to_state = np.linalg.norm(data.normalize(X_train) - data.normalize(data.row_from_state_action(state, 0.0)), axis=1)
        nearest_count = min(64, X_train.shape[0])
        nearest_idx = np.argsort(distance_to_state)[:nearest_count]
        points = data.normalize(np.vstack([data.row_from_state_action(state, 0.0).reshape(1, -1), X_train[nearest_idx]]))
        lower_raw = np.min(X_train, axis=0)
        upper_raw = np.max(X_train, axis=0)
        action_values = np.asarray(
            [map_planner_action_to_acceleration(action) for action in args.action_values],
            dtype=float,
        )
        lower_raw[-1] = min(float(np.min(action_values)), float(lower_raw[-1]))
        upper_raw[-1] = max(float(np.max(action_values)), float(upper_raw[-1]))
        lower_bounds = data.normalize(lower_raw)[0]
        upper_bounds = data.normalize(upper_raw)[0]
        steps = np.maximum((upper_bounds - lower_bounds) / 20.0, 1e-6)
        beta_f = _confidence_beta_from_delta(float(args.delta_f))
        beta_g = _confidence_beta_from_delta(float(args.delta_g))
        estimated_l_ell = estimate_ngsim_local_lcb_lipschitz_constant(
            safe_model=safe_model,
            points=points,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            steps=steps,
            quantile=_method_l_ell_quantile(args, method),
            beta_g=beta_g,
        )
        method_l_ell = _method_effective_l_ell(args, method, estimated_l_ell)
        planner_cfg = PlannerConfig(
            horizon=int(args.horizon),
            beta_f=beta_f,
            beta_g=beta_g,
            Lx=1.0,
            Ly=1.0,
            Lf=1.0,
            L_ell=float(method_l_ell),
            alpha_safety=float(alpha),
            salnx_criterion=_method_uncertainty_criterion(args, method),
            salnx_joint_mc_samples=int(args.salnx_mc_samples),
            tebbe_alpha=float(args.tebbe_alpha),
            tebbe_criterion=str(args.tebbe_uncertainty_criterion),
            tebbe_confidence_delta=float(args.tebbe_confidence_delta),
            tebbe_sample_start=int(args.tebbe_sample_start),
            tebbe_sample_stages=int(args.tebbe_sample_stages),
            fa_beam_width=int(args.fa_beam_width),
        )
        candidates = [
            (np.asarray(seq, dtype=float), data.normalize(_rollout_rows(data, state, seq)))
            for seq in action_sequences
        ]
        start_state = state.copy()
        planner = _select_planner(method, dyn_model, safe_model, planner_cfg, rng)
        if is_final_round:
            eval_planner = _select_planner(method, dyn_model, safe_model, planner_cfg, eval_rng)
            final_context_metrics = _evaluate_dead_end_free_contexts(
                planner=eval_planner,
                data=data,
                X_eval=X_eval,
                action_sequences=action_sequences,
            )
        selected_actions, selected_rows_norm, info, infos = planner.select_trajectory(candidates)
        if selected_actions is None or selected_rows_norm is None:
            selected_actions = np.full(
                int(args.horizon),
                planner_action_for_acceleration(0.0),
                dtype=float,
            )
            selected_rows = _rollout_rows(data, state, selected_actions)
            prob = 0.0
            score = float("-inf")
        else:
            selected_rows = data.denormalize(selected_rows_norm)
            prob = float(getattr(info, "trajectory_safety_prob", 0.0))
            score = float(getattr(info, "acquisition", float("-inf")))
        metrics = _candidate_metrics(data, infos)

        rollout_rows = []
        rollout_y = []
        rollout_g = []
        true_safety = []
        current = state.copy()
        for action in selected_actions[: int(args.horizon)]:
            acceleration = map_planner_action_to_acceleration(action)
            row = data.row_from_state_action(current, acceleration)
            next_state = data.step(current, acceleration, noise_std=args.dynamics_noise_std, rng=rng)
            y_next = float(next_state[0])
            g_value = data.safety_margin_from_headway(y_next)
            if args.safety_noise_std > 0.0 and g_value >= 0.0:
                g_value += float(rng.normal(0.0, args.safety_noise_std))
            rollout_rows.append(row)
            rollout_y.append(y_next)
            rollout_g.append(g_value)
            true_safety.append(data.safety_margin_from_headway(y_next))
            current = next_state
        if not bool(getattr(args, "reset_context_each_round", True)):
            state = current

        if rollout_rows:
            X_train = np.vstack([X_train, np.asarray(rollout_rows, dtype=float)])
            y_train = np.concatenate([y_train, np.asarray(rollout_y, dtype=float)])
            g_train = np.concatenate([g_train, np.asarray(rollout_g, dtype=float)])

        true_safety_arr = np.asarray(true_safety, dtype=float)
        violation = float(np.any(true_safety_arr < 0.0))
        dead_end_violation = float(start_state[0] >= data.min_headway and np.any(true_safety_arr < 0.0))
        violations += violation
        dead_end_violations += dead_end_violation

        if is_final_round:
            logs["evaluation_dynamics_rmse"].append(evaluation_rmse)
        logs["safety_violation_rate"].append(float(violations / (round_idx + 1)))
        logs["dead_end_safety_violation_rate"].append(float(dead_end_violations / (round_idx + 1)))
        if is_final_round:
            logs["safety_set_recovery_iou"].append(point_iou)
            logs["safety_set_recovery_iou_active"].append(point_iou)
            logs["dead_end_free_recovery_iou"].append(float(final_context_metrics["dead_end_free_recovery_iou"]))
            logs["dead_end_free_recovery_iou_active"].append(float(final_context_metrics["dead_end_free_recovery_iou"]))
            logs["false_certification_rate"].append(float(metrics["false_certification_rate"]))
            logs["feasible_sizes"].append(float(metrics["feasible_sizes"]))
            logs["selected_safety_probability"].append(prob)
        logs["selected_score"].append(score)
        logs["estimated_l_ell"].append(float(estimated_l_ell))
        logs["effective_l_ell"].append(float(method_l_ell))
    return logs


def summarize_trials(trials: Sequence[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    keys = sorted({key for trial in trials for key in trial})
    summary: Dict[str, List[float]] = {}
    for key in keys:
        arr = np.asarray([trial[key] for trial in trials], dtype=float)
        summary[f"{key}_mean"] = np.mean(arr, axis=0).tolist()
        summary[f"{key}_std"] = np.std(arr, axis=0).tolist()
    return summary


def _final_mean(summary: Dict[str, List[float]], metric: str) -> float:
    values = np.asarray(summary.get(f"{metric}_mean", ()), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(values[-1])


def _variant_result_line(label: str, summary: Dict[str, List[float]]) -> str:
    metrics = (
        ("RMSE", "evaluation_dynamics_rmse", ".3f"),
        ("SVR", "safety_violation_rate", ".3f"),
        ("DSVR", "dead_end_safety_violation_rate", ".3f"),
        ("DEFreeIoU", "dead_end_free_recovery_iou_active", ".3f"),
        ("FalseCert", "false_certification_rate", ".3f"),
    )
    parts = []
    for display_name, metric, fmt in metrics:
        value = _final_mean(summary, metric)
        parts.append(f"{display_name}={value:{fmt}}")
    return f"{label}: final | " + " | ".join(parts)


def _metric_display_name(metric: str) -> str:
    names = {
        "evaluation_dynamics_rmse": "Evaluation dynamics RMSE",
        "dynamics_rmse": "Global dynamics RMSE",
        "safety_violation_rate": "SVR",
        "dead_end_safety_violation_rate": "D-SVR",
        "false_certification_rate": "False certification",
        "safety_set_recovery_iou_active": "Global safety-set IoU",
        "dead_end_free_recovery_iou_active": "Dead-end safety-set IoU",
        "feasible_sizes": "Certified feasible sequences",
        "selected_safety_probability": "Selected safety probability",
        "estimated_l_ell": "Estimated L_ell",
        "effective_l_ell": "Effective L_ell",
    }
    return names.get(metric, metric)


def _final_metric_rows(summaries: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, object]]:
    preferred = (
        "evaluation_dynamics_rmse",
        "safety_violation_rate",
        "dead_end_safety_violation_rate",
        "false_certification_rate",
        "dead_end_free_recovery_iou_active",
        "dynamics_rmse",
        "safety_set_recovery_iou_active",
        "feasible_sizes",
        "selected_safety_probability",
        "estimated_l_ell",
        "effective_l_ell",
    )
    rows: List[Dict[str, object]] = []
    for label, summary in summaries.items():
        for metric in preferred:
            mean = np.asarray(summary.get(f"{metric}_mean", ()), dtype=float)
            if mean.size == 0:
                continue
            std = np.asarray(summary.get(f"{metric}_std", np.zeros_like(mean)), dtype=float)
            rows.append(
                {
                    "label": label,
                    "metric": metric,
                    "metric_name": _metric_display_name(metric),
                    "final_mean": float(mean[-1]),
                    "final_std": float(std[-1]) if std.size else 0.0,
                    "min_mean": float(np.nanmin(mean)),
                    "max_mean": float(np.nanmax(mean)),
                    "first_mean": float(mean[0]),
                    "rounds": int(mean.size),
                }
            )
    return rows


def _metric_summary_stem(args, variant: Optional[Dict[str, object]] = None) -> str:
    setting_stem = _setting_stem(args)
    if variant is None:
        return (
            f"metric_summary_all_m{int(args.horizon)}_rounds{int(args.rounds)}_"
            f"trials{int(args.trials)}_seed{int(args.seed)}_{setting_stem}"
        )
    result_id = _safe_filename_part(str(variant["result_id"]))
    return (
        f"metric_summary_{result_id}_m{int(args.horizon)}_rounds{int(args.rounds)}_"
        f"trials{int(args.trials)}_seed{int(args.seed)}_{setting_stem}"
    )


def write_metric_logs(
    save_dir: Path,
    summaries: Dict[str, Dict[str, List[float]]],
    elapsed_seconds: float,
    *,
    stem: str = "metric_summary",
) -> Tuple[Path, Path]:
    rows = _final_metric_rows(summaries)
    txt_path = save_dir / f"{stem}.txt"
    csv_path = save_dir / f"{stem}.csv"

    lines = [
        "NGSIM metric summary",
        f"elapsed_seconds: {elapsed_seconds:.3f}",
        "",
        "Each row reports the final mean +/- std across trials, plus min/max of the mean curve.",
        "",
    ]
    current_label = None
    for row in rows:
        label = str(row["label"])
        if label != current_label:
            if current_label is not None:
                lines.append("")
            lines.append(f"[{label}]")
            current_label = label
        lines.append(
            f"- {row['metric_name']}: "
            f"final={float(row['final_mean']):.6g} +/- {float(row['final_std']):.6g}, "
            f"first={float(row['first_mean']):.6g}, "
            f"min={float(row['min_mean']):.6g}, "
            f"max={float(row['max_mean']):.6g}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fieldnames = ("label", "metric", "metric_name", "final_mean", "final_std", "first_mean", "min_mean", "max_mean", "rounds")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    return txt_path, csv_path


def _payload_config(
    args,
    data: CarFollowingData,
    variants: Sequence[Dict[str, object]],
    worker_count: int,
    use_parallel: bool,
) -> Dict[str, object]:
    return {
        "dataset": "NGSIM car-following",
        "data_path": str(args.data_path),
        "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
        "metric_schema": "ngsim_v1",
        "methods": list(args.methods),
        "variant_labels": [str(variant["label"]) for variant in variants],
        "variants": [
            {
                "label": str(variant["label"]),
                "base_method": str(variant["base_method"]),
                "result_id": str(variant["result_id"]),
                "overrides": dict(variant.get("overrides", {})),
            }
            for variant in variants
        ],
        "rounds": args.rounds,
        "trials": args.trials,
        "parallel_workers": worker_count if use_parallel else 1,
        "horizon": args.horizon,
        "fa_sal_horizon": args.fa_sal_horizon,
        "fa_random_horizon": args.fa_random_horizon,
        "salnx_horizon": args.salnx_horizon,
        "tebbe_horizon": args.tebbe_horizon,
        "action_values": list(args.action_values),
        "applied_acceleration_values": [
            map_planner_action_to_acceleration(action) for action in args.action_values
        ],
        "min_headway": args.min_headway,
        "safety_headway_offset": args.safety_headway_offset,
        "effective_min_headway": data.min_headway,
        "dt": args.dt,
        "frame_stride": args.frame_stride,
        "effective_dt": data.dt,
        "n_init": args.n_init,
        "delta_f": args.delta_f,
        "delta_g": args.delta_g,
        "reset_context_each_round": args.reset_context_each_round,
        "safe_start_headway_max": args.safe_start_headway_max,
        "safe_start_rel_speed_max": args.safe_start_rel_speed_max,
        "safe_start_min_count": args.safe_start_min_count,
        "rmse_eval_headway_max": args.rmse_eval_headway_max,
        "rmse_eval_rel_speed_max": args.rmse_eval_rel_speed_max,
        "rmse_eval_min_count": args.rmse_eval_min_count,
        "collision_absorbing": args.collision_absorbing,
        "collision_headway": args.collision_headway,
        "collision_safety_penalty": args.collision_safety_penalty,
        "max_transitions": args.max_transitions,
        "fa_beam_width": args.fa_beam_width,
        "fa_sal_l_ell": args.fa_sal_l_ell,
        "fa_sal_l_ell_quantile": args.fa_sal_l_ell_quantile,
        "fa_sal_l_ell_quantile_sweep": list(args.fa_sal_l_ell_quantile_sweep),
        "fa_sal_l_ell_scale_sweep": list(args.fa_sal_l_ell_scale_sweep),
        "fa_random_l_ell": args.fa_random_l_ell,
        "fa_random_l_ell_quantile": args.fa_random_l_ell_quantile,
        "fa_random_l_ell_quantile_sweep": list(args.fa_random_l_ell_quantile_sweep),
        "fa_random_l_ell_scale_sweep": list(args.fa_random_l_ell_scale_sweep),
        "salnx_alpha_sweep": list(args.salnx_alpha_sweep),
    }


def _build_payload(
    args,
    data: CarFollowingData,
    variants: Sequence[Dict[str, object]],
    summaries: Dict[str, Dict[str, List[float]]],
    trials: Dict[str, Sequence[Dict[str, List[float]]]],
    elapsed_seconds: float,
    worker_count: int,
    use_parallel: bool,
) -> Dict[str, object]:
    return {
        "config": _payload_config(args, data, variants, worker_count, use_parallel),
        "summaries": summaries,
        "trials": trials,
        "elapsed_seconds": elapsed_seconds,
    }


def _write_aggregate_outputs(args, payload: Dict[str, object], elapsed_seconds: float) -> None:
    result_path = args.save_dir / "ngsim_results.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_path}")
    metric_txt_path, metric_csv_path = write_metric_logs(
        args.save_dir,
        payload["summaries"],
        elapsed_seconds,
        stem=_metric_summary_stem(args),
    )
    print(f"Saved {metric_txt_path}")
    print(f"Saved {metric_csv_path}")


def _write_variant_outputs(
    args,
    variant: Dict[str, object],
    payload: Dict[str, object],
    summaries: Dict[str, Dict[str, List[float]]],
    trials: Dict[str, Sequence[Dict[str, List[float]]]],
    elapsed_seconds: float,
) -> None:
    label = str(variant["label"])
    if label not in summaries:
        return
    variant_args = _namespace_with_overrides(args, dict(variant.get("overrides", {})))
    variant_payload = {
        **payload,
        "config": {
            **payload["config"],
            "methods": [str(variant["base_method"])],
            "horizon": int(variant_args.horizon),
            "variant_labels": [label],
            "variants": [
                {
                    "label": label,
                    "base_method": str(variant["base_method"]),
                    "result_id": str(variant["result_id"]),
                    "overrides": dict(variant.get("overrides", {})),
                }
            ],
        },
        "summaries": {label: summaries[label]},
        "trials": {label: trials[label]},
    }
    variant_path = _variant_result_path(variant_args, variant)
    variant_path.write_text(json.dumps(variant_payload, indent=2), encoding="utf-8")
    print(f"Saved {variant_path}")
    variant_metric_txt_path, variant_metric_csv_path = write_metric_logs(
        args.save_dir,
        {label: summaries[label]},
        elapsed_seconds,
        stem=_metric_summary_stem(variant_args, variant),
    )
    print(f"Saved {variant_metric_txt_path}")
    print(f"Saved {variant_metric_csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=_config_value("data", "data_path", Path("data_ngsim")))
    parser.add_argument("--split-manifest", type=Path, required=True, help="Required deterministic per-trial split and post-split RNG-state manifest.")
    parser.add_argument("--max-transitions", type=int, default=_config_value("data", "max_transitions", 5000))
    parser.add_argument("--distance-unit-scale", type=float, default=_config_value("data", "distance_unit_scale", 0.3048))
    parser.add_argument("--methods", nargs="+", default=list(_config_value("general", "methods", ("fa_sal", "fa_random", "salnx", "tebbe_abm"))))
    parser.add_argument("--rounds", type=int, default=_config_value("general", "rounds", 100))
    parser.add_argument("--trials", type=int, default=_config_value("general", "trials", 5))
    parser.add_argument("--seed", type=int, default=_config_value("general", "seed", 0))
    parser.add_argument("--save-dir", type=Path, default=_config_value("general", "save_dir", Path("result_ngsim")))
    parser.add_argument("--quiet", action="store_true", default=_config_value("general", "quiet", False))
    parser.add_argument("--parallel-workers", type=int, default=_config_value("general", "parallel_workers", 1), help="Number of worker processes for trial-level parallelism. Use 0 to auto-select, 1 for serial execution.")
    parser.add_argument("--horizon", type=int, default=_config_value("environment", "horizon", 6))
    parser.add_argument("--fa-sal-horizon", type=int, default=_config_value("fa_sal", "horizon", _config_value("environment", "horizon", 6)))
    parser.add_argument("--fa-random-horizon", type=int, default=_config_value("fa_random", "horizon", _config_value("environment", "horizon", 6)))
    parser.add_argument("--salnx-horizon", type=int, default=_config_value("salnx", "horizon", _config_value("environment", "horizon", 6)))
    parser.add_argument("--tebbe-horizon", type=int, default=_config_value("tebbe", "horizon", _config_value("environment", "horizon", 6)))
    parser.add_argument("--action-values", type=float, nargs="+", default=list(_config_value("environment", "action_values", (-4.0, -2.0, 0.0, 1.0, 2.0))))
    parser.add_argument("--min-headway", type=float, default=_config_value("environment", "min_headway", 8.0), help="Nominal d_min in meters.")
    parser.add_argument("--safety-headway-offset", type=float, default=_config_value("environment", "safety_headway_offset", 0.5), help="Offset subtracted from nominal d_min for all safety calculations.")
    parser.add_argument("--dt", type=float, default=_config_value("environment", "dt", 0.1))
    parser.add_argument("--frame-stride", type=int, default=_config_value("environment", "frame_stride", 1))
    parser.add_argument("--n-init", type=int, default=_config_value("environment", "n_init", 80))
    parser.add_argument("--n-eval", type=int, default=_config_value("environment", "n_eval", 400))
    parser.add_argument("--delta-f", type=float, default=_config_value("fa_sal", "delta_f", 0.1))
    parser.add_argument("--delta-g", type=float, default=_config_value("fa_sal", "delta_g", 0.1))
    parser.add_argument("--reset-context-each-round", action=argparse.BooleanOptionalAction, default=_config_value("environment", "reset_context_each_round", True))
    parser.add_argument("--safe-start-headway-max", type=float, default=_config_value("environment", "safe_start_headway_max", float("inf")))
    parser.add_argument("--safe-start-rel-speed-max", type=float, default=_config_value("environment", "safe_start_rel_speed_max", float("inf")))
    parser.add_argument("--safe-start-min-count", type=int, default=_config_value("environment", "safe_start_min_count", 0))
    parser.add_argument("--rmse-eval-headway-max", type=float, default=_config_value("environment", "rmse_eval_headway_max", 16.0), help="Maximum headway for Evaluation RMSE contexts; does not affect rollout starts.")
    parser.add_argument("--rmse-eval-rel-speed-max", type=float, default=_config_value("environment", "rmse_eval_rel_speed_max", 1.0), help="Maximum leader-minus-ego relative speed for Evaluation RMSE contexts; does not affect rollout starts.")
    parser.add_argument("--rmse-eval-min-count", type=int, default=_config_value("environment", "rmse_eval_min_count", 20))
    parser.add_argument("--residual-neighbors", type=int, default=_config_value("environment", "residual_neighbors", 16))
    parser.add_argument("--collision-absorbing", action=argparse.BooleanOptionalAction, default=_config_value("environment", "collision_absorbing", False), help="Treat headway violations as an absorbing collision state in the derived NGSIM benchmark.")
    parser.add_argument("--collision-headway", type=float, default=_config_value("environment", "collision_headway", 0.0), help="Headway assigned after an absorbing collision.")
    parser.add_argument("--collision-safety-penalty", type=float, default=_config_value("environment", "collision_safety_penalty", -10.0), help="Safety target assigned after an absorbing collision.")
    parser.add_argument("--dynamics-noise-std", type=float, default=_config_value("environment", "dynamics_noise_std", 0.15))
    parser.add_argument("--safety-noise-std", type=float, default=_config_value("environment", "safety_noise_std", 0.02))
    parser.add_argument("--kernel", choices=["se", "matern52"], default=_config_value("learning", "kernel", "se"))
    parser.add_argument("--length-scale", type=float, default=_config_value("learning", "length_scale", 1.0))
    parser.add_argument("--fa-beam-width", type=int, default=_config_value("fa_sal", "beam_width", 1))
    parser.add_argument("--fa-sal-l-ell", type=float, default=_config_value("fa_sal", "l_ell", 1.0))
    parser.add_argument("--fa-sal-l-ell-quantile", type=float, default=_config_value("fa_sal", "l_ell_quantile", 0.9))
    parser.add_argument("--fa-sal-l-ell-quantile-sweep", type=float, nargs="*", default=list(_config_value("fa_sal", "l_ell_quantile_sweep", ())))
    parser.add_argument("--fa-sal-l-ell-scale", type=float, default=_config_value("fa_sal", "l_ell_scale", 0.01))
    parser.add_argument("--fa-sal-l-ell-scale-sweep", type=float, nargs="*", default=list(_config_value("fa_sal", "l_ell_scale_sweep", ())))
    parser.add_argument("--fa-sal-uncertainty-criterion", choices=["logdet", "trace", "maxeig"], default=_config_value("fa_sal", "uncertainty_criterion", "logdet"))
    parser.add_argument("--fa-random-l-ell", type=float, default=_config_value("fa_random", "l_ell", 1.0))
    parser.add_argument("--fa-random-l-ell-quantile", type=float, default=_config_value("fa_random", "l_ell_quantile", 0.9))
    parser.add_argument("--fa-random-l-ell-quantile-sweep", type=float, nargs="*", default=list(_config_value("fa_random", "l_ell_quantile_sweep", ())))
    parser.add_argument("--fa-random-l-ell-scale", type=float, default=_config_value("fa_random", "l_ell_scale", 0.01))
    parser.add_argument("--fa-random-l-ell-scale-sweep", type=float, nargs="*", default=list(_config_value("fa_random", "l_ell_scale_sweep", ())))
    parser.add_argument("--salnx-alpha", type=float, default=_config_value("salnx", "alpha", 0.2))
    parser.add_argument("--salnx-alpha-sweep", type=float, nargs="*", default=list(_config_value("salnx", "alpha_sweep", ())))
    parser.add_argument("--salnx-mc-samples", type=int, default=_config_value("salnx", "mc_samples", 128))
    parser.add_argument("--salnx-uncertainty-criterion", choices=["logdet", "trace", "maxeig"], default=_config_value("salnx", "uncertainty_criterion", "logdet"))
    parser.add_argument("--tebbe-alpha", type=float, default=_config_value("tebbe", "alpha", 0.2))
    parser.add_argument("--tebbe-uncertainty-criterion", choices=["logdet", "trace", "maxeig"], default=_config_value("tebbe", "uncertainty_criterion", "logdet"))
    parser.add_argument("--tebbe-confidence-delta", type=float, default=_config_value("tebbe", "confidence_delta", 1e-2))
    parser.add_argument("--tebbe-sample-start", type=int, default=_config_value("tebbe", "sample_start", 32))
    parser.add_argument("--tebbe-sample-stages", type=int, default=_config_value("tebbe", "sample_stages", 6))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = time.perf_counter()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    data = load_ngsim_data(
        data_path=args.data_path,
        max_transitions=args.max_transitions,
        unit_scale=args.distance_unit_scale,
        min_headway=_effective_min_headway(args),
        dt=args.dt,
        frame_stride=args.frame_stride,
        residual_neighbors=args.residual_neighbors,
        seed=args.seed,
        collision_absorbing=args.collision_absorbing,
        collision_headway=args.collision_headway,
        collision_safety_penalty=args.collision_safety_penalty,
    )
    variants = _build_method_variants(args)
    all_summaries = {}
    all_trials = {}
    safe_context_count = int(_current_safe_indices(data).size)
    challenging_context_count = int(_challenging_safe_start_indices(args, data).size)
    rmse_context_args = argparse.Namespace(
        safe_start_headway_max=float(args.rmse_eval_headway_max),
        safe_start_rel_speed_max=float(args.rmse_eval_rel_speed_max),
        safe_start_min_count=int(args.rmse_eval_min_count),
    )
    rmse_eval_context_count = int(_challenging_safe_start_indices(rmse_context_args, data).size)
    action_sequence_counts = {
        int(_namespace_with_overrides(args, dict(variant.get("overrides", {}))).horizon): len(
            _make_action_sequences(args.action_values, int(_namespace_with_overrides(args, dict(variant.get("overrides", {}))).horizon))
        )
        for variant in variants
    }
    print(f"Loaded {data.X.shape[0]} NGSIM car-following transitions from {args.data_path}")
    action_count_text = ", ".join(f"m={h}: {count}" for h, count in sorted(action_sequence_counts.items()))
    print(f"Action sequences by horizon: {action_count_text} | step_dt={data.dt:.3f}s")
    print(
        f"Safety headway: nominal={float(args.min_headway):g}m "
        f"- offset={float(args.safety_headway_offset):g}m "
        f"= effective={data.min_headway:g}m"
    )
    print(f"Rollout start contexts: {challenging_context_count} / {safe_context_count}")
    print(f"Evaluation RMSE eval contexts: {rmse_eval_context_count} / {safe_context_count}")
    print(
        "Collision absorbing:",
        bool(args.collision_absorbing),
        f"| headway={float(args.collision_headway):g}",
        f"| safety_penalty={float(args.collision_safety_penalty):g}",
    )
    total_trial_count = int(args.trials) * max(1, len(variants))
    use_parallel = int(args.parallel_workers) != 1 and total_trial_count > 1
    worker_count = 1
    if use_parallel:
        auto_workers = os.cpu_count() or 1
        requested_workers = int(args.parallel_workers)
        worker_count = max(1, requested_workers) if requested_workers > 0 else min(auto_workers, total_trial_count)
        use_parallel = worker_count > 1
    if use_parallel:
        print(f"Parallel trial mode: using {worker_count} worker processes.")
        print(f"BLAS thread limits per worker: {BLAS_THREAD_LIMITS}")
    variant_iter = _iter_progress(
        variants,
        desc="NGSIM variants",
        dynamic_ncols=True,
        disable=bool(args.quiet),
    )
    for variant in variant_iter:
        method = str(variant["base_method"])
        label = str(variant["label"])
        variant_args = _namespace_with_overrides(args, dict(variant.get("overrides", {})))
        action_sequences = _make_action_sequences(args.action_values, int(variant_args.horizon))
        trials = [None] * int(args.trials)
        if use_parallel:
            tasks = [
                (variant_args, method, int(args.seed) + int(trial_idx), data, action_sequences)
                for trial_idx in range(int(args.trials))
            ]
            executor_cls = ProcessPoolExecutor
            try:
                executor = executor_cls(max_workers=worker_count)
            except Exception:
                executor_cls = ThreadPoolExecutor
                executor = executor_cls(max_workers=worker_count)
                print("Process-based parallelism is unavailable here; falling back to thread-based execution.")
            with executor:
                future_to_idx = {
                    executor.submit(_run_trial_task, task): trial_idx
                    for trial_idx, task in enumerate(tasks)
                }
                completed_iter = _iter_progress(
                    as_completed(future_to_idx),
                    total=len(future_to_idx),
                    desc=f"{label} trials",
                    leave=False,
                    dynamic_ncols=True,
                    disable=bool(args.quiet),
                )
                for future in completed_iter:
                    trial_idx = future_to_idx[future]
                    trials[trial_idx] = future.result()
        else:
            trial_iter = _iter_progress(
                range(int(args.trials)),
                desc=f"{label} trials",
                leave=False,
                dynamic_ncols=True,
                disable=bool(args.quiet),
            )
            for trial_idx in trial_iter:
                trial_logs = run_trial(
                    variant_args,
                    method=method,
                    seed=int(args.seed) + int(trial_idx),
                    data=data,
                    action_sequences=action_sequences,
                )
                trials[trial_idx] = trial_logs
        if any(trial is None for trial in trials):
            raise RuntimeError("One or more trials did not return results.")
        trials = list(trials)
        summary = summarize_trials(trials)
        all_summaries[label] = summary
        all_trials[label] = trials
        print(_variant_result_line(label, summary))
        elapsed_seconds = time.perf_counter() - start
        payload = _build_payload(
            args,
            data,
            variants,
            all_summaries,
            all_trials,
            elapsed_seconds,
            worker_count,
            use_parallel,
        )
        _write_variant_outputs(args, variant, payload, all_summaries, all_trials, elapsed_seconds)
        _write_aggregate_outputs(args, payload, elapsed_seconds)

    print("All requested NGSIM variants completed and saved.")


if __name__ == "__main__":
    main()
