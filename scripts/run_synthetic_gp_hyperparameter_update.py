

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import experiment as synthetic_experiment
import gp_models
from config import DEFAULT_EXPERIMENT_CONFIG
from planning import ConfidenceSchedule
from synthetic_preset import get_synthetic_environment_preset


CONDITIONS = {
    "nominal_fixed": {"length_scale": 1.0, "variance": 1.0, "retrain": False},
    "joint_low_fixed": {"length_scale": 0.5, "variance": 0.5, "retrain": False},
    "joint_low_retrained": {"length_scale": 0.5, "variance": 0.5, "retrain": True},
    "joint_high_fixed": {"length_scale": 2.0, "variance": 2.0, "retrain": False},
    "joint_high_retrained": {"length_scale": 2.0, "variance": 2.0, "retrain": True},
}


def _nll(log_params: np.ndarray, X: np.ndarray, y: np.ndarray, noise_var: float) -> float:
    length_scale = float(np.exp(log_params[0]))
    variance = float(np.exp(log_params[1]))
    cfg = gp_models.KernelConfig(kind="se", variance=variance, length_scale=length_scale)
    y = np.asarray(y, dtype=float).reshape(-1)
    y_scale = float(np.std(y))
    if y_scale < 1e-8:
        y_scale = 1.0
    y_norm = (y - float(np.mean(y))) / y_scale
    K = gp_models.kernel_matrix(np.asarray(X, dtype=float), np.asarray(X, dtype=float), cfg)
    C = K + (float(noise_var) + 1e-8) * np.eye(X.shape[0])
    try:
        L = np.linalg.cholesky(C)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    except np.linalg.LinAlgError:
        return 1e100
    return float(
        0.5 * y_norm @ alpha
        + np.sum(np.log(np.diag(L)))
        + 0.5 * X.shape[0] * math.log(2.0 * math.pi)
    )


def _optimize_kernel(gp: gp_models.GaussianProcess1D, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    initial = np.log([gp.kernel_cfg.length_scale, gp.kernel_cfg.variance])
    result = minimize(
        _nll,
        initial,
        args=(np.asarray(X, dtype=float), np.asarray(y, dtype=float), gp.noise_var),
        method="L-BFGS-B",
        bounds=[(math.log(0.1), math.log(10.0)), (math.log(0.1), math.log(10.0))],
        options={"maxiter": 50, "ftol": 1e-9},
    )
    gp.kernel_cfg.length_scale = float(np.exp(result.x[0]))
    gp.kernel_cfg.variance = float(np.exp(result.x[1]))
    return {
        "length_scale": gp.kernel_cfg.length_scale,
        "variance": gp.kernel_cfg.variance,
        "success": bool(result.success),
        "nll": float(result.fun),
        "nit": int(result.nit),
    }


def _common_kwargs() -> dict[str, Any]:
    defaults = DEFAULT_EXPERIMENT_CONFIG
    fa = defaults.fa_sal
    data = defaults.initial_data
    nominal_mpc = defaults.nominal_mpc
    salnx = defaults.salnx
    tebbe = defaults.tebbe
    safe_exploration = defaults.safe_exploration
    env = get_synthetic_environment_preset("penalty_stress")
    schedule = ConfidenceSchedule(
        delta_f=fa.delta_f,
        delta_g=fa.delta_g,
        y_min=fa.state_y_min,
        y_max=fa.state_y_max,
        y_grid_points=fa.state_grid_points,
    )
    return {
        "kernel_kind": "se",
        "T": 150,
        "fa_horizon": 4,
        "fa_beam_width": fa.beam_width,
        "fa_continuation_policy": "uncertainty-max",
        "fa_random_horizon": 4,
        "pointwise_horizon": 4,
        "safe_exploration_horizon": 1,
        "salnx_horizon": 4,
        "tebbe_horizon": 4,
        "nominal_mpc_horizon": nominal_mpc.horizon,
        "control_target": nominal_mpc.target,
        "control_q_y": nominal_mpc.q_y,
        "control_q_velocity": nominal_mpc.q_velocity,
        "control_r_u": nominal_mpc.r_u,
        "control_r_delta_u": nominal_mpc.r_delta_u,
        "control_terminal_weight": nominal_mpc.terminal_weight,
        "control_beam_width": nominal_mpc.beam_width,
        "active_mpc_information_weight": 1.0,
        "nominal_mpc_safety_margin": nominal_mpc.safety_margin,
        "n_init": data.n_init,
        "epsilon_margin": fa.epsilon_margin,
        "recovery_eval_interval": 5,
        "evaluation_horizon": 4,
        "episodic_reset_period": defaults.general.episodic_reset_period,
        "confidence_schedule": schedule,
        "fa_lx": fa.lx,
        "fa_ly": fa.ly,
        "fa_lf": fa.lf,
        "fa_lf_quantile": fa.lf_quantile,
        "fa_lf_estimator": "legacy_axis_quantile",
        "fa_lf_scale": 1.0,
        "fa_lf_cap": fa.lf_cap,
        "fa_l_ell": fa.l_ell,
        "fa_l_ell_quantile": fa.l_ell_quantile,
        "fa_l_ell_scale": 0.14,
        "fa_buffer_weight": fa.buffer_weight,
        "alpha_safety": salnx.alpha,
        "safe_exploration_alpha": safe_exploration.alpha,
        "salnx_criterion": salnx.criterion,
        "salnx_joint_mc_samples": salnx.joint_mc_samples,
        "tebbe_alpha": tebbe.alpha,
        "tebbe_criterion": tebbe.criterion,
        "tebbe_confidence_delta": tebbe.confidence_delta,
        "tebbe_sample_start": tebbe.sample_start,
        "tebbe_sample_stages": tebbe.sample_stages,
        "tebbe_endpoint_candidates": tebbe.endpoint_candidates,
        "tebbe_legacy_mc_prefix": False,
        "tebbe_local_refinement": False,
        "n_eval": 250,
        "snapshot_rounds": (),
        "snapshot_grid_size": defaults.snapshots.grid_size,
        "snapshot_u_slice": defaults.snapshots.u_slice,
        "env_kind": env["kind"],
        "env_y_max": env["y_max"],
        "env_noise_std": env["dynamics_noise_std"],
        "safety_noise_std": env["safety_noise_std"],
        "runaway_center": env["runaway_center"],
        "runaway_halfwidth": env["runaway_halfwidth"],
        "runaway_strength": env["runaway_strength"],
        "runaway_velocity_threshold": env["runaway_velocity_threshold"],
        "funnel_center": env["funnel_center"],
        "funnel_halfwidth": env["funnel_halfwidth"],
        "funnel_runaway_strength": env["funnel_runaway_strength"],
        "funnel_velocity_threshold": env["funnel_velocity_threshold"],
        "funnel_brake_fade": env["funnel_brake_fade"],
        "lattice_start": env["lattice_start"],
        "lattice_period": env["lattice_period"],
        "lattice_count": env["lattice_count"],
        "lattice_halfwidth": env["lattice_halfwidth"],
        "lattice_tailwidth": env["lattice_tailwidth"],
        "lattice_runaway_strength": env["lattice_runaway_strength"],
        "lattice_tail_runaway_multiplier": env["lattice_tail_runaway_multiplier"],
        "lattice_velocity_threshold": env["lattice_velocity_threshold"],
        "lattice_brake_fade": env["lattice_brake_fade"],
        "lattice_tail_brake_fade": env["lattice_tail_brake_fade"],
        "smooth_switches": bool(env.get("smooth_switches", False)),
        "switch_sharpness": float(env.get("switch_sharpness", 8.0)),
        "goal_y_min": env["goal_y_min"],
        "goal_y_max": env["goal_y_max"],
        "fail_y": env.get("fail_y", 5.2),
        "failure_absorbing": env.get("failure_absorbing", True),
        "init_y_range": (data.y_min, data.y_max),
        "init_velocity_range": (data.v_min, data.v_max),
        "start_y_tm1": data.start_y_tm1,
        "start_y_t": data.start_y_t,
        "show_progress": False,
        "progress_desc": "",
    }


def _scalar_metrics(trial: dict[str, Any]) -> dict[str, float]:
    unsafe = np.asarray(trial["unsafe_transitions"], dtype=float)
    chosen = np.asarray(trial["chosen_actions"], dtype=float)
    active = np.isfinite(chosen)
    return {
        "rmse": float(np.asarray(trial["dynamics_rmse"], dtype=float)[-1]),
        "svr": float(np.sum(unsafe[active]) / max(1, np.sum(active))),
        "d_svr": float(np.asarray(trial["dead_end_safety_violation_rate"], dtype=float)[-1]),
        "safety_iou": float(np.asarray(trial["safety_set_recovery_iou"], dtype=float)[-1]),
        "de_iou": float(np.asarray(trial["dead_end_free_recovery_iou"], dtype=float)[-1]),
        "false_certification": float(np.asarray(trial["false_certification_rate"], dtype=float)[-1]),
        "no_feasible_rate": float(np.mean(np.asarray(trial["no_feasible_action"], dtype=float)[active])),
    }


def _run_task(task: tuple[str, int]) -> dict[str, Any]:
    condition_name, seed = task
    condition = CONDITIONS[condition_name]
    registry: dict[str, Any] = {}
    original_dyn = synthetic_experiment.DynamicsGP
    original_safe = synthetic_experiment.SafetyGP
    original_env = synthetic_experiment.NARXDoubleIntegratorEnv

    class RecordingEnvironment(original_env):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.final_X = []
            self.final_y = []
            self.final_g = []
            self._inside_recorded_step = False
            registry["environment"] = self

        def step(self, z: np.ndarray, noise: bool = True) -> float:
            self._inside_recorded_step = True
            try:
                value = super().step(z, noise=noise)
            finally:
                self._inside_recorded_step = False
            if registry.get("collect_round_150", False) and noise:
                self.final_X.append(np.asarray(z, dtype=float).copy())
                self.final_y.append(float(value))
            return float(value)

        def failure_observation(self, z: np.ndarray, noise: bool = True) -> float:
            value = super().failure_observation(z, noise=noise)
            if registry.get("collect_round_150", False) and noise and not self._inside_recorded_step and self.final_y:
                self.final_y[-1] = float(value)
            return float(value)

        def observe_safety(self, z: np.ndarray, noise: bool = True) -> float:
            value = super().observe_safety(z, noise=noise)
            if registry.get("collect_round_150", False) and noise:
                self.final_g.append(float(value))
            return float(value)

    class AdaptiveDynamicsGP(original_dyn):
        def __init__(self, kernel_cfg: gp_models.KernelConfig, noise_std: float):
            own = gp_models.KernelConfig("se", condition.get("variance_f", condition["variance"]), condition.get("length_scale_f", condition["length_scale"]))
            super().__init__(own, noise_std)
            self.fit_calls = 0
            self.hyperparameter_history: list[dict[str, Any]] = []
            registry["dynamics"] = self

        def fit(self, X: np.ndarray, y: np.ndarray, kernel_train: np.ndarray | None = None) -> None:
            self.fit_calls += 1
            self.last_X = np.asarray(X, dtype=float).copy()
            self.last_y = np.asarray(y, dtype=float).copy()
            retrain_start = int(condition.get("retrain_start", 10))
            if condition["retrain"] and self.fit_calls > 1 and (self.fit_calls - 1) >= retrain_start and (self.fit_calls - 1) % 10 == 0:
                update = _optimize_kernel(self.gp, X, y)
                update["after_round"] = self.fit_calls - 1
                self.hyperparameter_history.append(update)
            super().fit(X, y)

    class AdaptiveSafetyGP(original_safe):
        def __init__(self, kernel_cfg: gp_models.KernelConfig, noise_std: float = 1e-4):
            own = gp_models.KernelConfig("se", condition.get("variance_g", condition["variance"]), condition.get("length_scale_g", condition["length_scale"]))
            super().__init__(own, noise_std)
            self.fit_calls = 0
            self.hyperparameter_history: list[dict[str, Any]] = []
            registry["safety"] = self

        def fit(self, X: np.ndarray, y: np.ndarray, kernel_train: np.ndarray | None = None) -> None:
            self.fit_calls += 1
            self.last_X = np.asarray(X, dtype=float).copy()
            self.last_y = np.asarray(y, dtype=float).copy()
            retrain_start = int(condition.get("retrain_start", 10))
            if condition["retrain"] and self.fit_calls > 1 and (self.fit_calls - 1) >= retrain_start and (self.fit_calls - 1) % 10 == 0:
                update = _optimize_kernel(self.gp, X, y)
                update["after_round"] = self.fit_calls - 1
                self.hyperparameter_history.append(update)
            super().fit(X, y)
            if condition["retrain"] and self.fit_calls == 150:
                registry["collect_round_150"] = True

    synthetic_experiment.DynamicsGP = AdaptiveDynamicsGP
    synthetic_experiment.SafetyGP = AdaptiveSafetyGP
    synthetic_experiment.NARXDoubleIntegratorEnv = RecordingEnvironment
    started = time.perf_counter()
    try:
        trial = synthetic_experiment.run_trial(
            planner_type="future",
            seed=seed,
            **_common_kwargs(),
        )
    finally:
        synthetic_experiment.DynamicsGP = original_dyn
        synthetic_experiment.SafetyGP = original_safe
        synthetic_experiment.NARXDoubleIntegratorEnv = original_env
    dyn = registry["dynamics"]
    safe = registry["safety"]
    env = registry["environment"]
    if condition["retrain"]:
        if not (len(env.final_X) == len(env.final_y) == len(env.final_g)):
            raise RuntimeError(f"Round-150 observation capture mismatch: X={len(env.final_X)}, y={len(env.final_y)}, g={len(env.final_g)}")
        final_X = np.vstack([dyn.last_X, np.asarray(env.final_X, dtype=float)])
        final_y = np.concatenate([dyn.last_y, np.asarray(env.final_y, dtype=float)])
        final_g = np.concatenate([safe.last_y, np.asarray(env.final_g, dtype=float)])
        dyn_update = _optimize_kernel(dyn.gp, final_X, final_y)
        dyn_update["after_round"] = 150
        dyn.hyperparameter_history.append(dyn_update)
        safe_update = _optimize_kernel(safe.gp, final_X, final_g)
        safe_update["after_round"] = 150
        safe.hyperparameter_history.append(safe_update)
    runtime = time.perf_counter() - started
    return {
        "condition": condition_name,
        "seed": seed,
        "metrics": _scalar_metrics(trial),
        "runtime_seconds": runtime,
        "final_hyperparameters": {
            "length_scale_f": dyn.gp.kernel_cfg.length_scale,
            "length_scale_g": safe.gp.kernel_cfg.length_scale,
            "variance_f": dyn.gp.kernel_cfg.variance,
            "variance_g": safe.gp.kernel_cfg.variance,
        },
        "hyperparameter_history": {
            "dynamics": dyn.hyperparameter_history,
            "safety": safe.hyperparameter_history,
        },
    }


def _summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    scalar_keys = list(trials[0]["metrics"]) + ["runtime_seconds"]
    for key in scalar_keys:
        values = np.asarray(
            [trial["runtime_seconds"] if key == "runtime_seconds" else trial["metrics"][key] for trial in trials],
            dtype=float,
        )
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    for key in ("length_scale_f", "length_scale_g", "variance_f", "variance_g"):
        values = np.asarray([trial["final_hyperparameters"][key] for trial in trials], dtype=float)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rounds-smoke", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result/synthetic_gp_update/results.json"),
    )
    args = parser.parse_args()
    if args.rounds_smoke:
        raise ValueError("Smoke rounds are run by the separate test command; production is fixed at 150 rounds.")
    tasks = [(condition, seed) for condition in CONDITIONS for seed in range(args.trials)]
    results: list[dict[str, Any]] = []
    print(f"Running {len(tasks)} FA-SAL paired trials with {args.workers} workers.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_task, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(tasks)}] {result['condition']} seed={result['seed']} "
                f"runtime={result['runtime_seconds']:.1f}s",
                flush=True,
            )
    ordered = sorted(results, key=lambda item: (list(CONDITIONS).index(item["condition"]), item["seed"]))
    by_condition = {
        condition: [item for item in ordered if item["condition"] == condition]
        for condition in CONDITIONS
    }
    payload = {
        "experiment": {
            "benchmark": "penalty_stress",
            "method": "FA-SAL",
            "horizon": 4,
            "rounds": 150,
            "paired_seeds": list(range(args.trials)),
            "kernel": "isotropic_se",
            "bounds": {"length_scale": [0.1, 10.0], "variance": [0.1, 10.0]},
            "retrain_after_rounds": list(range(10, 151, 10)),
            "l_ell_scale": 0.14,
            "conditions": CONDITIONS,
            "config": _json_ready(
                {
                    key: value
                    for key, value in _common_kwargs().items()
                    if key not in {"n_eval", "recovery_eval_interval"}
                }
            ),
        },
        "trials": ordered,
        "summary": {condition: _summarize(items) for condition, items in by_condition.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, ConfidenceSchedule):
        return value.__dict__
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
