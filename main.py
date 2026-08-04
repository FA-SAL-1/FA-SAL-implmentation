import argparse
import hashlib
import multiprocessing
import os
import pickle
from blas_threads import configure_blas_threads

BLAS_THREAD_LIMITS = configure_blas_threads()

import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
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

from config import DEFAULT_EXPERIMENT_CONFIG
from experiment import run_trial, summarize_trials
from planning import ConfidenceSchedule
from synthetic_preset import (
    SYNTHETIC_ENVIRONMENT_PRESETS,
    get_synthetic_environment_preset,
)


def _json_ready(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _format_sweep_value(value: float) -> str:
    return np.format_float_positional(
        float(value),
        precision=8,
        unique=True,
        fractional=False,
        trim="-",
    )


def _format_file_suffix(value: float) -> str:
    formatted = _format_sweep_value(value)
    return formatted.replace("-", "m").replace(".", "p")


def _safe_token(value: object) -> str:
    return str(value).strip().replace(" ", "_").replace("-", "_").replace(".", "p")


def _synthetic_result_setting_suffix(base_config: dict, variant: dict) -> str:
    horizon_by_method = {
        "fa_sal": base_config["fa_horizon"],
        "myopic_fa_sal": 1,
        "fa_random": base_config["fa_random_horizon"],
        "safe_exploration": base_config["safe_exploration_horizon"],
        "sal": base_config["safe_exploration_horizon"],
        "salnx": base_config["salnx_horizon"],
        "tebbe_abm": base_config["tebbe_horizon"],
        "pointwise": base_config["pointwise_horizon"],
        "nominal_mpc": base_config["nominal_mpc_horizon"],
        "al_mpc": base_config["nominal_mpc_horizon"],
    }
    method = str(variant["base_method"])
    planning_horizon = int(horizon_by_method.get(method, base_config["evaluation_horizon"]))
    parts = [
        f"m{planning_horizon}",
        f"evalm{int(base_config['evaluation_horizon'])}",
    ]
    if method == "tebbe_abm":
        parts.append(f"ucand{int(base_config['tebbe_endpoint_candidates'])}")
        parts.append(f"mcprefix{int(bool(base_config['tebbe_legacy_mc_prefix']))}")
        parts.append(f"localref{int(bool(base_config['tebbe_local_refinement']))}")
    parts.extend(
        [
            f"lfest{_safe_token(base_config['fa_lf_estimator'])}",
            f"lfq{_format_file_suffix(float(base_config['fa_lf_quantile']))}",
            f"lfgrid{int(base_config['state_grid_points'])}",
            f"rounds{int(base_config['rounds'])}",
            f"trials{int(base_config['trials'])}",
        ]
    )
    return "_".join(parts)


def _build_method_variants(
    selected_methods,
    method_catalog,
    fa_continuation_policy_sweep,
    fa_lf_scale_sweep,
    fa_l_ell_scale_sweep,
    salnx_alpha_sweep,
    tebbe_alpha_sweep,
    tebbe_confidence_delta_sweep,
):
    variants = []
    for method_name in selected_methods:
        meta = method_catalog[method_name]
        if method_name in {"fa_sal", "myopic_fa_sal", "fa_random"} and (
            fa_continuation_policy_sweep or fa_lf_scale_sweep or fa_l_ell_scale_sweep
        ):
            policies = (
                list(fa_continuation_policy_sweep)
                if method_name == "fa_sal" and fa_continuation_policy_sweep
                else [None]
            )
            lf_scales = list(fa_lf_scale_sweep) if fa_lf_scale_sweep else [None]
            l_ell_scales = list(fa_l_ell_scale_sweep) if fa_l_ell_scale_sweep else [None]
            for policy in policies:
                for lf_scale in lf_scales:
                    for l_ell_scale in l_ell_scales:
                        overrides = {}
                        label_parts = []
                        result_parts = [method_name]
                        if policy is not None:
                            policy_value = str(policy)
                            overrides["fa_continuation_policy"] = policy_value
                            label_parts.append(f"policy={policy_value}")
                            result_parts.append(f"policy_{_safe_token(policy_value)}")
                        if lf_scale is not None:
                            lf_value = float(lf_scale)
                            overrides["fa_lf_scale"] = lf_value
                            label_parts.append(f"lf_scale={_format_sweep_value(lf_value)}")
                            result_parts.append(f"lf_scale_{_format_file_suffix(lf_value)}")
                        if l_ell_scale is not None:
                            l_ell_value = float(l_ell_scale)
                            overrides["fa_l_ell_scale"] = l_ell_value
                            label_parts.append(f"l_ell_scale={_format_sweep_value(l_ell_value)}")
                            result_parts.append(f"l_ell_scale_{_format_file_suffix(l_ell_value)}")
                        variants.append(
                            {
                                "base_method": method_name,
                                "planner_type": str(meta["planner_type"]),
                                "seed_offset": int(meta["seed_offset"]),
                                "label": f"{meta['label']} ({', '.join(label_parts)})",
                                "result_id": "_".join(result_parts),
                                "overrides": overrides,
                            }
                        )
        elif method_name == "salnx" and salnx_alpha_sweep:
            for alpha in salnx_alpha_sweep:
                alpha_value = float(alpha)
                formatted = _format_sweep_value(alpha_value)
                suffix = _format_file_suffix(alpha_value)
                variants.append(
                    {
                        "base_method": method_name,
                        "planner_type": str(meta["planner_type"]),
                        "seed_offset": int(meta["seed_offset"]),
                        "label": f"{meta['label']} (alpha={formatted})",
                        "result_id": f"{method_name}_alpha_{suffix}",
                        "overrides": {"alpha_safety": alpha_value},
                    }
                )
        elif method_name == "tebbe_abm" and (
            tebbe_alpha_sweep or tebbe_confidence_delta_sweep
        ):
            alpha_values = list(tebbe_alpha_sweep) if tebbe_alpha_sweep else [None]
            delta_values = (
                list(tebbe_confidence_delta_sweep)
                if tebbe_confidence_delta_sweep
                else [None]
            )
            for alpha in alpha_values:
                for confidence_delta in delta_values:
                    overrides = {}
                    label_parts = []
                    result_parts = [method_name]
                    if alpha is not None:
                        alpha_value = float(alpha)
                        overrides["tebbe_alpha"] = alpha_value
                        label_parts.append(f"alpha={_format_sweep_value(alpha_value)}")
                        result_parts.append(f"alpha_{_format_file_suffix(alpha_value)}")
                    if confidence_delta is not None:
                        delta_value = float(confidence_delta)
                        overrides["tebbe_confidence_delta"] = delta_value
                        label_parts.append(f"confidence_delta={_format_sweep_value(delta_value)}")
                        result_parts.append(f"confidence_delta_{_format_file_suffix(delta_value)}")
                    variants.append(
                        {
                            "base_method": method_name,
                            "planner_type": str(meta["planner_type"]),
                            "seed_offset": int(meta["seed_offset"]),
                            "label": f"{meta['label']} ({', '.join(label_parts)})",
                            "result_id": "_".join(result_parts),
                            "overrides": overrides,
                        }
                    )
        else:
            variants.append(
                {
                    "base_method": method_name,
                    "planner_type": str(meta["planner_type"]),
                    "seed_offset": int(meta["seed_offset"]),
                    "label": str(meta["label"]),
                    "result_id": str(method_name),
                    "overrides": {},
                }
            )
    return variants


def _run_trial_task(task):
    variant = dict(task["variant"])
    common_kwargs = dict(task["common_kwargs"])
    common_kwargs.update(variant.get("overrides", {}))
    task_start = time.perf_counter()
    identity = multiprocessing.current_process()._identity
    worker_ordinal = int(identity[0]) if identity else 1
    worker_count = max(1, int(task.get("worker_count", 1)))
    progress_position = 1 + ((worker_ordinal - 1) % worker_count)
    print(
        f"Starting {variant['label']} seed_index={int(task['seed_index'])} "
        f"pid={os.getpid()}",
        flush=True,
    )
    trial = run_trial(
        planner_type=str(variant["planner_type"]),
        seed=int(task["seed"]),
        snapshot_rounds=tuple(task["snapshot_rounds"]),
        show_progress=True,
        progress_desc=(
            f"{variant['label']} seed={int(task['seed_index'])}"
        ),
        progress_position=progress_position,
        heartbeat_interval=int(task.get("trial_progress_interval", 0)),
        heartbeat_label=(
            f"{variant['label']} seed_index={int(task['seed_index'])} "
            f"pid={os.getpid()}"
        ),
        **common_kwargs,
    )
    checkpoint_dir = task.get("checkpoint_dir")
    if checkpoint_dir is not None:
        _save_trial_checkpoint(Path(checkpoint_dir), task, trial)
    print(
        f"Finished {variant['label']} seed_index={int(task['seed_index'])} "
        f"pid={os.getpid()} elapsed={time.perf_counter() - task_start:.1f}s",
        flush=True,
    )
    return {
        "variant_label": str(variant["label"]),
        "seed_index": int(task["seed_index"]),
        "trial": trial,
    }


def _checkpoint_signature(task):
    payload = {
        "variant": task["variant"], "seed_index": int(task["seed_index"]),
        "seed": int(task["seed"]), "snapshot_rounds": tuple(task["snapshot_rounds"]),
        "common_kwargs": task["common_kwargs"],
    }
    return hashlib.sha256(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def _checkpoint_path(checkpoint_dir, task):
    variant = task["variant"]
    name = (f"{_safe_token(variant['result_id'])}_seed_index_{int(task['seed_index'])}"
            f"_seed_{int(task['seed'])}_{_checkpoint_signature(task)[:16]}.pkl")
    return checkpoint_dir / name


def _load_trial_checkpoint(checkpoint_dir, task):
    path = _checkpoint_path(checkpoint_dir, task)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    return payload.get("trial") if payload.get("signature") == _checkpoint_signature(task) else None


def _save_trial_checkpoint(checkpoint_dir, task, trial):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, task)
    tmp = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    payload = {"signature": _checkpoint_signature(task), "trial": trial}
    try:
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()
    return path


def _compact_trial_record(trial):
    active_rounds = int(np.sum(~np.isnan(np.asarray(trial["chosen_actions"], dtype=float))))
    final_idx = max(0, active_rounds - 1)

    def final_value(key, default=0.0):
        values = np.asarray(trial.get(key, []), dtype=float)
        if values.size == 0:
            return float(default)
        return float(values[min(final_idx, values.size - 1)])

    unsafe = np.asarray(trial.get("unsafe_transitions", []), dtype=float)
    svr = float(np.sum(unsafe) / active_rounds) if active_rounds > 0 else 0.0
    feasible_ratios = np.asarray(trial.get("feasible_ratios", []), dtype=float)
    no_feasible = np.asarray(trial.get("no_feasible_action", []), dtype=float)

    def mean_active(key):
        values = np.asarray(trial.get(key, []), dtype=float)
        return float(np.mean(values[:active_rounds])) if active_rounds > 0 else 0.0

    return {
        "seed": int(trial.get("seed", -1)),
        "active_rounds": active_rounds,
        "estimated_lf": float(trial.get("estimated_lf", np.nan)),
        "effective_lf": float(trial.get("effective_lf", np.nan)),
        "runtime_seconds": float(trial.get("runtime_seconds", np.nan)),
        "peak_rss_mib": float(trial.get("peak_rss_mib", np.nan)),
        "svr": svr,
        "d_svr": final_value("dead_end_safety_violation_rate"),
        "rmse": final_value("dynamics_rmse"),
        "false_certification_rate": final_value("false_certification_rate"),
        "safety_iou": final_value("safety_set_recovery_iou"),
        "dead_end_iou": final_value("dead_end_free_recovery_iou"),
        "mean_feasible_ratio": float(np.mean(feasible_ratios[:active_rounds]))
        if active_rounds > 0
        else 0.0,
        "no_feasible_rate": float(np.mean(no_feasible[:active_rounds]))
        if active_rounds > 0
        else 0.0,
        "final_beta_f": final_value("beta_f"),
        "final_beta_g": final_value("beta_g"),
        "final_effective_l_ell": final_value("effective_l_ell"),
        "mean_training_size": mean_active("training_sizes"),
        "mean_candidate_count": mean_active("candidate_counts"),
        "mean_gp_fit_seconds": mean_active("gp_fit_seconds"),
        "mean_planning_seconds": mean_active("planning_seconds"),
        "mean_evaluation_seconds": mean_active("evaluation_seconds"),
        "mean_round_seconds": mean_active("round_total_seconds"),
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = DEFAULT_EXPERIMENT_CONFIG
    general = defaults.general
    fa_sal = defaults.fa_sal
    fa_random = defaults.fa_random
    pointwise = defaults.pointwise
    safe_exploration = defaults.safe_exploration
    salnx = defaults.salnx
    tebbe = defaults.tebbe
    nominal_mpc = defaults.nominal_mpc
    env = defaults.environment
    data = defaults.initial_data
    snapshots = defaults.snapshots
    parser = argparse.ArgumentParser(description="Future-Aware Safe Active Learning (FA-SAL) experiment runner.")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to run. Example: --methods fa_sal fa_random tebbe_abm")
    parser.add_argument("--allow-fewer-trials", action="store_true", help="Respect --trials even when it is below the default of 5.")
    parser.add_argument("--rounds", type=int, default=general.rounds, help="Number of online active-learning rounds per trial.")
    parser.add_argument("--trials", type=int, default=general.trials, help="Number of random seeds to average over.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Add this offset to every trial seed; use a nonzero value for independent validation runs.")
    parser.add_argument("--result-dir", type=Path, default=None, help="Result JSON directory. Defaults to the configured project result directory.")
    parser.add_argument("--r_i", dest="recovery_eval_interval", type=int, default=general.recovery_eval_interval, help="Compute dead-end-free recovery metrics every K rounds and reuse the latest value in between. Use 1 to evaluate every round.")
    parser.add_argument("--trial-progress-interval", type=int, default=0, help="Optional plain-text worker heartbeat interval. The per-trial tqdm bars are enabled independently; use 0 to disable heartbeats.")
    parser.add_argument("--evaluation-horizon", type=int, default=general.evaluation_horizon, help="Common oracle horizon used for dead-end/theorem/recovery metrics across all algorithms.")
    parser.add_argument("--fa-horizon", type=int, default=fa_sal.horizon, help="Planning horizon used by FA-SAL.")
    parser.add_argument("--fa-beam-width", type=int, default=fa_sal.beam_width, help="Beam width used by FA-SAL when certifying safe continuations.")
    parser.add_argument("--fa-continuation-policy", choices=["uncertainty-max", "random-safe", "greedy-margin"], default=fa_sal.continuation_policy)
    parser.add_argument("--fa-continuation-policy-sweep", choices=["uncertainty-max", "random-safe", "greedy-margin"], nargs="*", default=list(fa_sal.continuation_policy_sweep))
    parser.add_argument("--fa-random-horizon", type=int, default=fa_random.horizon, help="Planning horizon used by the FA-SAL-Random baseline.")
    parser.add_argument("--pointwise-horizon", type=int, default=pointwise.horizon, help="Planning horizon used by the Pointwise baseline.")
    parser.add_argument("--safe-exploration-horizon", type=int, default=safe_exploration.horizon, help="Planning horizon recorded for the Safe Exploration baseline; the planner itself is pointwise/myopic.")
    parser.add_argument("--salnx-horizon", type=int, default=salnx.horizon, help="Planning horizon used by the SAL-NX baseline.")
    parser.add_argument("--tebbe-horizon", type=int, default=tebbe.horizon, help="Planning horizon used by the Tebbe-ABM baseline.")
    parser.add_argument("--nominal-mpc-horizon", type=int, default=nominal_mpc.horizon)
    parser.add_argument("--control-target", type=float, default=nominal_mpc.target)
    parser.add_argument("--control-q-y", type=float, default=nominal_mpc.q_y)
    parser.add_argument("--control-q-velocity", type=float, default=nominal_mpc.q_velocity)
    parser.add_argument("--control-r-u", type=float, default=nominal_mpc.r_u)
    parser.add_argument("--control-r-delta-u", type=float, default=nominal_mpc.r_delta_u)
    parser.add_argument("--control-terminal-weight", type=float, default=nominal_mpc.terminal_weight)
    parser.add_argument("--control-beam-width", type=int, default=nominal_mpc.beam_width)
    parser.add_argument("--active-mpc-information-weight", type=float, default=1.0)
    parser.add_argument("--nominal-mpc-safety-margin", type=float, default=nominal_mpc.safety_margin)
    parser.add_argument("--n-init", type=int, default=data.n_init, help="Initial safe dataset size.")
    parser.add_argument("--kernel", choices=["se", "matern52"], default=general.kernel, help="GP kernel family.")
    parser.add_argument("--epsilon-margin", type=float, default=fa_sal.epsilon_margin, help="Interior/dead-end separation margin.")
    parser.add_argument("--episodic-reset-period", type=int, default=general.episodic_reset_period, help="Optional evaluation protocol: reset the rollout to the initial state every K rounds. Use 0 to disable.")
    parser.add_argument("--delta-f", type=float, default=fa_sal.delta_f, help="Confidence level delta_f in the finite-domain beta_t bound.")
    parser.add_argument("--delta-g", type=float, default=fa_sal.delta_g, help="Confidence level delta_g in the finite-domain beta_t bound.")
    parser.add_argument(
        "--fa-beta-multiplier",
        type=float,
        default=1.0,
        help="Multiplicative c_beta applied to both FA-SAL beta_t^f and beta_t^g.",
    )
    parser.add_argument("--state-grid-points", type=int, default=fa_sal.state_grid_points, help="Grid points per output axis used to instantiate |D|.")
    parser.add_argument("--state-y-min", type=float, default=fa_sal.state_y_min, help="Minimum y value used in the finite-domain grid.")
    parser.add_argument("--state-y-max", type=float, default=fa_sal.state_y_max, help="Maximum y value used in the finite-domain grid.")
    parser.add_argument("--fa-lx", type=float, default=fa_sal.lx, help="FA-SAL error-envelope constant Lx.")
    parser.add_argument("--fa-ly", type=float, default=fa_sal.ly, help="FA-SAL error-envelope constant Ly.")
    parser.add_argument("--fa-lf", type=float, default=fa_sal.lf, help="Minimum floor for the numerically estimated FA-SAL dynamics Lipschitz constant Lf.")
    parser.add_argument("--fa-lf-quantile", type=float, default=fa_sal.lf_quantile, help="Quantile of grid-Jacobian norms used when estimating FA-SAL's dynamics Lipschitz constant.")
    parser.add_argument("--fa-lf-estimator", choices=["jacobian", "legacy_axis_quantile"], default="legacy_axis_quantile", help="Dynamics Lipschitz estimator; jacobian is retained for compatibility.")
    parser.add_argument("--fa-lf-scale", type=float, default=fa_sal.lf_scale, help="Multiplicative scale applied to the grid-Jacobian estimate of Lf.")
    parser.add_argument("--fa-lf-scale-sweep", type=float, nargs="*", default=list(fa_sal.lf_scale_sweep), help="Optional Lf multiplier sweep. Example: --fa-lf-scale-sweep 0.5 0.75 1 1.5 2")
    parser.add_argument("--fa-lf-cap", type=float, default=fa_sal.lf_cap, help="Maximum cap applied to the numerically estimated FA-SAL dynamics Lipschitz constant.")
    parser.add_argument("--fa-l-ell", type=float, default=fa_sal.l_ell, help="Fallback FA-SAL LCB Lipschitz constant used only if the numerical estimate is invalid.")
    parser.add_argument("--fa-l-ell-quantile", type=float, default=fa_sal.l_ell_quantile, help="Robust quantile used when estimating FA-SAL's LCB Lipschitz constant.")
    parser.add_argument("--fa-l-ell-scale", type=float, default=fa_sal.l_ell_scale, help="Multiplicative scale applied directly to the numerically estimated FA-SAL LCB Lipschitz constant.")
    parser.add_argument("--fa-l-ell-scale-sweep", type=float, nargs="*", default=list(fa_sal.l_ell_scale_sweep), help="Optional FA-SAL parameter sweep for l_ell_scale. Example: --fa-l-ell-scale-sweep 0.5 0.6 0.7")
    parser.add_argument("--fa-buffer-weight", type=float, default=fa_sal.buffer_weight, help="Compatibility option; not used by the current selection rule.")
    parser.add_argument("--parallel-workers", type=int, default=general.parallel_workers, help="Number of worker processes used for parameter sweeps and seed runs. Use 0 to auto-select, 1 to force serial execution.")
    parser.add_argument("--max-tasks-per-child", type=int, default=1, help="Recycle each process worker after this many trials to release retained memory. Use 0 to keep workers alive.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory for atomic per-trial checkpoints. Defaults to result/checkpoints.")
    parser.add_argument("--resume-checkpoints", action=argparse.BooleanOptionalAction, default=True, help="Reuse valid completed-trial checkpoints from an identical configuration.")
    parser.add_argument("--paired-seeds", action=argparse.BooleanOptionalAction, default=True, help="Use identical environment/initial-data seeds across methods for paired tests.")
    parser.add_argument("--safe-exploration-alpha", type=float, default=safe_exploration.alpha, help="Allowed pointwise failure probability for the Safe Exploration/SAL baseline.")
    parser.add_argument("--salnx-alpha", type=float, default=salnx.alpha, help="Safety risk parameter alpha used by the SAL-NX baseline.")
    parser.add_argument("--salnx-alpha-sweep", type=float, nargs="*", default=list(salnx.alpha_sweep), help="Optional SAL-NX parameter sweep for alpha. Example: --salnx-alpha-sweep 0.1 0.2 0.3")
    parser.add_argument("--salnx-criterion", choices=["logdet", "trace", "maxeig"], default=salnx.criterion, help="Trajectory uncertainty criterion used by SAL-NX.")
    parser.add_argument("--salnx-joint-mc-samples", type=int, default=salnx.joint_mc_samples, help="Monte Carlo sample count used to approximate SAL-NX joint trajectory safety probability.")
    parser.add_argument("--tebbe-alpha", type=float, default=tebbe.alpha, help="Safety risk parameter alpha used by the Tebbe-ABM baseline.")
    parser.add_argument("--tebbe-alpha-sweep", type=float, nargs="*", default=list(tebbe.alpha_sweep), help="Optional Tebbe-ABM alpha sweep.")
    parser.add_argument("--tebbe-criterion", choices=["logdet", "trace", "maxeig"], default=tebbe.criterion, help="Trajectory uncertainty criterion used by the Tebbe-ABM baseline.")
    parser.add_argument("--tebbe-confidence-delta", type=float, default=tebbe.confidence_delta, help="Confidence parameter delta used in the Tebbe-ABM adaptive stopping bounds.")
    parser.add_argument("--tebbe-confidence-delta-sweep", type=float, nargs="*", default=list(tebbe.confidence_delta_sweep), help="Optional Tebbe-ABM confidence-delta sweep.")
    parser.add_argument("--tebbe-sample-start", type=int, default=tebbe.sample_start, help="Initial Monte Carlo sample batch size used by Tebbe-ABM.")
    parser.add_argument("--tebbe-sample-stages", type=int, default=tebbe.sample_stages, help="Number of doubling stages used by Tebbe-ABM.")
    parser.add_argument("--tebbe-endpoint-candidates", type=int, default=tebbe.endpoint_candidates, help="Finite endpoint candidate count used by Tebbe-ABM (default: 11, matching synthetic SAL-NX).")
    parser.add_argument("--tebbe-legacy-mc-prefix", action=argparse.BooleanOptionalAction, default=False, help="Use the legacy seed-0 full Monte Carlo sample array and stage prefixes.")
    parser.add_argument("--tebbe-local-refinement", action=argparse.BooleanOptionalAction, default=False, help="Enable optional Tebbe golden-section refinement.")
    parser.add_argument("--e_p", dest="eval_points", type=int, default=general.eval_points, help="Holdout points used for RMSE and safe-region evaluation.")
    parser.add_argument("--metric-eval-points", type=int, default=0, help="Generate --e_p points but report metrics on the deterministic first N points; 0 uses all points.")
    parser.add_argument("--environment-preset", choices=sorted(SYNTHETIC_ENVIRONMENT_PRESETS), default=env.preset)
    parser.add_argument("--env-kind", choices=["double_integrator", "runaway_zone", "commitment_funnel", "goal_funnel", "funnel_lattice", "diagnostic_deadend", "sparse_deadend"], default=env.kind, help="Synthetic environment family used in the benchmark.")
    parser.add_argument("--env-y-max", type=float, default=env.y_max, help="True safety boundary y_max of the synthetic environment.")
    parser.add_argument("--env-noise-std", type=float, default=env.dynamics_noise_std, help="Process/observation noise used in the environment simulator.")
    parser.add_argument("--safety-noise-std", type=float, default=env.safety_noise_std, help="Observation noise used for safety-function measurements.")
    parser.add_argument("--runaway-center", type=float, default=env.runaway_center, help="Center of the runaway zone used in the alternative synthetic environment.")
    parser.add_argument("--runaway-halfwidth", type=float, default=env.runaway_halfwidth, help="Half-width of the runaway zone used in the alternative synthetic environment.")
    parser.add_argument("--runaway-strength", type=float, default=env.runaway_strength, help="Extra positive drift strength inside the runaway zone.")
    parser.add_argument("--runaway-velocity-threshold", type=float, default=env.runaway_velocity_threshold, help="Minimum positive velocity required to activate the runaway zone.")
    parser.add_argument("--funnel-center", type=float, default=env.funnel_center, help="Center of the commitment funnel trap.")
    parser.add_argument("--funnel-halfwidth", type=float, default=env.funnel_halfwidth, help="Half-width of the commitment funnel trap.")
    parser.add_argument("--funnel-runaway-strength", type=float, default=env.funnel_runaway_strength, help="Additional drift strength inside the commitment funnel.")
    parser.add_argument("--funnel-velocity-threshold", type=float, default=env.funnel_velocity_threshold, help="Minimum velocity that activates the commitment funnel drift.")
    parser.add_argument("--funnel-brake-fade", type=float, default=env.funnel_brake_fade, help="Fraction of braking authority lost inside the commitment funnel.")
    parser.add_argument("--lattice-start", type=float, default=env.lattice_start, help="First dead-end lattice center.")
    parser.add_argument("--lattice-period", type=float, default=env.lattice_period, help="Spacing between repeated dead-end zones.")
    parser.add_argument("--lattice-count", type=int, default=env.lattice_count, help="Number of repeated dead-end zones.")
    parser.add_argument("--lattice-halfwidth", type=float, default=env.lattice_halfwidth, help="Half-width of each repeated dead-end zone.")
    parser.add_argument("--lattice-tailwidth", type=float, default=env.lattice_tailwidth, help="Downstream tail width that keeps the repeated dead-end zone active after crossing its center.")
    parser.add_argument("--lattice-runaway-strength", type=float, default=env.lattice_runaway_strength, help="Runaway drift strength inside repeated dead-end zones.")
    parser.add_argument("--lattice-tail-runaway-multiplier", type=float, default=env.lattice_tail_runaway_multiplier, help="Extra drift multiplier applied after crossing the center of a repeated dead-end zone.")
    parser.add_argument("--lattice-velocity-threshold", type=float, default=env.lattice_velocity_threshold, help="Velocity threshold that activates repeated dead-end zones.")
    parser.add_argument("--lattice-brake-fade", type=float, default=env.lattice_brake_fade, help="Braking fade inside repeated dead-end zones.")
    parser.add_argument("--lattice-tail-brake-fade", type=float, default=env.lattice_tail_brake_fade, help="Stronger braking fade applied in the downstream commitment tail of a repeated dead-end zone.")
    parser.add_argument("--smooth-switches", action=argparse.BooleanOptionalAction, default=bool(getattr(env, "smooth_switches", False)), help="Use differentiable lattice activation, braking fade, and runaway multipliers.")
    parser.add_argument("--switch-sharpness", type=float, default=float(getattr(env, "switch_sharpness", 8.0)), help="Sigmoid/softplus sharpness used by the smooth lattice dynamics.")
    parser.add_argument("--goal-y-min", type=float, default=env.goal_y_min, help="Lower bound of the terminal goal region in the natural episodic goal-funnel task.")
    parser.add_argument("--goal-y-max", type=float, default=env.goal_y_max, help="Upper bound of the terminal goal region in the natural episodic goal-funnel task.")
    parser.add_argument("--fail-y", type=float, default=getattr(env, "fail_y", 5.2), help="Failure-mode observation inserted after an actual safety violation.")
    parser.add_argument("--failure-absorbing", action=argparse.BooleanOptionalAction, default=getattr(env, "failure_absorbing", True), help="Whether the environment should use absorbing failure dynamics after a safety violation.")
    parser.add_argument("--init-y-min", type=float, default=data.y_min, help="Minimum y_t used when sampling the initial safe dataset.")
    parser.add_argument("--init-y-max", type=float, default=data.y_max, help="Maximum y_t used when sampling the initial safe dataset.")
    parser.add_argument("--init-v-min", type=float, default=data.v_min, help="Minimum velocity used when sampling the initial safe dataset.")
    parser.add_argument("--init-v-max", type=float, default=data.v_max, help="Maximum velocity used when sampling the initial safe dataset.")
    parser.add_argument("--start-y-tm1", type=float, default=data.start_y_tm1, help="Initial output y_{t-1} used to start each online rollout.")
    parser.add_argument("--start-y-t", type=float, default=data.start_y_t, help="Initial output y_t used to start each online rollout.")
    parser.add_argument("--snapshot-iters", type=int, nargs="*", default=list(snapshots.iters), help="Iterations to render in safe-region snapshot panels.")
    parser.add_argument("--snapshot-grid-size", type=int, default=snapshots.grid_size, help="Grid resolution per axis for the safe-region snapshot panels.")
    parser.add_argument("--snapshot-u-slice", type=float, default=snapshots.u_slice, help="Fixed u value used when slicing the 3D NARX regressor into a 2D snapshot plot.")
    return parser


def main() -> None:
    start_timestamp = datetime.now().astimezone()
    start_perf = time.perf_counter()
    args = build_parser().parse_args()
    defaults = DEFAULT_EXPERIMENT_CONFIG
    if args.environment_preset != defaults.environment.preset:
        preset_values = get_synthetic_environment_preset(args.environment_preset)
        destination_map = {
            "kind": "env_kind",
            "y_max": "env_y_max",
            "dynamics_noise_std": "env_noise_std",
            "safety_noise_std": "safety_noise_std",
        }
        for key, value in preset_values.items():
            setattr(args, destination_map.get(key, key), value)
    general = defaults.general
    if args.trials < 5 and not args.allow_fewer_trials:
        print(f"Requested {args.trials} seeds, but this experiment uses at least 5. Overriding trials to 5.")
        args.trials = 5
    project_root = Path(__file__).resolve().parent
    result_dir = args.result_dir if args.result_dir is not None else project_root / general.result_dirname
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else result_dir / "checkpoints"
    environment_preset = str(args.environment_preset)
    print(f"Start time: {start_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Using {args.trials} seeds per method.")
    print(f"BLAS thread limits per worker: {BLAS_THREAD_LIMITS}")

    confidence_schedule = ConfidenceSchedule(
        delta_f=args.delta_f,
        delta_g=args.delta_g,
        y_min=args.state_y_min,
        y_max=args.state_y_max,
        y_grid_points=args.state_grid_points,
        beta_multiplier=args.fa_beta_multiplier,
    )
    methods_config = tuple(args.methods) if args.methods is not None else defaults.general.methods
    if isinstance(methods_config, str):
        selected_methods = tuple(part.strip() for part in methods_config.split(",") if part.strip())
    else:
        selected_methods = tuple(methods_config)
    method_catalog = {
        "fa_sal": {"planner_type": "future", "seed_offset": 0, "label": "FA-SAL", "snapshot_prefix": "future"},
        "myopic_fa_sal": {"planner_type": "myopic_fa_sal", "seed_offset": 500, "label": "Myopic FA-SAL (m=1)", "snapshot_prefix": "myopic_fa_sal"},
        "fa_random": {"planner_type": "fa_random", "seed_offset": 1000, "label": "FA-SAL-Random", "snapshot_prefix": "fa_random"},
        "safe_exploration": {"planner_type": "safe_exploration", "seed_offset": 1500, "label": "Safe Exploration", "snapshot_prefix": "safe_exploration"},
        "sal": {"planner_type": "safe_exploration", "seed_offset": 1500, "label": "Safe Exploration", "snapshot_prefix": "safe_exploration"},
        "salnx": {"planner_type": "salnx", "seed_offset": 2000, "label": "SAL-NX", "snapshot_prefix": "salnx"},
        "tebbe_abm": {"planner_type": "tebbe_abm", "seed_offset": 2500, "label": "Tebbe-ABM", "snapshot_prefix": "tebbe_abm"},
        "pointwise": {"planner_type": "pointwise", "seed_offset": 3000, "label": "Pointwise", "snapshot_prefix": "pointwise"},
        "nominal_mpc": {"planner_type": "nominal_mpc", "seed_offset": 4500, "label": "Nominal MPC", "snapshot_prefix": "nominal_mpc"},
        "al_mpc": {"planner_type": "al_mpc", "seed_offset": 4750, "label": "AL-MPC", "snapshot_prefix": "al_mpc"},
    }
    invalid_methods = [name for name in selected_methods if name not in method_catalog]
    if invalid_methods:
        raise ValueError(
            f"Unknown methods in config.general.methods: {invalid_methods}. "
            f"Choose from {sorted(method_catalog.keys())}."
        )
    method_variants = _build_method_variants(
        selected_methods=selected_methods,
        method_catalog=method_catalog,
        fa_continuation_policy_sweep=args.fa_continuation_policy_sweep,
        fa_lf_scale_sweep=args.fa_lf_scale_sweep,
        fa_l_ell_scale_sweep=args.fa_l_ell_scale_sweep,
        salnx_alpha_sweep=args.salnx_alpha_sweep,
        tebbe_alpha_sweep=args.tebbe_alpha_sweep,
        tebbe_confidence_delta_sweep=args.tebbe_confidence_delta_sweep,
    )

    common_kwargs = dict(
        kernel_kind=args.kernel,
        T=args.rounds,
        fa_horizon=args.fa_horizon,
        fa_beam_width=args.fa_beam_width,
        fa_continuation_policy=args.fa_continuation_policy,
        fa_random_horizon=args.fa_random_horizon,
        pointwise_horizon=args.pointwise_horizon,
        safe_exploration_horizon=args.safe_exploration_horizon,
        salnx_horizon=args.salnx_horizon,
        tebbe_horizon=args.tebbe_horizon,
        nominal_mpc_horizon=args.nominal_mpc_horizon,
        control_target=args.control_target,
        control_q_y=args.control_q_y,
        control_q_velocity=args.control_q_velocity,
        control_r_u=args.control_r_u,
        control_r_delta_u=args.control_r_delta_u,
        control_terminal_weight=args.control_terminal_weight,
        control_beam_width=args.control_beam_width,
        active_mpc_information_weight=args.active_mpc_information_weight,
        nominal_mpc_safety_margin=args.nominal_mpc_safety_margin,
        n_init=args.n_init,
        epsilon_margin=args.epsilon_margin,
        recovery_eval_interval=args.recovery_eval_interval,
        evaluation_horizon=args.evaluation_horizon,
        episodic_reset_period=args.episodic_reset_period,
        confidence_schedule=confidence_schedule,
        fa_lx=args.fa_lx,
        fa_ly=args.fa_ly,
        fa_lf=args.fa_lf,
        fa_lf_quantile=args.fa_lf_quantile,
        fa_lf_estimator=args.fa_lf_estimator,
        fa_lf_scale=args.fa_lf_scale,
        fa_lf_cap=args.fa_lf_cap,
        fa_l_ell=args.fa_l_ell,
        fa_l_ell_quantile=args.fa_l_ell_quantile,
        fa_l_ell_scale=args.fa_l_ell_scale,
        fa_buffer_weight=args.fa_buffer_weight,
        alpha_safety=args.salnx_alpha,
        safe_exploration_alpha=args.safe_exploration_alpha,
        salnx_criterion=args.salnx_criterion,
        salnx_joint_mc_samples=args.salnx_joint_mc_samples,
        tebbe_alpha=args.tebbe_alpha,
        tebbe_criterion=args.tebbe_criterion,
        tebbe_confidence_delta=args.tebbe_confidence_delta,
        tebbe_sample_start=args.tebbe_sample_start,
        tebbe_sample_stages=args.tebbe_sample_stages,
        tebbe_endpoint_candidates=args.tebbe_endpoint_candidates,
        tebbe_legacy_mc_prefix=args.tebbe_legacy_mc_prefix,
        tebbe_local_refinement=args.tebbe_local_refinement,
        n_eval=args.eval_points,
        metric_eval_points=args.metric_eval_points,
        snapshot_grid_size=args.snapshot_grid_size,
        snapshot_u_slice=args.snapshot_u_slice,
        env_kind=args.env_kind,
        env_y_max=args.env_y_max,
        env_noise_std=args.env_noise_std,
        safety_noise_std=args.safety_noise_std,
        runaway_center=args.runaway_center,
        runaway_halfwidth=args.runaway_halfwidth,
        runaway_strength=args.runaway_strength,
        runaway_velocity_threshold=args.runaway_velocity_threshold,
        funnel_center=args.funnel_center,
        funnel_halfwidth=args.funnel_halfwidth,
        funnel_runaway_strength=args.funnel_runaway_strength,
        funnel_velocity_threshold=args.funnel_velocity_threshold,
        funnel_brake_fade=args.funnel_brake_fade,
        lattice_start=args.lattice_start,
        lattice_period=args.lattice_period,
        lattice_count=args.lattice_count,
        lattice_halfwidth=args.lattice_halfwidth,
        lattice_tailwidth=args.lattice_tailwidth,
        lattice_runaway_strength=args.lattice_runaway_strength,
        lattice_tail_runaway_multiplier=args.lattice_tail_runaway_multiplier,
        lattice_velocity_threshold=args.lattice_velocity_threshold,
        lattice_brake_fade=args.lattice_brake_fade,
        lattice_tail_brake_fade=args.lattice_tail_brake_fade,
        smooth_switches=args.smooth_switches,
        switch_sharpness=args.switch_sharpness,
        goal_y_min=args.goal_y_min,
        goal_y_max=args.goal_y_max,
        fail_y=args.fail_y,
        failure_absorbing=args.failure_absorbing,
        init_y_range=(args.init_y_min, args.init_y_max),
        init_velocity_range=(args.init_v_min, args.init_v_max),
        start_y_tm1=args.start_y_tm1,
        start_y_t=args.start_y_t,
    )

    trials_by_method = {}
    summaries = {}
    total_trial_count = len(method_variants) * args.trials
    use_parallel = args.parallel_workers != 1 and total_trial_count > 1
    worker_count = 1
    if use_parallel:
        auto_workers = os.cpu_count() or 1
        worker_count = max(1, int(args.parallel_workers)) if args.parallel_workers > 0 else min(auto_workers, total_trial_count)
        use_parallel = worker_count > 1
    if use_parallel:
        print(f"Parallel sweep mode: using {worker_count} worker processes.")
        if args.max_tasks_per_child > 0:
            print(f"Worker recycling after {args.max_tasks_per_child} trial(s) per process.")
    overall_progress = tqdm(
        total=total_trial_count,
        desc="Experiments",
        leave=True,
        dynamic_ncols=True,
    )
    if use_parallel:
        trials_by_method = {
            str(variant["label"]): [None] * args.trials
            for variant in method_variants
        }
        tasks = []
        for seed in range(args.trials):
            for variant in method_variants:
                tasks.append(
                    {
                        "variant": variant,
                        "seed_index": seed,
                        "seed": int(args.seed_offset) + (seed if args.paired_seeds else int(variant["seed_offset"]) + seed),
                        "snapshot_rounds": tuple(args.snapshot_iters) if seed == 0 else (),
                        "common_kwargs": common_kwargs,
                        "checkpoint_dir": str(checkpoint_dir),
                        "trial_progress_interval": args.trial_progress_interval,
                    }
                )
        pending_tasks = []
        resumed_count = 0
        for task in tasks:
            trial = _load_trial_checkpoint(checkpoint_dir, task) if args.resume_checkpoints else None
            if trial is None:
                pending_tasks.append(task)
            else:
                label = str(task["variant"]["label"])
                trials_by_method[label][int(task["seed_index"])] = trial
                resumed_count += 1
                overall_progress.update(1)
        if resumed_count:
            print(f"Resumed {resumed_count}/{total_trial_count} trials from {checkpoint_dir}.")
        for task in pending_tasks:
            task["worker_count"] = worker_count
        executor_cls = ProcessPoolExecutor
        try:
            executor_kwargs = {"max_workers": worker_count}
            if args.max_tasks_per_child > 0:
                executor_kwargs["max_tasks_per_child"] = args.max_tasks_per_child
            executor = executor_cls(**executor_kwargs)
        except (OSError, PermissionError):
            executor_cls = ThreadPoolExecutor
            executor = executor_cls(max_workers=worker_count)
            print("Process-based parallelism is unavailable here; falling back to thread-based parallel execution.")
        with executor:
            future_to_task = {
                executor.submit(_run_trial_task, task): task
                for task in pending_tasks
            }
            for future in as_completed(future_to_task):
                result = future.result()
                task = future_to_task[future]
                variant_label = str(result["variant_label"])
                seed_index = int(result["seed_index"])
                trials_by_method[variant_label][seed_index] = result["trial"]
                path = _save_trial_checkpoint(checkpoint_dir, task, result["trial"])
                print(f"Checkpointed {variant_label} seed_index={seed_index} to {path}")
                overall_progress.update(1)
        for variant in method_variants:
            label = str(variant["label"])
            method_trials = trials_by_method[label]
            summaries[label] = summarize_trials(method_trials)
    else:
        for variant in method_variants:
            label = str(variant["label"])
            variant_kwargs = dict(common_kwargs)
            variant_kwargs.update(variant.get("overrides", {}))
            method_trials = []
            seed_progress = tqdm(
                range(args.trials),
                desc=f"{label} seeds",
                leave=False,
                dynamic_ncols=True,
            )
            for seed in seed_progress:
                seed_progress.set_postfix_str(f"seed={seed}")
                task = {"variant": variant, "seed_index": seed,
                        "seed": int(args.seed_offset) + (seed if args.paired_seeds else int(variant["seed_offset"]) + seed),
                        "snapshot_rounds": tuple(args.snapshot_iters) if seed == 0 else (),
                        "common_kwargs": common_kwargs}
                trial = _load_trial_checkpoint(checkpoint_dir, task) if args.resume_checkpoints else None
                if trial is None:
                    trial = run_trial(planner_type=str(variant["planner_type"]), seed=int(task["seed"]),
                                      snapshot_rounds=task["snapshot_rounds"], show_progress=True,
                                      progress_desc=f"{label} seed {seed} rounds", **variant_kwargs)
                    _save_trial_checkpoint(checkpoint_dir, task, trial)
                method_trials.append(trial)
                overall_progress.update(1)
            seed_progress.close()
            trials_by_method[label] = method_trials
            summaries[label] = summarize_trials(method_trials)
    overall_progress.close()

    print("=== Aggregate Results ===")
    for name, summary in summaries.items():
        print(
            f"{name}"
            f" | Executed transitions: {summary['executed_transition_count_sum_mean'][0]:.2f} +/- {summary['executed_transition_count_sum_std'][0]:.2f}"
            f" | No-feasible rounds: {summary['no_feasible_action_sum_mean'][0]:.2f} +/- {summary['no_feasible_action_sum_std'][0]:.2f}"
            f" | False-cert: {summary['active_final_false_certification_rate_mean'][0]:.2f}"
            f" | InteriorRecall(eps): {summary['active_final_recall_mean'][0]:.2f}"
            f" | RMSE: {summary['active_final_rmse_mean'][0]:.3f}"
            f" | SVR: {summary['active_final_safety_violation_rate_mean'][0]:.2f}"
            f" | Safety IoU: {summary['active_final_safety_set_recovery_iou_mean'][0]:.2f}"
            f" | D-SVR: {summary['active_final_dead_end_safety_violation_rate_mean'][0]:.2f}"
            f" | Dead-end IoU: {summary['active_final_dead_end_free_recovery_iou_mean'][0]:.2f}"
        )

    result_dir.mkdir(parents=True, exist_ok=True)

    base_config = {
        "rounds": args.rounds,
        "trials": args.trials,
        "evaluation_horizon": args.evaluation_horizon,
        "methods": list(selected_methods),
        "fa_horizon": args.fa_horizon,
        "fa_beam_width": args.fa_beam_width,
        "fa_continuation_policy": args.fa_continuation_policy,
        "fa_continuation_policy_sweep": list(args.fa_continuation_policy_sweep),
        "paired_seeds": bool(args.paired_seeds),
        "fa_random_horizon": args.fa_random_horizon,
        "pointwise_horizon": args.pointwise_horizon,
        "safe_exploration_horizon": args.safe_exploration_horizon,
        "salnx_horizon": args.salnx_horizon,
        "tebbe_horizon": args.tebbe_horizon,
        "nominal_mpc_horizon": args.nominal_mpc_horizon,
        "control_target": args.control_target,
        "control_q_y": args.control_q_y,
        "control_q_velocity": args.control_q_velocity,
        "control_r_u": args.control_r_u,
        "control_r_delta_u": args.control_r_delta_u,
        "control_terminal_weight": args.control_terminal_weight,
        "control_beam_width": args.control_beam_width,
        "active_mpc_information_weight": args.active_mpc_information_weight,
        "nominal_mpc_safety_margin": args.nominal_mpc_safety_margin,
        "n_init": args.n_init,
        "kernel": args.kernel,
        "epsilon_margin": args.epsilon_margin,
        "episodic_reset_period": args.episodic_reset_period,
        "delta_f": args.delta_f,
        "delta_g": args.delta_g,
        "state_grid_points": args.state_grid_points,
        "state_y_min": args.state_y_min,
        "state_y_max": args.state_y_max,
        "fa_lx": args.fa_lx,
        "fa_ly": args.fa_ly,
        "fa_lf": args.fa_lf,
        "fa_lf_quantile": args.fa_lf_quantile,
        "fa_lf_estimator": args.fa_lf_estimator,
        "fa_lf_scale": args.fa_lf_scale,
        "fa_lf_scale_sweep": list(args.fa_lf_scale_sweep),
        "fa_lf_cap": args.fa_lf_cap,
        "fa_l_ell": args.fa_l_ell,
        "fa_l_ell_quantile": args.fa_l_ell_quantile,
        "fa_l_ell_scale": args.fa_l_ell_scale,
        "fa_l_ell_scale_sweep": list(args.fa_l_ell_scale_sweep),
        "fa_buffer_weight": args.fa_buffer_weight,
        "parallel_workers": worker_count if use_parallel else 1,
        "blas_thread_limits": BLAS_THREAD_LIMITS,
        "safe_exploration_alpha": args.safe_exploration_alpha,
        "salnx_alpha": args.salnx_alpha,
        "salnx_alpha_sweep": list(args.salnx_alpha_sweep),
        "salnx_criterion": args.salnx_criterion,
        "salnx_joint_mc_samples": args.salnx_joint_mc_samples,
        "tebbe_alpha": args.tebbe_alpha,
        "tebbe_alpha_sweep": list(args.tebbe_alpha_sweep),
        "tebbe_criterion": args.tebbe_criterion,
        "tebbe_confidence_delta": args.tebbe_confidence_delta,
        "tebbe_confidence_delta_sweep": list(args.tebbe_confidence_delta_sweep),
        "tebbe_sample_start": args.tebbe_sample_start,
        "tebbe_sample_stages": args.tebbe_sample_stages,
        "tebbe_endpoint_candidates": args.tebbe_endpoint_candidates,
        "tebbe_legacy_mc_prefix": args.tebbe_legacy_mc_prefix,
        "tebbe_local_refinement": args.tebbe_local_refinement,
        "env_kind": args.env_kind,
        "env_y_max": args.env_y_max,
        "env_noise_std": args.env_noise_std,
        "safety_noise_std": args.safety_noise_std,
        "runaway_center": args.runaway_center,
        "runaway_halfwidth": args.runaway_halfwidth,
        "runaway_strength": args.runaway_strength,
        "runaway_velocity_threshold": args.runaway_velocity_threshold,
        "funnel_center": args.funnel_center,
        "funnel_halfwidth": args.funnel_halfwidth,
        "funnel_runaway_strength": args.funnel_runaway_strength,
        "funnel_velocity_threshold": args.funnel_velocity_threshold,
        "funnel_brake_fade": args.funnel_brake_fade,
        "lattice_start": args.lattice_start,
        "lattice_period": args.lattice_period,
        "lattice_count": args.lattice_count,
        "lattice_halfwidth": args.lattice_halfwidth,
        "lattice_tailwidth": args.lattice_tailwidth,
        "lattice_runaway_strength": args.lattice_runaway_strength,
        "lattice_tail_runaway_multiplier": args.lattice_tail_runaway_multiplier,
        "lattice_velocity_threshold": args.lattice_velocity_threshold,
        "lattice_brake_fade": args.lattice_brake_fade,
        "lattice_tail_brake_fade": args.lattice_tail_brake_fade,
        "smooth_switches": bool(args.smooth_switches),
        "switch_sharpness": float(args.switch_sharpness),
        "goal_y_min": args.goal_y_min,
        "goal_y_max": args.goal_y_max,
        "fail_y": args.fail_y,
        "failure_absorbing": args.failure_absorbing,
        "init_y_min": args.init_y_min,
        "init_y_max": args.init_y_max,
        "init_v_min": args.init_v_min,
        "init_v_max": args.init_v_max,
        "start_y_tm1": args.start_y_tm1,
        "start_y_t": args.start_y_t,
        "snapshot_iters": args.snapshot_iters,
        "snapshot_grid_size": args.snapshot_grid_size,
        "snapshot_u_slice": args.snapshot_u_slice,
        "save_dir": str(result_dir),
        "environment_preset": environment_preset,
        "finite_domain": {
            "construction": "Cartesian grid over (y_t, y_{t-1}, u_t)",
            "y_min": args.state_y_min,
            "y_max": args.state_y_max,
            "points_per_y_axis": args.state_grid_points,
            "action_values": _json_ready(
                np.round(np.arange(-1.0, 1.0 + 1e-9, 0.2), 1)
            ),
            "domain_cardinality": int(
                args.state_grid_points
                * args.state_grid_points
                * len(np.arange(-1.0, 1.0 + 1e-9, 0.2))
            ),
        },
        "candidate_domain": {
            "construction": "All discrete actions returned by env.candidate_actions() at every round",
            "action_values": _json_ready(
                np.round(np.arange(-1.0, 1.0 + 1e-9, 0.2), 1)
            ),
            "candidate_count": int(len(np.arange(-1.0, 1.0 + 1e-9, 0.2))),
        },
        "gp_specification": {
            "kernel_family": args.kernel,
            "kernel_variance": 1.0,
            "kernel_length_scale": 1.0,
            "hyperparameter_rule": "fixed throughout each trial; no marginal-likelihood optimization",
            "dynamics_noise_std": args.env_noise_std,
            "safety_noise_std": args.safety_noise_std,
            "update_rule": "full exact-GP refit with dense Cholesky factorization at every round",
            "update_complexity": "O(n_t^3) time and O(n_t^2) memory per GP fit",
        },
        "confidence_schedule": {
            "formula": "beta_t^q = c_beta 2 log(|D| pi_t / delta_q), pi_t = pi^2 t^2 / 6",
            "delta_f": args.delta_f,
            "delta_g": args.delta_g,
            "beta_multiplier": args.fa_beta_multiplier,
        },
        "pipeline_complexity_note": (
            "Measured stage timings are stored per round. Current exact GP fitting "
            "has a dense O(n_t^3) Cholesky bottleneck; planning additionally scales "
            "with candidate count and horizon."
        ),
    }
    saved_paths = []
    for variant in method_variants:
        label = str(variant["label"])
        payload = {
            "config": {
                **base_config,
                "methods": [label],
                "base_method": str(variant["base_method"]),
                "result_id": str(variant["result_id"]),
                "overrides": _json_ready(variant.get("overrides", {})),
            },
            "summaries": {
                label: {k: v.tolist() for k, v in summaries[label].items()}
            },
            "trial_records": {
                label: [
                    _compact_trial_record(trial)
                    for trial in trials_by_method[label]
                ]
            },
            "snapshots": {
                label: _json_ready(trials_by_method[label][0].get("snapshots", []))
            },
        }
        setting_suffix = _synthetic_result_setting_suffix(base_config, variant)
        result_path = result_dir / (
            f"{_safe_token(environment_preset)}_{variant['result_id']}_{setting_suffix}_results.json"
        )
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        saved_paths.append(str(result_path))
    print("Saved results JSON files:")
    for path in saved_paths:
        print(path)

    end_timestamp = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - start_perf
    elapsed_minutes, remaining_seconds = divmod(elapsed_seconds, 60.0)
    elapsed_hours, remaining_minutes = divmod(elapsed_minutes, 60.0)
    print(f"End time: {end_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(
        "Elapsed time:"
        f" {int(elapsed_hours):02d}:{int(remaining_minutes):02d}:{remaining_seconds:05.2f}"
        f" ({elapsed_seconds:.2f} seconds)"
    )


if __name__ == "__main__":
    config = DEFAULT_EXPERIMENT_CONFIG

    print("Environment preset:", config.environment.preset)
    print("kind:", config.environment.kind)
    print("y_max:", config.environment.y_max)
    print("lattice_count:", config.environment.lattice_count)
    print("lattice_runaway_strength:", config.environment.lattice_runaway_strength)
    print("fail_y:", getattr(config.environment, "fail_y", 5.2))
    print("failure_absorbing:", getattr(config.environment, "failure_absorbing", True))
    main()
