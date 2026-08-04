

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_synthetic_gp_hyperparameter_update as base
import run_synthetic_gp_kernel_lf_separation as separation
import run_synthetic_gp_noise_delay as noise_delay
import run_synthetic_gp_option2 as option2
import run_synthetic_gp_option2_full_grid_median as full_grid


CONDITIONS = (
    "nominal_fixed",
    "joint_low_fixed",
    "joint_low_adaptive_lf",
    "joint_high_fixed",
    "joint_high_adaptive_lf",
)

base.CONDITIONS["joint_low_adaptive_lf"] = {
    "length_scale": 0.5,
    "variance": 0.5,
    "retrain": True,
    "retrain_start": 50,
}
base.CONDITIONS["joint_high_adaptive_lf"] = {
    "length_scale": 2.0,
    "variance": 2.0,
    "retrain": True,
    "retrain_start": 50,
}

_ORIGINAL_COMMON_KWARGS = base._common_kwargs


def _base_kwargs() -> dict[str, Any]:

    kwargs = _ORIGINAL_COMMON_KWARGS()
    kwargs["fa_lf_estimator"] = "legacy_axis_quantile"
    kwargs["fa_lf_quantile"] = 0.5
    kwargs["fa_lf_scale"] = 1.0
    kwargs["fa_l_ell_quantile"] = 0.9
    kwargs["fa_l_ell_scale"] = 0.14
    return kwargs


def _metrics(trial: dict[str, Any]) -> dict[str, float]:

    unsafe = np.asarray(trial["unsafe_transitions"], dtype=float)
    chosen = np.asarray(trial["chosen_actions"], dtype=float)
    active = np.isfinite(chosen)

    def final(active_key: str, fallback_key: str) -> float:
        values = trial.get(active_key, trial[fallback_key])
        return float(np.asarray(values, dtype=float)[-1])

    return {
        "rmse": float(np.asarray(trial["dynamics_rmse"], dtype=float)[-1]),
        "svr": float(np.sum(unsafe[active]) / max(1, np.sum(active))),
        "d_svr": float(
            np.asarray(trial["dead_end_safety_violation_rate"], dtype=float)[-1]
        ),
        "safety_iou": final(
            "safety_set_recovery_iou_active", "safety_set_recovery_iou"
        ),
        "de_iou": final(
            "dead_end_free_recovery_iou_active",
            "dead_end_free_recovery_iou",
        ),
        "false_certification": float(
            np.asarray(trial["false_certification_rate"], dtype=float)[-1]
        ),
        "no_feasible_rate": float(
            np.mean(np.asarray(trial["no_feasible_action"], dtype=float)[active])
        ),
    }


def _run(task: tuple[str, int]) -> dict[str, Any]:
    condition, seed = task

    def condition_kwargs() -> dict[str, Any]:
        kwargs = _base_kwargs()
        if condition == "nominal_fixed":
            kwargs["n_eval"] = 100
            kwargs["recovery_eval_interval"] = 10
        return kwargs

    base._common_kwargs = condition_kwargs
    base._scalar_metrics = _metrics
    adaptive = condition.endswith("_adaptive_lf")

    if adaptive:
        option2._anchors = full_grid._full_grid
        option2._sample_gradient_bound = separation._samplewise_maximum_median
        option2.LF_AGGREGATION = (
            "median of sample-wise maximum posterior-function gradient norms"
        )
        option2._install_hooks()
        base._optimize_kernel = lambda gp, X, y: noise_delay._optimize(
            gp, X, y, joint_noise=True
        )

    result = base._run_task((condition, seed))
    if adaptive:
        for name, history in result["hyperparameter_history"].items():
            if not history:
                continue
            suffix = "f" if name == "dynamics" else "g"
            result["final_hyperparameters"][f"variance_{suffix}"] = float(
                history[-1]["variance"]
            )
            result["final_hyperparameters"][f"noise_variance_{suffix}"] = float(
                history[-1]["noise_variance"]
            )
        lf_history = result["hyperparameter_history"]["dynamics"]
        result["lf_history"] = [
            {
                "after_round": int(item["after_round"]),
                "estimated_lf": float(item["estimated_lf"]),
            }
            for item in lf_history
            if "estimated_lf" in item
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "result/"
        "synthetic_gp_initial_scale_joint_noise50_lf_update_m4/results.json",
    )
    args = parser.parse_args()
    jobs = [
        (condition, seed)
        for condition in CONDITIONS
        for seed in range(int(args.trials))
    ]
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {pool.submit(_run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            item = future.result()
            records.append(item)
            print(
                f"[{index}/{len(jobs)}] {item['condition']} "
                f"seed={item['seed']} runtime={item['runtime_seconds']:.1f}s",
                flush=True,
            )

    records.sort(
        key=lambda item: (
            CONDITIONS.index(item["condition"]), int(item["seed"])
        )
    )
    groups = {
        condition: [
            item for item in records if item["condition"] == condition
        ]
        for condition in CONDITIONS
    }
    summary = {
        condition: base._summarize(items)
        for condition, items in groups.items()
    }
    for condition in (
        "joint_low_adaptive_lf",
        "joint_high_adaptive_lf",
    ):
        items = groups[condition]
        for key in ("noise_variance_f", "noise_variance_g"):
            values = np.asarray(
                [item["final_hyperparameters"][key] for item in items],
                dtype=float,
            )
            summary[condition][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
        final_lf = np.asarray(
            [item["lf_history"][-1]["estimated_lf"] for item in items],
            dtype=float,
        )
        summary[condition]["final_estimated_lf"] = {
            "mean": float(final_lf.mean()),
            "std": float(final_lf.std(ddof=0)),
        }

    payload = {
        "experiment": {
            "benchmark": "penalty_stress",
            "method": "FA-SAL uncertainty-max",
            "horizon": 4,
            "evaluation_horizon": 4,
            "rounds": 150,
            "paired_seeds": list(range(int(args.trials))),
            "conditions": list(CONDITIONS),
            "protocol": "legacy_axis_quantile",
            "adaptive_rule": (
                "joint physical-unit kernel/signal/noise multi-start MLE "
                "from round 50, every 10 rounds"
            ),
            "lf_rule": (
                "after each dynamics-GP update: median across posterior "
                "functions of each function's full-domain maximum gradient"
            ),
            "lf_posterior_function_samples": int(option2.GRADIENT_SAMPLES),
            "lf_domain": "fixed 31x31 state grid x all 11 actions",
            "l_ell_rule": "local_gradient",
        },
        "trials": groups,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
