

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_synthetic_gp_initial_scale_joint_noise50_mean_gradient_median_m4 as setup
import run_synthetic_gp_initial_scale_joint_noise50_lf_update_m4 as driver
import run_synthetic_gp_hyperparameter_update as base


CONDITION = "nominal_adaptive_lf"
base.CONDITIONS[CONDITION] = {
    "length_scale": 1.0,
    "variance": 1.0,
    "retrain": True,
    "retrain_start": 50,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "result"
        / "synthetic_gp_nominal_adaptive_mean_gradient_median_m4"
        / "results.json",
    )
    args = parser.parse_args()

    jobs = [(CONDITION, seed) for seed in range(args.trials)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(driver._run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            item = future.result()
            records.append(item)
            print(
                f"[{index}/{len(jobs)}] {CONDITION} seed={item['seed']} "
                f"runtime={item['runtime_seconds']:.1f}s",
                flush=True,
            )

    records.sort(key=lambda item: int(item["seed"]))
    summary = base._summarize(records)
    for key in ("noise_variance_f", "noise_variance_g"):
        values = np.asarray(
            [item["final_hyperparameters"][key] for item in records], dtype=float
        )
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    final_lf = np.asarray(
        [item["lf_history"][-1]["estimated_lf"] for item in records], dtype=float
    )
    summary["final_estimated_lf"] = {
        "mean": float(final_lf.mean()),
        "std": float(final_lf.std(ddof=0)),
    }

    payload = {
        "experiment": {
            "benchmark": "penalty_stress",
            "method": "FA-SAL uncertainty-max",
            "condition": CONDITION,
            "initial_scale": 1.0,
            "horizon": 4,
            "evaluation_horizon": 4,
            "rounds": 150,
            "paired_seeds": list(range(args.trials)),
            "adaptive_rule": (
                "joint kernel length-scale, signal-variance, and noise-variance "
                "MLE from round 50 every 10 rounds"
            ),
            "lf_rule": (
                "median absolute coordinate derivative of the updated dynamics-GP "
                "posterior mean on the fixed full-domain grid"
            ),
            "l_ell_rule": "local_gradient_per_round",
        },
        "trials": records,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
