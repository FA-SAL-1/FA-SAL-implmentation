#!/usr/bin/env python3

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


METRICS = (("SVR", "svr"), ("D-SVR", "d_svr"), ("RMSE", "rmse"))
BASELINES = (("SAL-NX", "salnx"), ("Tebbe-ABM", "tebbe"))


def one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        found = "\n".join(str(path) for path in matches) or "(none)"
        raise RuntimeError(f"Expected one result for {pattern}, found {len(matches)}:\n{found}")
    return matches[0]


def records(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("trial_records", {})
    if len(groups) != 1:
        raise RuntimeError(f"Expected one trial-record group in {path}, found {len(groups)}")
    values = next(iter(groups.values()))
    if len(values) != 10:
        raise RuntimeError(f"Expected 10 trials in {path}, found {len(values)}")
    return {int(value["seed"]): value for value in values}


def sign_flip(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(x - y, dtype=float)
    observed = float(differences.mean())
    statistics = np.asarray(
        [float(np.mean(differences * np.asarray(signs))) for signs in itertools.product((-1.0, 1.0), repeat=len(differences))]
    )
    p_value = float(np.mean(np.abs(statistics) >= abs(observed) - 1e-15))
    return observed, p_value


def holm(values: list[float]) -> list[float]:
    adjusted = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, (len(values) - rank) * value)
        adjusted[index] = min(1.0, running)
    return adjusted


def paths(root: Path, horizon: int) -> dict[str, Path]:
    suffix = f"m{horizon}_evalm{horizon}_lfestlegacy_axis_quantile_*_rounds150_trials10_results.json"
    return {
        "fa": one_match(root, f"penalty_stress_fa_sal_policy_uncertainty_max_lf_scale_1_l_ell_scale_0p001_{suffix}"),
        "salnx": one_match(root, f"penalty_stress_salnx_alpha_0p01_{suffix}"),
        "tebbe": one_match(root, f"penalty_stress_tebbe_abm_alpha_0p01_confidence_delta_0p01_m{horizon}_evalm{horizon}_ucand11_localref0_lfestlegacy_axis_quantile_*_rounds150_trials10_results.json"),
    }


def test_rows(root: Path, horizon: int) -> list[dict]:
    result_paths = paths(root, horizon)
    groups = {name: records(path) for name, path in result_paths.items()}
    rows = []
    for baseline_label, baseline_key in BASELINES:
        seeds = sorted(set(groups["fa"]) & set(groups[baseline_key]))
        if len(seeds) != 10:
            raise RuntimeError(f"Expected 10 paired seeds for m={horizon} and {baseline_label}, found {len(seeds)}")
        for metric_label, metric_key in METRICS:
            fa_values = np.asarray([float(groups["fa"][seed][metric_key]) for seed in seeds])
            baseline_values = np.asarray([float(groups[baseline_key][seed][metric_key]) for seed in seeds])
            difference, raw_p = sign_flip(fa_values, baseline_values)
            rows.append(
                {
                    "m": horizon,
                    "baseline": baseline_label,
                    "metric": metric_label,
                    "difference": difference,
                    "raw_p": raw_p,
                }
            )
    safety_indices = [index for index, row in enumerate(rows) if row["metric"] != "RMSE"]
    safety_adjusted = holm([rows[index]["raw_p"] for index in safety_indices])
    for index, value in zip(safety_indices, safety_adjusted):
        rows[index]["corrected_p"] = value
    rmse_indices = [index for index, row in enumerate(rows) if row["metric"] == "RMSE"]
    rmse_adjusted = holm([round(rows[index]["raw_p"], 5) for index in rmse_indices])
    for index, value in zip(rmse_indices, rmse_adjusted):
        rows[index]["corrected_p"] = value
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m4-result-dir", type=Path, required=True)
    parser.add_argument("--m8-result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = test_rows(args.m4_result_dir, 4) + test_rows(args.m8_result_dir, 8)
    lines = [
        "| $m$ | Baseline | Metric | FA-SAL $-$ Baseline ↓ | Raw $p$ | Corrected $p$¹ | p < 0.01 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    previous_m = None
    previous_baseline = None
    for row in rows:
        display_m = str(row["m"]) if row["m"] != previous_m else ""
        display_baseline = row["baseline"] if row["m"] != previous_m or row["baseline"] != previous_baseline else ""
        significant = "O" if row["corrected_p"] < 0.01 else "X"
        lines.append(
            f"| {display_m} | {display_baseline} | {row['metric']} | ${row['difference']:.3f}$ | "
            f"{row['raw_p']:.5f} | {row['corrected_p']:.5f} | {significant} |"
        )
        previous_m = row["m"]
        previous_baseline = row["baseline"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
