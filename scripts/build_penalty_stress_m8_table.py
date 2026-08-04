#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    ("RMSE ↓", "rmse", "min", 3),
    ("SVR ↓", "svr", "min", 3),
    ("D-SVR ↓", "d_svr", "min", 3),
    ("Safety IoU ↑", "safety_iou", "max", 3),
    ("DE IoU ↑", "dead_end_iou", "max", 3),
    ("False cert. ↓", "false_certification_rate", "min", 3),
    ("No-feasible rate ↓", "no_feasible_rate", "min", 3),
    ("Runtime (s) ↓", "runtime_seconds", "min", 2),
)


def one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        found = "\n".join(str(path) for path in matches) or "(none)"
        raise RuntimeError(f"Expected one result for {pattern}, found {len(matches)}:\n{found}")
    return matches[0]


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("trial_records", {})
    if len(groups) != 1:
        raise RuntimeError(f"Expected one trial-record group in {path}, found {len(groups)}")
    records = next(iter(groups.values()))
    if len(records) != 10:
        raise RuntimeError(f"Expected 10 trials in {path}, found {len(records)}")
    return records


def summarize(records: list[dict], ddof: int) -> dict[str, tuple[float, float]]:
    output = {}
    for _, key, _, _ in METRICS:
        values = np.asarray([float(record[key]) for record in records], dtype=float)
        output[key] = (float(values.mean()), float(values.std(ddof=ddof)))
    return output


def format_value(value: tuple[float, float], digits: int, bold: bool) -> str:
    text = f"{value[0]:.{digits}f} ± {value[1]:.{digits}f}"
    return f"**{text}**" if bold else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    patterns = (
        ("FA-SAL (UM)", 1, "penalty_stress_fa_sal_policy_uncertainty_max_lf_scale_1_l_ell_scale_0p001_m8_evalm8_lfestlegacy_axis_quantile_*_rounds150_trials10_results.json"),
        ("FA-SAL (RS)", 0, "penalty_stress_fa_sal_policy_random_safe_lf_scale_1_l_ell_scale_0p001_m8_evalm8_lfestlegacy_axis_quantile_*_rounds150_trials10_results.json"),
        ("FA-SAL (GM)", 0, "penalty_stress_fa_sal_policy_greedy_margin_lf_scale_1_l_ell_scale_0p001_m8_evalm8_lfestlegacy_axis_quantile_*_rounds150_trials10_results.json"),
        ("SAL-NX", 1, "penalty_stress_salnx_alpha_0p01_m8_evalm8_lfestlegacy_axis_quantile_*_rounds150_trials10_results.json"),
        ("Tebbe-ABM", 1, "penalty_stress_tebbe_abm_alpha_0p01_confidence_delta_0p01_m8_evalm8_ucand11_mcprefix0_localref0_lfestlegacy_axis_quantile_*_rounds150_trials10_results.json"),
    )
    summaries = [(method, summarize(load_records(one_match(args.result_dir, pattern)), ddof)) for method, ddof, pattern in patterns]

    rounded_best = {}
    for _, key, direction, digits in METRICS:
        values = [round(summary[key][0], digits) for _, summary in summaries]
        rounded_best[key] = min(values) if direction == "min" else max(values)

    lines = [
        "| Method | " + " | ".join(label for label, _, _, _ in METRICS) + " |",
        "| " + " | ".join(["---"] * (len(METRICS) + 1)) + " |",
    ]
    for method, summary in summaries:
        cells = []
        for _, key, _, digits in METRICS:
            mean = summary[key][0]
            cells.append(format_value(summary[key], digits, round(mean, digits) == rounded_best[key]))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
