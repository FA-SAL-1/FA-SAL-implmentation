#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    ("RMSE ↓", "rmse"),
    ("SVR ↓", "svr"),
    ("D-SVR ↓", "d_svr"),
    ("Safety IoU ↑", "safety_iou"),
    ("DE IoU ↑", "dead_end_iou"),
    ("False cert. ↓", "false_certification_rate"),
)


def one_match(root: Path, scale: str) -> Path:
    pattern = (
        "penalty_stress_fa_sal_policy_uncertainty_max_"
        f"lf_scale_{scale}_l_ell_scale_0p14_m1_evalm1_lfestjacobian_*_"
        "rounds150_trials10_results.json"
    )
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        found = "\n".join(str(path) for path in matches) or "(none)"
        raise RuntimeError(f"Expected one result for scale {scale}, found {len(matches)}:\n{found}")
    return matches[0]


def summarize(path: Path) -> dict[str, tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("trial_records", {})
    if len(groups) != 1:
        raise RuntimeError(f"Expected one trial-record group in {path}, found {len(groups)}")
    records = next(iter(groups.values()))
    if len(records) != 10:
        raise RuntimeError(f"Expected 10 trials in {path}, found {len(records)}")
    output = {}
    for _, key in METRICS:
        values = np.asarray([float(record[key]) for record in records], dtype=float)
        output[key] = (float(values.mean()), float(values.std(ddof=0)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scales = (("0p5", "0.5"), ("1", "1"), ("2", "2"))
    lines = [
        "| $L_f$ scale | " + " | ".join(label for label, _ in METRICS) + " |",
        "| " + " | ".join(["---"] * (len(METRICS) + 1)) + " |",
    ]
    for file_scale, display_scale in scales:
        summary = summarize(one_match(args.result_dir, file_scale))
        cells = [f"{summary[key][0]:.3f} ± {summary[key][1]:.3f}" for _, key in METRICS]
        lines.append(f"| ${display_scale}\\times$ | " + " | ".join(cells) + " |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
