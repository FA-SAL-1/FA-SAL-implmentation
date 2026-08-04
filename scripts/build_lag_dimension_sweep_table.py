import argparse
import json
from pathlib import Path


DIMENSIONS = ((2, 1), (4, 1), (8, 1), (2, 2), (2, 4), (2, 8), (4, 4), (8, 8))
METRICS = (
    ("dynamics_rmse", "RMSE ↓", 3),
    ("safety_violation_rate", "SVR ↓", 3),
    ("dead_end_safety_violation_rate", "D-SVR ↓", 3),
    ("safety_set_recovery_iou", "Safety IoU ↑", 3),
    ("dead_end_free_recovery_iou", "DE IoU ↑", 3),
    ("false_certification_rate", "False cert. ↓", 3),
    ("no_feasible_rate", "No-feasible ↓", 3),
    ("runtime_seconds", "Runtime (s) ↓", 2),
)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_summary(root, dy, du):
    folder = root / f"dy{dy}_du{du}"
    matches = sorted(folder.glob("penalty_stress_fa_sal_policy_random_safe_lf_scale_1_l_ell_scale_0p001_m4_evalm4_lfestlegacy_axis_quantile_lfq0p5_lfgrid31_rounds150_trials10_results.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one result in {folder}, found {len(matches)}")
    payload = json.loads(matches[0].read_text())
    config = payload["config"]
    expected = {
        "environment_preset": "penalty_stress",
        "rounds": 150,
        "trials": 10,
        "evaluation_horizon": 4,
        "fa_horizon": 4,
        "fa_lf_scale_sweep": [1.0],
        "fa_l_ell_scale_sweep": [0.001],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{matches[0]}: expected {key}={value}, got {config.get(key)}")
    return next(iter(payload["summaries"].values()))


def scalar(value):
    return value[-1] if isinstance(value, list) else value


def build_table(root):
    headers = ["$d_y$", "$d_u$", "$d_x$"] + [label for _, label, _ in METRICS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for dy, du in DIMENSIONS:
        summary = load_summary(root, dy, du)
        cells = [str(dy), str(du), str(dy + du)]
        for key, _, digits in METRICS:
            mean = float(scalar(summary[f"{key}_mean"]))
            std = float(scalar(summary[f"{key}_std"]))
            cell = f"{mean:.{digits}f} ± {std:.{digits}f}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    table = build_table(args.results_root)
    print(table, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table)


if __name__ == "__main__":
    main()
