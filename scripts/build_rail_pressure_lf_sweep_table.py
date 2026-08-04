import argparse
import json
import statistics
from pathlib import Path



SCALES = ("0.5", "1", "2", "4", "8")
METRICS = (
    ("rmse", "RMSE ↓"),
    ("safety_violation_rate", "SVR ↓"),
    ("dead_end_safety_violation_rate", "D-SVR ↓"),
    ("safe_coverage_iou", "Safety IoU ↑"),
    ("dead_end_free_recovery_iou", "DE IoU ↑"),
    ("false_certification_rate", "False cert. ↓"),
    ("mean_feasible_ratio", "Feasible ratio ↑"),
)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def result_file(root, scale):
    matches = sorted((root / f"lf_{scale}").glob("rail_pressure_fa_sal_l_ell_scale_0p0001_m5_rounds150_trials10_seed0_results.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one result for Lf scale {scale}, found {len(matches)}")
    return matches[0]


def final_values(path):
    payload = json.loads(path.read_text())
    config = payload["config"]
    expected = {"m": 5, "rounds": 150, "trials": 10, "seed": 0, "fa_sal_l_ell_scale_sweep": [0.0001]}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value}, got {config.get(key)}")
    records = next(iter(payload["trials"].values()))
    values = {}
    for key, _ in METRICS[:-1]:
        values[key] = [float(record[key][-1]) for record in records]
    values["mean_feasible_ratio"] = [float(record["feasible_sizes"][-1]) / 441.0 for record in records]
    return values


def format_value(values):
    return f"{statistics.fmean(values):.3f} ± {statistics.stdev(values):.3f}"


def build_table(root):
    headers = ["$L_f$ scale"] + [label for _, label in METRICS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for scale in SCALES:
        values = final_values(result_file(root, scale))
        cells = [f"${scale}\\times$"]
        for key, _ in METRICS:
            cell = format_value(values[key])
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
