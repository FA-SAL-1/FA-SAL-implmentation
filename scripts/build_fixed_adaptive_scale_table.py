import argparse
import json
from pathlib import Path


ROWS = (
    ("Fixed", "$1\\times$", "None", "nominal_fixed", "fixed"),
    ("Nominal-adaptive", "$1\\times$", "GP, $\\widehat L_{f,t}$, $\\widehat L_{\\ell,t}$", "nominal_adaptive_lf", "nominal"),
    ("Low-fixed", "$0.5\\times$", "None", "joint_low_fixed", "scale"),
    ("Low-adaptive", "$0.5\\times$", "GP, $\\widehat L_{f,t}$, $\\widehat L_{\\ell,t}$", "joint_low_adaptive_lf", "scale"),
    ("High-fixed", "$2\\times$", "None", "joint_high_fixed", "scale"),
    ("High-adaptive", "$2\\times$", "GP, $\\widehat L_{f,t}$, $\\widehat L_{\\ell,t}$", "joint_high_adaptive_lf", "scale"),
)
METRICS = (
    ("rmse", "RMSE ↓"),
    ("svr", "SVR ↓"),
    ("d_svr", "D-SVR ↓"),
    ("safety_iou", "Safety IoU ↑"),
    ("de_iou", "DE IoU ↑"),
    ("false_certification", "False Cert. ↓"),
    ("no_feasible_rate", "No-feasible Rate ↓"),
)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scale-results", type=Path, required=True)
    parser.add_argument("--nominal-adaptive-results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text())


def validate(payload, path):
    experiment = payload["experiment"]
    expected = {"benchmark": "penalty_stress", "horizon": 4, "evaluation_horizon": 4, "rounds": 150}
    for key, value in expected.items():
        if experiment.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value}, got {experiment.get(key)}")
    if experiment.get("paired_seeds") != list(range(10)):
        raise ValueError(f"{path}: expected paired seeds 0 through 9")


def build_table(args):
    payloads = {
        "fixed": load(args.input),
        "scale": load(args.scale_results),
        "nominal": load(args.nominal_adaptive_results),
    }
    paths = {"fixed": args.input, "scale": args.scale_results, "nominal": args.nominal_adaptive_results}
    for key, payload in payloads.items():
        validate(payload, paths[key])
    headers = ["Setting", "Scale $s$", "Update"] + [label for _, label in METRICS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for setting, scale, update, condition, source in ROWS:
        summary = payloads[source]["summary"]
        if source != "nominal":
            summary = summary[condition]
        cells = [setting, scale, update]
        for key, _ in METRICS:
            metric = summary[key]
            cell = f"{float(metric['mean']):.3f} ± {float(metric['std']):.3f}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    table = build_table(args)
    print(table, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table)


if __name__ == "__main__":
    main()
