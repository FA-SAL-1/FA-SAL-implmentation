import argparse
import json
from pathlib import Path
ROWS = (("FA-SAL", "$(1,1)$", "grid", "sf_1_sl_1"), ("FA-SAL", "$(0.5,0.5)$", "grid", "sf_0.5_sl_0.5"), ("FA-SAL", "$(0.5,1)$", "grid", "sf_0.5_sl_1"), ("FA-SAL", "$(0.5,2)$", "grid", "sf_0.5_sl_2"), ("FA-SAL", "$(1,0.5)$", "grid", "sf_1_sl_0.5"), ("FA-SAL", "$(1,2)$", "grid", "sf_1_sl_2"), ("FA-SAL", "$(2,0.5)$", "grid", "sf_2_sl_0.5"), ("FA-SAL", "$(2,1)$", "grid", "sf_2_sl_1"), ("FA-SAL", "$(2,2)$", "grid", "sf_2_sl_2"))
METRICS = (("rmse", "RMSE \u2193"), ("svr", "SVR \u2193"), ("d_svr", "D-SVR \u2193"), ("safety_iou", "Safety IoU \u2191"), ("de_iou", "DE IoU \u2191"), ("false_certification", "False Cert. \u2193"), ("no_feasible_rate", "No-feasible Rate \u2193"))
def scalar(value):
    return value[-1] if isinstance(value, list) else value
def generic(path):
    summary = next(iter(json.loads(path.read_text())["summaries"].values()))
    mapping = {"rmse": "dynamics_rmse", "svr": "safety_violation_rate", "d_svr": "dead_end_safety_violation_rate", "safety_iou": "safety_set_recovery_iou", "de_iou": "dead_end_free_recovery_iou", "false_certification": "false_certification_rate", "no_feasible_rate": "no_feasible_rate"}
    return {key: {"mean": scalar(summary[f"{source}_mean"]), "std": scalar(summary[f"{source}_std"])} for key, source in mapping.items()}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources = {"grid": json.loads(args.grid.read_text())["summary"]}
    headers = ["Method", "Scale $(s_f,s_\\ell)$"] + [label for _, label in METRICS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for method, scale, source, condition in ROWS:
        key = condition or source
        summary = sources[source] if source != "grid" else sources[source][condition]
        cells = [method, scale]
        for metric, _ in METRICS:
            value = summary[metric]
            cell = f"{float(value['mean']):.3f} ± {float(value['std']):.3f}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    print(table, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table)
if __name__ == "__main__":
    main()
