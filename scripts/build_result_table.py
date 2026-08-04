#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


METRICS = (
    ("RMSE ↓", ("dynamics_rmse", "rmse")),
    ("SVR ↓", ("safety_violation_rate",)),
    ("D-SVR ↓", ("dead_end_safety_violation_rate",)),
    (
        "Safety IoU ↑",
        (
            "safety_set_recovery_iou_active",
            "safe_coverage_iou",
            "safety_set_recovery_iou",
        ),
    ),
    (
        "DE IoU ↑",
        ("dead_end_free_recovery_iou_active", "dead_end_free_recovery_iou"),
    ),
    ("False cert. ↓", ("false_certification_rate",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def final_scalar(summary: dict, bases: tuple[str, ...], suffix: str) -> float:
    for base in bases:
        key = f"{base}_{suffix}"
        if key not in summary:
            continue
        value = summary[key]
        if isinstance(value, list):
            if not value:
                raise ValueError(f"Empty metric array: {key}")
            return float(value[-1])
        return float(value)
    raise KeyError(f"Missing metric with suffix {suffix}: {bases}")


def load_rows(result_dir: Path) -> dict[str, list[tuple[float, float]]]:
    rows: dict[str, list[tuple[float, float]]] = {}
    paths = sorted(result_dir.glob("*results.json"))
    for path in paths:
        if path.name == "rail_pressure_results.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries = payload.get("summaries")
        if not isinstance(summaries, dict):
            continue
        for method, summary in summaries.items():
            values = [
                (final_scalar(summary, bases, "mean"), final_scalar(summary, bases, "std"))
                for _, bases in METRICS
            ]
            if method in rows and rows[method] != values:
                raise RuntimeError(f"Multiple distinct summaries for {method} in {result_dir}")
            rows[method] = values
    if not rows:
        raise RuntimeError(f"No compatible result JSON files found in {result_dir}")
    return rows


def method_order(method: str) -> tuple[int, str]:
    prefixes = ("FA-SAL", "SAL-NX", "Tebbe-ABM")
    for index, prefix in enumerate(prefixes):
        if method.startswith(prefix):
            return index, method
    return len(prefixes), method


def build_table(rows: dict[str, list[tuple[float, float]]]) -> str:
    headers = ["Method"] + [label for label, _ in METRICS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for method in sorted(rows, key=method_order):
        cells = [method] + [f"{mean:.3f} ± {std:.3f}" for mean, std in rows[method]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    table = build_table(load_rows(args.result_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
