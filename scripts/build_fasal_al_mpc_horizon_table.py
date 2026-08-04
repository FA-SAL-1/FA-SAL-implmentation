#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    ("RMSE ↓", "rmse", "min"),
    ("SVR ↓", "svr", "min"),
    ("D-SVR ↓", "d_svr", "min"),
    ("Safety IoU ↑", "safety_iou", "max"),
    ("Dead-end IoU ↑", "dead_end_iou", "max"),
    ("False-cert ↓", "false_certification_rate", "min"),
)


def one_match(root: Path, patterns: str | tuple[str, ...]) -> Path:
    if isinstance(patterns, str):
        patterns = (patterns,)
    matches = sorted({path for pattern in patterns for path in root.glob(pattern)})
    if len(matches) != 1:
        found = "\n".join(str(path) for path in matches) or "(none)"
        expected = " or ".join(patterns)
        raise RuntimeError(f"Expected one result for {expected}, found {len(matches)}:\n{found}")
    return matches[0]


def summarize(path: Path, ddof: int) -> dict[str, tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("trial_records", {})
    if len(groups) != 1:
        raise RuntimeError(f"Expected one trial-record group in {path}, found {len(groups)}")
    records = next(iter(groups.values()))
    if len(records) != 10:
        raise RuntimeError(f"Expected 10 trials in {path}, found {len(records)}")
    output = {}
    for _, key, _ in METRICS:
        values = np.asarray([float(record[key]) for record in records], dtype=float)
        output[key] = (float(values.mean()), float(values.std(ddof=ddof)))
    return output


def fa_pattern(m: int) -> tuple[str, str]:
    if m == 1:
        suffix = "lf_scale_1_l_ell_scale_0p14_m1_evalm1_lfestjacobian_*_rounds150_trials10_results.json"
        return tuple(f"penalty_stress_fa_sal_{prefix}{suffix}" for prefix in ("policy_uncertainty_max_", ""))
    l_ell = "0p001" if m == 8 else "0p14"
    estimator = "jacobian" if m == 4 else "legacy_axis_quantile"
    suffix = f"lf_scale_1_l_ell_scale_{l_ell}_m{m}_evalm{m}_lfest{estimator}_*_rounds150_trials10_results.json"
    return tuple(f"penalty_stress_fa_sal_{prefix}{suffix}" for prefix in ("policy_uncertainty_max_", ""))


def al_pattern(m: int) -> str:
    return f"penalty_stress_al_mpc_m{m}_evalm{m}_lfestjacobian_*_rounds{150 * m}_trials10_results.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for m in (1, 2, 4, 8):
        fa_path = one_match(args.result_root / f"m{m}" / "fa_sal", fa_pattern(m))
        fa_summary = summarize(fa_path, 1 if m == 8 else 0)
        al_path = one_match(args.result_root / f"m{m}" / "al_mpc", al_pattern(m))
        al_summary = summarize(al_path, 0)
        rows.extend(((m, "FA-SAL", fa_summary), (m, "AL-MPC", al_summary)))

    lines = [
        "| m | Method | " + " | ".join(label for label, _, _ in METRICS) + " |",
        "| " + " | ".join(["---"] * (len(METRICS) + 2)) + " |",
    ]
    for index in range(0, len(rows), 2):
        pair = rows[index:index + 2]
        best = {}
        for _, key, direction in METRICS:
            values = [summary[key][0] for _, _, summary in pair]
            best[key] = min(values) if direction == "min" else max(values)
        for pair_index, (m, method, summary) in enumerate(pair):
            cells = []
            for _, key, _ in METRICS:
                value = summary[key]
                text = f"{value[0]:.3f} ± {value[1]:.3f}"
                cells.append(f"**{text}**" if value[0] == best[key] else text)
            lines.append(f"| {m if pair_index == 0 else ''} | {method} | " + " | ".join(cells) + " |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
