#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


def one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        found = "\n".join(str(path) for path in matches) or "(none)"
        raise RuntimeError(f"Expected one result for {pattern}, found {len(matches)}:\n{found}")
    return matches[0]


def summarize(path: Path) -> tuple[tuple[float, float], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("trial_records", {})
    if len(groups) != 1:
        raise RuntimeError(f"Expected one trial-record group in {path}, found {len(groups)}")
    records = next(iter(groups.values()))
    if len(records) != 10:
        raise RuntimeError(f"Expected 10 trials in {path}, found {len(records)}")
    values = (
        np.asarray([float(record["mean_feasible_ratio"]) * float(record["mean_candidate_count"]) for record in records]),
        np.asarray([float(record["mean_feasible_ratio"]) for record in records]),
        np.asarray([float(record["no_feasible_rate"]) for record in records]),
    )
    return tuple((float(value.mean()), float(value.std(ddof=1))) for value in values)


def format_value(value: tuple[float, float]) -> str:
    return f"{value[0] + 1e-12:.3f} ± {value[1] + 1e-12:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for m in (1, 2, 4, 8):
        suffix = f"m{m}_evalm{m}_lfestlegacy_axis_quantile_*_rounds150_trials10_results.json"
        fa_root = args.result_root / f"m{m}" / "fa_sal"
        um = one_match(fa_root, f"penalty_stress_fa_sal_policy_uncertainty_max_lf_scale_1_l_ell_scale_0p001_{suffix}")
        gm = one_match(fa_root, f"penalty_stress_fa_sal_policy_greedy_margin_lf_scale_1_l_ell_scale_0p001_{suffix}")
        al_root = args.result_root / f"m{m}" / "al_mpc"
        al = one_match(al_root, f"penalty_stress_al_mpc_m{m}_evalm{m}_lfestjacobian_*_rounds{150 * m}_trials10_results.json")
        rows.extend(((m, "FA-SAL", summarize(um)), (m, "FA-SAL (greedy-margin)", summarize(gm)), (m, "AL-MPC", summarize(al))))

    lines = [
        "| $m$ | Method | $\\lvert\\mathcal U_{t,m}^{\\mathrm{feas}}\\rvert$ ↑ | Feasible ratio ↑ | No-feasible rate ↓ |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, (m, method, summary) in enumerate(rows):
        display_m = str(m) if index % 3 == 0 else ""
        lines.append(f"| {display_m} | {method} | " + " | ".join(format_value(value) for value in summary) + " |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
