#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("RUN_ALL_DIR", REPO / "result" / "run_all"))


def env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key] = value
    return result


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown (waiting for timing data)"
    return str(timedelta(seconds=max(0, round(seconds))))


def historical_times() -> dict[str, list[int]]:
    values: dict[str, list[int]] = {}
    path = ROOT / "history.tsv"
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            fields = line.split("\t")
            if len(fields) >= 2 and fields[1].isdigit():
                values.setdefault(fields[0], []).append(int(fields[1]))
    return values


def render() -> str:
    latest = ROOT / "latest"
    if not latest.exists():
        return f"No run found under {ROOT}\nStart: ./scripts/run_all_sequential.sh --tmux"
    run_id = latest.read_text().strip()
    run_dir = ROOT / run_id
    meta = env_file(run_dir / "run.env")
    queue = (run_dir / "queue.txt").read_text().splitlines()
    rows = []
    state = run_dir / "state.tsv"
    if state.exists():
        for line in state.read_text(errors="replace").splitlines()[1:]:
            fields = line.split("\t")
            if len(fields) >= 9:
                rows.append(fields)

    now = int(time.time())
    total = int(meta.get("total", len(queue)))
    completed = len(rows)
    success = sum(row[2] == "SUCCESS" for row in rows)
    failed = sum(row[2] == "FAILED" for row in rows)
    current = env_file(run_dir / "current.env")
    current_name = current.get("script")
    current_elapsed = now - int(current["start_epoch"]) if current.get("start_epoch") else 0

    history = historical_times()
    observed = [int(row[5]) for row in rows]
    fallback = statistics.median(observed) if observed else None
    remaining_estimates = []
    for name in queue[completed:]:
        samples = history.get(name, [])
        remaining_estimates.append(statistics.median(samples) if samples else fallback)
    if current_name and remaining_estimates and remaining_estimates[0] is not None:
        remaining_estimates[0] = max(0, remaining_estimates[0] - current_elapsed)
    eta_seconds = None if any(x is None for x in remaining_estimates) else sum(remaining_estimates)

    started = int(meta.get("started_epoch", now))
    finished = meta.get("finished_epoch")
    lines = [
        f"FA-SAL sequential experiments — {datetime.now():%F %T}",
        f"Run:       {run_id}",
        f"Progress:  {completed}/{total} completed ({success} success, {failed} failed)",
        f"Elapsed:   {duration(now - started)}",
        f"Remaining: {duration(eta_seconds)}",
    ]
    if eta_seconds is not None:
        lines.append(f"ETA:       {datetime.now() + timedelta(seconds=eta_seconds):%F %T}")
    if finished:
        lines.append(f"State:     FINISHED at {datetime.fromtimestamp(int(finished)):%F %T}")
    elif current_name:
        lines.extend(
            [
                f"Running:   [{current.get('index')}/{total}] {current_name}",
                f"Run time:  {duration(current_elapsed)}",
                f"Live log:  {current.get('log')}",
            ]
        )
    else:
        lines.append("State:     between scripts / starting")

    if rows:
        lines.append("\nCompleted scripts:")
        for row in rows[-8:]:
            lines.append(f"  [{row[0]}/{row[1]}] {row[2]:7} {duration(int(row[5])):>10}  {row[7]}")
    lines.extend(
        [
            f"\nRun records: {run_dir}",
            f"Results:     {REPO / 'result'}",
            "Note: ETA becomes available after one script finishes, or from prior history.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="refresh every 10 seconds")
    args = parser.parse_args()
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        print(render(), flush=True)
        if not args.watch:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
