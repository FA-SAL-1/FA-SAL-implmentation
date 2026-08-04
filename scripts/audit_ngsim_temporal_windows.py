#!/usr/bin/env python3
"""Audit an NGSIM manifest against the exact transition pool used at runtime."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ngsim_experiment import _column_map, load_ngsim_data


ParticipantState = tuple[str, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT
        / "Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "ngsim_split.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "audits" / "ngsim_audit.json",
    )
    parser.add_argument("--max-transitions", type=int, default=5_000)
    parser.add_argument("--pool-seed", type=int, default=0)
    parser.add_argument("--distance-unit-scale", type=float, default=0.3048)
    parser.add_argument("--min-headway", type=float, default=8.0)
    parser.add_argument("--safety-headway-offset", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--residual-neighbors", type=int, default=16)
    return parser.parse_args()


def normalized_location(value: object) -> str:
    return str(value).strip().lower()


def record_state(record: dict[str, object]) -> ParticipantState:
    return (
        normalized_location(record["location"]),
        int(record["vehicle_id"]),
        int(record["frame_id"]),
    )


def load_one_file_with_metadata(
    path: Path,
    unit_scale: float,
    max_source_rows: int,
    dt: float,
    frame_stride: int,
) -> tuple[np.ndarray, np.ndarray, list[set[ParticipantState]]]:
    records: dict[tuple[int, int], dict[str, object]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return np.empty((0, 4)), np.empty((0,)), []
        cmap = _column_map(reader.fieldnames)
        location_column = next(
            (name for name in reader.fieldnames if name.strip().lower() == "location"),
            None,
        )
        if location_column is None:
            raise ValueError(f"CSV has no Location column: {path}")

        for row_index, raw in enumerate(reader):
            if max_source_rows > 0 and row_index >= max_source_rows:
                break
            try:
                vehicle_id = int(float(raw[cmap["vehicle_id"]]))
                frame_id = int(float(raw[cmap["frame_id"]]))
                preceding = int(float(raw[cmap["preceding"]]))
                record: dict[str, object] = {
                    "vehicle_id": vehicle_id,
                    "frame_id": frame_id,
                    "preceding": preceding,
                    "local_y": float(raw[cmap["local_y"]]),
                    "v_vel": float(raw[cmap["v_vel"]]),
                    "v_acc": float(raw[cmap["v_acc"]]),
                    "v_length": float(raw[cmap["v_length"]]) if "v_length" in cmap else 0.0,
                    "space_headway": (
                        float(raw[cmap["space_headway"]])
                        if "space_headway" in cmap
                        else float("nan")
                    ),
                    "location": raw[location_column],
                }
            except (KeyError, TypeError, ValueError):
                continue
            # This intentionally matches the runtime loader's current key and
            # overwrite behavior, including cross-location key collisions.
            records[(vehicle_id, frame_id)] = record

    rows: list[tuple[float, float, float, float]] = []
    targets: list[float] = []
    participants: list[set[ParticipantState]] = []
    scale = float(unit_scale)
    stride = max(1, int(frame_stride))

    for (vehicle_id, frame_id), ego in records.items():
        ego_next = records.get((vehicle_id, frame_id + stride))
        preceding = int(ego["preceding"])
        leader = records.get((preceding, frame_id)) if preceding > 0 else None
        current_participants = {record_state(ego)}
        if ego_next is not None:
            current_participants.add(record_state(ego_next))
        if leader is not None:
            current_participants.add(record_state(leader))

        headway_value = float(ego["space_headway"])
        next_headway_value = float(ego_next["space_headway"]) if ego_next is not None else float("nan")
        if ego_next is not None and np.isfinite(headway_value) and np.isfinite(next_headway_value):
            headway = headway_value * scale
            y_next = next_headway_value * scale
            action = float(ego["v_acc"]) * scale
            if leader is not None:
                relative_speed = (float(leader["v_vel"]) - float(ego["v_vel"])) * scale
            else:
                relative_speed = (y_next - headway) / max(float(dt), 1e-8) + float(dt) * action
            rows.append((headway, relative_speed, float(ego["v_vel"]) * scale, action))
            targets.append(y_next)
            participants.append(current_participants)
            continue

        if preceding <= 0:
            continue
        leader_next = records.get((preceding, frame_id + stride))
        if leader is None or ego_next is None or leader_next is None:
            continue
        current_participants.add(record_state(leader_next))
        leader_length = float(leader["v_length"])
        headway = (float(leader["local_y"]) - float(ego["local_y"]) - leader_length) * scale
        relative_speed = (float(leader["v_vel"]) - float(ego["v_vel"])) * scale
        action = float(ego["v_acc"]) * scale
        y_next = (float(leader_next["local_y"]) - float(ego_next["local_y"]) - leader_length) * scale
        rows.append((headway, relative_speed, float(ego["v_vel"]) * scale, action))
        targets.append(y_next)
        participants.append(current_participants)

    if not rows:
        return np.empty((0, 4)), np.empty((0,)), []
    X_raw = np.asarray(rows, dtype=float)
    y_raw = np.asarray(targets, dtype=float)
    mask = (
        np.isfinite(X_raw).all(axis=1)
        & np.isfinite(y_raw)
        & (X_raw[:, 0] > 0.5)
        & (X_raw[:, 0] < 150.0)
        & (y_raw > 0.0)
        & (y_raw < 150.0)
        & (X_raw[:, 2] >= 0.0)
        & (np.abs(X_raw[:, 1]) <= 40.0)
        & (np.abs(y_raw - X_raw[:, 0]) <= 12.0)
        & (np.abs(X_raw[:, 3]) < 12.0)
    )
    kept = np.flatnonzero(mask)
    return X_raw[mask], y_raw[mask], [participants[int(index)] for index in kept]


def load_runtime_pool_with_metadata(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[set[ParticipantState]]]:
    path = args.csv
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    max_source_rows = max(100_000, args.max_transitions * 200) if args.max_transitions > 0 else 0
    effective_dt = args.dt * max(1, args.frame_stride)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    participant_parts: list[set[ParticipantState]] = []
    for file_path in files:
        X, y, participants = load_one_file_with_metadata(
            file_path,
            args.distance_unit_scale,
            max_source_rows,
            effective_dt,
            args.frame_stride,
        )
        if X.size:
            X_parts.append(X)
            y_parts.append(y)
            participant_parts.extend(participants)
    if not X_parts:
        raise ValueError("No valid transitions were reconstructed.")
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    if args.max_transitions > 0 and len(X) > args.max_transitions:
        selected = np.random.default_rng(args.pool_seed).choice(
            len(X), size=args.max_transitions, replace=False
        )
        X = X[selected]
        y = y[selected]
        participant_parts = [participant_parts[int(index)] for index in selected]
    return X, y, participant_parts


def union_participants(indices: Iterable[int], metadata: list[set[ParticipantState]]) -> set[ParticipantState]:
    result: set[ParticipantState] = set()
    for index in indices:
        result.update(metadata[int(index)])
    return result


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    X_meta, _y_meta, metadata = load_runtime_pool_with_metadata(args)
    effective_min_headway = args.min_headway - args.safety_headway_offset
    data = load_ngsim_data(
        args.csv,
        args.max_transitions,
        args.distance_unit_scale,
        effective_min_headway,
        args.dt,
        args.frame_stride,
        args.residual_neighbors,
        args.pool_seed,
        True,
        0.0,
        -10.0,
    )
    if X_meta.shape != data.X.shape or not np.allclose(X_meta, data.X, equal_nan=True):
        raise RuntimeError("Audit metadata reconstruction does not match the runtime transition pool.")

    xy = np.column_stack((data.X, data.y_next))
    reports = []
    for trial in manifest["trials"]:
        train = set(map(int, trial["initial_train_pool_indices"]))
        global_eval = set(map(int, trial["global_eval_pool_indices"]))
        rmse_eval = set(map(int, trial["rmse_eval_pool_indices"]))
        evaluation = global_eval | rmse_eval
        for index in train | evaluation:
            if index < 0 or index >= len(data.X):
                raise IndexError(f"Manifest index {index} is outside the transition pool.")
        train_xy = {tuple(xy[index]) for index in train}
        evaluation_xy = {tuple(xy[index]) for index in evaluation}
        overlap = sorted(union_participants(train, metadata) & union_participants(evaluation, metadata))
        reports.append(
            {
                "trial_seed": int(trial["trial_seed"]),
                "initial_count": len(train),
                "global_eval_count": len(global_eval),
                "rmse_eval_count": len(rmse_eval),
                "index_overlap_count": len(train & evaluation),
                "exact_xy_overlap_count": len(train_xy & evaluation_xy),
                "participant_temporal_window_overlap_count": len(overlap),
                "participant_temporal_window_overlaps": [list(item) for item in overlap],
            }
        )

    report = {
        "schema": "ngsim_runtime_pool_leakage_audit_v2",
        "pool_semantics": "matches ngsim_experiment.py, including current record overwrite behavior",
        "transition_pool_count": len(data.X),
        "all_trials_pass": all(
            row["initial_count"] == 80
            and row["global_eval_count"] == 100
            and row["rmse_eval_count"] == 100
            and row["index_overlap_count"] == 0
            and row["exact_xy_overlap_count"] == 0
            and row["participant_temporal_window_overlap_count"] == 0
            for row in reports
        ),
        "trials": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
