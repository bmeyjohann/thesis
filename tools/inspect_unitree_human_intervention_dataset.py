#!/usr/bin/env python3
"""Audit a raw Unitree human-intervention dataset without modifying it."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit immutable Unitree human intervention NPZ chunks")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--min-action-delta", type=float, default=0.05)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = 0
    finite_rows = 0
    intervention_rows = 0
    nontrivial_intervention_rows = 0
    starts = 0
    ends = 0
    costful_rows = 0
    completed_rows = 0
    stale_rows = 0
    connected_rows = 0
    source_counts: Counter[str] = Counter()
    action_deltas: list[np.ndarray] = []
    intervention_lengths: list[int] = []
    active_length = 0
    expected_keys: set[str] | None = None

    for entry in manifest.get("parts", []):
        source = root / str(entry["file"])
        with np.load(source, allow_pickle=False) as chunk:
            keys = set(chunk.files)
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(f"Schema mismatch in {source}")
            count = int(chunk["intervened"].shape[0])
            if int(entry.get("rows", count)) != count:
                raise ValueError(f"Manifest row count mismatch in {source}")
            rows += count
            all_finite = np.ones(count, dtype=bool)
            for key in chunk.files:
                value = chunk[key]
                if value.dtype.kind == "f":
                    all_finite &= np.isfinite(value.reshape(count, -1)).all(axis=1)
            finite_rows += int(all_finite.sum())
            intervened = chunk["intervened"].astype(bool)
            starts += int(chunk["intervention_start"].astype(bool).sum())
            ends += int(chunk["intervention_end"].astype(bool).sum())
            intervention_rows += int(intervened.sum())
            delta = chunk["action_delta_to_policy"].astype(np.float64)
            nontrivial_intervention_rows += int((intervened & (delta >= float(args.min_action_delta))).sum())
            action_deltas.append(delta[intervened])
            costful_rows += int((chunk["cost"] > 0.0).sum())
            completed_rows += int(chunk["done"].astype(bool).sum())
            stale_rows += int(chunk["gamepad_stale"].astype(bool).sum())
            connected_rows += int(chunk["gamepad_connected"].astype(bool).sum())
            for source_id, count_for_source in Counter(chunk["control_source"].astype(int).tolist()).items():
                source_counts[manifest.get("control_source_names", {}).get(str(source_id), str(source_id))] += int(count_for_source)
            for active in intervened.tolist():
                if active:
                    active_length += 1
                elif active_length:
                    intervention_lengths.append(active_length)
                    active_length = 0
    if active_length:
        intervention_lengths.append(active_length)
    deltas = np.concatenate(action_deltas) if action_deltas else np.zeros((0,), dtype=np.float64)
    report = {
        "dataset_dir": str(root),
        "rows": rows,
        "parts": len(manifest.get("parts", [])),
        "finite_row_fraction": finite_rows / max(1, rows),
        "intervention_fraction": intervention_rows / max(1, rows),
        "nontrivial_intervention_fraction": nontrivial_intervention_rows / max(1, rows),
        "intervention_starts": starts,
        "intervention_ends": ends,
        "mean_intervention_segment_steps": float(np.mean(intervention_lengths)) if intervention_lengths else 0.0,
        "mean_intervention_action_delta": float(np.mean(deltas)) if deltas.size else 0.0,
        "costful_step_fraction": costful_rows / max(1, rows),
        "terminal_row_fraction": completed_rows / max(1, rows),
        "gamepad_connected_fraction": connected_rows / max(1, rows),
        "gamepad_stale_fraction": stale_rows / max(1, rows),
        "control_source_counts": dict(source_counts),
        "recommended_preference_rows": nontrivial_intervention_rows,
        "recommendation": (
            "Keep the raw dataset unchanged. Build any training subset from nontrivial intervention rows "
            "plus their policy context, after inspecting intervention segments and transport health."
        ),
    }
    output = Path(args.output).expanduser().resolve() if args.output else root / "audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
