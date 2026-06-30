#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _metadata(payload: dict[str, np.ndarray]) -> dict:
    raw = payload.get("metadata_json")
    if raw is None:
        return {}
    try:
        return json.loads(str(raw.item()))
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Concatenate Safety-Gym transition datasets with matching schemas.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    args = parser.parse_args()

    inputs = [Path(path).expanduser().resolve() for path in args.inputs]
    output = Path(args.output).expanduser().resolve()
    datasets = [_load(path) for path in inputs]
    if not datasets:
        raise ValueError("No input datasets provided.")

    required = ["observations", "actions", "next_observations", "rewards", "dones", "truncations"]
    optional = ["costs", "student_actions", "teacher_intervened", "episode_ids", "episode_steps"]
    first = datasets[0]
    for key in required:
        if key not in first:
            raise KeyError(f"Missing required key in first dataset: {key}")

    n_rows_total = 0
    merged: dict[str, np.ndarray] = {}
    for key in required + optional:
        if key not in first:
            continue
        arrays = []
        episode_offset = 0
        for ds_index, ds in enumerate(datasets):
            if key not in ds:
                raise KeyError(f"Dataset {inputs[ds_index]} is missing key present in first dataset: {key}")
            arr = np.asarray(ds[key])
            if key == "episode_ids":
                arr = arr.astype(np.int64, copy=True) + episode_offset
                episode_offset = int(arr.max()) + 1 if arr.size else episode_offset
            arrays.append(arr)
        merged[key] = np.concatenate(arrays, axis=0)
        if key == "actions":
            n_rows_total = int(merged[key].shape[0])

    obs_shape = tuple(merged["observations"].shape[1:])
    act_shape = tuple(merged["actions"].shape[1:])
    for key in ("next_observations",):
        if tuple(merged[key].shape[1:]) != obs_shape:
            raise ValueError(f"{key} shape {merged[key].shape[1:]} does not match observations shape {obs_shape}")
    for key in ("student_actions",):
        if key in merged and tuple(merged[key].shape[1:]) != act_shape:
            raise ValueError(f"{key} shape {merged[key].shape[1:]} does not match actions shape {act_shape}")

    metadata = {
        "source_datasets": [str(path) for path in inputs],
        "source_metadata": [_metadata(ds) for ds in datasets],
        "num_rows": n_rows_total,
        "merge_type": "concatenate",
    }
    merged["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=object)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": n_rows_total}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
