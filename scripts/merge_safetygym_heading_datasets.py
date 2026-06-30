#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge Safety-Gym heading-label datasets.")
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("datasets", nargs="+")
    return p


def _metadata(data) -> dict:
    if "metadata" not in data.files:
        return {}
    try:
        return json.loads(str(data["metadata"].item()))
    except Exception:
        return {}


def main() -> int:
    args = build_parser().parse_args()
    obs_parts: list[np.ndarray] = []
    angle_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    episode_id_parts: list[np.ndarray] = []
    episode_step_parts: list[np.ndarray] = []
    sources: list[dict] = []
    episode_offset = 0
    obs_dim: int | None = None

    for raw_path in args.datasets:
        path = Path(raw_path).expanduser()
        with np.load(path, allow_pickle=True) as data:
            obs = np.asarray(data["observations"], dtype=np.float32)
            angles = np.asarray(data["target_angles"], dtype=np.float32)
            actions = np.asarray(data["teacher_actions"], dtype=np.float32)
            episode_ids = np.asarray(data.get("episode_ids", np.zeros((obs.shape[0],), dtype=np.int64)), dtype=np.int64)
            episode_steps = np.asarray(data.get("episode_steps", np.zeros((obs.shape[0],), dtype=np.int64)), dtype=np.int64)
            if obs_dim is None:
                obs_dim = int(obs.shape[1])
            elif int(obs.shape[1]) != obs_dim:
                raise ValueError(f"Observation dim mismatch for {path}: {obs.shape[1]} != {obs_dim}")
            obs_parts.append(obs)
            angle_parts.append(angles)
            action_parts.append(actions)
            episode_id_parts.append(episode_ids + int(episode_offset))
            episode_step_parts.append(episode_steps)
            meta = _metadata(data)
            sources.append({"path": str(path), "rows": int(obs.shape[0]), "metadata": meta})
            episode_offset += int(episode_ids.max() + 1) if episode_ids.size else 0

    observations = np.concatenate(obs_parts, axis=0)
    target_angles = np.concatenate(angle_parts, axis=0)
    teacher_actions = np.concatenate(action_parts, axis=0)
    episode_ids = np.concatenate(episode_id_parts, axis=0)
    episode_steps = np.concatenate(episode_step_parts, axis=0)
    metadata = {
        "format": "safetygym_heading_labels",
        "merged": True,
        "sources": sources,
        "num_steps_collected": int(observations.shape[0]),
    }
    out = Path(args.output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        observations=observations,
        target_angles=target_angles,
        target_sincos=np.stack([np.sin(target_angles), np.cos(target_angles)], axis=1).astype(np.float32),
        teacher_actions=teacher_actions,
        episode_ids=episode_ids,
        episode_steps=episode_steps,
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )
    print({"output_path": str(out), "rows": int(observations.shape[0]), "sources": len(sources)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
