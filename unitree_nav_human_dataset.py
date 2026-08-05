"""Immutable, chunked transition datasets for Unitree human interventions."""

from __future__ import annotations

import json
import time
import atexit
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASET_VERSION = 1
CONTROL_SOURCE_NAMES = {0: "policy", 1: "keyboard", 2: "gamepad"}


@dataclass
class HumanDatasetSummary:
    rows: int = 0
    intervention_rows: int = 0
    intervention_starts: int = 0
    intervention_ends: int = 0
    chunks: int = 0


class UnitreeHumanDatasetWriter:
    """Append fixed-shape transitions as compressed NPZ chunks.

    The manifest is updated after each flush so an interrupted collection still
    leaves a valid prefix of the dataset. Raw demonstrations are never edited.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        metadata: dict[str, Any],
        chunk_size: int = 1024,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.parts_dir = self.root / "parts"
        self.manifest_path = self.root / "manifest.json"
        if self.manifest_path.exists():
            raise FileExistsError(f"Refusing to append to an existing raw dataset: {self.manifest_path}")
        self.parts_dir.mkdir(parents=True, exist_ok=False)
        self.chunk_size = max(1, int(chunk_size))
        self.rows: list[dict[str, np.ndarray]] = []
        self.summary = HumanDatasetSummary()
        self._closed = False
        self._manifest = {
            "format": "unitree_nav_human_intervention_npz",
            "version": DATASET_VERSION,
            "created_at_unix_s": time.time(),
            "control_source_names": CONTROL_SOURCE_NAMES,
            "metadata": metadata,
            "parts": [],
            "summary": asdict(self.summary),
        }
        self._write_manifest()
        atexit.register(self.close)

    @staticmethod
    def _as_array(value: Any, dtype) -> np.ndarray:
        return np.asarray(value, dtype=dtype).copy()

    def add(self, **row: Any) -> None:
        normalized = {
            "observations": self._as_array(row["observations"], np.float32),
            "base_observations": self._as_array(row["base_observations"], np.float32),
            "student_actions": self._as_array(row["student_actions"], np.float32),
            "human_actions": self._as_array(row["human_actions"], np.float32),
            "executed_actions": self._as_array(row["executed_actions"], np.float32),
            "next_observations": self._as_array(row["next_observations"], np.float32),
            "next_base_observations": self._as_array(row["next_base_observations"], np.float32),
            "reward": self._as_array(row["reward"], np.float32),
            "cost": self._as_array(row["cost"], np.float32),
            "done": self._as_array(row["done"], np.bool_),
            "terminal_success": self._as_array(row["terminal_success"], np.bool_),
            "intervened": self._as_array(row["intervened"], np.bool_),
            "intervention_start": self._as_array(row["intervention_start"], np.bool_),
            "intervention_end": self._as_array(row["intervention_end"], np.bool_),
            "control_source": self._as_array(row["control_source"], np.int8),
            "gamepad_connected": self._as_array(row["gamepad_connected"], np.bool_),
            "gamepad_stale": self._as_array(row["gamepad_stale"], np.bool_),
            "gamepad_command_norm": self._as_array(row["gamepad_command_norm"], np.float32),
            "action_delta_to_policy": self._as_array(row["action_delta_to_policy"], np.float32),
            "goal_distance": self._as_array(row["goal_distance"], np.float32),
            "next_goal_distance": self._as_array(row["next_goal_distance"], np.float32),
            "episode_index": self._as_array(row["episode_index"], np.int64),
            "step_index": self._as_array(row["step_index"], np.int64),
            "wall_time_unix_s": self._as_array(row["wall_time_unix_s"], np.float64),
        }
        if not all(np.isfinite(value).all() for key, value in normalized.items() if np.issubdtype(value.dtype, np.floating)):
            raise ValueError("Refusing to record a non-finite human intervention transition")
        self.rows.append(normalized)
        self.summary.rows += 1
        self.summary.intervention_rows += int(bool(normalized["intervened"].item()))
        self.summary.intervention_starts += int(bool(normalized["intervention_start"].item()))
        self.summary.intervention_ends += int(bool(normalized["intervention_end"].item()))
        if len(self.rows) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        keys = tuple(self.rows[0])
        if any(tuple(row) != keys for row in self.rows):
            raise RuntimeError("Human dataset row schema changed within one chunk")
        payload = {key: np.stack([row[key] for row in self.rows], axis=0) for key in keys}
        for key, value in payload.items():
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"Refusing to write non-finite values for {key}")
        part_name = f"part_{self.summary.chunks:05d}.npz"
        target = self.parts_dir / part_name
        np.savez_compressed(target, **payload)
        self._manifest["parts"].append({"file": f"parts/{part_name}", "rows": int(len(self.rows))})
        self.summary.chunks += 1
        self.rows.clear()
        self._write_manifest()

    def _write_manifest(self) -> None:
        self._manifest["summary"] = asdict(self.summary)
        temp = self.manifest_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self._manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(self.manifest_path)

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True
