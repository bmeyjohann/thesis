from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


class KNNBCPolicy:
    def __init__(
        self,
        *,
        observations: np.ndarray,
        actions: np.ndarray,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        k: int = 5,
    ):
        self.observations = np.asarray(observations, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32).reshape(-1)
        self.obs_std = np.maximum(np.asarray(obs_std, dtype=np.float32).reshape(-1), 1e-6)
        self.k = int(max(1, k))
        norm_obs = self._normalize(self.observations)
        self.tree = cKDTree(norm_obs)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return (np.asarray(obs, dtype=np.float32) - self.obs_mean) / self.obs_std

    def act(self, obs: np.ndarray) -> np.ndarray:
        q = self._normalize(np.asarray(obs, dtype=np.float32).reshape(1, -1))
        dist, idx = self.tree.query(q, k=min(self.k, int(self.actions.shape[0])))
        idx_arr = np.asarray(idx, dtype=np.int64).reshape(-1)
        act = self.actions[idx_arr]
        # Median preserves saturated maneuver modes better than a mean.
        return np.median(act, axis=0).astype(np.float32, copy=False)


def save_knn_policy(
    path: Path | str,
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    k: int = 5,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    obs = np.asarray(observations, dtype=np.float32)
    act = np.asarray(actions, dtype=np.float32)
    obs_mean = obs.mean(axis=0).astype(np.float32)
    obs_std = np.maximum(obs.std(axis=0).astype(np.float32), 1e-6)
    np.savez_compressed(
        target,
        format=np.asarray("knn_bc", dtype=object),
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
        k=np.asarray(int(k), dtype=np.int64),
        metadata_json=np.asarray(__import__("json").dumps(dict(metadata or {}), sort_keys=True), dtype=object),
    )
    return target


def load_knn_policy(path: Path | str) -> KNNBCPolicy:
    with np.load(Path(path).expanduser(), allow_pickle=True) as data:
        fmt = str(data["format"].item()) if "format" in data.files else ""
        if fmt != "knn_bc":
            raise ValueError(f"Not a knn_bc checkpoint: {path}")
        return KNNBCPolicy(
            observations=np.asarray(data["observations"], dtype=np.float32),
            actions=np.asarray(data["actions"], dtype=np.float32),
            obs_mean=np.asarray(data["obs_mean"], dtype=np.float32),
            obs_std=np.asarray(data["obs_std"], dtype=np.float32),
            k=int(np.asarray(data["k"]).item()) if "k" in data.files else 5,
        )
