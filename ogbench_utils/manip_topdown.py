from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


_DEFAULT_TARGET_BOUNDS = np.asarray([[0.3, -0.3], [0.55, 0.3]], dtype=np.float32)


def _vec3(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.isfinite(arr[:3]).all():
        return None
    return arr[:3].astype(np.float32, copy=False)


def _xy_bounds(env: Any) -> np.ndarray:
    try:
        bounds = np.asarray(getattr(env, "_target_sampling_bounds"), dtype=np.float32)
        if bounds.shape == (2, 2) and np.isfinite(bounds).all():
            return bounds
    except Exception:
        pass
    return _DEFAULT_TARGET_BOUNDS.copy()


def _num_cubes(env: Any, info: Optional[dict[str, Any]]) -> int:
    if isinstance(info, dict):
        try:
            total = int(info.get("diag/cubes_total", 0))
            if total > 0:
                return total
        except Exception:
            pass
        inferred = 0
        for key in info.keys():
            if isinstance(key, str) and key.startswith("privileged/block_") and key.endswith("_pos"):
                try:
                    idx = int(key[len("privileged/block_") : -len("_pos")])
                    inferred = max(inferred, idx + 1)
                except Exception:
                    continue
        if inferred > 0:
            return inferred
    try:
        return max(0, int(getattr(env, "_num_cubes", 0)))
    except Exception:
        return 0


def extract_manip_topdown_state(env: Any, info: Optional[dict[str, Any]]) -> dict[str, Any]:
    base = getattr(env, "unwrapped", env)
    bounds = _xy_bounds(base)
    effector_pos = _vec3(info.get("proprio/effector_pos")) if isinstance(info, dict) else None
    target_block = -1
    if isinstance(info, dict):
        try:
            target_block = int(info.get("privileged/target_block", -1))
        except Exception:
            target_block = -1
    cubes: list[dict[str, Any]] = []
    for idx in range(max(0, _num_cubes(base, info))):
        pos = _vec3(info.get(f"privileged/block_{idx}_pos")) if isinstance(info, dict) else None
        if pos is None:
            continue
        cubes.append(
            {
                "index": idx,
                "xy": [float(pos[0]), float(pos[1])],
                "z": float(pos[2]),
                "is_target": bool(idx == target_block),
            }
        )

    targets: list[dict[str, Any]] = []
    try:
        mocap_ids = list(getattr(base, "_cube_target_mocap_ids", []) or [])
        mocap_pos = np.asarray(getattr(getattr(base, "_data", None), "mocap_pos", None), dtype=np.float32)
        for idx, mocap_id in enumerate(mocap_ids):
            if mocap_pos.ndim != 2 or int(mocap_id) >= mocap_pos.shape[0]:
                continue
            pos = _vec3(mocap_pos[int(mocap_id)])
            if pos is None:
                continue
            targets.append(
                {
                    "index": idx,
                    "xy": [float(pos[0]), float(pos[1])],
                    "z": float(pos[2]),
                    "is_target": bool(idx == target_block),
                }
            )
    except Exception:
        targets = []

    active_target = _vec3(info.get("privileged/target_block_pos")) if isinstance(info, dict) else None
    return {
        "available": bool(effector_pos is not None or cubes or targets),
        "bounds": bounds.tolist(),
        "effector_xy": None if effector_pos is None else [float(effector_pos[0]), float(effector_pos[1])],
        "effector_z": 0.0 if effector_pos is None else float(effector_pos[2]),
        "target_block": int(target_block),
        "cubes": cubes,
        "targets": targets,
        "active_target_xy": None if active_target is None else [float(active_target[0]), float(active_target[1])],
        "active_target_z": 0.0 if active_target is None else float(active_target[2]),
        "source_env": str(getattr(base, "spec", None).id) if getattr(getattr(base, "spec", None), "id", None) else str(type(base).__name__),
    }
