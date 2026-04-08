from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict

DEFAULT_MANIP_DATASET_DIR = Path.home() / ".config" / "thesis" / "datasets" / "ogbench_manip"


def _slug(text: str) -> str:
    out = []
    for ch in str(text or ""):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "dataset"


def build_dataset_path(
    *,
    env_name: str,
    dataset_dir: Path | str = DEFAULT_MANIP_DATASET_DIR,
    label: str = "human_vr_demo",
    stamp: Optional[str] = None,
) -> Path:
    root = Path(dataset_dir).expanduser()
    stamp_text = stamp or time.strftime("%Y%m%d_%H%M%S")
    return root / f"{_slug(env_name)}__{_slug(label)}__{stamp_text}.npz"


def save_transition_dataset(
    *,
    path: Path | str,
    metadata: dict[str, Any],
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    truncations: np.ndarray,
    student_actions: Optional[np.ndarray] = None,
    teacher_intervened: Optional[np.ndarray] = None,
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True), dtype=object),
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1),
        "dones": np.asarray(dones, dtype=np.bool_).reshape(-1),
        "truncations": np.asarray(truncations, dtype=np.bool_).reshape(-1),
    }
    if student_actions is not None:
        payload["student_actions"] = np.asarray(student_actions, dtype=np.float32)
    if teacher_intervened is not None:
        payload["teacher_intervened"] = np.asarray(teacher_intervened, dtype=np.bool_).reshape(-1)
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.stem}_",
        suffix=target.suffix,
        delete=False,
    ) as tmp_npz:
        tmp_npz_path = Path(tmp_npz.name)
    try:
        np.savez_compressed(tmp_npz_path, **payload)
        os.replace(tmp_npz_path, target)
    finally:
        tmp_npz_path.unlink(missing_ok=True)

    sidecar = target.with_suffix(".json")
    with tempfile.NamedTemporaryFile(
        dir=sidecar.parent,
        prefix=f".{sidecar.stem}_",
        suffix=sidecar.suffix,
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as tmp_json:
        tmp_json.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        tmp_json_path = Path(tmp_json.name)
    try:
        os.replace(tmp_json_path, sidecar)
    finally:
        tmp_json_path.unlink(missing_ok=True)
    return target


def load_transition_dataset_metadata(path: Path | str) -> dict[str, Any]:
    source = Path(path).expanduser()
    sidecar = source.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            pass
    with np.load(source, allow_pickle=True) as data:
        raw = data.get("metadata_json", None)
        if raw is None:
            return {}
        try:
            return json.loads(str(raw.item()))
        except Exception:
            return {}


def find_latest_transition_dataset(
    *,
    env_name: str,
    dataset_dir: Path | str = DEFAULT_MANIP_DATASET_DIR,
) -> Optional[Path]:
    root = Path(dataset_dir).expanduser()
    if not root.exists():
        return None
    prefix = f"{_slug(env_name)}__"
    candidates = sorted(root.glob(f"{prefix}*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def extend_buffer_from_dataset(
    *,
    buffer,
    dataset_path: Path | str,
    device: torch.device,
    max_rows: int = 0,
    expected_obs_dim: Optional[int] = None,
    expected_act_dim: Optional[int] = None,
) -> dict[str, Any]:
    path = Path(dataset_path).expanduser()
    with np.load(path, allow_pickle=True) as data:
        obs = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        next_obs = np.asarray(data["next_observations"], dtype=np.float32)
        rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1)
        dones = np.asarray(data["dones"], dtype=np.bool_).reshape(-1)
        truncations = np.asarray(data["truncations"], dtype=np.bool_).reshape(-1)
        student_actions = (
            np.asarray(data["student_actions"], dtype=np.float32)
            if "student_actions" in data.files
            else actions
        )
        teacher_intervened = (
            np.asarray(data["teacher_intervened"], dtype=np.bool_).reshape(-1)
            if "teacher_intervened" in data.files
            else np.ones((obs.shape[0],), dtype=np.bool_)
        )

    if obs.ndim != 2 or next_obs.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"Dataset {path} must contain 2D observation/action arrays.")
    if expected_obs_dim is not None and int(obs.shape[1]) != int(expected_obs_dim):
        raise ValueError(f"Dataset obs_dim={obs.shape[1]} does not match expected obs_dim={expected_obs_dim}.")
    if expected_act_dim is not None and int(actions.shape[1]) != int(expected_act_dim):
        raise ValueError(f"Dataset act_dim={actions.shape[1]} does not match expected act_dim={expected_act_dim}.")

    limit = int(max_rows) if int(max_rows) > 0 else int(obs.shape[0])
    committed = 0
    for idx in range(min(limit, obs.shape[0])):
        td = TensorDict(
            {
                "observations": torch.as_tensor(obs[idx : idx + 1], device=device, dtype=torch.float32),
                "actions": torch.as_tensor(actions[idx : idx + 1], device=device, dtype=torch.float32),
                "student_actions": torch.as_tensor(student_actions[idx : idx + 1], device=device, dtype=torch.float32),
                "teacher_intervened": torch.as_tensor(teacher_intervened[idx : idx + 1], device=device, dtype=torch.bool),
                "next": {
                    "observations": torch.as_tensor(next_obs[idx : idx + 1], device=device, dtype=torch.float32),
                    "rewards": torch.as_tensor(rewards[idx : idx + 1], device=device, dtype=torch.float32),
                    "dones": torch.as_tensor(dones[idx : idx + 1], device=device, dtype=torch.bool),
                    "truncations": torch.as_tensor(truncations[idx : idx + 1], device=device, dtype=torch.bool),
                },
            },
            batch_size=(1,),
            device=device,
        )
        buffer.extend(td)
        committed += 1

    metadata = load_transition_dataset_metadata(path)
    return {
        "path": str(path),
        "rows_loaded": int(committed),
        "metadata": metadata,
    }


def _buffer_obs_to_numpy(buffer, env_idx: int, indices: torch.Tensor) -> np.ndarray:
    obs = buffer.observations[env_idx, indices]
    if getattr(buffer, "obs_is_pixel", False):
        obs = obs.to(torch.float32).div_(255.0).view(obs.shape[0], -1)
    else:
        obs = obs.to(torch.float32)
    return obs.detach().cpu().numpy().astype(np.float32, copy=False)


def save_buffer_as_transition_dataset(
    *,
    buffer,
    path: Path | str,
    metadata: dict[str, Any],
    max_rows: int = 0,
    filter_mode: str = "all",
) -> dict[str, Any]:
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    next_obs_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    done_rows: list[np.ndarray] = []
    trunc_rows: list[np.ndarray] = []
    student_action_rows: list[np.ndarray] = []
    teacher_intervened_rows: list[np.ndarray] = []

    rows_remaining = int(max_rows) if int(max_rows) > 0 else 0

    for env_idx in range(int(getattr(buffer, "n_env", 1))):
        cap = int(buffer.env_capacities[env_idx])
        if cap <= 1 or int(buffer.filled[env_idx].item()) <= 0:
            continue

        valid_mask = buffer.transition_ready[env_idx, :cap] & buffer.valid_next_mask[env_idx, :cap]
        if filter_mode == "intervened":
            valid_mask = valid_mask & buffer.teacher_intervened[env_idx, :cap]
        elif filter_mode == "non_intervened":
            valid_mask = valid_mask & (~buffer.teacher_intervened[env_idx, :cap])
        indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if indices.numel() <= 0:
            continue
        if rows_remaining > 0:
            indices = indices[:rows_remaining]
        if indices.numel() <= 0:
            break
        next_indices = (indices + 1) % cap

        obs_rows.append(_buffer_obs_to_numpy(buffer, env_idx, indices))
        next_obs_rows.append(_buffer_obs_to_numpy(buffer, env_idx, next_indices))
        action_rows.append(buffer.actions[env_idx, indices].detach().cpu().numpy().astype(np.float32, copy=False))
        student_action_rows.append(
            buffer.student_actions[env_idx, indices].detach().cpu().numpy().astype(np.float32, copy=False)
        )
        teacher_intervened_rows.append(
            buffer.teacher_intervened[env_idx, indices].detach().cpu().numpy().astype(np.bool_, copy=False)
        )
        reward_rows.append(buffer.rewards[env_idx, indices].detach().cpu().numpy().astype(np.float32, copy=False))
        done_rows.append(buffer.dones[env_idx, indices].detach().cpu().numpy().astype(np.bool_, copy=False))
        trunc_rows.append(buffer.truncations[env_idx, indices].detach().cpu().numpy().astype(np.bool_, copy=False))

        if rows_remaining > 0:
            rows_remaining -= int(indices.numel())
            if rows_remaining <= 0:
                break

    if not obs_rows:
        raise ValueError("Replay buffer does not contain any exportable transitions.")

    observations = np.concatenate(obs_rows, axis=0)
    actions = np.concatenate(action_rows, axis=0)
    next_observations = np.concatenate(next_obs_rows, axis=0)
    rewards = np.concatenate(reward_rows, axis=0).reshape(-1)
    dones = np.concatenate(done_rows, axis=0).reshape(-1)
    truncations = np.concatenate(trunc_rows, axis=0).reshape(-1)
    student_actions = np.concatenate(student_action_rows, axis=0)
    teacher_intervened = np.concatenate(teacher_intervened_rows, axis=0).reshape(-1)

    target = save_transition_dataset(
        path=path,
        metadata=metadata,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
        truncations=truncations,
        student_actions=student_actions,
        teacher_intervened=teacher_intervened,
    )
    return {
        "path": str(target),
        "rows_saved": int(observations.shape[0]),
        "metadata": metadata,
    }
