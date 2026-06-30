#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from safetygym_utils.env import (
    extract_agent_forward_xy,
    extract_agent_xy,
    extract_goal_distance,
    extract_goal_xy,
    extract_step_limit,
    _task_constrained_object_specs,
)
from safetygym_utils.io import save_args_json
from safetygym_utils.metrics import EpisodeWindow, augment_rollout_summary
from safetygym_utils.metrics import classify_outcome
from safetygym_utils.minimal_train import _make_env_with_wrappers
from safetygym_utils.policy_viz import (
    _extract_bounds,
    _extract_overlay_specs,
    plot_episode_contact_sheet,
    plot_eval_episode_trajectory,
)
from safetygym_utils.rendering import build_external_viewer, resolve_env_render_mode, wants_external_viewer
from safetygym_utils.controllers import ScriptedGeometricTeacherController
from safetygym_utils.dataset_io import load_transition_dataset


def _device(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, depth: int, layer_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = int(in_dim)
    for _ in range(max(1, int(depth))):
        layers.append(nn.Linear(last, int(hidden_dim)))
        if layer_norm:
            layers.append(nn.LayerNorm(int(hidden_dim)))
        layers.append(nn.SiLU())
        last = int(hidden_dim)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


class FlowChunkPolicy(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        chunk_len: int,
        hidden_dim: int,
        depth: int,
        layer_norm: bool,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.chunk_len = int(chunk_len)
        self.chunk_dim = self.act_dim * self.chunk_len
        self.net = _mlp(
            self.obs_dim + self.chunk_dim + 1,
            self.chunk_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            layer_norm=bool(layer_norm),
        )

    def forward(self, obs: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([obs, xt, t], dim=-1))

    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        *,
        num_steps: int,
        num_samples: int = 1,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]
        obs = obs.repeat_interleave(int(num_samples), dim=0)
        x = torch.randn((obs.shape[0], self.chunk_dim), device=obs.device, dtype=obs.dtype) * float(noise_scale)
        steps = max(1, int(num_steps))
        dt = 1.0 / float(steps)
        for idx in range(steps):
            t = torch.full((obs.shape[0], 1), float(idx) / float(steps), device=obs.device, dtype=obs.dtype)
            x = x + dt * self.forward(obs, x, t)
        x = torch.tanh(x).view(obs.shape[0], self.chunk_len, self.act_dim)
        if single:
            x = x.view(int(num_samples), self.chunk_len, self.act_dim)
        return x


@dataclass
class ChunkDataset:
    obs: np.ndarray
    chunks: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray

    @property
    def size(self) -> int:
        return int(self.obs.shape[0])


def _scale_to_unit(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    denom = np.maximum(high - low, 1e-6)
    return np.clip(2.0 * (np.asarray(action, dtype=np.float32) - low) / denom - 1.0, -1.0, 1.0)


def _scale_from_unit(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(low + 0.5 * (np.asarray(action, dtype=np.float32) + 1.0) * (high - low), low, high)


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _goal_teacher_action(env, args, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    act_dim = int(action_low.shape[0])
    action = np.zeros((act_dim,), dtype=np.float32)
    agent_xy = extract_agent_xy(env)
    goal_xy = extract_goal_xy(env)
    if agent_xy is None or goal_xy is None:
        return action
    delta = np.asarray(goal_xy - agent_xy, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-6:
        return action
    desired = delta / dist
    if str(getattr(args, "teacher_mode", "goal")).strip().lower() == "clearance":
        repel = _clearance_repulsion(env, agent_xy)
        desired = desired + float(getattr(args, "teacher_clearance_scale", 1.5)) * repel
        desired_norm = float(np.linalg.norm(desired))
        if desired_norm > 1e-6:
            desired = desired / desired_norm
    env_name = str(getattr(args, "env_name", "")).lower()
    point_mode = str(getattr(args, "point_action_mode", "native")).strip().lower()
    car_mode = str(getattr(args, "car_action_mode", "raw_wheels")).strip().lower()
    if "point" in env_name and point_mode == "world_velocity":
        action[: min(2, act_dim)] = desired[: min(2, act_dim)]
    else:
        forward = extract_agent_forward_xy(env)
        if "car" in env_name and car_mode == "throttle_turn":
            if forward is None:
                action[0] = 1.0
            else:
                heading = float(math.atan2(float(forward[1]), float(forward[0])))
                target = float(math.atan2(float(desired[1]), float(desired[0])))
                err = _wrap_angle(target - heading)
                action[0] = float(np.clip(math.cos(err), -0.25, 1.0))
                if act_dim >= 2:
                    action[1] = float(np.clip(err / (0.5 * math.pi), -1.0, 1.0))
        elif "point" in env_name:
            # Native Point actions are [forward, turn]; align the body to the
            # goal before moving forward.
            if forward is None:
                action[0] = 1.0
            else:
                heading = float(math.atan2(float(forward[1]), float(forward[0])))
                target = float(math.atan2(float(desired[1]), float(desired[0])))
                err = _wrap_angle(target - heading)
                action[0] = float(np.clip(max(0.0, math.cos(err)), 0.0, 1.0))
                if act_dim >= 2:
                    action[1] = float(np.clip(err / math.pi, -1.0, 1.0))
        elif forward is None:
            action[0] = 1.0
        else:
            heading = float(math.atan2(float(forward[1]), float(forward[0])))
            target = float(math.atan2(float(desired[1]), float(desired[0])))
            err = _wrap_angle(target - heading)
            action[0] = float(np.clip(math.cos(err), -0.25, 1.0))
            if act_dim >= 2:
                action[1] = float(np.clip(err / (0.5 * math.pi), -1.0, 1.0))
    return np.clip(action, action_low, action_high).astype(np.float32, copy=False)


def _clearance_repulsion(env, agent_xy: np.ndarray) -> np.ndarray:
    base = env.unwrapped
    task = getattr(base, "task", None)
    if task is None:
        return np.zeros((2,), dtype=np.float64)
    specs = _task_constrained_object_specs(task)
    agent_keepout = float(getattr(getattr(task, "agent", None), "keepout", 0.0) or 0.0)
    margin = 0.8
    out = np.zeros((2,), dtype=np.float64)
    for spec in specs:
        try:
            pos = np.asarray(task.data.body(str(spec["body_name"])).xpos[:2], dtype=np.float64)
        except Exception:
            continue
        delta = np.asarray(agent_xy, dtype=np.float64) - pos
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-6:
            continue
        clearance = dist - (agent_keepout + float(spec.get("keepout", 0.0)))
        if clearance >= margin:
            continue
        strength = ((margin - clearance) / margin) ** 2
        out += strength * (delta / dist)
    norm = float(np.linalg.norm(out))
    if norm > 1e-6:
        out = out / norm
    return out


def _collect_teacher_chunks(args, *, seed: int) -> ChunkDataset:
    env = _make_env_with_wrappers(args=args, seed=seed, with_intervention=False, controller=None, render_mode_override="none")
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    scripted_geo = None
    if str(getattr(args, "teacher_mode", "goal")).strip().lower() == "scripted_geo":
        scripted_geo = ScriptedGeometricTeacherController(action_low=low, action_high=high)
    max_steps = extract_step_limit(env)
    chunk_len = int(args.chunk_len)
    obs_rows: list[np.ndarray] = []
    chunk_rows: list[np.ndarray] = []
    episodes = 0
    while len(obs_rows) < int(args.dataset_steps):
        obs, _ = env.reset(seed=seed + episodes)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        done = False
        ep_len = 0
        while not done and ep_len < max_steps and len(obs_rows) < int(args.dataset_steps):
            if scripted_geo is not None:
                action = scripted_geo.get_action(obs=obs, env=env)
                if action is None:
                    action = np.zeros_like(low, dtype=np.float32)
                action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), low, high)
            else:
                action = _goal_teacher_action(env, args, low, high)
            next_obs, _reward, _cost, terminated, truncated, _info = env.step(action)
            ep_obs.append(obs.copy())
            ep_actions.append(_scale_to_unit(action, low, high))
            obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            done = bool(terminated or truncated)
            ep_len += 1
        for idx, row_obs in enumerate(ep_obs):
            seq = ep_actions[idx : idx + chunk_len]
            if not seq:
                continue
            while len(seq) < chunk_len:
                seq.append(seq[-1].copy())
            obs_rows.append(row_obs)
            chunk_rows.append(np.stack(seq, axis=0).astype(np.float32))
            if len(obs_rows) >= int(args.dataset_steps):
                break
        episodes += 1
        if episodes == 1 or episodes % int(max(1, args.dataset_log_episodes)) == 0:
            print(
                json.dumps(
                    {
                        "dataset/episodes": float(episodes),
                        "dataset/size": float(len(obs_rows)),
                        "dataset/target_size": float(args.dataset_steps),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    env.close()
    return ChunkDataset(
        obs=np.asarray(obs_rows, dtype=np.float32),
        chunks=np.asarray(chunk_rows, dtype=np.float32),
        action_low=low,
        action_high=high,
    )


def _load_teacher_chunks_from_dataset(args, *, action_low: np.ndarray, action_high: np.ndarray) -> ChunkDataset:
    data = load_transition_dataset(str(args.dataset_path))
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    episode_ids = np.asarray(data["episode_ids"], dtype=np.int64).reshape(-1)
    teacher_intervened = np.asarray(data.get("teacher_intervened"), dtype=np.bool_).reshape(-1)
    if bool(getattr(args, "dataset_teacher_only", True)):
        keep = teacher_intervened
        observations = observations[keep]
        actions = actions[keep]
        episode_ids = episode_ids[keep]
    if int(getattr(args, "dataset_max_rows", 0)) > 0:
        limit = int(args.dataset_max_rows)
        observations = observations[:limit]
        actions = actions[:limit]
        episode_ids = episode_ids[:limit]
    if observations.shape[0] == 0:
        raise ValueError(f"No rows available in dataset: {args.dataset_path}")

    scaled_actions = _scale_to_unit(actions, action_low, action_high)
    chunk_len = int(args.chunk_len)
    obs_rows: list[np.ndarray] = []
    chunk_rows: list[np.ndarray] = []
    start = 0
    n_rows = int(observations.shape[0])
    while start < n_rows:
        ep = int(episode_ids[start])
        stop = start + 1
        while stop < n_rows and int(episode_ids[stop]) == ep:
            stop += 1
        ep_obs = observations[start:stop]
        ep_actions = scaled_actions[start:stop]
        for idx, row_obs in enumerate(ep_obs):
            seq = [np.asarray(x, dtype=np.float32).copy() for x in ep_actions[idx : idx + chunk_len]]
            if not seq:
                continue
            while len(seq) < chunk_len:
                seq.append(seq[-1].copy())
            obs_rows.append(np.asarray(row_obs, dtype=np.float32).copy())
            chunk_rows.append(np.stack(seq, axis=0).astype(np.float32))
        start = stop
    print(
        json.dumps(
            {
                "dataset/path": str(args.dataset_path),
                "dataset/loaded_rows": float(observations.shape[0]),
                "dataset/chunks": float(len(obs_rows)),
                "dataset/episodes": float(len(set(episode_ids.tolist()))),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return ChunkDataset(
        obs=np.asarray(obs_rows, dtype=np.float32),
        chunks=np.asarray(chunk_rows, dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
    )


def _maybe_wandb(args, log_dir: Path):
    if not bool(args.use_wandb):
        return None
    import wandb

    kwargs: dict[str, Any] = {
        "project": str(args.wandb_project),
        "mode": str(args.wandb_mode),
        "config": vars(args),
        "dir": str(log_dir),
    }
    if str(args.wandb_entity).strip():
        kwargs["entity"] = str(args.wandb_entity).strip()
    if str(args.wandb_run_name).strip() or str(args.exp_name).strip():
        kwargs["name"] = str(args.wandb_run_name).strip() or str(args.exp_name).strip()
    if str(args.wandb_group).strip():
        kwargs["group"] = str(args.wandb_group).strip()
    return wandb.init(**kwargs)


def _prepare_dirs(args) -> tuple[Path, Path]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not str(args.exp_name).strip():
        args.exp_name = f"flowmatch_{args.env_name.replace('-', '_')}_{stamp}"
    log_dir = Path("logs") / "safetygym_flow" / args.exp_name
    model_dir = Path("models") / "safetygym_flow" / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_args_json(log_dir / "args.json", vars(args))
    save_args_json(model_dir / "args.json", vars(args))
    return log_dir, model_dir


def _save_checkpoint(path: Path, policy: FlowChunkPolicy, args, dataset: ChunkDataset, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
            "args": vars(args),
            "obs_dim": int(policy.obs_dim),
            "act_dim": int(policy.act_dim),
            "chunk_len": int(policy.chunk_len),
            "action_low": dataset.action_low,
            "action_high": dataset.action_high,
            "global_step": int(step),
        },
        path,
    )


def _load_checkpoint(path: str | Path, device: torch.device) -> tuple[FlowChunkPolicy, dict[str, Any], np.ndarray, np.ndarray]:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    ckpt_args = dict(ckpt.get("args", {}))
    policy = FlowChunkPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        act_dim=int(ckpt["act_dim"]),
        chunk_len=int(ckpt["chunk_len"]),
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        depth=int(ckpt_args.get("depth", 3)),
        layer_norm=bool(ckpt_args.get("layer_norm", False)),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return (
        policy,
        ckpt_args,
        np.asarray(ckpt["action_low"], dtype=np.float32).reshape(-1),
        np.asarray(ckpt["action_high"], dtype=np.float32).reshape(-1),
    )


def _score_goal_alignment(env, actions: np.ndarray) -> np.ndarray:
    agent_xy = extract_agent_xy(env)
    goal_xy = extract_goal_xy(env)
    if agent_xy is None or goal_xy is None:
        return np.zeros((int(actions.shape[0]),), dtype=np.float32)
    delta = np.asarray(goal_xy - agent_xy, dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-6:
        return np.zeros((int(actions.shape[0]),), dtype=np.float32)
    desired = (delta / norm).astype(np.float32)
    scores = np.zeros((int(actions.shape[0]),), dtype=np.float32)
    name = str(getattr(getattr(env, "spec", None), "id", "")).lower()
    if "point" in name:
        scores = np.asarray(actions[:, :2] @ desired[:2], dtype=np.float32)
    else:
        forward = extract_agent_forward_xy(env)
        if forward is None:
            # Fallback for raw wheel-like actions where we cannot infer heading.
            scores = np.asarray(actions[:, 0] - 0.15 * np.abs(actions[:, 1]), dtype=np.float32)
        else:
            heading = float(math.atan2(float(forward[1]), float(forward[0])))
            target = float(math.atan2(float(desired[1]), float(desired[0])))
            err = (target - heading + math.pi) % (2.0 * math.pi) - math.pi
            # In throttle-turn mode the first channel is forward throttle and
            # the second channel turns toward the goal. Reward steering in the
            # correct direction before rewarding throttle; otherwise the chunk
            # selector drives straight even when the car faces away from goal.
            throttle = np.asarray(actions[:, 0], dtype=np.float32)
            turn = np.asarray(actions[:, 1], dtype=np.float32)
            desired_turn = float(np.clip(err / (0.5 * math.pi), -1.0, 1.0))
            alignment_after_turn = 1.0 - np.minimum(1.0, np.abs(turn - desired_turn))
            forward_gate = float(max(0.0, math.cos(err)))
            scores = np.asarray(
                alignment_after_turn + 0.35 * forward_gate * throttle - 0.05 * np.abs(turn),
                dtype=np.float32,
            )
    return scores


def _constrained_object_circles(env) -> list[tuple[np.ndarray, float]]:
    base = env.unwrapped
    task = getattr(base, "task", None)
    if task is None:
        return []
    agent_keepout = float(getattr(getattr(task, "agent", None), "keepout", 0.0) or 0.0)
    out: list[tuple[np.ndarray, float]] = []
    for spec in _task_constrained_object_specs(task):
        try:
            pos = np.asarray(task.data.body(str(spec["body_name"])).xpos[:2], dtype=np.float64)
        except Exception:
            continue
        radius = agent_keepout + float(spec.get("keepout", 0.0))
        out.append((pos, radius))
    return out


def _score_goal_clearance_chunks(env, chunks: np.ndarray, args) -> np.ndarray:
    agent_xy = extract_agent_xy(env)
    goal_xy = extract_goal_xy(env)
    if agent_xy is None or goal_xy is None:
        return _score_goal_alignment(env, chunks[:, 0, :])

    start_xy = np.asarray(agent_xy, dtype=np.float64)
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    start_dist = float(np.linalg.norm(goal_xy - start_xy))
    if start_dist <= 1e-6:
        return np.zeros((int(chunks.shape[0]),), dtype=np.float32)

    forward = extract_agent_forward_xy(env)
    heading = 0.0
    if forward is not None:
        heading = float(math.atan2(float(forward[1]), float(forward[0])))

    env_name = str(getattr(args, "env_name", "")).lower()
    car_mode = str(getattr(args, "car_action_mode", "raw_wheels")).strip().lower()
    point_mode = str(getattr(args, "point_action_mode", "native")).strip().lower()
    step_scale = float(getattr(args, "eval_chunk_rollout_step_scale", 0.08))
    turn_scale = float(getattr(args, "eval_chunk_rollout_turn_scale", 0.35))
    margin = float(getattr(args, "eval_chunk_clearance_margin", 0.25))
    clearance_weight = float(getattr(args, "eval_chunk_clearance_weight", 4.0))
    goal_weight = float(getattr(args, "eval_chunk_goal_weight", 1.0))
    obstacles = _constrained_object_circles(env)

    scores = np.zeros((int(chunks.shape[0]),), dtype=np.float64)
    for i in range(int(chunks.shape[0])):
        pos = start_xy.copy()
        angle = float(heading)
        min_clearance = float("inf")
        path_progress = 0.0
        prev_dist = start_dist

        for action in np.asarray(chunks[i], dtype=np.float64):
            if "point" in env_name and point_mode == "world_velocity":
                delta = np.asarray(action[:2], dtype=np.float64)
                norm = float(np.linalg.norm(delta))
                if norm > 1.0:
                    delta = delta / norm
                pos = pos + step_scale * delta
            else:
                if "car" in env_name and car_mode == "throttle_turn":
                    throttle = float(action[0])
                    turn = float(action[1]) if action.shape[0] > 1 else 0.0
                elif "car" in env_name and action.shape[0] >= 2:
                    left = float(action[0])
                    right = float(action[1])
                    throttle = 0.5 * (left + right)
                    turn = 0.5 * (right - left)
                else:
                    throttle = float(action[0]) if action.shape[0] else 0.0
                    turn = float(action[1]) if action.shape[0] > 1 else 0.0
                angle = _wrap_angle(angle + turn_scale * turn)
                forward_vec = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
                pos = pos + step_scale * throttle * forward_vec

            dist = float(np.linalg.norm(goal_xy - pos))
            path_progress += prev_dist - dist
            prev_dist = dist
            for obstacle_xy, obstacle_radius in obstacles:
                clearance = float(np.linalg.norm(pos - obstacle_xy) - obstacle_radius)
                min_clearance = min(min_clearance, clearance)

        final_dist = float(np.linalg.norm(goal_xy - pos))
        score = goal_weight * (0.7 * (start_dist - final_dist) + 0.3 * path_progress)
        if obstacles and min_clearance < margin:
            score -= clearance_weight * ((margin - min_clearance) / max(margin, 1e-6)) ** 2
        scores[i] = score
    return scores.astype(np.float32)


def _sample_eval_action(policy: FlowChunkPolicy, obs: np.ndarray, device: torch.device, low: np.ndarray, high: np.ndarray, args, env) -> np.ndarray:
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
    chunks = policy.sample(
        obs_t,
        num_steps=int(args.sample_steps),
        num_samples=int(args.eval_num_samples),
        noise_scale=float(args.sample_noise_scale),
    )
    chunk_np = chunks.detach().cpu().numpy()
    first_actions = np.stack([_scale_from_unit(chunk_np[i, 0, :], low, high) for i in range(chunk_np.shape[0])], axis=0)
    mode = str(getattr(args, "eval_chunk_selector", "best_goal")).strip().lower()
    if mode == "mean":
        return np.mean(first_actions, axis=0).astype(np.float32)
    if mode == "first" or first_actions.shape[0] == 1:
        return first_actions[0].astype(np.float32)
    if mode == "best_goal_clearance":
        chunk_actions = np.stack(
            [
                np.stack([_scale_from_unit(chunk_np[i, j, :], low, high) for j in range(chunk_np.shape[1])], axis=0)
                for i in range(chunk_np.shape[0])
            ],
            axis=0,
        )
        scores = _score_goal_clearance_chunks(env, chunk_actions, args)
        best_idx = int(np.argmax(scores))
        return first_actions[best_idx].astype(np.float32)
    if mode != "best_goal":
        raise ValueError(f"Unsupported eval_chunk_selector: {mode}")
    scores = _score_goal_alignment(env, first_actions)
    best_idx = int(np.argmax(scores))
    return first_actions[best_idx].astype(np.float32)


def run_eval(policy: FlowChunkPolicy, args, device: torch.device, low: np.ndarray, high: np.ndarray, *, log_dir: Path | None, step: int) -> dict[str, float]:
    env = _make_env_with_wrappers(
        args=args,
        seed=int(args.seed + 20_000),
        with_intervention=False,
        controller=None,
        render_mode_override=resolve_env_render_mode(str(args.render_mode)) if bool(args.eval_interactive) else "none",
    )
    viewer = None
    if bool(args.eval_interactive) and wants_external_viewer(str(args.render_mode)):
        viewer = build_external_viewer(
            render_mode=str(args.render_mode),
            title=f"SafetyGym Flow Eval {args.env_name}",
            draw_hz=float(args.viewer_fps),
            scale=float(args.viewer_scale),
        )
    max_steps = extract_step_limit(env)
    win = EpisodeWindow(size=max(1, int(args.num_eval_episodes)))
    task = env.unwrapped.task
    overlay_specs = _extract_overlay_specs(task)
    bounds = _extract_bounds(task, x_range=None, y_range=None)
    plot_dir = None
    saved_plots: list[Path] = []
    if bool(args.eval_save_episode_plots) and log_dir is not None:
        plot_dir = log_dir / "eval_episode_plots" / f"step_{int(step)}"

    for ep in range(int(args.num_eval_episodes)):
        obs, _ = env.reset(seed=int(args.seed + 20_000 + ep))
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        done = False
        ep_len = 0
        ep_ret = 0.0
        ep_cost = 0.0
        goal_hits = 0
        first_goal_step: int | None = None
        path = [np.asarray(task.agent.pos[:2], dtype=np.float64).copy()]
        goal_positions = [np.asarray(task.goal.pos[:2], dtype=np.float64).copy()]
        hit_points: list[np.ndarray] = []
        if viewer is not None:
            viewer.draw_env(env)
        while not done and ep_len < max_steps:
            action = _sample_eval_action(policy, obs, device, low, high, args, env)
            next_obs, reward, cost, terminated, truncated, info = env.step(action)
            ep_len += 1
            ep_ret += float(reward)
            ep_cost += float(cost)
            obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            path.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
            if bool(info.get("goal_met", False)):
                goal_hits += 1
                if first_goal_step is None:
                    first_goal_step = int(ep_len)
                hit_points.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
            cur_goal = np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
            if np.linalg.norm(cur_goal - goal_positions[-1]) > 1e-6:
                goal_positions.append(cur_goal)
            done = bool(terminated or truncated)
            if viewer is not None:
                viewer.draw_env(env)
                if float(args.eval_sleep_seconds) > 0.0:
                    time.sleep(float(args.eval_sleep_seconds))
        final_dist = extract_goal_distance(env)
        terminated_f = bool(done and ep_len < max_steps)
        truncated_f = bool(ep_len >= max_steps and first_goal_step is None)
        outcome = classify_outcome(
            goal_met=bool(goal_hits > 0),
            episode_steps=int(ep_len),
            max_episode_steps=int(max_steps),
        )
        win.add(
            {
                "episode_return": float(ep_ret),
                "episode_cost_sum": float(ep_cost),
                "episode_cost_rate": float(ep_cost / max(1, ep_len)),
                "episode_length": float(ep_len),
                "intervention_steps": 0.0,
                "intervention_fraction": 0.0,
                "intervention_num_bursts": 0.0,
                "intervention_avg_burst_len": 0.0,
                "goal_met": 1.0 if goal_hits > 0 else 0.0,
                "goal_met_count": float(goal_hits),
                "first_goal_success": 1.0 if first_goal_step is not None else 0.0,
                "first_goal_hit_step": float(first_goal_step if first_goal_step is not None else max_steps),
                "first_goal_hit_step_success_only": float(first_goal_step if first_goal_step is not None else 0),
                "first_goal_within_100": 1.0 if first_goal_step is not None and first_goal_step <= 100 else 0.0,
                "first_goal_within_200": 1.0 if first_goal_step is not None and first_goal_step <= 200 else 0.0,
                "first_goal_reward_sum": float(ep_ret),
                "first_goal_dense_reward_sum": float(ep_ret),
                "final_distance_to_goal": float(final_dist),
                "outcome_success": 1.0 if outcome == "success" else 0.0,
                "outcome_timeout": 1.0 if outcome == "timeout" else 0.0,
                "outcome_kill": 1.0 if outcome == "kill" else 0.0,
                "outcome_other_failure": 0.0,
                "terminated": 1.0 if terminated_f else 0.0,
                "truncated": 1.0 if truncated_f else 0.0,
            }
        )
        if plot_dir is not None and len(saved_plots) < int(args.eval_episode_plot_max_episodes):
            saved_plots.append(
                Path(
                    plot_eval_episode_trajectory(
                        output_path=plot_dir / f"episode_{ep + 1:03d}.png",
                        task=task,
                        overlay_specs=overlay_specs,
                        bounds=bounds,
                        path=np.asarray(path, dtype=np.float64),
                        goal_positions=np.asarray(goal_positions, dtype=np.float64),
                        goal_hit_points=np.asarray(hit_points, dtype=np.float64) if hit_points else np.zeros((0, 2), dtype=np.float64),
                        episode_idx=ep + 1,
                        total_episodes=int(args.num_eval_episodes),
                        episode_reward=float(ep_ret),
                        goals_reached=int(goal_hits),
                        final_distance=float(final_dist),
                    )
                )
            )
    if saved_plots:
        sheet = plot_episode_contact_sheet(
            image_paths=saved_plots,
            output_path=plot_dir / "episode_contact_sheet.png",
            title=f"SafetyGym flow matching eval step {int(step)}",
            max_cols=3,
        )
        if sheet is not None:
            print(f"[EvalPlots] saved {sheet}", flush=True)
    if viewer is not None:
        viewer.close()
    env.close()
    return augment_rollout_summary(win.summary("eval"), "eval")


def train(args) -> None:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = _device(args.device)
    log_dir, model_dir = _prepare_dirs(args)
    wandb_run = _maybe_wandb(args, log_dir)
    if str(getattr(args, "dataset_path", "")).strip():
        env = _make_env_with_wrappers(args=args, seed=int(args.seed), with_intervention=False, controller=None, render_mode_override="none")
        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        env.close()
        dataset = _load_teacher_chunks_from_dataset(args, action_low=low, action_high=high)
    else:
        dataset = _collect_teacher_chunks(args, seed=int(args.seed))
    policy = FlowChunkPolicy(
        obs_dim=int(dataset.obs.shape[1]),
        act_dim=int(dataset.action_low.shape[0]),
        chunk_len=int(args.chunk_len),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        layer_norm=bool(args.layer_norm),
    ).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    obs = torch.as_tensor(dataset.obs, device=device, dtype=torch.float32)
    chunks = torch.as_tensor(dataset.chunks.reshape(dataset.size, -1), device=device, dtype=torch.float32)
    best_success = -1.0
    last_ckpt = model_dir / "final.pt"
    for step in range(1, int(args.train_steps) + 1):
        idx = torch.randint(0, dataset.size, (int(args.batch_size),), device=device)
        x1 = chunks[idx]
        x0 = torch.randn_like(x1) * float(args.train_noise_scale)
        t = torch.rand((x1.shape[0], 1), device=device)
        xt = (1.0 - t) * x0 + t * x1
        target = x1 - x0
        pred = policy(obs[idx], xt, t)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.max_grad_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(args.max_grad_norm))
        opt.step()
        if step == 1 or step % int(args.log_interval) == 0:
            logs = {"train/loss": float(loss.detach().cpu()), "train/step": float(step), "dataset/size": float(dataset.size)}
            print(json.dumps(logs, sort_keys=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(logs, step=step)
        if int(args.eval_interval) > 0 and (step == 1 or step % int(args.eval_interval) == 0):
            policy.eval()
            eval_logs = run_eval(policy, args, device, dataset.action_low, dataset.action_high, log_dir=log_dir, step=step)
            eval_logs["eval/step"] = float(step)
            print(json.dumps(eval_logs, sort_keys=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(eval_logs, step=step)
            success = float(eval_logs.get("eval/first_goal_success_rate", 0.0))
            if success >= best_success:
                best_success = success
                _save_checkpoint(model_dir / "best.pt", policy, args, dataset, step)
            policy.train()
        if int(args.save_interval) > 0 and step % int(args.save_interval) == 0:
            last_ckpt = model_dir / f"step_{step}.pt"
            _save_checkpoint(last_ckpt, policy, args, dataset, step)
    _save_checkpoint(model_dir / "final.pt", policy, args, dataset, int(args.train_steps))
    policy.eval()
    final_logs = run_eval(policy, args, device, dataset.action_low, dataset.action_high, log_dir=log_dir, step=int(args.train_steps))
    final_logs["eval/step"] = float(args.train_steps)
    print(json.dumps({"checkpoint": str(model_dir / "final.pt"), "best_checkpoint": str(model_dir / "best.pt"), **final_logs}, sort_keys=True), flush=True)
    if wandb_run is not None:
        wandb_run.log(final_logs, step=int(args.train_steps))
        wandb_run.finish()


def eval_only(args) -> None:
    device = _device(args.device)
    policy, ckpt_args, low, high = _load_checkpoint(args.checkpoint_path, device)
    eval_overrides = {
        "checkpoint_path",
        "eval_only",
        "eval_output_dir",
        "device",
        "render_mode",
        "viewer_fps",
        "viewer_scale",
        "eval_sleep_seconds",
        "num_eval_episodes",
        "eval_save_episode_plots",
        "eval_episode_plot_max_episodes",
        "eval_interactive",
        "eval_num_samples",
        "sample_steps",
        "sample_noise_scale",
        "eval_chunk_selector",
        "eval_chunk_goal_weight",
        "eval_chunk_clearance_weight",
        "eval_chunk_clearance_margin",
        "eval_chunk_rollout_step_scale",
        "eval_chunk_rollout_turn_scale",
    }
    for key, value in ckpt_args.items():
        if not hasattr(args, key) or key in eval_overrides:
            continue
        setattr(args, key, value)
    logs = run_eval(policy, args, device, low, high, log_dir=Path(args.eval_output_dir) if args.eval_output_dir else None, step=int(ckpt_args.get("global_step", 0)))
    print(json.dumps(logs, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone flow-matching action-chunk training for Safety-Gymnasium")
    p.add_argument("--env_name", type=str, default="SafetyPointGoal1-v0")
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dataset_steps", type=int, default=20_000)
    p.add_argument("--dataset_path", type=str, default="")
    p.add_argument("--dataset_max_rows", type=int, default=0)
    p.add_argument("--dataset_teacher_only", action="store_true", default=True)
    p.add_argument("--no_dataset_teacher_only", dest="dataset_teacher_only", action="store_false")
    p.add_argument("--dataset_log_episodes", type=int, default=10)
    p.add_argument("--teacher_mode", type=str, default="goal", choices=["goal", "clearance", "scripted_geo"])
    p.add_argument("--teacher_clearance_scale", type=float, default=1.5)
    p.add_argument("--train_steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--chunk_len", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--layer_norm", action="store_true", default=False)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--train_noise_scale", type=float, default=1.0)
    p.add_argument("--sample_noise_scale", type=float, default=1.0)
    p.add_argument("--sample_steps", type=int, default=8)
    p.add_argument("--eval_num_samples", type=int, default=8)
    p.add_argument("--eval_chunk_selector", type=str, default="best_goal", choices=["best_goal", "best_goal_clearance", "mean", "first"])
    p.add_argument("--eval_chunk_goal_weight", type=float, default=1.0)
    p.add_argument("--eval_chunk_clearance_weight", type=float, default=4.0)
    p.add_argument("--eval_chunk_clearance_margin", type=float, default=0.25)
    p.add_argument("--eval_chunk_rollout_step_scale", type=float, default=0.08)
    p.add_argument("--eval_chunk_rollout_turn_scale", type=float, default=0.35)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--eval_interval", type=int, default=2_000)
    p.add_argument("--save_interval", type=int, default=10_000)

    p.add_argument("--reward_mode", type=str, default="dense_plus_sparse", choices=["sparse", "dense", "dense_plus_sparse", "potential_diff", "dual", "native", "none"])
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--success_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--cost_penalty", type=float, default=0.0)
    p.add_argument("--clearance_penalty_scale", type=float, default=0.0)
    p.add_argument("--clearance_margin", type=float, default=0.0)
    p.add_argument("--clearance_penalty_power", type=float, default=1.0)
    p.add_argument("--clearance_penalty_mode", type=str, default="hinge_power", choices=["hinge_power", "softplus"])
    p.add_argument("--clearance_penalty_temperature", type=float, default=0.08)
    p.add_argument("--forward_reward_scale", type=float, default=0.0)
    p.add_argument("--backward_penalty_scale", type=float, default=0.0)
    p.add_argument("--heading_reward_scale", type=float, default=0.0)
    p.add_argument("--heading_positive_only", action="store_true", default=True)
    p.add_argument("--no_heading_positive_only", dest="heading_positive_only", action="store_false")
    p.add_argument("--adaptive_safety_curriculum", action="store_true", default=False)
    p.add_argument("--adaptive_safety_goal_target", type=float, default=1.0)
    p.add_argument("--adaptive_safety_window_episodes", type=int, default=10)
    p.add_argument("--adaptive_safety_step", type=float, default=0.05)
    p.add_argument("--adaptive_safety_init", type=float, default=0.0)
    p.add_argument("--adaptive_safety_min", type=float, default=0.0)
    p.add_argument("--adaptive_safety_max", type=float, default=1.0)
    p.add_argument("--cost_penalty_warmup_steps", type=int, default=0)
    p.add_argument("--cost_penalty_ramp_steps", type=int, default=0)
    p.add_argument("--clearance_penalty_warmup_steps", type=int, default=0)
    p.add_argument("--clearance_penalty_ramp_steps", type=int, default=0)

    p.add_argument("--render_mode", type=str, default="none", choices=["human", "rgb_array", "none", "pygame", "topdown"])
    p.add_argument("--viewer_fps", type=float, default=20.0)
    p.add_argument("--viewer_scale", type=float, default=1.0)
    p.add_argument("--eval_sleep_seconds", type=float, default=0.0)
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=2.0)
    p.add_argument("--car_force_scale", type=float, default=2.0)
    p.add_argument("--car_action_mode", type=str, default="throttle_turn", choices=["raw_wheels", "throttle_turn", "cardinal"])
    p.add_argument("--point_action_mode", type=str, default="world_velocity", choices=["native", "world_velocity"])
    p.add_argument("--point_turn_gain", type=float, default=2.5)
    p.add_argument("--point_alignment_power", type=float, default=1.0)
    p.add_argument("--point_allow_backward", action="store_true", default=False)
    p.add_argument(
        "--obs_mask_mode",
        type=str,
        default="goal_only_lidar",
        choices=["none", "goal_only_lidar", "privileged_geometry", "privileged_geometry_rich"],
    )
    p.add_argument("--max_episode_steps", type=int, default=250)
    p.add_argument("--terminate_on_goal", action="store_true", default=True)
    p.add_argument("--no_terminate_on_goal", dest="terminate_on_goal", action="store_false")
    p.add_argument("--terminate_on_cost", action="store_true", default=False)
    p.add_argument("--fixed_layout_preset", type=str, default="none")
    p.add_argument("--layout_curriculum", type=str, default="none")
    p.add_argument("--layout_curriculum_level", type=int, default=0)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--eval_save_episode_plots", action="store_true", default=False)
    p.add_argument("--eval_episode_plot_max_episodes", type=int, default=9)
    p.add_argument("--eval_interactive", action="store_true", default=False)

    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="thesis-safetygym")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")

    p.add_argument("--eval_only", action="store_true", default=False)
    p.add_argument("--checkpoint_path", type=str, default="")
    p.add_argument("--eval_output_dir", type=str, default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.eval_only):
        if not str(args.checkpoint_path).strip():
            raise SystemExit("--eval_only requires --checkpoint_path")
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
