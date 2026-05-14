#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FASTTD3_ROOT = REPO_ROOT / "fasttd3"
if str(FASTTD3_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTTD3_ROOT))

from scripts.recover_offline_actor_manip_from_checkpoint import (  # noqa: E402
    _auto_device,
    _build_models,
    _coerce_args_dict,
    _namespace_from_checkpoint_args,
)
from ogbench_utils.fastsac_ogbench_manip_env import build_manip_environment  # noqa: E402
from ogbench_utils.obs import prepare_observation  # noqa: E402


@dataclass
class ActorBundle:
    name: str
    kind: str
    actor_backbone: torch.nn.Module | None
    actor_head: torch.nn.Module | None
    obs_normalizer: Any | None
    act_dim: int

    @torch.no_grad()
    def action(self, obs_raw: Any, device: torch.device) -> np.ndarray:
        if self.kind == "random":
            return np.random.uniform(-1.0, 1.0, size=(self.act_dim,)).astype(np.float32)
        if self.kind == "oracle":
            return np.zeros((self.act_dim,), dtype=np.float32)
        if self.actor_backbone is None or self.actor_head is None:
            raise RuntimeError(f"Actor bundle {self.name} is missing actor modules")
        obs_tensor = prepare_observation(
            obs_raw,
            device=device,
            obs_mode="state",
            pixel_shape=None,
            flatten=True,
        )
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if self.obs_normalizer is not None:
            obs_tensor = self.obs_normalizer(obs_tensor)
        _, _, mean = self.actor_head(self.actor_backbone(obs_tensor))
        return mean[0].detach().cpu().numpy().astype(np.float32)


@dataclass
class CriticBundle:
    name: str
    critic_backbone: torch.nn.Module
    critic_heads: torch.nn.Module
    obs_normalizer: Any | None

    @torch.no_grad()
    def score(self, obs_raw: Any, action_np: np.ndarray, device: torch.device) -> dict[str, float]:
        obs_tensor = prepare_observation(
            obs_raw,
            device=device,
            obs_mode="state",
            pixel_shape=None,
            flatten=True,
        )
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if self.obs_normalizer is not None:
            obs_tensor = self.obs_normalizer(obs_tensor)
        action_tensor = torch.as_tensor(action_np, dtype=torch.float32, device=device).view(1, -1)
        features = self.critic_backbone(obs_tensor)
        q_stack = torch.stack(self.critic_heads(features, action_tensor), dim=0).squeeze(-1)
        q_min = float(q_stack.min(dim=0).values.mean().item())
        q_mean = float(q_stack.mean().item())
        q_max = float(q_stack.max(dim=0).values.mean().item())
        return {
            "q_min_mean": q_min,
            "q_mean_mean": q_mean,
            "q_max_mean": q_max,
        }


def _load_actor_bundle(checkpoint_path: Path, *, device: torch.device, env_obs_dim: int, env_act_dim: int, name: str) -> ActorBundle:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = _coerce_args_dict(checkpoint.get("args"))
    actor_backbone, actor_head, _, _, obs_normalizer = _build_models(
        ckpt_args=ckpt_args,
        checkpoint=checkpoint,
        obs_dim=env_obs_dim,
        act_dim=env_act_dim,
        device=device,
    )
    actor_backbone.eval()
    actor_head.eval()
    return ActorBundle(
        name=name,
        kind="learned",
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        obs_normalizer=obs_normalizer,
        act_dim=env_act_dim,
    )


def _load_critic_bundle(checkpoint_path: Path, *, device: torch.device, env_obs_dim: int, env_act_dim: int, name: str) -> CriticBundle:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = _coerce_args_dict(checkpoint.get("args"))
    _, _, critic_backbone, critic_heads, obs_normalizer = _build_models(
        ckpt_args=ckpt_args,
        checkpoint=checkpoint,
        obs_dim=env_obs_dim,
        act_dim=env_act_dim,
        device=device,
    )
    critic_backbone.eval()
    critic_heads.eval()
    return CriticBundle(
        name=name,
        critic_backbone=critic_backbone,
        critic_heads=critic_heads,
        obs_normalizer=obs_normalizer,
    )


def _record_progress(_: str) -> None:
    return None


def _base_args_from_checkpoint(checkpoint_path: Path) -> SimpleNamespace:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_dict = _coerce_args_dict(checkpoint.get("args"))
    args_ns = _namespace_from_checkpoint_args(args_dict)
    args_ns.num_envs = 1
    args_ns.eval_num_envs = 1
    args_ns.train_render_mode = "none"
    args_ns.eval_render_mode = "none"
    args_ns.visualize_intervention_colors = False
    args_ns.use_intervention = False
    args_ns.intervention_mode = "none"
    args_ns.intervention_episode_prob = 0.0
    args_ns.intervention_episode_prob_min = 0.0
    args_ns.intervention_episode_prob_decay_steps = 0
    args_ns.intervention_enable_after_steps = 0
    args_ns.static_reset_seed = None
    return args_ns


def _extract_done(done_value: Any) -> bool:
    if torch.is_tensor(done_value):
        if done_value.numel() == 1:
            return bool(done_value.item())
        return bool(done_value.reshape(-1)[0].item())
    arr = np.asarray(done_value).reshape(-1)
    if arr.size <= 0:
        return False
    return bool(arr[0])


def _extract_reward(reward_value: Any) -> float:
    if torch.is_tensor(reward_value):
        if reward_value.numel() == 1:
            return float(reward_value.item())
        return float(reward_value.reshape(-1)[0].item())
    arr = np.asarray(reward_value, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return 0.0
    return float(arr[0])


def _extract_success(infos: dict[str, Any]) -> float:
    for key in ("goals_reached", "goal_reached"):
        value = infos.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            flat = value.detach().cpu().reshape(-1)
            if flat.numel() > 0:
                return float(flat[0].item())
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return float(arr[0])
    log_dict = infos.get("log")
    if isinstance(log_dict, dict):
        value = log_dict.get("/Episode/goal_success_rate")
        if value is not None:
            if torch.is_tensor(value):
                flat = value.detach().cpu().reshape(-1)
                if flat.numel() > 0:
                    return float(flat[0].item())
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                return float(arr[0])
    return 0.0


def _build_rollout_env(args_ns: SimpleNamespace, *, device: torch.device, seed: int, oracle: bool):
    env_args = deepcopy(args_ns)
    env_args.num_envs = 1
    env_args.static_reset_seed = int(seed)
    env_args.visualize_intervention_colors = False
    env_args.train_render_mode = "none"
    if oracle:
        env_args.use_intervention = True
        env_args.intervention_mode = "agent"
        env_args.intervention_agent_mode = "always"
        env_args.intervention_episode_prob = 1.0
        env_args.intervention_episode_prob_min = 1.0
        env_args.intervention_episode_prob_decay_steps = 0
    else:
        env_args.use_intervention = False
        env_args.intervention_mode = "none"
        env_args.intervention_episode_prob = 0.0
        env_args.intervention_episode_prob_min = 0.0
        env_args.intervention_episode_prob_decay_steps = 0
    envs, _, _, _, obs_dim, act_dim, initial_obs_raw = build_manip_environment(env_args, device, _record_progress)
    return envs, initial_obs_raw, obs_dim, act_dim


def _rollout_actor(
    *,
    actor: ActorBundle,
    base_args: SimpleNamespace,
    critics: list[CriticBundle],
    device: torch.device,
    seeds: list[int],
) -> dict[str, Any]:
    critic_acc = {
        critic.name: {
            "q_min_sum": 0.0,
            "q_mean_sum": 0.0,
            "q_max_sum": 0.0,
            "num_steps": 0,
        }
        for critic in critics
    }
    episode_summaries: list[dict[str, Any]] = []

    for seed in seeds:
        envs, obs_raw, _, act_dim = _build_rollout_env(base_args, device=device, seed=seed, oracle=(actor.kind == "oracle"))
        try:
            ep_return = 0.0
            ep_len = 0
            ep_success = 0.0
            done = False
            while not done:
                action_np = actor.action(obs_raw, device=device)
                action_tensor = torch.as_tensor(action_np, dtype=torch.float32, device=device).view(1, act_dim)
                obs_before = obs_raw
                next_obs_raw, rewards, dones, infos = envs.step(action_tensor)
                applied_actions = infos.get("applied_actions")
                if applied_actions is None:
                    applied_np = action_np
                else:
                    if torch.is_tensor(applied_actions):
                        applied_np = applied_actions.detach().cpu().numpy().reshape(1, -1)[0].astype(np.float32)
                    else:
                        applied_np = np.asarray(applied_actions, dtype=np.float32).reshape(1, -1)[0]

                for critic in critics:
                    scored = critic.score(obs_before, applied_np, device=device)
                    critic_acc[critic.name]["q_min_sum"] += float(scored["q_min_mean"])
                    critic_acc[critic.name]["q_mean_sum"] += float(scored["q_mean_mean"])
                    critic_acc[critic.name]["q_max_sum"] += float(scored["q_max_mean"])
                    critic_acc[critic.name]["num_steps"] += 1

                ep_return += _extract_reward(rewards)
                ep_len += 1
                done = _extract_done(dones)
                if done:
                    ep_success = _extract_success(infos)
                obs_raw = next_obs_raw

            episode_summaries.append(
                {
                    "seed": int(seed),
                    "return": float(ep_return),
                    "length": int(ep_len),
                    "success": float(ep_success),
                }
            )
        finally:
            try:
                envs.close()
            except Exception:
                pass

    out: dict[str, Any] = {
        "actor": actor.name,
        "kind": actor.kind,
        "episodes": episode_summaries,
        "avg_return": float(np.mean([ep["return"] for ep in episode_summaries])) if episode_summaries else 0.0,
        "avg_length": float(np.mean([ep["length"] for ep in episode_summaries])) if episode_summaries else 0.0,
        "success_rate": float(np.mean([ep["success"] for ep in episode_summaries])) if episode_summaries else 0.0,
        "critics": {},
    }
    for critic in critics:
        acc = critic_acc[critic.name]
        denom = max(1, int(acc["num_steps"]))
        out["critics"][critic.name] = {
            "q_min_mean": float(acc["q_min_sum"]) / float(denom),
            "q_mean_mean": float(acc["q_mean_sum"]) / float(denom),
            "q_max_mean": float(acc["q_max_sum"]) / float(denom),
            "num_scored_steps": int(acc["num_steps"]),
        }
    return out


def _format_table(results: list[dict[str, Any]], critic_names: list[str]) -> str:
    headers = ["actor", "success", "avg_len", "avg_ret"]
    for critic_name in critic_names:
        headers.extend([f"{critic_name}:qmin", f"{critic_name}:qmean"])
    rows = [headers]
    for result in results:
        row = [
            result["actor"],
            f"{result['success_rate']:.3f}",
            f"{result['avg_length']:.1f}",
            f"{result['avg_return']:.3f}",
        ]
        for critic_name in critic_names:
            crit = result["critics"][critic_name]
            row.extend([f"{crit['q_min_mean']:.4f}", f"{crit['q_mean_mean']:.4f}"])
        rows.append(row)
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-score manipulation actors against scripted/human critics.")
    p.add_argument("--scripted_ckpt", type=str, required=True)
    p.add_argument("--human_ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--output_json",
        type=str,
        default=str(REPO_ROOT / "local" / "reports" / "manip_actor_critic_crossplay.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _auto_device(args.device)
    scripted_ckpt = Path(args.scripted_ckpt).expanduser().resolve()
    human_ckpt = Path(args.human_ckpt).expanduser().resolve()
    base_args = _base_args_from_checkpoint(scripted_ckpt)

    # Probe env dims once on the shared task configuration.
    probe_envs, initial_obs_raw, obs_dim, act_dim = _build_rollout_env(base_args, device=device, seed=int(args.seed_start), oracle=False)
    try:
        pass
    finally:
        probe_envs.close()

    critics = [
        _load_critic_bundle(scripted_ckpt, device=device, env_obs_dim=obs_dim, env_act_dim=act_dim, name="scripted_critic"),
        _load_critic_bundle(human_ckpt, device=device, env_obs_dim=obs_dim, env_act_dim=act_dim, name="human_critic"),
    ]
    actors = [
        ActorBundle(name="oracle_teacher", kind="oracle", actor_backbone=None, actor_head=None, obs_normalizer=None, act_dim=act_dim),
        _load_actor_bundle(scripted_ckpt, device=device, env_obs_dim=obs_dim, env_act_dim=act_dim, name="scripted_actor"),
        _load_actor_bundle(human_ckpt, device=device, env_obs_dim=obs_dim, env_act_dim=act_dim, name="human_actor"),
        ActorBundle(name="random_actor", kind="random", actor_backbone=None, actor_head=None, obs_normalizer=None, act_dim=act_dim),
    ]

    seeds = [int(args.seed_start) + idx for idx in range(int(args.episodes))]
    started = time.time()
    results = [
        _rollout_actor(actor=actor, base_args=base_args, critics=critics, device=device, seeds=seeds)
        for actor in actors
    ]
    payload = {
        "scripted_ckpt": str(scripted_ckpt),
        "human_ckpt": str(human_ckpt),
        "episodes": int(args.episodes),
        "seeds": seeds,
        "device": str(device),
        "duration_sec": float(time.time() - started),
        "results": results,
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table = _format_table(results, critic_names=[critic.name for critic in critics])
    print(table, flush=True)
    print(f"[CrossplayDone] json={output_json}", flush=True)


if __name__ == "__main__":
    main()
