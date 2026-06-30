#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_interactive_safetygym import _apply_ckpt_defaults, _build_env, build_parser
from safetygym_utils.env import extract_goal_distance
from safetygym_utils.policy_viz import _build_networks, _load_checkpoint


def _make_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = build_parser()
    eval_args = parser.parse_args(
        [
            "--model_path",
            str(args.model_path),
            "--controller",
            "policy",
            "--render_mode",
            "none",
            "--env_name",
            args.env_name,
            "--seed",
            str(args.seed),
            "--layout_curriculum",
            args.layout_curriculum,
            "--terminate_on_goal",
            "--reward_mode",
            "dense",
            "--dense_reward_scale",
            "1.0",
            "--success_reward_scale",
            "0.0",
            "--step_penalty",
            "0.0",
            "--clearance_penalty_scale",
            "0.0",
            "--car_action_mode",
            "raw_wheels",
            "--fps",
            "0",
        ]
    )
    _apply_ckpt_defaults(eval_args)
    eval_args.env_name = args.env_name
    eval_args.seed = int(args.seed)
    eval_args.layout_curriculum = args.layout_curriculum
    eval_args.terminate_on_goal = True
    eval_args.reward_mode = "dense"
    eval_args.dense_reward_scale = 1.0
    eval_args.success_reward_scale = 0.0
    eval_args.step_penalty = 0.0
    eval_args.clearance_penalty_scale = 0.0
    eval_args.car_action_mode = "raw_wheels"
    eval_args.intervention_mode = "none"
    return eval_args


def run(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = Path(args.model_path).expanduser().resolve()
    eval_args = _make_args(args)
    env = _build_env(eval_args, controller=None)
    checkpoint, train_args = _load_checkpoint(model_path, device)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    if act_dim != 2:
        raise ValueError(f"critic-greedy diagnostic expects 2D action space, got {act_dim}")
    _actor, critic, obs_preprocess = _build_networks(
        checkpoint=checkpoint,
        train_args=train_args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    xs = np.linspace(float(low[0]), float(high[0]), int(args.grid_size), dtype=np.float32)
    ys = np.linspace(float(low[1]), float(high[1]), int(args.grid_size), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    actions_np = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    actions_t = torch.as_tensor(actions_np, device=device, dtype=torch.float32)

    costs = []
    successes = []
    lengths = []
    hit_steps = []
    for ep in range(int(args.num_episodes)):
        obs, _info = env.reset(seed=int(args.seed) + ep)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        ep_cost = 0.0
        first_hit = None
        for step in range(1, int(args.max_steps) + 1):
            with torch.no_grad():
                obs_batch = np.repeat(obs.reshape(1, -1), actions_np.shape[0], axis=0)
                obs_t = obs_preprocess(torch.as_tensor(obs_batch, device=device, dtype=torch.float32))
                q_list = critic(obs_t, actions_t)
                q_min = torch.min(torch.stack(q_list, dim=0), dim=0).values.reshape(-1)
                action = actions_np[int(torch.argmax(q_min).detach().cpu().item())]
            obs, _reward, cost, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            ep_cost += float(cost)
            if first_hit is None and bool(info.get("goal_met", False)):
                first_hit = step
            if terminated or truncated:
                break
        costs.append(ep_cost)
        successes.append(1.0 if first_hit is not None else 0.0)
        lengths.append(float(step))
        hit_steps.append(float(first_hit if first_hit is not None else args.max_steps))
        print(
            f"episode={ep+1} success={successes[-1]:.0f} cost={ep_cost:.3f} len={step} "
            f"hit={hit_steps[-1]:.0f} final_dist={float(extract_goal_distance(env)):.3f}",
            flush=True,
        )
    env.close()
    summary = {
        "episodes": float(len(costs)),
        "success_rate": float(np.mean(successes)),
        "mean_cost": float(np.mean(costs)),
        "cost_rate": float(np.sum(costs) / max(1.0, float(np.sum(lengths)))),
        "mean_length": float(np.mean(lengths)),
        "mean_first_hit_step": float(np.mean(hit_steps)),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--env_name", default="SafetyCarGoal1-v0")
    p.add_argument("--layout_curriculum", default="car_random_blocked_filter")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--grid_size", type=int, default=21)
    p.add_argument("--device", default="auto")
    run(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
