#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_interactive_safetygym import _apply_ckpt_defaults, _build_env, build_parser
from safetygym_utils.controllers import ScriptedGeometricTeacherController
from safetygym_utils.env import extract_agent_xy, extract_goal_xy, scale_action_np
from safetygym_utils.policy_viz import _build_networks, _load_checkpoint


def _make_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = build_parser()
    eval_args = parser.parse_args(
        [
            "--model_path",
            str(args.model_path),
            "--controller",
            "policy",
            "--policy_format",
            "fastsac",
            "--render_mode",
            "none",
            "--env_name",
            args.env_name,
            "--seed",
            str(args.seed),
            "--layout_curriculum",
            args.layout_curriculum,
            "--reward_mode",
            args.reward_mode,
            "--dense_reward_scale",
            str(args.dense_reward_scale),
            "--success_reward_scale",
            str(args.success_reward_scale),
            "--step_penalty",
            str(args.step_penalty),
            "--clearance_penalty_scale",
            str(args.clearance_penalty_scale),
            "--car_action_mode",
            args.car_action_mode,
            "--num_episodes",
            str(args.num_episodes),
            "--fps",
            "0",
            "--terminate_on_goal",
        ]
    )
    _apply_ckpt_defaults(eval_args)
    # Keep audit overrides explicit after checkpoint defaults.
    eval_args.env_name = args.env_name
    eval_args.layout_curriculum = args.layout_curriculum
    eval_args.reward_mode = args.reward_mode
    eval_args.dense_reward_scale = args.dense_reward_scale
    eval_args.success_reward_scale = args.success_reward_scale
    eval_args.step_penalty = args.step_penalty
    eval_args.clearance_penalty_scale = args.clearance_penalty_scale
    eval_args.car_action_mode = args.car_action_mode
    eval_args.terminate_on_goal = True
    eval_args.render_mode = "none"
    eval_args.intervention_mode = "none"
    eval_args.seed = args.seed
    return eval_args


def _scale_if_needed(action: np.ndarray, action_space, scale_to_env: bool) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if scale_to_env:
        action = scale_action_np(action, action_space)
    return np.clip(action, action_space.low, action_space.high).astype(np.float32, copy=False)


def _actor_actions(
    actor,
    obs_preprocess,
    obs: np.ndarray,
    *,
    device: torch.device,
    action_space,
    scale_to_env: bool,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
    obs_t = obs_preprocess(obs_t)
    with torch.no_grad():
        _, _, mean_t = actor(obs_t)
        mean = mean_t[0].detach().cpu().numpy().astype(np.float32)
        samples = []
        batch = obs_t.repeat(int(max(1, num_samples)), 1)
        action_t, _, _ = actor(batch)
        for row in action_t.detach().cpu().numpy().astype(np.float32):
            samples.append(_scale_if_needed(row, action_space, scale_to_env))
    return _scale_if_needed(mean, action_space, scale_to_env), np.asarray(samples, dtype=np.float32)


def _q_grid(
    critic,
    obs_preprocess,
    obs: np.ndarray,
    *,
    device: torch.device,
    low: np.ndarray,
    high: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(float(low[0]), float(high[0]), int(grid_size), dtype=np.float32)
    ys = np.linspace(float(low[1]), float(high[1]), int(grid_size), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    actions = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    obs_batch = np.repeat(np.asarray(obs, dtype=np.float32).reshape(1, -1), actions.shape[0], axis=0)
    with torch.no_grad():
        obs_t = obs_preprocess(torch.as_tensor(obs_batch, device=device, dtype=torch.float32))
        act_t = torch.as_tensor(actions, device=device, dtype=torch.float32)
        q_list = critic(obs_t, act_t)
        q_min = torch.min(torch.stack(q_list, dim=0), dim=0).values.reshape(-1)
    return xs, ys, q_min.detach().cpu().numpy().reshape(int(grid_size), int(grid_size))


def _plot_case(
    *,
    out_path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    q: np.ndarray,
    actor_mean: np.ndarray,
    samples: np.ndarray,
    teacher_action: np.ndarray | None,
    cost: float,
    ep: int,
    step: int,
    agent_xy: np.ndarray | None,
    goal_xy: np.ndarray | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.4), dpi=140)
    im = ax.imshow(
        q,
        origin="lower",
        extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
        aspect="auto",
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="min critic Q(obs, action)")
    if samples.size:
        ax.scatter(samples[:, 0], samples[:, 1], s=12, c="white", alpha=0.38, label="actor stochastic samples")
    ax.scatter([actor_mean[0]], [actor_mean[1]], s=100, c="red", marker="x", linewidths=2.5, label="actor mean")
    if teacher_action is not None:
        ax.scatter(
            [teacher_action[0]],
            [teacher_action[1]],
            s=95,
            c="cyan",
            marker="*",
            edgecolors="black",
            linewidths=0.7,
            label="scripted teacher",
        )
    best_idx = np.unravel_index(int(np.nanargmax(q)), q.shape)
    ax.scatter([xs[best_idx[1]]], [ys[best_idx[0]]], s=85, c="orange", marker="o", edgecolors="black", label="grid argmax Q")
    title = f"episode {ep}, step {step}, incurred cost={cost:.1f}"
    if agent_xy is not None and goal_xy is not None:
        title += f"\nagent=({agent_xy[0]:.2f},{agent_xy[1]:.2f}) goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f})"
    ax.set_title(title)
    ax.set_xlabel("action[0] / left wheel")
    ax.set_ylabel("action[1] / right wheel")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    eval_args = _make_eval_args(args)
    env = _build_env(eval_args, controller=None)
    checkpoint, train_args = _load_checkpoint(model_path, device)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    if act_dim != 2:
        raise ValueError(f"action landscape diagnostic currently expects 2D actions, got {act_dim}")
    actor, critic, obs_preprocess = _build_networks(
        checkpoint=checkpoint,
        train_args=train_args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )
    teacher = ScriptedGeometricTeacherController(
        action_low=np.asarray(env.action_space.low, dtype=np.float32),
        action_high=np.asarray(env.action_space.high, dtype=np.float32),
    )
    scale_to_env = bool(train_args.get("scale_actor_to_env_bounds", getattr(eval_args, "scale_actor_to_env_bounds", False)))
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

    cases: list[dict[str, Any]] = []
    total_cost = 0.0
    total_success = 0
    for ep in range(int(args.num_episodes)):
        obs, _ = env.reset(seed=int(args.seed) + ep)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        for step in range(int(args.max_steps)):
            actor_mean, samples = _actor_actions(
                actor,
                obs_preprocess,
                obs,
                device=device,
                action_space=env.action_space,
                scale_to_env=scale_to_env,
                num_samples=int(args.actor_samples),
            )
            teacher_action = teacher.get_action(obs=obs, env=env)
            agent_xy = extract_agent_xy(env)
            goal_xy = extract_goal_xy(env)
            next_obs, _reward, cost, terminated, truncated, info = env.step(actor_mean)
            total_cost += float(cost)
            if bool(info.get("goal_met", False)):
                total_success += 1
            if float(cost) > 0.0 and len(cases) < int(args.max_cases):
                xs, ys, q = _q_grid(
                    critic,
                    obs_preprocess,
                    obs,
                    device=device,
                    low=low,
                    high=high,
                    grid_size=int(args.grid_size),
                )
                img_path = out_dir / f"case_{len(cases)+1:02d}_ep{ep+1:03d}_step{step+1:04d}.png"
                _plot_case(
                    out_path=img_path,
                    xs=xs,
                    ys=ys,
                    q=q,
                    actor_mean=actor_mean,
                    samples=samples,
                    teacher_action=teacher_action,
                    cost=float(cost),
                    ep=ep + 1,
                    step=step + 1,
                    agent_xy=None if agent_xy is None else np.asarray(agent_xy, dtype=np.float32).reshape(2),
                    goal_xy=None if goal_xy is None else np.asarray(goal_xy, dtype=np.float32).reshape(2),
                )
                best_idx = np.unravel_index(int(np.nanargmax(q)), q.shape)
                cases.append(
                    {
                        "episode": ep + 1,
                        "step": step + 1,
                        "cost": float(cost),
                        "actor_mean": actor_mean.tolist(),
                        "teacher_action": None if teacher_action is None else np.asarray(teacher_action).reshape(-1).tolist(),
                        "q_actor_mean": float(
                            torch.min(
                                torch.stack(
                                    critic(
                                        obs_preprocess(torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)),
                                        torch.as_tensor(actor_mean[None, :], device=device, dtype=torch.float32),
                                    ),
                                    dim=0,
                                ),
                                dim=0,
                            ).values.item()
                        ),
                        "q_grid_max": float(np.nanmax(q)),
                        "q_grid_argmax_action": [float(xs[best_idx[1]]), float(ys[best_idx[0]])],
                        "plot": str(img_path),
                    }
                )
            obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            if terminated or truncated:
                break
        if len(cases) >= int(args.max_cases):
            break
    env.close()
    summary = {
        "model_path": str(model_path),
        "out_dir": str(out_dir),
        "episodes_started": ep + 1 if "ep" in locals() else 0,
        "total_cost_until_stop": float(total_cost),
        "success_events_until_stop": int(total_success),
        "cases": cases,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose Safety-Gym actor/critic action landscape at cost-incurring states.")
    p.add_argument("--model_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--env_name", default="SafetyCarGoal1-v0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--max_cases", type=int, default=6)
    p.add_argument("--grid_size", type=int, default=61)
    p.add_argument("--actor_samples", type=int, default=128)
    p.add_argument("--layout_curriculum", default="car_random_blocked_filter")
    p.add_argument("--reward_mode", default="dense")
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--success_reward_scale", type=float, default=0.0)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--clearance_penalty_scale", type=float, default=0.0)
    p.add_argument("--car_action_mode", default="raw_wheels")
    summary = run(p.parse_args())
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
