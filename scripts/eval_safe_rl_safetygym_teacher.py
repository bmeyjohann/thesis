#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from safetygym_utils.policy_viz import (
    _extract_bounds,
    _extract_overlay_specs,
    plot_episode_contact_sheet,
    plot_eval_episode_trajectory,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_policy(obs_dim: int, act_dim: int, cfg: dict[str, Any], device: torch.device):
    from safe_rl.modules import ActorCritic

    policy_cfg = dict(cfg["policy"])
    policy_cfg.pop("class_name", None)
    alg_cfg = cfg["algorithm"]
    cost_limits = alg_cfg.get("cost_limits") or [1.0]
    policy_cfg["num_costs"] = len(cost_limits)
    return ActorCritic(obs_dim, obs_dim, act_dim, **policy_cfg).to(device)


def _extract_goal_metrics(info: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in ("goal_met", "goal_dist", "cost", "num_steps"):
        if key in info:
            try:
                metrics[key] = float(info[key])
            except Exception:
                pass
    return metrics


def _plot_actions(
    *,
    output_path: Path,
    actions: list[np.ndarray],
    costs: list[float],
    rewards: list[float],
    episode: int,
    first_goal_step: int | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actions_arr = np.asarray(actions, dtype=np.float64)
    costs_arr = np.asarray(costs, dtype=np.float64)
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    steps = np.arange(1, actions_arr.shape[0] + 1)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    if actions_arr.ndim == 2 and actions_arr.shape[1] >= 1:
        for dim in range(actions_arr.shape[1]):
            axes[0].plot(steps, actions_arr[:, dim], linewidth=1.2, label=f"action[{dim}]")
        axes[0].set_ylabel("action")
        axes[0].legend(loc="upper right", ncols=min(actions_arr.shape[1], 4), fontsize=8)
    axes[1].plot(steps, rewards_arr, color="#2b8cbe", linewidth=1.0)
    axes[1].set_ylabel("reward")
    axes[2].plot(steps, costs_arr, color="#cb181d", linewidth=1.0)
    axes[2].set_ylabel("cost")
    axes[2].set_xlabel("step")
    if first_goal_step is not None:
        for ax in axes:
            ax.axvline(int(first_goal_step), color="#31a354", linestyle="--", linewidth=1.0, alpha=0.8)
    cost_steps = steps[costs_arr > 0]
    if cost_steps.size:
        for ax in axes:
            ax.scatter(cost_steps, np.full_like(cost_steps, ax.get_ylim()[1]), color="#cb181d", s=8, alpha=0.4)
    fig.suptitle(f"Safe-RL teacher episode {episode} actions/rewards/costs")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import safety_gymnasium

    device = torch.device(args.device)
    cfg = _load_yaml(args.config)

    env = safety_gymnasium.make(args.env_id, render_mode=args.render_mode)
    obs, info = env.reset(seed=args.seed)
    obs_dim = int(obs.shape[0])
    act_dim = int(env.action_space.shape[0])

    policy = _make_policy(obs_dim, act_dim, cfg, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    rows: list[dict[str, Any]] = []
    trajectory_plot_paths: list[Path] = []
    action_plot_paths: list[Path] = []
    total_steps = 0
    task = getattr(env.unwrapped, "task", None)
    overlay_specs = _extract_overlay_specs(task) if task is not None else []
    bounds = _extract_bounds(task, None, None) if task is not None else (-2.0, 2.0, -2.0, 2.0)
    plot_dir: Path | None = None
    if args.out_dir and args.save_episode_plots:
        plot_dir = Path(args.out_dir) / f"{Path(args.checkpoint).stem}_plots"
    try:
        for ep in range(args.episodes):
            if ep > 0:
                obs, info = env.reset(seed=None if args.seed is None else args.seed + ep)
            ep_reward = 0.0
            ep_cost = 0.0
            ep_len = 0
            goal_seen = False
            goal_hits = 0
            first_goal_step: int | None = None
            final_info: dict[str, Any] = {}
            terminated = False
            truncated = False
            path: list[np.ndarray] = []
            goal_positions: list[np.ndarray] = []
            goal_hit_points: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            rewards: list[float] = []
            costs: list[float] = []
            if task is not None:
                path.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
                goal_positions.append(np.asarray(task.goal.pos[:2], dtype=np.float64).copy())

            while not (terminated or truncated):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.inference_mode():
                    action = policy.act_inference(obs_t).squeeze(0).detach().cpu().numpy()
                obs, reward, cost, terminated, truncated, info = env.step(action)
                actions.append(np.asarray(action, dtype=np.float64).copy())
                rewards.append(float(reward))
                costs.append(float(cost))
                if task is not None:
                    path.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
                if args.render_mode == "human" and args.fps > 0:
                    time.sleep(max(0.0, 1.0 / args.fps))
                ep_reward += float(reward)
                ep_cost += float(cost)
                ep_len += 1
                total_steps += 1
                final_info = info or {}
                if bool(final_info.get("goal_met", False)):
                    goal_seen = True
                    goal_hits += 1
                    if task is not None:
                        goal_hit_points.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
                        current_goal = np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
                        if not goal_positions or np.linalg.norm(goal_positions[-1] - current_goal) > 1e-6:
                            goal_positions.append(current_goal)
                    if first_goal_step is None:
                        first_goal_step = ep_len
                if args.max_steps and ep_len >= args.max_steps:
                    truncated = True

            row: dict[str, Any] = {
                "episode": ep,
                "reward": ep_reward,
                "cost": ep_cost,
                "length": ep_len,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "goal_met_any": bool(goal_seen),
                "goal_hit_count": int(goal_hits),
                "first_goal_step": first_goal_step if first_goal_step is not None else 0,
            }
            row.update({f"final_{k}": v for k, v in _extract_goal_metrics(final_info).items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
            if plot_dir is not None and task is not None and ep < int(args.plot_episodes):
                traj_path = plot_dir / f"episode_{ep:03d}_trajectory.png"
                action_path = plot_dir / f"episode_{ep:03d}_actions.png"
                trajectory_plot_paths.append(
                    plot_eval_episode_trajectory(
                        output_path=traj_path,
                        task=task,
                        overlay_specs=overlay_specs,
                        bounds=bounds,
                        path=np.asarray(path, dtype=np.float64),
                        goal_positions=np.asarray(goal_positions, dtype=np.float64),
                        goal_hit_points=np.asarray(goal_hit_points, dtype=np.float64) if goal_hit_points else None,
                        episode_idx=ep + 1,
                        total_episodes=args.episodes,
                        episode_reward=ep_reward,
                        goals_reached=goal_hits,
                        final_distance=float(task.dist_goal()),
                    )
                )
                action_plot_paths.append(
                    _plot_actions(
                        output_path=action_path,
                        actions=actions,
                        costs=costs,
                        rewards=rewards,
                        episode=ep + 1,
                        first_goal_step=first_goal_step,
                    )
                )
    finally:
        env.close()

    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r]
        return sum(vals) / max(len(vals), 1)

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "env_id": args.env_id,
        "episodes": len(rows),
        "mean_reward": mean("reward"),
        "mean_cost": mean("cost"),
        "mean_length": mean("length"),
        "goal_success_rate": sum(1 for r in rows if r.get("goal_met_any")) / max(len(rows), 1),
        "mean_goals_per_episode": mean("goal_hit_count"),
        "mean_first_goal_step": mean("first_goal_step"),
        "total_steps": total_steps,
        "checkpoint_iter": checkpoint.get("iter"),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.checkpoint).stem
        with open(out_dir / f"{stem}_episodes.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            writer.writeheader()
            writer.writerows(rows)
        with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if trajectory_plot_paths:
            contact = plot_episode_contact_sheet(
                image_paths=trajectory_plot_paths,
                output_path=out_dir / f"{stem}_trajectory_contact_sheet.png",
                title=f"{stem} trajectory episodes",
                max_cols=3,
            )
            summary["trajectory_contact_sheet"] = str(contact) if contact is not None else None
        if action_plot_paths:
            contact = plot_episode_contact_sheet(
                image_paths=action_plot_paths,
                output_path=out_dir / f"{stem}_action_contact_sheet.png",
                title=f"{stem} action episodes",
                max_cols=2,
            )
            summary["action_contact_sheet"] = str(contact) if contact is not None else None
        with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Safe-RL Safety-Gymnasium teacher checkpoint in a single env.")
    parser.add_argument("--env_id", default="SafetyCarGoal1-v0")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render_mode", default=None, choices=[None, "human", "rgb_array"], nargs="?")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--out_dir", default="logs/safetygym_safe_rl_teacher_eval")
    parser.add_argument("--save_episode_plots", action="store_true")
    parser.add_argument("--plot_episodes", type=int, default=10)
    args = parser.parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
