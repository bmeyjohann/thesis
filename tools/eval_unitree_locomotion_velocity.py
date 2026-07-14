#!/usr/bin/env python3
"""Benchmark a frozen Unitree G1 velocity policy through the navigation action term."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from eval_unitree_nav_baselines import make_env
from train_unitree_nav_thesis import DEFAULT_LOW_LEVEL, ROOT


def _make_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        low_level_policy_path=args.low_level_policy_path,
        task="Unitree-G1-Nav-Flat",
        device=args.device,
        episode_length_s=max(args.duration_s + 2.0, 12.0),
        debug_obstacle_width_min=0.0,
        debug_obstacle_width_max=0.0,
        debug_obstacle_height_min=0.0,
        debug_obstacle_height_max=0.0,
        debug_num_obstacles=0,
        debug_platform_width=0.0,
        debug_obstacle_border_width=0.0,
    )


def evaluate_command(env, command: tuple[float, float, float], args: argparse.Namespace) -> dict[str, float | list[float]]:
    device = torch.device(args.device)
    num_envs = int(args.num_envs)
    action = torch.tensor(command, device=device, dtype=torch.float32).repeat(num_envs, 1)
    env.reset()
    robot = env.env.unwrapped.scene["robot"]
    dt = float(env.env.unwrapped.step_dt)
    warmup_steps = round(float(args.warmup_s) / dt)
    measure_steps = round(float(args.duration_s) / dt)
    velocities: list[np.ndarray] = []
    yaw_rates: list[np.ndarray] = []
    done_count = 0
    for step in range(warmup_steps + measure_steps):
        _, _, done, _ = env.step(action)
        done_count += int(done.reshape(-1).sum().detach().cpu().item())
        if step >= warmup_steps:
            velocities.append(robot.data.root_link_lin_vel_b[:, :2].detach().cpu().numpy().copy())
            yaw_rates.append(robot.data.root_link_ang_vel_b[:, 2].detach().cpu().numpy().copy())
    vel = np.concatenate(velocities, axis=0)
    yaw_rate = np.concatenate(yaw_rates, axis=0)
    action_scale = np.asarray([1.0, 0.5, 0.8], dtype=np.float64)
    physical_command = np.asarray(command, dtype=np.float64) * action_scale
    requested_xy = physical_command[:2]
    requested_speed = float(np.linalg.norm(requested_xy))
    achieved_mean = vel.mean(axis=0)
    projected = vel @ (requested_xy / max(requested_speed, 1e-8)) if requested_speed > 0 else np.zeros(len(vel))
    return {
        "command": list(command),
        "physical_velocity_command": physical_command.tolist(),
        "mean_velocity_xy": achieved_mean.tolist(),
        "std_velocity_xy": vel.std(axis=0).tolist(),
        "mean_command_direction_velocity": float(projected.mean()),
        "tracking_ratio": float(projected.mean() / requested_speed) if requested_speed > 0 else 0.0,
        "mean_yaw_rate": float(yaw_rate.mean()),
        "std_yaw_rate": float(yaw_rate.std()),
        "yaw_tracking_ratio": float(yaw_rate.mean() / physical_command[2]) if abs(physical_command[2]) > 1e-8 else 0.0,
        "fall_or_reset_rate_per_env": float(done_count / max(1, num_envs)),
        "samples": int(len(vel)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-level-policy-path", default=str(DEFAULT_LOW_LEVEL))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--output", default=str(ROOT / "logs" / "unitree_mjlab" / "locomotion_velocity_baseline.json"))
    args = parser.parse_args()
    env = make_env(_make_args(args), num_envs=args.num_envs, render=False)
    commands = [
        (0.25, 0.0, 0.0),
        (0.50, 0.0, 0.0),
        (0.75, 0.0, 0.0),
        (1.00, 0.0, 0.0),
        (0.50, 0.25, 0.0),
        (0.00, 0.50, 0.0),
        (0.00, 0.00, 0.80),
    ]
    results = [evaluate_command(env, command, args) for command in commands]
    env.close()
    payload = {"low_level_policy_path": str(Path(args.low_level_policy_path).resolve()), "results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
