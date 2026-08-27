"""Evaluate the privileged Unitree expert with student-matched metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_unitree_modality_student import _teacher
from unitree_locomotion_cells import GEOMETRIES, MATERIALS, make_locomotion_cell_cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    parser.add_argument("--material", choices=MATERIALS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    cfg = make_locomotion_cell_cfg(
        args.geometry, args.material, play=True, num_envs=args.num_envs
    )
    cfg.seed = args.seed
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=str(device)))
    teacher, mean, std = _teacher(args.checkpoint, device)
    obs, _ = env.reset()
    reward_sum = tracking_sum = speed_sum = action_sum = action_delta_sum = 0.0
    fall_events = nonfinite = 0
    previous_action = None
    for _ in range(args.steps):
        with torch.inference_mode():
            action = teacher((obs["actor"] - mean) / std)
        nonfinite += int((~torch.isfinite(action)).sum().item())
        action_sum += float(action.abs().mean().item())
        if previous_action is not None:
            action_delta_sum += float((action - previous_action).abs().mean().item())
        previous_action = action
        obs, reward, dones, _ = env.step(action)
        raw = env.unwrapped
        command = raw.command_manager.get_command("twist")
        velocity = raw.scene["robot"].data.root_link_lin_vel_b[:, :2]
        tracking_sum += float(
            torch.linalg.vector_norm(command[:, :2] - velocity, dim=1).mean().item()
        )
        speed_sum += float(torch.linalg.vector_norm(velocity, dim=1).mean().item())
        reward_sum += float(reward.mean().item())
        fall_events += int(dones.sum().item())
    total_env_steps = args.num_envs * args.steps
    summary = {
        "architecture": "privileged_expert",
        "geometry": args.geometry, "material": args.material, "seed": args.seed,
        "num_envs": args.num_envs, "steps": args.steps,
        "checkpoint": str(args.checkpoint.resolve()),
        "mean_step_reward": reward_sum / args.steps,
        "mean_velocity_tracking_error": tracking_sum / args.steps,
        "mean_achieved_speed": speed_sum / args.steps,
        "mean_abs_action": action_sum / args.steps,
        "mean_abs_action_delta": action_delta_sum / max(args.steps - 1, 1),
        "fall_events": fall_events,
        "fall_events_per_1000_env_steps": 1000.0 * fall_events / total_env_steps,
        "nonfinite_action_count": nonfinite,
    }
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("[expert-reference] " + json.dumps(summary, sort_keys=True))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
