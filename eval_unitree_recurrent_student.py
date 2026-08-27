"""Evaluate a GRU or GRU-reconstruction modality student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_unitree_modality_student import _sensor_input
from unitree_locomotion_cells import GEOMETRIES, MATERIALS, make_locomotion_cell_cfg
from unitree_multimodal_locomotion import (
    MODALITIES,
    MultimodalGruActor,
    MultimodalGruReconstructionActor,
    modality_mask,
    preprocess_student_input,
)
from unitree_nav_vision import UnitreeVisionCfg, attach_vision_sensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    parser.add_argument("--material", choices=MATERIALS, required=True)
    parser.add_argument("--modality", choices=MODALITIES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = saved["architecture"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    cfg = make_locomotion_cell_cfg(args.geometry, args.material, play=True, num_envs=args.num_envs)
    cfg.seed = args.seed
    camera_cfg = saved["camera"]
    camera_names: tuple[str, ...] = ()
    if args.modality != "height_scan":
        mode = "rgbd" if args.modality == "depth" else args.modality
        camera_names = attach_vision_sensors(cfg, UnitreeVisionCfg(
            mode=mode, width=int(camera_cfg["width"]), height=int(camera_cfg["height"]),
            max_depth_m=float(camera_cfg["max_depth"]),
        ))
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=str(device)))
    if architecture == "gru":
        student = MultimodalGruActor(proprio_dim=98, action_dim=29).to(device)
    elif architecture == "gru_reconstruction":
        student = MultimodalGruReconstructionActor(
            proprio_dim=98, action_dim=29, reconstruction_dim=187
        ).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    student.load_state_dict(saved["student_state_dict"])
    student.eval()
    availability = modality_mask([args.modality] * args.num_envs, device)
    hidden = student.initial_hidden(args.num_envs, device)
    obs, _ = env.reset()
    reward_sum = tracking_sum = speed_sum = action_sum = action_delta_sum = 0.0
    fall_events = nonfinite = 0
    previous_action = None
    for _ in range(args.steps):
        actor_obs = obs["actor"]
        terrain_obs = _sensor_input(args.modality, obs, env, camera_names, float(camera_cfg["max_depth"]))
        proprio, terrain_obs = preprocess_student_input(
            actor_obs, terrain_obs, args.modality, saved
        )
        with torch.inference_mode():
            output = student(proprio, {args.modality: terrain_obs}, availability, hidden)
            if architecture == "gru":
                action, hidden = output
            else:
                action, _, hidden = output
        nonfinite += int((~torch.isfinite(action)).sum().item())
        action_sum += float(action.abs().mean().item())
        if previous_action is not None:
            action_delta_sum += float((action - previous_action).abs().mean().item())
        previous_action = action
        obs, reward, dones, _ = env.step(action)
        with torch.inference_mode():
            hidden = hidden * (1.0 - dones.float()).unsqueeze(1)
        raw = env.unwrapped
        command = raw.command_manager.get_command("twist")
        velocity = raw.scene["robot"].data.root_link_lin_vel_b[:, :2]
        tracking_sum += float(torch.linalg.vector_norm(command[:, :2] - velocity, dim=1).mean().item())
        speed_sum += float(torch.linalg.vector_norm(velocity, dim=1).mean().item())
        reward_sum += float(reward.mean().item())
        fall_events += int(dones.sum().item())
    total_env_steps = args.num_envs * args.steps
    summary = {
        "architecture": architecture, "modality": args.modality,
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
    print("[recurrent-eval] " + json.dumps(summary, sort_keys=True))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
