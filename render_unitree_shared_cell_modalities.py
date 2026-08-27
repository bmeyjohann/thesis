"""Render all requested modalities while reusing one terrain/material environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy as media
import numpy as np
import torch

from train_unitree_modality_student import _sensor_input
from unitree_locomotion_cells import GEOMETRIES, MATERIALS, make_locomotion_cell_cfg
from unitree_multimodal_locomotion import (
    MODALITIES,
    MultimodalGruActor,
    MultimodalGruReconstructionActor,
    MultimodalNoMemoryActor,
    modality_mask,
    preprocess_student_input,
)
from unitree_nav_vision import UnitreeVisionCfg, attach_vision_sensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    parser.add_argument("--material", choices=MATERIALS, required=True)
    parser.add_argument("--modalities", choices=MODALITIES, nargs="+", default=MODALITIES)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def _make_student(architecture: str, device: torch.device):
    if architecture == "nomemory":
        return MultimodalNoMemoryActor(proprio_dim=98, action_dim=29).to(device)
    if architecture == "gru":
        return MultimodalGruActor(proprio_dim=98, action_dim=29).to(device)
    if architecture == "gru_reconstruction":
        return MultimodalGruReconstructionActor(
            proprio_dim=98, action_dim=29, reconstruction_dim=187
        ).to(device)
    raise ValueError(f"Unsupported architecture: {architecture}")


def main() -> int:
    args = parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = saved["architecture"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    pending = []
    for modality in args.modalities:
        output_dir = args.output_root / modality / f"{args.geometry}__{args.material}"
        if not (output_dir / "summary.json").exists():
            pending.append((modality, output_dir))
    if not pending:
        return 0

    cfg = make_locomotion_cell_cfg(args.geometry, args.material, play=True, num_envs=1)
    cfg.seed = args.seed
    cfg.viewer.width = args.width
    cfg.viewer.height = args.height
    camera_cfg = saved["camera"]
    camera_names = attach_vision_sensors(cfg, UnitreeVisionCfg(
        mode="stereo_rgbd",
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        max_depth_m=float(camera_cfg["max_depth"]),
    ))
    raw_env = ManagerBasedRlEnv(cfg=cfg, device=str(device), render_mode="rgb_array")
    env = RslRlVecEnvWrapper(raw_env)
    student = _make_student(architecture, device)
    student.load_state_dict(saved["student_state_dict"])
    student.eval()
    fps = raw_env.metadata.get("render_fps", 30)

    for modality, output_dir in pending:
        output_dir.mkdir(parents=True, exist_ok=True)
        selected_cameras = camera_names if modality == "stereo_rgb" else camera_names[:1]
        availability = modality_mask([modality], device)
        hidden = student.initial_hidden(1, device) if architecture != "nomemory" else None
        obs, _ = env.reset()
        reward_sum = 0.0
        termination_count = 0
        frames: list[np.ndarray] = []
        for _ in range(args.steps):
            actor_obs = obs["actor"]
            terrain_obs = _sensor_input(
                modality, obs, env, selected_cameras, float(camera_cfg["max_depth"])
            )
            proprio, terrain_obs = preprocess_student_input(
                actor_obs, terrain_obs, modality, saved
            )
            with torch.inference_mode():
                output = student(
                    proprio, {modality: terrain_obs}, availability
                ) if architecture == "nomemory" else student(
                    proprio, {modality: terrain_obs}, availability, hidden
                )
                if architecture == "nomemory":
                    action = output
                elif architecture == "gru":
                    action, hidden = output
                else:
                    action, _, hidden = output
            obs, reward, dones, _ = env.step(action)
            if hidden is not None:
                hidden = hidden * (1.0 - dones.float()).unsqueeze(1)
            reward_sum += float(reward.mean().item())
            termination_count += int(dones.sum().item())
            frame = raw_env.render()
            if frame is not None:
                frames.append(frame[0] if frame.ndim == 4 else frame)
        video_path = output_dir / "rl-video-step-0.mp4"
        media.write_video(str(video_path), frames, fps=fps)
        summary = {
            "architecture": architecture,
            "checkpoint": str(args.checkpoint.resolve()),
            "modality": modality,
            "geometry": args.geometry,
            "material": args.material,
            "seed": args.seed,
            "steps": args.steps,
            "mean_step_reward": reward_sum / args.steps,
            "termination_count": termination_count,
            "batched_terrain_render": True,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print("[shared-cell-video] " + json.dumps(summary, sort_keys=True))

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
