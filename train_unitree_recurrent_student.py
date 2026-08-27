"""Train GRU and GRU-plus-height-reconstruction modality students."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_unitree_modality_student import _sensor_input, _teacher
from unitree_locomotion_cells import GEOMETRIES, MATERIALS, make_locomotion_cell_cfg
from unitree_multimodal_locomotion import (
    MODALITIES,
    MultimodalGruActor,
    MultimodalGruReconstructionActor,
    modality_mask,
)
from unitree_nav_vision import UnitreeVisionCfg, attach_vision_sensors


ARCHITECTURES = ("gru", "gru_reconstruction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    parser.add_argument("--material", choices=MATERIALS, required=True)
    parser.add_argument("--modality", choices=MODALITIES, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--unroll-length", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--camera-width", type=int, default=80)
    parser.add_argument("--camera-height", type=int, default=60)
    parser.add_argument("--camera-max-depth", type=float, default=5.0)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.architecture == "gru_reconstruction" and args.modality == "height_scan":
        raise ValueError("Height-scan reconstruction from height scan is a trivial target")
    if args.smoke:
        args.steps, args.num_envs, args.unroll_length = 3, min(args.num_envs, 2), 3
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    cfg = make_locomotion_cell_cfg(args.geometry, args.material, num_envs=args.num_envs)
    cfg.seed = args.seed
    camera_names: tuple[str, ...] = ()
    if args.modality != "height_scan":
        mode = "rgbd" if args.modality == "depth" else args.modality
        camera_names = attach_vision_sensors(cfg, UnitreeVisionCfg(
            mode=mode, width=args.camera_width, height=args.camera_height,
            max_depth_m=args.camera_max_depth,
        ))
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=str(device)))
    obs, _ = env.reset()
    teacher, teacher_mean, teacher_std = _teacher(args.expert_checkpoint, device)
    if args.architecture == "gru":
        student = MultimodalGruActor(proprio_dim=98, action_dim=29).to(device)
    else:
        student = MultimodalGruReconstructionActor(
            proprio_dim=98, action_dim=29, reconstruction_dim=187
        ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    start_step = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("architecture") != args.architecture:
            raise ValueError("Resume checkpoint architecture mismatch")
        student.load_state_dict(saved["student_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_step = int(saved.get("global_step", 0))

    availability = modality_mask([args.modality] * args.num_envs, device)
    hidden = student.initial_hidden(args.num_envs, device)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    pending_losses = []
    reward_total = 0.0
    done_total = 0
    last_bc = last_reconstruction = 0.0
    for local_step in range(args.steps):
        actor_obs = obs["actor"]
        with torch.inference_mode():
            inference_target = teacher((actor_obs - teacher_mean) / teacher_std)
        target = inference_target.clone()
        terrain_obs = _sensor_input(
            args.modality, obs, env, camera_names, args.camera_max_depth
        )
        if args.architecture == "gru":
            predicted, hidden = student(
                actor_obs[:, :98], {args.modality: terrain_obs}, availability, hidden
            )
            reconstruction_loss = predicted.new_zeros(())
        else:
            predicted, reconstruction, hidden = student(
                actor_obs[:, :98], {args.modality: terrain_obs}, availability, hidden
            )
            clean_height_target = obs["critic"][:, 98:285]
            reconstruction_loss = torch.nn.functional.mse_loss(
                reconstruction, clean_height_target
            )
        bc_loss = torch.nn.functional.mse_loss(predicted, target)
        pending_losses.append(bc_loss + args.reconstruction_weight * reconstruction_loss)
        last_bc = float(bc_loss.detach().item())
        last_reconstruction = float(reconstruction_loss.detach().item())
        obs, reward, dones, _ = env.step(target)
        hidden = hidden * (1.0 - dones.float()).unsqueeze(1)
        reward_total += float(reward.mean().item())
        done_total += int(dones.sum().item())
        global_step = start_step + local_step + 1
        update_due = len(pending_losses) == args.unroll_length or local_step + 1 == args.steps
        if update_due:
            loss = torch.stack(pending_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            hidden = hidden.detach()
            pending_losses.clear()
        if global_step % 100 == 0 or local_step + 1 == args.steps:
            row = {
                "global_step": global_step, "cell_step": local_step + 1,
                "bc_loss": last_bc, "reconstruction_loss": last_reconstruction,
                "mean_step_reward": reward_total / (local_step + 1),
                "termination_count": done_total, "architecture": args.architecture,
                "geometry": args.geometry, "material": args.material,
                "modality": args.modality, "unroll_length": args.unroll_length,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print("[recurrent-student] " + json.dumps(row, sort_keys=True), flush=True)
        if global_step % args.checkpoint_interval == 0 or local_step + 1 == args.steps:
            torch.save({
                "student_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step, "architecture": args.architecture,
                "modality": args.modality, "proprio_dim": 98, "action_dim": 29,
                "reconstruction_dim": 187, "unroll_length": args.unroll_length,
                "camera": {"width": args.camera_width, "height": args.camera_height, "max_depth": args.camera_max_depth},
            }, args.output / f"student_{global_step}.pt")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
