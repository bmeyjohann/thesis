"""Distill one Unitree locomotion student shared by all sensor modalities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from train_unitree_modality_student import _scan_image, _teacher
from unitree_locomotion_cells import GEOMETRIES, MATERIALS, make_locomotion_cell_cfg, make_locomotion_mixed_cfg
from unitree_multimodal_locomotion import (
    MODALITIES,
    MultimodalGruActor,
    MultimodalGruReconstructionActor,
    MultimodalNoMemoryActor,
    modality_mask,
)
from unitree_nav_vision import UnitreeVisionCfg, attach_vision_sensors

ARCHITECTURES = ("nomemory", "gru", "gru_reconstruction")


def balanced_modality_names(
    num_envs: int, global_step: int, hold_steps: int = 500
) -> list[str]:
    """Balance modalities across envs and rotate only at deployment-like intervals."""
    if hold_steps < 1:
        raise ValueError("hold_steps must be positive")
    phase = global_step // hold_steps
    return [MODALITIES[(index + phase) % len(MODALITIES)] for index in range(num_envs)]


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    availability: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct privileged terrain only when it was not directly observed."""
    rows = availability[:, MODALITIES.index("height_scan")] < 0.5
    if not torch.any(rows):
        return prediction.new_zeros(())
    return F.mse_loss(prediction[rows], target[rows])


def _all_sensor_inputs(
    obs,
    env,
    camera_names: tuple[str, ...],
    max_depth: float,
    teacher_mean: torch.Tensor,
    teacher_denominator: torch.Tensor,
):
    actor_obs = obs["actor"]
    left = env.unwrapped.scene.sensors.get(camera_names[0]).data
    right = env.unwrapped.scene.sensors.get(camera_names[1]).data
    if left.depth is None or left.rgb is None or right.rgb is None:
        raise RuntimeError("Stereo RGB-D rig did not return all requested tensors")
    depth = torch.nan_to_num(
        left.depth, nan=max_depth, posinf=max_depth, neginf=0.0
    ).permute(0, 3, 1, 2) / max_depth
    left_rgb = left.rgb.permute(0, 3, 1, 2).float() / 255.0
    right_rgb = right.rgb.permute(0, 3, 1, 2).float() / 255.0
    return {
        "height_scan": _scan_image(
            (actor_obs[:, 98:] - teacher_mean[:, 98:])
            / teacher_denominator[:, 98:]
        ),
        "depth": depth,
        "mono_rgb": left_rgb,
        "stereo_rgb": torch.cat((left_rgb, right_rgb), dim=1),
    }


def _make_student(architecture: str, device: torch.device):
    if architecture == "nomemory":
        return MultimodalNoMemoryActor(proprio_dim=98, action_dim=29).to(device)
    if architecture == "gru":
        return MultimodalGruActor(proprio_dim=98, action_dim=29).to(device)
    return MultimodalGruReconstructionActor(
        proprio_dim=98, action_dim=29, reconstruction_dim=187
    ).to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--geometry", choices=(*GEOMETRIES, "mixed"), required=True)
    parser.add_argument("--material", choices=(*MATERIALS, "mixed"), required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--unroll-length", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--modality-hold-steps", type=int, default=500)
    parser.add_argument("--camera-width", type=int, default=80)
    parser.add_argument("--camera-height", type=int, default=60)
    parser.add_argument("--camera-max-depth", type=float, default=5.0)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _save(
    args,
    student,
    optimizer,
    global_step: int,
    teacher_mean: torch.Tensor,
    teacher_denominator: torch.Tensor,
) -> Path:
    path = args.output / f"student_{global_step}.pt"
    torch.save({
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "architecture": args.architecture,
        "modalities": list(MODALITIES),
        "modality_sampling": "balanced_per_environment_held_one_hot",
        "modality_hold_steps": args.modality_hold_steps,
        "teacher_obs_normalization_epsilon": 1e-2,
        "input_normalization": {
            "proprio_mean": teacher_mean[:, :98].detach().cpu(),
            "proprio_denominator": teacher_denominator[:, :98].detach().cpu(),
            "height_scan_mean": teacher_mean[:, 98:].detach().cpu(),
            "height_scan_denominator": teacher_denominator[:, 98:].detach().cpu(),
        },
        "proprio_dim": 98,
        "action_dim": 29,
        "reconstruction_dim": 187,
        "unroll_length": args.unroll_length,
        "camera": {
            "width": args.camera_width,
            "height": args.camera_height,
            "max_depth": args.camera_max_depth,
        },
    }, path)
    return path


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.steps = 3
        args.num_envs = min(args.num_envs, 4)
        args.unroll_length = min(args.unroll_length, 3)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    if (args.geometry == "mixed") != (args.material == "mixed"):
        raise ValueError("mixed geometry and mixed material must be selected together")
    cfg = (
        make_locomotion_mixed_cfg(num_envs=args.num_envs)
        if args.geometry == "mixed"
        else make_locomotion_cell_cfg(
            args.geometry, args.material, num_envs=args.num_envs
        )
    )
    cfg.seed = args.seed
    camera_names = attach_vision_sensors(cfg, UnitreeVisionCfg(
        mode="stereo_rgbd",
        width=args.camera_width,
        height=args.camera_height,
        max_depth_m=args.camera_max_depth,
    ))
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=str(device)))
    obs, _ = env.reset()
    teacher, teacher_mean, teacher_std = _teacher(args.expert_checkpoint, device)
    student = _make_student(args.architecture, device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    start_step = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("architecture") != args.architecture:
            raise ValueError("Resume checkpoint architecture mismatch")
        if tuple(saved.get("modalities", ())) != MODALITIES:
            raise ValueError("Resume checkpoint is not a shared multimodal student")
        student.load_state_dict(saved["student_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_step = int(saved.get("global_step", 0))

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    recurrent = args.architecture != "nomemory"
    hidden = student.initial_hidden(args.num_envs, device) if recurrent else None
    pending_losses: list[torch.Tensor] = []
    reward_total = 0.0
    done_total = 0
    latest = {"bc_loss": 0.0, "reconstruction_loss": 0.0}
    latest.update({f"bc_loss_{name}": 0.0 for name in MODALITIES})

    for local_step in range(args.steps):
        global_step = start_step + local_step
        actor_obs = obs["actor"]
        with torch.inference_mode():
            inference_target = teacher((actor_obs - teacher_mean) / teacher_std)
        target = inference_target.clone()
        observations = _all_sensor_inputs(
            obs,
            env,
            camera_names,
            args.camera_max_depth,
            teacher_mean,
            teacher_std,
        )
        student_proprio = (
            actor_obs[:, :98] - teacher_mean[:, :98]
        ) / teacher_std[:, :98]
        names = balanced_modality_names(
            args.num_envs, global_step, args.modality_hold_steps
        )
        availability = modality_mask(names, device)
        if args.architecture == "nomemory":
            predicted = student(student_proprio, observations, availability)
            reconstruction_loss = predicted.new_zeros(())
        elif args.architecture == "gru":
            predicted, hidden = student(
                student_proprio, observations, availability, hidden
            )
            reconstruction_loss = predicted.new_zeros(())
        else:
            predicted, reconstruction, hidden = student(
                student_proprio, observations, availability, hidden
            )
            reconstruction_loss = masked_reconstruction_loss(
                reconstruction,
                (actor_obs[:, 98:285] - teacher_mean[:, 98:285])
                / teacher_std[:, 98:285],
                availability,
            )
        bc_loss = F.mse_loss(predicted, target)
        loss = bc_loss + args.reconstruction_weight * reconstruction_loss
        latest["bc_loss"] = float(bc_loss.detach().item())
        latest["reconstruction_loss"] = float(reconstruction_loss.detach().item())
        for index, name in enumerate(MODALITIES):
            rows = availability[:, index] > 0.5
            latest[f"bc_loss_{name}"] = float(F.mse_loss(predicted[rows], target[rows]).detach().item())

        if recurrent:
            pending_losses.append(loss)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        obs, reward, dones, _ = env.step(target)
        reward_total += float(reward.mean().item())
        done_total += int(dones.sum().item())
        if recurrent:
            hidden = hidden * (1.0 - dones.float()).unsqueeze(1)
            if len(pending_losses) == args.unroll_length or local_step + 1 == args.steps:
                optimizer.zero_grad(set_to_none=True)
                torch.stack(pending_losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                hidden = hidden.detach()
                pending_losses.clear()

        completed_step = global_step + 1
        if completed_step % 100 == 0 or local_step + 1 == args.steps:
            counts = {name: names.count(name) for name in MODALITIES}
            row = {
                "global_step": completed_step,
                "cell_step": local_step + 1,
                "architecture": args.architecture,
                "geometry": args.geometry,
                "material": args.material,
                "mean_step_reward": reward_total / (local_step + 1),
                "termination_count": done_total,
                **latest,
                **{f"assigned_{name}": count for name, count in counts.items()},
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print("[shared-student] " + json.dumps(row, sort_keys=True), flush=True)
        if completed_step % args.checkpoint_interval == 0 or local_step + 1 == args.steps:
            _save(
                args,
                student,
                optimizer,
                completed_step,
                teacher_mean,
                teacher_std,
            )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
