"""Online teacher-controlled distillation for one Unitree sensor modality."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

from unitree_locomotion_cells import GEOMETRIES, MATERIALS, make_locomotion_cell_cfg
from unitree_multimodal_locomotion import MODALITIES, MultimodalNoMemoryActor, modality_mask
from unitree_nav_vision import UnitreeVisionCfg, attach_vision_sensors

RSL_RL_NORMALIZATION_EPS = 1e-2


def _rsl_normalization_denominator(std: torch.Tensor) -> torch.Tensor:
    return std + RSL_RL_NORMALIZATION_EPS


def _teacher(checkpoint: Path, device: torch.device):
    value = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" not in value:
        from eval_unitree_velocity_policy import _split_to_legacy_checkpoint

        value = _split_to_legacy_checkpoint(value)
    state = value["model_state_dict"]
    layers = nn.Sequential(
        nn.Linear(285, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 29),
    ).to(device)
    actor_state = {key.removeprefix("actor."): tensor for key, tensor in state.items() if key.startswith("actor.")}
    layers.load_state_dict(actor_state)
    layers.eval()
    mean = state["actor_obs_normalizer._mean"].to(device)
    # Match rsl_rl.networks.normalization.EmpiricalNormalization.forward.
    # A clamp is not equivalent for constant features whose stored std is zero.
    std = _rsl_normalization_denominator(
        state["actor_obs_normalizer._std"].to(device)
    )
    return layers, mean, std


def _scan_image(scan: torch.Tensor) -> torch.Tensor:
    width = max(factor for factor in range(1, int(math.sqrt(scan.shape[1])) + 1) if scan.shape[1] % factor == 0)
    height = scan.shape[1] // width
    return scan.reshape(scan.shape[0], 1, width, height)


def _sensor_input(modality: str, obs, env, camera_names, max_depth: float) -> torch.Tensor:
    actor_obs = obs["actor"]
    if modality == "height_scan":
        return _scan_image(actor_obs[:, 98:])
    data = [env.unwrapped.scene.sensors.get(name).data for name in camera_names]
    if modality == "depth":
        depth = data[0].depth
        if depth is None:
            raise RuntimeError("Depth camera returned no depth tensor")
        return torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0).permute(0, 3, 1, 2) / max_depth
    rgbs = []
    for camera in data:
        if camera.rgb is None:
            raise RuntimeError("RGB camera returned no RGB tensor")
        rgbs.append(camera.rgb.permute(0, 3, 1, 2).float() / 255.0)
    return torch.cat(rgbs, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    parser.add_argument("--material", choices=MATERIALS, required=True)
    parser.add_argument("--modality", choices=MODALITIES, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--camera-width", type=int, default=80)
    parser.add_argument("--camera-height", type=int, default=60)
    parser.add_argument("--camera-max-depth", type=float, default=5.0)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.steps, args.num_envs = 2, min(args.num_envs, 2)
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
    student = MultimodalNoMemoryActor(proprio_dim=98, action_dim=29).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    start_step = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(saved["student_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_step = int(saved.get("global_step", 0))
    availability = modality_mask([args.modality] * args.num_envs, device)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    reward_total = 0.0
    done_total = 0
    for local_step in range(args.steps):
        actor_obs = obs["actor"]
        with torch.inference_mode():
            inference_target = teacher((actor_obs - teacher_mean) / teacher_std)
        target = inference_target.clone()
        terrain_obs = _sensor_input(
            args.modality, obs, env, camera_names, args.camera_max_depth
        )
        predicted = student(actor_obs[:, :98], {args.modality: terrain_obs}, availability)
        loss = torch.nn.functional.mse_loss(predicted, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        obs, reward, dones, _ = env.step(target)
        reward_total += float(reward.mean().item())
        done_total += int(dones.sum().item())
        global_step = start_step + local_step + 1
        if global_step % 100 == 0 or local_step + 1 == args.steps:
            row = {
                "global_step": global_step, "cell_step": local_step + 1,
                "bc_loss": float(loss.item()),
                "mean_step_reward": reward_total / (local_step + 1),
                "termination_count": done_total,
                "geometry": args.geometry, "material": args.material,
                "modality": args.modality,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print("[student] " + json.dumps(row, sort_keys=True), flush=True)
        if global_step % args.checkpoint_interval == 0 or local_step + 1 == args.steps:
            torch.save({
                "student_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "modality": args.modality,
                "proprio_dim": 98, "action_dim": 29,
                "camera": {"width": args.camera_width, "height": args.camera_height, "max_depth": args.camera_max_depth},
            }, args.output / f"student_{global_step}.pt")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
