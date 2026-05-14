#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.recover_offline_actor_manip_from_checkpoint import (  # noqa: E402
    _auto_device,
    _build_models,
    _coerce_args_dict,
    _eval_metrics,
    _format_metrics,
    _maybe_init_wandb,
    _namespace_from_checkpoint_args,
    _select_teacher_like_mask,
)
from fast_sac_utils import EmpiricalNormalization  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline manipulation preference-critic probe with actor recovery.")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--teacher_mask_mode", type=str, default="effective", choices=["raw", "diff", "effective", "all"])
    p.add_argument("--actor_init_mode", type=str, default="checkpoint", choices=["checkpoint", "random"])
    p.add_argument("--critic_steps", type=int, default=500)
    p.add_argument("--actor_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--pref_loss_type", type=str, default="hinge", choices=["hinge", "bradley_terry"])
    p.add_argument("--pref_margin", type=float, default=0.01)
    p.add_argument("--pref_weight", type=float, default=1.0)
    p.add_argument(
        "--linked_action_filter_mode",
        type=str,
        default="epsilon",
        choices=["epsilon", "angle", "component"],
    )
    p.add_argument(
        "--linked_action_scope",
        type=str,
        default="all",
        choices=["all", "xyz", "gripper", "non_gripper"],
    )
    p.add_argument(
        "--linked_action_filter_metric",
        type=str,
        default="l1",
        choices=["l1", "l2", "mean_abs"],
    )
    p.add_argument(
        "--linked_action_weight_metric",
        type=str,
        default="mean_abs",
        choices=["mean_abs", "l1", "l2"],
    )
    p.add_argument("--linked_action_epsilon", type=float, default=1e-6)
    p.add_argument("--linked_action_weight_scale", type=float, default=0.0)
    p.add_argument("--linked_action_angle_threshold_deg", type=float, default=-1.0)
    p.add_argument("--linked_component_xyz_threshold", type=float, default=-1.0)
    p.add_argument("--linked_component_yaw_threshold", type=float, default=-1.0)
    p.add_argument("--linked_component_gripper_threshold", type=float, default=-1.0)
    p.add_argument("--linked_component_adaptive_enable", action="store_true", default=False)
    p.add_argument("--linked_component_near_distance", type=float, default=-1.0)
    p.add_argument("--linked_component_far_distance", type=float, default=-1.0)
    p.add_argument("--linked_component_near_scale", type=float, default=-1.0)
    p.add_argument("--actor_q_weight", type=float, default=1.0)
    p.add_argument("--bc_teacher_weight", type=float, default=0.0)
    p.add_argument("--action_l2_weight", type=float, default=1e-3)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--eval_num_envs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--name", type=str, default="offline_pref_probe")
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--project", type=str, default="ogbench-manip-offline")
    p.add_argument("--entity", type=str, default="")
    p.add_argument("--group", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="", choices=["", "online", "offline", "disabled"])
    return p.parse_args()


def _weighted_mean(loss_tensor: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    if sample_weights is None:
        return loss_tensor.mean()
    weights = sample_weights.to(device=loss_tensor.device, dtype=loss_tensor.dtype)
    if loss_tensor.ndim == 1:
        weight_view = weights
    elif loss_tensor.ndim == 2:
        weight_view = weights.unsqueeze(-1)
    else:
        weight_view = weights.unsqueeze(0).unsqueeze(-1)
    denom = torch.clamp(weight_view.sum(), min=1e-8)
    return (loss_tensor * weight_view).sum() / denom


def _preference_loss(
    q_teacher: torch.Tensor,
    q_student: torch.Tensor,
    *,
    loss_type: str,
    margin: float,
    sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    delta = q_teacher - q_student
    if loss_type == "hinge":
        return _weighted_mean(F.relu(float(margin) - delta), sample_weights)
    if loss_type == "bradley_terry":
        return _weighted_mean(F.softplus(-(delta - float(margin))), sample_weights)
    raise ValueError(f"unsupported pref_loss_type={loss_type!r}")


def _resolve_float(value: float, ckpt_args: dict[str, object], key: str, fallback: float) -> float:
    if float(value) >= 0.0:
        return float(value)
    raw = ckpt_args.get(key, fallback)
    try:
        return float(raw)
    except Exception:
        return float(fallback)


def _gripper_index_for_size(size: int) -> Optional[int]:
    return (size - 1) if size >= 4 else None


def _yaw_index_for_size(size: int) -> Optional[int]:
    return 3 if size >= 5 else None


def _select_action_scope(actions: torch.Tensor, scope: str) -> torch.Tensor:
    scope_key = str(scope).strip().lower()
    if actions.ndim != 2:
        raise ValueError(f"expected 2D actions tensor, got shape={tuple(actions.shape)}")
    if scope_key == "all":
        return actions
    if scope_key == "xyz":
        return actions[:, : min(3, actions.shape[-1])]
    gripper_idx = _gripper_index_for_size(actions.shape[-1])
    if scope_key == "gripper":
        if gripper_idx is None:
            return actions[:, :0]
        return actions[:, gripper_idx : gripper_idx + 1]
    if scope_key == "non_gripper":
        if gripper_idx is None:
            return actions
        keep = [i for i in range(actions.shape[-1]) if i != gripper_idx]
        if not keep:
            return actions[:, :0]
        return actions[:, keep]
    raise ValueError(f"unsupported linked_action_scope={scope!r}")


def _delta_metric(delta: torch.Tensor, metric: str) -> torch.Tensor:
    if delta.ndim != 2:
        raise ValueError(f"expected 2D delta tensor, got shape={tuple(delta.shape)}")
    if delta.shape[-1] == 0:
        return torch.zeros(delta.shape[0], device=delta.device, dtype=delta.dtype)
    metric_key = str(metric).strip().lower()
    delta_abs = delta.abs()
    if metric_key == "l1":
        return delta_abs.sum(dim=-1)
    if metric_key == "l2":
        return torch.linalg.vector_norm(delta, dim=-1)
    if metric_key == "mean_abs":
        return delta_abs.mean(dim=-1)
    raise ValueError(f"unsupported action metric={metric!r}")


def _angle_deg(actions_a: torch.Tensor, actions_b: torch.Tensor, scope: str) -> torch.Tensor:
    a = _select_action_scope(actions_a, scope)
    b = _select_action_scope(actions_b, scope)
    if a.shape[-1] == 0:
        return torch.full((actions_a.shape[0],), 180.0, device=actions_a.device, dtype=actions_a.dtype)
    an = torch.linalg.vector_norm(a, dim=-1)
    bn = torch.linalg.vector_norm(b, dim=-1)
    denom = torch.clamp(an * bn, min=1e-8)
    cos = torch.clamp((a * b).sum(dim=-1) / denom, min=-1.0, max=1.0)
    ang = torch.rad2deg(torch.arccos(cos))
    zero_mask = (an < 1e-8) | (bn < 1e-8)
    return torch.where(zero_mask, torch.full_like(ang, 180.0), ang)


def _relative_distance_signal(observations: torch.Tensor, ckpt_args: dict[str, object]) -> torch.Tensor:
    include_rel = bool(ckpt_args.get("include_relative_cube_features", False))
    if not include_rel or observations.ndim != 2 or observations.shape[-1] < 10:
        return torch.full((observations.shape[0],), float("nan"), device=observations.device, dtype=observations.dtype)
    rel = observations[:, -10:]
    eff_to_cube_dist = rel[:, 3]
    cube_to_goal_dist = rel[:, 8]
    return torch.minimum(eff_to_cube_dist, cube_to_goal_dist)


def _component_scale(
    observations: torch.Tensor,
    *,
    ckpt_args: dict[str, object],
    adaptive_enable: bool,
    near_distance: float,
    far_distance: float,
    near_scale: float,
) -> torch.Tensor:
    if not adaptive_enable:
        return torch.ones(observations.shape[0], device=observations.device, dtype=observations.dtype)
    d = _relative_distance_signal(observations, ckpt_args)
    if torch.isnan(d).all():
        return torch.ones(observations.shape[0], device=observations.device, dtype=observations.dtype)
    near = max(0.0, float(near_distance))
    far = max(near + 1e-6, float(far_distance))
    near_s = float(np.clip(float(near_scale), 1e-3, 1.0))
    scale = torch.ones_like(d)
    scale = torch.where(d <= near, torch.full_like(scale, near_s), scale)
    mid_mask = torch.isfinite(d) & (d > near) & (d < far)
    alpha = torch.clamp((d - near) / (far - near), min=0.0, max=1.0)
    scale = torch.where(mid_mask, near_s + alpha * (1.0 - near_s), scale)
    return scale


def _component_mask(
    observations: torch.Tensor,
    actions: torch.Tensor,
    student_actions: torch.Tensor,
    *,
    ckpt_args: dict[str, object],
    xyz_threshold: float,
    yaw_threshold: float,
    gripper_threshold: float,
    adaptive_enable: bool,
    near_distance: float,
    far_distance: float,
    near_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    n = min(actions.shape[-1], student_actions.shape[-1])
    d = (actions[:, :n] - student_actions[:, :n]).abs()
    xyz_l2 = torch.linalg.vector_norm(d[:, : min(3, n)], dim=-1) if n >= 1 else torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
    yaw_idx = _yaw_index_for_size(n)
    gripper_idx = _gripper_index_for_size(n)
    yaw_abs = d[:, yaw_idx] if yaw_idx is not None else torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
    gripper_abs = d[:, gripper_idx] if gripper_idx is not None else torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    scale = _component_scale(
        observations,
        ckpt_args=ckpt_args,
        adaptive_enable=adaptive_enable,
        near_distance=near_distance,
        far_distance=far_distance,
        near_scale=near_scale,
    )
    thr_xyz = torch.full_like(scale, float(xyz_threshold)) * scale
    thr_yaw = torch.full_like(scale, float(yaw_threshold)) * scale
    thr_gripper = torch.full_like(scale, float(gripper_threshold)) * scale

    div_xyz = xyz_l2 > thr_xyz if xyz_threshold >= 0.0 else torch.zeros_like(scale, dtype=torch.bool)
    div_yaw = yaw_abs > thr_yaw if yaw_idx is not None and yaw_threshold >= 0.0 else torch.zeros_like(scale, dtype=torch.bool)
    div_gripper = (
        gripper_abs > thr_gripper if gripper_idx is not None and gripper_threshold >= 0.0 else torch.zeros_like(scale, dtype=torch.bool)
    )
    diverged = div_xyz | div_yaw | div_gripper
    diag = {
        "scale": scale,
        "xyz_l2": xyz_l2,
        "yaw_abs": yaw_abs,
        "gripper_abs": gripper_abs,
        "thr_xyz": thr_xyz,
        "thr_yaw": thr_yaw,
        "thr_gripper": thr_gripper,
        "div_xyz": div_xyz.to(scale.dtype),
        "div_yaw": div_yaw.to(scale.dtype),
        "div_gripper": div_gripper.to(scale.dtype),
    }
    return diverged, diag


def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = _auto_device(args.device)

    checkpoint_path = Path(args.checkpoint_path).expanduser()
    dataset_path = Path(args.dataset_path).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = _coerce_args_dict(checkpoint.get("args"))
    args_ns = _namespace_from_checkpoint_args(ckpt_args)
    args_ns.num_eval_episodes = int(args.num_eval_episodes)
    args_ns.eval_num_envs = int(args.eval_num_envs)
    args_ns.train_render_mode = "none"
    args_ns.eval_render_mode = "none"
    args_ns.visualize_intervention_colors = False
    args_ns.use_intervention = False
    args_ns.intervention_mode = "none"

    data = np.load(dataset_path)
    observations = torch.as_tensor(np.asarray(data["observations"], dtype=np.float32), device=device)
    actions = torch.as_tensor(np.asarray(data["actions"], dtype=np.float32), device=device)
    if "student_actions" not in data.files:
        raise ValueError("Preference probe requires dataset['student_actions']; demo-only datasets are not valid here.")
    student_actions = torch.as_tensor(np.asarray(data["student_actions"], dtype=np.float32), device=device)
    teacher_intervened = None
    if "teacher_intervened" in data.files:
        teacher_intervened = torch.as_tensor(np.asarray(data["teacher_intervened"], dtype=np.bool_), device=device)
    action_delta = actions - student_actions
    action_delta_abs = action_delta.abs()
    action_delta_l1 = action_delta_abs.sum(dim=-1)
    action_delta_mean_abs = action_delta_abs.mean(dim=-1)

    delta_scope = _select_action_scope(action_delta, str(args.linked_action_scope))
    filter_metric_values = _delta_metric(delta_scope, str(args.linked_action_filter_metric))
    weight_metric_values = _delta_metric(delta_scope, str(args.linked_action_weight_metric))
    angle_threshold_deg = _resolve_float(
        float(args.linked_action_angle_threshold_deg),
        ckpt_args,
        "tolerance_value",
        30.0,
    )
    component_xyz_threshold = _resolve_float(
        float(args.linked_component_xyz_threshold),
        ckpt_args,
        "tolerance_xyz_value",
        -1.0,
    )
    component_yaw_threshold = _resolve_float(
        float(args.linked_component_yaw_threshold),
        ckpt_args,
        "tolerance_yaw_value",
        -1.0,
    )
    component_gripper_threshold = _resolve_float(
        float(args.linked_component_gripper_threshold),
        ckpt_args,
        "tolerance_gripper_value",
        -1.0,
    )
    component_adaptive_enable = bool(args.linked_component_adaptive_enable) or bool(
        ckpt_args.get("tolerance_adaptive_enable", False)
    )
    component_near_distance = _resolve_float(
        float(args.linked_component_near_distance),
        ckpt_args,
        "tolerance_adaptive_near_distance",
        0.08,
    )
    component_far_distance = _resolve_float(
        float(args.linked_component_far_distance),
        ckpt_args,
        "tolerance_adaptive_far_distance",
        0.30,
    )
    component_near_scale = _resolve_float(
        float(args.linked_component_near_scale),
        ckpt_args,
        "tolerance_adaptive_near_scale",
        0.35,
    )
    filter_mode = str(args.linked_action_filter_mode).strip().lower()
    if filter_mode == "angle":
        filter_gate = _angle_deg(actions, student_actions, str(args.linked_action_scope)) > float(angle_threshold_deg)
        filter_diag: dict[str, torch.Tensor] = {"angle_deg": _angle_deg(actions, student_actions, str(args.linked_action_scope))}
    elif filter_mode == "component":
        filter_gate, filter_diag = _component_mask(
            observations,
            actions,
            student_actions,
            ckpt_args=ckpt_args,
            xyz_threshold=component_xyz_threshold,
            yaw_threshold=component_yaw_threshold,
            gripper_threshold=component_gripper_threshold,
            adaptive_enable=component_adaptive_enable,
            near_distance=component_near_distance,
            far_distance=component_far_distance,
            near_scale=component_near_scale,
        )
    else:
        filter_gate = filter_metric_values > float(args.linked_action_epsilon)
        filter_diag = {}

    base_diff_mask = action_delta_l1 > 1e-6
    teacher_like_base_mask = _select_teacher_like_mask(
        teacher_intervened=teacher_intervened,
        diff_mask=base_diff_mask,
        mode=str(args.teacher_mask_mode),
        num_rows=int(observations.shape[0]),
        device=device,
    )
    teacher_like_mask = teacher_like_base_mask & filter_gate
    mask_indices = torch.nonzero(teacher_like_mask).flatten()
    if mask_indices.numel() <= 0:
        raise ValueError("No teacher-like rows selected for preference probe.")
    selected_action_delta_mean = action_delta_mean_abs[mask_indices]
    selected_filter_metric = filter_metric_values[mask_indices]
    selected_weight_metric = weight_metric_values[mask_indices]
    relative_distance = _relative_distance_signal(observations, ckpt_args)
    selected_pair_weights = None
    if float(args.linked_action_weight_scale) > 0.0:
        selected_pair_weights = torch.clamp(
            selected_weight_metric / float(args.linked_action_weight_scale),
            min=0.0,
            max=1.0,
        )

    obs_dim = int(observations.shape[-1])
    act_dim = int(actions.shape[-1])
    actor_backbone, actor_head, critic_backbone, critic_heads, obs_normalizer = _build_models(
        ckpt_args=ckpt_args,
        checkpoint=checkpoint,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        load_actor_from_checkpoint=str(args.actor_init_mode).strip().lower() == "checkpoint",
    )
    if obs_normalizer is None:
        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        obs_normalizer.eval()
    critic_backbone.train()
    critic_heads.train()
    for param in list(critic_backbone.parameters()) + list(critic_heads.parameters()):
        param.requires_grad_(True)

    critic_optimizer = torch.optim.AdamW(
        list(critic_backbone.parameters()) + list(critic_heads.parameters()),
        lr=float(args.critic_lr),
        weight_decay=float(args.weight_decay),
    )
    actor_optimizer = torch.optim.AdamW(
        list(actor_backbone.parameters()) + list(actor_head.parameters()),
        lr=float(args.actor_lr),
        weight_decay=float(args.weight_decay),
    )

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (REPO_ROOT / "models" / "offline_actor_recover" / args.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    best_path = output_dir / "best_checkpoint.pt"

    wandb_run = _maybe_init_wandb(
        args,
        config={
            **vars(args),
            "checkpoint_path": str(checkpoint_path),
            "dataset_path": str(dataset_path),
            "dataset_rows": int(observations.shape[0]),
            "teacher_mask_rows": int(mask_indices.numel()),
            "linked_action_filter_mode": filter_mode,
            "linked_action_filter_metric": str(args.linked_action_filter_metric),
            "linked_action_weight_metric": str(args.linked_action_weight_metric),
            "linked_action_scope": str(args.linked_action_scope),
            "linked_action_epsilon": float(args.linked_action_epsilon),
            "linked_action_weight_scale": float(args.linked_action_weight_scale),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
        },
    )

    def _log_payload(payload: dict[str, float], *, step: int) -> None:
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        if wandb_run is not None:
            wandb_run.log(payload, step=step)

    with torch.no_grad():
        obs_pref = obs_normalizer(observations[mask_indices])
        feat_pref = critic_backbone(obs_pref)
        q_teacher = torch.stack(critic_heads(feat_pref, actions[mask_indices]), dim=0)
        q_student = torch.stack(critic_heads(feat_pref, student_actions[mask_indices]), dim=0)
        initial_q_teacher_mean = float(q_teacher.mean().item())
        initial_q_student_mean = float(q_student.mean().item())
        initial_q_delta = float((q_teacher - q_student).mean().item())

    initial_eval = _eval_metrics(
        args_ns=args_ns,
        device=device,
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        obs_normalizer=obs_normalizer,
    )
    initial_payload = {
        "phase": "initial",
        "step": 0,
        "Offline/dataset_rows": float(observations.shape[0]),
        "Offline/teacher_mask_rows": float(mask_indices.numel()),
        "Offline/teacher_like_fraction": float(teacher_like_mask.float().mean().item()),
        "Offline/teacher_like_base_fraction": float(teacher_like_base_mask.float().mean().item()),
        "Offline/teacher_flag_fraction": float(teacher_intervened.float().mean().item()) if teacher_intervened is not None else 0.0,
        "Offline/action_diff_fraction": float(base_diff_mask.float().mean().item()),
        "Offline/filter_pass_fraction": float(filter_gate.float().mean().item()),
        "Offline/action_delta_mean_selected": float(selected_action_delta_mean.mean().item()),
        "Offline/filter_metric_mean_selected": float(selected_filter_metric.mean().item()),
        "Offline/weight_metric_mean_selected": float(selected_weight_metric.mean().item()),
        "Offline/pair_weight_mean_selected": float(selected_pair_weights.mean().item()) if selected_pair_weights is not None else 1.0,
        "Offline/relative_distance_mean_selected": float(relative_distance[mask_indices].mean().item()) if torch.isfinite(relative_distance[mask_indices]).any() else float("nan"),
        "Offline/gate_angle_threshold_deg": float(angle_threshold_deg),
        "Offline/gate_component_xyz_threshold": float(component_xyz_threshold),
        "Offline/gate_component_yaw_threshold": float(component_yaw_threshold),
        "Offline/gate_component_gripper_threshold": float(component_gripper_threshold),
        "Pref/q_teacher_mean": initial_q_teacher_mean,
        "Pref/q_student_mean": initial_q_student_mean,
        "Pref/q_delta_mean": initial_q_delta,
        **{f"Eval/{k}": float(v) for k, v in initial_eval.items()},
    }
    if "angle_deg" in filter_diag:
        initial_payload["Offline/angle_deg_mean_selected"] = float(filter_diag["angle_deg"][mask_indices].mean().item())
    if "scale" in filter_diag:
        initial_payload["Offline/component_scale_mean_selected"] = float(filter_diag["scale"][mask_indices].mean().item())
        initial_payload["Offline/component_xyz_l2_mean_selected"] = float(filter_diag["xyz_l2"][mask_indices].mean().item())
        initial_payload["Offline/component_gripper_abs_mean_selected"] = float(filter_diag["gripper_abs"][mask_indices].mean().item())
    _log_payload(initial_payload, step=0)
    print(
        "[PrefProbeInit] "
        f"teacher_like_frac={float(teacher_like_mask.float().mean().item()):.4f} "
        f"q_teacher_mean={initial_q_teacher_mean:.4f} "
        f"q_student_mean={initial_q_student_mean:.4f} "
        f"q_delta_mean={initial_q_delta:.4f} "
        f"{_format_metrics(initial_eval)}",
        flush=True,
    )

    best_metrics = dict(initial_eval)
    best_success = float(initial_eval.get("success_rate", 0.0))
    best_ckpt = dict(checkpoint)
    best_ckpt["actor_backbone"] = actor_backbone.state_dict()
    best_ckpt["actor_head"] = actor_head.state_dict()
    best_ckpt["critic_backbone"] = critic_backbone.state_dict()
    best_ckpt["critic_heads"] = critic_heads.state_dict()
    if obs_normalizer is not None:
        best_ckpt["obs_normalizer_state"] = obs_normalizer.state_dict()
    torch.save(best_ckpt, best_path)

    start_time = time.time()
    for critic_step in range(1, int(args.critic_steps) + 1):
        sampled_mask_idx = torch.randint(0, mask_indices.numel(), (int(args.batch_size),), device=device)
        idx = mask_indices[sampled_mask_idx]
        obs_batch = obs_normalizer(observations[idx])
        feat_batch = critic_backbone(obs_batch)
        q_teacher_batch = torch.stack(critic_heads(feat_batch, actions[idx]), dim=0)
        q_student_batch = torch.stack(critic_heads(feat_batch, student_actions[idx]), dim=0)
        batch_pair_weights = selected_pair_weights[sampled_mask_idx] if selected_pair_weights is not None else None
        pref_loss = float(args.pref_weight) * _preference_loss(
            q_teacher_batch,
            q_student_batch,
            loss_type=str(args.pref_loss_type),
            margin=float(args.pref_margin),
            sample_weights=batch_pair_weights,
        )
        critic_optimizer.zero_grad(set_to_none=True)
        pref_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(critic_backbone.parameters()) + list(critic_heads.parameters()), max_norm=10.0)
        critic_optimizer.step()

        if critic_step == 1 or critic_step % max(1, int(args.eval_interval)) == 0 or critic_step == int(args.critic_steps):
            with torch.no_grad():
                obs_pref = obs_normalizer(observations[mask_indices])
                feat_pref = critic_backbone(obs_pref)
                q_teacher = torch.stack(critic_heads(feat_pref, actions[mask_indices]), dim=0)
                q_student = torch.stack(critic_heads(feat_pref, student_actions[mask_indices]), dim=0)
            payload = {
                "phase": "critic",
                "step": int(critic_step),
                "Train/pref_loss": float(pref_loss.detach().item()),
                "Train/pair_weight_mean": float(batch_pair_weights.mean().item()) if batch_pair_weights is not None else 1.0,
                "Offline/teacher_like_fraction": float(teacher_like_mask.float().mean().item()),
                "Offline/teacher_like_base_fraction": float(teacher_like_base_mask.float().mean().item()),
                "Offline/filter_pass_fraction": float(filter_gate.float().mean().item()),
                "Pref/q_teacher_mean": float(q_teacher.mean().item()),
                "Pref/q_student_mean": float(q_student.mean().item()),
                "Pref/q_delta_mean": float((q_teacher - q_student).mean().item()),
                "Perf/wall_s": float(time.time() - start_time),
            }
            _log_payload(payload, step=int(critic_step))
            print(
                f"[PrefProbeCritic] step={critic_step} pref_loss={payload['Train/pref_loss']:.4f} "
                f"q_teacher_mean={payload['Pref/q_teacher_mean']:.4f} "
                f"q_student_mean={payload['Pref/q_student_mean']:.4f} "
                f"q_delta_mean={payload['Pref/q_delta_mean']:.4f}",
                flush=True,
            )

    for param in list(critic_backbone.parameters()) + list(critic_heads.parameters()):
        param.requires_grad_(False)
    critic_backbone.eval()
    critic_heads.eval()

    actor_eval = _eval_metrics(
        args_ns=args_ns,
        device=device,
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        obs_normalizer=obs_normalizer,
    )
    _log_payload(
        {
            "phase": "actor",
            "step": 0,
            **{f"Eval/{k}": float(v) for k, v in actor_eval.items()},
        },
        step=int(args.critic_steps),
    )
    print(f"[PrefProbeActor] step=0 {_format_metrics(actor_eval)}", flush=True)

    for actor_step in range(1, int(args.actor_steps) + 1):
        idx = torch.randint(0, observations.shape[0], (int(args.batch_size),), device=device)
        obs_batch = observations[idx]
        obs_norm = obs_normalizer(obs_batch)
        features_actor = actor_backbone(obs_norm)
        _, _, mean_actions = actor_head(features_actor)
        with torch.no_grad():
            features_critic = critic_backbone(obs_norm)
        q_values = torch.stack(critic_heads(features_critic, mean_actions), dim=0)
        min_q = q_values.min(dim=0).values
        actor_q_loss = float(args.actor_q_weight) * (-min_q.mean())
        action_l2 = mean_actions.pow(2).mean()
        loss = actor_q_loss + float(args.action_l2_weight) * action_l2

        bc_teacher_loss_value = 0.0
        if float(args.bc_teacher_weight) > 0.0:
            teacher_idx = idx[teacher_like_mask[idx]]
            if teacher_idx.numel() > 0:
                teacher_obs = obs_normalizer(observations[teacher_idx])
                _, _, teacher_mean = actor_head(actor_backbone(teacher_obs))
                bc_teacher_loss = F.mse_loss(teacher_mean, actions[teacher_idx])
                loss = loss + float(args.bc_teacher_weight) * bc_teacher_loss
                bc_teacher_loss_value = float(bc_teacher_loss.detach().item())

        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(actor_backbone.parameters()) + list(actor_head.parameters()), max_norm=10.0)
        actor_optimizer.step()

        if actor_step == 1 or actor_step % max(1, int(args.eval_interval)) == 0 or actor_step == int(args.actor_steps):
            eval_metrics = _eval_metrics(
                args_ns=args_ns,
                device=device,
                actor_backbone=actor_backbone,
                actor_head=actor_head,
                obs_normalizer=obs_normalizer,
            )
            with torch.no_grad():
                sample_n = min(4096, observations.shape[0])
                q_idx = torch.randperm(observations.shape[0], device=device)[:sample_n]
                obs_eval = obs_normalizer(observations[q_idx])
                _, _, mean_eval = actor_head(actor_backbone(obs_eval))
                q_actor = torch.stack(critic_heads(critic_backbone(obs_eval), mean_eval), dim=0).min(dim=0).values
            payload = {
                "phase": "actor",
                "step": int(actor_step),
                "Train/loss": float(loss.detach().item()),
                "Train/actor_q_loss": float(actor_q_loss.detach().item()),
                "Train/action_l2": float(action_l2.detach().item()),
                "Train/bc_teacher_loss": float(bc_teacher_loss_value),
                "Diag/q_actor_mean": float(q_actor.mean().item()),
                "Perf/wall_s": float(time.time() - start_time),
                **{f"Eval/{k}": float(v) for k, v in eval_metrics.items()},
            }
            _log_payload(payload, step=int(args.critic_steps) + int(actor_step))
            print(
                f"[PrefProbeActor] step={actor_step} loss={payload['Train/loss']:.4f} "
                f"actor_q_loss={payload['Train/actor_q_loss']:.4f} "
                f"q_actor_mean={payload['Diag/q_actor_mean']:.4f} "
                f"{_format_metrics(eval_metrics)}",
                flush=True,
            )
            success = float(eval_metrics.get("success_rate", 0.0))
            place_rate = float(eval_metrics.get("placed_episode_rate", 0.0))
            if (success > best_success + 1e-6) or (abs(success - best_success) <= 1e-6 and place_rate > float(best_metrics.get("placed_episode_rate", 0.0))):
                best_success = success
                best_metrics = dict(eval_metrics)
                best_ckpt["actor_backbone"] = actor_backbone.state_dict()
                best_ckpt["actor_head"] = actor_head.state_dict()
                best_ckpt["critic_backbone"] = critic_backbone.state_dict()
                best_ckpt["critic_heads"] = critic_heads.state_dict()
                torch.save(best_ckpt, best_path)
                print(f"[PrefProbeBest] step={actor_step} saved={best_path}", flush=True)

    summary = {
        "dataset_rows": int(observations.shape[0]),
        "teacher_mask_rows": int(mask_indices.numel()),
        "teacher_mask_fraction": float(teacher_like_mask.float().mean().item()),
        "teacher_mask_base_fraction": float(teacher_like_base_mask.float().mean().item()),
        "filter_pass_fraction": float(filter_gate.float().mean().item()),
        "linked_action_filter_mode": filter_mode,
        "linked_action_filter_metric": str(args.linked_action_filter_metric),
        "linked_action_weight_metric": str(args.linked_action_weight_metric),
        "linked_action_scope": str(args.linked_action_scope),
        "linked_action_epsilon": float(args.linked_action_epsilon),
        "linked_action_weight_scale": float(args.linked_action_weight_scale),
        "selected_action_delta_mean": float(selected_action_delta_mean.mean().item()),
        "selected_filter_metric_mean": float(selected_filter_metric.mean().item()),
        "selected_weight_metric_mean": float(selected_weight_metric.mean().item()),
        "selected_pair_weight_mean": float(selected_pair_weights.mean().item()) if selected_pair_weights is not None else 1.0,
        "initial_q_teacher_mean": initial_q_teacher_mean,
        "initial_q_student_mean": initial_q_student_mean,
        "initial_q_delta": initial_q_delta,
        "best_metrics": best_metrics,
        "best_checkpoint_path": str(best_path),
        "metrics_path": str(metrics_path),
    }
    if "angle_deg" in filter_diag:
        summary["selected_angle_deg_mean"] = float(filter_diag["angle_deg"][mask_indices].mean().item())
    if "scale" in filter_diag:
        summary["selected_component_scale_mean"] = float(filter_diag["scale"][mask_indices].mean().item())
        summary["selected_component_xyz_l2_mean"] = float(filter_diag["xyz_l2"][mask_indices].mean().item())
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[PrefProbeDone] summary={summary_path}", flush=True)
    if wandb_run is not None:
        if hasattr(wandb_run, "summary"):
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    wandb_run.summary[key] = value
            wandb_run.summary["best_checkpoint_path"] = str(best_path)
            for key, value in best_metrics.items():
                wandb_run.summary[f"best/{key}"] = float(value)
        wandb_run.finish()


if __name__ == "__main__":
    main()
