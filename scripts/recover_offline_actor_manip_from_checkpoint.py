#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FASTTD3_ROOT = REPO_ROOT / "fasttd3"
if str(FASTTD3_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTTD3_ROOT))
FAST_SAC_ROOT = REPO_ROOT / "fasttd3" / "fast_sac"
if "fast_sac_utils" not in sys.modules:
    spec = importlib.util.spec_from_file_location("fast_sac_utils", FAST_SAC_ROOT / "fast_sac_utils.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load fast_sac_utils from {FAST_SAC_ROOT / 'fast_sac_utils.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fast_sac_utils"] = module
    spec.loader.exec_module(module)

from fast_sac_utils import EmpiricalNormalization  # noqa: E402

from ogbench_utils import CriticEnsemble, GaussianPolicyHead, MLPBackbone  # noqa: E402
from ogbench_utils.fastsac_ogbench_loop import run_eval_metrics  # noqa: E402
from ogbench_utils.fastsac_ogbench_manip_env import build_manip_eval_environment  # noqa: E402
from ogbench_utils.fastsac_ogbench_types import AMPComponents  # noqa: E402


def _auto_device(requested: str) -> torch.device:
    mode = str(requested or "auto").strip().lower()
    if mode == "cuda":
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _coerce_args_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None:
        return {}
    if hasattr(raw, "__dict__"):
        return dict(vars(raw))
    return {}


def _namespace_from_checkpoint_args(args_dict: dict[str, Any]) -> SimpleNamespace:
    merged = dict(args_dict)
    merged.setdefault("num_envs", 1)
    merged.setdefault("eval_num_envs", 5)
    merged.setdefault("num_eval_episodes", 10)
    merged.setdefault("reward_switch_after_steps", 0)
    merged.setdefault("reward_type", "sparse")
    merged.setdefault("cube_reward_mode", "dense")
    merged.setdefault("include_goal", False)
    merged.setdefault("include_distance", False)
    merged.setdefault("include_direction", False)
    merged.setdefault("include_velocity", False)
    merged.setdefault("include_relative_cube_features", True)
    merged.setdefault("relative_only_obs", True)
    merged.setdefault("disable_rotation", True)
    merged.setdefault("binary_gripper_actions", False)
    merged.setdefault("binary_gripper_threshold", 0.0)
    merged.setdefault("hard_gripper_intervention", False)
    merged.setdefault("gripper_intervene_pick_radius", 0.06)
    merged.setdefault("gripper_intervene_place_radius", 0.08)
    merged.setdefault("gripper_intervene_contact_threshold", 0.02)
    merged.setdefault("hard_block_lethal", False)
    merged.setdefault("intervention_enable_after_steps", 0)
    merged.setdefault("intervention_safety_margin_frac", 0.0)
    merged.setdefault("intervention_release_steps", 0)
    merged.setdefault("intervention_reward_patience_steps", 5)
    merged.setdefault("intervention_reward_improvement_epsilon", 0.0)
    merged.setdefault("intervention_episode_prob", 1.0)
    merged.setdefault("intervention_episode_prob_min", 1.0)
    merged.setdefault("intervention_episode_prob_decay_steps", 0)
    merged.setdefault("intervention_episode_prob_decay_start", 0)
    merged.setdefault("intervention_episode_prob_seed", 0)
    merged.setdefault("teacher_target_mode", "sequential")
    merged.setdefault("cube_success_tolerance", 0.04)
    merged.setdefault("tolerance_channel_weights", None)
    merged.setdefault("tolerance_xyz_value", -1.0)
    merged.setdefault("tolerance_yaw_value", -1.0)
    merged.setdefault("tolerance_gripper_value", -1.0)
    merged.setdefault("tolerance_adaptive_enable", True)
    merged.setdefault("tolerance_adaptive_near_distance", 0.08)
    merged.setdefault("tolerance_adaptive_far_distance", 0.30)
    merged.setdefault("tolerance_adaptive_near_scale", 0.35)
    merged.setdefault("dense_reward_scale", 1.0)
    merged.setdefault("step_penalty", 0.0)
    merged.setdefault("cube_success_reward", 1.0)
    merged.setdefault("cube_subgoal_grasp_reward", 0.25)
    merged.setdefault("cube_subgoal_place_reward", 1.0)
    merged.setdefault("cube_subgoal_drop_penalty", 0.0)
    merged.setdefault("cube_subgoal_grasp_error_threshold", 0.08)
    merged.setdefault("cube_dense_progress_scale", 1.0)
    merged.setdefault("cube_dense_progress_clip", 0.0)
    merged.setdefault("teacher_type", "cube_markov")
    merged.setdefault("train_render_mode", "none")
    merged.setdefault("eval_render_mode", "none")
    merged.setdefault("visualize_intervention_colors", False)
    merged.setdefault("hold_targets_on_zero_action", False)
    merged.setdefault("noop_action_threshold", 1e-6)
    merged.setdefault("static_reset_seed", None)
    return SimpleNamespace(**merged)


def _build_models(
    *,
    ckpt_args: dict[str, Any],
    checkpoint: dict[str, Any],
    obs_dim: int,
    act_dim: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module, EmpiricalNormalization | None]:
    use_layer_norm = bool(ckpt_args.get("use_layer_norm", False))
    layer_norm_eps = float(ckpt_args.get("layer_norm_eps", 1e-5))
    init_scale = float(ckpt_args.get("init_scale", 0.01))
    actor_hidden_dim = int(ckpt_args.get("actor_hidden_dim", 512))
    critic_hidden_dim = int(ckpt_args.get("critic_hidden_dim", 1024))
    num_critics = int(ckpt_args.get("num_critics", 2))
    arch_shared_trunk = bool(ckpt_args.get("arch_shared_trunk", False))
    shared_hidden_dim = int(ckpt_args.get("shared_hidden_dim", actor_hidden_dim))

    if arch_shared_trunk:
        feature_dim = shared_hidden_dim
        actor_backbone = MLPBackbone(
            obs_dim,
            feature_dim,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
        ).to(device)
        critic_backbone = actor_backbone
    else:
        actor_backbone = MLPBackbone(
            obs_dim,
            actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
        ).to(device)
        critic_backbone = MLPBackbone(
            obs_dim,
            critic_hidden_dim,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
        ).to(device)

    actor_head = GaussianPolicyHead(
        actor_backbone.output_dim,
        act_dim,
        actor_hidden_dim,
        init_scale,
        use_layer_norm=use_layer_norm,
        layer_norm_eps=layer_norm_eps,
    ).to(device)
    critic_heads = CriticEnsemble(
        critic_backbone.output_dim,
        act_dim,
        critic_hidden_dim,
        num_critics,
        use_layer_norm=use_layer_norm,
        layer_norm_eps=layer_norm_eps,
    ).to(device)

    actor_backbone.load_state_dict(checkpoint["actor_backbone"])
    actor_head.load_state_dict(checkpoint["actor_head"])
    critic_heads.load_state_dict(checkpoint["critic_heads"])
    if arch_shared_trunk:
        actor_backbone.load_state_dict(checkpoint["shared_backbone"])
    else:
        critic_backbone.load_state_dict(checkpoint["critic_backbone"])

    obs_normalizer = None
    obs_norm_state = checkpoint.get("obs_normalizer_state")
    if isinstance(obs_norm_state, dict):
        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        obs_normalizer.load_state_dict(obs_norm_state)
        obs_normalizer.eval()

    actor_backbone.train()
    actor_head.train()
    critic_backbone.eval()
    critic_heads.eval()
    for module in (critic_backbone, critic_heads):
        for param in module.parameters():
            param.requires_grad_(False)

    return actor_backbone, actor_head, critic_backbone, critic_heads, obs_normalizer


def _eval_metrics(
    *,
    args_ns: SimpleNamespace,
    device: torch.device,
    actor_backbone: torch.nn.Module,
    actor_head: torch.nn.Module,
    obs_normalizer: EmpiricalNormalization | None,
) -> dict[str, float]:
    if obs_normalizer is None:
        obs_space = build_manip_eval_environment(args_ns, device).observation_space
        obs_dim = int(np.prod(obs_space.shape))
        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        obs_normalizer.eval()
    eval_envs = build_manip_eval_environment(args_ns, device)
    amp = AMPComponents(enabled=False, device_type=device.type, dtype=torch.float32, scaler=torch.amp.GradScaler(enabled=False))
    try:
        metrics = run_eval_metrics(
            args=args_ns,
            device=device,
            eval_envs=eval_envs,
            actor_backbone=actor_backbone,
            actor_head=actor_head,
            obs_normalizer=obs_normalizer,
            amp=amp,
        )
    finally:
        eval_envs.close()
    return metrics


def _format_metrics(metrics: dict[str, float]) -> str:
    keys = [
        "avg_return",
        "avg_length",
        "success_rate",
        "avg_success_length",
        "placed_episode_rate",
        "grasped_episode_rate",
        "avg_episode_place_events",
        "avg_dense_phase_cumulative_episode",
        "avg_cube_max_target_error",
        "timeout_rate",
    ]
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return ", ".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline actor recovery from a saved manip FastSAC checkpoint and replay dataset.")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num_gradient_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--action_l2_weight", type=float, default=1e-3)
    p.add_argument("--bc_teacher_weight", type=float, default=0.0)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--eval_num_envs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--name", type=str, default="offline_actor_recover")
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
    student_actions = torch.as_tensor(np.asarray(data["student_actions"], dtype=np.float32), device=device)
    teacher_intervened = torch.as_tensor(np.asarray(data["teacher_intervened"], dtype=np.bool_), device=device)
    diff_mask = (torch.abs(actions - student_actions).sum(dim=-1) > 1e-6)
    teacher_like_mask = diff_mask if int(teacher_intervened.sum().item()) <= 0 else teacher_intervened | diff_mask

    obs_dim = int(observations.shape[-1])
    act_dim = int(actions.shape[-1])
    actor_backbone, actor_head, critic_backbone, critic_heads, obs_normalizer = _build_models(
        ckpt_args=ckpt_args,
        checkpoint=checkpoint,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )
    if obs_normalizer is None:
        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        obs_normalizer.eval()

    optimizer = torch.optim.AdamW(
        list(actor_backbone.parameters()) + list(actor_head.parameters()),
        lr=float(args.actor_lr),
        weight_decay=float(args.weight_decay),
    )

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (REPO_ROOT / "models" / "offline_actor_recover" / args.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_checkpoint.pt"
    metrics_path = output_dir / "metrics.jsonl"

    def checkpoint_q_stats() -> dict[str, float]:
        with torch.no_grad():
            sample_n = min(4096, observations.shape[0])
            idx = torch.randperm(observations.shape[0], device=device)[:sample_n]
            obs_batch = observations[idx]
            obs_norm = obs_normalizer(obs_batch)
            _, _, mean_actions = actor_head(actor_backbone(obs_norm))
            features = critic_backbone(obs_norm)
            q_actor = torch.stack(critic_heads(features, mean_actions), dim=0).min(dim=0).values
            q_exec = torch.stack(critic_heads(features, actions[idx]), dim=0).min(dim=0).values
            q_student = torch.stack(critic_heads(features, student_actions[idx]), dim=0).min(dim=0).values
            return {
                "q_actor_mean": float(q_actor.mean().item()),
                "q_exec_mean": float(q_exec.mean().item()),
                "q_student_mean": float(q_student.mean().item()),
            }

    initial_q = checkpoint_q_stats()
    print(
        "[RecoverInit] "
        f"dataset_rows={int(observations.shape[0])} "
        f"teacher_flag_frac={float(teacher_intervened.float().mean().item()):.4f} "
        f"action_diff_frac={float(diff_mask.float().mean().item()):.4f} "
        f"teacher_like_frac={float(teacher_like_mask.float().mean().item()):.4f} "
        f"q_actor_mean={initial_q['q_actor_mean']:.4f} "
        f"q_exec_mean={initial_q['q_exec_mean']:.4f} "
        f"q_student_mean={initial_q['q_student_mean']:.4f}",
        flush=True,
    )

    initial_eval = _eval_metrics(
        args_ns=args_ns,
        device=device,
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        obs_normalizer=obs_normalizer,
    )
    print(f"[RecoverEval] step=0 {_format_metrics(initial_eval)}", flush=True)

    best_success = float(initial_eval.get("success_rate", 0.0))
    best_metrics = dict(initial_eval)
    best_ckpt = dict(checkpoint)

    start_time = time.time()
    for step in range(1, int(args.num_gradient_steps) + 1):
        idx = torch.randint(0, observations.shape[0], (int(args.batch_size),), device=device)
        obs_batch = observations[idx]
        obs_norm = obs_normalizer(obs_batch)
        features_actor = actor_backbone(obs_norm)
        _, _, mean_actions = actor_head(features_actor)
        with torch.no_grad():
            features_critic = critic_backbone(obs_norm)
        q_values = torch.stack(critic_heads(features_critic, mean_actions), dim=0)
        min_q = q_values.min(dim=0).values
        actor_loss = -min_q.mean()
        action_l2 = mean_actions.pow(2).mean()
        loss = actor_loss + float(args.action_l2_weight) * action_l2

        bc_teacher_loss_value = 0.0
        if float(args.bc_teacher_weight) > 0.0:
            teacher_idx = idx[teacher_like_mask[idx]]
            if teacher_idx.numel() > 0:
                teacher_obs = observations[teacher_idx]
                teacher_obs_norm = obs_normalizer(teacher_obs)
                _, _, teacher_mean = actor_head(actor_backbone(teacher_obs_norm))
                teacher_target = actions[teacher_idx]
                bc_teacher_loss = F.mse_loss(teacher_mean, teacher_target)
                loss = loss + float(args.bc_teacher_weight) * bc_teacher_loss
                bc_teacher_loss_value = float(bc_teacher_loss.detach().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(actor_backbone.parameters()) + list(actor_head.parameters()), max_norm=10.0)
        optimizer.step()

        if step == 1 or step % max(1, int(args.eval_interval)) == 0 or step == int(args.num_gradient_steps):
            eval_metrics = _eval_metrics(
                args_ns=args_ns,
                device=device,
                actor_backbone=actor_backbone,
                actor_head=actor_head,
                obs_normalizer=obs_normalizer,
            )
            q_stats = checkpoint_q_stats()
            wall_s = time.time() - start_time
            payload = {
                "step": int(step),
                "wall_s": float(wall_s),
                "loss": float(loss.detach().item()),
                "actor_q_loss": float(actor_loss.detach().item()),
                "action_l2": float(action_l2.detach().item()),
                "bc_teacher_loss": float(bc_teacher_loss_value),
                **q_stats,
                **eval_metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
            print(
                f"[RecoverEval] step={step} loss={payload['loss']:.4f} "
                f"actor_q_loss={payload['actor_q_loss']:.4f} action_l2={payload['action_l2']:.4f} "
                f"q_actor_mean={payload['q_actor_mean']:.4f} q_exec_mean={payload['q_exec_mean']:.4f} "
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
                torch.save(best_ckpt, best_path)
                print(f"[RecoverBest] step={step} saved={best_path}", flush=True)

    summary = {
        "best_metrics": best_metrics,
        "best_checkpoint_path": str(best_path if best_path.exists() else ""),
        "metrics_path": str(metrics_path),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[RecoverDone] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
