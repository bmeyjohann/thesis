#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FAST_SAC_ROOT = REPO_ROOT / "fasttd3" / "fast_sac"
if str(FAST_SAC_ROOT) not in sys.path:
    sys.path.insert(0, str(FAST_SAC_ROOT))

from fast_sac_utils import EmpiricalNormalization

from ogbench_utils.bc_ogbench_manip_train import (
    BehaviorCloningPolicy,
    _prepare_bc_run_dirs,
    _run_bc_eval,
    _save_bc_checkpoint,
)
from ogbench_utils.cli import build_train_parser
from ogbench_utils.fastsac_ogbench_loop import run_eval_metrics
from ogbench_utils.fastsac_ogbench_manip_env import build_manip_environment, build_manip_eval_environment
from ogbench_utils.fastsac_ogbench_setup import (
    build_updater_from_components,
    create_replay_buffer,
    initialize_amp,
    initialize_buffers,
    initialize_models,
    select_device,
)
from ogbench_utils.fastsac_ogbench_types import AMPComponents
from ogbench_utils.hgdagger_ogbench_manip_train import (
    HGDaggerEnsemble,
    _prepare_run_dirs as _prepare_hg_run_dirs,
    _run_hg_eval,
    _save_hg_checkpoint,
)
from ogbench_utils.logging import CheckpointManager
from ogbench_utils.manip_dataset_io import load_transition_dataset_metadata
from ogbench_utils.pvp_td3_ogbench_manip_train import (
    PVPReplayBuffer,
    TD3Actor,
    TwinCritic,
    _prepare_run_dirs as _prepare_pvp_run_dirs,
    _run_pvp_eval,
    _sample_balanced_batch,
    _save_pvp_checkpoint,
)


OFFLINE_METHODS = (
    "bc",
    "hg_dagger",
    "pvp",
    "eil",
    "hilserl",
    "own",
    "own_nogate",
)


def _default_wandb_mode() -> str:
    explicit = str(os.environ.get("WANDB_MODE", "")).strip()
    if explicit:
        return explicit
    cluster_markers = ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_JOB_NODELIST")
    on_cluster = any(str(os.environ.get(key, "")).strip() for key in cluster_markers)
    return "offline" if on_cluster else "online"


def _bool_from_any(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _parse_args() -> argparse.Namespace:
    parser = build_train_parser()
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--eval_num_envs", type=int, default=5)
    parser.add_argument("--method", type=str, required=True, choices=OFFLINE_METHODS)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--offline_updates", type=int, default=2_000)
    parser.add_argument(
        "--teacher_mask_mode",
        type=str,
        default="effective",
        choices=["raw", "diff", "effective", "all"],
        help="How to define teacher-like rows from the fixed offline dataset.",
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--entity", type=str, default="")
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="", choices=["", "online", "offline", "disabled"])
    parser.add_argument(
        "--eil_use_teacher_rows_as_good",
        action="store_true",
        default=True,
        help="Keep intervened rows marked as good in the offline EIL reconstruction.",
    )
    parser.add_argument(
        "--no_eil_use_teacher_rows_as_good",
        dest="eil_use_teacher_rows_as_good",
        action="store_false",
    )
    args = parser.parse_args()

    if not getattr(args, "exp_name", ""):
        args.exp_name = str(args.name or f"offline_{args.method}_{time.strftime('%Y%m%d_%H%M%S')}")
    if not getattr(args, "name", ""):
        args.name = str(args.exp_name)
    if not getattr(args, "project", ""):
        args.project = "ogbench-manip-offline"
    return args


@dataclass
class OfflineDataset:
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    truncations: torch.Tensor
    student_actions: torch.Tensor
    teacher_intervened: torch.Tensor
    diff_mask: torch.Tensor
    teacher_mask: torch.Tensor
    metadata: dict[str, Any]

    @property
    def num_rows(self) -> int:
        return int(self.observations.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def act_dim(self) -> int:
        return int(self.actions.shape[-1])


def _select_teacher_like_mask(
    *,
    teacher_intervened: torch.Tensor,
    diff_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    mode_key = str(mode).strip().lower()
    if mode_key == "all":
        return torch.ones_like(teacher_intervened, dtype=torch.bool)
    if mode_key == "raw":
        return teacher_intervened.to(torch.bool)
    if mode_key == "diff":
        return diff_mask.to(torch.bool)
    return teacher_intervened.to(torch.bool) | diff_mask.to(torch.bool)


def _load_dataset(*, path: str, device: torch.device, teacher_mask_mode: str) -> OfflineDataset:
    dataset_path = Path(path).expanduser()
    metadata = load_transition_dataset_metadata(dataset_path)
    with np.load(dataset_path, allow_pickle=True) as data:
        observations = torch.as_tensor(np.asarray(data["observations"], dtype=np.float32), device=device)
        actions = torch.as_tensor(np.asarray(data["actions"], dtype=np.float32), device=device)
        next_observations = torch.as_tensor(np.asarray(data["next_observations"], dtype=np.float32), device=device)
        rewards = torch.as_tensor(np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1), device=device)
        dones = torch.as_tensor(np.asarray(data["dones"], dtype=np.bool_).reshape(-1), device=device)
        truncations = torch.as_tensor(np.asarray(data["truncations"], dtype=np.bool_).reshape(-1), device=device)
        if "student_actions" in data.files:
            student_actions = torch.as_tensor(np.asarray(data["student_actions"], dtype=np.float32), device=device)
        else:
            student_actions = actions.clone()
        if "teacher_intervened" in data.files:
            teacher_intervened = torch.as_tensor(np.asarray(data["teacher_intervened"], dtype=np.bool_).reshape(-1), device=device)
        else:
            teacher_intervened = torch.zeros(actions.shape[0], dtype=torch.bool, device=device)

    diff_mask = (torch.abs(actions - student_actions).sum(dim=-1) > 1e-6)
    teacher_mask = _select_teacher_like_mask(
        teacher_intervened=teacher_intervened,
        diff_mask=diff_mask,
        mode=teacher_mask_mode,
    )
    return OfflineDataset(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
        truncations=truncations,
        student_actions=student_actions,
        teacher_intervened=teacher_intervened,
        diff_mask=diff_mask,
        teacher_mask=teacher_mask,
        metadata=metadata,
    )


def _fit_obs_normalizer(
    *,
    obs_normalizer: EmpiricalNormalization,
    observations: torch.Tensor,
    next_observations: torch.Tensor,
    chunk_size: int = 2048,
) -> None:
    was_training = obs_normalizer.training
    obs_normalizer.train()
    for source in (observations, next_observations):
        for start in range(0, int(source.shape[0]), int(max(1, chunk_size))):
            obs_normalizer.update(source[start : start + chunk_size])
    obs_normalizer.eval()
    if was_training:
        obs_normalizer.train()


def _maybe_apply_dataset_metadata_defaults(args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    if not getattr(args, "env_name", "") and metadata.get("env_name"):
        args.env_name = str(metadata["env_name"])
    if metadata.get("disable_rotation") is not None:
        args.disable_rotation = _bool_from_any(metadata.get("disable_rotation"), bool(getattr(args, "disable_rotation", False)))
    if metadata.get("gamma") is not None:
        try:
            args.gamma = float(metadata["gamma"])
        except Exception:
            pass


def _prepare_offline_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    eval_args = argparse.Namespace(**vars(args))
    eval_args.train_render_mode = "none"
    eval_args.eval_render_mode = "none"
    eval_args.use_intervention = False
    eval_args.intervention_mode = "none"
    eval_args.visualize_intervention_colors = False
    eval_args.num_envs = 1
    return eval_args


def _ensure_output_dir(path_text: str, default_dir: Path) -> Path:
    path = Path(path_text).expanduser() if str(path_text).strip() else default_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    if not bool(getattr(args, "use_wandb", False)):
        return None
    mode = str(getattr(args, "wandb_mode", "")).strip()
    if mode:
        os.environ["WANDB_MODE"] = mode
    else:
        os.environ.setdefault("WANDB_MODE", _default_wandb_mode())
    import wandb

    init_kwargs: dict[str, Any] = {
        "project": str(getattr(args, "project", "ogbench-manip-offline")),
        "name": str(getattr(args, "name", "")),
        "config": config,
        "reinit": True,
    }
    entity = str(getattr(args, "entity", "")).strip()
    group = str(getattr(args, "group", "")).strip()
    if entity:
        init_kwargs["entity"] = entity
    if group:
        init_kwargs["group"] = group
    return wandb.init(**init_kwargs)


def _write_metrics(metrics_path: Path, payload: dict[str, Any]) -> None:
    with metrics_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _teacher_idx(dataset: OfflineDataset) -> torch.Tensor:
    idx = torch.nonzero(dataset.teacher_mask, as_tuple=False).squeeze(-1)
    if idx.numel() <= 0:
        raise ValueError("Offline dataset does not contain any teacher-like rows under the selected teacher_mask_mode.")
    return idx


def _dataset_summary(dataset: OfflineDataset) -> dict[str, float]:
    gripper_idx = dataset.act_dim - 1 if dataset.act_dim >= 4 else None
    xyz_dim = min(3, dataset.act_dim)
    delta = dataset.actions - dataset.student_actions
    xyz_delta = delta[:, :xyz_dim]
    xyz_nonzero = (torch.abs(xyz_delta).sum(dim=-1) > 1e-6)
    gripper_nonzero = (
        (torch.abs(delta[:, gripper_idx]) > 1e-6) if gripper_idx is not None else torch.zeros_like(dataset.teacher_mask)
    )
    teacher_rows = dataset.teacher_mask
    teacher_count = max(1.0, float(teacher_rows.to(torch.float32).sum().item()))
    gripper_only = teacher_rows & (~xyz_nonzero) & gripper_nonzero
    return {
        "Offline/dataset_rows": float(dataset.num_rows),
        "Offline/teacher_flag_fraction": float(dataset.teacher_intervened.float().mean().item()),
        "Offline/action_diff_fraction": float(dataset.diff_mask.float().mean().item()),
        "Offline/teacher_like_fraction": float(dataset.teacher_mask.float().mean().item()),
        "Offline/teacher_like_rows": float(dataset.teacher_mask.to(torch.int64).sum().item()),
        "Offline/intervened_gripper_only_fraction": float(gripper_only.float().sum().item() / teacher_count),
    }


def _save_fastsac_checkpoint(
    *,
    path: Path,
    args: argparse.Namespace,
    model,
    obs_normalizer,
) -> None:
    checkpoint = {
        "step": int(getattr(args, "_offline_save_step", 0)),
        "actor_backbone": model.actor_backbone.state_dict(),
        "actor_head": model.actor_head.state_dict(),
        "critic_backbone": (None if args.arch_shared_trunk else model.critic_backbone.state_dict()),
        "shared_backbone": model.actor_backbone.state_dict() if args.arch_shared_trunk else None,
        "critic_heads": model.critic_heads.state_dict(),
        "critic_target_backbone": model.critic_target_backbone.state_dict(),
        "critic_target_heads": model.critic_target_heads.state_dict(),
        "obs_normalizer_state": obs_normalizer.state_dict() if hasattr(obs_normalizer, "state_dict") else None,
        "critic_obs_normalizer_state": None,
        "log_alpha": model.log_alpha.detach().cpu().item(),
        "pixel_shape": None,
        "args": vars(args),
    }
    torch.save(checkpoint, path, _use_new_zipfile_serialization=True)


def _make_transition_tensordict(
    *,
    obs: torch.Tensor,
    action: torch.Tensor,
    student_action: torch.Tensor,
    next_obs: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    truncation: torch.Tensor,
    teacher_intervened: torch.Tensor,
    device: torch.device,
    eil_good: bool = False,
    eil_bad: bool = False,
) -> TensorDict:
    return TensorDict(
        {
            "observations": obs.unsqueeze(0).to(device=device, dtype=torch.float32),
            "actions": action.unsqueeze(0).to(device=device, dtype=torch.float32),
            "student_actions": student_action.unsqueeze(0).to(device=device, dtype=torch.float32),
            "teacher_intervened": teacher_intervened.reshape(1).to(device=device, dtype=torch.bool),
            "eil_good": torch.tensor([eil_good], device=device, dtype=torch.bool),
            "eil_bad": torch.tensor([eil_bad], device=device, dtype=torch.bool),
            "next": {
                "observations": next_obs.unsqueeze(0).to(device=device, dtype=torch.float32),
                "rewards": reward.reshape(1).to(device=device, dtype=torch.float32),
                "dones": done.reshape(1).to(device=device, dtype=torch.bool),
                "truncations": truncation.reshape(1).to(device=device, dtype=torch.bool),
            },
        },
        batch_size=(1,),
        device=device,
    )


def _reconstruct_eil_labels(
    *,
    dataset: OfflineDataset,
    bad_pre_steps: int,
    use_teacher_rows_as_good: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = dataset.num_rows
    good = torch.zeros(n, dtype=torch.bool, device=dataset.observations.device)
    bad = torch.zeros(n, dtype=torch.bool, device=dataset.observations.device)
    pending: deque[int] = deque()
    teacher_active = False

    episode_done = dataset.dones | dataset.truncations
    for idx in range(n):
        teacher_row = bool(dataset.teacher_mask[idx].item())
        if teacher_row:
            if (not teacher_active) and pending:
                split_idx = max(0, len(pending) - int(max(0, bad_pre_steps)))
                pending_list = list(pending)
                for pos, row_idx in enumerate(pending_list):
                    if pos < split_idx:
                        good[row_idx] = True
                    else:
                        bad[row_idx] = True
                pending.clear()
            if use_teacher_rows_as_good:
                good[idx] = True
            teacher_active = True
        else:
            if teacher_active:
                teacher_active = False
            pending.append(idx)
            while len(pending) > int(max(0, bad_pre_steps)):
                row_idx = pending.popleft()
                good[row_idx] = True
        if bool(episode_done[idx].item()):
            while pending:
                row_idx = pending.popleft()
                good[row_idx] = True
            teacher_active = False
    while pending:
        row_idx = pending.popleft()
        good[row_idx] = True
    return good, bad


def _run_bc_offline(
    *,
    args: argparse.Namespace,
    dataset: OfflineDataset,
    device: torch.device,
    output_dir: Path,
    metrics_path: Path,
    wandb_run,
) -> dict[str, float]:
    run_log_dir, run_model_dir, record_progress, progress_file = _prepare_bc_run_dirs(args)
    del run_log_dir
    eval_args = _prepare_offline_eval_args(args)
    eval_envs = build_manip_eval_environment(eval_args, device)
    teacher_idx = _teacher_idx(dataset)
    obs_normalizer = EmpiricalNormalization(shape=dataset.obs_dim, device=device)
    _fit_obs_normalizer(
        obs_normalizer=obs_normalizer,
        observations=dataset.observations,
        next_observations=dataset.next_observations,
    )
    obs_normalizer.eval()
    policy = BehaviorCloningPolicy(obs_dim=dataset.obs_dim, act_dim=dataset.act_dim, args=args, device=device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(args.actor_learning_rate), weight_decay=1e-5)

    best_metrics = _run_bc_eval(args=eval_args, device=device, eval_envs=eval_envs, policy=policy, obs_normalizer=obs_normalizer)
    best_success = float(best_metrics.get("success_rate", 0.0))
    dataset_logs = _dataset_summary(dataset)
    initial_payload = {
        "step": 0,
        **dataset_logs,
        **{f"Eval/{k}": float(v) for k, v in best_metrics.items()},
        "Train/bc_loss": 0.0,
    }
    _write_metrics(metrics_path, initial_payload)
    if wandb_run is not None:
        wandb_run.log(initial_payload, step=0)
    best_path = output_dir / "best_checkpoint.pt"
    final_path = output_dir / "final_checkpoint.pt"
    step_interval = max(1, int(args.eval_interval))
    log_interval = max(1, int(args.log_interval))
    save_interval = max(1, int(args.save_interval))
    try:
        init_ckpt = _save_bc_checkpoint(
            run_model_dir=run_model_dir,
            tag="step0",
            step_value=0,
            run_prefix=args.env_name.replace("-", "_"),
            policy=policy,
            obs_normalizer=obs_normalizer,
            args=args,
        )
        torch.save(torch.load(init_ckpt, map_location="cpu", weights_only=False), best_path)
        for step in range(1, int(args.offline_updates) + 1):
            batch_idx = teacher_idx[torch.randint(0, teacher_idx.numel(), (int(args.batch_size),), device=device)]
            optimizer.zero_grad(set_to_none=True)
            batch_obs = obs_normalizer(dataset.observations[batch_idx])
            pred_actions = policy.mean_actions(batch_obs)
            loss = F.mse_loss(pred_actions, dataset.actions[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=float(args.max_grad_norm) if float(args.max_grad_norm) > 0 else float("inf"),
            )
            optimizer.step()

            if step == 1 or step % log_interval == 0 or step == int(args.offline_updates):
                payload = {
                    "step": int(step),
                    **dataset_logs,
                    "Train/bc_loss": float(loss.detach().cpu().item()),
                    "Train/expert_buffer_size": float(teacher_idx.numel()),
                    "Train/teacher_like_fraction": float(dataset.teacher_mask.float().mean().item()),
                }
                _write_metrics(metrics_path, payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=step)
                print(f"[OfflineBC] step={step} bc_loss={payload['Train/bc_loss']:.6f}", flush=True)

            if step % step_interval == 0 or step == int(args.offline_updates):
                eval_metrics = _run_bc_eval(
                    args=eval_args,
                    device=device,
                    eval_envs=eval_envs,
                    policy=policy,
                    obs_normalizer=obs_normalizer,
                )
                payload = {"step": int(step), **{f"Eval/{k}": float(v) for k, v in eval_metrics.items()}}
                _write_metrics(metrics_path, payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=step)
                if float(eval_metrics.get("success_rate", 0.0)) >= best_success:
                    best_success = float(eval_metrics.get("success_rate", 0.0))
                    best_metrics = dict(eval_metrics)
                    ckpt = _save_bc_checkpoint(
                        run_model_dir=run_model_dir,
                        tag="best",
                        step_value=step,
                        run_prefix=args.env_name.replace("-", "_"),
                        policy=policy,
                        obs_normalizer=obs_normalizer,
                        args=args,
                    )
                    if ckpt != best_path:
                        torch.save(torch.load(ckpt, map_location="cpu", weights_only=False), best_path)

            if step % save_interval == 0 or step == int(args.offline_updates):
                ckpt = _save_bc_checkpoint(
                    run_model_dir=run_model_dir,
                    tag=f"step{step}",
                    step_value=step,
                    run_prefix=args.env_name.replace("-", "_"),
                    policy=policy,
                    obs_normalizer=obs_normalizer,
                    args=args,
                )
                if step == int(args.offline_updates):
                    torch.save(torch.load(ckpt, map_location="cpu", weights_only=False), final_path)
        return best_metrics
    finally:
        progress_file.close()
        eval_envs.close()


def _run_hg_offline(
    *,
    args: argparse.Namespace,
    dataset: OfflineDataset,
    device: torch.device,
    output_dir: Path,
    metrics_path: Path,
    wandb_run,
) -> dict[str, float]:
    run_log_dir, run_model_dir, record_progress, progress_file = _prepare_hg_run_dirs(args)
    del run_log_dir, record_progress
    eval_args = _prepare_offline_eval_args(args)
    eval_envs = build_manip_eval_environment(eval_args, device)
    teacher_idx = _teacher_idx(dataset)
    obs_normalizer = EmpiricalNormalization(shape=dataset.obs_dim, device=device)
    _fit_obs_normalizer(
        obs_normalizer=obs_normalizer,
        observations=dataset.observations,
        next_observations=dataset.next_observations,
    )
    obs_normalizer.eval()
    ensemble = HGDaggerEnsemble(obs_dim=dataset.obs_dim, act_dim=dataset.act_dim, args=args, device=device)
    optimizer = torch.optim.AdamW(ensemble.parameters(), lr=float(args.actor_learning_rate), weight_decay=1e-5)
    best_metrics = _run_hg_eval(args=eval_args, device=device, eval_envs=eval_envs, ensemble=ensemble, obs_normalizer=obs_normalizer)
    best_success = float(best_metrics.get("success_rate", 0.0))
    dataset_logs = _dataset_summary(dataset)
    best_path = output_dir / "best_checkpoint.pt"
    final_path = output_dir / "final_checkpoint.pt"
    initial_payload = {
        "step": 0,
        **dataset_logs,
        **{f"Eval/{k}": float(v) for k, v in best_metrics.items()},
        "Train/bc_loss": 0.0,
    }
    _write_metrics(metrics_path, initial_payload)
    if wandb_run is not None:
        wandb_run.log(initial_payload, step=0)
    step_interval = max(1, int(args.eval_interval))
    log_interval = max(1, int(args.log_interval))
    save_interval = max(1, int(args.save_interval))
    try:
        init_ckpt = _save_hg_checkpoint(
            run_model_dir=run_model_dir,
            tag="step0",
            step_value=0,
            run_prefix=args.env_name.replace("-", "_"),
            ensemble=ensemble,
            obs_normalizer=obs_normalizer,
            args=args,
            tau_estimate=None,
        )
        torch.save(torch.load(init_ckpt, map_location="cpu", weights_only=False), best_path)
        for step in range(1, int(args.offline_updates) + 1):
            batch_idx = teacher_idx[torch.randint(0, teacher_idx.numel(), (int(args.batch_size),), device=device)]
            optimizer.zero_grad(set_to_none=True)
            batch_obs = obs_normalizer(dataset.observations[batch_idx])
            member_means = ensemble.mean_actions(batch_obs)
            loss = torch.tensor(0.0, device=device)
            for member_idx in range(member_means.shape[0]):
                loss = loss + F.mse_loss(member_means[member_idx], dataset.actions[batch_idx])
            loss = loss / float(member_means.shape[0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                ensemble.parameters(),
                max_norm=float(args.max_grad_norm) if float(args.max_grad_norm) > 0 else float("inf"),
            )
            optimizer.step()

            if step == 1 or step % log_interval == 0 or step == int(args.offline_updates):
                with torch.no_grad():
                    action_var = torch.var(member_means.detach(), dim=0, unbiased=False).mean()
                payload = {
                    "step": int(step),
                    **dataset_logs,
                    "Train/bc_loss": float(loss.detach().cpu().item()),
                    "Train/expert_buffer_size": float(teacher_idx.numel()),
                    "Train/hg_action_variance_mean": float(action_var.detach().cpu().item()),
                }
                _write_metrics(metrics_path, payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=step)
                print(f"[OfflineHG] step={step} bc_loss={payload['Train/bc_loss']:.6f}", flush=True)

            if step % step_interval == 0 or step == int(args.offline_updates):
                eval_metrics = _run_hg_eval(
                    args=eval_args,
                    device=device,
                    eval_envs=eval_envs,
                    ensemble=ensemble,
                    obs_normalizer=obs_normalizer,
                )
                payload = {"step": int(step), **{f"Eval/{k}": float(v) for k, v in eval_metrics.items()}}
                _write_metrics(metrics_path, payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=step)
                if float(eval_metrics.get("success_rate", 0.0)) >= best_success:
                    best_success = float(eval_metrics.get("success_rate", 0.0))
                    best_metrics = dict(eval_metrics)
                    ckpt = _save_hg_checkpoint(
                        run_model_dir=run_model_dir,
                        tag="best",
                        step_value=step,
                        run_prefix=args.env_name.replace("-", "_"),
                        ensemble=ensemble,
                        obs_normalizer=obs_normalizer,
                        args=args,
                        tau_estimate=None,
                    )
                    if ckpt != best_path:
                        torch.save(torch.load(ckpt, map_location="cpu", weights_only=False), best_path)

            if step % save_interval == 0 or step == int(args.offline_updates):
                ckpt = _save_hg_checkpoint(
                    run_model_dir=run_model_dir,
                    tag=f"step{step}",
                    step_value=step,
                    run_prefix=args.env_name.replace("-", "_"),
                    ensemble=ensemble,
                    obs_normalizer=obs_normalizer,
                    args=args,
                    tau_estimate=None,
                )
                if step == int(args.offline_updates):
                    torch.save(torch.load(ckpt, map_location="cpu", weights_only=False), final_path)
        return best_metrics
    finally:
        progress_file.close()
        eval_envs.close()


def _populate_pvp_buffers(
    *,
    dataset: OfflineDataset,
    novice_buffer,
    human_buffer,
) -> None:
    last_teacher = False
    episode_done = dataset.dones | dataset.truncations
    for idx in range(dataset.num_rows):
        teacher_row = bool(dataset.teacher_mask[idx].item())
        intervention_start = teacher_row and (not last_teacher)
        target_buffer = human_buffer if teacher_row else novice_buffer
        target_buffer.append(
            state=dataset.observations[idx],
            action_behavior=dataset.actions[idx],
            action_novice=dataset.student_actions[idx],
            next_state=dataset.next_observations[idx],
            reward=float(dataset.rewards[idx].item()),
            done=bool(dataset.dones[idx].item()),
            intervened=teacher_row,
            intervention_start=intervention_start,
        )
        last_teacher = teacher_row
        if bool(episode_done[idx].item()):
            last_teacher = False


def _run_pvp_offline(
    *,
    args: argparse.Namespace,
    dataset: OfflineDataset,
    device: torch.device,
    output_dir: Path,
    metrics_path: Path,
    wandb_run,
) -> dict[str, float]:
    run_log_dir, run_model_dir, record_progress, progress_file = _prepare_pvp_run_dirs(args)
    del run_log_dir, record_progress
    eval_args = _prepare_offline_eval_args(args)
    eval_envs = build_manip_eval_environment(eval_args, device)
    obs_normalizer = EmpiricalNormalization(shape=dataset.obs_dim, device=device)
    _fit_obs_normalizer(
        obs_normalizer=obs_normalizer,
        observations=dataset.observations,
        next_observations=dataset.next_observations,
    )
    obs_normalizer.eval()

    actor = TD3Actor(dataset.obs_dim, dataset.act_dim, hidden_dim=int(getattr(args, "actor_hidden_dim", 256))).to(device)
    actor_target = TD3Actor(dataset.obs_dim, dataset.act_dim, hidden_dim=int(getattr(args, "actor_hidden_dim", 256))).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic = TwinCritic(dataset.obs_dim, dataset.act_dim, hidden_dim=int(getattr(args, "critic_hidden_dim", 256))).to(device)
    critic_target = TwinCritic(dataset.obs_dim, dataset.act_dim, hidden_dim=int(getattr(args, "critic_hidden_dim", 256))).to(device)
    critic_target.load_state_dict(critic.state_dict())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(args.actor_learning_rate))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(args.critic_learning_rate))
    novice_buffer = PVPReplayBuffer(
        capacity=max(1, int(dataset.num_rows)),
        obs_dim=dataset.obs_dim,
        act_dim=dataset.act_dim,
    )
    human_buffer = PVPReplayBuffer(
        capacity=max(1, int(dataset.num_rows)),
        obs_dim=dataset.obs_dim,
        act_dim=dataset.act_dim,
    )
    _populate_pvp_buffers(dataset=dataset, novice_buffer=novice_buffer, human_buffer=human_buffer)

    dataset_logs = _dataset_summary(dataset)
    best_metrics = _run_pvp_eval(args=eval_args, device=device, eval_envs=eval_envs, actor=actor, obs_normalizer=obs_normalizer)
    best_success = float(best_metrics.get("success_rate", 0.0))
    best_path = output_dir / "best_checkpoint.pt"
    final_path = output_dir / "final_checkpoint.pt"
    initial_payload = {
        "step": 0,
        **dataset_logs,
        "Train/buffer_novice_size": float(novice_buffer.size),
        "Train/buffer_human_size": float(human_buffer.size),
        **{f"Eval/{k}": float(v) for k, v in best_metrics.items()},
    }
    _write_metrics(metrics_path, initial_payload)
    if wandb_run is not None:
        wandb_run.log(initial_payload, step=0)

    step_interval = max(1, int(args.eval_interval))
    log_interval = max(1, int(args.log_interval))
    save_interval = max(1, int(args.save_interval))
    try:
        init_ckpt = _save_pvp_checkpoint(
            run_model_dir=run_model_dir,
            tag="step0",
            step_value=0,
            run_prefix=args.env_name.replace("-", "_"),
            actor=actor,
            critic=critic,
            actor_target=actor_target,
            critic_target=critic_target,
            obs_normalizer=obs_normalizer,
            args=args,
        )
        torch.save(torch.load(init_ckpt, map_location="cpu", weights_only=False), best_path)
        actor_loss_value = 0.0
        critic_loss_value = 0.0
        q_min_data_mean = 0.0
        q_min_teacher_mean = 0.0
        q_min_non_teacher_mean = 0.0
        target_q_mean = 0.0
        q_disagreement_mean = 0.0
        proxy_teacher_loss_value = 0.0
        proxy_student_loss_value = 0.0
        pvp_intervened_batch_fraction = 0.0
        pvp_td_reward_mean = 0.0
        for step in range(1, int(args.offline_updates) + 1):
            batch = _sample_balanced_batch(
                novice_buffer=novice_buffer,
                human_buffer=human_buffer,
                batch_size=int(args.batch_size),
                device=device,
                use_balance_sample=bool(getattr(args, "pvp_balance_sample", True)),
            )
            if batch is None:
                raise ValueError("Offline PVP buffers are too small for the requested batch_size.")
            batch_obs = obs_normalizer(batch.states)
            batch_next_obs = obs_normalizer(batch.next_states)
            with torch.no_grad():
                noise = torch.randn_like(batch.actions_behavior) * float(getattr(args, "pvp_target_policy_noise", 0.2))
                noise = noise.clamp(
                    -float(getattr(args, "pvp_target_noise_clip", 0.5)),
                    float(getattr(args, "pvp_target_noise_clip", 0.5)),
                )
                next_actions = (actor_target(batch_next_obs) + noise).clamp(-1.0, 1.0)
                target_q = critic_target.min_q(batch_next_obs, next_actions)
                td_rewards = batch.rewards if bool(getattr(args, "pvp_include_env_reward_in_td", False)) else torch.zeros_like(batch.rewards)
                target_q = td_rewards + (1.0 - batch.dones) * float(args.gamma) * target_q
            q1_behavior, q2_behavior = critic(batch_obs, batch.actions_behavior)
            q1_novice, q2_novice = critic(batch_obs, batch.actions_novice)
            td_mask = torch.ones_like(batch.interventions)
            if bool(getattr(args, "pvp_stop_td_on_intervention_start", True)):
                td_mask = 1.0 - batch.intervention_starts
            q_value_bound = float(getattr(args, "pvp_proxy_value_bound", 1.0))
            cql_coeff = float(getattr(args, "pvp_cql_coefficient", 1.0))
            critic_loss = 0.5 * F.mse_loss(td_mask * q1_behavior, td_mask * target_q)
            critic_loss = critic_loss + 0.5 * F.mse_loss(td_mask * q2_behavior, td_mask * target_q)
            proxy_teacher_loss = (
                batch.interventions * cql_coeff * F.mse_loss(q1_behavior, q_value_bound * torch.ones_like(q1_behavior), reduction="none")
            ).mean()
            proxy_teacher_loss = proxy_teacher_loss + (
                batch.interventions * cql_coeff * F.mse_loss(q2_behavior, q_value_bound * torch.ones_like(q2_behavior), reduction="none")
            ).mean()
            proxy_student_loss = (
                batch.interventions * cql_coeff * F.mse_loss(q1_novice, -q_value_bound * torch.ones_like(q1_novice), reduction="none")
            ).mean()
            proxy_student_loss = proxy_student_loss + (
                batch.interventions * cql_coeff * F.mse_loss(q2_novice, -q_value_bound * torch.ones_like(q2_novice), reduction="none")
            ).mean()
            critic_loss_total = critic_loss + proxy_teacher_loss + proxy_student_loss

            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(),
                max_norm=float(args.max_grad_norm) if float(args.max_grad_norm) > 0 else float("inf"),
            )
            critic_optimizer.step()

            if step % int(max(1, getattr(args, "pvp_policy_delay", 2))) == 0:
                actor_actions = actor(batch_obs)
                actor_loss = -critic.min_q(batch_obs, actor_actions).mean()
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    max_norm=float(args.max_grad_norm) if float(args.max_grad_norm) > 0 else float("inf"),
                )
                actor_optimizer.step()
                tau = float(args.tau)
                for src_param, tgt_param in zip(actor.parameters(), actor_target.parameters()):
                    tgt_param.data.mul_(1.0 - tau).add_(tau * src_param.data)
                for src_param, tgt_param in zip(critic.parameters(), critic_target.parameters()):
                    tgt_param.data.mul_(1.0 - tau).add_(tau * src_param.data)
                actor_loss_value = float(actor_loss.detach().cpu().item())

            with torch.no_grad():
                qmin_data = torch.minimum(q1_behavior, q2_behavior)
                qdis_data = torch.abs(q1_behavior - q2_behavior)
                teacher_rows = batch.interventions.squeeze(-1) > 0.5
                non_teacher_rows = ~teacher_rows
                q_min_data_mean = float(qmin_data.mean().detach().cpu().item())
                q_disagreement_mean = float(qdis_data.mean().detach().cpu().item())
                q_min_teacher_mean = float(qmin_data[teacher_rows].mean().detach().cpu().item()) if bool(teacher_rows.any().item()) else 0.0
                q_min_non_teacher_mean = float(qmin_data[non_teacher_rows].mean().detach().cpu().item()) if bool(non_teacher_rows.any().item()) else 0.0
                target_q_mean = float(target_q.mean().detach().cpu().item())
                pvp_intervened_batch_fraction = float(batch.interventions.mean().detach().cpu().item())
                pvp_td_reward_mean = float(td_rewards.mean().detach().cpu().item())
            critic_loss_value = float(critic_loss_total.detach().cpu().item())
            proxy_teacher_loss_value = float(proxy_teacher_loss.detach().cpu().item())
            proxy_student_loss_value = float(proxy_student_loss.detach().cpu().item())

            if step == 1 or step % log_interval == 0 or step == int(args.offline_updates):
                payload = {
                    "step": int(step),
                    **dataset_logs,
                    "Train/actor_loss": float(actor_loss_value),
                    "Train/critic_loss": float(critic_loss_value),
                    "Train/target_q_mean": float(target_q_mean),
                    "Train/q_min_data_mean": float(q_min_data_mean),
                    "Train/q_min_teacher_action_mean": float(q_min_teacher_mean),
                    "Train/q_min_non_teacher_action_mean": float(q_min_non_teacher_mean),
                    "Train/q_disagreement_data_mean": float(q_disagreement_mean),
                    "Train/pvp_proxy_teacher_loss": float(proxy_teacher_loss_value),
                    "Train/pvp_proxy_student_loss": float(proxy_student_loss_value),
                    "Train/pvp_intervened_batch_fraction": float(pvp_intervened_batch_fraction),
                    "Train/pvp_td_reward_mean": float(pvp_td_reward_mean),
                    "Train/buffer_novice_size": float(novice_buffer.size),
                    "Train/buffer_human_size": float(human_buffer.size),
                }
                _write_metrics(metrics_path, payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=step)
                print(
                    f"[OfflinePVP] step={step} qloss={payload['Train/critic_loss']:.6f} "
                    f"teacher_frac={payload['Train/pvp_intervened_batch_fraction']:.4f}",
                    flush=True,
                )

            if step % step_interval == 0 or step == int(args.offline_updates):
                eval_metrics = _run_pvp_eval(
                    args=eval_args,
                    device=device,
                    eval_envs=eval_envs,
                    actor=actor,
                    obs_normalizer=obs_normalizer,
                )
                payload = {"step": int(step), **{f"Eval/{k}": float(v) for k, v in eval_metrics.items()}}
                _write_metrics(metrics_path, payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=step)
                if float(eval_metrics.get("success_rate", 0.0)) >= best_success:
                    best_success = float(eval_metrics.get("success_rate", 0.0))
                    best_metrics = dict(eval_metrics)
                    ckpt = _save_pvp_checkpoint(
                        run_model_dir=run_model_dir,
                        tag="best",
                        step_value=step,
                        run_prefix=args.env_name.replace("-", "_"),
                        actor=actor,
                        critic=critic,
                        actor_target=actor_target,
                        critic_target=critic_target,
                        obs_normalizer=obs_normalizer,
                        args=args,
                    )
                    if ckpt != best_path:
                        torch.save(torch.load(ckpt, map_location="cpu", weights_only=False), best_path)

            if step % save_interval == 0 or step == int(args.offline_updates):
                ckpt = _save_pvp_checkpoint(
                    run_model_dir=run_model_dir,
                    tag=f"step{step}",
                    step_value=step,
                    run_prefix=args.env_name.replace("-", "_"),
                    actor=actor,
                    critic=critic,
                    actor_target=actor_target,
                    critic_target=critic_target,
                    obs_normalizer=obs_normalizer,
                    args=args,
                )
                if step == int(args.offline_updates):
                    torch.save(torch.load(ckpt, map_location="cpu", weights_only=False), final_path)
        return best_metrics
    finally:
        progress_file.close()
        eval_envs.close()


def _fastsac_batch_sizes(args: argparse.Namespace, replay_buffer, demo_buffer) -> tuple[int, int, int]:
    base_batch = max(1, int(args.batch_size))
    pref_sampling_mode = str(getattr(args, "pref_sampling_mode", "independent")).strip().lower()
    if pref_sampling_mode == "linked":
        b_pref = 0
    else:
        pref_buffer = None
        b_pref = int(base_batch * float(getattr(args, "pref_sample_ratio", 0.0))) if pref_buffer is not None else 0
    b_demo = int(base_batch * float(getattr(args, "demo_sample_ratio", 0.0))) if demo_buffer is not None else 0
    main_batch = max(1, base_batch - b_pref - b_demo)
    return base_batch, main_batch, b_demo


def _populate_fastsac_buffers(
    *,
    args: argparse.Namespace,
    dataset: OfflineDataset,
    device: torch.device,
    replay_buffer,
    demo_buffer,
) -> None:
    eil_good = torch.zeros(dataset.num_rows, dtype=torch.bool, device=device)
    eil_bad = torch.zeros(dataset.num_rows, dtype=torch.bool, device=device)
    if str(args.method) == "eil":
        eil_good, eil_bad = _reconstruct_eil_labels(
            dataset=dataset,
            bad_pre_steps=int(getattr(args, "eil_bad_pre_steps", 8)),
            use_teacher_rows_as_good=bool(getattr(args, "eil_use_teacher_rows_as_good", True)),
        )

    for idx in range(dataset.num_rows):
        td = _make_transition_tensordict(
            obs=dataset.observations[idx],
            action=dataset.actions[idx],
            student_action=dataset.student_actions[idx],
            next_obs=dataset.next_observations[idx],
            reward=dataset.rewards[idx],
            done=dataset.dones[idx],
            truncation=dataset.truncations[idx],
            teacher_intervened=dataset.teacher_mask[idx],
            device=device,
            eil_good=bool(eil_good[idx].item()),
            eil_bad=bool(eil_bad[idx].item()),
        )
        replay_buffer.extend(td)
        if demo_buffer is not None and bool(dataset.teacher_mask[idx].item()):
            demo_buffer.extend(td)


def _build_fastsac_args_for_method(args: argparse.Namespace) -> argparse.Namespace:
    method_args = argparse.Namespace(**vars(args))
    method_args.train_render_mode = "none"
    method_args.eval_render_mode = "none"
    method_args.use_intervention = False
    method_args.intervention_mode = "none"
    method_args.visualize_intervention_colors = False
    method_args.num_envs = 1
    method_args.demo_prefill_episodes = 0
    method_args.demo_prefill_num_envs = 0
    method_args.demo_prefill_target = "demo"
    method_args.demo_prefill_intervention_mode = "agent_always"
    method_args.pref_buffer_enable = False
    method_args.pref_sample_ratio = 0.0
    method_args.pref_capacity = 0
    method_args.actor_bc_weight_demo = 0.0
    method_args.actor_bc_weight_pref = 0.0

    if args.method == "own":
        method_args.algo_variant = "own"
        method_args.demo_buffer_enable = False
        method_args.demo_sample_ratio = 0.0
        method_args.pref_sampling_mode = "linked"
        method_args.pref_rank_weight = float(getattr(args, "pref_rank_weight", 1.0))
        method_args.pref_rank_margin = float(getattr(args, "pref_rank_margin", 0.01))
        method_args.pref_loss_type = str(getattr(args, "pref_loss_type", "hinge"))
        method_args.pref_critic_scope = str(getattr(args, "pref_critic_scope", "all"))
        method_args.pref_linked_action_epsilon = float(getattr(args, "pref_linked_action_epsilon", 0.01))
        method_args.pref_linked_action_weight_scale = float(getattr(args, "pref_linked_action_weight_scale", 0.25))
    elif args.method == "own_nogate":
        method_args.algo_variant = "own"
        method_args.demo_buffer_enable = False
        method_args.demo_sample_ratio = 0.0
        method_args.pref_sampling_mode = "linked"
        method_args.pref_rank_weight = float(getattr(args, "pref_rank_weight", 1.0))
        method_args.pref_rank_margin = float(getattr(args, "pref_rank_margin", 0.01))
        method_args.pref_loss_type = str(getattr(args, "pref_loss_type", "hinge"))
        method_args.pref_critic_scope = str(getattr(args, "pref_critic_scope", "all"))
        method_args.pref_linked_action_epsilon = 1e-6
        method_args.pref_linked_action_weight_scale = 0.0
    elif args.method == "hilserl":
        method_args.algo_variant = "own"
        method_args.demo_buffer_enable = True
        method_args.demo_sample_ratio = float(getattr(args, "demo_sample_ratio", 0.5))
        method_args.pref_sampling_mode = "linked"
        method_args.pref_rank_weight = 0.0
        method_args.pref_linked_action_epsilon = 1e-6
        method_args.pref_linked_action_weight_scale = 0.0
    elif args.method == "eil":
        method_args.algo_variant = "eil"
        method_args.demo_buffer_enable = False
        method_args.demo_sample_ratio = 0.0
        method_args.pref_sampling_mode = "linked"
        method_args.pref_rank_weight = 0.0
        method_args.fixed_alpha = 0.0
        method_args.alpha_init = 0.0
        method_args.alpha_min = 0.0
        method_args.alpha_max = 0.0
    else:
        raise ValueError(f"unsupported fastsac method={args.method!r}")
    return method_args


def _run_fastsac_offline(
    *,
    args: argparse.Namespace,
    dataset: OfflineDataset,
    device: torch.device,
    output_dir: Path,
    metrics_path: Path,
    wandb_run,
) -> dict[str, float]:
    method_args = _build_fastsac_args_for_method(args)
    probe_envs, _, _, _, n_obs, n_act, _ = build_manip_environment(method_args, device, lambda _: None)
    probe_envs.close()
    if int(n_obs) != dataset.obs_dim or int(n_act) != dataset.act_dim:
        raise ValueError(
            f"Dataset dimensions obs={dataset.obs_dim}, act={dataset.act_dim} do not match env dims obs={n_obs}, act={n_act}."
        )
    model = initialize_models(method_args, device, n_obs, n_act, lambda _: None)
    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    _fit_obs_normalizer(
        obs_normalizer=obs_normalizer,
        observations=dataset.observations,
        next_observations=dataset.next_observations,
    )
    obs_normalizer.eval()
    buffers = initialize_buffers(method_args, device, n_obs, n_act, obs_normalizer)
    amp = initialize_amp(method_args, device)
    replay_buffer = create_replay_buffer(
        method_args,
        device,
        n_obs,
        n_act,
        buffer_size=max(dataset.num_rows + 8, dataset.num_rows + 8),
        n_env_override=1,
    )
    demo_buffer = None
    if bool(getattr(method_args, "demo_buffer_enable", False)):
        demo_buffer = create_replay_buffer(
            method_args,
            device,
            n_obs,
            n_act,
            buffer_size=max(int(dataset.teacher_mask.to(torch.int64).sum().item()) + 8, 8),
            n_env_override=1,
        )
    _populate_fastsac_buffers(args=args, dataset=dataset, device=device, replay_buffer=replay_buffer, demo_buffer=demo_buffer)
    updater = build_updater_from_components(
        args=method_args,
        device=device,
        model=model,
        buffers=buffers,
        obs_normalizer=obs_normalizer,
        amp=amp,
    )
    eval_args = _prepare_offline_eval_args(method_args)
    eval_envs = build_manip_eval_environment(eval_args, device)
    dataset_logs = _dataset_summary(dataset)
    if args.method == "eil":
        eil_good, eil_bad = _reconstruct_eil_labels(
            dataset=dataset,
            bad_pre_steps=int(getattr(args, "eil_bad_pre_steps", 8)),
            use_teacher_rows_as_good=bool(getattr(args, "eil_use_teacher_rows_as_good", True)),
        )
        dataset_logs["Offline/eil_good_fraction"] = float(eil_good.float().mean().item())
        dataset_logs["Offline/eil_bad_fraction"] = float(eil_bad.float().mean().item())

    best_metrics = run_eval_metrics(
        args=eval_args,
        device=device,
        eval_envs=eval_envs,
        actor_backbone=model.actor_backbone,
        actor_head=model.actor_head,
        obs_normalizer=obs_normalizer,
        amp=AMPComponents(enabled=False, device_type=device.type, dtype=torch.float32, scaler=torch.amp.GradScaler(enabled=False)),
    )
    best_success = float(best_metrics.get("success_rate", 0.0))
    best_path = output_dir / "best_checkpoint.pt"
    final_path = output_dir / "final_checkpoint.pt"
    initial_payload = {
        "step": 0,
        **dataset_logs,
        "Train/replay_size": float(getattr(replay_buffer, "size", 0)),
        "Train/demo_size": float(getattr(demo_buffer, "size", 0)) if demo_buffer is not None else 0.0,
        **{f"Eval/{k}": float(v) for k, v in best_metrics.items()},
    }
    _write_metrics(metrics_path, initial_payload)
    if wandb_run is not None:
        wandb_run.log(initial_payload, step=0)

    base_batch, main_batch, b_demo = _fastsac_batch_sizes(method_args, replay_buffer, demo_buffer)
    log_interval = max(1, int(args.log_interval))
    step_interval = max(1, int(args.eval_interval))
    save_interval = max(1, int(args.save_interval))

    try:
        method_args._offline_save_step = 0
        _save_fastsac_checkpoint(path=best_path, args=method_args, model=model, obs_normalizer=obs_normalizer)
        for step in range(1, int(args.offline_updates) + 1):
            metrics_accumulator, updates_count = updater.update(
                replay_buffer=replay_buffer,
                demo_buffer=demo_buffer,
                total_env_steps=step,
                main_batch=main_batch,
                base_batch=base_batch,
                b_pref=0,
                b_demo=b_demo,
            )
            if updates_count <= 0:
                raise ValueError("FastSAC offline updater did not perform any updates.")
            scale = float(max(1, updates_count))
            mean_metrics = {k: float(v) / scale for k, v in metrics_accumulator.items()}
            if step == 1 or step % log_interval == 0 or step == int(args.offline_updates):
                payload = {
                    "step": int(step),
                    **dataset_logs,
                    "Train/replay_size": float(getattr(replay_buffer, "size", 0)),
                    "Train/demo_size": float(getattr(demo_buffer, "size", 0)) if demo_buffer is not None else 0.0,
                    "Train/critic_loss": float(mean_metrics.get("critic_loss", 0.0)),
                    "Train/critic_loss_replay": float(mean_metrics.get("critic_loss_replay", 0.0)),
                    "Train/critic_loss_pref": float(mean_metrics.get("critic_loss_pref", 0.0)),
                    "Train/critic_loss_pref_weighted": float(mean_metrics.get("critic_loss_pref_weighted", 0.0)),
                    "Train/actor_loss": float(mean_metrics.get("actor_loss", 0.0)),
                    "Train/actor_loss_sac": float(mean_metrics.get("actor_loss_sac", 0.0)),
                    "Train/target_q_mean": float(mean_metrics.get("target_q", 0.0)),
                    "Train/q_min_data_mean": float(mean_metrics.get("q_min_data", 0.0)),
                    "Train/q_min_pi_mean": float(mean_metrics.get("q_min_pi", 0.0)),
                    "Train/q_disagreement_data_mean": float(mean_metrics.get("q_disagreement_data", 0.0)),
                    "Train/q_disagreement_pi_mean": float(mean_metrics.get("q_disagreement_pi", 0.0)),
                    "Train/pref_linked_fraction": float(mean_metrics.get("pref_linked_fraction", 0.0)),
                    "Train/pref_linked_effective_fraction": float(mean_metrics.get("pref_linked_effective_fraction", 0.0)),
                    "Train/pref_q_delta_mean": float(
                        mean_metrics.get("pref_q_delta_sum", 0.0) / max(1e-8, mean_metrics.get("pref_q_delta_count", 0.0))
                    ) if mean_metrics.get("pref_q_delta_count", 0.0) > 0 else 0.0,
                    "Train/pref_action_delta_mean": float(
                        mean_metrics.get("pref_action_delta_sum", 0.0) / max(1e-8, mean_metrics.get("pref_action_delta_count", 0.0))
                    ) if mean_metrics.get("pref_action_delta_count", 0.0) > 0 else 0.0,
                    "Train/pref_action_weight_mean": float(
                        mean_metrics.get("pref_action_weight_sum", 0.0) / max(1e-8, mean_metrics.get("pref_action_weight_count", 0.0))
                    ) if mean_metrics.get("pref_action_weight_count", 0.0) > 0 else 0.0,
                    "Train/pref_lambda": float(mean_metrics.get("pref_lambda", 0.0)),
                    "Train/eil_good_loss": float(mean_metrics.get("eil_good_loss", 0.0)),
                    "Train/eil_bad_loss": float(mean_metrics.get("eil_bad_loss", 0.0)),
                    "Train/eil_pair_loss": float(mean_metrics.get("eil_pair_loss", 0.0)),
                    "Train/eil_good_batch_fraction": float(mean_metrics.get("eil_good_batch_fraction", 0.0)),
                    "Train/eil_bad_batch_fraction": float(mean_metrics.get("eil_bad_batch_fraction", 0.0)),
                    "Train/demo_rows_sampled": float(mean_metrics.get("demo_rows_sampled", 0.0)),
                    "Train/demo_fallback_updates": float(mean_metrics.get("demo_fallback_updates", 0.0)),
                }
                _write_metrics(metrics_path, payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=step)
                print(
                    f"[OfflineFastSAC:{args.method}] step={step} "
                    f"critic={payload['Train/critic_loss']:.6f} actor={payload['Train/actor_loss']:.6f}",
                    flush=True,
                )

            if step % step_interval == 0 or step == int(args.offline_updates):
                eval_metrics = run_eval_metrics(
                    args=eval_args,
                    device=device,
                    eval_envs=eval_envs,
                    actor_backbone=model.actor_backbone,
                    actor_head=model.actor_head,
                    obs_normalizer=obs_normalizer,
                    amp=AMPComponents(enabled=False, device_type=device.type, dtype=torch.float32, scaler=torch.amp.GradScaler(enabled=False)),
                )
                payload = {"step": int(step), **{f"Eval/{k}": float(v) for k, v in eval_metrics.items()}}
                _write_metrics(metrics_path, payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=step)
                if float(eval_metrics.get("success_rate", 0.0)) >= best_success:
                    best_success = float(eval_metrics.get("success_rate", 0.0))
                    best_metrics = dict(eval_metrics)
                    method_args._offline_save_step = int(step)
                    _save_fastsac_checkpoint(path=best_path, args=method_args, model=model, obs_normalizer=obs_normalizer)

            if step % save_interval == 0 or step == int(args.offline_updates):
                method_args._offline_save_step = int(step)
                checkpoint_path = output_dir / f"step_{step}.pt"
                _save_fastsac_checkpoint(path=checkpoint_path, args=method_args, model=model, obs_normalizer=obs_normalizer)
                if step == int(args.offline_updates):
                    _save_fastsac_checkpoint(path=final_path, args=method_args, model=model, obs_normalizer=obs_normalizer)
        return best_metrics
    finally:
        eval_envs.close()


def _method_runner(method: str) -> Callable[..., dict[str, float]]:
    if method == "bc":
        return _run_bc_offline
    if method == "hg_dagger":
        return _run_hg_offline
    if method == "pvp":
        return _run_pvp_offline
    if method in {"eil", "hilserl", "own", "own_nogate"}:
        return _run_fastsac_offline
    raise ValueError(f"unsupported method={method!r}")


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
    os.environ.setdefault("WANDB_MODE", _default_wandb_mode())
    os.environ.setdefault("WANDB_CONSOLE", "off")
    os.environ.setdefault("WANDB_SILENT", "true")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = select_device(args)
    dataset = _load_dataset(
        path=args.dataset_path,
        device=device,
        teacher_mask_mode=str(args.teacher_mask_mode),
    )
    _maybe_apply_dataset_metadata_defaults(args, dataset.metadata)
    output_dir = _ensure_output_dir(
        args.output_dir,
        REPO_ROOT / "models" / "offline_method_compare" / str(args.name),
    )
    metrics_path = output_dir / "metrics.jsonl"
    config = {
        **vars(args),
        "device_resolved": str(device),
        "dataset_rows": dataset.num_rows,
        "dataset_obs_dim": dataset.obs_dim,
        "dataset_act_dim": dataset.act_dim,
        "dataset_metadata": dataset.metadata,
    }
    wandb_run = _maybe_init_wandb(args, config)
    print(
        f"[OfflineCompare] method={args.method} device={device} "
        f"dataset_rows={dataset.num_rows} teacher_like_frac={float(dataset.teacher_mask.float().mean().item()):.4f}",
        flush=True,
    )
    runner = _method_runner(str(args.method))
    best_metrics = runner(
        args=args,
        dataset=dataset,
        device=device,
        output_dir=output_dir,
        metrics_path=metrics_path,
        wandb_run=wandb_run,
    )
    summary = {
        "method": str(args.method),
        "dataset_path": str(Path(args.dataset_path).expanduser()),
        "output_dir": str(output_dir),
        "best_metrics": {k: float(v) for k, v in best_metrics.items()},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OfflineCompareDone] method={args.method} best={json.dumps(summary['best_metrics'], sort_keys=True)}", flush=True)
    if wandb_run is not None:
        wandb_run.summary["best_success_rate"] = float(best_metrics.get("success_rate", 0.0))
        wandb_run.summary["best_avg_length"] = float(best_metrics.get("avg_length", 0.0))
        wandb_run.finish()


if __name__ == "__main__":
    main()
