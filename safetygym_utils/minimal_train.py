from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import sys

import numpy as np
import torch
from tensordict import TensorDict

_FAST_SAC_PATH = Path(__file__).resolve().parent.parent / "fasttd3" / "fast_sac"
if _FAST_SAC_PATH.exists():
    _fast_sac_path_str = str(_FAST_SAC_PATH)
    if _fast_sac_path_str not in sys.path:
        sys.path.insert(0, _fast_sac_path_str)

from fast_sac import Actor, Critic
from fast_sac_utils import EmpiricalNormalization, SimpleReplayBuffer

from .dataset_io import (
    DEFAULT_SAFETYGYM_DATASET_DIR,
    build_dataset_path,
    extend_buffer_from_dataset,
    find_latest_transition_dataset,
    load_transition_dataset,
    save_buffer_as_transition_dataset,
)
from .env import clip_action_to_space, extract_goal_distance, extract_step_limit, make_safety_env
from .io import save_args_json
from .metrics import EpisodeWindow, augment_rollout_summary, classify_outcome
from .policy_viz import _extract_bounds, _extract_overlay_specs, plot_episode_contact_sheet, plot_eval_episode_trajectory
from .sac import SACUpdateMetrics, SACTensors, SafetyActor, SafetyCritic, sac_update_step
from .wrappers import HumanInterventionWrapper, RewardModeWrapper, TerminateOnGoalWrapper
from .controllers import (
    DEFAULT_SAFETY_GAMEPAD_CACHE_PATH,
    DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
    DEFAULT_SAFETY_GAMEPAD_PORT,
    build_human_controller,
)


@dataclass
class MinimalUpdateMetrics:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    target_q_mean: float
    q_min_pi_mean: float
    q_min_data_mean: float
    q_gap_mean: float
    replay_reward_mean: float
    actor_updates: float
    alpha_updates: float


def _critic_pair(critic, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = critic(obs, actions)
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError(f"Expected 2 critics, got tuple of length {len(out)}")
        return out[0], out[1]
    if isinstance(out, list):
        if len(out) < 2:
            raise ValueError(f"Expected at least 2 critics, got list of length {len(out)}")
        return out[0], out[1]
    raise TypeError(f"Unsupported critic output type: {type(out)!r}")


def _prepare_run_dirs(args) -> tuple[Path, Path]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not getattr(args, "exp_name", ""):
        args.exp_name = f"minimal_{args.env_name.replace('-', '_')}_{stamp}"
    log_dir = Path("logs") / "safetygym_minimal" / args.exp_name
    model_dir = Path("models") / "safetygym_minimal" / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_args_json(log_dir / "args.json", vars(args))
    save_args_json(model_dir / "args.json", vars(args))
    return log_dir, model_dir


def _resolved_gamepad_cache_path(args) -> str:
    raw = str(getattr(args, "gamepad_cache_path", "") or "").strip()
    return raw or str(DEFAULT_SAFETY_GAMEPAD_CACHE_PATH)


def _resolved_gamepad_config_path(args) -> str:
    raw = str(getattr(args, "gamepad_config_path", "") or "").strip()
    return raw or str(DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH)


def _make_env_with_wrappers(
    *,
    args,
    seed: int,
    with_intervention: bool,
    controller,
    render_mode_override: str | None = None,
):
    env = make_safety_env(
        args.env_name,
        render_mode=(
            str(render_mode_override)
            if render_mode_override is not None
            else str(getattr(args, "render_mode", "none"))
        ),
        max_episode_steps=int(args.max_episode_steps),
        surface_mode=args.surface_mode,
        car_wheel_command_limit=float(args.car_wheel_command_limit),
        car_force_scale=float(args.car_force_scale),
        seed=seed,
    )
    env = RewardModeWrapper(
        env,
        reward_mode=args.reward_mode,
        dense_reward_scale=float(args.dense_reward_scale),
        step_penalty=float(args.step_penalty),
        cost_penalty=float(getattr(args, "cost_penalty", 0.0)),
        cost_penalty_warmup_steps=int(getattr(args, "cost_penalty_warmup_steps", 0)),
        cost_penalty_ramp_steps=int(getattr(args, "cost_penalty_ramp_steps", 0)),
        clearance_penalty_scale=float(getattr(args, "clearance_penalty_scale", 0.0)),
        clearance_margin=float(getattr(args, "clearance_margin", 0.0)),
        clearance_penalty_power=float(getattr(args, "clearance_penalty_power", 1.0)),
        clearance_penalty_warmup_steps=int(getattr(args, "clearance_penalty_warmup_steps", 0)),
        clearance_penalty_ramp_steps=int(getattr(args, "clearance_penalty_ramp_steps", 0)),
        forward_reward_scale=float(getattr(args, "forward_reward_scale", 0.0)),
        backward_penalty_scale=float(getattr(args, "backward_penalty_scale", 0.0)),
        heading_reward_scale=float(getattr(args, "heading_reward_scale", 0.0)),
        heading_positive_only=bool(getattr(args, "heading_positive_only", True)),
    )
    if bool(getattr(args, "terminate_on_goal", False)):
        env = TerminateOnGoalWrapper(env)
    if with_intervention:
        if controller is None:
            raise ValueError("Intervention requested but controller is None")
        env = HumanInterventionWrapper(
            env,
            controller=controller,
            threshold=float(args.intervention_threshold),
            hold_seconds=float(args.intervention_hold_seconds),
            clearance_override_threshold=float(getattr(args, "teacher_override_clearance_threshold", -1.0)),
        )
    return env


def _maybe_init_wandb(args, log_dir: Path):
    if not bool(getattr(args, "use_wandb", False)):
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError("--use_wandb was set but wandb could not be imported.") from exc

    kwargs: Dict[str, Any] = {
        "project": str(getattr(args, "wandb_project", "thesis-safetygym")),
        "mode": str(getattr(args, "wandb_mode", "offline")),
        "config": vars(args),
        "dir": str(log_dir),
    }
    run_name = str(getattr(args, "wandb_run_name", "")).strip() or str(getattr(args, "exp_name", "")).strip()
    if run_name:
        kwargs["name"] = run_name
    group = str(getattr(args, "wandb_group", "")).strip()
    if group:
        kwargs["group"] = group
    entity = str(getattr(args, "wandb_entity", "")).strip()
    if entity:
        kwargs["entity"] = entity
    return wandb.init(**kwargs)


def _maybe_render_policy_map(
    *,
    args,
    checkpoint_path: Path,
    step_value: int,
    log_dir: Path,
    wandb_run,
) -> None:
    if not bool(getattr(args, "viz_on_checkpoint", False)):
        return
    first_step = int(max(0, getattr(args, "viz_first_step", 0)))
    if step_value < first_step:
        return
    try:
        from .policy_viz import generate_safety_policy_maps

        result = generate_safety_policy_maps(
            model_path=checkpoint_path,
            output_dir=log_dir / "policy_maps",
            tag=str(step_value),
            env_name=str(args.env_name),
            device=str(getattr(args, "viz_device", "cpu")),
            grid_resolution=int(getattr(args, "viz_grid_resolution", 48)),
            quiver_stride=int(getattr(args, "viz_quiver_stride", 4)),
            seed=int(getattr(args, "viz_seed", 0)),
            headings_deg=str(getattr(args, "viz_headings_deg", "0,90,180,270")),
            num_rollouts=int(getattr(args, "viz_num_rollouts", 4)),
        )
        print(f"[Viz] generated {result['map_png']}", flush=True)
        if wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb_run.log(
                    {
                        "viz/policy_map": wandb.Image(str(result["map_png"]), caption=f"step={step_value}"),
                        "viz/policy_rollouts": wandb.Image(str(result["rollout_png"]), caption=f"step={step_value}"),
                    },
                    step=int(step_value),
                )
            except Exception:
                pass
    except Exception as exc:
        print(f"[Viz] failed at step {step_value}: {exc}", flush=True)


def _maybe_export_final_buffer_dataset(
    *,
    args,
    buffer,
    env_name: str,
    variant: str,
    reward_mode: str,
    label_suffix: str,
    explicit_path: str,
    enabled: bool,
    filter_mode: str = "all",
) -> Dict[str, float]:
    if buffer is None or not enabled:
        return {}
    dataset_path = str(explicit_path or "").strip()
    if not dataset_path:
        dataset_root = str(getattr(args, "export_dataset_dir", "") or "").strip() or str(DEFAULT_SAFETYGYM_DATASET_DIR)
        dataset_path = str(
            build_dataset_path(
                env_name=env_name,
                dataset_dir=dataset_root,
                label=f"{variant}_{reward_mode}_{label_suffix}",
            )
        )
    metadata = {
        "env_name": str(env_name),
        "variant": str(variant),
        "reward_mode": str(reward_mode),
        "source_buffer": str(label_suffix),
        "filter_mode": str(filter_mode),
        "total_timesteps": int(getattr(args, "total_timesteps", 0)),
        "init_checkpoint_path": str(getattr(args, "init_checkpoint_path", "") or ""),
    }
    stats = save_buffer_as_transition_dataset(
        buffer=buffer,
        path=dataset_path,
        metadata=metadata,
        max_rows=int(getattr(args, "export_dataset_max_rows", 0)),
        filter_mode=filter_mode,
    )
    print(f"[Dataset] exported {stats['rows_saved']} rows from {label_suffix} buffer to {stats['path']}", flush=True)
    metric_prefix = f"train/export_{label_suffix}"
    return {
        f"{metric_prefix}_rows": float(stats["rows_saved"]),
        f"{metric_prefix}_saved": 1.0,
    }


def _build_transition(
    *,
    obs: np.ndarray,
    action: np.ndarray,
    next_obs: np.ndarray,
    reward: float,
    done: bool,
    truncated: bool,
    device: torch.device,
    student_action: np.ndarray | None = None,
    teacher_intervened: bool | None = None,
) -> TensorDict:
    payload = {
        "observations": torch.as_tensor(obs, device=device, dtype=torch.float32).view(1, -1),
        "actions": torch.as_tensor(action, device=device, dtype=torch.float32).view(1, -1),
        "next": {
            "observations": torch.as_tensor(next_obs, device=device, dtype=torch.float32).view(1, -1),
            "rewards": torch.as_tensor([reward], device=device, dtype=torch.float32),
            "dones": torch.as_tensor([done], device=device, dtype=torch.long),
            "truncations": torch.as_tensor([truncated], device=device, dtype=torch.long),
        },
    }
    if student_action is not None:
        payload["student_actions"] = torch.as_tensor(student_action[None, :], device=device, dtype=torch.float32)
    if teacher_intervened is not None:
        payload["teacher_intervened"] = torch.as_tensor([bool(teacher_intervened)], device=device, dtype=torch.bool)
    return TensorDict(payload, batch_size=(1,), device=device)


def _scale_action_tensor(actions: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    center = 0.5 * (high + low)
    half = 0.5 * (high - low)
    return center + actions * half


def _scale_action_np(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    center = 0.5 * (high + low)
    half = 0.5 * (high - low)
    return (center + action * half).astype(np.float32, copy=False)


def _episode_metrics(
    *,
    info: Dict[str, Any],
    ep_return: float,
    ep_cost: float,
    ep_reward_raw_env: float,
    ep_reward_dense: float,
    ep_reward_sparse: float,
    ep_reward_step_penalty: float,
    ep_reward_cost_penalty: float,
    ep_reward_clearance_penalty: float,
    ep_reward_forward: float,
    ep_reward_backward_penalty: float,
    ep_reward_heading: float,
    ep_mean_constrained_clearance: float,
    ep_min_constrained_clearance: float,
    ep_len: int,
    max_episode_steps: int,
    terminated: bool,
    truncated: bool,
    final_distance: float,
    first_goal_hit_step: int | None,
    first_goal_reward_sum: float,
    first_goal_dense_reward_sum: float,
) -> Dict[str, float]:
    goal_met_count = int(info.get("goal_hit_count", 0))
    goal_met_any = goal_met_count > 0
    outcome = classify_outcome(
        goal_met=goal_met_any,
        episode_steps=ep_len,
        max_episode_steps=max_episode_steps,
    )
    first_hit = int(first_goal_hit_step) if first_goal_hit_step is not None else int(max_episode_steps)
    return {
        "episode_return": float(ep_return),
        "episode_cost_sum": float(ep_cost),
        "episode_cost_rate": float(ep_cost / max(1, ep_len)),
        "episode_length": float(ep_len),
        "reward_shaped_sum": float(ep_return),
        "reward_raw_env_sum": float(ep_reward_raw_env),
        "reward_dense_sum": float(ep_reward_dense),
        "reward_sparse_sum": float(ep_reward_sparse),
        "reward_step_penalty_sum": float(ep_reward_step_penalty),
        "reward_cost_penalty_sum": float(ep_reward_cost_penalty),
        "reward_clearance_penalty_sum": float(ep_reward_clearance_penalty),
        "reward_forward_sum": float(ep_reward_forward),
        "reward_backward_penalty_sum": float(ep_reward_backward_penalty),
        "reward_heading_sum": float(ep_reward_heading),
        "mean_constrained_clearance": float(ep_mean_constrained_clearance),
        "min_constrained_clearance": float(ep_min_constrained_clearance),
        "intervention_steps": float(info.get("teacher_intervention_steps", 0.0)),
        "intervention_fraction": float(info.get("teacher_fraction_steps", 0.0)),
        "intervention_num_bursts": float(info.get("teacher_num_bursts", 0.0)),
        "intervention_avg_burst_len": float(info.get("teacher_avg_burst_len", 0.0)),
        "goal_met": 1.0 if goal_met_any else 0.0,
        "goal_met_count": float(goal_met_count),
        "first_goal_success": 1.0 if first_goal_hit_step is not None else 0.0,
        "first_goal_hit_step": float(first_hit),
        "first_goal_hit_step_success_only": float(first_hit if first_goal_hit_step is not None else 0.0),
        "first_goal_within_100": 1.0 if first_goal_hit_step is not None and first_hit <= 100 else 0.0,
        "first_goal_within_200": 1.0 if first_goal_hit_step is not None and first_hit <= 200 else 0.0,
        "first_goal_reward_sum": float(first_goal_reward_sum),
        "first_goal_dense_reward_sum": float(first_goal_dense_reward_sum),
        "final_distance_to_goal": float(final_distance),
        "outcome_success": 1.0 if outcome == "success" else 0.0,
        "outcome_timeout": 1.0 if outcome == "timeout" else 0.0,
        "outcome_kill": 1.0 if outcome == "kill" else 0.0,
        "outcome_other_failure": 0.0,
        "terminated": 1.0 if terminated else 0.0,
        "truncated": 1.0 if truncated else 0.0,
    }


def _select_eval_action(
    actor: Actor,
    obs: np.ndarray,
    device: torch.device,
    low_np: np.ndarray,
    high_np: np.ndarray,
    *,
    scale_to_env: bool,
    obs_normalizer,
) -> np.ndarray:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
        obs_t = obs_normalizer(obs_t)
        _, _, mean_t = actor(obs_t)
        action = mean_t[0].detach().cpu().numpy().astype(np.float32)
    if scale_to_env:
        action = _scale_action_np(action, low_np, high_np)
    return np.clip(action, low_np, high_np).astype(np.float32, copy=False)


def _run_eval(
    actor: Actor,
    args,
    device: torch.device,
    *,
    obs_normalizer,
    log_dir: Path | None = None,
    step_value: int | None = None,
) -> Dict[str, float]:
    env = _make_env_with_wrappers(
        args=args,
        seed=int(args.seed + 10_000),
        with_intervention=False,
        controller=None,
        render_mode_override="none",
    )
    if hasattr(env, "set_total_steps"):
        env.set_total_steps(int(max(0, step_value if step_value is not None else 0)))
    low_np = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high_np = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    max_steps = extract_step_limit(env)
    win = EpisodeWindow(size=max(1, int(args.num_eval_episodes)))
    task = env.unwrapped.task
    overlay_specs = _extract_overlay_specs(task)
    bounds = _extract_bounds(task, x_range=None, y_range=None)
    save_episode_plots = bool(getattr(args, "eval_save_episode_plots", False))
    episode_plot_max = int(max(0, getattr(args, "eval_episode_plot_max_episodes", 9)))
    saved_episode_plot_paths: list[Path] = []
    plot_dir: Path | None = None
    if save_episode_plots and log_dir is not None and step_value is not None:
        plot_dir = log_dir / "eval_episode_plots" / f"step_{int(step_value)}"

    for episode_idx in range(int(args.num_eval_episodes)):
        obs, info = env.reset(seed=int(args.seed + 10_000 + episode_idx))
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        ep_ret = 0.0
        ep_cost = 0.0
        ep_reward_raw_env = 0.0
        ep_reward_dense = 0.0
        ep_reward_sparse = 0.0
        ep_reward_step_penalty = 0.0
        ep_reward_cost_penalty = 0.0
        ep_reward_clearance_penalty = 0.0
        ep_reward_forward = 0.0
        ep_reward_backward_penalty = 0.0
        ep_reward_heading = 0.0
        ep_clearance_sum = 0.0
        ep_clearance_count = 0
        ep_min_clearance = float("inf")
        ep_len = 0
        goal_hit_count = 0
        first_goal_hit_step: int | None = None
        first_goal_reward_sum = 0.0
        first_goal_dense_reward_sum = 0.0
        done = False
        ep_path = [np.asarray(task.agent.pos[:2], dtype=np.float64).copy()]
        ep_goal_positions = [np.asarray(task.goal.pos[:2], dtype=np.float64).copy()]
        ep_goal_hit_points: list[np.ndarray] = []

        while not done and ep_len < max_steps:
            action = _select_eval_action(
                actor,
                obs,
                device,
                low_np,
                high_np,
                scale_to_env=bool(args.scale_actor_to_env_bounds),
                obs_normalizer=obs_normalizer,
            )
            next_obs, reward, cost, terminated, truncated, info = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            ep_ret += float(reward)
            ep_cost += float(cost)
            ep_reward_raw_env += float(info.get("reward_raw_env", 0.0))
            ep_reward_dense += float(info.get("reward_dense_component", 0.0))
            ep_reward_sparse += float(info.get("reward_sparse_component", 0.0))
            ep_reward_step_penalty += float(info.get("reward_step_penalty_component", 0.0))
            ep_reward_cost_penalty += float(info.get("reward_cost_penalty_component", 0.0))
            ep_reward_clearance_penalty += float(info.get("reward_clearance_penalty_component", 0.0))
            ep_reward_forward += float(info.get("reward_forward_component", 0.0))
            ep_reward_backward_penalty += float(info.get("reward_backward_penalty_component", 0.0))
            ep_reward_heading += float(info.get("reward_heading_component", 0.0))
            step_clearance = info.get("min_constrained_clearance", float("nan"))
            if np.isfinite(step_clearance):
                ep_clearance_sum += float(step_clearance)
                ep_clearance_count += 1
                ep_min_clearance = min(ep_min_clearance, float(step_clearance))
            ep_len += 1
            ep_path.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
            if bool(info.get("goal_met", False)):
                goal_hit_count += 1
                if first_goal_hit_step is None:
                    first_goal_hit_step = int(ep_len)
                    first_goal_reward_sum = float(ep_ret)
                    first_goal_dense_reward_sum = float(ep_reward_dense)
                ep_goal_hit_points.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
            current_goal_xy = np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
            if np.linalg.norm(current_goal_xy - np.asarray(ep_goal_positions[-1], dtype=np.float64)) > 1e-6:
                ep_goal_positions.append(current_goal_xy)
            done = bool(terminated or truncated)
            obs = next_obs

        final_dist = extract_goal_distance(env)
        info = dict(info)
        info["goal_hit_count"] = goal_hit_count
        if save_episode_plots and plot_dir is not None and len(saved_episode_plot_paths) < episode_plot_max:
            plot_path = plot_eval_episode_trajectory(
                output_path=plot_dir / f"episode_{episode_idx + 1:03d}.png",
                task=task,
                overlay_specs=overlay_specs,
                bounds=bounds,
                path=np.asarray(ep_path, dtype=np.float64),
                goal_positions=np.asarray(ep_goal_positions, dtype=np.float64),
                goal_hit_points=(
                    np.asarray(ep_goal_hit_points, dtype=np.float64)
                    if ep_goal_hit_points
                    else np.zeros((0, 2), dtype=np.float64)
                ),
                episode_idx=episode_idx + 1,
                total_episodes=int(args.num_eval_episodes),
                episode_reward=float(ep_ret),
                goals_reached=int(goal_hit_count),
                final_distance=float(final_dist),
            )
            saved_episode_plot_paths.append(Path(plot_path))
        win.add(
            _episode_metrics(
                info=info,
                ep_return=ep_ret,
                ep_cost=ep_cost,
                ep_reward_raw_env=ep_reward_raw_env,
                ep_reward_dense=ep_reward_dense,
                ep_reward_sparse=ep_reward_sparse,
                ep_reward_step_penalty=ep_reward_step_penalty,
                ep_reward_cost_penalty=ep_reward_cost_penalty,
                ep_reward_clearance_penalty=ep_reward_clearance_penalty,
                ep_reward_forward=ep_reward_forward,
                ep_reward_backward_penalty=ep_reward_backward_penalty,
                ep_reward_heading=ep_reward_heading,
                ep_mean_constrained_clearance=(
                    float(ep_clearance_sum / max(1, ep_clearance_count)) if ep_clearance_count > 0 else float("nan")
                ),
                ep_min_constrained_clearance=(float(ep_min_clearance) if np.isfinite(ep_min_clearance) else float("nan")),
                ep_len=ep_len,
                max_episode_steps=max_steps,
                terminated=bool(info.get("terminated", False)),
                truncated=bool(done and ep_len >= max_steps),
                final_distance=final_dist,
                first_goal_hit_step=first_goal_hit_step,
                first_goal_reward_sum=(first_goal_reward_sum if first_goal_hit_step is not None else float(ep_ret)),
                first_goal_dense_reward_sum=(
                    first_goal_dense_reward_sum if first_goal_hit_step is not None else float(ep_reward_dense)
                ),
            )
        )

    if save_episode_plots and plot_dir is not None and saved_episode_plot_paths:
        sheet_path = plot_episode_contact_sheet(
            image_paths=saved_episode_plot_paths,
            output_path=plot_dir / "episode_contact_sheet.png",
            title=f"SafetyGym minimal eval step {int(step_value or 0)}",
            max_cols=3,
        )
        if sheet_path is not None:
            print(f"[EvalPlots] saved {sheet_path}", flush=True)
    env.close()
    return augment_rollout_summary(win.summary("eval"), "eval")


def _save_checkpoint(
    *,
    path: Path,
    actor: Actor,
    critic: Critic,
    critic_target: Critic,
    actor_optimizer,
    critic_optimizer,
    alpha_optimizer,
    log_alpha: torch.Tensor,
    obs_normalizer,
    step: int,
    save_optimizer_state: bool,
) -> None:
    def _cpuify(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: _cpuify(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_cpuify(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_cpuify(v) for v in value)
        return value

    payload = {
        "actor_state_dict": _cpuify(actor.state_dict()),
        "critic_state_dict": _cpuify(critic.state_dict()),
        "critic_target_state_dict": _cpuify(critic_target.state_dict()),
        "log_alpha": log_alpha.detach().cpu(),
        "obs_normalizer_state_dict": _cpuify(obs_normalizer.state_dict()) if hasattr(obs_normalizer, "state_dict") else {},
        "global_step": int(step),
    }
    if save_optimizer_state:
        payload.update(
            {
                "actor_optimizer_state_dict": _cpuify(actor_optimizer.state_dict()),
                "critic_optimizer_state_dict": _cpuify(critic_optimizer.state_dict()),
                "alpha_optimizer_state_dict": _cpuify(alpha_optimizer.state_dict()),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _select_student_action(
    actor,
    obs: np.ndarray,
    device: torch.device,
    low_np: np.ndarray,
    high_np: np.ndarray,
    *,
    scale_to_env: bool,
    obs_normalizer,
) -> np.ndarray:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
        obs_t = obs_normalizer(obs_t)
        action_norm_t, _, _ = actor(obs_t)
        action = action_norm_t[0].detach().cpu().numpy().astype(np.float32)
    if scale_to_env:
        action = _scale_action_np(action, low_np, high_np)
    return _clip_to_bounds(action, low_np, high_np)


def _clip_to_bounds(action: np.ndarray, low_np: np.ndarray, high_np: np.ndarray) -> np.ndarray:
    return np.clip(action, low_np, high_np).astype(np.float32, copy=False)


def _sample_pref_batch(pref_pairs: list[Dict[str, np.ndarray]], batch_size: int, device: torch.device):
    if not pref_pairs:
        return None
    n = min(int(batch_size), len(pref_pairs))
    if n <= 0:
        return None
    idx = np.random.choice(len(pref_pairs), size=n, replace=False)
    obs = torch.as_tensor(np.stack([pref_pairs[i]["obs"] for i in idx], axis=0), device=device, dtype=torch.float32)
    teacher = torch.as_tensor(
        np.stack([pref_pairs[i]["teacher_actions"] for i in idx], axis=0), device=device, dtype=torch.float32
    )
    student = torch.as_tensor(
        np.stack([pref_pairs[i]["student_actions"] for i in idx], axis=0), device=device, dtype=torch.float32
    )
    return {
        "obs": obs,
        "teacher_actions": teacher,
        "student_actions": student,
    }


def _sample_linked_pref_batch(replay_buffer: SimpleReplayBuffer, batch_size: int, device: torch.device):
    if int(getattr(replay_buffer, "n_env", 1)) != 1:
        raise ValueError("Linked preference sampling expects a single-env replay buffer.")
    cap = int(replay_buffer.env_capacities[0])
    if cap <= 1 or int(replay_buffer.filled[0].item()) <= 0:
        return None
    valid_mask = (
        replay_buffer.transition_ready[0, :cap]
        & replay_buffer.valid_next_mask[0, :cap]
        & replay_buffer.teacher_intervened[0, :cap]
    )
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if valid_indices.numel() <= 0:
        return None
    n = min(int(batch_size), int(valid_indices.numel()))
    if n <= 0:
        return None
    sample_ids = torch.randint(0, valid_indices.numel(), (n,), device=replay_buffer.storage_device)
    idx = valid_indices.index_select(0, sample_ids)
    obs = replay_buffer._gather_observations(0, idx).to(device, non_blocking=True)
    teacher = replay_buffer.actions[0, idx].to(device=device, dtype=torch.float32, non_blocking=True)
    student = replay_buffer.student_actions[0, idx].to(device=device, dtype=torch.float32, non_blocking=True)
    return {
        "obs": obs,
        "teacher_actions": teacher,
        "student_actions": student,
    }


def _maybe_load_checkpoint(*, args, sac: SACTensors, obs_normalizer, device: torch.device) -> None:
    checkpoint_path = str(getattr(args, "init_checkpoint_path", "") or "").strip()
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if bool(getattr(args, "load_actor_from_checkpoint", True)) and "actor_state_dict" in checkpoint:
        sac.actor.load_state_dict(checkpoint["actor_state_dict"])
    if bool(getattr(args, "load_critic_from_checkpoint", True)) and "critic_state_dict" in checkpoint:
        sac.critic.load_state_dict(checkpoint["critic_state_dict"])
    if bool(getattr(args, "load_critic_target_from_checkpoint", True)) and "critic_target_state_dict" in checkpoint:
        sac.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
    elif bool(getattr(args, "load_critic_from_checkpoint", True)) and "critic_state_dict" in checkpoint:
        sac.critic_target.load_state_dict(sac.critic.state_dict())
    if bool(getattr(args, "load_alpha_from_checkpoint", True)) and "log_alpha" in checkpoint:
        log_alpha = torch.as_tensor(checkpoint["log_alpha"], device=device, dtype=torch.float32).reshape_as(sac.log_alpha)
        sac.log_alpha.data.copy_(log_alpha)
    if bool(getattr(args, "obs_normalization", True)) and hasattr(obs_normalizer, "load_state_dict"):
        state = checkpoint.get("obs_normalizer_state_dict")
        if state:
            obs_normalizer.load_state_dict(state, strict=False)
    if bool(getattr(args, "load_optimizer_state_from_checkpoint", False)):
        if "actor_optimizer_state_dict" in checkpoint:
            sac.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            sac.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        if "alpha_optimizer_state_dict" in checkpoint:
            sac.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])


def _reset_critic_stack(
    *,
    sac: SACTensors,
    module_impl: str,
    obs_dim: int,
    act_dim: int,
    hidden_critic: int,
    use_layer_norm: bool,
    layer_norm_eps: float,
    lr_critic: float,
    weight_decay: float,
    device: torch.device,
) -> None:
    if module_impl == "custom":
        critic = SafetyCritic(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=hidden_critic,
            num_critics=2,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
            device=device,
        )
        critic_target = SafetyCritic(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=hidden_critic,
            num_critics=2,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
            device=device,
        )
    else:
        critic = Critic(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=hidden_critic,
            device=device,
        )
        critic_target = Critic(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=hidden_critic,
            device=device,
        )
    critic_target.load_state_dict(critic.state_dict())
    sac.critic = critic
    sac.critic_target = critic_target
    sac.critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=lr_critic, weight_decay=weight_decay)


def run_minimal_training(args) -> None:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.set_num_threads(int(args.torch_num_threads))
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(int(args.torch_num_interop_threads))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    log_dir, model_dir = _prepare_run_dirs(args)
    wandb_run = _maybe_init_wandb(args, log_dir)
    variant = str(getattr(args, "variant", "plain")).strip().lower()
    if variant not in {"plain", "own", "pvp", "hilserl"}:
        raise ValueError(f"Unsupported variant: {variant}")

    controller = None
    env = _make_env_with_wrappers(args=args, seed=int(args.seed), with_intervention=False, controller=None)
    obs, _ = env.reset(seed=int(args.seed))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    obs_dim = int(obs.shape[0])
    act_dim = int(np.prod(env.action_space.shape))
    low_np = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high_np = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    low_t = torch.as_tensor(low_np, device=device, dtype=torch.float32).view(1, -1)
    high_t = torch.as_tensor(high_np, device=device, dtype=torch.float32).view(1, -1)
    if bool(getattr(args, "use_intervention", False)):
        env.close()
        controller = build_human_controller(
            input_device=str(getattr(args, "human_input_device", "keyboard")),
            action_dim=act_dim,
            obs_dim=obs_dim,
            env_name=str(args.env_name),
            action_scale=float(getattr(args, "human_action_scale", 1.0)),
            wheel_command_limit=float(getattr(args, "car_wheel_command_limit", 2.0)),
            overlay_fps_limit=int(getattr(args, "controller_fps_limit", 0)),
            overlay_draw_hz=float(getattr(args, "controller_overlay_hz", 20.0)),
            gamepad_mode=str(getattr(args, "gamepad_mode", "local")),
            gamepad_host=str(getattr(args, "gamepad_host", "")),
            gamepad_port=int(getattr(args, "gamepad_port", 0) or DEFAULT_SAFETY_GAMEPAD_PORT),
            gamepad_cache_path=_resolved_gamepad_cache_path(args),
            gamepad_reconnect_seconds=float(getattr(args, "gamepad_reconnect_seconds", 2.0)),
            gamepad_config_path=_resolved_gamepad_config_path(args),
            gamepad_use_saved_config=bool(getattr(args, "gamepad_use_saved_config", True)),
            gamepad_device_index=int(getattr(args, "gamepad_device_index", 0)),
            action_low=low_np,
            action_high=high_np,
            expert_checkpoint_path=str(getattr(args, "expert_checkpoint_path", "") or ""),
            expert_safe_checkpoint_path=str(getattr(args, "expert_safe_checkpoint_path", "") or ""),
            expert_switch_clearance_threshold=float(getattr(args, "expert_switch_clearance_threshold", 0.08)),
            expert_device=str(getattr(args, "expert_device", "cpu")),
            prefer_separate_keyboard_window=str(getattr(args, "render_mode", "none")).lower() == "human",
        )
        env = _make_env_with_wrappers(args=args, seed=int(args.seed), with_intervention=True, controller=controller)
        obs, _ = env.reset(seed=int(args.seed))
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

    module_impl = str(getattr(args, "module_impl", "fastsac")).strip().lower()
    if module_impl == "custom":
        actor = SafetyActor(
            n_obs=obs_dim,
            n_act=act_dim,
            num_envs=1,
            init_scale=float(args.init_scale),
            hidden_dim=int(args.actor_hidden_dim),
            use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
            layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
            device=device,
        )
        critic = SafetyCritic(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=int(args.critic_hidden_dim),
            num_critics=2,
            use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
            layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
            device=device,
        )
        critic_target = SafetyCritic(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=int(args.critic_hidden_dim),
            num_critics=2,
            use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
            layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
            device=device,
        )
    else:
        actor = Actor(
            n_obs=obs_dim,
            n_act=act_dim,
            num_envs=1,
            init_scale=float(args.init_scale),
            hidden_dim=int(args.actor_hidden_dim),
            device=device,
        )
        critic = Critic(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=int(args.critic_hidden_dim),
            device=device,
        )
        critic_target = Critic(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=int(args.critic_hidden_dim),
            device=device,
        )
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(args.actor_learning_rate),
        weight_decay=float(args.weight_decay),
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=float(args.critic_learning_rate),
        weight_decay=float(args.weight_decay),
    )
    log_alpha = torch.ones(1, requires_grad=True, device=device)
    log_alpha.data.copy_(torch.tensor([np.log(float(args.alpha_init))], device=device))
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=float(args.critic_learning_rate))
    sac = SACTensors(
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        log_alpha=log_alpha,
        target_entropy=-float(act_dim),
        pref_lambda=float(getattr(args, "pref_lambda_init", 1.0)),
        pref_violation_ema=0.0,
    )

    if bool(args.obs_normalization):
        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    else:
        obs_normalizer = torch.nn.Identity()
    _maybe_load_checkpoint(args=args, sac=sac, obs_normalizer=obs_normalizer, device=device)

    def _make_rb() -> SimpleReplayBuffer:
        return SimpleReplayBuffer(
            n_env=1,
            buffer_size=int(args.buffer_size),
            n_obs=obs_dim,
            n_act=act_dim,
            n_critic_obs=obs_dim,
            asymmetric_obs=False,
            n_steps=1,
            gamma=float(args.gamma),
            device=device,
        )

    main_rb = _make_rb()
    demo_rb = (
        _make_rb()
        if (
            variant in {"own", "hilserl"}
            and (
                float(getattr(args, "demo_sample_ratio", 0.0)) > 0.0
                or int(getattr(args, "prefill_demo_episodes", 0)) > 0
                or bool(getattr(args, "store_intervened_in_demo_buffer", False))
                or int(getattr(args, "demo_pretrain_updates", 0)) > 0
                or (
                    (
                        bool(str(getattr(args, "demo_dataset_path", "") or "").strip())
                        or bool(getattr(args, "demo_dataset_auto_load", False))
                    )
                    and str(getattr(args, "demo_dataset_target", "variant")).strip().lower() in {"demo", "variant"}
                )
            )
        )
        else None
    )
    novice_rb = _make_rb() if variant == "pvp" else None
    human_rb = _make_rb() if variant == "pvp" else None
    pref_pairs: list[Dict[str, np.ndarray]] = []
    pref_capacity = int(max(0, getattr(args, "pref_capacity", 0)))
    pref_sampling_mode = str(getattr(args, "pref_sampling_mode", "linked")).strip().lower()
    if pref_sampling_mode not in {"linked", "separate"}:
        raise ValueError(f"Unsupported pref_sampling_mode: {pref_sampling_mode}")
    dataset_target = str(getattr(args, "demo_dataset_target", "variant")).strip().lower()
    dataset_requested = bool(str(getattr(args, "demo_dataset_path", "") or "").strip()) or bool(
        getattr(args, "demo_dataset_auto_load", False)
    )

    demo_dataset_path = str(getattr(args, "demo_dataset_path", "") or "").strip()
    if not demo_dataset_path and bool(getattr(args, "demo_dataset_auto_load", False)):
        dataset_root = str(getattr(args, "demo_dataset_dir", "") or "").strip() or str(DEFAULT_SAFETYGYM_DATASET_DIR)
        found = find_latest_transition_dataset(env_name=args.env_name, dataset_dir=dataset_root)
        if found is not None:
            demo_dataset_path = str(found)
            print(f"[Dataset] auto-selected SafetyGym dataset {demo_dataset_path}", flush=True)
    if demo_dataset_path:
        max_rows = int(getattr(args, "demo_dataset_max_rows", 0))
        if dataset_target == "replay":
            stats = extend_buffer_from_dataset(
                buffer=main_rb,
                dataset_path=demo_dataset_path,
                device=device,
                max_rows=max_rows,
                expected_obs_dim=obs_dim,
                expected_act_dim=act_dim,
            )
            print(f"[Dataset] loaded {stats['rows_loaded']} rows into replay buffer", flush=True)
        elif dataset_target == "demo":
            if demo_rb is None:
                raise ValueError("--demo_dataset_target demo requires a demo buffer.")
            stats = extend_buffer_from_dataset(
                buffer=demo_rb,
                dataset_path=demo_dataset_path,
                device=device,
                max_rows=max_rows,
                expected_obs_dim=obs_dim,
                expected_act_dim=act_dim,
            )
            print(f"[Dataset] loaded {stats['rows_loaded']} rows into demo buffer", flush=True)
        else:
            variant_data = load_transition_dataset(demo_dataset_path)
            stats_main = extend_buffer_from_dataset(
                buffer=main_rb,
                dataset_path=demo_dataset_path,
                device=device,
                max_rows=max_rows,
                expected_obs_dim=obs_dim,
                expected_act_dim=act_dim,
            )
            print(f"[Dataset] loaded {stats_main['rows_loaded']} rows into replay buffer", flush=True)
            if variant == "hilserl":
                if demo_rb is None:
                    raise ValueError("variant dataset load for hilserl requires a demo buffer.")
                stats_demo = extend_buffer_from_dataset(
                    buffer=demo_rb,
                    dataset_path=demo_dataset_path,
                    device=device,
                    max_rows=max_rows,
                    expected_obs_dim=obs_dim,
                    expected_act_dim=act_dim,
                    filter_mode="intervened",
                )
                print(f"[Dataset] loaded {stats_demo['rows_loaded']} intervened rows into demo buffer", flush=True)
            elif variant == "pvp":
                if human_rb is None or novice_rb is None:
                    raise ValueError("variant dataset load for pvp requires human_rb and novice_rb.")
                stats_h = extend_buffer_from_dataset(
                    buffer=human_rb,
                    dataset_path=demo_dataset_path,
                    device=device,
                    max_rows=max_rows,
                    expected_obs_dim=obs_dim,
                    expected_act_dim=act_dim,
                    filter_mode="intervened",
                )
                stats_n = extend_buffer_from_dataset(
                    buffer=novice_rb,
                    dataset_path=demo_dataset_path,
                    device=device,
                    max_rows=max_rows,
                    expected_obs_dim=obs_dim,
                    expected_act_dim=act_dim,
                    filter_mode="non_intervened",
                )
                print(
                    f"[Dataset] loaded {stats_h['rows_loaded']} intervened rows into human_rb "
                    f"and {stats_n['rows_loaded']} non-intervened rows into novice_rb",
                    flush=True,
                )
            elif variant == "own" and pref_capacity > 0 and pref_sampling_mode != "linked":
                teacher_intervened = np.asarray(variant_data["teacher_intervened"], dtype=np.bool_)
                teacher_actions = np.asarray(variant_data["actions"], dtype=np.float32)
                student_actions = np.asarray(variant_data["student_actions"], dtype=np.float32)
                observations = np.asarray(variant_data["observations"], dtype=np.float32)
                for idx in np.nonzero(teacher_intervened)[0].tolist():
                    pref_pairs.append(
                        {
                            "obs": observations[idx].copy(),
                            "teacher_actions": clip_action_to_space(teacher_actions[idx], env.action_space),
                            "student_actions": clip_action_to_space(student_actions[idx], env.action_space),
                        }
                    )
                if len(pref_pairs) > pref_capacity:
                    pref_pairs = pref_pairs[-pref_capacity:]
                print(f"[Dataset] loaded {len(pref_pairs)} preference pairs from intervened dataset rows", flush=True)

    def _reset_obs(reset_seed: int | None):
        if reset_seed is None:
            ob, _info = env.reset()
        else:
            ob, _info = env.reset(seed=reset_seed)
        return np.asarray(ob, dtype=np.float32).reshape(-1)

    prefill_steps = 0
    prefill_logs: Dict[str, float] = {}
    if int(getattr(args, "prefill_demo_episodes", 0)) > 0:
        if not bool(getattr(args, "use_intervention", False)):
            raise ValueError("--prefill_demo_episodes requires --use_intervention so teacher actions can be collected.")
        total_prefill_steps = 0
        demo_steps = 0
        episode_cap = int(max(0, getattr(args, "prefill_max_steps_per_episode", 0)))
        prefill_policy = str(getattr(args, "prefill_policy", "student")).strip().lower()
        for ep_idx in range(int(args.prefill_demo_episodes)):
            obs = _reset_obs(int(args.seed) + 10_000 + ep_idx)
            ep_steps = 0
            while True:
                if prefill_policy == "random":
                    student_action = env.action_space.sample().astype(np.float32)
                elif prefill_policy == "zero":
                    student_action = np.zeros_like(low_np, dtype=np.float32)
                else:
                    student_action = _select_student_action(
                        sac.actor,
                        obs,
                        device,
                        low_np,
                        high_np,
                        scale_to_env=bool(args.scale_actor_to_env_bounds),
                        obs_normalizer=obs_normalizer,
                    )
                next_obs, reward, cost, terminated, truncated, info = env.step(student_action)
                next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                teacher_intervened = bool(info.get("teacher_intervened", False))
                applied_action = np.asarray(info.get("teacher_action", student_action), dtype=np.float32)
                applied_action = clip_action_to_space(applied_action, env.action_space)
                transition = _build_transition(
                    obs=obs,
                    action=applied_action,
                    next_obs=next_obs,
                    reward=float(reward),
                    done=bool(terminated or truncated),
                    truncated=bool(truncated),
                    device=device,
                    student_action=student_action,
                    teacher_intervened=teacher_intervened,
                )
                main_rb.extend(transition)
                if bool(args.obs_normalization) and hasattr(obs_normalizer, "update"):
                    obs_normalizer.update(torch.as_tensor(obs[None, :], device=device, dtype=torch.float32))
                    obs_normalizer.update(torch.as_tensor(next_obs[None, :], device=device, dtype=torch.float32))
                if variant == "pvp":
                    if teacher_intervened and human_rb is not None:
                        human_rb.extend(transition)
                    elif novice_rb is not None:
                        novice_rb.extend(transition)
                else:
                    if teacher_intervened and demo_rb is not None and (
                        variant == "hilserl" or bool(getattr(args, "store_intervened_in_demo_buffer", False))
                    ):
                        demo_rb.extend(transition)
                    if variant == "own" and teacher_intervened and pref_capacity > 0 and pref_sampling_mode != "linked":
                        pref_pairs.append(
                            {
                                "obs": obs.copy(),
                                "teacher_actions": applied_action.copy(),
                                "student_actions": student_action.copy(),
                            }
                        )
                        if len(pref_pairs) > pref_capacity:
                            pref_pairs = pref_pairs[-pref_capacity:]
                total_prefill_steps += 1
                ep_steps += 1
                if teacher_intervened:
                    demo_steps += 1
                obs = next_obs
                force_end = bool(episode_cap > 0 and ep_steps >= episode_cap and not (terminated or truncated))
                if terminated or truncated or force_end:
                    break
        prefill_steps = int(total_prefill_steps)
        prefill_logs = {
            "train/prefill_episodes": float(int(args.prefill_demo_episodes)),
            "train/prefill_steps": float(total_prefill_steps),
            "train/prefill_demo_steps": float(demo_steps),
            "train/prefill_demo_fraction": float(demo_steps / max(1, total_prefill_steps)),
        }
        print(json.dumps(prefill_logs, sort_keys=True), flush=True)
        if wandb_run is not None:
            wandb_run.log(prefill_logs, step=0)

    if int(getattr(args, "demo_pretrain_updates", 0)) > 0 and demo_rb is not None:
        demo_batch_size = int(getattr(args, "demo_pretrain_batch_size", 0)) or int(args.batch_size)
        if demo_rb.size >= demo_batch_size:
            pretrain_updates = int(args.demo_pretrain_updates)
            pretrain_last: SACUpdateMetrics | None = None
            for update_idx in range(1, pretrain_updates + 1):
                pretrain_last = sac_update_step(
                    sac=sac,
                    batch=demo_rb.sample(demo_batch_size),
                    gamma=float(args.gamma),
                    tau=float(args.tau),
                    max_grad_norm=float(args.max_grad_norm),
                    obs_preprocess=obs_normalizer,
                    pref_batch=None,
                    pref_rank_weight=0.0,
                    pref_rank_margin=float(getattr(args, "pref_rank_margin", 0.1)),
                    pref_loss_type=str(getattr(args, "pref_loss_type", "margin")),
                    pref_stopgrad_positive=bool(getattr(args, "pref_stopgrad_positive", False)),
                    pref_lambda_lr=float(getattr(args, "pref_lambda_lr", 1e-3)),
                    pref_lambda_max=float(getattr(args, "pref_lambda_max", 10.0)),
                    pref_lambda_ema=float(getattr(args, "pref_lambda_ema", 0.9)),
                    pref_violation_clip=float(getattr(args, "pref_violation_clip", 10.0)),
                    pref_violation_target=float(getattr(args, "pref_violation_target", 0.0)),
                    pref_lagrangian_violation_type=str(getattr(args, "pref_lagrangian_violation_type", "hinge")),
                    alpha_min=float(args.alpha_min),
                    alpha_max=float(args.alpha_max),
                    scale_actor_to_env_bounds=bool(args.scale_actor_to_env_bounds),
                    action_low=low_t,
                    action_high=high_t,
                    update_actor=(update_idx % int(max(1, args.policy_frequency)) == 0),
                    critic_loss_reduction=str(args.critic_loss_reduction),
                )
            if bool(getattr(args, "critic_reset_after_pretrain", False)):
                _reset_critic_stack(
                    sac=sac,
                    module_impl=module_impl,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    hidden_critic=int(args.critic_hidden_dim),
                    use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
                    layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
                    lr_critic=float(args.critic_learning_rate),
                    weight_decay=float(args.weight_decay),
                    device=device,
                )
            if pretrain_last is not None:
                pretrain_logs = {
                    "train/pretrain_updates_run": float(pretrain_updates),
                    "train/pretrain_critic_loss_mean": float(pretrain_last.critic_loss),
                    "train/pretrain_actor_loss_mean": float(pretrain_last.actor_loss),
                    "train/pretrain_target_q_mean": float(pretrain_last.target_q_mean),
                }
                print(json.dumps(pretrain_logs, sort_keys=True), flush=True)
                if wandb_run is not None:
                    wandb_run.log(pretrain_logs, step=0)

    if prefill_steps > 0:
        obs = _reset_obs(int(args.seed) + 20_000)

    win = EpisodeWindow(size=100)
    max_steps = extract_step_limit(env)
    start_time = time.time()
    next_log = int(args.log_interval)
    next_eval = int(args.eval_interval)
    next_save = int(args.save_interval)
    effective_learning_starts = max(0, int(args.learning_starts) - prefill_steps)

    ep_ret = 0.0
    ep_cost = 0.0
    ep_reward_raw_env = 0.0
    ep_reward_dense = 0.0
    ep_reward_sparse = 0.0
    ep_reward_step_penalty = 0.0
    ep_reward_cost_penalty = 0.0
    ep_reward_clearance_penalty = 0.0
    ep_reward_forward = 0.0
    ep_reward_backward_penalty = 0.0
    ep_reward_heading = 0.0
    ep_clearance_sum = 0.0
    ep_clearance_count = 0
    ep_min_clearance = float("inf")
    ep_len = 0
    ep_goal_hit_count = 0
    ep_first_goal_hit_step: int | None = None
    ep_first_goal_reward_sum = 0.0
    ep_first_goal_dense_reward_sum = 0.0
    total_updates = 0
    episode_idx = 0
    last_update: SACUpdateMetrics | None = None

    for step in range(1, int(args.total_timesteps) + 1):
        if step <= int(effective_learning_starts):
            student_action = env.action_space.sample().astype(np.float32)
        else:
            student_action = _select_student_action(
                sac.actor,
                obs,
                device,
                low_np,
                high_np,
                scale_to_env=bool(args.scale_actor_to_env_bounds),
                obs_normalizer=obs_normalizer,
            )

        next_obs, reward, cost, terminated, truncated, info = env.step(student_action)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        if bool(info.get("goal_met", False)):
            ep_goal_hit_count += 1
        teacher_intervened = bool(info.get("teacher_intervened", False))
        applied_action = np.asarray(info.get("teacher_action", student_action), dtype=np.float32)
        applied_action = clip_action_to_space(applied_action, env.action_space)

        transition = _build_transition(
            obs=obs,
            action=applied_action,
            next_obs=next_obs,
            reward=float(reward),
            done=bool(terminated or truncated),
            truncated=bool(truncated),
            device=device,
            student_action=student_action,
            teacher_intervened=teacher_intervened,
        )
        main_rb.extend(transition)

        if bool(args.obs_normalization) and hasattr(obs_normalizer, "update"):
            obs_normalizer.update(torch.as_tensor(obs[None, :], device=device, dtype=torch.float32))
            obs_normalizer.update(torch.as_tensor(next_obs[None, :], device=device, dtype=torch.float32))

        if variant == "pvp":
            if teacher_intervened and human_rb is not None:
                human_rb.extend(transition)
            elif novice_rb is not None:
                novice_rb.extend(transition)
        else:
            if teacher_intervened and demo_rb is not None and (
                variant == "hilserl" or bool(getattr(args, "store_intervened_in_demo_buffer", False))
            ):
                demo_rb.extend(transition)
            if variant == "own" and teacher_intervened and pref_capacity > 0 and pref_sampling_mode != "linked":
                pref_pairs.append(
                    {
                        "obs": obs.copy(),
                        "teacher_actions": applied_action.copy(),
                        "student_actions": student_action.copy(),
                    }
                )
                if len(pref_pairs) > pref_capacity:
                    pref_pairs = pref_pairs[-pref_capacity:]

        obs = next_obs
        ep_ret += float(reward)
        ep_cost += float(cost)
        ep_reward_raw_env += float(info.get("reward_raw_env", 0.0))
        ep_reward_dense += float(info.get("reward_dense_component", 0.0))
        ep_reward_sparse += float(info.get("reward_sparse_component", 0.0))
        ep_reward_step_penalty += float(info.get("reward_step_penalty_component", 0.0))
        ep_reward_cost_penalty += float(info.get("reward_cost_penalty_component", 0.0))
        ep_reward_clearance_penalty += float(info.get("reward_clearance_penalty_component", 0.0))
        ep_reward_forward += float(info.get("reward_forward_component", 0.0))
        ep_reward_backward_penalty += float(info.get("reward_backward_penalty_component", 0.0))
        ep_reward_heading += float(info.get("reward_heading_component", 0.0))
        step_clearance = info.get("min_constrained_clearance", float("nan"))
        if np.isfinite(step_clearance):
            ep_clearance_sum += float(step_clearance)
            ep_clearance_count += 1
            ep_min_clearance = min(ep_min_clearance, float(step_clearance))
        ep_len += 1
        if bool(info.get("goal_met", False)) and ep_first_goal_hit_step is None:
            ep_first_goal_hit_step = int(ep_len)
            ep_first_goal_reward_sum = float(ep_ret)
            ep_first_goal_dense_reward_sum = float(ep_reward_dense)

        rb_ready = main_rb.size >= int(args.batch_size)
        if variant == "pvp" and novice_rb is not None:
            rb_ready = novice_rb.size >= max(1, int(args.batch_size // 2))

        if step > int(effective_learning_starts) and rb_ready:
            for _ in range(int(args.num_updates)):
                total_updates += 1
                if variant == "pvp" and novice_rb is not None and human_rb is not None:
                    half = max(1, int(args.batch_size // 2))
                    if human_rb.size >= half and novice_rb.size >= max(1, int(args.batch_size - half)):
                        batch_n = novice_rb.sample(max(1, int(args.batch_size - half)))
                        batch_h = human_rb.sample(half)
                        batch = TensorDict.cat([batch_n, batch_h], dim=0)
                    elif novice_rb.size >= int(args.batch_size):
                        batch = novice_rb.sample(int(args.batch_size))
                    else:
                        batch = main_rb.sample(int(args.batch_size))
                elif (
                    variant in {"own", "hilserl"}
                    and demo_rb is not None
                    and demo_rb.size > 0
                    and float(getattr(args, "demo_sample_ratio", 0.0)) > 0.0
                ):
                    demo_n = int(max(1, args.batch_size * float(getattr(args, "demo_sample_ratio", 0.0))))
                    base_n = max(1, int(args.batch_size - demo_n))
                    if main_rb.size >= base_n and demo_rb.size >= demo_n:
                        batch = TensorDict.cat([main_rb.sample(base_n), demo_rb.sample(demo_n)], dim=0)
                    else:
                        batch = main_rb.sample(int(args.batch_size))
                else:
                    batch = main_rb.sample(int(args.batch_size))

                pref_batch = None
                if variant == "own" and float(getattr(args, "pref_sample_ratio", 0.0)) > 0.0:
                    pref_n = max(1, int(args.batch_size * float(getattr(args, "pref_sample_ratio", 0.0))))
                    if pref_sampling_mode == "linked":
                        pref_batch = _sample_linked_pref_batch(main_rb, pref_n, device)
                    else:
                        pref_batch = _sample_pref_batch(pref_pairs, pref_n, device)

                last_update = sac_update_step(
                    sac=sac,
                    batch=batch,
                    gamma=float(args.gamma),
                    tau=float(args.tau),
                    max_grad_norm=float(args.max_grad_norm),
                    obs_preprocess=obs_normalizer,
                    pref_batch=pref_batch,
                    pref_rank_weight=float(getattr(args, "pref_rank_weight", 0.0)) if variant == "own" else 0.0,
                    pref_rank_margin=float(getattr(args, "pref_rank_margin", 0.1)),
                    pref_loss_type=str(getattr(args, "pref_loss_type", "margin")),
                    pref_stopgrad_positive=bool(getattr(args, "pref_stopgrad_positive", False)),
                    pref_lambda_lr=float(getattr(args, "pref_lambda_lr", 1e-3)),
                    pref_lambda_max=float(getattr(args, "pref_lambda_max", 10.0)),
                    pref_lambda_ema=float(getattr(args, "pref_lambda_ema", 0.9)),
                    pref_violation_clip=float(getattr(args, "pref_violation_clip", 10.0)),
                    pref_violation_target=float(getattr(args, "pref_violation_target", 0.0)),
                    pref_lagrangian_violation_type=str(getattr(args, "pref_lagrangian_violation_type", "hinge")),
                    alpha_min=float(args.alpha_min),
                    alpha_max=float(args.alpha_max),
                    scale_actor_to_env_bounds=bool(args.scale_actor_to_env_bounds),
                    action_low=low_t,
                    action_high=high_t,
                    update_actor=(total_updates % int(max(1, args.policy_frequency)) == 0),
                    critic_loss_reduction=str(args.critic_loss_reduction),
                )

        if terminated or truncated:
            final_dist = extract_goal_distance(env)
            info = dict(info)
            info["goal_hit_count"] = ep_goal_hit_count
            win.add(
                _episode_metrics(
                    info=info,
                    ep_return=ep_ret,
                    ep_cost=ep_cost,
                    ep_reward_raw_env=ep_reward_raw_env,
                    ep_reward_dense=ep_reward_dense,
                    ep_reward_sparse=ep_reward_sparse,
                    ep_reward_step_penalty=ep_reward_step_penalty,
                    ep_reward_cost_penalty=ep_reward_cost_penalty,
                    ep_reward_clearance_penalty=ep_reward_clearance_penalty,
                    ep_reward_forward=ep_reward_forward,
                    ep_reward_backward_penalty=ep_reward_backward_penalty,
                    ep_reward_heading=ep_reward_heading,
                    ep_mean_constrained_clearance=(
                        float(ep_clearance_sum / max(1, ep_clearance_count)) if ep_clearance_count > 0 else float("nan")
                    ),
                    ep_min_constrained_clearance=(
                        float(ep_min_clearance) if np.isfinite(ep_min_clearance) else float("nan")
                    ),
                    ep_len=ep_len,
                    max_episode_steps=max_steps,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    final_distance=final_dist,
                    first_goal_hit_step=ep_first_goal_hit_step,
                    first_goal_reward_sum=(
                        ep_first_goal_reward_sum if ep_first_goal_hit_step is not None else float(ep_ret)
                    ),
                    first_goal_dense_reward_sum=(
                        ep_first_goal_dense_reward_sum if ep_first_goal_hit_step is not None else float(ep_reward_dense)
                    ),
                )
            )
            episode_idx += 1
            reset_seed = int(args.seed + episode_idx) if bool(getattr(args, "reseed_on_episode_reset", False)) else None
            obs = _reset_obs(reset_seed)
            ep_ret = 0.0
            ep_cost = 0.0
            ep_reward_raw_env = 0.0
            ep_reward_dense = 0.0
            ep_reward_sparse = 0.0
            ep_reward_step_penalty = 0.0
            ep_reward_cost_penalty = 0.0
            ep_reward_clearance_penalty = 0.0
            ep_reward_forward = 0.0
            ep_reward_backward_penalty = 0.0
            ep_reward_heading = 0.0
            ep_clearance_sum = 0.0
            ep_clearance_count = 0
            ep_min_clearance = float("inf")
            ep_len = 0
            ep_goal_hit_count = 0
            ep_first_goal_hit_step = None
            ep_first_goal_reward_sum = 0.0
            ep_first_goal_dense_reward_sum = 0.0

        if step >= next_log:
            next_log += int(args.log_interval)
            elapsed = max(1e-6, time.time() - start_time)
            logs: Dict[str, float] = {
                "train/step": float(step),
                "train/fps": float(step / elapsed),
                "train/replay_size": float(main_rb.size),
                "train/buffer_main_size": float(main_rb.size),
            }
            if demo_rb is not None:
                logs["train/buffer_demo_size"] = float(demo_rb.size)
            if novice_rb is not None:
                logs["train/buffer_novice_size"] = float(novice_rb.size)
            if human_rb is not None:
                logs["train/buffer_human_size"] = float(human_rb.size)
            if variant == "own":
                if pref_sampling_mode == "linked":
                    logs["train/buffer_pref_size"] = float(main_rb.linked_pref_pair_count())
                else:
                    logs["train/buffer_pref_size"] = float(len(pref_pairs))
            logs.update(augment_rollout_summary(win.summary("train"), "train"))
            if last_update is not None:
                logs.update(
                    {
                        "train/critic_loss": float(last_update.critic_loss),
                        "train/actor_loss": float(last_update.actor_loss),
                        "train/alpha_loss": float(last_update.alpha_loss),
                        "train/alpha": float(last_update.alpha),
                        "train/target_q_mean": float(last_update.target_q_mean),
                        "train/q_min_pi_mean": float(last_update.q_min_pi_mean),
                        "train/q_min_data_mean": float(last_update.q_min_data_mean),
                        "train/q_gap_mean": float(last_update.q_disagreement_data_mean),
                        "train/replay_reward_mean": float(last_update.replay_reward_mean),
                        "train/actor_updates_per_iter": float(last_update.actor_updates),
                        "train/alpha_updates_per_iter": float(last_update.alpha_updates),
                    }
                )
                if variant == "own":
                    logs.update(
                        {
                            "train/critic_loss_pref": float(last_update.critic_loss_pref),
                            "train/critic_loss_pref_weighted": float(last_update.critic_loss_pref_weighted),
                            "train/pref_q_delta": float(last_update.pref_q_delta),
                            "train/pref_lambda": float(last_update.pref_lambda),
                            "train/pref_violation": float(last_update.pref_violation),
                            "train/pref_violation_ema": float(last_update.pref_violation_ema),
                        }
                    )
            print(json.dumps(logs, sort_keys=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(logs, step=step)

        if int(args.eval_interval) > 0 and step >= next_eval:
            next_eval += int(args.eval_interval)
            eval_logs = _run_eval(
                sac.actor,
                args,
                device,
                obs_normalizer=obs_normalizer,
                log_dir=log_dir,
                step_value=step,
            )
            eval_logs["eval/step"] = float(step)
            print(json.dumps(eval_logs, sort_keys=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(eval_logs, step=step)

        if int(args.save_interval) > 0 and step >= next_save:
            next_save += int(args.save_interval)
            checkpoint_path = model_dir / f"step_{step}.pt"
            _save_checkpoint(
                path=checkpoint_path,
                actor=sac.actor,
                critic=sac.critic,
                critic_target=sac.critic_target,
                actor_optimizer=sac.actor_optimizer,
                critic_optimizer=sac.critic_optimizer,
                alpha_optimizer=sac.alpha_optimizer,
                log_alpha=sac.log_alpha,
                obs_normalizer=obs_normalizer,
                step=step,
                save_optimizer_state=bool(getattr(args, "save_optimizer_state_in_checkpoints", True)),
            )
            _maybe_render_policy_map(
                args=args,
                checkpoint_path=checkpoint_path,
                step_value=int(step),
                log_dir=log_dir,
                wandb_run=wandb_run,
            )

    final_checkpoint = model_dir / "final.pt"
    _save_checkpoint(
        path=final_checkpoint,
        actor=sac.actor,
        critic=sac.critic,
        critic_target=sac.critic_target,
        actor_optimizer=sac.actor_optimizer,
        critic_optimizer=sac.critic_optimizer,
        alpha_optimizer=sac.alpha_optimizer,
        log_alpha=sac.log_alpha,
        obs_normalizer=obs_normalizer,
        step=int(args.total_timesteps),
        save_optimizer_state=bool(getattr(args, "save_optimizer_state_in_checkpoints", True)),
    )
    _maybe_render_policy_map(
        args=args,
        checkpoint_path=final_checkpoint,
        step_value=int(args.total_timesteps),
        log_dir=log_dir,
        wandb_run=wandb_run,
    )

    final_eval = _run_eval(
        sac.actor,
        args,
        device,
        obs_normalizer=obs_normalizer,
        log_dir=log_dir,
        step_value=int(args.total_timesteps),
    )
    final_eval["eval/step"] = float(args.total_timesteps)
    print(json.dumps(final_eval, sort_keys=True), flush=True)
    export_logs: Dict[str, float] = {}
    export_logs.update(
        _maybe_export_final_buffer_dataset(
            args=args,
            buffer=main_rb,
            env_name=str(args.env_name),
            variant=variant,
            reward_mode=str(args.reward_mode),
            label_suffix="replay",
            explicit_path=str(getattr(args, "export_final_replay_dataset_path", "") or ""),
            enabled=bool(getattr(args, "export_final_replay_dataset", False)),
        )
    )
    export_logs.update(
        _maybe_export_final_buffer_dataset(
            args=args,
            buffer=demo_rb,
            env_name=str(args.env_name),
            variant=variant,
            reward_mode=str(args.reward_mode),
            label_suffix="demo",
            explicit_path=str(getattr(args, "export_final_demo_dataset_path", "") or ""),
            enabled=bool(getattr(args, "export_final_demo_dataset", False)),
            filter_mode="all",
        )
    )
    if wandb_run is not None:
        wandb_run.log(final_eval, step=int(args.total_timesteps))
        if export_logs:
            wandb_run.log(export_logs, step=int(args.total_timesteps))
        wandb_run.finish()
    env.close()
    if controller is not None and hasattr(controller, "close"):
        controller.close()
