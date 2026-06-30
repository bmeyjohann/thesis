from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import sys

import numpy as np
import torch
from tensordict import TensorDict

# Ensure local FastSAC utilities are importable without installation.
_FAST_SAC_PATH = Path(__file__).resolve().parent.parent / "fasttd3" / "fast_sac"
if _FAST_SAC_PATH.exists():
    _fast_sac_path_str = str(_FAST_SAC_PATH)
    if _fast_sac_path_str not in sys.path:
        sys.path.insert(0, _fast_sac_path_str)

from fast_sac_utils import EmpiricalNormalization, SimpleReplayBuffer

from .controllers import build_human_controller
from .dataset_io import (
    DEFAULT_SAFETYGYM_DATASET_DIR,
    build_dataset_path,
    extend_buffer_from_dataset,
    find_latest_transition_dataset,
    load_transition_dataset,
    save_buffer_as_transition_dataset,
)
from .gamepad import DEFAULT_SAFETY_GAMEPAD_CACHE_PATH, DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH, DEFAULT_SAFETY_GAMEPAD_PORT
from .env import clip_action_to_space, extract_goal_distance, extract_step_limit, make_safety_env, resolve_control_scheme, scale_action_np
from .io import save_args_json
from .metrics import EpisodeWindow, augment_rollout_summary, classify_outcome
from .policy_viz import _extract_bounds, _extract_overlay_specs, plot_episode_contact_sheet, plot_eval_episode_trajectory
from .rendering import build_external_viewer, resolve_env_render_mode, wants_external_viewer
from .sac import (
    SACTensors,
    SACUpdateMetrics,
    build_sac,
    compute_q_disagreement,
    reset_critic,
    sac_update_step,
)
from .wrappers import HumanInterventionWrapper, RewardModeWrapper


@dataclass
class RunPaths:
    log_dir: Path
    model_dir: Path


@dataclass
class TimingWindow:
    steps: int = 0
    action_s: float = 0.0
    env_s: float = 0.0
    data_s: float = 0.0
    update_s: float = 0.0
    uncertainty_s: float = 0.0
    episode_end_s: float = 0.0
    misc_s: float = 0.0
    total_s: float = 0.0

    def add(
        self,
        *,
        action_s: float,
        env_s: float,
        data_s: float,
        update_s: float,
        uncertainty_s: float,
        episode_end_s: float,
        misc_s: float,
        total_s: float,
    ) -> None:
        self.steps += 1
        self.action_s += float(action_s)
        self.env_s += float(env_s)
        self.data_s += float(data_s)
        self.update_s += float(update_s)
        self.uncertainty_s += float(uncertainty_s)
        self.episode_end_s += float(episode_end_s)
        self.misc_s += float(misc_s)
        self.total_s += float(total_s)

    def summary(self, prefix: str) -> Dict[str, float]:
        if self.steps <= 0:
            return {}
        out: Dict[str, float] = {
            f"{prefix}/steps": float(self.steps),
            f"{prefix}/step_ms_mean": float((self.total_s / self.steps) * 1e3),
            f"{prefix}/action_ms_mean": float((self.action_s / self.steps) * 1e3),
            f"{prefix}/env_step_ms_mean": float((self.env_s / self.steps) * 1e3),
            f"{prefix}/data_ms_mean": float((self.data_s / self.steps) * 1e3),
            f"{prefix}/update_ms_mean": float((self.update_s / self.steps) * 1e3),
            f"{prefix}/uncertainty_ms_mean": float((self.uncertainty_s / self.steps) * 1e3),
            f"{prefix}/episode_end_ms_mean": float((self.episode_end_s / self.steps) * 1e3),
            f"{prefix}/misc_ms_mean": float((self.misc_s / self.steps) * 1e3),
        }
        denom = max(1e-9, self.total_s)
        out[f"{prefix}/pct_action"] = float(100.0 * self.action_s / denom)
        out[f"{prefix}/pct_env_step"] = float(100.0 * self.env_s / denom)
        out[f"{prefix}/pct_data"] = float(100.0 * self.data_s / denom)
        out[f"{prefix}/pct_update"] = float(100.0 * self.update_s / denom)
        out[f"{prefix}/pct_uncertainty"] = float(100.0 * self.uncertainty_s / denom)
        out[f"{prefix}/pct_episode_end"] = float(100.0 * self.episode_end_s / denom)
        out[f"{prefix}/pct_misc"] = float(100.0 * self.misc_s / denom)
        return out

    def reset(self) -> None:
        self.steps = 0
        self.action_s = 0.0
        self.env_s = 0.0
        self.data_s = 0.0
        self.update_s = 0.0
        self.uncertainty_s = 0.0
        self.episode_end_s = 0.0
        self.misc_s = 0.0
        self.total_s = 0.0


def _maybe_render_policy_map(
    *,
    args,
    checkpoint_path: Path,
    step_value: int,
    run_paths: RunPaths,
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
            output_dir=run_paths.log_dir / "policy_maps",
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


def _summary_stats(values: list[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_count": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_count": float(arr.shape[0]),
    }


def _window_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _compute_uncertainty_signal(
    *,
    sac: SACTensors,
    obs: np.ndarray,
    action: np.ndarray,
    device: torch.device,
    obs_preprocess=None,
) -> Dict[str, float]:
    obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
    if obs_preprocess is not None:
        was_training = bool(getattr(obs_preprocess, "training", False))
        if hasattr(obs_preprocess, "eval"):
            obs_preprocess.eval()
        obs_t = obs_preprocess(obs_t)
        if was_training and hasattr(obs_preprocess, "train"):
            obs_preprocess.train()
    act_t = torch.as_tensor(action[None, :], device=device, dtype=torch.float32)
    qd = compute_q_disagreement(sac=sac, obs=obs_t, actions=act_t)
    return {
        "abs_diff": float(qd.abs_diff_mean),
        "q1": float(qd.q1_mean),
        "q2": float(qd.q2_mean),
        "q_min": float(qd.q_min_mean),
        "q_max": float(qd.q_max_mean),
    }


def _prepare_run_dirs(args, variant_tag: str) -> RunPaths:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not getattr(args, "exp_name", ""):
        env_tag = args.env_name.replace("-", "_")
        args.exp_name = f"{variant_tag}_{env_tag}_{stamp}"
    log_dir = Path("logs") / "safetygym" / args.exp_name
    model_dir = Path("models") / "safetygym" / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    args_payload = vars(args)
    save_args_json(log_dir / "args.json", args_payload)
    save_args_json(model_dir / "args.json", args_payload)
    return RunPaths(log_dir=log_dir, model_dir=model_dir)


def _maybe_init_wandb(args, *, variant: str, run_paths: RunPaths):
    if not bool(getattr(args, "use_wandb", False)):
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError("--use_wandb was set, but wandb could not be imported.") from exc

    project = str(getattr(args, "wandb_project", "thesis-safetygym"))
    mode = str(getattr(args, "wandb_mode", "offline"))
    name = str(getattr(args, "wandb_run_name", "")).strip() or str(getattr(args, "exp_name", "")).strip()
    entity = str(getattr(args, "wandb_entity", "")).strip()
    group = str(getattr(args, "wandb_group", "")).strip()

    init_kwargs = {
        "project": project,
        "mode": mode,
        "config": {**vars(args), "variant": variant},
        "dir": str(run_paths.log_dir),
    }
    if name:
        init_kwargs["name"] = name
    if entity:
        init_kwargs["entity"] = entity
    if group:
        init_kwargs["group"] = group
    return wandb.init(**init_kwargs)


def _emit_metrics(
    *,
    payload: Dict[str, float],
    log_path: Path,
    wandb_run=None,
    step: Optional[int] = None,
) -> None:
    summary_line = _console_metrics_line(payload)
    if summary_line:
        print(summary_line, flush=True)
    line = json.dumps(payload, sort_keys=True)
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    if wandb_run is not None:
        if step is None:
            wandb_run.log(payload)
        else:
            wandb_run.log(payload, step=int(step))


def _fmt_metric(payload: Dict[str, float], key: str, *, digits: int = 3) -> str:
    value = payload.get(key)
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _console_metrics_line(payload: Dict[str, float]) -> str:
    if "eval/step" in payload:
        goals_solved = int(round(float(payload.get("eval/goals_solved", 0.0))))
        goals_attempted = int(round(float(payload.get("eval/goals_attempted", payload.get("eval/episodes", 0.0)))))
        return (
            "[Eval] "
            f"step={int(payload.get('eval/step', 0.0))} "
            f"reward={_fmt_metric(payload, 'eval/mean_reward')} "
            f"dense={_fmt_metric(payload, 'eval/mean_dense_reward')} "
            f"sparse={_fmt_metric(payload, 'eval/mean_sparse_reward')} "
            f"cost={_fmt_metric(payload, 'eval/episode_cost_sum_mean')} "
            f"success={_fmt_metric(payload, 'eval/outcome_success_mean')} "
            f"goals={goals_solved}/{goals_attempted} "
            f"timeout={_fmt_metric(payload, 'eval/outcome_timeout_mean')} "
            f"kill={_fmt_metric(payload, 'eval/outcome_kill_mean')} "
            f"final_dist={_fmt_metric(payload, 'eval/final_distance_to_goal_mean')} "
            f"episodes={int(payload.get('eval/episodes', 0.0))}"
        )
    if "train/pretrain_updates_requested" in payload:
        return (
            "[Pretrain] "
            f"updates={int(payload.get('train/pretrain_updates_run', 0.0))}/"
            f"{int(payload.get('train/pretrain_updates_requested', 0.0))} "
            f"critic_loss={_fmt_metric(payload, 'train/pretrain_critic_loss_mean')} "
            f"actor_loss={_fmt_metric(payload, 'train/pretrain_actor_loss_mean')} "
            f"target_q={_fmt_metric(payload, 'train/pretrain_target_q_mean')}"
        )
    if "train/prefill_episodes" in payload:
        return (
            "[Prefill] "
            f"episodes={int(payload.get('train/prefill_episodes', 0.0))} "
            f"steps={int(payload.get('train/prefill_steps', 0.0))} "
            f"demo_steps={int(payload.get('train/prefill_demo_steps', 0.0))} "
            f"demo_frac={_fmt_metric(payload, 'train/prefill_demo_fraction')} "
            f"duration_s={_fmt_metric(payload, 'train/prefill_duration_s', digits=2)}"
        )
    if "train/step" in payload:
        goals_solved = int(round(float(payload.get("train/goals_solved", 0.0))))
        goals_attempted = int(round(float(payload.get("train/goals_attempted", payload.get("train/episodes", 0.0)))))
        return (
            "[Train] "
            f"step={int(payload.get('train/step', 0.0))} "
            f"fps={_fmt_metric(payload, 'train/fps', digits=1)} "
            f"reward={_fmt_metric(payload, 'train/mean_reward')} "
            f"dense={_fmt_metric(payload, 'train/mean_dense_reward')} "
            f"sparse={_fmt_metric(payload, 'train/mean_sparse_reward')} "
            f"cost={_fmt_metric(payload, 'train/episode_cost_sum_mean')} "
            f"success={_fmt_metric(payload, 'train/outcome_success_mean')} "
            f"goals={goals_solved}/{goals_attempted} "
            f"final_dist={_fmt_metric(payload, 'train/final_distance_to_goal_mean')} "
            f"teacher_frac={_fmt_metric(payload, 'train/teacher_fraction_steps')} "
            f"replay={int(payload.get('train/buffer_main_size', 0.0))} "
            f"demo={int(payload.get('train/buffer_demo_size', 0.0))} "
            f"pref={int(payload.get('train/buffer_pref_size', 0.0))}"
        )
    return ""


def _make_env_with_wrappers(
    *,
    args,
    seed: int,
    with_intervention: bool,
    controller,
):
    env = make_safety_env(
        args.env_name,
        render_mode=resolve_env_render_mode(args.render_mode),
        max_episode_steps=args.max_episode_steps,
        surface_mode=getattr(args, "surface_mode", "default"),
        car_wheel_command_limit=float(getattr(args, "car_wheel_command_limit", 2.0)),
        car_force_scale=float(getattr(args, "car_force_scale", 2.0)),
        car_action_mode=str(getattr(args, "car_action_mode", "raw_wheels")),
        seed=seed,
    )
    env = RewardModeWrapper(
        env,
        reward_mode=args.reward_mode,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
        cost_penalty=float(getattr(args, "cost_penalty", 0.0)),
    )
    if with_intervention:
        if controller is None:
            raise ValueError("Intervention requested but controller is None")
        env = HumanInterventionWrapper(
            env,
            controller=controller,
            threshold=args.intervention_threshold,
            hold_seconds=args.intervention_hold_seconds,
        )
    return env


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
    payload: Dict[str, Any] = {
        "observations": torch.as_tensor(obs[None, :], device=device, dtype=torch.float32),
        "actions": torch.as_tensor(action[None, :], device=device, dtype=torch.float32),
        "next": {
            "observations": torch.as_tensor(next_obs[None, :], device=device, dtype=torch.float32),
            "rewards": torch.as_tensor([reward], device=device, dtype=torch.float32),
            "dones": torch.as_tensor([done], device=device, dtype=torch.bool),
            "truncations": torch.as_tensor([truncated], device=device, dtype=torch.bool),
            "effective_n_steps": torch.ones(1, device=device, dtype=torch.float32),
        },
    }
    if student_action is not None:
        payload["student_actions"] = torch.as_tensor(student_action[None, :], device=device, dtype=torch.float32)
    if teacher_intervened is not None:
        payload["teacher_intervened"] = torch.as_tensor([bool(teacher_intervened)], device=device, dtype=torch.bool)
    return TensorDict(payload, batch_size=(1,), device=device)


def _episode_metrics_from_info(
    *,
    info: Dict[str, Any],
    ep_return: float,
    ep_cost: float,
    ep_reward_raw_env: float,
    ep_reward_dense: float,
    ep_reward_sparse: float,
    ep_reward_step_penalty: float,
    ep_reward_cost_penalty: float,
    ep_len: int,
    max_episode_steps: int,
    terminated: bool,
    truncated: bool,
    final_distance_to_goal: float,
    goal_met_any: bool,
    goal_met_count: int,
    first_goal_hit_step: int | None,
    first_goal_reward_sum: float,
    first_goal_dense_reward_sum: float,
) -> Dict[str, float]:
    goal_met = bool(goal_met_any)
    outcome = classify_outcome(goal_met=goal_met, episode_steps=ep_len, max_episode_steps=max_episode_steps)
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
        "intervention_steps": float(info.get("teacher_intervention_steps", 0.0)),
        "intervention_fraction": float(info.get("teacher_fraction_steps", 0.0)),
        "intervention_num_bursts": float(info.get("teacher_num_bursts", 0.0)),
        "intervention_avg_burst_len": float(info.get("teacher_avg_burst_len", 0.0)),
        "goal_met": 1.0 if goal_met else 0.0,
        "goal_met_count": float(max(0, int(goal_met_count))),
        "first_goal_success": 1.0 if first_goal_hit_step is not None else 0.0,
        "first_goal_hit_step": float(first_hit),
        "first_goal_hit_step_success_only": float(first_hit if first_goal_hit_step is not None else 0.0),
        "first_goal_within_100": 1.0 if first_goal_hit_step is not None and first_hit <= 100 else 0.0,
        "first_goal_within_200": 1.0 if first_goal_hit_step is not None and first_hit <= 200 else 0.0,
        "first_goal_reward_sum": float(first_goal_reward_sum),
        "first_goal_dense_reward_sum": float(first_goal_dense_reward_sum),
        "final_distance_to_goal": float(final_distance_to_goal),
        "outcome_success": 1.0 if outcome == "success" else 0.0,
        "outcome_timeout": 1.0 if outcome == "timeout" else 0.0,
        "outcome_kill": 1.0 if outcome == "kill" else 0.0,
        "outcome_other_failure": 1.0 if outcome not in {"success", "timeout", "kill"} else 0.0,
        "terminated": 1.0 if terminated else 0.0,
        "truncated": 1.0 if truncated else 0.0,
    }


def _select_prefill_action(
    *,
    policy: str,
    obs: np.ndarray,
    env,
    sac: SACTensors,
    device: torch.device,
    obs_preprocess=None,
    scale_actor_to_env_bounds: bool = False,
) -> np.ndarray:
    mode = str(policy).lower()
    if mode == "random":
        return env.action_space.sample().astype(np.float32)
    if mode == "zero":
        return np.zeros_like(np.asarray(env.action_space.low, dtype=np.float32).reshape(-1), dtype=np.float32)
    if mode == "student":
        with torch.no_grad():
            obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
            if obs_preprocess is not None:
                obs_t = obs_preprocess(obs_t)
            act_t, _, _ = sac.actor(obs_t)
            action = act_t[0].detach().cpu().numpy().astype(np.float32)
        if scale_actor_to_env_bounds:
            action = scale_action_np(action, env.action_space)
        return action
    raise ValueError(f"Unsupported --prefill_policy: {policy}")


def _run_demo_prefill(
    *,
    args,
    env,
    sac: SACTensors,
    device: torch.device,
    main_rb: SimpleReplayBuffer,
    demo_rb: Optional[SimpleReplayBuffer],
    novice_rb: Optional[SimpleReplayBuffer],
    human_rb: Optional[SimpleReplayBuffer],
    variant: str,
    obs_preprocess=None,
    viewer: PygameRGBArrayViewer | None = None,
) -> Dict[str, float]:
    if int(getattr(args, "prefill_demo_episodes", 0)) <= 0:
        return {}
    if not bool(args.use_intervention):
        raise ValueError("--prefill_demo_episodes requires --use_intervention so human demos can be collected.")

    start = time.time()
    total_steps = 0
    demo_steps = 0
    episode_cap = int(max(0, getattr(args, "prefill_max_steps_per_episode", 0)))
    target_episodes = int(max(0, getattr(args, "prefill_demo_episodes", 0)))
    prefill_policy = str(getattr(args, "prefill_policy", "student")).lower()

    for ep_idx in range(target_episodes):
        obs, _ = env.reset(seed=args.seed + 10_000 + ep_idx)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if viewer is not None:
            viewer.draw_env(env)
        ep_steps = 0
        while True:
            student_action = _select_prefill_action(
                policy=prefill_policy,
                obs=obs,
                env=env,
                sac=sac,
                device=device,
                obs_preprocess=obs_preprocess,
                scale_actor_to_env_bounds=bool(getattr(args, "scale_actor_to_env_bounds", False)),
            )
            student_action = clip_action_to_space(student_action, env.action_space)
            next_obs, reward, _cost, terminated, truncated, info = env.step(student_action)
            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            if viewer is not None:
                viewer.draw_env(env)

            ep_steps += 1
            total_steps += 1
            teacher_intervened = bool(info.get("teacher_intervened", False))
            if teacher_intervened:
                demo_steps += 1

            force_end = bool(episode_cap > 0 and ep_steps >= episode_cap and not (terminated or truncated))
            done_for_rb = bool(terminated or truncated or force_end)
            truncated_for_rb = bool(truncated or force_end)
            applied_action = np.asarray(info.get("teacher_action", student_action), dtype=np.float32)
            applied_action = clip_action_to_space(applied_action, env.action_space)

            transition = _build_transition(
                obs=obs,
                action=applied_action,
                next_obs=next_obs,
                reward=float(reward),
                done=done_for_rb,
                truncated=truncated_for_rb,
                device=device,
            )

            main_rb.extend(transition)
            if bool(getattr(args, "obs_normalization", True)) and hasattr(obs_preprocess, "update"):
                obs_preprocess.update(torch.as_tensor(obs[None, :], device=device, dtype=torch.float32))
                obs_preprocess.update(torch.as_tensor(next_obs[None, :], device=device, dtype=torch.float32))
            if teacher_intervened and demo_rb is not None:
                demo_rb.extend(transition)
            if variant == "pvp":
                if teacher_intervened and human_rb is not None:
                    human_rb.extend(transition)
                elif novice_rb is not None:
                    novice_rb.extend(transition)

            obs = next_obs
            if done_for_rb:
                break

    return {
        "train/prefill_episodes": float(target_episodes),
        "train/prefill_steps": float(total_steps),
        "train/prefill_demo_steps": float(demo_steps),
        "train/prefill_demo_fraction": float(demo_steps / max(1, total_steps)),
        "train/prefill_duration_s": float(time.time() - start),
    }


def _run_demo_pretrain(
    *,
    args,
    sac: SACTensors,
    demo_rb: Optional[SimpleReplayBuffer],
    obs_preprocess=None,
) -> tuple[Optional[SACUpdateMetrics], Dict[str, float]]:
    updates = int(max(0, getattr(args, "demo_pretrain_updates", 0)))
    if updates <= 0 or demo_rb is None:
        return None, {}
    batch_size = int(getattr(args, "demo_pretrain_batch_size", 0))
    if batch_size <= 0:
        batch_size = int(args.batch_size)
    if demo_rb.size < batch_size:
        return None, {
            "train/pretrain_updates_requested": float(updates),
            "train/pretrain_updates_run": 0.0,
            "train/pretrain_skipped_small_demo_buffer": 1.0,
        }

    start = time.time()
    last_update: Optional[SACUpdateMetrics] = None
    critic_losses: list[float] = []
    actor_losses: list[float] = []
    alpha_losses: list[float] = []
    target_q_means: list[float] = []
    for _ in range(updates):
        batch = demo_rb.sample(batch_size)
        last_update = sac_update_step(
            sac=sac,
            batch=batch,
            gamma=float(args.gamma),
            tau=float(args.tau),
            max_grad_norm=float(args.max_grad_norm),
            obs_preprocess=obs_preprocess,
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
            pref_action_delta_min=float(getattr(args, "pref_action_delta_min", 0.0)),
            alpha_min=float(getattr(args, "alpha_min", 0.0)),
            alpha_max=float(getattr(args, "alpha_max", 1.0)),
        )
        critic_losses.append(float(last_update.critic_loss))
        actor_losses.append(float(last_update.actor_loss))
        alpha_losses.append(float(last_update.alpha_loss))
        target_q_means.append(float(last_update.target_q_mean))

    logs = {
        "train/pretrain_updates_requested": float(updates),
        "train/pretrain_updates_run": float(updates),
        "train/pretrain_duration_s": float(time.time() - start),
        "train/pretrain_critic_loss_mean": float(np.mean(np.asarray(critic_losses, dtype=np.float64))),
        "train/pretrain_actor_loss_mean": float(np.mean(np.asarray(actor_losses, dtype=np.float64))),
        "train/pretrain_alpha_loss_mean": float(np.mean(np.asarray(alpha_losses, dtype=np.float64))),
        "train/pretrain_target_q_mean": float(np.mean(np.asarray(target_q_means, dtype=np.float64))),
    }
    return last_update, logs


def _maybe_load_sac_checkpoint(
    *,
    args,
    sac: SACTensors,
    device: torch.device,
    obs_preprocess=None,
) -> Dict[str, float]:
    checkpoint_path = str(getattr(args, "init_checkpoint_path", "") or "").strip()
    if not checkpoint_path:
        return {}

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    loaded_actor = 0.0
    loaded_critic = 0.0
    loaded_target = 0.0
    loaded_alpha = 0.0
    loaded_optim = 0.0
    loaded_obs_normalizer = 0.0

    if bool(getattr(args, "load_actor_from_checkpoint", True)) and "actor_state_dict" in checkpoint:
        sac.actor.load_state_dict(checkpoint["actor_state_dict"])
        loaded_actor = 1.0
    if bool(getattr(args, "load_critic_from_checkpoint", True)) and "critic_state_dict" in checkpoint:
        sac.critic.load_state_dict(checkpoint["critic_state_dict"])
        loaded_critic = 1.0
    if bool(getattr(args, "load_critic_target_from_checkpoint", True)) and "critic_target_state_dict" in checkpoint:
        sac.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        loaded_target = 1.0
    elif loaded_critic > 0.0:
        sac.critic_target.load_state_dict(sac.critic.state_dict())
        loaded_target = 1.0
    if bool(getattr(args, "load_alpha_from_checkpoint", True)) and "log_alpha" in checkpoint:
        log_alpha = torch.as_tensor(checkpoint["log_alpha"], device=device, dtype=torch.float32).reshape_as(sac.log_alpha)
        sac.log_alpha.data.copy_(log_alpha)
        loaded_alpha = 1.0
    if (
        bool(getattr(args, "obs_normalization", True))
        and obs_preprocess is not None
        and hasattr(obs_preprocess, "load_state_dict")
        and "obs_normalizer_state_dict" in checkpoint
    ):
        obs_preprocess.load_state_dict(checkpoint["obs_normalizer_state_dict"], strict=False)
        loaded_obs_normalizer = 1.0

    if bool(getattr(args, "load_optimizer_state_from_checkpoint", False)):
        if loaded_actor > 0.0 and "actor_optimizer_state_dict" in checkpoint:
            sac.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            loaded_optim = 1.0
        if loaded_critic > 0.0 and "critic_optimizer_state_dict" in checkpoint:
            sac.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            loaded_optim = 1.0
        if loaded_alpha > 0.0 and "alpha_optimizer_state_dict" in checkpoint:
            sac.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
            loaded_optim = 1.0

    print(f"[Checkpoint] initialized SafetyGym training from {checkpoint_path}", flush=True)
    return {
        "train/init_checkpoint_loaded": 1.0,
        "train/init_checkpoint_loaded_actor": loaded_actor,
        "train/init_checkpoint_loaded_critic": loaded_critic,
        "train/init_checkpoint_loaded_critic_target": loaded_target,
        "train/init_checkpoint_loaded_alpha": loaded_alpha,
        "train/init_checkpoint_loaded_obs_normalizer": loaded_obs_normalizer,
        "train/init_checkpoint_loaded_optimizer_state": loaded_optim,
    }


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


def _run_eval(
    actor,
    args,
    device: torch.device,
    eval_seed: int,
    *,
    obs_preprocess=None,
    run_paths: RunPaths | None = None,
    step_value: int | None = None,
) -> Dict[str, float]:
    env = _make_env_with_wrappers(args=args, seed=eval_seed, with_intervention=False, controller=None)
    scale_actor_to_env_bounds = bool(getattr(args, "scale_actor_to_env_bounds", False))
    max_steps = extract_step_limit(env)
    win = EpisodeWindow(size=max(10, args.num_eval_episodes))
    task = env.unwrapped.task
    overlay_specs = _extract_overlay_specs(task)
    bounds = _extract_bounds(task, x_range=None, y_range=None)
    save_episode_plots = bool(getattr(args, "eval_save_episode_plots", False))
    episode_plot_max = int(max(0, getattr(args, "eval_episode_plot_max_episodes", 9)))
    saved_episode_plot_paths: list[Path] = []
    plot_dir: Path | None = None
    if save_episode_plots and run_paths is not None and step_value is not None:
        plot_dir = run_paths.log_dir / "eval_episode_plots" / f"step_{int(step_value)}"

    episodes = 0
    obs, _ = env.reset(seed=eval_seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    ep_ret = 0.0
    ep_cost = 0.0
    ep_reward_raw_env = 0.0
    ep_reward_dense = 0.0
    ep_reward_sparse = 0.0
    ep_reward_step_penalty = 0.0
    ep_reward_cost_penalty = 0.0
    ep_len = 0
    ep_goal_met_any = False
    ep_goal_met_count = 0
    ep_first_goal_hit_step: int | None = None
    ep_first_goal_reward_sum = 0.0
    ep_first_goal_dense_reward_sum = 0.0
    ep_path = [np.asarray(task.agent.pos[:2], dtype=np.float64).copy()]
    ep_goal_positions = [np.asarray(task.goal.pos[:2], dtype=np.float64).copy()]
    ep_goal_hit_points: list[np.ndarray] = []

    obs_preprocess_was_training = bool(getattr(obs_preprocess, "training", False)) if obs_preprocess is not None else False
    if obs_preprocess is not None and hasattr(obs_preprocess, "eval"):
        obs_preprocess.eval()

    while episodes < args.num_eval_episodes:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
            if obs_preprocess is not None:
                obs_t = obs_preprocess(obs_t)
            _, _, mean = actor(obs_t)
            action = mean[0].detach().cpu().numpy().astype(np.float32)
        if scale_actor_to_env_bounds:
            action = scale_action_np(action, env.action_space)
        action = clip_action_to_space(action, env.action_space)
        next_obs, reward, cost, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        ep_ret += float(reward)
        ep_cost += float(cost)
        ep_reward_raw_env += float(info.get("reward_raw_env", 0.0))
        ep_reward_dense += float(info.get("reward_dense_component", 0.0))
        ep_reward_sparse += float(info.get("reward_sparse_component", 0.0))
        ep_reward_step_penalty += float(info.get("reward_step_penalty_component", 0.0))
        ep_reward_cost_penalty += float(info.get("reward_cost_penalty_component", 0.0))
        ep_len += 1
        ep_path.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
        if bool(info.get("goal_met", False)):
            ep_goal_met_any = True
            ep_goal_met_count += 1
            if ep_first_goal_hit_step is None:
                ep_first_goal_hit_step = int(ep_len)
                ep_first_goal_reward_sum = float(ep_ret)
                ep_first_goal_dense_reward_sum = float(ep_reward_dense)
            ep_goal_hit_points.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
        current_goal_xy = np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
        if np.linalg.norm(current_goal_xy - np.asarray(ep_goal_positions[-1], dtype=np.float64)) > 1e-6:
            ep_goal_positions.append(current_goal_xy)

        if terminated or truncated:
            final_dist = extract_goal_distance(env)
            epm = _episode_metrics_from_info(
                info=dict(info),
                ep_return=ep_ret,
                ep_cost=ep_cost,
                ep_reward_raw_env=ep_reward_raw_env,
                ep_reward_dense=ep_reward_dense,
                ep_reward_sparse=ep_reward_sparse,
                ep_reward_step_penalty=ep_reward_step_penalty,
                ep_reward_cost_penalty=ep_reward_cost_penalty,
                ep_len=ep_len,
                max_episode_steps=max_steps,
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_distance_to_goal=final_dist,
                goal_met_any=ep_goal_met_any,
                goal_met_count=ep_goal_met_count,
                first_goal_hit_step=ep_first_goal_hit_step,
                first_goal_reward_sum=(
                    ep_first_goal_reward_sum if ep_first_goal_hit_step is not None else float(ep_ret)
                ),
                first_goal_dense_reward_sum=(
                    ep_first_goal_dense_reward_sum if ep_first_goal_hit_step is not None else float(ep_reward_dense)
                ),
            )
            win.add(epm)
            if save_episode_plots and plot_dir is not None and len(saved_episode_plot_paths) < episode_plot_max:
                plot_path = plot_eval_episode_trajectory(
                    output_path=plot_dir / f"episode_{episodes + 1:03d}.png",
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
                    episode_idx=episodes + 1,
                    total_episodes=int(args.num_eval_episodes),
                    episode_reward=float(ep_ret),
                    goals_reached=int(ep_goal_met_count),
                    final_distance=float(final_dist),
                )
                saved_episode_plot_paths.append(Path(plot_path))
            episodes += 1
            obs, _ = env.reset(seed=eval_seed + episodes)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            ep_ret = 0.0
            ep_cost = 0.0
            ep_reward_raw_env = 0.0
            ep_reward_dense = 0.0
            ep_reward_sparse = 0.0
            ep_reward_step_penalty = 0.0
            ep_reward_cost_penalty = 0.0
            ep_len = 0
            ep_goal_met_any = False
            ep_goal_met_count = 0
            ep_first_goal_hit_step = None
            ep_first_goal_reward_sum = 0.0
            ep_first_goal_dense_reward_sum = 0.0
            ep_path = [np.asarray(task.agent.pos[:2], dtype=np.float64).copy()]
            ep_goal_positions = [np.asarray(task.goal.pos[:2], dtype=np.float64).copy()]
            ep_goal_hit_points = []
        else:
            obs = next_obs

    if save_episode_plots and plot_dir is not None and saved_episode_plot_paths:
        sheet_path = plot_episode_contact_sheet(
            image_paths=saved_episode_plot_paths,
            output_path=plot_dir / "episode_contact_sheet.png",
            title=f"SafetyGym train eval step {int(step_value or 0)}",
            max_cols=3,
        )
        if sheet_path is not None:
            print(f"[EvalPlots] saved {sheet_path}", flush=True)
    if obs_preprocess is not None and obs_preprocess_was_training and hasattr(obs_preprocess, "train"):
        obs_preprocess.train()
    env.close()
    return augment_rollout_summary(win.summary("eval"), "eval")


def run_training(args, *, variant: str) -> None:
    if int(args.num_envs) != 1:
        raise ValueError("SafetyGym human-intervention pipeline currently supports --num_envs 1 only.")

    torch_num_threads = int(getattr(args, "torch_num_threads", 0))
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    torch_num_interop = int(getattr(args, "torch_num_interop_threads", 0))
    if torch_num_interop > 0:
        try:
            torch.set_num_interop_threads(torch_num_interop)
        except RuntimeError:
            # torch may disallow changing interop threads after initialization.
            pass

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_paths = _prepare_run_dirs(args, variant_tag=variant)
    log_path = run_paths.log_dir / "training.log"
    wandb_run = _maybe_init_wandb(args, variant=variant, run_paths=run_paths)
    viewer = build_external_viewer(
        render_mode=getattr(args, "render_mode", "human"),
        title=f"SafetyGym {args.env_name}",
        draw_hz=float(getattr(args, "viewer_fps", 20.0)),
        scale=float(getattr(args, "viewer_scale", 1.0)),
    ) if wants_external_viewer(getattr(args, "render_mode", "human")) else None

    controller = None
    with_intervention = bool(args.use_intervention)
    if with_intervention:
        controller = build_human_controller(
            input_device=str(getattr(args, "human_input_device", "keyboard")),
            action_dim=2,
            env_name=args.env_name,
            action_scale=float(args.human_action_scale),
            wheel_command_limit=float(getattr(args, "car_wheel_command_limit", 2.0)),
            overlay_fps_limit=int(getattr(args, "controller_fps_limit", 0)),
            overlay_draw_hz=float(getattr(args, "controller_overlay_hz", 20.0)),
            gamepad_mode=str(getattr(args, "gamepad_mode", "local")),
            gamepad_host=str(getattr(args, "gamepad_host", "")),
            gamepad_port=int(getattr(args, "gamepad_port", 0) or DEFAULT_SAFETY_GAMEPAD_PORT),
            gamepad_cache_path=getattr(args, "gamepad_cache_path", DEFAULT_SAFETY_GAMEPAD_CACHE_PATH),
            gamepad_reconnect_seconds=float(getattr(args, "gamepad_reconnect_seconds", 2.0)),
            gamepad_config_path=getattr(args, "gamepad_config_path", DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH),
            gamepad_use_saved_config=bool(getattr(args, "gamepad_use_saved_config", True)),
            gamepad_device_index=int(getattr(args, "gamepad_device_index", 0)),
            prefer_separate_keyboard_window=wants_external_viewer(getattr(args, "render_mode", "human")),
            control_scheme_override=resolve_control_scheme(
                str(args.env_name),
                car_action_mode=str(getattr(args, "car_action_mode", "raw_wheels")),
            ),
        )

    env = _make_env_with_wrappers(
        args=args,
        seed=args.seed,
        with_intervention=with_intervention,
        controller=controller,
    )

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if viewer is not None:
        viewer.draw_env(env)
    obs_dim = int(obs.shape[0])
    act_dim = int(np.prod(env.action_space.shape))

    if with_intervention and controller is not None and controller.action_dim != act_dim:
        controller.close()
        controller = build_human_controller(
            input_device=str(getattr(args, "human_input_device", "keyboard")),
            action_dim=act_dim,
            env_name=args.env_name,
            action_scale=float(args.human_action_scale),
            wheel_command_limit=float(getattr(args, "car_wheel_command_limit", 2.0)),
            overlay_fps_limit=int(getattr(args, "controller_fps_limit", 0)),
            overlay_draw_hz=float(getattr(args, "controller_overlay_hz", 20.0)),
            gamepad_mode=str(getattr(args, "gamepad_mode", "local")),
            gamepad_host=str(getattr(args, "gamepad_host", "")),
            gamepad_port=int(getattr(args, "gamepad_port", 0) or DEFAULT_SAFETY_GAMEPAD_PORT),
            gamepad_cache_path=getattr(args, "gamepad_cache_path", DEFAULT_SAFETY_GAMEPAD_CACHE_PATH),
            gamepad_reconnect_seconds=float(getattr(args, "gamepad_reconnect_seconds", 2.0)),
            gamepad_config_path=getattr(args, "gamepad_config_path", DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH),
            gamepad_use_saved_config=bool(getattr(args, "gamepad_use_saved_config", True)),
            gamepad_device_index=int(getattr(args, "gamepad_device_index", 0)),
            prefer_separate_keyboard_window=wants_external_viewer(getattr(args, "render_mode", "human")),
            control_scheme_override=resolve_control_scheme(
                str(args.env_name),
                car_action_mode=str(getattr(args, "car_action_mode", "raw_wheels")),
            ),
        )
        env.close()
        env = _make_env_with_wrappers(args=args, seed=args.seed, with_intervention=True, controller=controller)
        obs, _ = env.reset(seed=args.seed)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if viewer is not None:
            viewer.draw_env(env)

    max_steps = extract_step_limit(env)
    if bool(getattr(args, "obs_normalization", True)):
        obs_preprocess = EmpiricalNormalization(shape=obs_dim, device=device)
    else:
        obs_preprocess = torch.nn.Identity()

    sac: SACTensors = build_sac(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_actor=args.actor_hidden_dim,
        hidden_critic=args.critic_hidden_dim,
        num_critics=int(getattr(args, "num_critics", 2)),
        use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
        layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
        init_scale=args.init_scale,
        lr_actor=args.actor_learning_rate,
        lr_critic=args.critic_learning_rate,
        weight_decay=args.weight_decay,
        num_envs=1,
        device=device,
        alpha_init=float(getattr(args, "alpha_init", 1e-3)),
    )
    sac.pref_lambda = float(getattr(args, "pref_lambda_init", 0.0))
    sac.pref_violation_ema = 0.0
    init_logs = _maybe_load_sac_checkpoint(args=args, sac=sac, device=device, obs_preprocess=obs_preprocess)
    if init_logs:
        _emit_metrics(payload=init_logs, log_path=log_path, wandb_run=wandb_run, step=0)

    main_rb = SimpleReplayBuffer(
        n_env=1,
        buffer_size=args.buffer_size,
        n_obs=obs_dim,
        n_act=act_dim,
        n_critic_obs=obs_dim,
        asymmetric_obs=False,
        playground_mode=False,
        n_steps=1,
        gamma=args.gamma,
        device=device,
        pixel_shape=None,
    )
    action_low_t = torch.as_tensor(env.action_space.low.reshape(1, -1), device=device, dtype=torch.float32)
    action_high_t = torch.as_tensor(env.action_space.high.reshape(1, -1), device=device, dtype=torch.float32)

    novice_rb = None
    human_rb = None
    demo_rb: Optional[SimpleReplayBuffer] = None
    if variant == "pvp":
        novice_rb = SimpleReplayBuffer(
            n_env=1,
            buffer_size=args.buffer_size,
            n_obs=obs_dim,
            n_act=act_dim,
            n_critic_obs=obs_dim,
            asymmetric_obs=False,
            playground_mode=False,
            n_steps=1,
            gamma=args.gamma,
            device=device,
            pixel_shape=None,
        )
        human_rb = SimpleReplayBuffer(
            n_env=1,
            buffer_size=args.buffer_size,
            n_obs=obs_dim,
            n_act=act_dim,
            n_critic_obs=obs_dim,
            asymmetric_obs=False,
            playground_mode=False,
            n_steps=1,
            gamma=args.gamma,
            device=device,
            pixel_shape=None,
        )
    dataset_target = str(getattr(args, "demo_dataset_target", "variant")).strip().lower()
    dataset_requested = bool(str(getattr(args, "demo_dataset_path", "") or "").strip()) or bool(
        getattr(args, "demo_dataset_auto_load", False)
    )
    if (
        variant == "hilserl"
        or bool(getattr(args, "store_intervened_in_demo_buffer", False))
        or int(getattr(args, "prefill_demo_episodes", 0)) > 0
        or int(getattr(args, "demo_pretrain_updates", 0)) > 0
        or (dataset_requested and dataset_target in {"demo", "variant"})
    ):
        demo_rb = SimpleReplayBuffer(
            n_env=1,
            buffer_size=args.buffer_size,
            n_obs=obs_dim,
            n_act=act_dim,
            n_critic_obs=obs_dim,
            asymmetric_obs=False,
            playground_mode=False,
            n_steps=1,
            gamma=args.gamma,
            device=device,
            pixel_shape=None,
        )

    pref_pairs: list[Dict[str, np.ndarray]] = []
    pref_capacity = int(max(0, getattr(args, "pref_capacity", 0)))
    pref_sampling_mode = str(getattr(args, "pref_sampling_mode", "linked")).strip().lower()

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

    train_win = EpisodeWindow(size=200)

    prefill_logs = _run_demo_prefill(
        args=args,
        env=env,
        sac=sac,
        device=device,
        main_rb=main_rb,
        demo_rb=demo_rb,
        novice_rb=novice_rb,
        human_rb=human_rb,
        variant=variant,
        obs_preprocess=obs_preprocess,
        viewer=viewer,
    )
    if prefill_logs:
        _emit_metrics(payload=prefill_logs, log_path=log_path, wandb_run=wandb_run, step=0)

    last_update: Optional[SACUpdateMetrics] = None
    update_window_count = 0
    update_window_actor_updates = 0.0
    update_window_alpha_updates = 0.0
    update_window_policy_entropy = 0.0
    update_window_log_pi_mean = 0.0
    update_window_action_l2 = 0.0
    update_window_q_min_pi_mean = 0.0
    update_window_q_min_data_mean = 0.0
    update_window_q_disagreement_data_mean = 0.0
    update_window_q_disagreement_pi_mean = 0.0
    update_window_replay_reward_mean = 0.0
    update_window_replay_reward_abs_mean = 0.0
    total_update_steps = 0
    pretrain_update, pretrain_logs = _run_demo_pretrain(
        args=args,
        sac=sac,
        demo_rb=demo_rb,
        obs_preprocess=obs_preprocess,
    )
    if pretrain_update is not None:
        last_update = pretrain_update
    if pretrain_logs:
        _emit_metrics(payload=pretrain_logs, log_path=log_path, wandb_run=wandb_run, step=0)

    pretrain_updates_run = int(pretrain_logs.get("train/pretrain_updates_run", 0.0)) if pretrain_logs else 0
    if bool(getattr(args, "critic_reset_after_pretrain", False)) and pretrain_updates_run > 0:
        reset_critic(
            sac=sac,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_critic=int(args.critic_hidden_dim),
            num_critics=int(getattr(args, "num_critics", 2)),
            use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
            layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
            lr_critic=float(args.critic_learning_rate),
            weight_decay=float(args.weight_decay),
            device=device,
        )
        _emit_metrics(
            payload={"train/critic_reset_after_pretrain": 1.0},
            log_path=log_path,
            wandb_run=wandb_run,
            step=0,
        )
    elif bool(getattr(args, "critic_reset_after_pretrain", False)):
        _emit_metrics(
            payload={"train/critic_reset_after_pretrain": 0.0},
            log_path=log_path,
            wandb_run=wandb_run,
            step=0,
        )

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if viewer is not None:
        viewer.draw_env(env)
    prefill_steps_completed = int(prefill_logs.get("train/prefill_steps", 0.0)) if prefill_logs else 0
    effective_learning_starts = max(0, int(args.learning_starts) - prefill_steps_completed)

    ep_ret = 0.0
    ep_cost = 0.0
    ep_reward_raw_env = 0.0
    ep_reward_dense = 0.0
    ep_reward_sparse = 0.0
    ep_reward_step_penalty = 0.0
    ep_reward_cost_penalty = 0.0
    ep_len = 0
    ep_goal_met_any = False
    ep_goal_met_count = 0
    ep_first_goal_hit_step: int | None = None
    ep_first_goal_reward_sum = 0.0
    ep_first_goal_dense_reward_sum = 0.0

    next_log = int(args.log_interval)
    next_eval = int(args.eval_interval) if args.eval_interval > 0 else -1
    next_save = int(args.save_interval) if args.save_interval > 0 else -1

    start_time = time.time()
    live_window_steps = 0
    live_window_intervention_steps = 0
    total_interventions = 0
    live_window_controller_ms = 0.0
    live_window_oversight_required_steps = 0
    live_window_uncertainty_all: list[float] = []
    live_window_uncertainty_intervention: list[float] = []
    live_window_reward_raw_env: list[float] = []
    live_window_reward_shaped: list[float] = []
    live_window_reward_dense: list[float] = []
    live_window_reward_sparse: list[float] = []
    live_window_reward_step_penalty: list[float] = []
    live_window_reward_cost_penalty: list[float] = []
    prev_teacher_intervened = False
    oversight_mode = str(getattr(args, "uncertainty_oversight_mode", "signal_only")).lower()
    oversight_threshold = float(getattr(args, "uncertainty_oversight_threshold", 0.0))
    oversight_enabled = oversight_mode != "off" and oversight_threshold > 0.0
    oversight_ema_alpha = float(np.clip(float(getattr(args, "uncertainty_oversight_ema_alpha", 0.05)), 1e-6, 1.0))
    oversight_ema: Optional[float] = None
    uncertainty_enabled = bool(getattr(args, "uncertainty_log_every_step", True)) or oversight_enabled
    timing_enabled = bool(getattr(args, "profile_timing", False))
    timing_window = TimingWindow()
    env_fps_limit = float(max(0.0, getattr(args, "env_fps_limit", 0.0)))
    env_step_period = (1.0 / env_fps_limit) if env_fps_limit > 0.0 else 0.0

    for step in range(1, int(args.total_timesteps) + 1):
        step_t0 = time.perf_counter()
        t_action = 0.0
        t_env = 0.0
        t_data = 0.0
        t_update = 0.0
        t_uncertainty = 0.0
        t_episode_end = 0.0

        t0 = time.perf_counter()
        if step <= int(effective_learning_starts):
            student_action = env.action_space.sample().astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
                obs_t = obs_preprocess(obs_t)
                action_t, _, _ = sac.actor(obs_t)
                student_action = action_t[0].detach().cpu().numpy().astype(np.float32)
            if bool(getattr(args, "scale_actor_to_env_bounds", False)):
                student_action = scale_action_np(student_action, env.action_space)
        student_action = clip_action_to_space(student_action, env.action_space)
        t_action += time.perf_counter() - t0

        t0 = time.perf_counter()
        next_obs, reward, cost, terminated, truncated, info = env.step(student_action)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        if viewer is not None:
            viewer.draw_env(env)
        t_env += time.perf_counter() - t0

        t0 = time.perf_counter()
        teacher_intervened = bool(info.get("teacher_intervened", False))
        live_window_steps += 1
        live_window_controller_ms += float(info.get("teacher_controller_ms", 0.0))
        if teacher_intervened:
            live_window_intervention_steps += 1
            total_interventions += 1
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
        if bool(getattr(args, "obs_normalization", True)) and hasattr(obs_preprocess, "update"):
            obs_preprocess.update(torch.as_tensor(obs[None, :], device=device, dtype=torch.float32))
            obs_preprocess.update(torch.as_tensor(next_obs[None, :], device=device, dtype=torch.float32))
        if variant == "pvp":
            if teacher_intervened and human_rb is not None:
                human_rb.extend(transition)
            elif novice_rb is not None:
                novice_rb.extend(transition)
        if teacher_intervened and demo_rb is not None and (
            variant == "hilserl" or bool(getattr(args, "store_intervened_in_demo_buffer", False))
        ):
            demo_rb.extend(transition)

        if teacher_intervened and pref_capacity > 0 and variant == "own" and pref_sampling_mode != "linked":
            teacher_action = np.asarray(info.get("teacher_action", applied_action), dtype=np.float32)
            student_logged = np.asarray(info.get("student_action", student_action), dtype=np.float32)
            pref_pairs.append(
                {
                    "obs": obs.copy(),
                    "teacher_actions": clip_action_to_space(teacher_action, env.action_space),
                    "student_actions": clip_action_to_space(student_logged, env.action_space),
                }
            )
            if len(pref_pairs) > pref_capacity:
                pref_pairs = pref_pairs[-pref_capacity:]

        if uncertainty_enabled:
            tu0 = time.perf_counter()
            unc = _compute_uncertainty_signal(
                sac=sac,
                obs=obs,
                action=applied_action,
                device=device,
                obs_preprocess=obs_preprocess,
            )
            unc_val = float(unc["abs_diff"])
            live_window_uncertainty_all.append(unc_val)
            if teacher_intervened:
                live_window_uncertainty_intervention.append(unc_val)

            if oversight_enabled:
                if oversight_ema is None:
                    oversight_ema = unc_val
                else:
                    oversight_ema = float(oversight_ema_alpha * unc_val + (1.0 - oversight_ema_alpha) * oversight_ema)
                oversight_required = bool(unc_val > oversight_threshold)
                if oversight_required:
                    live_window_oversight_required_steps += 1

            t_uncertainty += time.perf_counter() - tu0
        prev_teacher_intervened = bool(teacher_intervened)
        t_data += time.perf_counter() - t0

        obs = next_obs
        ep_ret += float(reward)
        ep_cost += float(cost)
        ep_reward_raw_env += float(info.get("reward_raw_env", 0.0))
        ep_reward_dense += float(info.get("reward_dense_component", 0.0))
        ep_reward_sparse += float(info.get("reward_sparse_component", 0.0))
        ep_reward_step_penalty += float(info.get("reward_step_penalty_component", 0.0))
        ep_reward_cost_penalty += float(info.get("reward_cost_penalty_component", 0.0))
        live_window_reward_raw_env.append(float(info.get("reward_raw_env", 0.0)))
        live_window_reward_shaped.append(float(reward))
        live_window_reward_dense.append(float(info.get("reward_dense_component", 0.0)))
        live_window_reward_sparse.append(float(info.get("reward_sparse_component", 0.0)))
        live_window_reward_step_penalty.append(float(info.get("reward_step_penalty_component", 0.0)))
        live_window_reward_cost_penalty.append(float(info.get("reward_cost_penalty_component", 0.0)))
        ep_len += 1
        if bool(info.get("goal_met", False)):
            ep_goal_met_any = True
            ep_goal_met_count += 1
            if ep_first_goal_hit_step is None:
                ep_first_goal_hit_step = int(ep_len)
                ep_first_goal_reward_sum = float(ep_ret)
                ep_first_goal_dense_reward_sum = float(ep_reward_dense)

        rb_ready = main_rb.size >= int(args.batch_size)
        if variant == "pvp" and novice_rb is not None:
            rb_ready = novice_rb.size >= max(1, int(args.batch_size // 2))

        update_every = max(1, int(getattr(args, "update_every", 1)))
        updates_per_cycle = max(1, int(getattr(args, "updates_per_cycle", 1)))
        if step > int(effective_learning_starts) and rb_ready and (step % update_every == 0):
            t0 = time.perf_counter()
            for _ in range(updates_per_cycle):
                total_update_steps += 1
                if variant == "pvp" and novice_rb is not None and human_rb is not None:
                    half = max(1, int(args.batch_size // 2))
                    if human_rb.size >= half:
                        batch_n = novice_rb.sample(max(1, int(args.batch_size - half)))
                        batch_h = human_rb.sample(half)
                        batch = TensorDict.cat([batch_n, batch_h], dim=0)
                    else:
                        batch = novice_rb.sample(int(args.batch_size))
                elif (
                    variant in {"hilserl", "own"}
                    and demo_rb is not None
                    and demo_rb.size > 0
                    and float(getattr(args, "demo_sample_ratio", 0.0)) > 0.0
                ):
                    demo_ratio = float(getattr(args, "demo_sample_ratio", 0.0))
                    demo_n = int(max(1, args.batch_size * demo_ratio))
                    base_n = max(1, int(args.batch_size - demo_n))
                    if main_rb.size >= base_n and demo_rb.size >= demo_n:
                        batch_main = main_rb.sample(base_n)
                        batch_demo = demo_rb.sample(demo_n)
                        batch = TensorDict.cat([batch_main, batch_demo], dim=0)
                    else:
                        batch = main_rb.sample(int(args.batch_size))
                else:
                    batch = main_rb.sample(int(args.batch_size))

                pref_batch = None
                if (
                    variant == "own"
                    and float(args.pref_sample_ratio) > 0.0
                    and pref_sampling_mode == "linked"
                ):
                    pref_batch = None
                elif variant == "own" and float(args.pref_sample_ratio) > 0.0 and pref_pairs:
                    pref_n = max(1, int(args.batch_size * float(args.pref_sample_ratio)))
                    pref_batch = _sample_pref_batch(pref_pairs, pref_n, device)

                last_update = sac_update_step(
                    sac=sac,
                    batch=batch,
                    gamma=float(args.gamma),
                    tau=float(args.tau),
                    max_grad_norm=float(args.max_grad_norm),
                    obs_preprocess=obs_preprocess,
                    pref_batch=pref_batch,
                    pref_sampling_mode=pref_sampling_mode if variant == "own" else "separate",
                    pref_rank_weight=float(getattr(args, "pref_rank_weight", 0.0)),
                    pref_rank_margin=float(getattr(args, "pref_rank_margin", 0.1)),
                    pref_loss_type=str(getattr(args, "pref_loss_type", "margin")),
                    pref_stopgrad_positive=bool(getattr(args, "pref_stopgrad_positive", False)),
                    pref_lambda_lr=float(getattr(args, "pref_lambda_lr", 1e-3)),
                    pref_lambda_max=float(getattr(args, "pref_lambda_max", 10.0)),
                    pref_lambda_ema=float(getattr(args, "pref_lambda_ema", 0.9)),
                    pref_violation_clip=float(getattr(args, "pref_violation_clip", 10.0)),
                    pref_violation_target=float(getattr(args, "pref_violation_target", 0.0)),
                    pref_lagrangian_violation_type=str(getattr(args, "pref_lagrangian_violation_type", "hinge")),
                    pref_action_delta_min=float(getattr(args, "pref_action_delta_min", 0.0)),
                    alpha_min=float(getattr(args, "alpha_min", 0.0)),
                    alpha_max=float(getattr(args, "alpha_max", 1.0)),
                    scale_actor_to_env_bounds=bool(getattr(args, "scale_actor_to_env_bounds", False)),
                    action_low=action_low_t,
                    action_high=action_high_t,
                    update_actor=(total_update_steps % max(1, int(getattr(args, "policy_frequency", 1))) == 0),
                    critic_loss_reduction=str(getattr(args, "critic_loss_reduction", "mean")),
                )
                update_window_count += 1
                update_window_actor_updates += float(getattr(last_update, "actor_updates", 0.0))
                update_window_alpha_updates += float(getattr(last_update, "alpha_updates", 0.0))
                update_window_policy_entropy += float(getattr(last_update, "policy_entropy", 0.0))
                update_window_log_pi_mean += float(getattr(last_update, "log_pi_mean", 0.0))
                update_window_action_l2 += float(getattr(last_update, "action_l2", 0.0))
                update_window_q_min_pi_mean += float(getattr(last_update, "q_min_pi_mean", 0.0))
                update_window_q_min_data_mean += float(getattr(last_update, "q_min_data_mean", 0.0))
                update_window_q_disagreement_data_mean += float(
                    getattr(last_update, "q_disagreement_data_mean", 0.0)
                )
                update_window_q_disagreement_pi_mean += float(
                    getattr(last_update, "q_disagreement_pi_mean", 0.0)
                )
                update_window_replay_reward_mean += float(getattr(last_update, "replay_reward_mean", 0.0))
                update_window_replay_reward_abs_mean += float(getattr(last_update, "replay_reward_abs_mean", 0.0))
            t_update += time.perf_counter() - t0

        if terminated or truncated:
            t0 = time.perf_counter()
            final_dist = extract_goal_distance(env)
            epm = _episode_metrics_from_info(
                info=dict(info),
                ep_return=ep_ret,
                ep_cost=ep_cost,
                ep_reward_raw_env=ep_reward_raw_env,
                ep_reward_dense=ep_reward_dense,
                ep_reward_sparse=ep_reward_sparse,
                ep_reward_step_penalty=ep_reward_step_penalty,
                ep_reward_cost_penalty=ep_reward_cost_penalty,
                ep_len=ep_len,
                max_episode_steps=max_steps,
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_distance_to_goal=final_dist,
                goal_met_any=ep_goal_met_any,
                goal_met_count=ep_goal_met_count,
                first_goal_hit_step=ep_first_goal_hit_step,
                first_goal_reward_sum=(
                    ep_first_goal_reward_sum if ep_first_goal_hit_step is not None else float(ep_ret)
                ),
                first_goal_dense_reward_sum=(
                    ep_first_goal_dense_reward_sum if ep_first_goal_hit_step is not None else float(ep_reward_dense)
                ),
            )
            train_win.add(epm)

            if bool(getattr(args, "reseed_on_episode_reset", True)):
                obs, _ = env.reset(seed=args.seed + step)
            else:
                obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            if viewer is not None:
                viewer.draw_env(env)
            ep_ret = 0.0
            ep_cost = 0.0
            ep_reward_raw_env = 0.0
            ep_reward_dense = 0.0
            ep_reward_sparse = 0.0
            ep_reward_step_penalty = 0.0
            ep_len = 0
            ep_goal_met_any = False
            ep_goal_met_count = 0
            ep_first_goal_hit_step = None
            ep_first_goal_reward_sum = 0.0
            ep_first_goal_dense_reward_sum = 0.0
            prev_teacher_intervened = False
            t_episode_end += time.perf_counter() - t0

        if env_step_period > 0.0:
            remaining = env_step_period - (time.perf_counter() - step_t0)
            if remaining > 0.0:
                time.sleep(remaining)

        step_total = time.perf_counter() - step_t0
        known = t_action + t_env + t_data + t_update + t_uncertainty + t_episode_end
        t_misc = max(0.0, step_total - known)
        if timing_enabled:
            timing_window.add(
                action_s=t_action,
                env_s=t_env,
                data_s=t_data,
                update_s=t_update,
                uncertainty_s=t_uncertainty,
                episode_end_s=t_episode_end,
                misc_s=t_misc,
                total_s=step_total,
            )

        if next_log > 0 and step >= next_log:
            next_log += int(args.log_interval)
            elapsed = max(1e-6, time.time() - start_time)
            fps = float(step / elapsed)
            logs: Dict[str, float] = {
                "train/step": float(step),
                "train/fps": fps,
                "train/replay_size": float(main_rb.size),
                "train/replay_capacity": float(getattr(main_rb, "capacity", getattr(args, "buffer_size", 0))),
                "train/effective_learning_starts": float(effective_learning_starts),
                "train/live_window_steps": float(live_window_steps),
                "train/live_intervention_steps": float(live_window_intervention_steps),
                "train/live_intervention_fraction": float(live_window_intervention_steps / max(1, live_window_steps)),
                "train/total_interventions": float(total_interventions),
                "train/live_controller_ms_per_step": float(live_window_controller_ms / max(1, live_window_steps)),
            }
            # Always emit buffer state for consistent dashboards across variants.
            logs["train/buffer_main_size"] = float(main_rb.size)
            logs["train/buffer_main_capacity"] = float(
                getattr(main_rb, "capacity", getattr(args, "buffer_size", 0))
            )
            logs["train/buffer_demo_size"] = float(demo_rb.size if demo_rb is not None else 0)
            logs["train/buffer_demo_capacity"] = float(
                getattr(demo_rb, "capacity", getattr(args, "buffer_size", 0)) if demo_rb is not None else 0
            )
            logs["train/buffer_novice_size"] = float(novice_rb.size if novice_rb is not None else 0)
            logs["train/buffer_novice_capacity"] = float(
                getattr(novice_rb, "capacity", getattr(args, "buffer_size", 0)) if novice_rb is not None else 0
            )
            logs["train/buffer_human_size"] = float(human_rb.size if human_rb is not None else 0)
            logs["train/buffer_human_capacity"] = float(
                getattr(human_rb, "capacity", getattr(args, "buffer_size", 0)) if human_rb is not None else 0
            )
            if variant == "own" and pref_sampling_mode == "linked":
                logs["train/buffer_pref_size"] = float(main_rb.linked_pref_pair_count())
                logs["train/buffer_pref_capacity"] = float(
                    getattr(main_rb, "capacity", getattr(args, "buffer_size", 0))
                )
            else:
                logs["train/buffer_pref_size"] = float(len(pref_pairs))
                logs["train/buffer_pref_capacity"] = float(pref_capacity)
            logs["train/pref_sampling_mode_linked"] = 1.0 if pref_sampling_mode == "linked" else 0.0
            if uncertainty_enabled:
                logs.update(_summary_stats(live_window_uncertainty_all, "train/uncertainty_all"))
                logs.update(_summary_stats(live_window_uncertainty_intervention, "train/uncertainty_intervention"))
            if oversight_enabled:
                logs["train/oversight_threshold"] = float(oversight_threshold)
                logs["train/oversight_required_steps"] = float(live_window_oversight_required_steps)
                logs["train/oversight_required_fraction"] = float(
                    live_window_oversight_required_steps / max(1, live_window_steps)
                )
                logs["train/oversight_ema"] = float(oversight_ema if oversight_ema is not None else 0.0)
            if timing_enabled:
                logs.update(timing_window.summary("train_timing"))
            logs["train/mean_step_reward"] = _window_mean(live_window_reward_shaped)
            logs["train/mean_step_env_reward"] = _window_mean(live_window_reward_raw_env)
            logs["train/mean_step_dense_reward"] = _window_mean(live_window_reward_dense)
            logs["train/mean_step_sparse_reward"] = _window_mean(live_window_reward_sparse)
            logs["train/mean_step_step_penalty_reward"] = _window_mean(live_window_reward_step_penalty)
            logs["train/mean_step_cost_penalty_reward"] = _window_mean(live_window_reward_cost_penalty)
            logs.update(augment_rollout_summary(train_win.summary("train"), "train"))
            if "train/mean_reward" not in logs and "train/episode_return_mean" in logs:
                logs["train/mean_reward"] = float(logs["train/episode_return_mean"])
            if variant == "pvp" and novice_rb is not None and human_rb is not None:
                logs["train/novice_replay_size"] = float(novice_rb.size)
                logs["train/human_replay_size"] = float(human_rb.size)
            logs["train/demo_replay_size"] = float(demo_rb.size if demo_rb is not None else 0)
            if last_update is not None:
                logs.update(
                    {
                        "train/critic_loss": last_update.critic_loss,
                        "train/critic_loss_pref": last_update.critic_loss_pref,
                        "train/critic_loss_pref_weighted": last_update.critic_loss_pref_weighted,
                        "train/actor_loss": last_update.actor_loss,
                        "train/alpha_loss": last_update.alpha_loss,
                        "train/alpha": last_update.alpha,
                        "train/target_q_mean": last_update.target_q_mean,
                        "train/pref_q_delta": last_update.pref_q_delta,
                        "train/pref_lambda": last_update.pref_lambda,
                        "train/pref_lambda_delta": last_update.pref_lambda_delta,
                        "train/pref_dual_violation": last_update.pref_dual_violation,
                        "train/pref_dual_signal": last_update.pref_dual_signal,
                        "train/pref_violation": last_update.pref_violation,
                        "train/pref_violation_ema": last_update.pref_violation_ema,
                        "train/pref_lagrangian_loss": last_update.pref_lagrangian_loss,
                    }
                )
                update_denom = float(max(1, update_window_count))
                actor_update_denom = float(max(1.0, update_window_actor_updates))
                alpha_update_denom = float(max(1.0, update_window_alpha_updates))
                logs["train/policy_entropy"] = float(update_window_policy_entropy / actor_update_denom)
                logs["train/log_pi_mean"] = float(update_window_log_pi_mean / actor_update_denom)
                logs["train/action_l2"] = float(update_window_action_l2 / actor_update_denom)
                logs["train/q_min_pi_mean"] = float(update_window_q_min_pi_mean / actor_update_denom)
                logs["train/q_min_data_mean"] = float(update_window_q_min_data_mean / update_denom)
                logs["train/q_disagreement_data_mean"] = float(
                    update_window_q_disagreement_data_mean / update_denom
                )
                logs["train/q_disagreement_pi_mean"] = float(
                    update_window_q_disagreement_pi_mean / actor_update_denom
                )
                logs["train/replay_reward_mean"] = float(update_window_replay_reward_mean / update_denom)
                logs["train/replay_reward_abs_mean"] = float(
                    update_window_replay_reward_abs_mean / update_denom
                )
                logs["train/actor_updates_per_iter"] = float(update_window_actor_updates)
                logs["train/alpha_updates_per_iter"] = float(update_window_alpha_updates)
                pref_loss_type = str(getattr(args, "pref_loss_type", "margin")).strip().lower()
                pref_lagrangian_violation_type = str(
                    getattr(args, "pref_lagrangian_violation_type", "hinge")
                ).strip().lower()
                logs["train/pref_lagrangian_enabled"] = 1.0 if pref_loss_type == "lagrangian" else 0.0
                logs["train/pref_lagrangian_violation_is_smooth"] = (
                    1.0 if pref_lagrangian_violation_type == "smooth" else 0.0
                )
                logs["train/pref_stopgrad_positive_enabled"] = (
                    1.0 if bool(getattr(args, "pref_stopgrad_positive", False)) else 0.0
                )
                logs["train/pref_lambda_lr"] = float(getattr(args, "pref_lambda_lr", 0.0))
                logs["train/pref_lambda_max"] = float(getattr(args, "pref_lambda_max", 0.0))
                logs["train/pref_lambda_ema_cfg"] = float(getattr(args, "pref_lambda_ema", 0.0))
                logs["train/pref_violation_clip"] = float(getattr(args, "pref_violation_clip", 0.0))
                logs["train/pref_violation_target"] = float(getattr(args, "pref_violation_target", 0.0))

            _emit_metrics(payload=logs, log_path=log_path, wandb_run=wandb_run, step=step)
            live_window_steps = 0
            live_window_intervention_steps = 0
            live_window_controller_ms = 0.0
            live_window_oversight_required_steps = 0
            live_window_uncertainty_all = []
            live_window_uncertainty_intervention = []
            live_window_reward_raw_env = []
            live_window_reward_shaped = []
            live_window_reward_dense = []
            live_window_reward_sparse = []
            live_window_reward_step_penalty = []
            live_window_reward_cost_penalty = []
            update_window_count = 0
            update_window_actor_updates = 0.0
            update_window_alpha_updates = 0.0
            update_window_policy_entropy = 0.0
            update_window_log_pi_mean = 0.0
            update_window_action_l2 = 0.0
            update_window_q_min_pi_mean = 0.0
            update_window_q_min_data_mean = 0.0
            update_window_q_disagreement_data_mean = 0.0
            update_window_q_disagreement_pi_mean = 0.0
            update_window_replay_reward_mean = 0.0
            update_window_replay_reward_abs_mean = 0.0
            if timing_enabled:
                timing_window.reset()

        if next_eval > 0 and step >= next_eval:
            next_eval += int(args.eval_interval)
            eval_metrics = _run_eval(
                sac.actor,
                args,
                device,
                eval_seed=args.seed + 100000 + step,
                obs_preprocess=obs_preprocess,
                run_paths=run_paths,
                step_value=int(step),
            )
            eval_metrics["eval/step"] = float(step)
            _emit_metrics(payload=eval_metrics, log_path=log_path, wandb_run=wandb_run, step=step)

        if next_save > 0 and step >= next_save:
            next_save += int(args.save_interval)
            ckpt_path = run_paths.model_dir / f"step_{step}.pt"
            torch.save(
                {
                    "actor_state_dict": sac.actor.state_dict(),
                    "critic_state_dict": sac.critic.state_dict(),
                    "critic_target_state_dict": sac.critic_target.state_dict(),
                    "actor_optimizer_state_dict": sac.actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": sac.critic_optimizer.state_dict(),
                    "alpha_optimizer_state_dict": sac.alpha_optimizer.state_dict(),
                    "log_alpha": sac.log_alpha.detach().cpu(),
                    "obs_normalizer_state_dict": (
                        obs_preprocess.state_dict() if hasattr(obs_preprocess, "state_dict") else {}
                    ),
                    "args": vars(args),
                    "global_step": int(step),
                    "variant": str(variant),
                },
                ckpt_path,
            )
            _maybe_render_policy_map(
                args=args,
                checkpoint_path=ckpt_path,
                step_value=int(step),
                run_paths=run_paths,
                wandb_run=wandb_run,
            )

    final_ckpt = run_paths.model_dir / "final.pt"
    torch.save(
        {
            "actor_state_dict": sac.actor.state_dict(),
            "critic_state_dict": sac.critic.state_dict(),
            "critic_target_state_dict": sac.critic_target.state_dict(),
            "actor_optimizer_state_dict": sac.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": sac.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": sac.alpha_optimizer.state_dict(),
            "log_alpha": sac.log_alpha.detach().cpu(),
            "obs_normalizer_state_dict": (
                obs_preprocess.state_dict() if hasattr(obs_preprocess, "state_dict") else {}
            ),
            "args": vars(args),
            "global_step": int(args.total_timesteps),
            "variant": str(variant),
        },
        final_ckpt,
    )
    _maybe_render_policy_map(
        args=args,
        checkpoint_path=final_ckpt,
        step_value=int(args.total_timesteps),
        run_paths=run_paths,
        wandb_run=wandb_run,
    )
    final_export_logs: Dict[str, float] = {}
    final_export_logs.update(
        _maybe_export_final_buffer_dataset(
            args=args,
            buffer=main_rb,
            env_name=args.env_name,
            variant=variant,
            reward_mode=args.reward_mode,
            label_suffix="replay",
            explicit_path=str(getattr(args, "export_final_replay_dataset_path", "") or ""),
            enabled=bool(getattr(args, "export_final_replay_dataset", False)),
            filter_mode="all",
        )
    )
    final_export_logs.update(
        _maybe_export_final_buffer_dataset(
            args=args,
            buffer=demo_rb,
            env_name=args.env_name,
            variant=variant,
            reward_mode=args.reward_mode,
            label_suffix="demo",
            explicit_path=str(getattr(args, "export_final_demo_dataset_path", "") or ""),
            enabled=bool(getattr(args, "export_final_demo_dataset", False)),
            filter_mode="all",
        )
    )
    if final_export_logs:
        _emit_metrics(
            payload=final_export_logs,
            log_path=log_path,
            wandb_run=wandb_run,
            step=int(args.total_timesteps),
        )

    env.close()
    if controller is not None:
        controller.close()
    if viewer is not None:
        viewer.close()
    if wandb_run is not None:
        wandb_run.finish()
