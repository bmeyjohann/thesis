from __future__ import annotations

import json
import random
import time
from collections import deque
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

from fast_sac_utils import SimpleReplayBuffer

from .controllers import KeyboardConfig, PygameKeyboardController, infer_control_scheme
from .env import clip_action_to_space, extract_goal_distance, extract_step_limit, make_safety_env
from .io import save_args_json
from .metrics import EpisodeWindow, classify_outcome
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


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(y.shape[0], dtype=np.float64)
    x_center = x - np.mean(x)
    denom = float(np.sum(x_center * x_center))
    if denom <= 0.0:
        return 0.0
    y_center = y - np.mean(y)
    return float(np.sum(x_center * y_center) / denom)


def _compute_uncertainty_signal(
    *,
    sac: SACTensors,
    obs: np.ndarray,
    action: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
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


def _make_env_with_wrappers(
    *,
    args,
    seed: int,
    with_intervention: bool,
    controller: Optional[PygameKeyboardController],
):
    env = make_safety_env(
        args.env_name,
        render_mode=args.render_mode,
        max_episode_steps=args.max_episode_steps,
        surface_mode=getattr(args, "surface_mode", "default"),
        car_wheel_command_limit=float(getattr(args, "car_wheel_command_limit", 2.0)),
        car_force_scale=float(getattr(args, "car_force_scale", 2.0)),
        seed=seed,
    )
    env = RewardModeWrapper(
        env,
        reward_mode=args.reward_mode,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
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
) -> TensorDict:
    return TensorDict(
        {
            "observations": torch.as_tensor(obs[None, :], device=device, dtype=torch.float32),
            "actions": torch.as_tensor(action[None, :], device=device, dtype=torch.float32),
            "next": {
                "observations": torch.as_tensor(next_obs[None, :], device=device, dtype=torch.float32),
                "rewards": torch.as_tensor([reward], device=device, dtype=torch.float32),
                "dones": torch.as_tensor([done], device=device, dtype=torch.bool),
                "truncations": torch.as_tensor([truncated], device=device, dtype=torch.bool),
                "effective_n_steps": torch.ones(1, device=device, dtype=torch.float32),
            },
        },
        batch_size=(1,),
        device=device,
    )


def _episode_metrics_from_info(
    *,
    info: Dict[str, Any],
    ep_return: float,
    ep_cost: float,
    ep_len: int,
    max_episode_steps: int,
    terminated: bool,
    truncated: bool,
    final_distance_to_goal: float,
) -> Dict[str, float]:
    goal_met = bool(info.get("goal_met", False))
    outcome = classify_outcome(goal_met=goal_met, episode_steps=ep_len, max_episode_steps=max_episode_steps)
    return {
        "episode_return": float(ep_return),
        "episode_cost_sum": float(ep_cost),
        "episode_cost_rate": float(ep_cost / max(1, ep_len)),
        "episode_length": float(ep_len),
        "intervention_steps": float(info.get("teacher_intervention_steps", 0.0)),
        "intervention_fraction": float(info.get("teacher_fraction_steps", 0.0)),
        "intervention_num_bursts": float(info.get("teacher_num_bursts", 0.0)),
        "intervention_avg_burst_len": float(info.get("teacher_avg_burst_len", 0.0)),
        "goal_met": 1.0 if goal_met else 0.0,
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
) -> np.ndarray:
    mode = str(policy).lower()
    if mode == "random":
        return env.action_space.sample().astype(np.float32)
    if mode == "zero":
        return np.zeros_like(np.asarray(env.action_space.low, dtype=np.float32).reshape(-1), dtype=np.float32)
    if mode == "student":
        with torch.no_grad():
            obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
            act_t, _, _ = sac.actor(obs_t)
            return act_t[0].detach().cpu().numpy().astype(np.float32)
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
        ep_steps = 0
        while True:
            student_action = _select_prefill_action(policy=prefill_policy, obs=obs, env=env, sac=sac, device=device)
            student_action = clip_action_to_space(student_action, env.action_space)
            next_obs, reward, _cost, terminated, truncated, info = env.step(student_action)
            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

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
            pref_batch=None,
            pref_rank_weight=0.0,
            pref_rank_margin=float(getattr(args, "pref_rank_margin", 0.1)),
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


def _run_eval(actor, args, device: torch.device, eval_seed: int) -> Dict[str, float]:
    env = _make_env_with_wrappers(args=args, seed=eval_seed, with_intervention=False, controller=None)
    max_steps = extract_step_limit(env)
    win = EpisodeWindow(size=max(10, args.num_eval_episodes))

    episodes = 0
    obs, _ = env.reset(seed=eval_seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    ep_ret = 0.0
    ep_cost = 0.0
    ep_len = 0

    while episodes < args.num_eval_episodes:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
            _, _, mean = actor(obs_t)
            action = mean[0].detach().cpu().numpy().astype(np.float32)
        action = clip_action_to_space(action, env.action_space)
        next_obs, reward, cost, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        ep_ret += float(reward)
        ep_cost += float(cost)
        ep_len += 1

        if terminated or truncated:
            final_dist = extract_goal_distance(env)
            epm = _episode_metrics_from_info(
                info=dict(info),
                ep_return=ep_ret,
                ep_cost=ep_cost,
                ep_len=ep_len,
                max_episode_steps=max_steps,
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_distance_to_goal=final_dist,
            )
            win.add(epm)
            episodes += 1
            obs, _ = env.reset(seed=eval_seed + episodes)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            ep_ret = 0.0
            ep_cost = 0.0
            ep_len = 0
        else:
            obs = next_obs

    env.close()
    return win.summary("eval")


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

    controller = None
    with_intervention = bool(args.use_intervention)
    control_scheme = infer_control_scheme(args.env_name)
    if with_intervention:
        controller = PygameKeyboardController(
            action_dim=2,
            config=KeyboardConfig(
                action_scale=args.human_action_scale,
                control_scheme=control_scheme,
                overlay_fps_limit=int(getattr(args, "controller_fps_limit", 0)),
                overlay_draw_hz=float(getattr(args, "controller_overlay_hz", 20.0)),
                wheel_command_limit=float(getattr(args, "car_wheel_command_limit", 2.0)),
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
    obs_dim = int(obs.shape[0])
    act_dim = int(np.prod(env.action_space.shape))

    if with_intervention and controller is not None and controller.action_dim != act_dim:
        controller.close()
        controller = PygameKeyboardController(
            action_dim=act_dim,
            config=KeyboardConfig(
                action_scale=args.human_action_scale,
                control_scheme=control_scheme,
                overlay_fps_limit=int(getattr(args, "controller_fps_limit", 0)),
                overlay_draw_hz=float(getattr(args, "controller_overlay_hz", 20.0)),
                wheel_command_limit=float(getattr(args, "car_wheel_command_limit", 2.0)),
            ),
        )
        env.close()
        env = _make_env_with_wrappers(args=args, seed=args.seed, with_intervention=True, controller=controller)
        obs, _ = env.reset(seed=args.seed)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

    max_steps = extract_step_limit(env)

    sac: SACTensors = build_sac(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_actor=args.actor_hidden_dim,
        hidden_critic=args.critic_hidden_dim,
        init_scale=args.init_scale,
        lr_actor=args.actor_learning_rate,
        lr_critic=args.critic_learning_rate,
        weight_decay=args.weight_decay,
        num_envs=1,
        device=device,
    )

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
    if variant == "hilserl" or int(getattr(args, "prefill_demo_episodes", 0)) > 0 or int(getattr(args, "demo_pretrain_updates", 0)) > 0:
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
    )
    if prefill_logs:
        prefill_line = json.dumps(prefill_logs, sort_keys=True)
        print(prefill_line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(prefill_line + "\n")

    last_update: Optional[SACUpdateMetrics] = None
    pretrain_update, pretrain_logs = _run_demo_pretrain(args=args, sac=sac, demo_rb=demo_rb)
    if pretrain_update is not None:
        last_update = pretrain_update
    if pretrain_logs:
        pretrain_line = json.dumps(pretrain_logs, sort_keys=True)
        print(pretrain_line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(pretrain_line + "\n")

    pretrain_updates_run = int(pretrain_logs.get("train/pretrain_updates_run", 0.0)) if pretrain_logs else 0
    if bool(getattr(args, "critic_reset_after_pretrain", False)) and pretrain_updates_run > 0:
        reset_critic(
            sac=sac,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_critic=int(args.critic_hidden_dim),
            lr_critic=float(args.critic_learning_rate),
            weight_decay=float(args.weight_decay),
            device=device,
        )
        reset_line = json.dumps({"train/critic_reset_after_pretrain": 1.0}, sort_keys=True)
        print(reset_line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(reset_line + "\n")
    elif bool(getattr(args, "critic_reset_after_pretrain", False)):
        reset_skip_line = json.dumps({"train/critic_reset_after_pretrain": 0.0}, sort_keys=True)
        print(reset_skip_line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(reset_skip_line + "\n")

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    prefill_steps_completed = int(prefill_logs.get("train/prefill_steps", 0.0)) if prefill_logs else 0
    effective_learning_starts = max(0, int(args.learning_starts) - prefill_steps_completed)

    ep_ret = 0.0
    ep_cost = 0.0
    ep_len = 0

    next_log = int(args.log_interval)
    next_eval = int(args.eval_interval) if args.eval_interval > 0 else -1
    next_save = int(args.save_interval) if args.save_interval > 0 else -1

    start_time = time.time()
    live_window_steps = 0
    live_window_intervention_steps = 0
    live_window_controller_ms = 0.0
    live_window_oversight_required_steps = 0
    live_window_uncertainty_all: list[float] = []
    live_window_uncertainty_intervention: list[float] = []
    live_window_uncertainty_no_intervention: list[float] = []
    live_window_uncertainty_qmin: list[float] = []
    live_window_uncertainty_qmax: list[float] = []
    live_window_preint_delta: list[float] = []
    live_window_preint_slope: list[float] = []
    live_window_preint_z: list[float] = []
    preint_hist = deque(maxlen=max(2, int(getattr(args, "uncertainty_pre_intervention_window", 25))))
    ep_uncertainty_values: list[float] = []
    prev_teacher_intervened = False
    oversight_mode = str(getattr(args, "uncertainty_oversight_mode", "signal_only")).lower()
    oversight_threshold = float(getattr(args, "uncertainty_oversight_threshold", 0.0))
    oversight_enabled = oversight_mode != "off" and oversight_threshold > 0.0
    oversight_ema_alpha = float(np.clip(float(getattr(args, "uncertainty_oversight_ema_alpha", 0.05)), 1e-6, 1.0))
    oversight_ema: Optional[float] = None
    uncertainty_enabled = bool(getattr(args, "uncertainty_log_every_step", True)) or oversight_enabled
    timing_enabled = bool(getattr(args, "profile_timing", False))
    timing_window = TimingWindow()

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
                action_t, _, _ = sac.actor(obs_t)
                student_action = action_t[0].detach().cpu().numpy().astype(np.float32)
        student_action = clip_action_to_space(student_action, env.action_space)
        t_action += time.perf_counter() - t0

        t0 = time.perf_counter()
        next_obs, reward, cost, terminated, truncated, info = env.step(student_action)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        t_env += time.perf_counter() - t0

        t0 = time.perf_counter()
        teacher_intervened = bool(info.get("teacher_intervened", False))
        live_window_steps += 1
        live_window_controller_ms += float(info.get("teacher_controller_ms", 0.0))
        if teacher_intervened:
            live_window_intervention_steps += 1
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
        )

        main_rb.extend(transition)
        if variant == "pvp":
            if teacher_intervened and human_rb is not None:
                human_rb.extend(transition)
            elif novice_rb is not None:
                novice_rb.extend(transition)
        if teacher_intervened and demo_rb is not None:
            demo_rb.extend(transition)

        if teacher_intervened and pref_capacity > 0 and variant == "own":
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
            )
            unc_val = float(unc["abs_diff"])
            live_window_uncertainty_all.append(unc_val)
            live_window_uncertainty_qmin.append(float(unc["q_min"]))
            live_window_uncertainty_qmax.append(float(unc["q_max"]))
            if teacher_intervened:
                live_window_uncertainty_intervention.append(unc_val)
            else:
                live_window_uncertainty_no_intervention.append(unc_val)

            if teacher_intervened and not prev_teacher_intervened and len(preint_hist) >= 2:
                hist_vals = [float(v) for v in list(preint_hist)]
                delta = float(hist_vals[-1] - hist_vals[0])
                slope = float(_linear_slope(hist_vals))
                if len(ep_uncertainty_values) >= 2:
                    ep_mu = float(np.mean(np.asarray(ep_uncertainty_values, dtype=np.float64)))
                    ep_std = float(np.std(np.asarray(ep_uncertainty_values, dtype=np.float64)))
                    z = float((hist_vals[-1] - ep_mu) / max(1e-6, ep_std))
                else:
                    z = 0.0
                live_window_preint_delta.append(delta)
                live_window_preint_slope.append(slope)
                live_window_preint_z.append(z)

            preint_hist.append(unc_val)
            ep_uncertainty_values.append(unc_val)

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
        ep_len += 1

        rb_ready = main_rb.size >= int(args.batch_size)
        if variant == "pvp" and novice_rb is not None:
            rb_ready = novice_rb.size >= max(1, int(args.batch_size // 2))

        update_every = max(1, int(getattr(args, "update_every", 1)))
        updates_per_cycle = max(1, int(getattr(args, "updates_per_cycle", 1)))
        if step > int(effective_learning_starts) and rb_ready and (step % update_every == 0):
            t0 = time.perf_counter()
            for _ in range(updates_per_cycle):
                if variant == "pvp" and novice_rb is not None and human_rb is not None:
                    half = max(1, int(args.batch_size // 2))
                    if human_rb.size >= half:
                        batch_n = novice_rb.sample(max(1, int(args.batch_size - half)))
                        batch_h = human_rb.sample(half)
                        batch = TensorDict.cat([batch_n, batch_h], dim=0)
                    else:
                        batch = novice_rb.sample(int(args.batch_size))
                elif variant == "hilserl" and demo_rb is not None and demo_rb.size > 0 and float(args.demo_sample_ratio) > 0.0:
                    demo_n = int(max(1, args.batch_size * float(args.demo_sample_ratio)))
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
                if variant == "own" and float(args.pref_sample_ratio) > 0.0 and pref_pairs:
                    pref_n = max(1, int(args.batch_size * float(args.pref_sample_ratio)))
                    pref_batch = _sample_pref_batch(pref_pairs, pref_n, device)

                last_update = sac_update_step(
                    sac=sac,
                    batch=batch,
                    gamma=float(args.gamma),
                    tau=float(args.tau),
                    max_grad_norm=float(args.max_grad_norm),
                    pref_batch=pref_batch,
                    pref_rank_weight=float(getattr(args, "pref_rank_weight", 0.0)),
                    pref_rank_margin=float(getattr(args, "pref_rank_margin", 0.1)),
                )
            t_update += time.perf_counter() - t0

        if terminated or truncated:
            t0 = time.perf_counter()
            final_dist = extract_goal_distance(env)
            epm = _episode_metrics_from_info(
                info=dict(info),
                ep_return=ep_ret,
                ep_cost=ep_cost,
                ep_len=ep_len,
                max_episode_steps=max_steps,
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_distance_to_goal=final_dist,
            )
            train_win.add(epm)

            obs, _ = env.reset(seed=args.seed + step)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            ep_ret = 0.0
            ep_cost = 0.0
            ep_len = 0
            preint_hist.clear()
            ep_uncertainty_values = []
            prev_teacher_intervened = False
            t_episode_end += time.perf_counter() - t0

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
                "train/effective_learning_starts": float(effective_learning_starts),
                "train/live_window_steps": float(live_window_steps),
                "train/live_intervention_steps": float(live_window_intervention_steps),
                "train/live_intervention_fraction": float(live_window_intervention_steps / max(1, live_window_steps)),
                "train/live_controller_ms_per_step": float(live_window_controller_ms / max(1, live_window_steps)),
            }
            if uncertainty_enabled:
                logs.update(_summary_stats(live_window_uncertainty_all, "train/uncertainty_all"))
                logs.update(_summary_stats(live_window_uncertainty_intervention, "train/uncertainty_intervention"))
                logs.update(_summary_stats(live_window_uncertainty_no_intervention, "train/uncertainty_no_intervention"))
                logs.update(_summary_stats(live_window_uncertainty_qmin, "train/qmin_all"))
                logs.update(_summary_stats(live_window_uncertainty_qmax, "train/qmax_all"))
                logs["train/uncertainty_gap_intervention_minus_no_intervention"] = float(
                    logs.get("train/uncertainty_intervention_mean", 0.0)
                    - logs.get("train/uncertainty_no_intervention_mean", 0.0)
                )
                logs["train/uncertainty_preint_events"] = float(len(live_window_preint_delta))
                logs["train/uncertainty_preint_delta_mean"] = float(
                    np.mean(np.asarray(live_window_preint_delta, dtype=np.float64))
                    if live_window_preint_delta
                    else 0.0
                )
                logs["train/uncertainty_preint_slope_mean"] = float(
                    np.mean(np.asarray(live_window_preint_slope, dtype=np.float64))
                    if live_window_preint_slope
                    else 0.0
                )
                logs["train/uncertainty_preint_z_mean"] = float(
                    np.mean(np.asarray(live_window_preint_z, dtype=np.float64))
                    if live_window_preint_z
                    else 0.0
                )
            if oversight_enabled:
                logs["train/oversight_threshold"] = float(oversight_threshold)
                logs["train/oversight_required_steps"] = float(live_window_oversight_required_steps)
                logs["train/oversight_required_fraction"] = float(
                    live_window_oversight_required_steps / max(1, live_window_steps)
                )
                logs["train/oversight_ema"] = float(oversight_ema if oversight_ema is not None else 0.0)
            if timing_enabled:
                logs.update(timing_window.summary("train_timing"))
            logs.update(train_win.summary("train"))
            if variant == "pvp" and novice_rb is not None and human_rb is not None:
                logs["train/novice_replay_size"] = float(novice_rb.size)
                logs["train/human_replay_size"] = float(human_rb.size)
            if demo_rb is not None:
                logs["train/demo_replay_size"] = float(demo_rb.size)
            if last_update is not None:
                logs.update(
                    {
                        "train/critic_loss": last_update.critic_loss,
                        "train/actor_loss": last_update.actor_loss,
                        "train/alpha_loss": last_update.alpha_loss,
                        "train/alpha": last_update.alpha,
                        "train/target_q_mean": last_update.target_q_mean,
                    }
                )

            line = json.dumps(logs, sort_keys=True)
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            live_window_steps = 0
            live_window_intervention_steps = 0
            live_window_controller_ms = 0.0
            live_window_oversight_required_steps = 0
            live_window_uncertainty_all = []
            live_window_uncertainty_intervention = []
            live_window_uncertainty_no_intervention = []
            live_window_uncertainty_qmin = []
            live_window_uncertainty_qmax = []
            live_window_preint_delta = []
            live_window_preint_slope = []
            live_window_preint_z = []
            if timing_enabled:
                timing_window.reset()

        if next_eval > 0 and step >= next_eval:
            next_eval += int(args.eval_interval)
            eval_metrics = _run_eval(sac.actor, args, device, eval_seed=args.seed + 100000 + step)
            eval_metrics["eval/step"] = float(step)
            line = json.dumps(eval_metrics, sort_keys=True)
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        if next_save > 0 and step >= next_save:
            next_save += int(args.save_interval)
            ckpt_path = run_paths.model_dir / f"step_{step}.pt"
            torch.save(
                {
                    "actor_state_dict": sac.actor.state_dict(),
                    "critic_state_dict": sac.critic.state_dict(),
                    "critic_target_state_dict": sac.critic_target.state_dict(),
                    "log_alpha": sac.log_alpha.detach().cpu(),
                    "args": vars(args),
                },
                ckpt_path,
            )

    final_ckpt = run_paths.model_dir / "final.pt"
    torch.save(
        {
            "actor_state_dict": sac.actor.state_dict(),
            "critic_state_dict": sac.critic.state_dict(),
            "critic_target_state_dict": sac.critic_target.state_dict(),
            "log_alpha": sac.log_alpha.detach().cpu(),
            "args": vars(args),
        },
        final_ckpt,
    )

    env.close()
    if controller is not None:
        controller.close()
