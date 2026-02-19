#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict
import sys

import numpy as np
import torch

# Ensure local FastSAC package is importable without installation.
_FAST_SAC_PATH = Path(__file__).resolve().parent / "fasttd3" / "fast_sac"
if _FAST_SAC_PATH.exists():
    _fast_sac_path_str = str(_FAST_SAC_PATH)
    if _fast_sac_path_str not in sys.path:
        sys.path.insert(0, _fast_sac_path_str)

from fast_sac import Actor

from safetygym_utils.controllers import KeyboardConfig, PygameKeyboardController, infer_control_scheme
from safetygym_utils.env import clip_action_to_space, extract_goal_distance, extract_step_limit, make_safety_env
from safetygym_utils.io import load_args_json, maybe_find_args_json_from_model
from safetygym_utils.metrics import EpisodeWindow, classify_outcome
from safetygym_utils.wrappers import HumanInterventionWrapper, RewardModeWrapper


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive evaluation for Safety-Gymnasium FastSAC checkpoints")
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--controller", type=str, default="policy", choices=["policy", "random", "human", "keyboard"])
    p.add_argument("--intervention_mode", type=str, default="none", choices=["none", "human"])
    p.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array", "none"])
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=2.0)
    p.add_argument("--car_force_scale", type=float, default=2.0)
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--fps", type=int, default=30)

    p.add_argument("--reward_mode", type=str, default="sparse", choices=["sparse", "dense", "none"])
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)

    p.add_argument("--intervention_threshold", type=float, default=0.1)
    p.add_argument("--intervention_hold_seconds", type=float, default=0.25)
    p.add_argument("--human_action_scale", type=float, default=1.0)
    p.add_argument("--controller_fps_limit", type=int, default=0)
    p.add_argument("--controller_overlay_hz", type=float, default=20.0)

    p.add_argument("--actor_hidden_dim", type=int, default=512)
    p.add_argument("--init_scale", type=float, default=0.01)
    p.add_argument("--load_checkpoint_args", action="store_true", default=True)
    p.add_argument("--no_load_checkpoint_args", dest="load_checkpoint_args", action="store_false")
    return p


def _apply_ckpt_defaults(args: argparse.Namespace) -> None:
    if not args.model_path or not args.load_checkpoint_args:
        return
    args_path = maybe_find_args_json_from_model(Path(args.model_path))
    if args_path is None or not args_path.exists():
        return
    cfg = load_args_json(args_path)

    def _set_if_default(name: str, default_val):
        if getattr(args, name) == default_val and name in cfg:
            setattr(args, name, cfg[name])

    _set_if_default("env_name", "SafetyCarGoal2-v0")
    _set_if_default("reward_mode", "sparse")
    _set_if_default("dense_reward_scale", 1.0)
    _set_if_default("step_penalty", 0.0)
    _set_if_default("surface_mode", "default")
    _set_if_default("car_wheel_command_limit", 2.0)
    _set_if_default("car_force_scale", 2.0)
    _set_if_default("actor_hidden_dim", 512)
    _set_if_default("init_scale", 0.01)


def _build_env(args: argparse.Namespace, controller: PygameKeyboardController | None):
    env = make_safety_env(
        args.env_name,
        render_mode=args.render_mode,
        max_episode_steps=args.max_episode_steps,
        surface_mode=args.surface_mode,
        car_wheel_command_limit=args.car_wheel_command_limit,
        car_force_scale=args.car_force_scale,
        seed=args.seed,
    )
    env = RewardModeWrapper(
        env,
        reward_mode=args.reward_mode,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
    )
    if args.intervention_mode == "human":
        if controller is None:
            raise ValueError("controller is required for intervention_mode=human")
        env = HumanInterventionWrapper(
            env,
            controller=controller,
            threshold=args.intervention_threshold,
            hold_seconds=args.intervention_hold_seconds,
        )
    return env


def _episode_metrics(info: Dict[str, Any], ep_return: float, ep_cost: float, ep_len: int, max_steps: int, terminated: bool, truncated: bool, final_distance: float) -> Dict[str, float]:
    goal_met = bool(info.get("goal_met", False))
    outcome = classify_outcome(goal_met=goal_met, episode_steps=ep_len, max_episode_steps=max_steps)
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
        "final_distance_to_goal": float(final_distance),
        "outcome_success": 1.0 if outcome == "success" else 0.0,
        "outcome_timeout": 1.0 if outcome == "timeout" else 0.0,
        "outcome_kill": 1.0 if outcome == "kill" else 0.0,
        "outcome_other_failure": 1.0 if outcome not in {"success", "timeout", "kill"} else 0.0,
        "terminated": 1.0 if terminated else 0.0,
        "truncated": 1.0 if truncated else 0.0,
    }


def main() -> int:
    args = build_parser().parse_args()
    _apply_ckpt_defaults(args)

    if args.controller in {"human", "keyboard"} and args.intervention_mode == "human":
        # Avoid double-human override (controller action + intervention wrapper action).
        args.intervention_mode = "none"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    controller = None
    if args.controller in {"human", "keyboard"} or args.intervention_mode == "human":
        control_scheme = infer_control_scheme(args.env_name)
        # Temporary env for action-dim detection.
        tmp_env = make_safety_env(
            args.env_name,
            render_mode="none",
            max_episode_steps=args.max_episode_steps,
            surface_mode=args.surface_mode,
            car_wheel_command_limit=args.car_wheel_command_limit,
            car_force_scale=args.car_force_scale,
            seed=args.seed,
        )
        act_dim = int(np.prod(tmp_env.action_space.shape))
        tmp_env.close()
        controller = PygameKeyboardController(
            action_dim=act_dim,
            config=KeyboardConfig(
                action_scale=args.human_action_scale,
                control_scheme=control_scheme,
                overlay_fps_limit=int(args.controller_fps_limit),
                overlay_draw_hz=float(args.controller_overlay_hz),
                wheel_command_limit=float(args.car_wheel_command_limit),
            ),
        )

    env = _build_env(args, controller)
    max_steps = extract_step_limit(env)

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    obs_dim = int(obs.shape[0])
    act_dim = int(np.prod(env.action_space.shape))

    actor = None
    if args.controller == "policy":
        if not args.model_path:
            raise ValueError("--model_path is required for --controller policy")
        actor = Actor(
            n_obs=obs_dim,
            n_act=act_dim,
            num_envs=1,
            init_scale=args.init_scale,
            hidden_dim=args.actor_hidden_dim,
            device=device,
        )
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()

    win = EpisodeWindow(size=max(10, args.num_episodes))

    ep_ret = 0.0
    ep_cost = 0.0
    ep_len = 0
    episodes = 0

    while episodes < args.num_episodes:
        if args.controller == "policy":
            with torch.no_grad():
                obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
                _, _, mean = actor(obs_t)
                action = mean[0].detach().cpu().numpy().astype(np.float32)
        elif args.controller == "random":
            action = env.action_space.sample().astype(np.float32)
        else:
            if controller is None:
                raise RuntimeError("Controller requested but not initialized")
            action = controller.get_action().astype(np.float32)

        action = clip_action_to_space(action, env.action_space)
        next_obs, reward, cost, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        ep_ret += float(reward)
        ep_cost += float(cost)
        ep_len += 1

        if terminated or truncated:
            final_dist = extract_goal_distance(env)
            ep = _episode_metrics(
                dict(info),
                ep_return=ep_ret,
                ep_cost=ep_cost,
                ep_len=ep_len,
                max_steps=max_steps,
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_distance=final_dist,
            )
            win.add(ep)
            outcome = "success" if ep["outcome_success"] > 0.5 else ("timeout" if ep["outcome_timeout"] > 0.5 else "kill")
            print(
                f"episode={episodes+1} outcome={outcome} return={ep_ret:.3f} cost={ep_cost:.3f} "
                f"len={ep_len} final_dist={final_dist:.3f} interventions={int(ep['intervention_steps'])}",
                flush=True,
            )
            episodes += 1
            obs, _ = env.reset(seed=args.seed + episodes)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            ep_ret = 0.0
            ep_cost = 0.0
            ep_len = 0
        else:
            obs = next_obs

        if args.fps > 0:
            time.sleep(1.0 / float(args.fps))

    summary = win.summary("eval")
    print(json.dumps(summary, sort_keys=True), flush=True)

    env.close()
    if controller is not None:
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
