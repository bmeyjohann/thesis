#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.controllers import ExpertPolicyController, LearnedInterventionPolicyController
from safetygym_utils.env import clip_action_to_space, extract_goal_distance, make_safety_env
from safetygym_utils.metrics import EpisodeWindow, augment_rollout_summary
from safetygym_utils.rendering import build_external_viewer, resolve_env_render_mode, wants_external_viewer
from safetygym_utils.wrappers import HumanInterventionWrapper, RewardModeWrapper, TerminateOnGoalWrapper


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a learned Safety-Gym imitation intervention teacher")
    p.add_argument("--teacher_checkpoint_path", type=str, required=True)
    p.add_argument("--student_checkpoint_path", type=str, default="")
    p.add_argument("--student_policy", type=str, default="random", choices=["random", "zero", "checkpoint"])
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--render_mode", type=str, default="none", choices=["human", "rgb_array", "none", "pygame", "topdown"])
    p.add_argument("--viewer_fps", type=float, default=20.0)
    p.add_argument("--fps", type=float, default=0.0)
    p.add_argument("--reward_mode", type=str, default="dense", choices=["sparse", "dense", "dense_plus_sparse", "native", "none"])
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=2.0)
    p.add_argument("--car_force_scale", type=float, default=2.0)
    p.add_argument("--car_action_mode", type=str, default="raw_wheels", choices=["raw_wheels", "throttle_turn", "cardinal"])
    p.add_argument("--point_action_mode", type=str, default="native", choices=["native", "world_velocity"])
    p.add_argument(
        "--obs_mask_mode",
        type=str,
        default="none",
        choices=["none", "goal_only_lidar", "privileged_geometry", "privileged_geometry_rich"],
    )
    p.add_argument("--terminate_on_goal", action="store_true", default=False)
    p.add_argument("--intervention_threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    return p


def _episode_metrics(
    *,
    ep_ret: float,
    ep_cost: float,
    ep_len: int,
    interventions: int,
    intervention_bursts: int,
    burst_steps_sum: int,
    probs: list[float],
    goal_met_count: int,
    first_goal_hit_step: int | None,
    first_goal_reward_sum: float,
    first_goal_dense_reward_sum: float,
    final_distance: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, float]:
    goal_met = bool(goal_met_count > 0)
    first_goal_success = 1.0 if goal_met else 0.0
    first_hit = float(first_goal_hit_step if first_goal_hit_step is not None else ep_len)
    timeout = bool((not goal_met) and truncated)
    kill = bool((not goal_met) and terminated)
    avg_burst = float(burst_steps_sum) / max(1, int(intervention_bursts))
    return {
        "episode_return": float(ep_ret),
        "episode_cost_sum": float(ep_cost),
        "episode_cost_rate": float(ep_cost / max(1, ep_len)),
        "episode_length": float(ep_len),
        "intervention_steps": float(interventions),
        "intervention_fraction": float(interventions / max(1, ep_len)),
        "intervention_num_bursts": float(intervention_bursts),
        "intervention_avg_burst_len": float(avg_burst),
        "goal_met": float(first_goal_success),
        "goal_met_count": float(goal_met_count),
        "first_goal_success": float(first_goal_success),
        "first_goal_hit_step": float(first_hit),
        "first_goal_hit_step_success_only": float(first_hit if goal_met else 0.0),
        "first_goal_within_100": float(1.0 if goal_met and first_hit <= 100 else 0.0),
        "first_goal_within_200": float(1.0 if goal_met and first_hit <= 200 else 0.0),
        "first_goal_reward_sum": float(first_goal_reward_sum if goal_met else ep_ret),
        "first_goal_dense_reward_sum": float(first_goal_dense_reward_sum if goal_met else ep_ret),
        "final_distance_to_goal": float(final_distance if np.isfinite(final_distance) else 0.0),
        "outcome_success": float(first_goal_success),
        "outcome_timeout": float(timeout),
        "outcome_kill": float(kill),
        "outcome_other_failure": float(0.0 if goal_met or timeout or kill else 1.0),
        "terminated": float(terminated),
        "truncated": float(truncated),
        "teacher_fraction": float(interventions / max(1, ep_len)),
        "teacher_steps": float(interventions),
        "teacher_prob_mean": float(np.mean(probs) if probs else 0.0),
        "teacher_prob_max": float(np.max(probs) if probs else 0.0),
    }


def main() -> int:
    args = build_parser().parse_args()
    env = make_safety_env(
        args.env_name,
        render_mode=resolve_env_render_mode(args.render_mode),
        max_episode_steps=int(args.max_episode_steps),
        surface_mode=str(args.surface_mode),
        car_wheel_command_limit=float(args.car_wheel_command_limit),
        car_force_scale=float(args.car_force_scale),
        car_action_mode=str(args.car_action_mode),
        point_action_mode=str(args.point_action_mode),
        obs_mask_mode=str(args.obs_mask_mode),
        seed=int(args.seed),
    )
    env = RewardModeWrapper(
        env,
        reward_mode=str(args.reward_mode),
        dense_reward_scale=float(args.dense_reward_scale),
        step_penalty=float(args.step_penalty),
    )
    if bool(args.terminate_on_goal):
        env = TerminateOnGoalWrapper(env)

    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    obs, _ = env.reset(seed=int(args.seed))
    obs_dim = int(np.asarray(obs, dtype=np.float32).reshape(-1).shape[0])
    act_dim = int(low.shape[0])

    teacher = LearnedInterventionPolicyController(
        checkpoint_path=args.teacher_checkpoint_path,
        action_low=low,
        action_high=high,
        intervention_threshold=float(args.intervention_threshold),
        device=str(args.device),
    )
    env = HumanInterventionWrapper(env, controller=teacher, threshold=0.0, hold_seconds=0.0)

    student = None
    if args.student_policy == "checkpoint":
        if not str(args.student_checkpoint_path).strip():
            raise ValueError("--student_policy checkpoint requires --student_checkpoint_path")
        student = ExpertPolicyController(
            checkpoint_path=args.student_checkpoint_path,
            obs_dim=obs_dim,
            action_low=low,
            action_high=high,
            device=str(args.device),
        )

    viewer = (
        build_external_viewer(
            render_mode=args.render_mode,
            title=f"SafetyGym learned teacher eval {args.env_name}",
            draw_hz=float(args.viewer_fps),
            scale=1.0,
        )
        if wants_external_viewer(args.render_mode)
        else None
    )
    win = EpisodeWindow(size=max(1, int(args.num_episodes)))
    try:
        for ep in range(int(args.num_episodes)):
            obs, _info = env.reset(seed=int(args.seed) + ep)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            done = False
            ep_ret = 0.0
            ep_cost = 0.0
            ep_len = 0
            interventions = 0
            intervention_bursts = 0
            burst_steps_sum = 0
            in_burst = False
            current_burst_len = 0
            probs: list[float] = []
            goal_met_count = 0
            first_goal_hit_step: int | None = None
            first_goal_reward_sum = 0.0
            first_goal_dense_reward_sum = 0.0
            final_info = {}
            terminated = False
            truncated = False
            if viewer is not None:
                viewer.draw_env(env)
            while not done:
                if args.student_policy == "zero":
                    student_action = np.zeros((act_dim,), dtype=np.float32)
                elif args.student_policy == "checkpoint" and student is not None:
                    student_action = student.get_action(obs=obs, env=env)
                    if student_action is None:
                        student_action = np.zeros((act_dim,), dtype=np.float32)
                else:
                    student_action = env.action_space.sample().astype(np.float32)
                student_action = clip_action_to_space(student_action, env.action_space)
                obs, reward, cost, terminated, truncated, info = env.step(student_action)
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                ep_ret += float(reward)
                ep_cost += float(cost)
                ep_len += 1
                if bool(info.get("teacher_intervened", False)):
                    interventions += 1
                    current_burst_len += 1
                    if not in_burst:
                        in_burst = True
                        intervention_bursts += 1
                elif in_burst:
                    burst_steps_sum += int(current_burst_len)
                    current_burst_len = 0
                    in_burst = False
                if bool(info.get("goal_met", False)):
                    goal_met_count += 1
                    if first_goal_hit_step is None:
                        first_goal_hit_step = int(ep_len)
                        first_goal_reward_sum = float(ep_ret)
                        first_goal_dense_reward_sum = float(ep_ret)
                probs.append(float(info.get("teacher_intervention_probability", 0.0)))
                if viewer is not None:
                    viewer.draw_env(env)
                final_info = dict(info)
                done = bool(terminated or truncated)
                if float(args.fps) > 0:
                    time.sleep(1.0 / float(args.fps))
            if in_burst:
                burst_steps_sum += int(current_burst_len)
            final_distance = extract_goal_distance(env)
            if not np.isfinite(final_distance):
                final_distance = float(final_info.get("goal_distance", 0.0) or 0.0)
            metrics = _episode_metrics(
                ep_ret=ep_ret,
                ep_cost=ep_cost,
                ep_len=ep_len,
                interventions=interventions,
                intervention_bursts=intervention_bursts,
                burst_steps_sum=burst_steps_sum,
                probs=probs,
                goal_met_count=goal_met_count,
                first_goal_hit_step=first_goal_hit_step,
                first_goal_reward_sum=first_goal_reward_sum,
                first_goal_dense_reward_sum=first_goal_dense_reward_sum,
                final_distance=float(final_distance),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            win.add(metrics)
            print(json.dumps({"episode": ep + 1, **metrics}, sort_keys=True), flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        if student is not None:
            student.close()
        env.close()

    print(json.dumps(augment_rollout_summary(win.summary("eval"), "eval"), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
