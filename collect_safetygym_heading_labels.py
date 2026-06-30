#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.controllers import ScriptedGeometricTeacherController, extract_agent_forward_xy
from safetygym_utils.env import make_safety_env, resolve_control_scheme
from safetygym_utils.heading_policy import load_heading_policy
from safetygym_utils.rendering import resolve_env_render_mode
from safetygym_utils.wrappers import RewardModeWrapper, SafetyLayoutCurriculumWrapper, TerminateOnCostWrapper, TerminateOnGoalWrapper


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect Safety-Gym scripted-geometric desired-heading labels.")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--env_name", type=str, default="SafetyCarGoal1-v0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--render_mode", type=str, default="none", choices=["human", "rgb_array", "none", "pygame", "topdown"])
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=1.0)
    p.add_argument("--car_force_scale", type=float, default=1.0)
    p.add_argument("--car_action_mode", type=str, default="throttle_turn", choices=["raw_wheels", "throttle_turn", "cardinal"])
    p.add_argument(
        "--obs_mask_mode",
        type=str,
        default="privileged_geometry_rich",
        choices=["none", "goal_only_lidar", "privileged_geometry", "privileged_geometry_rich"],
    )
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--layout_curriculum", type=str, default="car_random_blocked_filter")
    p.add_argument("--layout_curriculum_level", type=int, default=0)
    p.add_argument("--reward_mode", type=str, default="dense_plus_sparse")
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--success_reward_scale", type=float, default=5.0)
    p.add_argument("--step_penalty", type=float, default=-0.001)
    p.add_argument("--clearance_penalty_scale", type=float, default=4.0)
    p.add_argument("--clearance_margin", type=float, default=0.0)
    p.add_argument("--clearance_penalty_mode", type=str, default="softplus", choices=["hinge_power", "softplus"])
    p.add_argument("--clearance_penalty_temperature", type=float, default=0.001)
    p.add_argument("--terminate_on_goal", action="store_true", default=True)
    p.add_argument("--terminate_on_cost", action="store_true", default=False)
    p.add_argument("--rollout_policy", type=str, default="teacher", choices=["teacher", "heading_bc"])
    p.add_argument("--student_model_path", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_episodes", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=50000)
    return p


def _signed_angle(src: np.ndarray, dst: np.ndarray) -> float:
    src = np.asarray(src, dtype=np.float64).reshape(2)
    dst = np.asarray(dst, dtype=np.float64).reshape(2)
    src = src / max(1e-9, float(np.linalg.norm(src)))
    dst = dst / max(1e-9, float(np.linalg.norm(dst)))
    cross = float(src[0] * dst[1] - src[1] * dst[0])
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    return float(np.arctan2(cross, dot))


def _make_env(args: argparse.Namespace):
    env = make_safety_env(
        args.env_name,
        render_mode=resolve_env_render_mode(args.render_mode),
        max_episode_steps=int(args.max_episode_steps),
        surface_mode=str(args.surface_mode),
        car_wheel_command_limit=float(args.car_wheel_command_limit),
        car_force_scale=float(args.car_force_scale),
        car_action_mode=str(args.car_action_mode),
        obs_mask_mode=str(args.obs_mask_mode),
        seed=int(args.seed),
    )
    if str(args.layout_curriculum).strip().lower() != "none":
        env = SafetyLayoutCurriculumWrapper(env, curriculum=str(args.layout_curriculum), level=int(args.layout_curriculum_level))
    env = RewardModeWrapper(
        env,
        reward_mode=str(args.reward_mode),
        dense_reward_scale=float(args.dense_reward_scale),
        success_reward_scale=float(args.success_reward_scale),
        step_penalty=float(args.step_penalty),
        clearance_penalty_scale=float(args.clearance_penalty_scale),
        clearance_margin=float(args.clearance_margin),
        clearance_penalty_mode=str(args.clearance_penalty_mode),
        clearance_penalty_temperature=float(args.clearance_penalty_temperature),
    )
    if bool(args.terminate_on_cost):
        env = TerminateOnCostWrapper(env)
    if bool(args.terminate_on_goal):
        env = TerminateOnGoalWrapper(env)
    return env


def main() -> int:
    args = build_parser().parse_args()
    if args.rollout_policy == "heading_bc" and not str(args.student_model_path).strip():
        raise ValueError("--student_model_path is required for --rollout_policy heading_bc")
    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)
    env = _make_env(args)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    teacher = ScriptedGeometricTeacherController(action_low=action_low, action_high=action_high)
    student = None
    if args.rollout_policy == "heading_bc":
        student = load_heading_policy(Path(args.student_model_path).expanduser(), device=device)

    obs_rows: list[np.ndarray] = []
    angle_rows: list[float] = []
    action_rows: list[np.ndarray] = []
    episode_id_rows: list[int] = []
    episode_step_rows: list[int] = []

    obs, _ = env.reset(seed=int(args.seed))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    episodes = 0
    episode_steps = 0
    steps = 0
    try:
        while episodes < int(args.num_episodes) and steps < int(args.max_steps):
            forward = extract_agent_forward_xy(env)
            desired = teacher._desired_direction(env)  # Intentionally label the teacher's planner target.
            action = teacher.get_action(obs=obs, env=env)
            if forward is None or desired is None or action is None:
                obs, _ = env.reset(seed=int(args.seed) + episodes + 1)
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                episodes += 1
                episode_steps = 0
                continue
            angle = _signed_angle(np.asarray(forward, dtype=np.float64), np.asarray(desired, dtype=np.float64))
            obs_rows.append(obs.copy())
            angle_rows.append(float(angle))
            action_rows.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())
            episode_id_rows.append(int(episodes))
            episode_step_rows.append(int(episode_steps))

            if student is not None:
                step_action = student.act(obs, env=env, device=device)
            else:
                step_action = action
            next_obs, _reward, _cost, terminated, truncated, _info = env.step(step_action)
            steps += 1
            episode_steps += 1
            if terminated or truncated:
                episodes += 1
                print(f"episode={episodes} rows={len(obs_rows)}", flush=True)
                episode_steps = 0
                obs, _ = env.reset(seed=int(args.seed) + episodes)
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            else:
                obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
    finally:
        env.close()

    target = Path(args.dataset_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "safetygym_heading_labels",
        "env_name": str(args.env_name),
        "label_teacher": "scripted_geo_desired_heading",
        "obs_mask_mode": str(args.obs_mask_mode),
        "car_action_mode": str(args.car_action_mode),
        "layout_curriculum": str(args.layout_curriculum),
        "layout_curriculum_level": int(args.layout_curriculum_level),
        "num_episodes_collected": int(episodes),
        "num_steps_collected": int(len(obs_rows)),
        "rollout_policy": str(args.rollout_policy),
        "student_model_path": str(args.student_model_path),
    }
    angles = np.asarray(angle_rows, dtype=np.float32)
    np.savez_compressed(
        target,
        observations=np.asarray(obs_rows, dtype=np.float32),
        target_angles=angles,
        target_sincos=np.stack([np.sin(angles), np.cos(angles)], axis=1).astype(np.float32),
        teacher_actions=np.asarray(action_rows, dtype=np.float32),
        episode_ids=np.asarray(episode_id_rows, dtype=np.int64),
        episode_steps=np.asarray(episode_step_rows, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )
    print({"dataset_path": str(target), "episodes": int(episodes), "rows": int(len(obs_rows))}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
