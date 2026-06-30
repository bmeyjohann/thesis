#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.controllers import LearnedInterventionPolicyController, build_human_controller
from safetygym_utils.dataset_io import save_transition_dataset
from safetygym_utils.env import clip_action_to_space, make_safety_env, resolve_control_scheme, scale_action_np
from safetygym_utils.io import load_args_json, maybe_find_args_json_from_model
from safetygym_utils.rendering import resolve_env_render_mode
from safetygym_utils.sac import SafetyActor
from safetygym_utils.wrappers import (
    FixedSafetyLayoutWrapper,
    FrameStackObservationWrapper,
    HumanInterventionWrapper,
    RewardModeWrapper,
    SafetyLayoutCurriculumWrapper,
    TerminateOnGoalWrapper,
)

_FAST_SAC_PATH = _ROOT / "fasttd3" / "fast_sac"
if _FAST_SAC_PATH.exists() and str(_FAST_SAC_PATH) not in sys.path:
    sys.path.insert(0, str(_FAST_SAC_PATH))

try:
    from fast_sac_utils import EmpiricalNormalization  # type: ignore
except Exception:
    EmpiricalNormalization = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect DAgger rows with scripted gate/action labels on learned-teacher visited states.")
    p.add_argument("--base_model_path", type=str, required=True, help="Goal-reaching student policy checkpoint.")
    p.add_argument("--base_policy", type=str, default="fastsac", choices=["fastsac", "ppo"])
    p.add_argument("--learned_teacher_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--env_name", type=str, default="SafetyCarGoal1-v0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--render_mode", type=str, default="none", choices=["human", "rgb_array", "none", "pygame", "topdown"])
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=1.0)
    p.add_argument("--car_force_scale", type=float, default=1.0)
    p.add_argument("--car_action_mode", type=str, default="raw_wheels", choices=["raw_wheels", "throttle_turn", "cardinal"])
    p.add_argument("--obs_mask_mode", type=str, default="privileged_geometry", choices=["none", "goal_only_lidar", "privileged_geometry", "privileged_geometry_rich"])
    p.add_argument("--actor_hidden_dim", type=int, default=256)
    p.add_argument("--use_layer_norm", action="store_true", default=True)
    p.add_argument("--layer_norm_eps", type=float, default=1e-5)
    p.add_argument("--temporal_encoder", type=str, default="none", choices=["none", "attention"])
    p.add_argument("--init_scale", type=float, default=0.01)
    p.add_argument("--scale_actor_to_env_bounds", action="store_true", default=True)
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--fixed_layout_preset", type=str, default="none")
    p.add_argument("--layout_curriculum", type=str, default="car_random_blocked_filter")
    p.add_argument("--layout_curriculum_level", type=int, default=0)
    p.add_argument("--obs_frame_stack", type=int, default=1)
    p.add_argument("--reward_mode", type=str, default="dense")
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--success_reward_scale", type=float, default=0.0)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--clearance_penalty_scale", type=float, default=0.0)
    p.add_argument("--clearance_margin", type=float, default=0.0)
    p.add_argument("--clearance_penalty_mode", type=str, default="softplus", choices=["hinge_power", "softplus"])
    p.add_argument("--clearance_penalty_temperature", type=float, default=0.001)
    p.add_argument("--terminate_on_goal", action="store_true", default=True)
    p.add_argument("--num_episodes", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--learned_intervention_threshold", type=float, default=0.5)
    p.add_argument("--teacher_override_clearance_threshold", type=float, default=0.08)
    p.add_argument("--teacher_override_clearance_exit_threshold", type=float, default=0.12)
    p.add_argument("--teacher_override_mode", type=str, default="clearance_or_progress")
    p.add_argument("--teacher_progress_bad_steps", type=int, default=3)
    p.add_argument("--teacher_progress_good_steps", type=int, default=5)
    p.add_argument("--teacher_progress_epsilon", type=float, default=1e-4)
    p.add_argument("--teacher_progress_trigger_mode", type=str, default="worse")
    p.add_argument("--teacher_progress_release_mode", type=str, default="improve")
    p.add_argument("--teacher_progress_score_mode", type=str, default="potential_field")
    p.add_argument("--teacher_progress_dense_scale", type=float, default=1.0)
    p.add_argument("--teacher_progress_clearance_scale", type=float, default=4.0)
    p.add_argument("--teacher_progress_clearance_margin", type=float, default=0.12)
    p.add_argument("--teacher_progress_clearance_mode", type=str, default="softplus")
    p.add_argument("--teacher_progress_clearance_temperature", type=float, default=0.001)
    p.add_argument("--action_disagreement_margin", type=float, default=0.25)
    return p


def _load_ckpt_defaults(args: argparse.Namespace) -> None:
    args_path = maybe_find_args_json_from_model(Path(args.base_model_path))
    if args_path is None or not args_path.exists():
        return
    cfg = load_args_json(args_path)
    for name in (
        "env_name",
        "surface_mode",
        "car_wheel_command_limit",
        "car_force_scale",
        "car_action_mode",
        "obs_mask_mode",
        "actor_hidden_dim",
        "use_layer_norm",
        "layer_norm_eps",
        "temporal_encoder",
        "init_scale",
        "scale_actor_to_env_bounds",
        "layout_curriculum",
        "layout_curriculum_level",
        "obs_frame_stack",
    ):
        if name in cfg:
            setattr(args, name, cfg[name])


def _make_env(args: argparse.Namespace):
    kwargs = {}
    if int(args.max_episode_steps) > 0:
        kwargs["max_episode_steps"] = int(args.max_episode_steps)
    env = make_safety_env(
        args.env_name,
        render_mode=resolve_env_render_mode(args.render_mode),
        surface_mode=str(args.surface_mode),
        car_wheel_command_limit=float(args.car_wheel_command_limit),
        car_force_scale=float(args.car_force_scale),
        car_action_mode=str(args.car_action_mode),
        obs_mask_mode=str(args.obs_mask_mode),
        seed=int(args.seed),
        **kwargs,
    )
    if str(args.fixed_layout_preset).strip().lower() != "none":
        env = FixedSafetyLayoutWrapper(env, preset=str(args.fixed_layout_preset))
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
    if bool(args.terminate_on_goal):
        env = TerminateOnGoalWrapper(env)
    if int(args.obs_frame_stack) > 1:
        env = FrameStackObservationWrapper(env, num_frames=int(args.obs_frame_stack))
    return env


def _vecnormalize_path_for_ppo_model(model_path: Path) -> Path | None:
    candidates: list[Path] = []
    stem = model_path.stem
    if stem.startswith("ppo_step_") and stem.endswith("_steps"):
        step_text = stem[len("ppo_step_") :]
        candidates.append(model_path.parent / f"ppo_step_vecnormalize_{step_text}.pkl")
    candidates.append(model_path.parent / "vecnormalize.pkl")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    args = build_parser().parse_args()
    if str(args.base_policy).strip().lower() == "fastsac":
        _load_ckpt_defaults(args)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    env = _make_env(args)
    obs, _ = env.reset(seed=int(args.seed))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    obs_dim = int(obs.shape[0])
    act_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    base_policy = str(args.base_policy).strip().lower()
    actor = None
    obs_preprocess = torch.nn.Identity()
    ppo_student = None
    ppo_vecnormalize = None
    if base_policy == "fastsac":
        actor = SafetyActor(
            n_obs=obs_dim,
            n_act=act_dim,
            num_envs=1,
            init_scale=float(args.init_scale),
            hidden_dim=int(args.actor_hidden_dim),
            use_layer_norm=bool(args.use_layer_norm),
            layer_norm_eps=float(args.layer_norm_eps),
            temporal_encoder=str(args.temporal_encoder),
            obs_frame_stack=int(args.obs_frame_stack),
            device=device,
        )
        checkpoint = torch.load(args.base_model_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()
        obs_norm_state = checkpoint.get("obs_normalizer_state_dict", None)
        if obs_norm_state and EmpiricalNormalization is not None:
            obs_preprocess = EmpiricalNormalization(shape=obs_dim, device=device)
            obs_preprocess.load_state_dict(obs_norm_state, strict=False)
            obs_preprocess.eval()
    elif base_policy == "ppo":
        model_path = Path(args.base_model_path).expanduser().resolve()
        ppo_student = PPO.load(str(model_path), device=device)
        vecnormalize_path = _vecnormalize_path_for_ppo_model(model_path)
        if vecnormalize_path is not None:
            tmp_env = DummyVecEnv([lambda: _make_env(copy.copy(args))])
            ppo_vecnormalize = VecNormalize.load(str(vecnormalize_path), tmp_env)
            ppo_vecnormalize.training = False
            ppo_vecnormalize.norm_reward = False
    else:
        raise ValueError(f"Unsupported base_policy: {args.base_policy}")

    learned_teacher = LearnedInterventionPolicyController(
        checkpoint_path=args.learned_teacher_path,
        action_low=action_low,
        action_high=action_high,
        intervention_threshold=float(args.learned_intervention_threshold),
        device=str(device),
    )
    scripted = build_human_controller(
        input_device="scripted_geo",
        action_dim=act_dim,
        obs_dim=obs_dim,
        env_name=str(args.env_name),
        action_scale=1.0,
        wheel_command_limit=float(args.car_wheel_command_limit),
        overlay_fps_limit=0,
        overlay_draw_hz=20.0,
        action_low=action_low,
        action_high=action_high,
        control_scheme_override=resolve_control_scheme(str(args.env_name), car_action_mode=str(args.car_action_mode)),
    )
    oracle = HumanInterventionWrapper(
        env,
        controller=scripted,
        threshold=0.0,
        hold_seconds=0.0,
        clearance_override_threshold=float(args.teacher_override_clearance_threshold),
        clearance_override_exit_threshold=float(args.teacher_override_clearance_exit_threshold),
        clearance_override_mode=str(args.teacher_override_mode),
        teacher_progress_bad_steps=int(args.teacher_progress_bad_steps),
        teacher_progress_good_steps=int(args.teacher_progress_good_steps),
        teacher_progress_epsilon=float(args.teacher_progress_epsilon),
        teacher_progress_trigger_mode=str(args.teacher_progress_trigger_mode),
        teacher_progress_release_mode=str(args.teacher_progress_release_mode),
        teacher_progress_score_mode=str(args.teacher_progress_score_mode),
        teacher_progress_dense_scale=float(args.teacher_progress_dense_scale),
        teacher_progress_clearance_scale=float(args.teacher_progress_clearance_scale),
        teacher_progress_clearance_margin=float(args.teacher_progress_clearance_margin),
        teacher_progress_clearance_mode=str(args.teacher_progress_clearance_mode),
        teacher_progress_clearance_temperature=float(args.teacher_progress_clearance_temperature),
    )
    obs, _ = oracle.reset(seed=int(args.seed))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    oracle._last_obs = obs.copy()

    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    next_obs_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    done_rows: list[bool] = []
    trunc_rows: list[bool] = []
    cost_rows: list[float] = []
    student_rows: list[np.ndarray] = []
    intervened_rows: list[bool] = []
    episode_id_rows: list[int] = []
    episode_step_rows: list[int] = []
    gate_disagreements = 0
    action_disagreements = 0
    learned_interventions = 0
    oracle_interventions = 0
    episodes = 0
    episode_steps = 0
    steps = 0
    try:
        while episodes < int(args.num_episodes) and steps < int(args.max_steps):
            if base_policy == "fastsac":
                assert actor is not None
                with torch.inference_mode():
                    obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
                    obs_t = obs_preprocess(obs_t)
                    _, _, mean = actor(obs_t)
                    base_action = mean[0].detach().cpu().numpy().astype(np.float32)
                if bool(args.scale_actor_to_env_bounds):
                    base_action = scale_action_np(base_action, env.action_space)
            else:
                assert ppo_student is not None
                ppo_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                expected_shape = getattr(ppo_student.observation_space, "shape", None)
                expected_dim = int(np.prod(expected_shape)) if expected_shape else int(ppo_obs.shape[-1])
                if ppo_obs.shape[-1] > expected_dim:
                    ppo_obs = ppo_obs[:, :expected_dim]
                if ppo_vecnormalize is not None:
                    ppo_obs = ppo_vecnormalize.normalize_obs(ppo_obs)
                base_action, _ = ppo_student.predict(ppo_obs, deterministic=True)
                base_action = np.asarray(base_action, dtype=np.float32).reshape(-1)
            base_action = clip_action_to_space(base_action, env.action_space)

            oracle_action_raw = oracle._current_teacher_action(student_action=base_action)
            oracle_gate = oracle_action_raw is not None
            if oracle_gate:
                oracle_action = clip_action_to_space(np.asarray(oracle_action_raw, dtype=np.float32), env.action_space)
            else:
                oracle_action = base_action.copy()

            learned_raw = learned_teacher.get_action(obs=obs, env=env, student_action=base_action)
            learned_gate = learned_raw is not None
            if learned_gate:
                applied_action = clip_action_to_space(np.asarray(learned_raw, dtype=np.float32), env.action_space)
            else:
                applied_action = base_action.copy()
            if learned_gate:
                learned_interventions += 1
            if oracle_gate:
                oracle_interventions += 1
            if bool(learned_gate) != bool(oracle_gate):
                gate_disagreements += 1
            if oracle_gate and float(np.linalg.norm(applied_action - oracle_action)) > float(args.action_disagreement_margin):
                action_disagreements += 1

            next_obs, reward, cost, terminated, truncated, _info = env.step(applied_action)
            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            obs_rows.append(obs.copy())
            action_rows.append(oracle_action.copy())
            next_obs_rows.append(next_obs.copy())
            reward_rows.append(float(reward))
            done_rows.append(bool(terminated or truncated))
            trunc_rows.append(bool(truncated))
            cost_rows.append(float(cost))
            student_rows.append(base_action.copy())
            intervened_rows.append(bool(oracle_gate))
            episode_id_rows.append(int(episodes))
            episode_step_rows.append(int(episode_steps))

            steps += 1
            episode_steps += 1
            if terminated or truncated:
                episodes += 1
                print(
                    json.dumps(
                        {
                            "episode": int(episodes),
                            "collected_steps": int(steps),
                            "gate_disagreement_rate": gate_disagreements / max(1, steps),
                            "oracle_intervention_rate": oracle_interventions / max(1, steps),
                            "learned_intervention_rate": learned_interventions / max(1, steps),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                episode_steps = 0
                learned_teacher.reset()
                obs, _ = oracle.reset(seed=int(args.seed) + episodes)
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                oracle._last_obs = obs.copy()
            else:
                obs = next_obs
                oracle._last_obs = obs.copy()
    finally:
        learned_teacher.close()
        scripted.close()
        env.close()

    metadata = {
        "env_name": str(args.env_name),
        "source_base_model_path": str(Path(args.base_model_path).expanduser()),
        "source_learned_teacher_path": str(Path(args.learned_teacher_path).expanduser()),
        "label_teacher": "scripted_geo_gate",
        "executed_policy": "base_policy_plus_learned_teacher",
        "obs_mask_mode": str(args.obs_mask_mode),
        "car_action_mode": str(args.car_action_mode),
        "num_episodes_collected": int(episodes),
        "num_steps_collected": int(len(obs_rows)),
        "gate_disagreement_rate": float(gate_disagreements / max(1, len(obs_rows))),
        "action_disagreement_rate_when_oracle_intervenes": float(action_disagreements / max(1, oracle_interventions)),
        "oracle_intervention_rate": float(oracle_interventions / max(1, len(obs_rows))),
        "learned_intervention_rate": float(learned_interventions / max(1, len(obs_rows))),
    }
    path = save_transition_dataset(
        path=args.dataset_path,
        metadata=metadata,
        observations=np.asarray(obs_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.float32),
        next_observations=np.asarray(next_obs_rows, dtype=np.float32),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.bool_),
        truncations=np.asarray(trunc_rows, dtype=np.bool_),
        costs=np.asarray(cost_rows, dtype=np.float32),
        student_actions=np.asarray(student_rows, dtype=np.float32),
        teacher_intervened=np.asarray(intervened_rows, dtype=np.bool_),
        episode_ids=np.asarray(episode_id_rows, dtype=np.int64),
        episode_steps=np.asarray(episode_step_rows, dtype=np.int64),
    )
    print(json.dumps({"dataset_path": str(path), "episodes": int(episodes), "rows": int(len(obs_rows)), **metadata}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
