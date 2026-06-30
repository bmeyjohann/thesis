#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from safetygym_utils.io import save_args_json
from safetygym_utils.minimal_train import _make_env_with_wrappers
from safetygym_utils.metrics import EpisodeWindow, augment_rollout_summary, classify_outcome
from safetygym_utils.policy_viz import (
    _extract_bounds,
    _extract_overlay_specs,
    _body_xy_and_yaw,
    plot_episode_contact_sheet,
    plot_eval_episode_trajectory,
)
from safetygym_utils.wrappers import fixed_layout_preset_names, layout_curriculum_names


class SafetyGymnasiumToGymnasium(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.setdefault("cost", float(cost))
        return obs, float(reward), bool(terminated), bool(truncated), info


class WandbMetricCallback(BaseCallback):
    def __init__(self, *, wandb_run, log_freq: int = 2048):
        super().__init__()
        self.wandb_run = wandb_run
        self.log_freq = int(max(1, log_freq))
        self._ep_cost: np.ndarray | None = None
        self._ep_len: np.ndarray | None = None
        self._ep_goal_hits: np.ndarray | None = None
        self._ep_min_clearance: np.ndarray | None = None
        self._completed: list[dict[str, float]] = []

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self._ep_cost = np.zeros(n_envs, dtype=np.float64)
        self._ep_len = np.zeros(n_envs, dtype=np.float64)
        self._ep_goal_hits = np.zeros(n_envs, dtype=np.float64)
        self._ep_min_clearance = np.full(n_envs, np.inf, dtype=np.float64)

    def _record_step_infos(self) -> None:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if self._ep_cost is None or self._ep_len is None or self._ep_goal_hits is None or self._ep_min_clearance is None:
            return
        if dones is None:
            dones = [False] * len(infos)
        for idx, info in enumerate(infos):
            if idx >= len(self._ep_cost):
                continue
            info = dict(info or {})
            self._ep_len[idx] += 1.0
            self._ep_cost[idx] += float(info.get("cost", 0.0) or 0.0)
            if bool(info.get("goal_met", False)):
                self._ep_goal_hits[idx] += 1.0
            clearance = info.get("min_constrained_clearance")
            if clearance is not None and np.isfinite(float(clearance)):
                self._ep_min_clearance[idx] = min(float(self._ep_min_clearance[idx]), float(clearance))
            if bool(dones[idx]):
                ep_len = max(1.0, float(self._ep_len[idx]))
                goal_hits = float(self._ep_goal_hits[idx])
                min_clearance = float(self._ep_min_clearance[idx])
                self._completed.append(
                    {
                        "episode_cost_sum": float(self._ep_cost[idx]),
                        "episode_cost_rate": float(self._ep_cost[idx] / ep_len),
                        "episode_length": ep_len,
                        "goal_success": 1.0 if goal_hits > 0 else 0.0,
                        "goals_per_episode": goal_hits,
                        "first_goal_success": 1.0 if goal_hits > 0 else 0.0,
                        "min_constrained_clearance": min_clearance if np.isfinite(min_clearance) else 0.0,
                    }
                )
                if len(self._completed) > 1000:
                    del self._completed[: len(self._completed) - 1000]
                self._ep_cost[idx] = 0.0
                self._ep_len[idx] = 0.0
                self._ep_goal_hits[idx] = 0.0
                self._ep_min_clearance[idx] = np.inf

    def _on_step(self) -> bool:
        self._record_step_infos()
        if self.wandb_run is None or (self.num_timesteps % self.log_freq) != 0:
            return True
        metrics: dict[str, float] = {"train/step": float(self.num_timesteps)}
        for key, value in self.model.logger.name_to_value.items():
            if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                metrics[f"sb3/{key}"] = float(value)
        if self._completed:
            recent = self._completed[-100:]
            keys = sorted(recent[0].keys())
            for key in keys:
                vals = [float(item[key]) for item in recent if key in item and np.isfinite(float(item[key]))]
                if vals:
                    metrics[f"evalish/{key}_mean100"] = float(np.mean(vals))
            metrics["evalish/episodes_logged"] = float(len(self._completed))
        self.wandb_run.log(metrics, step=int(self.num_timesteps))
        return True


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        args,
        log_dir: Path,
        wandb_run,
        eval_freq: int,
        num_episodes: int,
        save_plots: bool,
        plot_max_episodes: int,
        reward_surface: bool,
    ):
        super().__init__()
        self.args = args
        self.log_dir = Path(log_dir)
        self.wandb_run = wandb_run
        self.eval_freq = int(max(0, eval_freq))
        self.num_episodes = int(max(1, num_episodes))
        self.save_plots = bool(save_plots)
        self.plot_max_episodes = int(max(0, plot_max_episodes))
        self.reward_surface = bool(reward_surface)
        self._next_eval = self.eval_freq if self.eval_freq > 0 else -1

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps < self._next_eval:
            return True
        while self._next_eval > 0 and self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
        self._run_eval(step=int(self.num_timesteps))
        return True

    def _obs_for_policy(self, obs: np.ndarray) -> np.ndarray:
        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm is None:
            return np.asarray(obs, dtype=np.float32)
        normed = vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))
        return np.asarray(normed, dtype=np.float32).reshape(-1)

    def _eval_args(self):
        eval_args = copy.copy(self.args)
        fixed = str(getattr(self.args, "eval_fixed_layout_preset", "train")).strip().lower()
        if fixed != "train":
            eval_args.fixed_layout_preset = fixed
        curriculum = str(getattr(self.args, "eval_layout_curriculum", "train")).strip().lower()
        if curriculum != "train":
            eval_args.layout_curriculum = curriculum
        level = int(getattr(self.args, "eval_layout_curriculum_level", -1))
        if level >= 0:
            eval_args.layout_curriculum_level = level
        eval_args.render_mode = "none"
        return eval_args

    def _snapshot_layout(self, *, env, task, overlay_specs: list[dict[str, Any]], reset_info: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, float]]]:
        from safetygym_utils.env import extract_agent_xy, extract_goal_xy

        overlay_poses: dict[str, tuple[np.ndarray, float]] = {}
        obstacles: list[dict[str, Any]] = []
        for spec in overlay_specs:
            pose = _body_xy_and_yaw(task, str(spec["name"]))
            if pose is None:
                continue
            pos_xy, yaw = pose
            overlay_poses[str(spec["name"])] = (np.asarray(pos_xy, dtype=np.float64).reshape(2).copy(), float(yaw))
            size = np.asarray(spec.get("size", [0.1]), dtype=np.float64).reshape(-1)
            obstacles.append(
                {
                    "name": str(spec["name"]),
                    "geom_type": str(spec.get("geom_type", "sphere")),
                    "xy": [float(pos_xy[0]), float(pos_xy[1])],
                    "yaw": float(yaw),
                    "size": [float(x) for x in size.tolist()],
                }
            )
        agent_xy = extract_agent_xy(env)
        goal_xy = extract_goal_xy(env)
        layout = {
            "layout_curriculum": str(reset_info.get("layout_curriculum", getattr(self._eval_args(), "layout_curriculum", "none"))),
            "layout_curriculum_level": int(reset_info.get("layout_curriculum_level", getattr(self._eval_args(), "layout_curriculum_level", -1))),
            "fixed_layout_preset": str(reset_info.get("fixed_layout_preset", getattr(self._eval_args(), "fixed_layout_preset", "none"))),
            "agent_xy": [float(x) for x in np.asarray(agent_xy if agent_xy is not None else [float("nan"), float("nan")], dtype=np.float64).reshape(2).tolist()],
            "goal_xy": [float(x) for x in np.asarray(goal_xy if goal_xy is not None else [float("nan"), float("nan")], dtype=np.float64).reshape(2).tolist()],
            "obstacles": obstacles,
        }
        layout.update(self._layout_challenge_diagnostics(layout))
        return layout, overlay_poses

    @staticmethod
    def _layout_challenge_diagnostics(layout: dict[str, Any]) -> dict[str, float]:
        agent = np.asarray(layout.get("agent_xy", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        goal = np.asarray(layout.get("goal_xy", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        seg = goal - agent
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6 or not np.isfinite(seg_len):
            return {
                "challenge_start_goal_distance": float(seg_len),
                "challenge_min_line_clearance": float("nan"),
                "challenge_obstacles_near_corridor": 0.0,
                "challenge_direct_path_blocked": 0.0,
            }
        unit = seg / seg_len
        min_clearance = float("inf")
        near_count = 0
        for obstacle in layout.get("obstacles", []):
            name = str(obstacle.get("name", "")).lower()
            if not any(key in name for key in ("hazard", "vase", "pillar", "gremlin", "wall")):
                continue
            pos = np.asarray(obstacle.get("xy", [np.nan, np.nan]), dtype=np.float64).reshape(2)
            if not np.isfinite(pos).all():
                continue
            rel = pos - agent
            t = float(np.clip(np.dot(rel, unit) / seg_len, 0.0, 1.0))
            closest = agent + t * seg
            size = np.asarray(obstacle.get("size", [0.1]), dtype=np.float64).reshape(-1)
            geom = str(obstacle.get("geom_type", "sphere")).lower()
            if geom == "box":
                radius = float(np.linalg.norm(size[:2])) if size.size >= 2 else float(size[0])
            else:
                radius = float(size[0]) if size.size >= 1 else 0.1
            clearance = float(np.linalg.norm(pos - closest) - radius)
            min_clearance = min(min_clearance, clearance)
            if clearance < 0.25 and 0.05 < t < 0.95:
                near_count += 1
        if not np.isfinite(min_clearance):
            min_clearance = float("nan")
        return {
            "challenge_start_goal_distance": float(seg_len),
            "challenge_min_line_clearance": float(min_clearance),
            "challenge_obstacles_near_corridor": float(near_count),
            "challenge_direct_path_blocked": 1.0 if np.isfinite(min_clearance) and min_clearance < 0.05 else 0.0,
        }

    def _run_eval(self, *, step: int) -> None:
        eval_args = self._eval_args()
        eval_dir = self.log_dir / "periodic_eval" / f"step_{step}"
        plot_dir = eval_dir / "episode_plots"
        eval_dir.mkdir(parents=True, exist_ok=True)
        win = EpisodeWindow(size=self.num_episodes)
        saved_plots: list[Path] = []
        episode_records: list[dict[str, Any]] = []
        for ep_idx in range(self.num_episodes):
            env = _make_env_with_wrappers(
                args=eval_args,
                seed=int(getattr(self.args, "seed", 1)) + 100_000 + step + ep_idx,
                with_intervention=False,
                controller=None,
                render_mode_override="none",
            )
            obs, reset_info = env.reset()
            reset_info = dict(reset_info or {})
            task = env.unwrapped.task
            overlay_specs = _extract_overlay_specs(task)
            bounds = _extract_bounds(task, x_range=None, y_range=None)
            layout_snapshot, overlay_poses = self._snapshot_layout(env=env, task=task, overlay_specs=overlay_specs, reset_info=reset_info)
            initial_goal_xy = np.asarray(layout_snapshot["goal_xy"], dtype=np.float64).reshape(2)
            ep_ret = 0.0
            ep_cost = 0.0
            ep_len = 0
            ep_goal_hits = 0
            ep_first_goal_hit_step = None
            ep_path = []
            ep_goal_positions = []
            ep_goal_hit_points = []
            ep_min_clearance = float("inf")
            terminated = truncated = False
            while not (terminated or truncated):
                try:
                    from safetygym_utils.env import extract_agent_xy, extract_goal_distance, extract_goal_xy

                    agent_xy = extract_agent_xy(env)
                    goal_xy = extract_goal_xy(env)
                    if agent_xy is not None:
                        ep_path.append(np.asarray(agent_xy, dtype=np.float64).reshape(2))
                    if goal_xy is not None and (not ep_goal_positions or np.linalg.norm(np.asarray(ep_goal_positions[-1]) - goal_xy) > 1e-6):
                        ep_goal_positions.append(np.asarray(goal_xy, dtype=np.float64).reshape(2))
                    final_dist = extract_goal_distance(env)
                except Exception:
                    final_dist = float("nan")
                action, _ = self.model.predict(self._obs_for_policy(obs), deterministic=True)
                obs, reward, cost, terminated, truncated, info = env.step(action)
                info = dict(info or {})
                ep_ret += float(reward)
                ep_cost += float(cost)
                ep_len += 1
                clearance = info.get("min_constrained_clearance")
                if clearance is not None and np.isfinite(float(clearance)):
                    ep_min_clearance = min(ep_min_clearance, float(clearance))
                if bool(info.get("goal_met", False)):
                    ep_goal_hits += 1
                    if ep_first_goal_hit_step is None:
                        ep_first_goal_hit_step = ep_len
                    if ep_path:
                        ep_goal_hit_points.append(np.asarray(ep_path[-1], dtype=np.float64).reshape(2))
                if ep_len >= int(getattr(eval_args, "max_episode_steps", 0) or 1000):
                    break
            max_steps = int(getattr(eval_args, "max_episode_steps", 0) or 1000)
            first_hit = ep_first_goal_hit_step if ep_first_goal_hit_step is not None else max_steps
            ep = {
                "episode_return": float(ep_ret),
                "episode_cost_sum": float(ep_cost),
                "episode_cost_rate": float(ep_cost) / max(1, int(ep_len)),
                "episode_length": float(ep_len),
                "intervention_steps": 0.0,
                "intervention_fraction": 0.0,
                "intervention_num_bursts": 0.0,
                "intervention_avg_burst_len": 0.0,
                "goal_met": 1.0 if ep_goal_hits > 0 else 0.0,
                "goal_met_count": float(ep_goal_hits),
                "first_goal_success": 1.0 if ep_goal_hits > 0 else 0.0,
                "first_goal_hit_step": float(first_hit),
                "first_goal_hit_step_success_only": float(ep_first_goal_hit_step or 0),
                "first_goal_within_100": 1.0 if ep_first_goal_hit_step is not None and ep_first_goal_hit_step <= 100 else 0.0,
                "first_goal_within_200": 1.0 if ep_first_goal_hit_step is not None and ep_first_goal_hit_step <= 200 else 0.0,
                "first_goal_reward_sum": float(ep_ret),
                "first_goal_dense_reward_sum": float(ep_ret),
                "final_distance_to_goal": float(final_dist),
                "outcome_success": 1.0 if ep_goal_hits > 0 else 0.0,
                "outcome_timeout": 1.0 if classify_outcome(goal_met=ep_goal_hits > 0, episode_steps=ep_len, max_episode_steps=max_steps) == "timeout" else 0.0,
                "outcome_kill": 1.0 if classify_outcome(goal_met=ep_goal_hits > 0, episode_steps=ep_len, max_episode_steps=max_steps) == "kill" else 0.0,
                "outcome_other_failure": 0.0,
                "terminated": 1.0 if terminated else 0.0,
                "truncated": 1.0 if truncated else 0.0,
            }
            for key in (
                "challenge_start_goal_distance",
                "challenge_min_line_clearance",
                "challenge_obstacles_near_corridor",
                "challenge_direct_path_blocked",
            ):
                ep[key] = float(layout_snapshot.get(key, 0.0))
            win.add(ep)
            episode_records.append({"episode": ep_idx + 1, "metrics": ep, "layout": layout_snapshot})
            save_args_json(plot_dir / f"episode_{ep_idx + 1:03d}_layout.json", layout_snapshot)
            if self.save_plots and len(saved_plots) < self.plot_max_episodes:
                reward_surface_config = None
                if self.reward_surface:
                    reward_surface_config = {
                        "resolution": 100,
                        "dense_reward_scale": float(getattr(eval_args, "dense_reward_scale", 1.0)),
                        "clearance_penalty_scale": float(getattr(eval_args, "clearance_penalty_scale", 0.0)),
                        "clearance_margin": float(getattr(eval_args, "clearance_margin", 0.0)),
                        "clearance_penalty_power": float(getattr(eval_args, "clearance_penalty_power", 1.0)),
                        "clearance_penalty_mode": str(getattr(eval_args, "clearance_penalty_mode", "softplus")),
                        "clearance_penalty_temperature": float(getattr(eval_args, "clearance_penalty_temperature", 0.001)),
                    }
                plot_path = plot_eval_episode_trajectory(
                    output_path=plot_dir / f"episode_{ep_idx + 1:03d}.png",
                    task=task,
                    overlay_specs=overlay_specs,
                    bounds=bounds,
                    path=np.asarray(ep_path, dtype=np.float64),
                    goal_positions=np.asarray(ep_goal_positions, dtype=np.float64),
                    goal_hit_points=np.asarray(ep_goal_hit_points, dtype=np.float64) if ep_goal_hit_points else np.zeros((0, 2)),
                    episode_idx=ep_idx + 1,
                    total_episodes=self.num_episodes,
                    episode_reward=float(ep_ret),
                    goals_reached=int(ep_goal_hits),
                    final_distance=float(final_dist),
                    reward_surface_config=reward_surface_config,
                    initial_goal_xy=initial_goal_xy,
                    overlay_poses=overlay_poses,
                )
                saved_plots.append(plot_path)
            env.close()

        summary = augment_rollout_summary(win.summary(prefix="periodic_eval"), prefix="periodic_eval")
        metrics = {k: float(v) for k, v in summary.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v))}
        metrics["periodic_eval/step"] = float(step)
        metrics["periodic_eval/layout_curriculum_level"] = float(getattr(eval_args, "layout_curriculum_level", -1))
        save_args_json(eval_dir / "metrics.json", metrics)
        save_args_json(eval_dir / "episodes.json", {"episodes": episode_records})
        sheet = None
        if saved_plots:
            sheet = plot_episode_contact_sheet(
                image_paths=saved_plots,
                output_path=eval_dir / "episode_contact_sheet.png",
                title=f"{getattr(self.args, 'exp_name', '')} step {step}",
            )
        print(f"[PeriodicEval] step={step} metrics={metrics} plots={sheet}", flush=True)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
            if sheet is not None:
                try:
                    import wandb  # type: ignore

                    self.wandb_run.log({"periodic_eval/trajectory_sheet": wandb.Image(str(sheet))}, step=step)
                except Exception:
                    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal PPO training for Safety-Gymnasium with thesis reward wrappers.")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--vec_env", type=str, default="subproc", choices=["dummy", "subproc"])
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--net_arch", type=str, default="256,256")
    p.add_argument("--activation_fn", type=str, default="tanh", choices=["tanh", "relu", "elu"])
    p.add_argument(
        "--safe_rl_checkpoint_path",
        type=str,
        default="",
        help="Optional Safe-RL ActorCritic PPO checkpoint to map into the SB3 PPO policy before training.",
    )
    p.add_argument(
        "--init_ppo_model_path",
        type=str,
        default="",
        help="Optional SB3 PPO .zip checkpoint to continue from with the current environment/wrapper settings.",
    )
    p.add_argument("--reset_ppo_optimizer", action="store_true", default=False)
    p.add_argument("--safe_rl_load_critic", action="store_true", default=True)
    p.add_argument("--no_safe_rl_load_critic", dest="safe_rl_load_critic", action="store_false")
    p.add_argument(
        "--safe_rl_actor_std_override",
        type=float,
        default=0.0,
        help="If >0, replace the Safe-RL checkpoint actor std with this SB3 Gaussian std.",
    )
    p.add_argument("--normalize_obs", action="store_true", default=True)
    p.add_argument("--no_normalize_obs", dest="normalize_obs", action="store_false")
    p.add_argument("--normalize_reward", action="store_true", default=False)
    p.add_argument("--render_mode", type=str, default="none", choices=["human", "none"])
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=1.0)
    p.add_argument("--car_force_scale", type=float, default=1.0)
    p.add_argument("--car_action_mode", type=str, default="raw_wheels", choices=["raw_wheels", "throttle_turn", "cardinal"])
    p.add_argument("--point_action_mode", type=str, default="native", choices=["native", "world_velocity"])
    p.add_argument("--point_turn_gain", type=float, default=2.5)
    p.add_argument("--point_alignment_power", type=float, default=1.0)
    p.add_argument("--point_allow_backward", action="store_true", default=False)
    p.add_argument(
        "--obs_mask_mode",
        type=str,
        default="none",
        choices=["none", "goal_only_lidar", "privileged_geometry", "privileged_geometry_rich"],
    )
    p.add_argument(
        "--fixed_layout_preset",
        type=str,
        default="none",
        choices=["none", *fixed_layout_preset_names()],
    )
    p.add_argument("--layout_curriculum", type=str, default="none", choices=["none", *layout_curriculum_names()])
    p.add_argument("--layout_curriculum_level", type=int, default=0)
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--terminate_on_goal", action="store_true", default=False)
    p.add_argument("--terminate_on_cost", action="store_true", default=False)
    p.add_argument("--reseed_on_episode_reset", action="store_true", default=False)
    p.add_argument("--no_reseed_on_episode_reset", dest="reseed_on_episode_reset", action="store_false")
    p.add_argument("--reward_mode", type=str, default="dense_plus_sparse", choices=["sparse", "dense", "dense_plus_sparse", "potential_diff", "native", "none"])
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--success_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=-0.001)
    p.add_argument("--cost_penalty", type=float, default=0.0)
    p.add_argument("--cost_penalty_warmup_steps", type=int, default=0)
    p.add_argument("--cost_penalty_ramp_steps", type=int, default=0)
    p.add_argument("--clearance_penalty_scale", type=float, default=1.1)
    p.add_argument("--clearance_margin", type=float, default=0.0)
    p.add_argument("--clearance_penalty_power", type=float, default=1.0)
    p.add_argument("--clearance_penalty_mode", type=str, default="softplus", choices=["hinge_power", "softplus"])
    p.add_argument("--clearance_penalty_temperature", type=float, default=0.001)
    p.add_argument("--clearance_penalty_warmup_steps", type=int, default=0)
    p.add_argument("--clearance_penalty_ramp_steps", type=int, default=0)
    p.add_argument("--forward_reward_scale", type=float, default=0.0)
    p.add_argument("--backward_penalty_scale", type=float, default=0.0)
    p.add_argument("--heading_reward_scale", type=float, default=0.0)
    p.add_argument("--heading_positive_only", action="store_true", default=True)
    p.add_argument("--no_heading_positive_only", dest="heading_positive_only", action="store_false")
    p.add_argument("--adaptive_safety_curriculum", action="store_true", default=False)
    p.add_argument("--adaptive_safety_goal_target", type=float, default=1.0)
    p.add_argument("--adaptive_safety_window_episodes", type=int, default=10)
    p.add_argument("--adaptive_safety_step", type=float, default=0.05)
    p.add_argument("--adaptive_safety_init", type=float, default=0.0)
    p.add_argument("--adaptive_safety_min", type=float, default=0.0)
    p.add_argument("--adaptive_safety_max", type=float, default=1.0)
    p.add_argument("--save_interval", type=int, default=20_000)
    p.add_argument("--log_interval", type=int, default=2048)
    p.add_argument("--eval_interval", type=int, default=0)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_fixed_layout_preset", type=str, default="train", choices=["train", "none", *fixed_layout_preset_names()])
    p.add_argument("--eval_layout_curriculum", type=str, default="train", choices=["train", "none", *layout_curriculum_names()])
    p.add_argument("--eval_layout_curriculum_level", type=int, default=-1)
    p.add_argument("--eval_save_plots", action="store_true", default=False)
    p.add_argument("--eval_plot_max_episodes", type=int, default=6)
    p.add_argument("--eval_reward_surface", action="store_true", default=False)
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="thesis-safetygym")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    return p


def _prepare_run_dirs(args) -> tuple[Path, Path]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not args.exp_name:
        args.exp_name = f"ppo_{args.env_name.replace('-', '_')}_{stamp}"
    log_dir = Path("logs") / "safetygym_ppo" / args.exp_name
    model_dir = Path("models") / "safetygym_ppo" / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_args_json(log_dir / "args.json", vars(args))
    save_args_json(model_dir / "args.json", vars(args))
    return log_dir, model_dir


def _make_single_env(args, seed: int):
    env = _make_env_with_wrappers(args=args, seed=seed, with_intervention=False, controller=None)
    env = SafetyGymnasiumToGymnasium(env)
    return Monitor(env)


def _vecnormalize_path_for_model(model_path: Path) -> Path | None:
    candidates: list[Path] = []
    if model_path.name.startswith("ppo_step_") and model_path.name.endswith("_steps.zip"):
        step = model_path.name.removeprefix("ppo_step_").removesuffix("_steps.zip")
        candidates.append(model_path.parent / f"ppo_step_vecnormalize_{step}_steps.pkl")
    candidates.append(model_path.parent / "vecnormalize.pkl")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _activation_fn(name: str):
    import torch.nn as nn

    return {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[str(name).lower()]


def _copy_tensor(dst, src, *, name: str) -> None:
    import torch

    src_t = torch.as_tensor(src, device=dst.device, dtype=dst.dtype)
    if tuple(dst.shape) != tuple(src_t.shape):
        raise ValueError(f"{name}: shape mismatch, target={tuple(dst.shape)} source={tuple(src_t.shape)}")
    dst.data.copy_(src_t)


def _load_safe_rl_ppo_checkpoint(model: PPO, path: str, *, load_critic: bool, actor_std_override: float) -> None:
    import torch

    ckpt_path = Path(path).expanduser()
    checkpoint = torch.load(ckpt_path, map_location=model.device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    policy = model.policy

    _copy_tensor(policy.mlp_extractor.policy_net[0].weight, state["actor.network.0.weight"], name="actor fc0 weight")
    _copy_tensor(policy.mlp_extractor.policy_net[0].bias, state["actor.network.0.bias"], name="actor fc0 bias")
    _copy_tensor(policy.mlp_extractor.policy_net[2].weight, state["actor.network.2.weight"], name="actor fc1 weight")
    _copy_tensor(policy.mlp_extractor.policy_net[2].bias, state["actor.network.2.bias"], name="actor fc1 bias")
    _copy_tensor(policy.mlp_extractor.policy_net[4].weight, state["actor.network.4.weight"], name="actor fc2 weight")
    _copy_tensor(policy.mlp_extractor.policy_net[4].bias, state["actor.network.4.bias"], name="actor fc2 bias")
    _copy_tensor(policy.action_net.weight, state["actor.network.6.weight"], name="actor head weight")
    _copy_tensor(policy.action_net.bias, state["actor.network.6.bias"], name="actor head bias")

    if float(actor_std_override) > 0.0:
        actor_std = torch.full_like(policy.log_std, float(actor_std_override))
    else:
        actor_std = torch.as_tensor(state["actor.std"], device=policy.log_std.device, dtype=policy.log_std.dtype)
    _copy_tensor(policy.log_std, torch.log(actor_std.clamp_min(1e-8)), name="actor log_std")

    critic_prefix = "critic.network"
    if f"{critic_prefix}.0.weight" not in state and "critics.0.network.0.weight" in state:
        critic_prefix = "critics.0.network"
    if load_critic and f"{critic_prefix}.0.weight" in state:
        _copy_tensor(policy.mlp_extractor.value_net[0].weight, state[f"{critic_prefix}.0.weight"], name="critic fc0 weight")
        _copy_tensor(policy.mlp_extractor.value_net[0].bias, state[f"{critic_prefix}.0.bias"], name="critic fc0 bias")
        _copy_tensor(policy.mlp_extractor.value_net[2].weight, state[f"{critic_prefix}.2.weight"], name="critic fc1 weight")
        _copy_tensor(policy.mlp_extractor.value_net[2].bias, state[f"{critic_prefix}.2.bias"], name="critic fc1 bias")
        _copy_tensor(policy.mlp_extractor.value_net[4].weight, state[f"{critic_prefix}.4.weight"], name="critic fc2 weight")
        _copy_tensor(policy.mlp_extractor.value_net[4].bias, state[f"{critic_prefix}.4.bias"], name="critic fc2 bias")
        _copy_tensor(policy.value_net.weight, state[f"{critic_prefix}.6.weight"], name="critic head weight")
        _copy_tensor(policy.value_net.bias, state[f"{critic_prefix}.6.bias"], name="critic head bias")

    print(
        "[SafeRLWarmStart] loaded "
        f"{ckpt_path} iter={checkpoint.get('iter', '<unknown>')} "
        f"actor_std_mean={float(actor_std.mean().detach().cpu()):.6f} "
        f"critic={'yes' if load_critic and f'{critic_prefix}.0.weight' in state else 'no'}"
    )


def _maybe_init_wandb(args, log_dir: Path):
    if not bool(args.use_wandb):
        return None
    import wandb

    kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "mode": args.wandb_mode,
        "config": vars(args),
        "dir": str(log_dir),
        "name": args.wandb_run_name or args.exp_name,
        "group": args.wandb_group or None,
    }
    if args.wandb_entity:
        kwargs["entity"] = args.wandb_entity
    return wandb.init(**kwargs)


def main() -> None:
    args = build_parser().parse_args()
    log_dir, model_dir = _prepare_run_dirs(args)
    wandb_run = _maybe_init_wandb(args, log_dir)

    vec_cls = SubprocVecEnv if args.vec_env == "subproc" and args.num_envs > 1 else DummyVecEnv
    env_fns = [
        (lambda rank=rank: _make_single_env(args, seed=int(args.seed) + rank))
        for rank in range(int(args.num_envs))
    ]
    env = vec_cls(env_fns)
    init_ppo_path = Path(str(getattr(args, "init_ppo_model_path", "")).strip()).expanduser()
    init_vecnormalize_path = _vecnormalize_path_for_model(init_ppo_path) if str(init_ppo_path).strip() else None
    if (args.normalize_obs or args.normalize_reward) and init_vecnormalize_path is not None:
        print(f"[PPOWarmStart] loading VecNormalize stats: {init_vecnormalize_path}", flush=True)
        env = VecNormalize.load(str(init_vecnormalize_path), env)
        env.training = True
        env.norm_obs = bool(args.normalize_obs)
        env.norm_reward = bool(args.normalize_reward)
        env.gamma = float(args.gamma)
    elif args.normalize_obs or args.normalize_reward:
        env = VecNormalize(
            env,
            norm_obs=bool(args.normalize_obs),
            norm_reward=bool(args.normalize_reward),
            gamma=float(args.gamma),
        )

    net_arch = [int(x.strip()) for x in str(args.net_arch).split(",") if x.strip()]
    policy_kwargs = {
        "net_arch": {"pi": net_arch, "vf": net_arch},
        "activation_fn": _activation_fn(args.activation_fn),
    }
    if str(getattr(args, "init_ppo_model_path", "")).strip():
        init_path = Path(str(args.init_ppo_model_path).strip()).expanduser()
        print(f"[PPOWarmStart] loading SB3 PPO checkpoint: {init_path}", flush=True)
        model = PPO.load(
            str(init_path),
            env=env,
            device=args.device,
            seed=int(args.seed),
            tensorboard_log=str(log_dir / "tb"),
            custom_objects={
                "learning_rate": float(args.learning_rate),
                "clip_range": float(args.clip_range),
                "n_steps": int(args.n_steps),
                "batch_size": int(args.batch_size),
                "n_epochs": int(args.n_epochs),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "ent_coef": float(args.ent_coef),
                "vf_coef": float(args.vf_coef),
                "max_grad_norm": float(args.max_grad_norm),
            },
        )
        if bool(getattr(args, "reset_ppo_optimizer", False)):
            lr_now = float(args.learning_rate)
            model.policy.optimizer = model.policy.optimizer_class(
                model.policy.parameters(),
                lr=lr_now,
                **model.policy.optimizer_kwargs,
            )
            print(f"[PPOWarmStart] reset policy optimizer lr={lr_now}", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=int(args.seed),
            device=args.device,
            tensorboard_log=str(log_dir / "tb"),
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            learning_rate=float(args.learning_rate),
            clip_range=float(args.clip_range),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            max_grad_norm=float(args.max_grad_norm),
            policy_kwargs=policy_kwargs,
        )
    if str(args.safe_rl_checkpoint_path).strip():
        _load_safe_rl_ppo_checkpoint(
            model,
            str(args.safe_rl_checkpoint_path).strip(),
            load_critic=bool(args.safe_rl_load_critic),
            actor_std_override=float(getattr(args, "safe_rl_actor_std_override", 0.0)),
        )
    callbacks = [
        CheckpointCallback(
            save_freq=max(1, int(args.save_interval) // max(1, int(args.num_envs))),
            save_path=str(model_dir),
            name_prefix="ppo_step",
            save_vecnormalize=bool(args.normalize_obs or args.normalize_reward),
        ),
        WandbMetricCallback(wandb_run=wandb_run, log_freq=int(args.log_interval)),
    ]
    if int(getattr(args, "eval_interval", 0)) > 0:
        callbacks.append(
            PeriodicEvalCallback(
                args=args,
                log_dir=log_dir,
                wandb_run=wandb_run,
                eval_freq=int(args.eval_interval),
                num_episodes=int(args.eval_episodes),
                save_plots=bool(args.eval_save_plots),
                plot_max_episodes=int(args.eval_plot_max_episodes),
                reward_surface=bool(args.eval_reward_surface),
            )
        )
    model.learn(total_timesteps=int(args.total_timesteps), callback=callbacks, progress_bar=False)
    model.save(str(model_dir / "final.zip"))
    if isinstance(env, VecNormalize):
        env.save(str(model_dir / "vecnormalize.pkl"))
    env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
