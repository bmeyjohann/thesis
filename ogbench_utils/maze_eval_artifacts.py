from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from .env_wrappers_maze import build_ogbench_maze_wrapper, maybe_set_goal_color
from .obs import prepare_observation

WALL_TILE_IDS = {1}


@dataclass
class MazeTrajectoryEpisode:
    positions: np.ndarray
    goal_xy: np.ndarray
    success: bool
    timeout: bool
    lethal: bool
    total_reward: float
    length: int


def _maze_make_kwargs(args) -> dict:
    kwargs: dict = {"render_mode": "rgb_array"}
    max_episode_steps = int(getattr(args, "max_episode_steps", 0) or 0)
    if max_episode_steps > 0:
        kwargs["max_episode_steps"] = max_episode_steps
    return kwargs


def _build_single_maze_eval_env(args) -> gym.Env:
    env = gym.make(str(args.env_name), **_maze_make_kwargs(args))
    wrapper = build_ogbench_maze_wrapper(
        env_name=str(args.env_name),
        obs_mode=str(getattr(args, "obs_mode", "state")),
        include_goal=bool(getattr(args, "include_goal", True)),
        include_distance=bool(getattr(args, "include_distance", False)),
        include_direction=bool(getattr(args, "include_direction", False)),
        include_velocity=bool(getattr(args, "include_velocity", False)),
        reward_type=str(getattr(args, "reward_type", "sparse")),
        dense_reward_scale=float(getattr(args, "dense_reward_scale", 0.01)),
        step_penalty=float(getattr(args, "step_penalty", 0.0)),
        reward_switch_after_steps=int(getattr(args, "reward_switch_after_steps", 0) or 0),
        intervention_mode="none",
        teacher_type=str(getattr(args, "teacher_type", "bfs")),
        tolerance_type=str(getattr(args, "tolerance_type", "angle")),
        tolerance_value=float(getattr(args, "tolerance_value", 30.0)),
        tolerance_channel_weights=getattr(args, "tolerance_channel_weights", None),
        tolerance_xyz_value=float(getattr(args, "tolerance_xyz_value", -1.0)),
        tolerance_yaw_value=float(getattr(args, "tolerance_yaw_value", -1.0)),
        tolerance_gripper_value=float(getattr(args, "tolerance_gripper_value", -1.0)),
        tolerance_adaptive_enable=bool(getattr(args, "tolerance_adaptive_enable", True)),
        tolerance_adaptive_near_distance=float(getattr(args, "tolerance_adaptive_near_distance", 0.08)),
        tolerance_adaptive_far_distance=float(getattr(args, "tolerance_adaptive_far_distance", 0.30)),
        tolerance_adaptive_near_scale=float(getattr(args, "tolerance_adaptive_near_scale", 0.35)),
        binary_gripper_actions=bool(getattr(args, "binary_gripper_actions", False)),
        binary_gripper_threshold=float(getattr(args, "binary_gripper_threshold", 0.0)),
        hard_gripper_intervention=bool(getattr(args, "hard_gripper_intervention", False)),
        gripper_intervene_pick_radius=float(getattr(args, "gripper_intervene_pick_radius", 0.06)),
        gripper_intervene_place_radius=float(getattr(args, "gripper_intervene_place_radius", 0.06)),
        gripper_intervene_contact_threshold=float(getattr(args, "gripper_intervene_contact_threshold", 0.3)),
        hard_block_lethal=bool(getattr(args, "hard_block_lethal", True)),
        intervention_enable_after_steps=int(getattr(args, "intervention_enable_after_steps", 0) or 0),
        intervention_safety_margin_frac=float(getattr(args, "intervention_safety_margin_frac", 0.0)),
        intervention_release_steps=int(getattr(args, "intervention_release_steps", 3) or 3),
        intervention_reward_patience_steps=int(getattr(args, "intervention_reward_patience_steps", 5) or 5),
        intervention_reward_improvement_epsilon=float(
            getattr(args, "intervention_reward_improvement_epsilon", 1e-6)
        ),
        intervention_episode_prob=0.0,
        intervention_episode_prob_min=0.0,
        intervention_episode_prob_decay_steps=0,
        intervention_episode_prob_decay_start=0,
        intervention_episode_prob_seed=getattr(args, "intervention_episode_prob_seed", None),
        static_reset_seed=getattr(args, "static_reset_seed", None),
    )
    env = wrapper(env)
    maybe_set_goal_color(env, str(getattr(args, "goal_marker_color", "auto")))
    return env


def _maze_layout_spec(env: gym.Env) -> tuple[np.ndarray | None, float | None, tuple[float, float] | None, int | None]:
    base = getattr(env, "unwrapped", env)
    maze_map = getattr(base, "maze_map", None)
    maze_unit = getattr(base, "_maze_unit", None)
    offset_x = getattr(base, "_offset_x", None)
    offset_y = getattr(base, "_offset_y", None)
    dangerous_id = getattr(base, "_dangerous_tile_id", None)
    if maze_map is None or maze_unit is None or offset_x is None or offset_y is None:
        return None, None, None, None
    return np.array(maze_map, copy=True), float(maze_unit), (float(offset_x), float(offset_y)), int(dangerous_id)


def _current_xy_goal(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    base = getattr(env, "unwrapped", env)
    xy = np.asarray(base.get_xy(), dtype=np.float32).reshape(2)
    goal_xy = np.asarray(getattr(base, "cur_goal_xy"), dtype=np.float32).reshape(2)
    return xy, goal_xy


def rollout_maze_policy_episodes(
    *,
    args,
    device: torch.device,
    policy_step_fn: Callable[[torch.Tensor], torch.Tensor],
    num_episodes: int,
    max_steps: int | None = None,
) -> tuple[list[MazeTrajectoryEpisode], tuple[np.ndarray | None, float | None, tuple[float, float] | None, int | None]]:
    env = _build_single_maze_eval_env(args)
    try:
        layout_spec = _maze_layout_spec(env)
        episodes: list[MazeTrajectoryEpisode] = []
        for ep_idx in range(max(1, int(num_episodes))):
            obs, info = env.reset(seed=(int(getattr(args, "viz_seed", 0)) + ep_idx))
            start_xy, goal_xy = _current_xy_goal(env)
            positions = [start_xy.copy()]
            total_reward = 0.0
            step_count = 0
            success = False
            timeout = False
            lethal = False
            episode_limit = int(max_steps or getattr(env, "_max_episode_steps", 0) or getattr(getattr(env, "spec", None), "max_episode_steps", 0) or 1000)
            while step_count < episode_limit:
                obs_tensor = prepare_observation(
                    obs,
                    device=device,
                    obs_mode="state",
                    pixel_shape=None,
                    flatten=True,
                )
                with torch.no_grad():
                    action_tensor = policy_step_fn(obs_tensor)
                action = (
                    action_tensor.detach()
                    .reshape(-1)
                    .to(device="cpu", dtype=torch.float32, non_blocking=False)
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                obs, reward, terminated, truncated, info = env.step(action)
                xy_now, goal_xy = _current_xy_goal(env)
                positions.append(xy_now.copy())
                total_reward += float(reward)
                step_count += 1
                done = bool(terminated or truncated)
                if done:
                    success = bool(info.get("goal_reached", False))
                    timeout = bool(truncated or info.get("truncated", False))
                    lethal = bool(info.get("killed", False))
                    break
            episodes.append(
                MazeTrajectoryEpisode(
                    positions=np.asarray(positions, dtype=np.float32),
                    goal_xy=np.asarray(goal_xy, dtype=np.float32),
                    success=success,
                    timeout=timeout,
                    lethal=lethal,
                    total_reward=float(total_reward),
                    length=int(step_count),
                )
            )
        return episodes, layout_spec
    finally:
        env.close()


def plot_maze_trajectory_episodes(
    *,
    episodes: list[MazeTrajectoryEpisode],
    output_path: Path,
    env_name: str,
    maze_layout: np.ndarray | None,
    maze_unit: float | None,
    offsets: tuple[float, float] | None,
    dangerous_id: int | None,
    title_suffix: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    if maze_layout is not None and maze_unit is not None and offsets is not None:
        off_x, off_y = offsets
        rows, cols = maze_layout.shape
        for i in range(rows):
            for j in range(cols):
                tile = int(maze_layout[i, j])
                x = j * maze_unit - off_x
                y = i * maze_unit - off_y
                if tile in WALL_TILE_IDS:
                    patch = patches.Rectangle(
                        (x - maze_unit / 2.0, y - maze_unit / 2.0),
                        maze_unit,
                        maze_unit,
                        linewidth=0.5,
                        edgecolor="black",
                        facecolor="#343434",
                        alpha=0.85,
                    )
                    ax.add_patch(patch)
                elif dangerous_id is not None and tile == int(dangerous_id):
                    patch = patches.Rectangle(
                        (x - maze_unit / 2.0, y - maze_unit / 2.0),
                        maze_unit,
                        maze_unit,
                        linewidth=0.3,
                        edgecolor="#b81d13",
                        facecolor="#ef4136",
                        alpha=0.35,
                    )
                    ax.add_patch(patch)
        x_min = -off_x - maze_unit / 2.0
        x_max = (cols - 1) * maze_unit - off_x + maze_unit / 2.0
        y_min = -off_y - maze_unit / 2.0
        y_max = (rows - 1) * maze_unit - off_y + maze_unit / 2.0
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    success_count = 0
    timeout_count = 0
    lethal_count = 0
    for idx, episode in enumerate(episodes):
        color = "#2ca02c" if episode.success else ("#d62728" if episode.lethal else "#ff7f0e")
        label = None
        if episode.success:
            label = "success"
            success_count += 1
        elif episode.lethal:
            label = "lethal"
            lethal_count += 1
        elif episode.timeout:
            label = "timeout"
            timeout_count += 1
        xy = np.asarray(episode.positions, dtype=np.float32)
        if xy.ndim == 2 and xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.0, alpha=0.78)
            ax.scatter(xy[0, 0], xy[0, 1], color=color, s=25, marker="o", alpha=0.9)
            ax.scatter(xy[-1, 0], xy[-1, 1], color=color, s=28, marker="x", alpha=0.95)
        goal_xy = np.asarray(episode.goal_xy, dtype=np.float32).reshape(2)
        ax.scatter(goal_xy[0], goal_xy[1], color="white", edgecolors="black", linewidths=0.8, marker="*", s=140)
        if label is not None:
            ax.text(
                float(goal_xy[0]) + 0.08,
                float(goal_xy[1]) + 0.08,
                label,
                fontsize=8,
                color=color,
                alpha=0.9,
            )

    avg_len = float(np.mean([ep.length for ep in episodes])) if episodes else 0.0
    avg_reward = float(np.mean([ep.total_reward for ep in episodes])) if episodes else 0.0
    title = f"{env_name} maze eval trajectories"
    if title_suffix:
        title = f"{title} | {title_suffix}"
    subtitle = (
        f"episodes={len(episodes)} success={success_count} timeout={timeout_count} "
        f"lethal={lethal_count} avg_len={avg_len:.1f} avg_return={avg_reward:.3f}"
    )
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


class MazeEvalArtifactManager:
    def __init__(self, *, args, output_dir: Path, record_progress):
        self.args = args
        self.output_dir = output_dir
        self.record_progress = record_progress

    def enabled(self) -> bool:
        return bool(getattr(self.args, "eval_save_trajectory_plot", False))

    def maybe_save_trajectory_plot(
        self,
        *,
        device: torch.device,
        tag: str,
        step_value: int,
        policy_step_fn: Callable[[torch.Tensor], torch.Tensor],
        wandb_run,
        max_steps: int | None = None,
    ) -> Optional[Path]:
        if not self.enabled():
            return None
        try:
            num_eps = int(max(1, getattr(self.args, "eval_trajectory_plot_episodes", 5)))
            episodes, (maze_layout, maze_unit, offsets, dangerous_id) = rollout_maze_policy_episodes(
                args=self.args,
                device=device,
                policy_step_fn=policy_step_fn,
                num_episodes=num_eps,
                max_steps=max_steps,
            )
            out_path = self.output_dir / f"{tag}.png"
            plot_maze_trajectory_episodes(
                episodes=episodes,
                output_path=out_path,
                env_name=str(getattr(self.args, "env_name", "maze")),
                maze_layout=maze_layout,
                maze_unit=maze_unit,
                offsets=offsets,
                dangerous_id=dangerous_id,
                title_suffix=f"step={step_value}",
            )
            self.record_progress(f"[MazeEvalViz] saved trajectory plot {out_path}")
            if wandb_run is not None and bool(getattr(self.args, "use_wandb", False)):
                import wandb

                wandb_run.log({"viz/maze_trajectory_plot": wandb.Image(str(out_path), caption=tag)}, step=step_value)
            return out_path
        except Exception as exc:
            self.record_progress(f"[MazeEvalViz] failed to save trajectory plot: {exc}")
            return None
