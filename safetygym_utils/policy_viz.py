from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from .env import clip_action_to_space, extract_step_limit, make_safety_env, scale_action_np
from .io import load_args_json, maybe_find_args_json_from_model
from .metrics import EpisodeWindow, augment_rollout_summary, classify_outcome
from .sac import SafetyActor, SafetyCritic

_FAST_SAC_PATH = Path(__file__).resolve().parent.parent / "fasttd3" / "fast_sac"
if _FAST_SAC_PATH.exists():
    _fast_sac_path_str = str(_FAST_SAC_PATH)
    if _fast_sac_path_str not in sys.path:
        sys.path.insert(0, _fast_sac_path_str)

try:
    from fast_sac_utils import EmpiricalNormalization  # type: ignore
except Exception:  # pragma: no cover
    EmpiricalNormalization = None


def _parse_float_list(value: str | Iterable[float] | None, default: tuple[float, ...]) -> list[float]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        vals = [float(part) for part in parts if part]
        return vals if vals else list(default)
    return [float(v) for v in value]


def _merge_checkpoint_args(model_path: Path, checkpoint_args: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(checkpoint_args or {})
    args_path = maybe_find_args_json_from_model(model_path)
    if args_path is not None and args_path.exists():
        try:
            file_cfg = load_args_json(args_path)
        except Exception:
            file_cfg = {}
        if isinstance(file_cfg, dict):
            for key, value in file_cfg.items():
                cfg.setdefault(key, value)
    return cfg


def _load_checkpoint(model_path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "actor_state_dict" not in checkpoint or "critic_state_dict" not in checkpoint:
        raise KeyError(f"{model_path} is not a SafetyGym FastSAC checkpoint")
    train_args = _merge_checkpoint_args(model_path, checkpoint.get("args", {}) or {})
    return checkpoint, train_args


def _quat_wxyz_from_yaw(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _heading_label(heading_deg: float) -> str:
    value = float(heading_deg)
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}deg"
    return f"{value:.1f}deg"


def _set_goal_xy(task, goal_xy: np.ndarray) -> None:
    goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    task.world_info.layout["goal"] = goal_xy.copy()
    task.world_info.world_config_dict["geoms"]["goal"]["pos"][:2] = goal_xy.copy()
    task._set_goal(goal_xy)


def _set_agent_pose(task, state_template: dict[str, Any], xy: np.ndarray, yaw_rad: float, agent_z: float) -> None:
    xy = np.asarray(xy, dtype=np.float64).reshape(2)
    state = {
        "time": float(state_template["time"]),
        "qpos": np.array(state_template["qpos"], copy=True),
        "qvel": np.zeros_like(state_template["qvel"]),
        "act": None if state_template.get("act") is None else np.zeros_like(state_template["act"]),
    }
    state["qpos"][0] = float(xy[0])
    state["qpos"][1] = float(xy[1])
    state["qpos"][2] = float(agent_z)
    state["qpos"][3:7] = _quat_wxyz_from_yaw(yaw_rad)
    task.world.set_state(state)
    task.world_info.layout["agent"] = xy.copy()
    import mujoco

    mujoco.mj_forward(task.model, task.data)


def _extract_bounds(task, x_range: list[float] | None, y_range: list[float] | None) -> tuple[float, float, float, float]:
    if x_range is not None and y_range is not None:
        return (float(x_range[0]), float(x_range[1]), float(y_range[0]), float(y_range[1]))
    extents = getattr(getattr(task, "placements_conf", None), "extents", None)
    arr = None if extents is None else np.asarray(extents, dtype=np.float64).reshape(-1)
    if arr is not None and arr.size == 4 and np.isfinite(arr).all():
        x_min, y_min, x_max, y_max = map(float, arr.tolist())
    else:
        x_min, y_min, x_max, y_max = -2.0, -2.0, 2.0, 2.0
    if x_range is not None:
        x_min, x_max = float(x_range[0]), float(x_range[1])
    if y_range is not None:
        y_min, y_max = float(y_range[0]), float(y_range[1])
    return (x_min, x_max, y_min, y_max)


def _extract_overlay_specs(task) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    world_cfg = getattr(getattr(task, "world_info", None), "world_config_dict", None)
    if not isinstance(world_cfg, dict):
        return specs
    for section in ("geoms", "free_geoms"):
        items = world_cfg.get(section, {})
        if not isinstance(items, dict):
            continue
        for name, cfg in items.items():
            if not isinstance(cfg, dict) or str(name) in {"agent", "goal"}:
                continue
            geom_list = cfg.get("geoms", [])
            if not geom_list:
                continue
            geom = geom_list[0]
            specs.append(
                {
                    "name": str(name),
                    "section": str(section),
                    "geom_type": str(geom.get("type", "sphere")).lower(),
                    "size": np.asarray(geom.get("size", [0.1, 0.1, 0.1]), dtype=np.float64).reshape(-1),
                }
            )
    return specs


def _body_xy_and_yaw(task, body_name: str) -> tuple[np.ndarray, float] | None:
    try:
        body = task.data.body(str(body_name))
        pos = np.asarray(body.xpos, dtype=np.float64).reshape(-1)
        mat = np.asarray(body.xmat, dtype=np.float64).reshape(3, 3)
    except Exception:
        return None
    if pos.size < 2 or not np.isfinite(pos[:2]).all() or not np.isfinite(mat).all():
        return None
    yaw = math.atan2(float(mat[1, 0]), float(mat[0, 0]))
    return pos[:2].copy(), yaw


def _overlay_world(ax: plt.Axes, *, task, overlay_specs: list[dict[str, Any]], goal_xy: np.ndarray) -> None:
    color_map = {
        "hazard": ("#d65244", 0.45),
        "vase": ("#f1b24a", 0.55),
        "pillar": ("#8a6fd1", 0.55),
    }
    for spec in overlay_specs:
        body_pose = _body_xy_and_yaw(task, spec["name"])
        if body_pose is None:
            continue
        pos_xy, yaw = body_pose
        size = spec["size"]
        label = str(spec["name"]).lower()
        face_color, alpha = next(
            (value for key, value in color_map.items() if key in label),
            ("#8b9bb4", 0.35),
        )
        geom_type = spec["geom_type"]
        if geom_type in {"cylinder", "sphere"}:
            radius = float(size[0]) if size.size >= 1 else 0.1
            patch = patches.Circle((pos_xy[0], pos_xy[1]), radius=radius, facecolor=face_color, edgecolor="black", linewidth=0.6, alpha=alpha)
        elif geom_type == "box":
            half_w = float(size[0]) if size.size >= 1 else 0.1
            half_h = float(size[1]) if size.size >= 2 else half_w
            patch = patches.Rectangle(
                (pos_xy[0] - half_w, pos_xy[1] - half_h),
                width=2.0 * half_w,
                height=2.0 * half_h,
                angle=np.degrees(yaw),
                rotation_point="center",
                facecolor=face_color,
                edgecolor="black",
                linewidth=0.6,
                alpha=alpha,
            )
        else:
            radius = float(size[0]) if size.size >= 1 else 0.1
            patch = patches.Circle((pos_xy[0], pos_xy[1]), radius=radius, facecolor=face_color, edgecolor="black", linewidth=0.6, alpha=alpha)
        ax.add_patch(patch)
    ax.scatter(float(goal_xy[0]), float(goal_xy[1]), marker="*", s=150, c="white", edgecolors="black", linewidths=0.8, zorder=6)


def _build_networks(
    *,
    checkpoint: dict[str, Any],
    train_args: dict[str, Any],
    obs_dim: int,
    act_dim: int,
    device: torch.device,
) -> tuple[SafetyActor, SafetyCritic, torch.nn.Module]:
    actor = SafetyActor(
        n_obs=obs_dim,
        n_act=act_dim,
        num_envs=1,
        init_scale=float(train_args.get("init_scale", 0.01)),
        hidden_dim=int(train_args.get("actor_hidden_dim", 256)),
        use_layer_norm=bool(train_args.get("use_layer_norm", False)),
        layer_norm_eps=float(train_args.get("layer_norm_eps", 1e-5)),
        device=device,
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    critic = SafetyCritic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=int(train_args.get("critic_hidden_dim", 512)),
        num_critics=int(train_args.get("num_critics", 2)),
        use_layer_norm=bool(train_args.get("use_layer_norm", False)),
        layer_norm_eps=float(train_args.get("layer_norm_eps", 1e-5)),
        device=device,
    )
    critic.load_state_dict(checkpoint["critic_state_dict"])
    critic.eval()
    obs_preprocess = torch.nn.Identity()
    obs_norm_state = checkpoint.get("obs_normalizer_state_dict", None)
    if obs_norm_state and EmpiricalNormalization is not None:
        obs_preprocess = EmpiricalNormalization(shape=obs_dim, device=device)
        obs_preprocess.load_state_dict(obs_norm_state, strict=False)
        obs_preprocess.eval()
    return actor, critic, obs_preprocess


def _compute_vector_field(
    *,
    env,
    task,
    state_template: dict[str, Any],
    goal_xy: np.ndarray,
    heading_deg: float,
    xs: np.ndarray,
    ys: np.ndarray,
    actions: np.ndarray,
    quiver_stride: int,
    agent_z: float,
    scale_actor_to_env_bounds: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stride = max(1, int(quiver_stride))
    xs_sub = xs[::stride]
    ys_sub = ys[::stride]
    actions_sub = actions[::stride, ::stride, :]
    grid_x, grid_y = np.meshgrid(xs_sub, ys_sub)
    arrows_dx = np.full_like(grid_x, np.nan, dtype=np.float64)
    arrows_dy = np.full_like(grid_y, np.nan, dtype=np.float64)
    yaw_rad = math.radians(float(heading_deg))
    _set_goal_xy(task, goal_xy)
    for row_idx, y in enumerate(ys_sub):
        for col_idx, x in enumerate(xs_sub):
            action = np.asarray(actions_sub[row_idx, col_idx], dtype=np.float32)
            if scale_actor_to_env_bounds:
                action = scale_action_np(action, env.action_space)
            _set_agent_pose(task, state_template, np.asarray([x, y], dtype=np.float64), yaw_rad, agent_z)
            before = np.asarray(task.agent.pos[:2], dtype=np.float64)
            task.simulation_forward(clip_action_to_space(action, env.action_space))
            after = np.asarray(task.agent.pos[:2], dtype=np.float64)
            delta = after - before
            arrows_dx[row_idx, col_idx] = float(delta[0])
            arrows_dy[row_idx, col_idx] = float(delta[1])
    return grid_x, grid_y, arrows_dx, arrows_dy


def _evaluate_heading_grid(
    *,
    env,
    task,
    actor: SafetyActor,
    critic: SafetyCritic,
    obs_preprocess,
    device: torch.device,
    xs: np.ndarray,
    ys: np.ndarray,
    goal_xy: np.ndarray,
    heading_deg: float,
    quiver_stride: int,
    state_template: dict[str, Any],
    agent_z: float,
    scale_actor_to_env_bounds: bool,
) -> dict[str, Any]:
    pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    yaw_rad = math.radians(float(heading_deg))
    _set_goal_xy(task, goal_xy)
    observations: list[np.ndarray] = []
    for xy in pts:
        _set_agent_pose(task, state_template, xy, yaw_rad, agent_z)
        obs = np.asarray(task.obs(), dtype=np.float32).reshape(-1)
        observations.append(obs)
    obs_np = np.stack(observations, axis=0)
    obs_t = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
    obs_t = obs_preprocess(obs_t)
    with torch.no_grad():
        _, _, mean_actions = actor(obs_t)
        critic_actions = mean_actions
        if scale_actor_to_env_bounds:
            low_t = torch.as_tensor(env.action_space.low.reshape(1, -1), device=device, dtype=torch.float32)
            high_t = torch.as_tensor(env.action_space.high.reshape(1, -1), device=device, dtype=torch.float32)
            center = 0.5 * (high_t + low_t)
            half = 0.5 * (high_t - low_t)
            critic_actions = center + mean_actions * half
        q_outputs = critic(obs_t, critic_actions)
        q_stack = torch.stack(list(q_outputs), dim=0)
        value = torch.min(q_stack, dim=0).values.detach().cpu().numpy().reshape(len(ys), len(xs))
        disagreement = (
            torch.max(q_stack, dim=0).values - torch.min(q_stack, dim=0).values
        ).detach().cpu().numpy().reshape(len(ys), len(xs))
        actions = mean_actions.detach().cpu().numpy().reshape(len(ys), len(xs), -1)

    left = actions[..., 0]
    right = actions[..., 1] if actions.shape[-1] > 1 else np.zeros_like(left)
    throttle = 0.5 * (left + right)
    turn = 0.5 * (right - left)
    qx, qy, arrows_dx, arrows_dy = _compute_vector_field(
        env=env,
        task=task,
        state_template=state_template,
        goal_xy=goal_xy,
        heading_deg=heading_deg,
        xs=xs,
        ys=ys,
        actions=actions,
        quiver_stride=quiver_stride,
        agent_z=agent_z,
        scale_actor_to_env_bounds=scale_actor_to_env_bounds,
    )
    return {
        "heading_deg": float(heading_deg),
        "value": value,
        "disagreement": disagreement,
        "throttle": throttle,
        "turn": turn,
        "quiver_x": qx,
        "quiver_y": qy,
        "quiver_dx": arrows_dx,
        "quiver_dy": arrows_dy,
    }


def _plot_heading_maps(
    *,
    output_dir: Path,
    base_name: str,
    xs: np.ndarray,
    ys: np.ndarray,
    heading_maps: list[dict[str, Any]],
    task,
    overlay_specs: list[dict[str, Any]],
    goal_xy: np.ndarray,
) -> Path:
    rows = max(1, len(heading_maps))
    fig, axes = plt.subplots(rows, 3, figsize=(19, 5.8 * rows), constrained_layout=True)
    axes_2d = np.asarray(axes, dtype=object).reshape(rows, 3)
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    extent = [float(xs.min() - dx / 2), float(xs.max() + dx / 2), float(ys.min() - dy / 2), float(ys.max() + dy / 2)]

    for row_idx, result in enumerate(heading_maps):
        heading = result["heading_deg"]
        row_axes = axes_2d[row_idx]
        step_mag = np.sqrt(np.square(result["quiver_dx"]) + np.square(result["quiver_dy"]))
        finite_mag = step_mag[np.isfinite(step_mag)]
        mag_vmax = float(np.quantile(finite_mag, 0.95)) if finite_mag.size else 1.0
        mag_vmax = max(mag_vmax, 1e-6)
        panels = [
            ("Critic min(Q)", result["value"], "viridis", None, None),
            ("Critic disagreement", result["disagreement"], "magma", 0.0, None),
            ("Policy local displacement", step_mag, "cividis", 0.0, mag_vmax),
        ]
        for ax, (title, grid, cmap, vmin, vmax) in zip(row_axes, panels):
            im = ax.imshow(
                np.asarray(grid, dtype=np.float64),
                extent=extent,
                origin="lower",
                aspect="equal",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            _overlay_world(ax, task=task, overlay_specs=overlay_specs, goal_xy=goal_xy)
            ax.set_title(f"{title}\nheading={_heading_label(heading)}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.82)
        row_axes[2].quiver(
            result["quiver_x"],
            result["quiver_y"],
            result["quiver_dx"],
            result["quiver_dy"],
            step_mag,
            angles="xy",
            scale_units="xy",
            scale=max(mag_vmax * 10.0, 1e-6),
            width=0.006,
            cmap="coolwarm",
            pivot="mid",
            minlength=0.0,
        )
        row_axes[2].text(
            0.02,
            0.96,
            "arrows: one-step world displacement\ncolor/length: displacement magnitude",
            transform=row_axes[2].transAxes,
            color="white",
            fontsize=9,
            va="top",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=4),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{base_name}.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return png_path


def _run_rollout_eval(
    *,
    env_name: str,
    train_args: dict[str, Any],
    actor: SafetyActor,
    obs_preprocess,
    device: torch.device,
    seed: int,
    num_episodes: int,
    max_episode_steps: int,
) -> tuple[dict[str, float], list[dict[str, Any]], Any, list[dict[str, Any]]]:
    env = make_safety_env(
        env_name,
        render_mode="none",
        max_episode_steps=max_episode_steps,
        surface_mode=str(train_args.get("surface_mode", "default")),
        car_wheel_command_limit=float(train_args.get("car_wheel_command_limit", 2.0)),
        car_force_scale=float(train_args.get("car_force_scale", 2.0)),
        car_action_mode=str(train_args.get("car_action_mode", "raw_wheels")),
        obs_mask_mode=str(train_args.get("obs_mask_mode", "none")),
        seed=seed,
    )
    task = env.unwrapped.task
    overlay_specs = _extract_overlay_specs(task)
    max_steps = extract_step_limit(env)
    win = EpisodeWindow(size=max(10, int(num_episodes)))
    trajectories: list[dict[str, Any]] = []
    scale_actor_to_env_bounds = bool(train_args.get("scale_actor_to_env_bounds", False))

    for episode_idx in range(int(num_episodes)):
        obs, _ = env.reset(seed=int(seed + episode_idx))
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        goal_xy = np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
        path = [np.asarray(task.agent.pos[:2], dtype=np.float64).copy()]
        ep_ret = 0.0
        ep_cost = 0.0
        ep_len = 0
        ep_goal_hits = 0
        ep_first_goal_hit_step: int | None = None
        ep_first_goal_reward_sum = 0.0
        ep_first_goal_dense_reward_sum = 0.0
        while ep_len < max_steps:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
                obs_t = obs_preprocess(obs_t)
                _, _, mean_action = actor(obs_t)
                action = mean_action[0].detach().cpu().numpy().astype(np.float32)
            if scale_actor_to_env_bounds:
                action = scale_action_np(action, env.action_space)
            action = clip_action_to_space(action, env.action_space)
            next_obs, reward, cost, terminated, truncated, info = env.step(action)
            obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            path.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
            ep_ret += float(reward)
            ep_cost += float(cost)
            ep_len += 1
            if bool(info.get("goal_met", False)):
                ep_goal_hits += 1
                if ep_first_goal_hit_step is None:
                    ep_first_goal_hit_step = int(ep_len)
                    ep_first_goal_reward_sum = float(ep_ret)
                    ep_first_goal_dense_reward_sum = float(ep_ret)
            if terminated or truncated:
                outcome = classify_outcome(goal_met=ep_goal_hits > 0, episode_steps=ep_len, max_episode_steps=max_steps)
                first_hit = int(ep_first_goal_hit_step) if ep_first_goal_hit_step is not None else int(max_steps)
                win.add(
                    {
                        "episode_return": float(ep_ret),
                        "episode_cost_sum": float(ep_cost),
                        "episode_cost_rate": float(ep_cost / max(1, ep_len)),
                        "episode_length": float(ep_len),
                        "reward_shaped_sum": float(ep_ret),
                        "reward_raw_env_sum": float(ep_ret),
                        "reward_dense_sum": float(ep_ret),
                        "reward_sparse_sum": float(0.0),
                        "reward_step_penalty_sum": float(0.0),
                        "intervention_steps": 0.0,
                        "intervention_fraction": 0.0,
                        "intervention_num_bursts": 0.0,
                        "intervention_avg_burst_len": 0.0,
                        "goal_met": 1.0 if ep_goal_hits > 0 else 0.0,
                        "goal_met_count": float(ep_goal_hits),
                        "first_goal_success": 1.0 if ep_first_goal_hit_step is not None else 0.0,
                        "first_goal_hit_step": float(first_hit),
                        "first_goal_hit_step_success_only": float(first_hit if ep_first_goal_hit_step is not None else 0.0),
                        "first_goal_within_100": 1.0 if ep_first_goal_hit_step is not None and first_hit <= 100 else 0.0,
                        "first_goal_within_200": 1.0 if ep_first_goal_hit_step is not None and first_hit <= 200 else 0.0,
                        "first_goal_reward_sum": float(ep_first_goal_reward_sum),
                        "first_goal_dense_reward_sum": float(ep_first_goal_dense_reward_sum),
                        "final_distance_to_goal": float(task.dist_goal()),
                        "outcome_success": 1.0 if outcome == "success" else 0.0,
                        "outcome_timeout": 1.0 if outcome == "timeout" else 0.0,
                        "outcome_kill": 1.0 if outcome == "kill" else 0.0,
                        "outcome_other_failure": 1.0 if outcome not in {"success", "timeout", "kill"} else 0.0,
                        "terminated": 1.0 if terminated else 0.0,
                        "truncated": 1.0 if truncated else 0.0,
                    }
                )
                trajectories.append(
                    {
                        "goal_xy": goal_xy,
                        "path": np.stack(path, axis=0),
                        "goal_hits": int(ep_goal_hits),
                    }
                )
                break

    summary = augment_rollout_summary(win.summary("eval"), "eval")
    env.close()
    return summary, trajectories, task, overlay_specs


def _plot_rollouts(
    *,
    output_dir: Path,
    base_name: str,
    trajectories: list[dict[str, Any]],
    task,
    overlay_specs: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    rollout_summary: dict[str, float],
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    x_min, x_max, y_min, y_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    goal_xy = np.asarray(task.goal.pos[:2], dtype=np.float64)
    _overlay_world(ax, task=task, overlay_specs=overlay_specs, goal_xy=goal_xy)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(trajectories))))
    for idx, traj in enumerate(trajectories):
        path = np.asarray(traj["path"], dtype=np.float64)
        goal = np.asarray(traj["goal_xy"], dtype=np.float64)
        ax.plot(path[:, 0], path[:, 1], color=colors[idx], linewidth=2.0, alpha=0.9)
        ax.scatter(path[0, 0], path[0, 1], color=colors[idx], s=28, marker="o")
        ax.scatter(path[-1, 0], path[-1, 1], color=colors[idx], s=36, marker="x")
        ax.scatter(goal[0], goal[1], color=colors[idx], s=70, marker="*", edgecolors="black", linewidths=0.5)
    ax.set_title(
        "SafetyGym policy rollouts\n"
        f"success={rollout_summary.get('eval/goal_success_rate', 0.0):.2f} "
        f"goals/ep={rollout_summary.get('eval/goals_per_episode', 0.0):.2f} "
        f"final_dist={rollout_summary.get('eval/final_distance_to_goal_mean', 0.0):.2f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{base_name}.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return png_path


def plot_eval_episode_trajectory(
    *,
    output_path: str | Path,
    task,
    overlay_specs: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    path: np.ndarray,
    goal_positions: np.ndarray,
    goal_hit_points: np.ndarray | None,
    episode_idx: int,
    total_episodes: int,
    episode_reward: float,
    goals_reached: int,
    final_distance: float,
) -> Path:
    output_path = Path(output_path).resolve()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    x_min, x_max, y_min, y_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    current_goal = np.asarray(task.goal.pos[:2], dtype=np.float64)
    _overlay_world(ax, task=task, overlay_specs=overlay_specs, goal_xy=current_goal)

    path = np.asarray(path, dtype=np.float64)
    if path.ndim == 2 and path.shape[0] >= 2:
        ax.plot(path[:, 0], path[:, 1], color="#2b8cbe", linewidth=2.2, alpha=0.95, zorder=7)
        ax.scatter(path[0, 0], path[0, 1], color="#08519c", s=34, marker="o", zorder=8)
        ax.scatter(path[-1, 0], path[-1, 1], color="#cb181d", s=42, marker="x", zorder=8)

    goal_positions = np.asarray(goal_positions, dtype=np.float64)
    for idx, goal_xy in enumerate(goal_positions):
        marker_color = "#252525" if idx == 0 else "#6a51a3"
        ax.scatter(goal_xy[0], goal_xy[1], color=marker_color, s=90, marker="*", edgecolors="white", linewidths=0.6, zorder=9)
        ax.text(
            float(goal_xy[0]) + 0.04,
            float(goal_xy[1]) + 0.04,
            str(idx + 1),
            fontsize=10,
            color="black",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
            zorder=10,
        )

    if goal_hit_points is not None:
        hits = np.asarray(goal_hit_points, dtype=np.float64)
        if hits.size > 0:
            ax.scatter(hits[:, 0], hits[:, 1], color="#31a354", s=46, marker="P", edgecolors="black", linewidths=0.4, zorder=10)

    ax.set_title(
        "SafetyGym eval episode\n"
        f"ep={int(episode_idx)}/{int(total_episodes)} reward={float(episode_reward):.3f} "
        f"goals={int(goals_reached)} final_dist={float(final_distance):.2f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_episode_contact_sheet(
    *,
    image_paths: Iterable[str | Path],
    output_path: str | Path,
    title: str,
    max_cols: int = 3,
) -> Path | None:
    paths = [Path(p).resolve() for p in image_paths]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    images = [plt.imread(str(p)) for p in paths]
    cols = max(1, min(int(max_cols), len(images)))
    rows = int(math.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes, dtype=object).reshape(rows, cols)
    for ax in axes_arr.reshape(-1):
        ax.axis("off")
    for ax, img, path in zip(axes_arr.reshape(-1), images, paths):
        ax.imshow(img)
        ax.set_title(path.stem, fontsize=10)
        ax.axis("off")
    fig.suptitle(str(title), fontsize=14)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def generate_safety_policy_maps(
    *,
    model_path: str | Path,
    output_dir: str | Path,
    tag: str | None = None,
    env_name: str | None = None,
    device: str = "cpu",
    grid_resolution: int = 48,
    quiver_stride: int = 4,
    seed: int = 0,
    goal_override: Iterable[float] | None = None,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    headings_deg: str | Iterable[float] | None = None,
    num_rollouts: int = 4,
    rollout_seed: int | None = None,
    rollout_max_steps: int = 0,
) -> dict[str, Any]:
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    device_t = torch.device(device)
    checkpoint, train_args = _load_checkpoint(model_path, device_t)
    env_id = str(env_name or train_args.get("env_name") or "SafetyCarGoal2-v0")
    headings = _parse_float_list(headings_deg, default=(0.0, 90.0, 180.0, 270.0))

    env = make_safety_env(
        env_id,
        render_mode="none",
        max_episode_steps=int(train_args.get("max_episode_steps", 0) or 0),
        surface_mode=str(train_args.get("surface_mode", "default")),
        car_wheel_command_limit=float(train_args.get("car_wheel_command_limit", 2.0)),
        car_force_scale=float(train_args.get("car_force_scale", 2.0)),
        car_action_mode=str(train_args.get("car_action_mode", "raw_wheels")),
        obs_mask_mode=str(train_args.get("obs_mask_mode", "none")),
        seed=int(seed),
    )
    obs, _ = env.reset(seed=int(seed))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    task = env.unwrapped.task
    if not getattr(task, "observation_flatten", True):
        task.toggle_observation_space()
        obs = np.asarray(task.obs(), dtype=np.float32).reshape(-1)
    act_dim = int(np.prod(env.action_space.shape))
    actor, critic, obs_preprocess = _build_networks(
        checkpoint=checkpoint,
        train_args=train_args,
        obs_dim=int(obs.shape[0]),
        act_dim=act_dim,
        device=device_t,
    )
    state_template = task.world.get_state()
    agent_z = float(state_template["qpos"][2])
    overlay_specs = _extract_overlay_specs(task)
    goal_xy = (
        np.asarray(list(goal_override), dtype=np.float64).reshape(2)
        if goal_override is not None
        else np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
    )
    _set_goal_xy(task, goal_xy)
    bounds = _extract_bounds(task, x_range=x_range, y_range=y_range)
    xs = np.linspace(bounds[0], bounds[1], int(grid_resolution), dtype=np.float64)
    ys = np.linspace(bounds[2], bounds[3], int(grid_resolution), dtype=np.float64)

    heading_maps = [
        _evaluate_heading_grid(
            env=env,
            task=task,
            actor=actor,
            critic=critic,
            obs_preprocess=obs_preprocess,
            device=device_t,
            xs=xs,
            ys=ys,
            goal_xy=goal_xy,
            heading_deg=heading,
            quiver_stride=quiver_stride,
            state_template=state_template,
            agent_z=agent_z,
            scale_actor_to_env_bounds=bool(train_args.get("scale_actor_to_env_bounds", False)),
        )
        for heading in headings
    ]

    base_name = f"safety_viz_{model_path.stem}" + (f"_{tag}" if tag else "")
    map_png = _plot_heading_maps(
        output_dir=output_dir,
        base_name=base_name,
        xs=xs,
        ys=ys,
        heading_maps=heading_maps,
        task=task,
        overlay_specs=overlay_specs,
        goal_xy=goal_xy,
    )
    env.close()

    rollout_summary, trajectories, rollout_task, rollout_overlay_specs = _run_rollout_eval(
        env_name=env_id,
        train_args=train_args,
        actor=actor,
        obs_preprocess=obs_preprocess,
        device=device_t,
        seed=int(seed if rollout_seed is None else rollout_seed),
        num_episodes=int(max(1, num_rollouts)),
        max_episode_steps=int(rollout_max_steps or train_args.get("max_episode_steps", 0) or 0),
    )
    rollout_png = _plot_rollouts(
        output_dir=output_dir,
        base_name=f"{base_name}_rollouts",
        trajectories=trajectories,
        task=rollout_task,
        overlay_specs=rollout_overlay_specs,
        bounds=bounds,
        rollout_summary=rollout_summary,
    )

    meta = {
        "model_path": str(model_path),
        "env_name": env_id,
        "goal_xy": goal_xy.tolist(),
        "x_range": [float(bounds[0]), float(bounds[1])],
        "y_range": [float(bounds[2]), float(bounds[3])],
        "grid_resolution": int(grid_resolution),
        "quiver_stride": int(quiver_stride),
        "headings_deg": [float(v) for v in headings],
        "rollout_summary": {k: float(v) for k, v in rollout_summary.items()},
        "heading_stats": [
            {
                "heading_deg": float(item["heading_deg"]),
                "value_min": float(np.nanmin(item["value"])),
                "value_max": float(np.nanmax(item["value"])),
                "disagreement_mean": float(np.nanmean(item["disagreement"])),
                "throttle_mean": float(np.nanmean(item["throttle"])),
                "turn_mean": float(np.nanmean(item["turn"])),
            }
            for item in heading_maps
        ],
    }
    meta_path = output_dir / f"{base_name}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "map_png": map_png,
        "rollout_png": rollout_png,
        "meta_json": meta_path,
        "meta": meta,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize SafetyGym FastSAC value/policy maps over XY for fixed headings.")
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--env_name", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--grid_resolution", type=int, default=48)
    p.add_argument("--quiver_stride", type=int, default=4)
    p.add_argument("--output_dir", type=Path, default=Path("visualizations") / "safetygym")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout_seed", type=int, default=None)
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--rollout_max_steps", type=int, default=0)
    p.add_argument("--headings_deg", type=str, default="0,90,180,270")
    p.add_argument("--goal", type=float, nargs=2, default=None)
    p.add_argument("--x_range", type=float, nargs=2, default=None)
    p.add_argument("--y_range", type=float, nargs=2, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    result = generate_safety_policy_maps(
        model_path=args.model_path,
        output_dir=args.output_dir,
        tag=args.tag,
        env_name=args.env_name,
        device=args.device,
        grid_resolution=args.grid_resolution,
        quiver_stride=args.quiver_stride,
        seed=args.seed,
        goal_override=args.goal,
        x_range=args.x_range,
        y_range=args.y_range,
        headings_deg=args.headings_deg,
        num_rollouts=args.num_rollouts,
        rollout_seed=args.rollout_seed,
        rollout_max_steps=args.rollout_max_steps,
    )
    print(f"Saved policy maps to {result['map_png']}")
    print(f"Saved rollout figure to {result['rollout_png']}")
    print(f"Saved metadata to {result['meta_json']}")


if __name__ == "__main__":
    main()
