#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from safetygym_utils.env import clip_action_to_space, extract_agent_xy, extract_goal_distance, extract_step_limit, scale_action_np
from safetygym_utils.io import load_args_json
from safetygym_utils.minimal_train import _make_env_with_wrappers
from safetygym_utils.policy_viz import (
    _build_networks,
    _extract_bounds,
    _extract_overlay_specs,
    _load_checkpoint,
    _overlay_world,
)
from tools.visualize_safetygym_reward_surface import _compute_surface, _extent


class SafetyGymnasiumToGymnasium(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.setdefault("cost", float(cost))
        return obs, float(reward), bool(terminated), bool(truncated), info


def _make_ppo_single_env(args, seed: int):
    env = _make_env_with_wrappers(args=args, seed=seed, with_intervention=False, controller=None)
    return SafetyGymnasiumToGymnasium(env)


def _load_json_args(path: Path) -> dict[str, Any]:
    return load_args_json(path) if path.exists() else {}


def _namespace_with_defaults(cfg: dict[str, Any]) -> SimpleNamespace:
    defaults = {
        "env_name": "SafetyCarGoal2-v0",
        "render_mode": "none",
        "max_episode_steps": 0,
        "surface_mode": "default",
        "car_wheel_command_limit": 1.0,
        "car_force_scale": 1.0,
        "car_action_mode": "raw_wheels",
        "obs_mask_mode": "none",
        "reward_mode": "dense_plus_sparse",
        "dense_reward_scale": 1.0,
        "step_penalty": -0.001,
        "cost_penalty": 0.0,
        "cost_penalty_warmup_steps": 0,
        "cost_penalty_ramp_steps": 0,
        "clearance_penalty_scale": 1.1,
        "clearance_margin": 0.0,
        "clearance_penalty_power": 1.0,
        "clearance_penalty_mode": "softplus",
        "clearance_penalty_temperature": 0.001,
        "clearance_penalty_warmup_steps": 0,
        "clearance_penalty_ramp_steps": 0,
        "forward_reward_scale": 0.0,
        "backward_penalty_scale": 0.0,
        "heading_reward_scale": 0.0,
        "heading_positive_only": True,
        "adaptive_safety_curriculum": False,
        "adaptive_safety_goal_target": 1.0,
        "adaptive_safety_window_episodes": 10,
        "adaptive_safety_step": 0.05,
        "adaptive_safety_init": 0.0,
        "adaptive_safety_min": 0.0,
        "adaptive_safety_max": 1.0,
        "terminate_on_goal": False,
    }
    merged = dict(defaults)
    merged.update(cfg)
    return SimpleNamespace(**merged)


def _surface_kwargs(train_args: dict[str, Any]) -> dict[str, float]:
    return {
        "dense_scale": float(train_args.get("dense_reward_scale", 1.0)),
        "sparse_goal_bonus": 1.0 if str(train_args.get("reward_mode", "dense_plus_sparse")) in {"sparse", "dense_plus_sparse"} else 0.0,
        "clearance_scale": float(train_args.get("clearance_penalty_scale", 0.0)),
        "clearance_margin": float(train_args.get("clearance_margin", 0.0)),
        "clearance_temperature": float(train_args.get("clearance_penalty_temperature", 0.001)),
    }


def _plot_episode_overlay(
    *,
    output_path: Path,
    env,
    task,
    state_template: dict[str, Any],
    agent_z: float,
    overlay_specs: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    path: np.ndarray,
    goal_xy: np.ndarray,
    train_args: dict[str, Any],
    title: str,
    grid_resolution: int,
) -> dict[str, Any]:
    xs = np.linspace(bounds[0], bounds[1], int(grid_resolution), dtype=np.float64)
    ys = np.linspace(bounds[2], bounds[3], int(grid_resolution), dtype=np.float64)
    grids = _compute_surface(
        env=env,
        task=task,
        state_template=state_template,
        goal_xy=np.asarray(goal_xy, dtype=np.float64),
        xs=xs,
        ys=ys,
        agent_z=float(agent_z),
        **_surface_kwargs(train_args),
    )
    total = np.asarray(grids["total"], dtype=np.float64)
    finite = total[np.isfinite(total)]
    norm = None
    if finite.size and float(np.nanmin(finite)) < 0.0 < float(np.nanmax(finite)):
        norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=float(np.nanmin(finite)), vmax=float(np.nanmax(finite)))

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 8.0), constrained_layout=True)
    im = ax.imshow(
        total,
        extent=_extent(xs, ys),
        origin="lower",
        aspect="equal",
        cmap="coolwarm",
        norm=norm,
    )
    _overlay_world(ax, task=task, overlay_specs=overlay_specs, goal_xy=np.asarray(goal_xy, dtype=np.float64))
    path = np.asarray(path, dtype=np.float64)
    if path.ndim == 2 and len(path) >= 2:
        ax.plot(path[:, 0], path[:, 1], color="#111827", linewidth=2.3, alpha=0.95, zorder=8)
        ax.scatter(path[0, 0], path[0, 1], color="#1d4ed8", s=44, marker="o", edgecolors="white", linewidths=0.6, zorder=9)
        ax.scatter(path[-1, 0], path[-1, 1], color="#dc2626", s=58, marker="x", linewidths=1.6, zorder=9)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.82, label="state potential diagnostic")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    stats = {
        "surface_min": float(np.nanmin(total)),
        "surface_max": float(np.nanmax(total)),
        "surface_p50": float(np.nanquantile(total, 0.50)),
        "surface_p95": float(np.nanquantile(total, 0.95)),
    }
    return stats


def _make_contact_sheet(paths: list[Path], output_path: Path, title: str) -> None:
    images = [plt.imread(str(p)) for p in paths if p.exists()]
    if not images:
        return
    cols = min(3, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.7 * cols, 5.7 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes, dtype=object).reshape(rows, cols)
    for ax in axes_arr.reshape(-1):
        ax.axis("off")
    for ax, image, path in zip(axes_arr.reshape(-1), images, paths):
        ax.imshow(image)
        ax.set_title(path.stem, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _eval_fastsac(
    *,
    model_path: Path,
    output_dir: Path,
    label: str,
    seed: int,
    num_episodes: int,
    grid_resolution: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint, train_args = _load_checkpoint(model_path, device)
    env_args = _namespace_with_defaults(train_args)
    env = _make_env_with_wrappers(args=env_args, seed=int(seed), with_intervention=False, controller=None, render_mode_override="none")
    obs, _ = env.reset(seed=int(seed))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    task = env.unwrapped.task
    if not getattr(task, "observation_flatten", True):
        task.toggle_observation_space()
        obs = np.asarray(task.obs(), dtype=np.float32).reshape(-1)
    act_dim = int(np.prod(env.action_space.shape))
    actor, _, obs_preprocess = _build_networks(
        checkpoint=checkpoint,
        train_args=train_args,
        obs_dim=int(obs.shape[0]),
        act_dim=act_dim,
        device=device,
    )
    bounds = _extract_bounds(task, x_range=None, y_range=None)
    max_steps = int(extract_step_limit(env) or getattr(env_args, "max_episode_steps", 1000) or 1000)
    episode_paths: list[Path] = []
    episodes: list[dict[str, Any]] = []
    scale_to_env = bool(train_args.get("scale_actor_to_env_bounds", True))
    for ep in range(int(num_episodes)):
        obs, _ = env.reset(seed=int(seed + ep))
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        task = env.unwrapped.task
        state_template = task.world.get_state()
        agent_z = float(state_template["qpos"][2])
        overlay_specs = _extract_overlay_specs(task)
        goal_xy = np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
        path = [np.asarray(task.agent.pos[:2], dtype=np.float64).copy()]
        ep_ret = 0.0
        ep_cost = 0.0
        first_goal_step = None
        for step in range(max_steps):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=device)
                obs_t = obs_preprocess(obs_t)
                _, _, mean_action = actor(obs_t)
                action = mean_action[0].detach().cpu().numpy().astype(np.float32)
            if scale_to_env:
                action = scale_action_np(action, env.action_space)
            action = clip_action_to_space(action, env.action_space)
            obs, reward, cost, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            path.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
            ep_ret += float(reward)
            ep_cost += float(cost)
            if first_goal_step is None and bool(info.get("goal_met", False)):
                first_goal_step = step + 1
            if terminated or truncated:
                break
        final_dist = float(extract_goal_distance(env))
        ep_path = output_dir / label / f"episode_{ep + 1:02d}_reward_surface_overlay.png"
        stats = _plot_episode_overlay(
            output_path=ep_path,
            env=env,
            task=task,
            state_template=state_template,
            agent_z=agent_z,
            overlay_specs=overlay_specs,
            bounds=bounds,
            path=np.stack(path, axis=0),
            goal_xy=goal_xy,
            train_args=train_args,
            title=(
                f"{label} ep {ep + 1}: return={ep_ret:.2f} cost={ep_cost:.1f} "
                f"first_goal={first_goal_step if first_goal_step is not None else 'none'} final_dist={final_dist:.2f}"
            ),
            grid_resolution=grid_resolution,
        )
        episode_paths.append(ep_path)
        episodes.append(
            {
                "episode": ep + 1,
                "return": float(ep_ret),
                "cost_sum": float(ep_cost),
                "first_goal_step": first_goal_step,
                "final_distance": final_dist,
                **stats,
            }
        )
    contact = output_dir / f"{label}_reward_surface_contact_sheet.png"
    _make_contact_sheet(episode_paths, contact, f"{label}: trajectories over reward surface")
    env.close()
    return {"label": label, "model_path": str(model_path), "contact_sheet": str(contact), "episodes": episodes}


def _eval_ppo(
    *,
    model_path: Path,
    output_dir: Path,
    label: str,
    seed: int,
    num_episodes: int,
    grid_resolution: int,
) -> dict[str, Any]:
    train_args = _load_json_args(model_path.parent / "args.json")
    env_args = _namespace_with_defaults(train_args)
    raw_env = _make_ppo_single_env(env_args, seed=int(seed))
    task = raw_env.unwrapped.task
    bounds = _extract_bounds(task, x_range=None, y_range=None)
    max_steps = int(extract_step_limit(raw_env) or getattr(env_args, "max_episode_steps", 1000) or 1000)
    venv = DummyVecEnv([lambda: raw_env])
    vecnorm_path = model_path.parent / "vecnormalize.pkl"
    if vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
    model = PPO.load(str(model_path), env=venv)

    episode_paths: list[Path] = []
    episodes: list[dict[str, Any]] = []
    for ep in range(int(num_episodes)):
        obs = venv.reset()
        task = raw_env.unwrapped.task
        state_template = task.world.get_state()
        agent_z = float(state_template["qpos"][2])
        overlay_specs = _extract_overlay_specs(task)
        goal_xy = np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
        path = []
        ep_ret = 0.0
        ep_cost = 0.0
        first_goal_step = None
        for step in range(max_steps):
            agent_xy = extract_agent_xy(raw_env)
            if agent_xy is not None:
                path.append(np.asarray(agent_xy, dtype=np.float64).reshape(2))
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            info = dict(infos[0])
            ep_ret += float(rewards[0])
            ep_cost += float(info.get("cost", 0.0))
            if first_goal_step is None and bool(info.get("goal_met", False)):
                first_goal_step = step + 1
            if bool(dones[0]):
                break
        agent_xy = extract_agent_xy(raw_env)
        if agent_xy is not None:
            path.append(np.asarray(agent_xy, dtype=np.float64).reshape(2))
        final_dist = float(extract_goal_distance(raw_env))
        ep_path = output_dir / label / f"episode_{ep + 1:02d}_reward_surface_overlay.png"
        stats = _plot_episode_overlay(
            output_path=ep_path,
            env=raw_env,
            task=task,
            state_template=state_template,
            agent_z=agent_z,
            overlay_specs=overlay_specs,
            bounds=bounds,
            path=np.stack(path, axis=0),
            goal_xy=goal_xy,
            train_args=train_args,
            title=(
                f"{label} ep {ep + 1}: return={ep_ret:.2f} cost={ep_cost:.1f} "
                f"first_goal={first_goal_step if first_goal_step is not None else 'none'} final_dist={final_dist:.2f}"
            ),
            grid_resolution=grid_resolution,
        )
        episode_paths.append(ep_path)
        episodes.append(
            {
                "episode": ep + 1,
                "return": float(ep_ret),
                "cost_sum": float(ep_cost),
                "first_goal_step": first_goal_step,
                "final_distance": final_dist,
                **stats,
            }
        )
    contact = output_dir / f"{label}_reward_surface_contact_sheet.png"
    _make_contact_sheet(episode_paths, contact, f"{label}: trajectories over reward surface")
    venv.close()
    return {"label": label, "model_path": str(model_path), "contact_sheet": str(contact), "episodes": episodes}


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in str(value):
        path = Path(value).resolve()
        return path.stem, path
    label, path_s = str(value).split("=", 1)
    return label.strip(), Path(path_s).resolve()


def _plot_easy_reference(output_dir: Path, grid_resolution: int, seed: int, env_name: str) -> Path:
    cfg = _namespace_with_defaults({"env_name": str(env_name)})
    env = _make_env_with_wrappers(args=cfg, seed=int(seed), with_intervention=False, controller=None, render_mode_override="none")
    env.reset(seed=int(seed))
    task = env.unwrapped.task
    state_template = task.world.get_state()
    path = np.asarray([task.agent.pos[:2]], dtype=np.float64)
    safe_env = str(env_name).replace("-", "_")
    output_path = output_dir / f"{safe_env}_reward_surface_reference.png"
    _plot_episode_overlay(
        output_path=output_path,
        env=env,
        task=task,
        state_template=state_template,
        agent_z=float(state_template["qpos"][2]),
        overlay_specs=_extract_overlay_specs(task),
        bounds=_extract_bounds(task, x_range=None, y_range=None),
        path=path,
        goal_xy=np.asarray(task.goal.pos[:2], dtype=np.float64).copy(),
        train_args=vars(cfg),
        title=f"{env_name} reference: same softplus clearance reward surface",
        grid_resolution=grid_resolution,
    )
    env.close()
    return output_path


def main() -> int:
    p = argparse.ArgumentParser(description="Plot Safety-Gym trajectories on top of the shaped reward surface diagnostic.")
    p.add_argument("--output_dir", type=Path, default=Path("logs") / "safetygym_softplus_comparison_20260511" / "reward_surface_trajectory_overlay")
    p.add_argument("--num_episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--grid_resolution", type=int, default=70)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip_default_models", action="store_true")
    p.add_argument("--ppo_model", action="append", default=[], help="Optional label=path PPO model. Can be repeated.")
    p.add_argument("--fastsac_model", action="append", default=[], help="Optional label=path FastSAC checkpoint. Can be repeated.")
    p.add_argument("--reference_env", type=str, default="SafetyCarGoal1-v0")
    args = p.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results: list[dict[str, Any]] = []
    custom_models = bool(args.ppo_model or args.fastsac_model)
    if not args.skip_default_models and not custom_models:
        results.append(
            _eval_ppo(
                model_path=PROJECT_ROOT / "models/safetygym_ppo/safetycar_goal2_softplus_m000_s11_ppo_20260511/final.zip",
                output_dir=output_dir,
                label="ppo_final_200k",
                seed=int(args.seed),
                num_episodes=int(args.num_episodes),
                grid_resolution=int(args.grid_resolution),
            )
        )
        results.append(
            _eval_fastsac(
                model_path=PROJECT_ROOT / "models/safetygym_minimal/safetycar_goal2_softplus_m000_s11_fastsac_20260511/step_40000.pt",
                output_dir=output_dir,
                label="fastsac_original_40k",
                seed=int(args.seed),
                num_episodes=int(args.num_episodes),
                grid_resolution=int(args.grid_resolution),
                device=device,
            )
        )
        results.append(
            _eval_fastsac(
                model_path=PROJECT_ROOT / "models/safetygym_minimal/safetycar_goal2_softplus_m000_s11_fastsac_resume40k_20260511/step_30000.pt",
                output_dir=output_dir,
                label="fastsac_resume_plus30k",
                seed=int(args.seed),
                num_episodes=int(args.num_episodes),
                grid_resolution=int(args.grid_resolution),
                device=device,
            )
        )
    for item in args.ppo_model:
        label, path = _parse_label_path(item)
        results.append(
            _eval_ppo(
                model_path=path,
                output_dir=output_dir,
                label=label,
                seed=int(args.seed),
                num_episodes=int(args.num_episodes),
                grid_resolution=int(args.grid_resolution),
            )
        )
    for item in args.fastsac_model:
        label, path = _parse_label_path(item)
        results.append(
            _eval_fastsac(
                model_path=path,
                output_dir=output_dir,
                label=label,
                seed=int(args.seed),
                num_episodes=int(args.num_episodes),
                grid_resolution=int(args.grid_resolution),
                device=device,
            )
        )
    easy_ref = _plot_easy_reference(
        output_dir=output_dir,
        grid_resolution=int(args.grid_resolution),
        seed=int(args.seed),
        env_name=str(args.reference_env),
    )
    summary = {"results": results, "easy_reference": str(easy_ref)}
    summary_path = output_dir / "reward_surface_overlay_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved overlay summary to {summary_path}")
    for item in results:
        print(f"{item['label']}: {item['contact_sheet']}")
    print(f"goal1_easy_reference: {easy_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
