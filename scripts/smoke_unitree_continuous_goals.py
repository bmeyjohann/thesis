#!/usr/bin/env python3
"""Live invariant probe for Unitree continuous-goal placement."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_interactive_unitree_nav import _apply_strict_blocked_obstacle_profile  # noqa: E402
from eval_unitree_nav_baselines import (  # noqa: E402
    _goal_positions_xy,
    _reset_until_feasible,
    _resample_continuous_goals,
    _robot_positions_xy,
    make_env,
    parse_args,
)


def main() -> int:
    sys.argv = [
        sys.argv[0],
        "--controller",
        "direct_goal",
        "--device",
        "cuda:0",
        "--num-envs",
        "1",
        "--num-episodes",
        "1",
        "--navigation-episode-mode",
        "continuous_goals",
        "--continuous-goal-distance-min",
        "8.0",
        "--continuous-goal-distance-max",
        "14.0",
        "--continuous-goal-region-mode",
        "terrain_bank",
        "--continuous-goal-require-blocked-corridor",
        "--continuous-goal-blocked-probability",
        "1.0",
        "--min-goal-obstacle-clearance",
        "1.0",
    ]
    args = parse_args()
    _apply_strict_blocked_obstacle_profile(args)
    args.navigation_episode_mode = "continuous_goals"
    args.continuous_goal_distance_min = 8.0
    args.continuous_goal_distance_max = 14.0
    args.continuous_goal_region_mode = "terrain_bank"
    args.continuous_goal_require_blocked_corridor = True
    args.continuous_goal_blocked_probability = 1.0
    args.continuous_goal_resample_attempts = 1024
    args.min_goal_obstacle_clearance = 1.0
    env = make_env(args, num_envs=1, render=False)
    try:
        initial_clearances = []
        for _ in range(10):
            _, _, initial_clearance, _, obstacle_cells = _reset_until_feasible(args, env)
            initial_clearances.append(float(initial_clearance[0].item()))
            if initial_clearances[-1] < 1.0:
                raise AssertionError(f"initial goal clearance={initial_clearances[-1]:.4f}")
        root_before = _robot_positions_xy(env).copy()
        terrain_origins = env.env.unwrapped.scene.terrain.terrain_origins.detach().cpu().numpy().reshape(-1, 3)[:, :2]
        sampled: list[float] = []
        blocked: list[bool] = []
        goal_tiles: list[int] = []
        for _ in range(20):
            _, clearance, stats = _resample_continuous_goals(
                args,
                env,
                obstacle_cells,
                torch.zeros(1, dtype=torch.long, device=args.device),
            )
            sampled.append(float(clearance[0].item()))
            blocked.append(bool(stats[0]["blocked"]))
            goal_xy = _goal_positions_xy(env)[0]
            goal_tiles.append(int(np.argmin(np.linalg.norm(terrain_origins - goal_xy.reshape(1, 2), axis=1))))
            if not np.allclose(root_before, _robot_positions_xy(env), atol=1e-6):
                raise AssertionError("continuous goal resampling changed humanoid root position")
        if not all(blocked):
            raise AssertionError(f"blocked corridor rate={np.mean(blocked):.3f}, expected 1.0")
        if len(set(goal_tiles)) < 2:
            raise AssertionError(f"continuous goals remained on one tile: {goal_tiles}")
        print(
            f"continuous-goal smoke OK: initial_layouts={len(initial_clearances)} "
            f"initial_min_clearance={min(initial_clearances):.4f}m samples={len(sampled)} "
            f"continuous_min_clearance={min(sampled):.4f}m blocked_rate={np.mean(blocked):.3f} "
            f"unique_goal_tiles={len(set(goal_tiles))} root_unchanged=1",
            flush=True,
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
