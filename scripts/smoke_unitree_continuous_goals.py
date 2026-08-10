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
    _goal_clearances,
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
        "2.0",
        "--continuous-goal-distance-max",
        "5.0",
        "--min-goal-obstacle-clearance",
        "1.0",
    ]
    args = parse_args()
    _apply_strict_blocked_obstacle_profile(args)
    args.navigation_episode_mode = "continuous_goals"
    args.continuous_goal_distance_min = 2.0
    args.continuous_goal_distance_max = 5.0
    args.min_goal_obstacle_clearance = 1.0
    env = make_env(args, num_envs=1, render=False)
    try:
        _, _, initial_clearance, _, obstacle_cells = _reset_until_feasible(args, env)
        if float(initial_clearance[0].item()) < 1.0:
            raise AssertionError(f"initial goal clearance={float(initial_clearance[0].item()):.4f}")
        root_before = _robot_positions_xy(env).copy()
        sampled = []
        for _ in range(20):
            _, clearance, _ = _resample_continuous_goals(
                args,
                env,
                obstacle_cells,
                torch.zeros(1, dtype=torch.long, device=args.device),
            )
            sampled.append(float(clearance[0].item()))
            if not np.allclose(root_before, _robot_positions_xy(env), atol=1e-6):
                raise AssertionError("continuous goal resampling changed humanoid root position")
        print(
            f"continuous-goal smoke OK: samples={len(sampled)} "
            f"min_clearance={min(sampled):.4f}m root_unchanged=1",
            flush=True,
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
