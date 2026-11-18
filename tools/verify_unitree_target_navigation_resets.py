from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_unitree_nav_baselines import (
    _environment_origins_xy,
    _goal_positions_xy,
    _resample_close_goals,
    _robot_positions_xy,
    _terrain_obstacle_cells_by_env,
)
from train_unitree_nav_thesis import make_env
from unitree_target_navigation import (
    _point_clear_of_features,
    _point_clear_of_obstacles,
    _segment_feature_labels,
    update_target_goal_curriculum,
)
from unitree_target_terrain import generate_target_terrain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-args", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--resets", type=int, default=3)
    parser.add_argument("--terrain-seed", type=int, default=3)
    args = parser.parse_args()

    config = json.loads(args.base_args.read_text())
    config.update(
        num_envs=args.num_envs,
        seed=args.terrain_seed,
        target_terrain_seed=args.terrain_seed,
        target_terrain_start_boundary_margin=1.25,
        target_terrain_start_clearance=0.75,
        target_terrain_start_attempts=256,
        target_terrain_goal_feature_fraction=1.0,
        target_terrain_goal_curriculum=True,
        target_terrain_goal_curriculum_steps=20_000,
        target_terrain_goal_curriculum_start_min=1.5,
        target_terrain_goal_curriculum_start_max=3.0,
        target_terrain_goal_curriculum_start_feature_fraction=0.0,
    )
    run_args = argparse.Namespace(**config)
    update_target_goal_curriculum(run_args, 0)
    layout = generate_target_terrain(
        args.terrain_seed,
        preset=run_args.target_terrain_preset,
        arena_size=run_args.target_terrain_arena_size,
    )
    start_half = run_args.target_terrain_arena_size / 2.0 - run_args.target_terrain_start_boundary_margin
    goal_half = run_args.target_terrain_arena_size / 2.0 - run_args.target_terrain_goal_boundary_margin

    env = make_env(run_args, render=False)
    samples: list[dict[str, object]] = []
    try:
        for reset_index in range(args.resets):
            curriculum_step = 0 if reset_index < max(1, args.resets // 2) else 20_000
            update_target_goal_curriculum(run_args, curriculum_step)
            stage = "close" if curriculum_step == 0 else "far"
            env.reset()
            _resample_close_goals(run_args, env, _terrain_obstacle_cells_by_env(env))
            starts = _robot_positions_xy(env)
            goals = _goal_positions_xy(env)
            origins = _environment_origins_xy(env)
            for env_index, (start_world, goal_world, origin) in enumerate(zip(starts, goals, origins)):
                start = start_world - origin
                goal = goal_world - origin
                distance = float(np.linalg.norm(goal - start))
                features = _segment_feature_labels(start, goal, layout)
                start_valid = _point_clear_of_features(
                    start, layout, run_args.target_terrain_start_clearance, start_half
                )
                goal_valid = _point_clear_of_obstacles(
                    goal, layout, run_args.target_terrain_goal_clearance, goal_half
                )
                samples.append(
                    {
                        "reset": reset_index,
                        "stage": stage,
                        "env": env_index,
                        "origin_xy": origin.tolist(),
                        "start_local_xy": start.tolist(),
                        "goal_local_xy": goal.tolist(),
                        "distance_m": distance,
                        "route_features": list(features),
                        "start_valid": bool(start_valid),
                        "goal_valid": bool(goal_valid),
                    }
                )
                if not start_valid or not goal_valid:
                    raise AssertionError(f"Invalid target reset sample: {samples[-1]}")
                min_distance = 1.5 if stage == "close" else float(run_args.goal_distance_min)
                max_distance = 3.0 if stage == "close" else float(run_args.goal_distance_max)
                if not min_distance - 1e-3 <= distance <= max_distance + 1e-3:
                    raise AssertionError(f"Goal distance outside curriculum stage: {samples[-1]}")
                if stage == "far" and not features:
                    raise AssertionError(f"Feature-free route despite required fraction 1: {samples[-1]}")
    finally:
        env.close()

    starts = np.asarray([sample["start_local_xy"] for sample in samples])
    result = {
        "sample_count": len(samples),
        "unique_env_origins": len({tuple(sample["origin_xy"]) for sample in samples}),
        "start_span_x_m": float(np.ptp(starts[:, 0])),
        "start_span_y_m": float(np.ptp(starts[:, 1])),
        "far_routes_all_feature_crossing": all(
            bool(sample["route_features"]) for sample in samples if sample["stage"] == "far"
        ),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
