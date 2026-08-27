from __future__ import annotations

import argparse
from pathlib import Path

import torch


UNITREE_EVAL_CONFIG_KEYS = (
    "task",
    "target_terrain",
    "target_terrain_preset",
    "target_terrain_seed",
    "target_terrain_arena_size",
    "target_terrain_material_resolution",
    "low_level_policy_path",
    "episode_length_s",
    "navigation_episode_mode",
    "continuous_environment_horizon_s",
    "continuous_goal_distance_min",
    "continuous_goal_distance_max",
    "continuous_goal_region_mode",
    "continuous_goal_resample_attempts",
    "continuous_goal_boundary_margin",
    "continuous_goal_require_blocked_corridor",
    "continuous_goal_blocked_probability",
    "success_dist",
    "terminate_on_goal",
    "height_scan_resolution",
    "height_scan_pattern",
    "height_scan_frustum_near",
    "height_scan_frustum_far",
    "height_scan_frustum_fov_deg",
    "height_scan_frustum_side",
    "height_scan_forward_size",
    "height_scan_lateral_size",
    "scan_history",
    "scan_history_stride",
    "action_history",
    "student_action_scale",
    "mask_height_scan",
    "mask_proprioception",
    "mask_goal_heading",
    "goal_encoding",
    "goal_distance_scale",
    "velocity_scale",
    "goal_distance_min",
    "goal_distance_max",
    "min_goal_obstacle_clearance",
    "goal_clearance_resample_attempts",
    "min_start_obstacle_clearance",
    "start_position_range",
    "start_clearance_resample_attempts",
    "require_blocked_corridor",
    "blocked_corridor_radius",
    "blocked_corridor_ignore_end_radius",
    "blocked_corridor_min_cells",
    "blocked_corridor_resample_attempts",
    "blocked_goal_max_distance",
    "blocked_goal_distance_sampling",
    "blocked_goal_placement_mode",
    "blocked_goal_distance_multiplier_min",
    "blocked_goal_distance_multiplier_max",
    "blocked_goal_candidate_attempts",
    "debug_goal_through_obstacle",
    "goal_through_obstacle_prob",
    "debug_goal_distance",
    "debug_goal_obstacle_min_dist",
    "debug_goal_obstacle_max_dist",
    "debug_obstacle_width_min",
    "debug_obstacle_width_max",
    "strict_min_size_obstacles",
    "debug_obstacle_height_min",
    "debug_obstacle_height_max",
    "debug_num_obstacles",
    "disable_obstacles",
    "debug_platform_width",
    "debug_obstacle_border_width",
    "debug_terrain_rows",
    "debug_terrain_cols",
    "resample_terrain_tiles",
    "hidden_dim",
    "use_layer_norm",
    "policy_encoder",
)


def apply_unitree_checkpoint_config(args: argparse.Namespace) -> dict:
    """Make a policy checkpoint's environment manifest authoritative for eval."""
    model_path = str(getattr(args, "model_path", "") or "")
    enabled = bool(getattr(args, "checkpoint_env_config", True))
    if not enabled or not model_path or str(getattr(args, "controller", "")) != "policy":
        return {}
    checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=False)
    saved = dict(checkpoint.get("args", {}))
    applied: dict[str, object] = {}
    for key in UNITREE_EVAL_CONFIG_KEYS:
        if key in saved:
            setattr(args, key, saved[key])
            applied[key] = saved[key]
    if "action_history" not in applied:
        args.action_history = 0
    if "student_action_smoothing" in saved and hasattr(args, "policy_action_smoothing"):
        args.policy_action_smoothing = saved["student_action_smoothing"]
        applied["policy_action_smoothing"] = saved["student_action_smoothing"]
    print(
        "[checkpoint-config] restored policy environment manifest "
        f"from {model_path}: scan={getattr(args, 'height_scan_resolution', None)} "
        f"history={getattr(args, 'scan_history', None)} action_history={getattr(args, 'action_history', 0)} "
        f"obstacle_width={getattr(args, 'debug_obstacle_width_min', None)}-"
        f"{getattr(args, 'debug_obstacle_width_max', None)} success_dist={getattr(args, 'success_dist', None)} "
        f"action_smoothing={getattr(args, 'policy_action_smoothing', None)}",
        flush=True,
    )
    return applied
