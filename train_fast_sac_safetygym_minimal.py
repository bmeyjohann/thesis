#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from safetygym_utils.minimal_train import run_minimal_training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal plain FastSAC training for Safety-Gymnasium")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--variant", type=str, default="plain", choices=["plain", "own", "pvp", "eil", "hilserl"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--torch_num_threads", type=int, default=1)
    p.add_argument("--torch_num_interop_threads", type=int, default=1)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--learning_starts", type=int, default=5_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_updates", type=int, default=2)
    p.add_argument("--policy_frequency", type=int, default=2)
    p.add_argument(
        "--actor_update_start_step",
        type=int,
        default=0,
        help="If >0, suppress actor/alpha updates until this env step so critic/preference learning can warm up.",
    )
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor_learning_rate", type=float, default=3e-4)
    p.add_argument("--critic_learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--actor_hidden_dim", type=int, default=512)
    p.add_argument("--critic_hidden_dim", type=int, default=1024)
    p.add_argument("--module_impl", type=str, default="fastsac", choices=["fastsac", "custom"])
    p.add_argument("--use_layer_norm", action="store_true", default=False)
    p.add_argument("--layer_norm_eps", type=float, default=1e-5)
    p.add_argument("--temporal_encoder", type=str, default="none", choices=["none", "attention"])
    p.add_argument("--init_scale", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--alpha_init", type=float, default=1e-3)
    p.add_argument("--alpha_min", type=float, default=0.0)
    p.add_argument("--alpha_max", type=float, default=1.0)
    p.add_argument("--critic_loss_reduction", type=str, default="sum", choices=["sum", "mean"])
    p.add_argument("--actor_bc_weight", type=float, default=0.0)
    p.add_argument("--actor_bc_reward_weight_scale", type=float, default=0.0)
    p.add_argument("--actor_bc_reward_weight_max", type=float, default=10.0)
    p.add_argument("--actor_bc_obstacle_lidar_weight_scale", type=float, default=0.0)
    p.add_argument("--actor_bc_obstacle_lidar_weight_max", type=float, default=10.0)
    p.add_argument("--actor_bc_goal_block_weight_scale", type=float, default=0.0)
    p.add_argument("--actor_bc_goal_block_weight_max", type=float, default=10.0)
    p.add_argument("--actor_bc_only_until_step", type=int, default=0)
    p.add_argument("--actor_bc_only_pretrain_updates", type=int, default=0)
    p.add_argument("--actor_bc_teacher_only", action="store_true", default=True)
    p.add_argument("--no_actor_bc_teacher_only", dest="actor_bc_teacher_only", action="store_false")
    p.add_argument(
        "--actor_reference_distill_weight",
        type=float,
        default=0.0,
        help="MSE penalty keeping actor mean close to a frozen copy after checkpoint load.",
    )
    p.add_argument("--intervention_aux_head", action="store_true", default=False)
    p.add_argument("--intervention_aux_weight", type=float, default=0.0)
    p.add_argument("--intervention_aux_pos_weight", type=float, default=0.0)
    p.add_argument("--intervention_aux_pretrain_updates", type=int, default=0)
    p.add_argument("--intervention_aux_pretrain_batch_size", type=int, default=0)
    p.add_argument("--intervention_aux_pretrain_distill_weight", type=float, default=1.0)
    p.add_argument("--intervention_aux_pretrain_lr", type=float, default=0.0)
    p.add_argument("--obs_normalization", action="store_true", default=True)
    p.add_argument("--no_obs_normalization", dest="obs_normalization", action="store_false")
    p.add_argument(
        "--freeze_obs_normalizer_after_load",
        action="store_true",
        default=False,
        help="Keep loaded/fitted observation normalization fixed during online collection and updates.",
    )

    p.add_argument(
        "--reward_mode",
        type=str,
        default="dense",
        choices=["sparse", "dense", "dense_plus_sparse", "potential_diff", "dual", "native", "none"],
    )
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--success_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--cost_penalty", type=float, default=0.0)
    p.add_argument("--cost_penalty_warmup_steps", type=int, default=0)
    p.add_argument("--cost_penalty_ramp_steps", type=int, default=0)
    p.add_argument("--footprint_cost", action="store_true", default=False)
    p.add_argument("--footprint_cost_mode", type=str, default="visual", choices=["visual", "keepout"])
    p.add_argument("--footprint_cost_margin", type=float, default=0.0)
    p.add_argument("--footprint_cost_value", type=float, default=1.0)
    p.add_argument("--clearance_penalty_scale", type=float, default=0.0)
    p.add_argument("--clearance_margin", type=float, default=0.0)
    p.add_argument("--clearance_penalty_power", type=float, default=1.0)
    p.add_argument("--clearance_penalty_mode", type=str, default="hinge_power", choices=["hinge_power", "softplus"])
    p.add_argument("--clearance_penalty_temperature", type=float, default=0.08)
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
    p.add_argument("--obs_frame_stack", type=int, default=1)
    p.add_argument("--fixed_layout_preset", type=str, default="none")
    p.add_argument("--layout_curriculum", type=str, default="none")
    p.add_argument("--layout_curriculum_level", type=int, default=0)
    p.add_argument("--layout_seed_replay", type=str, default="")
    p.add_argument("--layout_seed_replay_prob", type=float, default=0.0)
    p.add_argument("--layout_seed_replay_mode", type=str, default="cycle", choices=["cycle", "random"])
    p.add_argument("--scale_actor_to_env_bounds", action="store_true", default=True)
    p.add_argument("--no_scale_actor_to_env_bounds", dest="scale_actor_to_env_bounds", action="store_false")
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--terminate_on_goal", action="store_true", default=False)
    p.add_argument("--terminate_on_cost", action="store_true", default=False)
    p.add_argument("--reseed_on_episode_reset", action="store_true", default=False)
    p.add_argument("--no_reseed_on_episode_reset", dest="reseed_on_episode_reset", action="store_false")
    p.add_argument("--use_intervention", action="store_true", default=False)
    p.add_argument("--intervention_threshold", type=float, default=0.1)
    p.add_argument("--intervention_hold_seconds", type=float, default=0.25)
    p.add_argument("--debug_intervention_console", action="store_true", default=False)
    p.add_argument("--teacher_override_clearance_threshold", type=float, default=-1.0)
    p.add_argument("--teacher_override_clearance_exit_threshold", type=float, default=-1.0)
    p.add_argument("--teacher_clearance_source", type=str, default="keepout", choices=["keepout", "visual", "footprint", "visual_footprint", "footprint_cost"])
    p.add_argument(
        "--teacher_override_mode",
        type=str,
        default="clearance",
        choices=[
            "clearance",
            "clearance_projected_release",
            "clearance_projected_release_or_progress",
            "teacher_goal_progress",
            "student_forward_clearance",
            "student_projected_clearance",
            "reward_progress",
            "clearance_or_progress",
            "pcpo_value_progress",
            "pcpo_cost_value_progress",
        ],
    )
    p.add_argument("--teacher_goal_progress_steps", type=int, default=3)
    p.add_argument("--teacher_goal_progress_epsilon", type=float, default=1e-3)
    p.add_argument("--teacher_progress_bad_steps", type=int, default=3)
    p.add_argument("--teacher_progress_good_steps", type=int, default=5)
    p.add_argument("--teacher_progress_epsilon", type=float, default=1e-4)
    p.add_argument(
        "--teacher_progress_trigger_mode",
        type=str,
        default="worse",
        choices=["worse", "not_improving", "no_progress", "not_progressing", "no-improve"],
    )
    p.add_argument(
        "--teacher_progress_release_mode",
        type=str,
        default="improve",
        choices=["improve", "non_worse", "not_worse", "not-worse", "stable", "plateau"],
    )
    p.add_argument(
        "--teacher_progress_score_mode",
        type=str,
        default="reward_wrapper",
        choices=["reward_wrapper", "euclidean", "potential_field", "reward", "wrapper", "potential", "clearance", "clearance_potential"],
    )
    p.add_argument("--teacher_progress_dense_scale", type=float, default=1.0)
    p.add_argument("--teacher_progress_clearance_scale", type=float, default=-1.0)
    p.add_argument("--teacher_progress_clearance_margin", type=float, default=0.0)
    p.add_argument(
        "--teacher_progress_clearance_mode",
        type=str,
        default="softplus",
        choices=["softplus", "hinge", "quadratic_hinge", "exp_soft", "hinge_power"],
    )
    p.add_argument("--teacher_progress_clearance_temperature", type=float, default=0.001)
    p.add_argument("--human_action_scale", type=float, default=1.0)
    p.add_argument(
        "--human_input_device",
        type=str,
        default="keyboard",
        choices=[
            "keyboard",
            "gamepad",
            "expert",
            "expert_switch",
            "safe_rl",
            "scripted",
            "scripted_goal_geom",
            "scripted_geo",
            "scripted_geo_legacy",
            "scripted_visual_mpc",
            "heading_bc",
            "learned",
            "learned_intervention",
            "imitation",
        ],
    )
    p.add_argument("--controller_fps_limit", type=int, default=0)
    p.add_argument("--controller_overlay_hz", type=float, default=20.0)
    p.add_argument("--env_fps_limit", type=float, default=0.0)
    p.add_argument("--scripted_geo_heading_tolerance", type=float, default=0.20)
    p.add_argument("--scripted_geo_lookahead", type=float, default=1.0)
    p.add_argument("--scripted_geo_safety_margin", type=float, default=0.18)
    p.add_argument("--scripted_geo_grid_resolution", type=float, default=0.08)
    p.add_argument("--scripted_geo_emergency_clearance", type=float, default=0.08)
    p.add_argument("--scripted_geo_action_shield_steps", type=int, default=1)
    p.add_argument("--gamepad_mode", type=str, default="local", choices=["local", "connect"])
    p.add_argument("--gamepad_host", type=str, default="")
    p.add_argument("--gamepad_port", type=int, default=0)
    p.add_argument("--gamepad_cache_path", type=str, default="")
    p.add_argument("--gamepad_reconnect_seconds", type=float, default=2.0)
    p.add_argument("--gamepad_config_path", type=str, default="")
    p.add_argument("--gamepad_use_saved_config", action="store_true", default=True)
    p.add_argument("--no_gamepad_use_saved_config", dest="gamepad_use_saved_config", action="store_false")
    p.add_argument("--gamepad_device_index", type=int, default=0)
    p.add_argument("--expert_checkpoint_path", type=str, default="")
    p.add_argument("--expert_config_path", type=str, default="")
    p.add_argument("--expert_safe_checkpoint_path", type=str, default="")
    p.add_argument("--expert_switch_clearance_threshold", type=float, default=0.08)
    p.add_argument("--learned_intervention_threshold", type=float, default=0.5)
    p.add_argument("--expert_device", type=str, default="cpu")

    p.add_argument("--pref_capacity", type=int, default=100_000)
    p.add_argument("--pref_sampling_mode", type=str, default="linked", choices=["linked", "separate"])
    p.add_argument("--pref_sample_ratio", type=float, default=0.0)
    p.add_argument("--pref_replay_sample_ratio", type=float, default=0.0)
    p.add_argument("--pref_rank_weight", type=float, default=0.0)
    p.add_argument("--pref_rank_margin", type=float, default=0.1)
    p.add_argument("--pref_loss_type", type=str, default="margin", choices=["margin", "bradley_terry", "lagrangian"])
    p.add_argument("--pref_lambda_init", type=float, default=1.0)
    p.add_argument("--pref_lambda_lr", type=float, default=1e-3)
    p.add_argument("--pref_lambda_max", type=float, default=10.0)
    p.add_argument("--pref_lambda_ema", type=float, default=0.9)
    p.add_argument("--pref_violation_clip", type=float, default=10.0)
    p.add_argument("--pref_violation_target", type=float, default=0.0)
    p.add_argument("--pref_lagrangian_violation_type", type=str, default="hinge", choices=["hinge", "smooth"])
    p.add_argument("--pref_stopgrad_positive", action="store_true", default=False)
    p.add_argument("--pref_obs_noise_std", type=float, default=0.0)
    p.add_argument("--pref_action_noise_std", type=float, default=0.0)
    p.add_argument("--pref_action_noise_copies", type=int, default=1)
    p.add_argument("--pref_action_delta_min", type=float, default=0.0)
    p.add_argument("--pref_action_delta_weight_scale", type=float, default=0.0)
    p.add_argument("--pref_action_delta_weight_max", type=float, default=10.0)
    p.add_argument("--pvp_proxy_value_bound", type=float, default=1.0)
    p.add_argument("--pvp_include_env_reward_in_td", action="store_true", default=False)
    p.add_argument("--eil_threshold", type=float, default=0.0)
    p.add_argument("--eil_good_margin", type=float, default=0.0)
    p.add_argument("--eil_bad_margin", type=float, default=0.01)
    p.add_argument("--eil_pair_margin", type=float, default=0.01)
    p.add_argument("--eil_bad_pre_steps", type=int, default=8)
    p.add_argument("--demo_sample_ratio", type=float, default=0.0)
    p.add_argument("--prefill_demo_episodes", type=int, default=0)
    p.add_argument("--prefill_max_steps_per_episode", type=int, default=0)
    p.add_argument("--prefill_policy", type=str, default="student", choices=["student", "random", "zero"])
    p.add_argument("--store_intervened_in_demo_buffer", action="store_true", default=False)
    p.add_argument("--demo_dataset_path", type=str, default="")
    p.add_argument("--demo_dataset_dir", type=str, default="")
    p.add_argument("--demo_dataset_auto_load", action="store_true", default=False)
    p.add_argument("--demo_dataset_target", type=str, default="variant", choices=["variant", "replay", "demo"])
    p.add_argument("--demo_dataset_max_rows", type=int, default=0)
    p.add_argument(
        "--demo_dataset_clear_teacher_flags_in_replay",
        action="store_true",
        default=False,
        help="Load dataset rows into replay as non-intervention rows so linked prefs only come from online interventions.",
    )
    p.add_argument("--demo_pretrain_updates", type=int, default=0)
    p.add_argument("--demo_pretrain_batch_size", type=int, default=0)
    p.add_argument("--critic_reset_after_pretrain", action="store_true", default=False)
    p.add_argument("--init_checkpoint_path", type=str, default="")
    p.add_argument("--load_actor_from_checkpoint", action="store_true", default=True)
    p.add_argument("--no_load_actor_from_checkpoint", dest="load_actor_from_checkpoint", action="store_false")
    p.add_argument("--load_critic_from_checkpoint", action="store_true", default=True)
    p.add_argument("--no_load_critic_from_checkpoint", dest="load_critic_from_checkpoint", action="store_false")
    p.add_argument("--load_critic_target_from_checkpoint", action="store_true", default=True)
    p.add_argument("--no_load_critic_target_from_checkpoint", dest="load_critic_target_from_checkpoint", action="store_false")
    p.add_argument("--load_alpha_from_checkpoint", action="store_true", default=True)
    p.add_argument("--no_load_alpha_from_checkpoint", dest="load_alpha_from_checkpoint", action="store_false")
    p.add_argument("--load_obs_normalizer_from_checkpoint", action="store_true", default=True)
    p.add_argument(
        "--no_load_obs_normalizer_from_checkpoint",
        dest="load_obs_normalizer_from_checkpoint",
        action="store_false",
    )
    p.add_argument("--load_optimizer_state_from_checkpoint", action="store_true", default=False)
    p.add_argument("--save_optimizer_state_in_checkpoints", action="store_true", default=True)
    p.add_argument(
        "--no_save_optimizer_state_in_checkpoints",
        dest="save_optimizer_state_in_checkpoints",
        action="store_false",
    )
    p.add_argument("--export_dataset_dir", type=str, default="")
    p.add_argument("--export_dataset_max_rows", type=int, default=0)
    p.add_argument("--offline_only", action="store_true", default=False)
    p.add_argument("--export_replay_dataset_interval", type=int, default=0)
    p.add_argument("--export_replay_dataset_path", type=str, default="")
    p.add_argument("--export_replay_dataset_dir", type=str, default="")
    p.add_argument("--export_replay_dataset_label", type=str, default="online_replay")
    p.add_argument("--export_replay_dataset_max_rows", type=int, default=0)
    p.add_argument("--export_final_replay_dataset", action="store_true", default=False)
    p.add_argument("--export_final_replay_dataset_path", type=str, default="")
    p.add_argument("--export_final_demo_dataset", action="store_true", default=False)
    p.add_argument("--export_final_demo_dataset_path", type=str, default="")
    p.add_argument("--bc_eval_hotkey_enable", action="store_true", default=True)
    p.add_argument("--no_bc_eval_hotkey_enable", dest="bc_eval_hotkey_enable", action="store_false")
    p.add_argument(
        "--bc_eval_interval",
        type=int,
        default=0,
        help="If >0, periodically export replay, train a learned intervention teacher, and evaluate it every N env steps.",
    )
    p.add_argument("--bc_eval_epochs", type=int, default=10)
    p.add_argument("--bc_eval_batch_size", type=int, default=256)
    p.add_argument("--bc_eval_context_len", type=int, default=8)
    p.add_argument("--bc_eval_hidden_dim", type=int, default=256)
    p.add_argument("--bc_eval_num_layers", type=int, default=3)
    p.add_argument("--bc_eval_intervention_threshold", type=float, default=0.5)
    p.add_argument("--bc_eval_num_episodes", type=int, default=3)
    p.add_argument("--bc_eval_render_mode", type=str, default="pygame", choices=["human", "rgb_array", "none", "pygame", "topdown"])
    p.add_argument("--bc_eval_fps", type=float, default=30.0)
    p.add_argument("--bc_eval_device", type=str, default="cpu")
    p.add_argument("--bc_eval_student_policy", type=str, default="checkpoint", choices=["random", "zero", "checkpoint"])
    p.add_argument("--bc_eval_student_checkpoint_path", type=str, default="")
    p.add_argument("--bc_eval_dataset_label", type=str, default="hotkey_human_intervention_replay")

    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="thesis-safetygym")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")

    p.add_argument("--log_interval", type=int, default=2_000)
    p.add_argument("--eval_interval", type=int, default=20_000)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument(
        "--eval_layout_seed_replay_prob",
        type=float,
        default=-1.0,
        help=(
            "If >=0, override layout_seed_replay_prob only for train-time eval. "
            "Use 0.0 to evaluate the standard random layout distribution while training with replayed hard seeds."
        ),
    )
    p.add_argument("--save_interval", type=int, default=50_000)
    p.add_argument("--viz_on_checkpoint", action="store_true", default=False)
    p.add_argument("--viz_grid_resolution", type=int, default=48)
    p.add_argument("--viz_quiver_stride", type=int, default=4)
    p.add_argument("--viz_device", type=str, default="cpu")
    p.add_argument("--viz_seed", type=int, default=0)
    p.add_argument("--viz_first_step", type=int, default=0)
    p.add_argument("--viz_headings_deg", type=str, default="0,90,180,270")
    p.add_argument("--viz_num_rollouts", type=int, default=4)
    p.add_argument("--eval_save_episode_plots", action="store_true", default=False)
    p.add_argument("--eval_episode_plot_max_episodes", type=int, default=9)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_minimal_training(args)


if __name__ == "__main__":
    main()
