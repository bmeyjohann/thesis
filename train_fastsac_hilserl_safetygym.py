#!/usr/bin/env python3
from __future__ import annotations

import argparse

from safetygym_utils.train import run_training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FastSAC HILSERL-style training for Safety-Gymnasium")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--torch_num_threads", type=int, default=1)
    p.add_argument("--torch_num_interop_threads", type=int, default=1)
    p.add_argument("--total_timesteps", type=int, default=500_000)
    p.add_argument("--learning_starts", type=int, default=5_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--update_every", type=int, default=1)
    p.add_argument("--updates_per_cycle", type=int, default=1)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor_learning_rate", type=float, default=3e-4)
    p.add_argument("--critic_learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--actor_hidden_dim", type=int, default=512)
    p.add_argument("--critic_hidden_dim", type=int, default=1024)
    p.add_argument("--init_scale", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=10.0)

    p.add_argument("--reward_mode", type=str, default="sparse", choices=["sparse", "dense", "none"])
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)

    p.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array", "none"])
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=2.0)
    p.add_argument("--car_force_scale", type=float, default=2.0)
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--use_intervention", action="store_true", default=False)
    p.add_argument("--intervention_threshold", type=float, default=0.1)
    p.add_argument("--intervention_hold_seconds", type=float, default=0.25)
    p.add_argument("--human_action_scale", type=float, default=1.0)
    p.add_argument("--controller_fps_limit", type=int, default=0)
    p.add_argument("--controller_overlay_hz", type=float, default=20.0)

    p.add_argument("--demo_sample_ratio", type=float, default=0.5)
    p.add_argument("--prefill_demo_episodes", type=int, default=0)
    p.add_argument("--prefill_max_steps_per_episode", type=int, default=0)
    p.add_argument("--prefill_policy", type=str, default="student", choices=["student", "random", "zero"])
    p.add_argument("--demo_pretrain_updates", type=int, default=0)
    p.add_argument("--demo_pretrain_batch_size", type=int, default=0)
    p.add_argument("--critic_reset_after_pretrain", action="store_true", default=False)
    p.add_argument("--uncertainty_log_every_step", dest="uncertainty_log_every_step", action="store_true")
    p.add_argument("--no_uncertainty_log_every_step", dest="uncertainty_log_every_step", action="store_false")
    p.add_argument("--uncertainty_pre_intervention_window", type=int, default=25)
    p.add_argument("--uncertainty_oversight_mode", type=str, default="signal_only", choices=["off", "signal_only"])
    p.add_argument("--uncertainty_oversight_threshold", type=float, default=0.0)
    p.add_argument("--uncertainty_oversight_ema_alpha", type=float, default=0.05)
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="thesis-safetygym")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")

    p.set_defaults(uncertainty_log_every_step=True)

    p.add_argument("--log_interval", type=int, default=2_000)
    p.add_argument("--eval_interval", type=int, default=20_000)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=50_000)
    p.add_argument("--profile_timing", action="store_true", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_training(args, variant="hilserl")


if __name__ == "__main__":
    main()
