"""Manipulation-specific faithful PVP-TD3 CLI parsing."""

from __future__ import annotations

import argparse

from .cli import build_train_parser

_EVAL_ARGS = (
    ("--eval_interval", dict(type=int, default=5_000, help="Environment step interval between evaluation runs (0 disables)")),
    ("--num_eval_episodes", dict(type=int, default=5, help="Number of evaluation episodes to run each interval")),
    ("--eval_num_envs", dict(type=int, default=1, help="Number of parallel envs to use for evaluation")),
)

_PVP_DEFAULTS = {
    "env_name": "cube-single-singletask-task1-v0",
    "teacher_type": "cube_markov",
    "obs_mode": "state",
    "save_interval": 10_000,
    "log_interval": 64,
    "algo_variant": "pvp_td3",
    "batch_size": 128,
    "actor_learning_rate": 1e-4,
    "critic_learning_rate": 1e-4,
    "tau": 0.005,
    "gamma": 0.99,
    "learning_starts": 100,
    "num_updates": 1,
}


def _attach_eval_args(parser: argparse.ArgumentParser) -> None:
    for flag, kwargs in _EVAL_ARGS:
        parser.add_argument(flag, **kwargs)


def build_pvp_td3_manip_parser() -> argparse.ArgumentParser:
    parser = build_train_parser()
    _attach_eval_args(parser)
    parser.add_argument(
        "--pvp_policy_delay",
        type=int,
        default=2,
        help="Delayed actor update interval for the faithful TD3-style PVP trainer.",
    )
    parser.add_argument(
        "--pvp_target_policy_noise",
        type=float,
        default=0.2,
        help="Stddev of target policy smoothing noise for the faithful TD3-style PVP trainer.",
    )
    parser.add_argument(
        "--pvp_target_noise_clip",
        type=float,
        default=0.5,
        help="Absolute clip for target policy smoothing noise in the faithful TD3-style PVP trainer.",
    )
    parser.add_argument(
        "--pvp_cql_coefficient",
        type=float,
        default=1.0,
        help="Scale for the proxy value teacher/student losses on intervention rows.",
    )
    parser.add_argument(
        "--pvp_stop_td_on_intervention_start",
        action="store_true",
        default=True,
        help="Mask TD loss on takeover-start rows, matching the official PVP default.",
    )
    parser.add_argument(
        "--no_pvp_stop_td_on_intervention_start",
        dest="pvp_stop_td_on_intervention_start",
        action="store_false",
        help="Disable the takeover-start TD mask in the faithful PVP trainer.",
    )
    parser.add_argument(
        "--pvp_balance_sample",
        action="store_true",
        default=True,
        help="Sample half batches from novice and human buffers when both contain enough data.",
    )
    parser.add_argument(
        "--no_pvp_balance_sample",
        dest="pvp_balance_sample",
        action="store_false",
        help="Use a single merged human buffer instead of balanced novice/human sampling.",
    )
    parser.set_defaults(**_PVP_DEFAULTS)
    return parser


def parse_pvp_td3_manip_args():
    args = build_pvp_td3_manip_parser().parse_args()
    if getattr(args, "obs_mode", "state") != "state":
        raise ValueError("Faithful PVP manipulation entrypoint only supports obs_mode='state'.")
    env_name = str(getattr(args, "env_name", "") or "").lower()
    forbidden = ("maze", "pointmaze", "antmaze")
    if any(tok in env_name for tok in forbidden):
        raise ValueError(f"Faithful PVP manipulation entrypoint received maze env_name={args.env_name!r}.")
    if getattr(args, "train_render_mode", "none") == "human" and int(getattr(args, "num_envs", 1)) != 1:
        raise ValueError("--train_render_mode=human requires --num_envs=1.")
    if int(getattr(args, "num_envs", 1)) < 1:
        raise ValueError("--num_envs must be >= 1.")
    if int(getattr(args, "pvp_policy_delay", 1)) < 1:
        raise ValueError("--pvp_policy_delay must be >= 1.")
    return args
