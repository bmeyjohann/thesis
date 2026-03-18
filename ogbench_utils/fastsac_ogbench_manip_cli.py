"""Manipulation-specific FastSAC OGBench CLI parsing."""

from __future__ import annotations

import argparse

from .cli import build_train_parser

_FASTSAC_EVAL_ARGS = (
    ("--eval_interval", dict(type=int, default=10_000, help="Environment step interval between evaluation runs (0 disables)")),
    ("--num_eval_episodes", dict(type=int, default=1, help="Number of evaluation episodes to run each interval")),
    ("--eval_num_envs", dict(type=int, default=10, help="Number of parallel envs to use for evaluation")),
)

_MANIP_DEFAULTS = {
    "env_name": "cube-single-v0",
    "teacher_type": "cube_plan",
    "obs_mode": "state",
    "save_interval": 10_000,
    "log_interval": 64,
}


def _attach_eval_args(parser: argparse.ArgumentParser) -> None:
    for flag, kwargs in _FASTSAC_EVAL_ARGS:
        parser.add_argument(flag, **kwargs)


def build_fastsac_manip_parser() -> argparse.ArgumentParser:
    parser = build_train_parser()
    _attach_eval_args(parser)
    parser.set_defaults(**_MANIP_DEFAULTS)
    return parser


def parse_fastsac_manip_args():
    args = build_fastsac_manip_parser().parse_args()
    if getattr(args, "obs_mode", "state") != "state":
        raise ValueError("FastSAC manipulation entrypoint only supports obs_mode='state'.")
    env_name = str(getattr(args, "env_name", "") or "").lower()
    forbidden = ("maze", "pointmaze", "antmaze")
    if any(tok in env_name for tok in forbidden):
        raise ValueError(f"Manip FastSAC entrypoint received maze env_name={args.env_name!r}.")
    if getattr(args, "train_render_mode", "none") == "human" and int(getattr(args, "num_envs", 1)) != 1:
        raise ValueError("--train_render_mode=human requires --num_envs=1.")
    if int(getattr(args, "num_critics", 2)) < 1:
        raise ValueError("--num_critics must be >= 1.")
    return args
