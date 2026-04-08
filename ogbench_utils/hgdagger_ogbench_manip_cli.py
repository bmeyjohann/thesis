"""Manipulation-specific HG-DAgger CLI parsing."""

from __future__ import annotations

import argparse

from .cli import build_train_parser

_EVAL_ARGS = (
    ("--eval_interval", dict(type=int, default=10_000, help="Environment step interval between evaluation runs (0 disables)")),
    ("--num_eval_episodes", dict(type=int, default=1, help="Number of evaluation episodes to run each interval")),
    ("--eval_num_envs", dict(type=int, default=10, help="Number of parallel envs to use for evaluation")),
)

_HGDAGGER_DEFAULTS = {
    "env_name": "cube-single-v0",
    "teacher_type": "cube_plan",
    "obs_mode": "state",
    "save_interval": 10_000,
    "log_interval": 64,
    "algo_variant": "hg_dagger",
}


def _attach_eval_args(parser: argparse.ArgumentParser) -> None:
    for flag, kwargs in _EVAL_ARGS:
        parser.add_argument(flag, **kwargs)


def build_hgdagger_manip_parser() -> argparse.ArgumentParser:
    parser = build_train_parser()
    _attach_eval_args(parser)
    parser.add_argument(
        "--hg_ensemble_size",
        type=int,
        default=5,
        help="Number of deterministic novice policies in the HG-DAgger ensemble.",
    )
    parser.add_argument(
        "--hg_doubt_percentile",
        type=float,
        default=75.0,
        help="Percentile used to define the high-doubt tail when estimating tau from intervention-time doubt values.",
    )
    parser.set_defaults(**_HGDAGGER_DEFAULTS)
    return parser


def parse_hgdagger_manip_args():
    args = build_hgdagger_manip_parser().parse_args()
    if getattr(args, "obs_mode", "state") != "state":
        raise ValueError("HG-DAgger manipulation entrypoint only supports obs_mode='state'.")
    env_name = str(getattr(args, "env_name", "") or "").lower()
    forbidden = ("maze", "pointmaze", "antmaze")
    if any(tok in env_name for tok in forbidden):
        raise ValueError(f"HG-DAgger manipulation entrypoint received maze env_name={args.env_name!r}.")
    if getattr(args, "train_render_mode", "none") == "human" and int(getattr(args, "num_envs", 1)) != 1:
        raise ValueError("--train_render_mode=human requires --num_envs=1.")
    if int(getattr(args, "num_envs", 1)) < 1:
        raise ValueError("--num_envs must be >= 1.")
    if int(getattr(args, "hg_ensemble_size", 1)) < 1:
        raise ValueError("--hg_ensemble_size must be >= 1.")
    return args
