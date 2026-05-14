"""Maze-specific HG-DAgger CLI parsing."""

from __future__ import annotations

import argparse

from .cli import build_train_parser

_EVAL_ARGS = (
    ("--eval_interval", dict(type=int, default=10_000, help="Environment step interval between evaluation runs (0 disables)")),
    ("--num_eval_episodes", dict(type=int, default=10, help="Number of evaluation episodes to run each interval")),
    ("--eval_num_envs", dict(type=int, default=8, help="Number of parallel envs to use for evaluation")),
)

_HGDAGGER_DEFAULTS = {
    "env_name": "pointmaze-arena-danger-lethal-v0",
    "teacher_type": "bfs",
    "obs_mode": "state",
    "save_interval": 10_000,
    "log_interval": 200,
    "algo_variant": "hg_dagger",
}


def _attach_eval_args(parser: argparse.ArgumentParser) -> None:
    for flag, kwargs in _EVAL_ARGS:
        parser.add_argument(flag, **kwargs)


def build_hgdagger_maze_parser() -> argparse.ArgumentParser:
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


def parse_hgdagger_maze_args():
    args = build_hgdagger_maze_parser().parse_args()
    if getattr(args, "obs_mode", "state") != "state":
        raise ValueError("HG-DAgger maze entrypoint only supports obs_mode='state'.")
    env_name = str(getattr(args, "env_name", "") or "").lower()
    forbidden = ("cube", "scene", "puzzle", "manip")
    if any(tok in env_name for tok in forbidden):
        raise ValueError(f"HG-DAgger maze entrypoint received non-maze env_name={args.env_name!r}.")
    if int(getattr(args, "num_envs", 1)) < 1:
        raise ValueError("--num_envs must be >= 1.")
    if int(getattr(args, "hg_ensemble_size", 1)) < 1:
        raise ValueError("--hg_ensemble_size must be >= 1.")
    return args
