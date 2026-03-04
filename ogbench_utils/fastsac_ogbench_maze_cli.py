"""Maze-specific FastSAC OGBench CLI parsing."""

from __future__ import annotations

import argparse

from .cli import build_train_parser

_FASTSAC_EVAL_ARGS = (
    ("--eval_interval", dict(type=int, default=50_000, help="Environment step interval between evaluation runs (0 disables)")),
    ("--num_eval_episodes", dict(type=int, default=10, help="Number of evaluation episodes to run each interval")),
    ("--eval_num_envs", dict(type=int, default=8, help="Number of parallel envs to use for evaluation")),
)

_MAZE_DEFAULTS = {
    "env_name": "pointmaze-arena-danger-lethal-v0",
    "teacher_type": "bfs",
    "obs_mode": "state",
}


def _attach_eval_args(parser: argparse.ArgumentParser) -> None:
    for flag, kwargs in _FASTSAC_EVAL_ARGS:
        parser.add_argument(flag, **kwargs)


def build_fastsac_maze_parser() -> argparse.ArgumentParser:
    parser = build_train_parser()
    _attach_eval_args(parser)
    parser.set_defaults(**_MAZE_DEFAULTS)
    return parser


def parse_fastsac_maze_args():
    args = build_fastsac_maze_parser().parse_args()
    if getattr(args, "obs_mode", "state") != "state":
        raise ValueError("FastSAC maze entrypoint only supports obs_mode='state'.")
    env_name = str(getattr(args, "env_name", "") or "").lower()
    forbidden = ("cube", "scene", "puzzle", "manip")
    if any(tok in env_name for tok in forbidden):
        raise ValueError(f"Maze FastSAC entrypoint received non-maze env_name={args.env_name!r}.")
    if int(getattr(args, "num_critics", 2)) < 1:
        raise ValueError("--num_critics must be >= 1.")
    return args
