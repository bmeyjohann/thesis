#!/usr/bin/env python3
"""Maze-focused FastSAC entrypoint.

This entrypoint keeps maze workflow explicit:
- parse maze-specific FastSAC args
- run the maze-specific FastSAC OGBench pipeline
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ogbench_utils.fastsac_ogbench_maze_cli import parse_fastsac_maze_args

TOOLS_PATH = Path(__file__).resolve().parent / "tools"
if TOOLS_PATH.exists():
    sys.path.append(str(TOOLS_PATH))
try:
    from visualize_policy_map import generate_policy_map  # type: ignore
except Exception:  # pragma: no cover
    generate_policy_map = None


def _configure_mujoco_gl_for_train() -> None:
    configured = str(os.environ.get("MUJOCO_GL", "")).strip().lower()
    if sys.platform.startswith("win") and configured in {"", "egl"}:
        os.environ["MUJOCO_GL"] = "glfw"


def main() -> None:
    args = parse_fastsac_maze_args()
    _configure_mujoco_gl_for_train()
    import ogbench  # noqa: F401  # Registers environments.
    from ogbench_utils.fastsac_ogbench_maze_train import run_fastsac_ogbench_maze

    run_fastsac_ogbench_maze(args, generate_policy_map=generate_policy_map)


if __name__ == "__main__":
    main()
