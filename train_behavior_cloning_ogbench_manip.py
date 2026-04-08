#!/usr/bin/env python3
"""Manipulation-focused behavior cloning entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import ogbench  # noqa: F401  # Registers environments.

from ogbench_utils.bc_ogbench_manip_cli import parse_bc_manip_args
from ogbench_utils.bc_ogbench_manip_train import run_bc_ogbench_manip

TOOLS_PATH = Path(__file__).resolve().parent / "tools"
if TOOLS_PATH.exists():
    sys.path.append(str(TOOLS_PATH))


def main() -> None:
    args = parse_bc_manip_args()
    run_bc_ogbench_manip(args)


if __name__ == "__main__":
    main()
