#!/usr/bin/env python3
"""Manipulation-focused faithful PVP-TD3 entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import ogbench  # noqa: F401  # Registers environments.

from ogbench_utils.pvp_td3_ogbench_manip_cli import parse_pvp_td3_manip_args
from ogbench_utils.pvp_td3_ogbench_manip_train import run_pvp_td3_ogbench_manip

TOOLS_PATH = Path(__file__).resolve().parent / "tools"
if TOOLS_PATH.exists():
    sys.path.append(str(TOOLS_PATH))


def main() -> None:
    args = parse_pvp_td3_manip_args()
    run_pvp_td3_ogbench_manip(args)


if __name__ == "__main__":
    main()
