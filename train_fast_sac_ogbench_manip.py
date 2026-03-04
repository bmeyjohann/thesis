#!/usr/bin/env python3
"""Manipulation-focused FastSAC entrypoint.

This entrypoint keeps manipulation workflow explicit:
- parse manip-specific FastSAC args
- run the manipulation-specific FastSAC OGBench pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

import ogbench  # noqa: F401  # Registers environments.

from ogbench_utils.fastsac_ogbench_manip_cli import parse_fastsac_manip_args
from ogbench_utils.fastsac_ogbench_manip_train import run_fastsac_ogbench_manip

TOOLS_PATH = Path(__file__).resolve().parent / "tools"
if TOOLS_PATH.exists():
    sys.path.append(str(TOOLS_PATH))
try:
    from visualize_policy_map import generate_policy_map  # type: ignore
except Exception:  # pragma: no cover
    generate_policy_map = None


def main() -> None:
    args = parse_fastsac_manip_args()
    run_fastsac_ogbench_manip(args, generate_policy_map=generate_policy_map)


if __name__ == "__main__":
    main()
