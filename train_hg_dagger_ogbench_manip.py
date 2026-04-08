#!/usr/bin/env python3
"""Manipulation-focused HG-DAgger entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import ogbench  # noqa: F401  # Registers environments.

from ogbench_utils.hgdagger_ogbench_manip_cli import parse_hgdagger_manip_args
from ogbench_utils.hgdagger_ogbench_manip_train import run_hgdagger_ogbench_manip

TOOLS_PATH = Path(__file__).resolve().parent / "tools"
if TOOLS_PATH.exists():
    sys.path.append(str(TOOLS_PATH))


def main() -> None:
    args = parse_hgdagger_manip_args()
    run_hgdagger_ogbench_manip(args)


if __name__ == "__main__":
    main()
