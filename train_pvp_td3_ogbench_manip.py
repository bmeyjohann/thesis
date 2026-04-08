#!/usr/bin/env python3
"""Manipulation-focused faithful PVP-TD3 entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OGBENCH_SUBMODULE_PATH = REPO_ROOT / "ogbench"
FASTTD3_SUBMODULE_PATH = REPO_ROOT / "fasttd3"
for _path in (FASTTD3_SUBMODULE_PATH, OGBENCH_SUBMODULE_PATH):
    _path_str = str(_path)
    if _path_str in sys.path:
        sys.path.remove(_path_str)
    sys.path.insert(0, _path_str)

from ogbench_utils.pvp_td3_ogbench_manip_cli import parse_pvp_td3_manip_args

TOOLS_PATH = REPO_ROOT / "tools"
if TOOLS_PATH.exists():
    sys.path.append(str(TOOLS_PATH))


def _configure_mujoco_gl_for_train() -> None:
    configured = str(os.environ.get("MUJOCO_GL", "")).strip().lower()
    if sys.platform.startswith("win") and configured in {"", "egl"}:
        os.environ["MUJOCO_GL"] = "glfw"


def main() -> None:
    args = parse_pvp_td3_manip_args()
    _configure_mujoco_gl_for_train()
    import ogbench  # noqa: F401  # Registers environments.
    from ogbench_utils.pvp_td3_ogbench_manip_train import run_pvp_td3_ogbench_manip

    run_pvp_td3_ogbench_manip(args)


if __name__ == "__main__":
    main()
