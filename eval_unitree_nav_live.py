#!/usr/bin/env python3
"""Interactive Unitree navigation controller viewer.

This launches the mjlab Viser/native viewer with a lightweight controller
callable. It is for human inspection; metrics/video should still use
eval_unitree_nav_baselines.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from eval_unitree_nav_baselines import _reset_until_feasible, controller_action, make_env
from train_unitree_nav_thesis import DEFAULT_LOW_LEVEL, ROOT, _extract_actor_obs


class ControllerPolicy:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def __call__(self, obs):
        actor_obs = _extract_actor_obs(obs).to(self.args.device, dtype=torch.float32)
        with torch.inference_mode():
            return controller_action(actor_obs, self.args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["direct_goal", "scan_teacher"], default="scan_teacher")
    parser.add_argument("--task", default="Unitree-G1-Nav-Obstacles-Safe-Collision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--episode-length-s", type=float, default=16.0)
    parser.add_argument("--low-level-policy-path", default=str(DEFAULT_LOW_LEVEL))
    parser.add_argument("--viewer", choices=["auto", "native", "viser"], default="viser")
    parser.add_argument("--num-steps", type=int, default=0, help="0 means run until the viewer is closed.")
    parser.add_argument("--teacher-scan-block-threshold", type=float, default=0.12)
    parser.add_argument("--teacher-sector-half-width", type=float, default=0.35)
    parser.add_argument("--teacher-align-angle", type=float, default=0.45)
    parser.add_argument("--teacher-max-vx", type=float, default=0.75)
    parser.add_argument("--teacher-max-vy", type=float, default=0.5)
    parser.add_argument("--teacher-yaw-gain", type=float, default=1.2)
    parser.add_argument("--teacher-clearance-weight", type=float, default=0.0)
    parser.add_argument("--teacher-clearance-power", type=float, default=2.0)
    parser.add_argument("--teacher-speed-clearance-scale", type=float, default=0.0)
    parser.add_argument("--teacher-num-sectors", type=int, default=13)
    parser.add_argument("--teacher-min-forward-scale", type=float, default=0.25)
    parser.add_argument("--teacher-escape-risk-threshold", type=float, default=0.0)
    parser.add_argument("--teacher-escape-forward-scale", type=float, default=0.0)
    parser.add_argument("--teacher-escape-lateral-scale", type=float, default=1.0)
    parser.add_argument("--teacher-escape-radius", type=float, default=1.0)
    parser.add_argument("--teacher-escape-all-directions", action="store_true")
    parser.add_argument("--teacher-bypass-angle", type=float, default=0.0)
    parser.add_argument("--teacher-goal-stop-dist", type=float, default=0.0)
    parser.add_argument("--min-goal-obstacle-clearance", type=float, default=0.0)
    parser.add_argument("--goal-clearance-resample-attempts", type=int, default=50)
    parser.add_argument("--min-start-obstacle-clearance", type=float, default=0.0)
    parser.add_argument("--start-clearance-resample-attempts", type=int, default=20)
    parser.add_argument("--require-blocked-corridor", action="store_true")
    parser.add_argument("--blocked-corridor-radius", type=float, default=0.45)
    parser.add_argument("--blocked-corridor-ignore-end-radius", type=float, default=0.75)
    parser.add_argument("--blocked-corridor-min-cells", type=int, default=1)
    parser.add_argument("--blocked-corridor-resample-attempts", type=int, default=100)
    parser.add_argument("--debug-obstacle-width-min", type=float, default=0.0)
    parser.add_argument("--debug-obstacle-width-max", type=float, default=0.0)
    parser.add_argument("--debug-obstacle-height-min", type=float, default=0.0)
    parser.add_argument("--debug-obstacle-height-max", type=float, default=0.0)
    parser.add_argument("--debug-num-obstacles", type=int, default=0)
    parser.add_argument("--debug-platform-width", type=float, default=0.0)
    parser.add_argument("--debug-obstacle-border-width", type=float, default=0.0)
    parser.add_argument("--debug-goal-through-obstacle", action="store_true")
    parser.add_argument("--debug-goal-distance", type=float, default=3.2)
    parser.add_argument("--debug-goal-obstacle-min-dist", type=float, default=0.8)
    parser.add_argument("--debug-goal-obstacle-max-dist", type=float, default=2.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env = make_env(args, num_envs=args.num_envs, render=False)
    if (
        float(args.min_goal_obstacle_clearance) > 0.0
        or float(args.min_start_obstacle_clearance) > 0.0
        or bool(args.require_blocked_corridor)
    ):
        if int(args.num_envs) != 1:
            raise ValueError("Filtered live inspection currently requires --num-envs 1")
        _reset_until_feasible(args, env)
    policy = ControllerPolicy(args)

    if args.viewer == "auto":
        resolved_viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    else:
        resolved_viewer = args.viewer

    if resolved_viewer == "native":
        from mjlab.viewer import NativeMujocoViewer

        viewer = NativeMujocoViewer(env, policy)
    else:
        from mjlab.viewer import ViserPlayViewer

        print("[unitree-live] Open the Viser HTTP URL printed below to inspect the rollout.")
        viewer = ViserPlayViewer(env, policy)

    viewer.run(num_steps=None if int(args.num_steps) <= 0 else int(args.num_steps))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
