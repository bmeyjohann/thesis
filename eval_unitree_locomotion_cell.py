"""Evaluate a G1 checkpoint on an explicitly selected terrain/material cell."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from unitree_locomotion_cells import (
    GEOMETRIES,
    MATERIALS,
    cell_metadata,
    make_locomotion_cell_cfg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    parser.add_argument("--material", choices=MATERIALS, required=True)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary-file", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
    from src.tasks.velocity.rl import VelocityOnPolicyRunner

    task_id = f"Unitree-G1-Cell-Eval-{args.geometry}-{args.material}"
    register_mjlab_task(
        task_id,
        make_locomotion_cell_cfg(args.geometry, args.material, num_envs=args.num_envs),
        make_locomotion_cell_cfg(args.geometry, args.material, play=True, num_envs=args.num_envs),
        unitree_g1_ppo_runner_cfg(),
        VelocityOnPolicyRunner,
    )
    forwarded = [
        "eval_unitree_velocity_policy.py", "--task", task_id,
        "--checkpoint-file", str(args.checkpoint_file),
        "--seed", str(args.seed), "--num-envs", str(args.num_envs),
        "--steps", str(args.steps), "--device", args.device,
        "--checkpoint-observation-mode", "task",
    ]
    import eval_unitree_velocity_policy

    old_argv = sys.argv
    output = io.StringIO()
    try:
        sys.argv = forwarded
        with contextlib.redirect_stdout(output):
            result = eval_unitree_velocity_policy.main()
    finally:
        sys.argv = old_argv
    rendered = output.getvalue()
    print(rendered, end="")
    marker = "[velocity-eval] {"
    summaries = [line[len("[velocity-eval] "):] for line in rendered.splitlines() if line.startswith(marker)]
    if not summaries:
        raise RuntimeError("Velocity evaluator did not emit a JSON summary")
    summary = json.loads(summaries[-1])
    summary.update(cell_metadata(args.geometry, args.material))
    summary["evaluation_seed"] = int(args.seed)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[cell-eval] summary={args.summary_file.resolve()}")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
