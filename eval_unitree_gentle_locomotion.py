"""Register and inspect the gentle-terrain G1 locomotion task."""

from __future__ import annotations

import sys

from unitree_gentle_locomotion import make_gentle_locomotion_cfg


def main() -> int:
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
    from src.tasks.velocity.rl import VelocityOnPolicyRunner

    task_id = "Unitree-G1-Gentle-Curriculum-Eval"
    register_mjlab_task(
        task_id,
        make_gentle_locomotion_cfg(num_envs=1),
        make_gentle_locomotion_cfg(play=True, num_envs=1),
        unitree_g1_ppo_runner_cfg(),
        VelocityOnPolicyRunner,
    )

    import eval_unitree_velocity_policy

    sys.argv = [sys.argv[0], "--task", task_id, *sys.argv[1:]]
    return int(eval_unitree_velocity_policy.main())


if __name__ == "__main__":
    raise SystemExit(main())
