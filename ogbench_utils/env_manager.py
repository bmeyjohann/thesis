from __future__ import annotations

from typing import Optional, Tuple


def maybe_switch_env(
    *,
    args,
    envs,
    current_env_name: str,
    wrappers,
    total_env_steps: int,
    env_switch_global_step: Optional[int],
    record_progress,
    wandb_run,
) -> Tuple[bool, str, Optional[object], object]:
    """Switch underlying environment when curriculum conditions trigger."""
    if (
        args.switch_env_name is None
        or env_switch_global_step is None
        or total_env_steps < env_switch_global_step
        or args.switch_env_name == current_env_name
    ):
        return False, current_env_name, None, wandb_run

    per_env_progress = total_env_steps // envs.num_envs
    new_env_name = args.switch_env_name
    record_progress(f"[Env] Switching from {current_env_name} to {new_env_name} at total_steps={total_env_steps}")
    obs = envs.switch_env(
        new_env_name,
        wrappers=wrappers,
        curriculum_steps=per_env_progress,
    )
    if args.use_wandb and wandb_run is not None:
        import wandb

        wandb_run.log(
            {
                "env/switch_event": 1,
                "env/current_env": new_env_name,
            },
            step=total_env_steps,
        )
    return True, new_env_name, obs, wandb_run
