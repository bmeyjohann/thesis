"""Train the G1 gentle-terrain locomotion curriculum."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

from unitree_gentle_locomotion import (
    REPO_ROOT,
    UNITREE_ROOT,
    gentle_locomotion_metadata,
    make_gentle_locomotion_cfg,
)


DEFAULT_WARM_START = (
    REPO_ROOT
    / "external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/"
    "supervisor_rough_model9999/model_9999.pt"
)


def _load_train_module():
    path = UNITREE_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("unitree_mjlab_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configure_warm_start(rl_cfg, source: Path) -> None:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    warm_run = f"gentle_warmstart_{digest}"
    warm_dir = REPO_ROOT / "external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity" / warm_run
    warm_dir.mkdir(parents=True, exist_ok=True)
    linked = warm_dir / f"legacy_{source.name}"
    if not linked.exists():
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            linked.symlink_to(source)
        else:
            from eval_unitree_velocity_policy import _split_to_legacy_checkpoint

            torch.save(_split_to_legacy_checkpoint(checkpoint), linked)
    rl_cfg.resume = True
    rl_cfg.load_run = warm_run
    rl_cfg.load_checkpoint = linked.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--run-name", default="g1_gentle_turn_curriculum")
    parser.add_argument("--warm-start-checkpoint", type=Path, default=DEFAULT_WARM_START)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(UNITREE_ROOT)
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
    from src.tasks.velocity.rl import VelocityOnPolicyRunner

    num_envs = min(args.num_envs, 16) if args.smoke else args.num_envs
    iterations = 1 if args.smoke else args.iterations
    task_id = "Unitree-G1-Gentle-Curriculum"
    env_cfg = make_gentle_locomotion_cfg(num_envs=num_envs)
    play_cfg = make_gentle_locomotion_cfg(play=True, num_envs=1)
    rl_cfg = unitree_g1_ppo_runner_cfg()
    rl_cfg.seed = args.seed
    rl_cfg.max_iterations = iterations
    rl_cfg.save_interval = min(args.save_interval, iterations)
    rl_cfg.experiment_name = "g1_velocity"
    rl_cfg.run_name = args.run_name
    rl_cfg.logger = args.logger
    rl_cfg.upload_model = False
    if args.smoke:
        rl_cfg.run_name = "smoke_" + rl_cfg.run_name
        rl_cfg.num_steps_per_env = 4
    if not args.scratch:
        _configure_warm_start(rl_cfg, args.warm_start_checkpoint)

    register_mjlab_task(task_id, env_cfg, play_cfg, rl_cfg, VelocityOnPolicyRunner)
    resolved = {
        **gentle_locomotion_metadata(),
        "seed": args.seed,
        "num_envs": num_envs,
        "iterations": iterations,
        "smoke": args.smoke,
        "task_id": task_id,
        "warm_start_checkpoint": None
        if args.scratch
        else str(args.warm_start_checkpoint.expanduser().resolve()),
    }
    print("[gentle-locomotion-config] " + json.dumps(resolved, sort_keys=True), flush=True)
    train_module = _load_train_module()
    train_module.launch_training(task_id, train_module.TrainConfig.from_task(task_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
