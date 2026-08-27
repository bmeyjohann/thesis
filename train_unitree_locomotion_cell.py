"""Train one staged G1 locomotion expert without modifying the submodule."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

from unitree_locomotion_cells import (
    GEOMETRIES,
    MATERIALS,
    REPO_ROOT,
    UNITREE_ROOT,
    cell_metadata,
    make_locomotion_cell_cfg,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    parser.add_argument("--material", choices=MATERIALS, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--wandb-project", default="unitree-g1-multimodal-locomotion")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
    from src.tasks.velocity.rl import VelocityOnPolicyRunner

    num_envs = min(args.num_envs, 16) if args.smoke else args.num_envs
    iterations = 1 if args.smoke else args.iterations
    task_id = f"Unitree-G1-Cell-{args.geometry}-{args.material}"
    train_cfg = make_locomotion_cell_cfg(args.geometry, args.material, num_envs=num_envs)
    play_cfg = make_locomotion_cell_cfg(args.geometry, args.material, play=True, num_envs=1)
    rl_cfg = unitree_g1_ppo_runner_cfg()
    rl_cfg.seed = args.seed
    rl_cfg.max_iterations = iterations
    rl_cfg.save_interval = min(args.save_interval, iterations)
    rl_cfg.experiment_name = "g1_multimodal_cell_experts"
    rl_cfg.run_name = args.run_name or f"{args.geometry}_{args.material}_seed{args.seed}"
    if args.smoke:
        rl_cfg.run_name = "smoke_" + rl_cfg.run_name
        rl_cfg.num_steps_per_env = 4
    rl_cfg.logger = args.logger
    rl_cfg.wandb_project = args.wandb_project
    rl_cfg.upload_model = False
    if args.warm_start_checkpoint is not None:
        source = args.warm_start_checkpoint.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {source}")
        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
        warm_run = f"warmstart_{digest}"
        warm_dir = REPO_ROOT / "logs/rsl_rl" / rl_cfg.experiment_name / warm_run
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
    register_mjlab_task(task_id, train_cfg, play_cfg, rl_cfg, VelocityOnPolicyRunner)

    metadata = {
        **cell_metadata(args.geometry, args.material),
        "seed": args.seed,
        "num_envs": num_envs,
        "iterations": iterations,
        "smoke": args.smoke,
        "task_id": task_id,
        "warm_start_checkpoint": (
            str(args.warm_start_checkpoint.expanduser().resolve())
            if args.warm_start_checkpoint is not None else None
        ),
    }
    print("[cell-config] " + json.dumps(metadata, sort_keys=True), flush=True)
    train_module = _load_train_module()
    train_module.launch_training(task_id, train_module.TrainConfig.from_task(task_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
