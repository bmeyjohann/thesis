#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_ppo_safetygym_minimal import PeriodicEvalCallback, _make_single_env, _vecnormalize_path_for_model


def _load_args(model_path: Path) -> SimpleNamespace:
    args_path = model_path.parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing checkpoint args: {args_path}")
    with args_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SimpleNamespace(**data)


def main() -> int:
    p = argparse.ArgumentParser(description="Posthoc PPO Safety-Gym eval with curriculum layout artifacts.")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--step_label", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval_fixed_layout_preset", type=str, default="train")
    p.add_argument("--eval_layout_curriculum", type=str, default="train")
    p.add_argument("--eval_layout_curriculum_level", type=int, default=-1)
    p.add_argument("--car_wheel_command_limit", type=float, default=0.0)
    p.add_argument("--car_force_scale", type=float, default=0.0)
    p.add_argument("--plot_max_episodes", type=int, default=9)
    p.add_argument("--reward_surface", action="store_true", default=False)
    parsed = p.parse_args()

    model_path = Path(parsed.model_path).expanduser().resolve()
    train_args = _load_args(model_path)
    eval_args = copy.copy(train_args)
    eval_args.seed = int(parsed.seed)
    eval_args.eval_fixed_layout_preset = str(parsed.eval_fixed_layout_preset)
    eval_args.eval_layout_curriculum = str(parsed.eval_layout_curriculum)
    eval_args.eval_layout_curriculum_level = int(parsed.eval_layout_curriculum_level)
    if float(parsed.car_wheel_command_limit) > 0.0:
        eval_args.car_wheel_command_limit = float(parsed.car_wheel_command_limit)
    if float(parsed.car_force_scale) > 0.0:
        eval_args.car_force_scale = float(parsed.car_force_scale)
    eval_args.render_mode = "none"
    eval_args.use_wandb = False
    eval_args.exp_name = Path(parsed.output_dir).name

    env = DummyVecEnv([lambda: _make_single_env(train_args, seed=int(parsed.seed))])
    vecnormalize_path = _vecnormalize_path_for_model(model_path)
    if vecnormalize_path is not None:
        print(f"[PosthocEval] loading VecNormalize stats: {vecnormalize_path}", flush=True)
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = False
        env.norm_reward = False
    model = PPO.load(str(model_path), env=env, device="auto")

    cb = PeriodicEvalCallback(
        args=eval_args,
        log_dir=Path(parsed.output_dir).expanduser().resolve(),
        wandb_run=None,
        eval_freq=1,
        num_episodes=int(parsed.num_episodes),
        save_plots=True,
        plot_max_episodes=int(parsed.plot_max_episodes),
        reward_surface=bool(parsed.reward_surface),
    )
    cb.model = model
    cb.num_timesteps = int(parsed.step_label)
    cb._run_eval(step=int(parsed.step_label))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
