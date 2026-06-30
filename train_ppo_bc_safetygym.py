#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.dataset_io import load_transition_dataset
from safetygym_utils.io import save_args_json
from safetygym_utils.minimal_train import _make_env_with_wrappers
from safetygym_utils.wrappers import fixed_layout_preset_names, layout_curriculum_names
from train_ppo_safetygym_minimal import SafetyGymnasiumToGymnasium, _activation_fn


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Behavior-clone a SB3 PPO Safety-Gym policy from transition data.")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--env_name", type=str, default="SafetyCarGoal1-v0")
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--output_dir", type=str, default="models/safetygym_ppo_bc")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--net_arch", type=str, default="256,256")
    p.add_argument("--activation_fn", type=str, default="tanh", choices=["tanh", "relu", "elu"])
    p.add_argument("--initial_log_std", type=float, default=-2.0)
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=1.0)
    p.add_argument("--car_force_scale", type=float, default=1.0)
    p.add_argument("--car_action_mode", type=str, default="raw_wheels", choices=["raw_wheels", "throttle_turn", "cardinal"])
    p.add_argument("--point_action_mode", type=str, default="native", choices=["native", "world_velocity"])
    p.add_argument("--point_turn_gain", type=float, default=2.5)
    p.add_argument("--point_alignment_power", type=float, default=1.0)
    p.add_argument("--point_allow_backward", action="store_true", default=False)
    p.add_argument("--obs_mask_mode", type=str, default="none", choices=["none", "goal_only_lidar", "privileged_geometry", "privileged_geometry_rich"])
    p.add_argument("--fixed_layout_preset", type=str, default="none", choices=["none", *fixed_layout_preset_names()])
    p.add_argument("--layout_curriculum", type=str, default="none", choices=["none", *layout_curriculum_names()])
    p.add_argument("--layout_curriculum_level", type=int, default=0)
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--terminate_on_goal", action="store_true", default=False)
    p.add_argument("--terminate_on_cost", action="store_true", default=False)
    p.add_argument("--reseed_on_episode_reset", action="store_true", default=False)
    p.add_argument("--no_reseed_on_episode_reset", dest="reseed_on_episode_reset", action="store_false")
    p.add_argument("--reward_mode", type=str, default="dense_plus_sparse", choices=["sparse", "dense", "dense_plus_sparse", "potential_diff", "native", "none"])
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--success_reward_scale", type=float, default=5.0)
    p.add_argument("--step_penalty", type=float, default=-0.001)
    p.add_argument("--cost_penalty", type=float, default=0.0)
    p.add_argument("--cost_penalty_warmup_steps", type=int, default=0)
    p.add_argument("--cost_penalty_ramp_steps", type=int, default=0)
    p.add_argument("--clearance_penalty_scale", type=float, default=4.0)
    p.add_argument("--clearance_margin", type=float, default=0.0)
    p.add_argument("--clearance_penalty_power", type=float, default=1.0)
    p.add_argument("--clearance_penalty_mode", type=str, default="softplus", choices=["hinge_power", "softplus"])
    p.add_argument("--clearance_penalty_temperature", type=float, default=0.001)
    p.add_argument("--clearance_penalty_warmup_steps", type=int, default=0)
    p.add_argument("--clearance_penalty_ramp_steps", type=int, default=0)
    p.add_argument("--forward_reward_scale", type=float, default=0.0)
    p.add_argument("--backward_penalty_scale", type=float, default=0.0)
    p.add_argument("--heading_reward_scale", type=float, default=0.0)
    p.add_argument("--heading_positive_only", action="store_true", default=True)
    p.add_argument("--no_heading_positive_only", dest="heading_positive_only", action="store_false")
    return p


def _make_single_env(args, seed: int):
    env = _make_env_with_wrappers(args=args, seed=seed, with_intervention=False, controller=None)
    return Monitor(SafetyGymnasiumToGymnasium(env))


def _split_indices(n_rows: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n_rows, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(round(n_rows * min(max(float(val_fraction), 0.0), 0.9)))
    return idx[n_val:], idx[:n_val]


def _eval_mse(policy, obs: torch.Tensor, actions: torch.Tensor, batch_size: int) -> float:
    policy.eval()
    total = 0.0
    rows = 0
    with torch.inference_mode():
        for start in range(0, int(obs.shape[0]), int(batch_size)):
            stop = min(int(obs.shape[0]), start + int(batch_size))
            dist = policy.get_distribution(obs[start:stop])
            pred = dist.distribution.mean
            loss = F.mse_loss(pred, actions[start:stop], reduction="sum")
            total += float(loss.detach().cpu())
            rows += int(stop - start)
    return float(total / max(1, rows))


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    data = load_transition_dataset(args.dataset_path)
    obs = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if int(args.max_rows) > 0:
        obs = obs[: int(args.max_rows)]
        actions = actions[: int(args.max_rows)]

    exp_name = str(args.exp_name or f"ppo_bc_{args.env_name.replace('-', '_')}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir = Path(args.output_dir).expanduser() / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_args_json(out_dir / "args.json", vars(args))

    env = DummyVecEnv([lambda: _make_single_env(args, seed=int(args.seed))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=0.99)
    env.obs_rms.update(obs.astype(np.float64))
    env.training = False
    env.norm_obs = True
    env.norm_reward = False

    net_arch = [int(x.strip()) for x in str(args.net_arch).split(",") if x.strip()]
    model = PPO(
        "MlpPolicy",
        env,
        seed=int(args.seed),
        device=device,
        verbose=0,
        n_steps=512,
        batch_size=512,
        n_epochs=1,
        gamma=0.99,
        policy_kwargs={"net_arch": {"pi": net_arch, "vf": net_arch}, "activation_fn": _activation_fn(args.activation_fn)},
    )
    model.policy.log_std.data.fill_(float(args.initial_log_std))
    opt = torch.optim.AdamW(model.policy.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    norm_obs = env.normalize_obs(obs.copy()).astype(np.float32)
    obs_t = torch.as_tensor(norm_obs, device=device, dtype=torch.float32)
    act_t = torch.as_tensor(actions, device=device, dtype=torch.float32)
    train_idx, val_idx = _split_indices(int(obs_t.shape[0]), float(args.val_fraction), int(args.seed))
    train_idx_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    val_idx_t = torch.as_tensor(val_idx, device=device, dtype=torch.long)
    best = float("inf")
    best_path = out_dir / "best.zip"
    n_train = int(train_idx_t.numel())
    for epoch in range(1, int(args.epochs) + 1):
        perm = train_idx_t[torch.randperm(n_train, device=device)]
        model.policy.train()
        total = 0.0
        rows = 0
        for start in range(0, n_train, int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            dist = model.policy.get_distribution(obs_t[idx])
            pred = dist.distribution.mean
            action_loss = F.mse_loss(pred, act_t[idx])
            opt.zero_grad(set_to_none=True)
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 10.0)
            opt.step()
            total += float(action_loss.detach().cpu()) * int(idx.numel())
            rows += int(idx.numel())
        train_mse = float(total / max(1, rows))
        val_mse = _eval_mse(model.policy, obs_t[val_idx_t], act_t[val_idx_t], int(args.batch_size)) if int(val_idx_t.numel()) else train_mse
        print(json.dumps({"epoch": epoch, "train_action_mse": train_mse, "val_action_mse": val_mse, "rows": int(obs_t.shape[0])}, sort_keys=True), flush=True)
        if val_mse < best:
            best = float(val_mse)
            model.save(str(best_path))
            env.save(str(out_dir / "best_vecnormalize.pkl"))
            # Keep shared PPO evaluators simple: they look for vecnormalize.pkl
            # next to arbitrary .zip checkpoints that are not step-named.
            env.save(str(out_dir / "vecnormalize.pkl"))

    model.save(str(out_dir / "final.zip"))
    env.save(str(out_dir / "vecnormalize.pkl"))
    (out_dir / "summary.json").write_text(json.dumps({"best_val_action_mse": best, "rows": int(obs_t.shape[0])}, indent=2, sort_keys=True) + "\n")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
