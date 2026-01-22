#!/usr/bin/env python3
"""
FastSAC training entrypoint for HighwayEnv continuous-control tasks.

This is intentionally minimal and mirrors the logging/run structure used by
train_fast_sac_ogbench.py while relying only on the pieces we need.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")

# Make HighwayEnv and FastSAC importable without installation.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

FAST_SAC_PATH = Path(__file__).resolve().parent / "fasttd3" / "fast_sac"
HIGHWAY_PATH = Path(__file__).resolve().parent / "HighwayEnv"
if FAST_SAC_PATH.exists():
    os.sys.path.append(str(FAST_SAC_PATH))
if HIGHWAY_PATH.exists():
    os.sys.path.append(str(HIGHWAY_PATH))

from fast_sac import Actor, Critic  # noqa: E402
from fast_sac_utils import EmpiricalNormalization, SimpleReplayBuffer  # noqa: E402

import highway_env  # noqa: F401  # Registers gymnasium environments.  # noqa: E402

try:  # noqa: E402
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - gymnasium expected in runtime env
    raise ImportError("gymnasium is required for HighwayEnv training") from exc

try:  # noqa: E402
    import wandb
except Exception:  # pragma: no cover - optional
    wandb = None


torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastSAC for HighwayEnv (continuous control).")
    parser.add_argument("--env_name", type=str, default="highway-v0")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=20_000_000)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_eval_envs", type=int, default=4)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200_000)
    parser.add_argument("--log_interval", type=int, default=50_000)
    parser.add_argument("--save_interval", type=int, default=500_000)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=32_768)
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--num_updates", type=int, default=2)
    parser.add_argument("--actor_hidden_dim", type=int, default=512)
    parser.add_argument("--critic_hidden_dim", type=int, default=1024)
    parser.add_argument("--actor_learning_rate", type=float, default=3e-4)
    parser.add_argument("--critic_learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--init_scale", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--obs_normalization", action="store_true", default=True)
    parser.add_argument("--no_obs_normalization", action="store_false", dest="obs_normalization")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--project", type=str, default="fastsac_highway")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--device_rank", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=("bf16", "fp16"))
    parser.add_argument("--disable_bootstrap", action="store_true", default=False)

    parser.add_argument("--lanes_count", type=int, default=4)
    parser.add_argument("--traffic_vehicles", type=int, default=50)
    parser.add_argument("--obs_vehicles", type=int, default=15)
    parser.add_argument("--duration", type=int, default=40)
    parser.add_argument("--collision_reward", type=float, default=-2.0)
    parser.add_argument("--right_lane_reward", type=float, default=0.0)
    parser.add_argument("--high_speed_reward", type=float, default=1.0)
    parser.add_argument("--lane_change_reward", type=float, default=0.0)
    parser.add_argument("--reward_speed_min", type=float, default=20.0)
    parser.add_argument("--reward_speed_max", type=float, default=30.0)
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--offroad_terminal", action="store_true", default=False)
    parser.add_argument("--offroad_penalty", type=float, default=0.0)
    return parser.parse_args()


def ensure_experiment_name(args: argparse.Namespace) -> None:
    if args.exp_name:
        return
    env_tag = args.env_name.replace("-v0", "").replace("-", "_")
    components = [
        env_tag,
        f"nenv{args.num_envs}",
        f"steps{args.total_timesteps}",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ]
    args.exp_name = "_".join(components)


def prepare_run_dirs(args: argparse.Namespace) -> Tuple[Path, Path, callable]:
    logs_root = Path("logs") / "fast_sac_highway"
    models_root = Path("models") / "fast_sac_highway"
    logs_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    run_log_dir = logs_root / args.exp_name
    run_model_dir = models_root / args.exp_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = run_log_dir / "training.log"
    progress_file = open(log_file_path, "a", encoding="utf-8")
    progress_file.write(f"# logging started {datetime.now().isoformat()}\n")
    progress_file.flush()

    def record_progress(message: str) -> None:
        progress_file.write(f"{datetime.now().isoformat()} {message}\n")
        progress_file.flush()

    config_path = run_log_dir / "args.json"
    with open(config_path, "w", encoding="utf-8") as cfg_file:
        json.dump(vars(args), cfg_file, indent=2)
    print(f"Saved run config: {config_path}")

    return run_log_dir, run_model_dir, record_progress


def build_env_config(args: argparse.Namespace) -> Dict:
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": args.obs_vehicles,
        },
        "action": {"type": "ContinuousAction"},
        "lanes_count": args.lanes_count,
        "vehicles_count": args.traffic_vehicles,
        "duration": args.duration,
        "collision_reward": args.collision_reward,
        "right_lane_reward": args.right_lane_reward,
        "high_speed_reward": args.high_speed_reward,
        "lane_change_reward": args.lane_change_reward,
        "reward_speed_range": [args.reward_speed_min, args.reward_speed_max],
        "normalize_reward": args.normalize_reward,
        "offroad_terminal": args.offroad_terminal,
    }


def make_env(env_name: str, config: Dict, seed: int | None = None) -> gym.Env:
    env = gym.make(env_name)
    env.unwrapped.configure(config)
    env = gym.wrappers.FlattenObservation(env)
    env.reset(seed=seed)
    return env


def reset_envs(envs: List[gym.Env], seed: int | None) -> Tuple[np.ndarray, List[Dict]]:
    observations = []
    infos: List[Dict] = []
    for idx, env in enumerate(envs):
        obs, info = env.reset(seed=None if seed is None else seed + idx)
        observations.append(obs)
        infos.append(info)
    return np.asarray(observations, dtype=np.float32), infos


def step_envs(
    envs: List[gym.Env],
    actions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    observations = []
    rewards = []
    terminated = []
    truncated = []
    infos: List[Dict] = []
    for env, action in zip(envs, actions):
        obs, reward, term, trunc, info = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        terminated.append(term)
        truncated.append(trunc)
        infos.append(info)
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(terminated, dtype=bool),
        np.asarray(truncated, dtype=bool),
        infos,
    )


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.lerp_(param.data, tau)


@dataclass
class UpdateMetrics:
    qf_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    qf_min: float
    qf_max: float


def run_eval(
    envs: List[gym.Env],
    actor: Actor,
    obs_normalizer: EmpiricalNormalization | torch.nn.Identity,
    device: torch.device,
    num_episodes: int,
    seed: int,
) -> Dict[str, float]:
    obs_normalizer.eval()
    returns: List[float] = []
    lengths: List[int] = []
    crashes: List[int] = []
    speeds: List[float] = []

    obs, _ = reset_envs(envs, seed)
    ep_returns = np.zeros(len(envs), dtype=np.float32)
    ep_lengths = np.zeros(len(envs), dtype=np.int32)
    while len(returns) < num_episodes:
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
        with torch.no_grad():
            norm_obs = obs_normalizer(obs_tensor)
            _, _, action_mean = actor(norm_obs)
        actions = action_mean.cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = step_envs(envs, actions)
        done = terminated | truncated
        ep_returns += rewards
        ep_lengths += 1
        for idx, is_done in enumerate(done):
            if not is_done:
                continue
            returns.append(float(ep_returns[idx]))
            lengths.append(int(ep_lengths[idx]))
            crash = int(bool(infos[idx].get("crashed", False)))
            crashes.append(crash)
            speeds.append(float(infos[idx].get("speed", 0.0)))
            ep_returns[idx] = 0.0
            ep_lengths[idx] = 0
            obs_reset, _ = envs[idx].reset(seed=seed + 1000 + idx)
            next_obs[idx] = obs_reset
        obs = next_obs

    obs_normalizer.train()
    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/ep_len_mean": float(np.mean(lengths)),
        "eval/crash_rate": float(np.mean(crashes)) if crashes else 0.0,
        "eval/speed_mean": float(np.mean(speeds)) if speeds else 0.0,
    }


def main() -> None:
    args = parse_args()
    ensure_experiment_name(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise RuntimeError("No GPU available")
    print(f"Using device: {device}")

    run_log_dir, run_model_dir, record_progress = prepare_run_dirs(args)
    record_progress(f"[Init] exp_name={args.exp_name} env={args.env_name} num_envs={args.num_envs}")
    print(f"Log directory: {run_log_dir}")
    print(f"Model directory: {run_model_dir}")

    env_config = build_env_config(args)
    envs = [make_env(args.env_name, env_config, seed=args.seed + i) for i in range(args.num_envs)]
    eval_envs = [make_env(args.env_name, env_config, seed=args.seed + 100 + i) for i in range(args.num_eval_envs)]

    obs, _ = reset_envs(envs, args.seed)
    obs_dim = int(np.prod(envs[0].observation_space.shape))
    act_dim = int(np.prod(envs[0].action_space.shape))

    obs_normalizer: EmpiricalNormalization | torch.nn.Identity
    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    else:
        obs_normalizer = torch.nn.Identity()

    actor = Actor(
        n_obs=obs_dim,
        n_act=act_dim,
        num_envs=args.num_envs,
        init_scale=args.init_scale,
        hidden_dim=args.actor_hidden_dim,
        device=device,
    )
    critic = Critic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=args.critic_hidden_dim,
        device=device,
    )
    critic_target = Critic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=args.critic_hidden_dim,
        device=device,
    )
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=args.actor_learning_rate,
        weight_decay=args.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_learning_rate,
        weight_decay=args.weight_decay,
    )

    target_entropy = -float(act_dim)
    log_alpha = torch.ones(1, requires_grad=True, device=device)
    log_alpha.data.copy_(torch.tensor([np.log(0.001)], device=device))
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.critic_learning_rate)

    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=obs_dim,
        n_act=act_dim,
        n_critic_obs=obs_dim,
        asymmetric_obs=False,
        playground_mode=False,
        n_steps=1,
        gamma=args.gamma,
        device=device,
        pixel_shape=None,
    )

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        log_alpha.data.copy_(checkpoint["log_alpha"])
        record_progress(f"[Init] loaded checkpoint {args.checkpoint_path}")

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    if args.use_wandb and wandb is not None:
        wandb.init(project=args.project, name=args.exp_name, config=vars(args), save_code=True)

    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_crashes = deque(maxlen=100)
    ep_return = np.zeros(args.num_envs, dtype=np.float32)
    ep_length = np.zeros(args.num_envs, dtype=np.int32)
    speed_accum = 0.0
    speed_count = 0
    lane_counts = np.zeros(args.lanes_count, dtype=np.int64)
    lane_total = 0

    next_log_step = args.log_interval if args.log_interval > 0 else None
    next_eval_step = args.eval_interval if args.eval_interval > 0 else None
    next_save_step = args.save_interval if args.save_interval > 0 else None

    global_step = 0
    start_time = time.time()
    record_progress("[Train] start")

    while global_step < args.total_timesteps:
        obs_batch = obs.copy()
        if global_step < args.learning_starts:
            actions = np.asarray([env.action_space.sample() for env in envs], dtype=np.float32)
        else:
            obs_tensor = torch.as_tensor(obs_batch, device=device, dtype=torch.float32)
            with torch.no_grad(), torch.autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                norm_obs = obs_normalizer(obs_tensor)
                actions_t, _, _ = actor(norm_obs)
            actions = actions_t.cpu().numpy()
        actions = np.clip(actions, -1.0, 1.0)

        next_obs, rewards, terminated, truncated, infos = step_envs(envs, actions)
        done = terminated | truncated

        for idx, info in enumerate(infos):
            speed_accum += float(info.get("speed", 0.0))
            speed_count += 1
            try:
                lane_index = envs[idx].unwrapped.vehicle.lane_index
                if lane_index is not None:
                    lane_id = int(lane_index[2])
                    if 0 <= lane_id < args.lanes_count:
                        lane_counts[lane_id] += 1
                        lane_total += 1
                if args.offroad_penalty != 0.0 and not envs[idx].unwrapped.vehicle.on_road:
                    rewards[idx] += float(args.offroad_penalty)
            except Exception:
                pass

        transition = TensorDict(
            {
                "observations": torch.as_tensor(obs_batch, device=device, dtype=torch.float32),
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float32),
                "next": {
                    "observations": torch.as_tensor(next_obs, device=device, dtype=torch.float32),
                    "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float32),
                    "dones": torch.as_tensor(done, device=device, dtype=torch.bool),
                    "truncations": torch.as_tensor(truncated, device=device, dtype=torch.bool),
                    "effective_n_steps": torch.ones(args.num_envs, device=device, dtype=torch.float32),
                },
            },
            batch_size=(args.num_envs,),
            device=device,
        )
        rb.extend(transition)

        ep_return += rewards
        ep_length += 1
        for idx, is_done in enumerate(done):
            if not is_done:
                continue
            episode_returns.append(float(ep_return[idx]))
            episode_lengths.append(int(ep_length[idx]))
            episode_crashes.append(int(bool(infos[idx].get("crashed", False))))
            ep_return[idx] = 0.0
            ep_length[idx] = 0
            obs_reset, _ = envs[idx].reset(seed=args.seed + 10000 + global_step + idx)
            next_obs[idx] = obs_reset

        obs = next_obs
        global_step += args.num_envs

        metrics: UpdateMetrics | None = None
        if global_step > args.learning_starts and rb.size >= args.batch_size:
            per_env_batch = max(1, args.batch_size // args.num_envs)
            for update_idx in range(args.num_updates):
                data = rb.sample(per_env_batch)
                if args.obs_normalization:
                    data["observations"] = obs_normalizer(data["observations"])
                    data["next"]["observations"] = obs_normalizer(data["next"]["observations"])

                with torch.autocast(
                    device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    observations = data["observations"]
                    next_observations = data["next"]["observations"]
                    actions_t = data["actions"]
                    rewards_t = data["next"]["rewards"].unsqueeze(-1)
                    dones_t = data["next"]["dones"].bool().unsqueeze(-1)
                    trunc_t = data["next"]["truncations"].bool().unsqueeze(-1)
                    if args.disable_bootstrap:
                        bootstrap = (~dones_t).float()
                    else:
                        bootstrap = (trunc_t | ~dones_t).float()
                    discount = args.gamma ** data["next"]["effective_n_steps"].unsqueeze(-1)

                    with torch.no_grad():
                        next_actions, next_log_pi, _ = actor(next_observations)
                        q1_next, q2_next = critic_target(next_observations, next_actions)
                        min_q_next = torch.min(q1_next, q2_next) - log_alpha.exp() * next_log_pi
                        target_q = rewards_t + bootstrap * discount * min_q_next

                    q1, q2 = critic(observations, actions_t)
                    qf_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                critic_optimizer.zero_grad(set_to_none=True)
                scaler.scale(qf_loss).backward()
                scaler.unscale_(critic_optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                scaler.step(critic_optimizer)
                scaler.update()

                do_policy = (
                    update_idx % args.policy_frequency == 1
                    if args.num_updates > 1
                    else global_step % args.policy_frequency == 0
                )
                if do_policy:
                    with torch.autocast(
                        device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        pi, log_pi, _ = actor(observations)
                        q1_pi, q2_pi = critic(observations, pi)
                        q_pi = torch.min(q1_pi, q2_pi)
                        actor_loss = ((log_alpha.exp().detach() * log_pi) - q_pi).mean()

                    actor_optimizer.zero_grad(set_to_none=True)
                    scaler.scale(actor_loss).backward()
                    scaler.unscale_(actor_optimizer)
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                    scaler.step(actor_optimizer)
                    scaler.update()

                    alpha_optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        _, log_pi, _ = actor(observations)
                    alpha_loss = -log_alpha.exp() * (log_pi + target_entropy).detach().mean()
                    scaler.scale(alpha_loss).backward()
                    scaler.unscale_(alpha_optimizer)
                    alpha_optimizer.step()
                else:
                    actor_loss = torch.tensor(0.0, device=device)
                    alpha_loss = torch.tensor(0.0, device=device)

                soft_update(critic, critic_target, args.tau)

                metrics = UpdateMetrics(
                    qf_loss=float(qf_loss.detach().cpu().item()),
                    actor_loss=float(actor_loss.detach().cpu().item()),
                    alpha_loss=float(alpha_loss.detach().cpu().item()),
                    alpha=float(log_alpha.exp().detach().cpu().item()),
                    qf_min=float(target_q.min().detach().cpu().item()),
                    qf_max=float(target_q.max().detach().cpu().item()),
                )

        if next_log_step is not None and global_step >= next_log_step:
            next_log_step += args.log_interval
            elapsed = max(1e-6, time.time() - start_time)
            fps = global_step / elapsed
            avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
            avg_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
            crash_rate = float(np.mean(episode_crashes)) if episode_crashes else 0.0
            mean_speed = speed_accum / max(1, speed_count)
            log_items = {
                "step": global_step,
                "fps": fps,
                "buffer_size": rb.size,
                "train/return_mean": avg_return,
                "train/ep_len_mean": avg_len,
                "train/crash_rate": crash_rate,
                "train/speed_mean": mean_speed,
            }
            if lane_total > 0:
                lane_fracs = lane_counts / max(1, lane_total)
                for lane_id, frac in enumerate(lane_fracs):
                    log_items[f"train/lane_frac_{lane_id}"] = float(frac)
                entropy = float(-np.sum(lane_fracs * np.log(lane_fracs + 1e-8)))
                log_items["train/lane_entropy"] = entropy
            if metrics is not None:
                log_items.update(
                    {
                        "train/qf_loss": metrics.qf_loss,
                        "train/actor_loss": metrics.actor_loss,
                        "train/alpha_loss": metrics.alpha_loss,
                        "train/alpha": metrics.alpha,
                        "train/qf_min": metrics.qf_min,
                        "train/qf_max": metrics.qf_max,
                    }
                )
            record_progress(f"[Log] {json.dumps(log_items, sort_keys=True)}")
            print(log_items, flush=True)
            if args.use_wandb and wandb is not None:
                wandb.log(log_items, step=global_step)
            speed_accum = 0.0
            speed_count = 0
            lane_counts.fill(0)
            lane_total = 0

        if next_eval_step is not None and global_step >= next_eval_step:
            next_eval_step += args.eval_interval
            eval_metrics = run_eval(
                eval_envs,
                actor,
                obs_normalizer,
                device,
                args.num_eval_episodes,
                args.seed + 5000 + global_step,
            )
            record_progress(f"[Eval] {json.dumps(eval_metrics, sort_keys=True)}")
            print(eval_metrics, flush=True)
            if args.use_wandb and wandb is not None:
                wandb.log(eval_metrics, step=global_step)

        if next_save_step is not None and global_step >= next_save_step:
            next_save_step += args.save_interval
            ckpt_path = run_model_dir / f"step_{global_step}.pt"
            torch.save(
                {
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "critic_target_state_dict": critic_target.state_dict(),
                    "obs_normalizer_state": obs_normalizer.state_dict(),
                    "log_alpha": log_alpha.detach().cpu(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            record_progress(f"[Checkpoint] saved {ckpt_path}")

    record_progress("[Train] finished")


if __name__ == "__main__":
    main()
