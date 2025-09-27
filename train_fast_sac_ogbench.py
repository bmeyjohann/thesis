#!/usr/bin/env python3
"""
FastSAC training for OGBench environments using VectorizedOGBenchEnv and teacher interventions.

Matches CLI style of train_rsl_rl_integrated.py where practical.
"""

import os
import sys
import argparse
import time
import json
import copy
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tensordict import TensorDict, from_module

# Add FastSAC path
sys.path.append('fasttd3/fast_sac')

from fast_sac import Actor, Critic
from fast_sac_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    SimpleReplayBuffer,
    save_params,
)

# Register OGBench envs and wrappers
import ogbench
from ogbench.wrappers import FlexibleObsWrapper, DetailedRewardWrapper, InterventionWrapper
from fasttd3.fast_sac.environments.ogbench_env import OGBenchVecEnvAdapter

TOOLS_PATH = Path(__file__).resolve().parent / "tools"
if TOOLS_PATH.exists():
    sys.path.append(str(TOOLS_PATH))

try:
    from visualize_policy_map import generate_policy_map  # type: ignore
except Exception:  # pragma: no cover - optional dependency for viz
    generate_policy_map = None


def parse_args():
    p = argparse.ArgumentParser()
    # Env
    p.add_argument('--env_name', type=str, default='pointmaze-arena-danger-lethal-v0')
    p.add_argument('--num_envs', type=int, default=64)
    p.add_argument('--total_timesteps', type=int, default=1_000_000)
    p.add_argument('--device', type=str, default='auto')
    # Observations
    p.add_argument('--include_goal', action='store_true', default=True)
    p.add_argument('--include_distance', action='store_true', default=False)
    p.add_argument('--include_direction', action='store_true', default=False)
    p.add_argument('--include_velocity', action='store_true', default=False)
    # Rewards
    p.add_argument('--reward_type', type=str, default='sparse', choices=['sparse','dense','combined'])
    p.add_argument('--dense_reward_scale', type=float, default=0.01)
    p.add_argument('--step_penalty', type=float, default=0.0)
    p.add_argument('--reward_switch_after_steps', type=int, default=0)
    # Intervention / Teacher
    p.add_argument('--use_intervention', action='store_true', default=False)
    p.add_argument('--intervention_mode', type=str, default='agent', choices=['human','agent'])
    p.add_argument('--teacher_type', type=str, default='bfs', choices=['bfs'])
    p.add_argument('--tolerance_type', type=str, default='angle', choices=['angle','l2'])
    p.add_argument('--tolerance_value', type=float, default=30.0)
    p.add_argument('--hard_block_lethal', action='store_true', default=True)
    p.add_argument('--no_hard_block_lethal', dest='hard_block_lethal', action='store_false')
    p.add_argument('--intervention_enable_after_steps', type=int, default=0)
    # SAC core (trimmed reasonable defaults)
    p.add_argument('--actor_learning_rate', type=float, default=3e-4)
    p.add_argument('--critic_learning_rate', type=float, default=3e-4)
    p.add_argument('--batch_size', type=int, default=32768)
    p.add_argument('--buffer_size', type=int, default=1024*50)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--policy_frequency', type=int, default=2)
    p.add_argument('--num_updates', type=int, default=2)
    p.add_argument('--learning_starts', type=int, default=1000)
    p.add_argument('--max_grad_norm', type=float, default=0.0)
    p.add_argument('--init_scale', type=float, default=0.01)
    p.add_argument('--actor_hidden_dim', type=int, default=512)
    p.add_argument('--critic_hidden_dim', type=int, default=1024)
    p.add_argument('--store_denied_actions', action='store_true', default=False,
                   help='Add denied student actions to replay buffer with penalty reward')
    p.add_argument('--denied_action_penalty', type=float, default=-1.0,
                   help='Reward assigned to denied student actions when stored')
    # Intervention reward shaping variants
    p.add_argument('--intervention_reward_mode', type=str, default='none',
                   choices=['none', 'penalty_student', 'bonus_teacher'],
                   help='How to shape reward/actions on intervention')
    p.add_argument('--intervention_reward_value', type=float, default=0.0,
                   help='Magnitude for intervention reward shaping (e.g., 0.1)')
    # Logging
    p.add_argument('--use_wandb', action='store_true', default=False)
    p.add_argument('--project', type=str, default='ogbench-rsl-rl')
    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--save_interval', type=int, default=200000,
                   help='Env-step interval for checkpoint saves (0 disables)')
    p.add_argument('--log_interval', type=int, default=200)
    # Misc
    p.add_argument('--compile', action='store_true', default=False)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16','fp16'])
    # Counterfactual buffer (student-denied actions) for fast adaptation
    p.add_argument('--cf_buffer_enable', action='store_true', default=False,
                   help='Enable counterfactual buffer for denied student actions')
    p.add_argument('--cf_capacity', type=int, default=100000,
                   help='Capacity of CF buffer (rows)')
    p.add_argument('--cf_sample_ratio', type=float, default=0.5,
                   help='Fraction of batch for CF critic loss (0..1)')
    p.add_argument('--cf_penalty', type=float, default=1.0,
                   help='Positive penalty magnitude; critic target becomes -abs(value) for denied actions')
    p.add_argument('--cf_q_weight', type=float, default=1.0,
                   help='Weight for CF critic penalty loss')
    # Replay buffer reset on curriculum switch
    p.add_argument('--reset_replay_on_switch', action='store_true', default=False,
                   help='Reset main replay buffer when reward_switch_after_steps is reached')
    p.add_argument('--reset_critic_on_switch', action='store_true', default=False,
                   help='Reload critic weights/optimiser to initial state at curriculum switch')
    # Policy visualization
    p.add_argument('--viz_on_checkpoint', action='store_true', default=False,
                   help='Render policy/critic maps whenever a checkpoint is saved')
    p.add_argument('--viz_grid_resolution', type=int, default=64,
                   help='Grid resolution for policy maps if enabled')
    p.add_argument('--viz_quiver_stride', type=int, default=1,
                   help='Stride for quiver arrows in policy maps')
    p.add_argument('--viz_device', type=str, default='cpu',
                   help='Device to use when generating policy maps')
    p.add_argument('--viz_seed', type=int, default=0,
                   help='Seed used to sample/lock the visualization goal location')
    return p.parse_args()


def make_wrappers(args):
    def _apply(env):
        env = FlexibleObsWrapper(
            env,
            include_goal=args.include_goal,
            include_distance=args.include_distance,
            include_direction=args.include_direction,
            include_velocity=args.include_velocity,
        )
        env = DetailedRewardWrapper(
            env,
            reward_type=args.reward_type,
            dense_reward_scale=args.dense_reward_scale,
            step_penalty=args.step_penalty,
            switch_reward_to_sparse_after_steps_per_env=(args.reward_switch_after_steps // max(1, args.num_envs)),
        )
        if args.use_intervention and args.intervention_mode == 'agent':
            env = InterventionWrapper(
                env,
                mode='agent',
                teacher_type=args.teacher_type,
                tolerance_type=args.tolerance_type,
                tolerance_value=args.tolerance_value,
                hard_block_lethal=args.hard_block_lethal,
                enable_after_steps=args.intervention_enable_after_steps,
            )
        return env
    return [_apply]


def main():
    args = parse_args()
    device = torch.device('cuda' if (args.device=='auto' and torch.cuda.is_available()) or args.device=='cuda' else 'cpu')

    cf_penalty_target = -abs(float(args.cf_penalty))
    setattr(args, 'cf_penalty_target', cf_penalty_target)

    default_buffer_size = 1024 * 50
    if not args.exp_name:
        env_tag = args.env_name.replace('-v0', '').replace('-', '_')
        components = [env_tag]
        components.append(args.reward_type)
        if args.use_intervention:
            components.append(f"teacher_tol{int(args.tolerance_value)}")
            if args.intervention_enable_after_steps > 0:
                components.append(f"warmup{args.intervention_enable_after_steps}")
        else:
            components.append('student')
        if args.buffer_size != default_buffer_size:
            components.append(f"buf{args.buffer_size}")
        if args.store_denied_actions:
            components.append('denied')
        components.append(f"nenv{args.num_envs}")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        components.append(timestamp)
        args.exp_name = '_'.join(components)

    logs_root = Path('logs') / 'fast_sac'
    models_root = Path('models') / 'fast_sac'
    logs_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    run_log_dir = logs_root / args.exp_name
    run_model_dir = models_root / args.exp_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)
    viz_output_dir = run_log_dir / 'policy_maps'
    viz_cache_path = run_log_dir / 'policy_map_goal.json'

    print(f"FastSAC OGBench on {args.env_name} device={device}")
    print(f"Log directory: {run_log_dir}")
    print(f"Model directory: {run_model_dir}")

    # Open progress log early so we can write init breadcrumbs
    log_file_path = run_log_dir / 'training.log'
    progress_file = open(log_file_path, 'a', encoding='utf-8')
    progress_file.write(f"# logging started {datetime.now().isoformat()}\n")
    progress_file.flush()

    def record_progress(message: str):
        progress_file.write(f"{datetime.now().isoformat()} {message}\n")
        progress_file.flush()

    config_path = run_log_dir / 'args.json'
    with open(config_path, 'w', encoding='utf-8') as cfg_file:
        json.dump(vars(args), cfg_file, indent=2)
    print(f"Saved run config: {config_path}")

    wrappers = make_wrappers(args)
    record_progress("[Init] constructing vector env adapter")
    envs = OGBenchVecEnvAdapter(
        env_name=args.env_name,
        num_envs=args.num_envs,
        device=device,
        wrappers=wrappers,
        clip_actions=1.0,
    )
    record_progress("[Init] env adapter constructed")
    print("[Init] Env adapter constructed", flush=True)

    n_obs = envs.num_obs
    n_act = envs.num_actions
    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    critic_obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    reward_normalizer = RewardNormalizer(gamma=args.gamma, device=device, g_max=10.0)

    actor = Actor(n_obs=n_obs, n_act=n_act, num_envs=args.num_envs, init_scale=args.init_scale, hidden_dim=args.actor_hidden_dim, device=device)
    # Match FastTD3 vendor behavior: use a separate "exploration" actor whose
    # parameters share storage with the trainable actor. This keeps rollouts
    # stable while allowing the trainable actor to update.
    actor_detach = Actor(n_obs=n_obs, n_act=n_act, num_envs=args.num_envs, init_scale=args.init_scale, hidden_dim=args.actor_hidden_dim, device=device)
    from_module(actor).data.to_module(actor_detach)
    policy_fn = actor_detach.forward
    qnet = Critic(n_obs=n_obs, n_act=n_act, hidden_dim=args.critic_hidden_dim, device=device)
    qnet_target = Critic(n_obs=n_obs, n_act=n_act, hidden_dim=args.critic_hidden_dim, device=device)
    qnet_target.load_state_dict(qnet.state_dict())

    initial_qnet_state = copy.deepcopy(qnet.state_dict())
    initial_qnet_target_state = copy.deepcopy(qnet_target.state_dict())

    q_optimizer = optim.AdamW(list(qnet.parameters()), lr=args.critic_learning_rate, weight_decay=0.1)
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=args.actor_learning_rate, weight_decay=0.1)

    target_entropy = -float(n_act)
    log_alpha = torch.ones(1, requires_grad=True, device=device)
    log_alpha.data.copy_(torch.tensor([np.log(0.001)], device=device))
    alpha_optimizer = optim.Adam([log_alpha], lr=args.critic_learning_rate)

    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_obs,
        asymmetric_obs=False,
        playground_mode=False,
        n_steps=1,
        gamma=args.gamma,
        device=device,
    )

    # Initialize a simple counterfactual buffer (on-device ring buffer) if enabled
    if args.cf_buffer_enable:
        cf_capacity = int(max(1, args.cf_capacity))
        cf_obs = torch.empty((cf_capacity, n_obs), dtype=torch.float32, device=device)
        cf_act = torch.empty((cf_capacity, n_act), dtype=torch.float32, device=device)
        cf_ptr = 0
        cf_size = 0

        def cf_append(s_batch: torch.Tensor, a_batch: torch.Tensor):
            nonlocal cf_ptr, cf_size
            if s_batch is None or a_batch is None:
                return
            b = int(s_batch.shape[0])
            if b <= 0:
                return
            # If incoming is larger than capacity, keep only the last cf_capacity rows
            if b > cf_capacity:
                s_batch = s_batch[-cf_capacity:]
                a_batch = a_batch[-cf_capacity:]
                b = cf_capacity
            end = cf_ptr + b
            if end <= cf_capacity:
                cf_obs[cf_ptr:end].copy_(s_batch)
                cf_act[cf_ptr:end].copy_(a_batch)
            else:
                first = cf_capacity - cf_ptr
                cf_obs[cf_ptr:].copy_(s_batch[:first])
                cf_act[cf_ptr:].copy_(a_batch[:first])
                remain = b - first
                cf_obs[:remain].copy_(s_batch[first:])
                cf_act[:remain].copy_(a_batch[first:])
            cf_ptr = (cf_ptr + b) % cf_capacity
            cf_size = min(cf_size + b, cf_capacity)

        def cf_sample(b: int):
            if cf_size <= 0:
                return None, None
            idx = torch.randint(low=0, high=cf_size, size=(int(b),), device=device)
            s = cf_obs[idx]
            a = cf_act[idx]
            # Normalize states consistently with main replay samples
            s = _normalize_obs(s)
            return s, a
    else:
        cf_obs = None
        cf_act = None
        cf_ptr = 0
        cf_size = 0

    amp_enabled = args.amp and (device.type=='cuda')
    amp_device_type = 'cuda' if device.type=='cuda' else 'cpu'
    amp_dtype = torch.bfloat16 if args.amp_dtype=='bf16' else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype==torch.float16)

    # Compile optional
    def _normalize_obs(x): return obs_normalizer(x)
    if args.compile:
        actor = torch.compile(actor)
        # Compile the exploration policy function as well
        policy_fn = torch.compile(policy_fn)
        qnet = torch.compile(qnet)
        qnet_target = torch.compile(qnet_target)
        _normalize_obs = torch.compile(_normalize_obs)

    try:
        # Initialize wandb early so runs appear even if logging hasn't triggered yet
        if args.use_wandb:
            import wandb
            wandb_run = wandb.init(
                project=args.project,
                name=args.exp_name,
                id=args.exp_name,
                config=vars(args),
                reinit=True,
                resume="allow",
            )
        else:
            wandb_run = None
        obs = envs.reset()
        record_progress("[Init] envs.reset() returned; entering loop")
        print("[Init] Env reset complete; starting training loop", flush=True)
        total_env_steps = 0
        iteration_idx = 0
        start_time = time.time()
        # wandb_run may be set above
        # Episode buffers similar to RSL-RL
        cur_reward_sum = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
        cur_episode_length = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
        rewbuffer = []
        lenbuffer = []
        last_denied_samples = 0
        next_log_step = args.log_interval if args.log_interval > 0 else None
        next_save_step = args.save_interval if args.save_interval > 0 else None

        run_prefix = args.env_name.replace('-', '_')

        # Learning warm-up threshold (also reused after any replay reset)
        next_learning_starts_at = int(args.learning_starts)

        def save_checkpoint(tag: str, step_value: int):
            save_path = run_model_dir / f"{run_prefix}_{tag}.pt"
            save_params(
                step_value,
                actor,
                qnet,
                qnet_target,
                obs_normalizer,
                critic_obs_normalizer,
                args,
                str(save_path),
            )
            record_progress(f"[Checkpoint] saved {save_path}")
            return save_path

        def maybe_render_policy_map(tag: str, step_value: int, checkpoint_path: Path):
            if not args.viz_on_checkpoint or generate_policy_map is None:
                return
            try:
                png_path, meta_path, _ = generate_policy_map(
                    model_path=checkpoint_path,
                    output_dir=viz_output_dir,
                    tag=tag,
                    env_name=args.env_name,
                    device=args.viz_device,
                    grid_resolution=args.viz_grid_resolution,
                    quiver_stride=args.viz_quiver_stride,
                    seed=args.viz_seed,
                    cache_path=viz_cache_path,
                )
                record_progress(f"[Viz] generated {png_path}")
                if args.use_wandb and wandb_run is not None:
                    import wandb
                    wandb_run.log({
                        f"viz/{tag}": wandb.Image(str(png_path)),
                    }, step=step_value)
            except Exception as exc:  # pragma: no cover - best effort logging
                record_progress(f"[Viz] failed for {tag}: {exc}")

        print("[Init] Starting training loop", flush=True)

        # Track if we have reset the main replay buffer at the curriculum switch
        did_reset_replay = False

        while total_env_steps < args.total_timesteps:
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                norm_obs = _normalize_obs(obs)
                actions, _, _ = policy_fn(norm_obs)
                
            next_obs, rewards, dones, infos = envs.step(actions.float())
            truncations = infos.get('time_outs', torch.zeros_like(dones, device=device))
            applied_actions = infos.get('applied_actions', actions)
            student_actions = infos.get('student_actions')
            teacher_mask = infos.get('teacher_intervened_mask')
            # Build mask if missing by comparing applied vs student actions
            if teacher_mask is None and student_actions is not None and applied_actions is not None:
                try:
                    teacher_mask = (torch.abs(applied_actions - student_actions).sum(dim=-1) > 1e-6)
                except Exception:
                    teacher_mask = None

            # Decide actions to store and reward shaping
            rewards_eff = rewards
            used_actions = applied_actions
            dones_eff = dones
            last_denied_samples = 0
            if teacher_mask is not None:
                denied_ids = torch.nonzero(teacher_mask, as_tuple=False).flatten()
                if denied_ids.numel() > 0:
                    last_denied_samples = int(denied_ids.numel())
                    mode = args.intervention_reward_mode
                    val = float(args.intervention_reward_value)
                    if mode == 'penalty_student':
                        # Store the STUDENT action for intervened rows and apply a negative adjustment
                        if student_actions is not None:
                            used_actions = used_actions.clone()
                            used_actions[denied_ids] = student_actions[denied_ids]
                        if val != 0.0:
                            rewards_eff = rewards_eff.clone()
                            rewards_eff[denied_ids] = rewards_eff[denied_ids] - abs(val)
                        # Mark these rows as terminal so bootstrapping stops
                        dones_eff = dones_eff.clone()
                        dones_eff[denied_ids] = 1
                    elif mode == 'bonus_teacher' and val != 0.0:
                        rewards_eff = rewards_eff.clone()
                        rewards_eff[denied_ids] = rewards_eff[denied_ids] + abs(val)
            
            # Build transition
            obs_detached = obs.detach()
            next_obs_detached = next_obs.detach()
            transition = TensorDict(
                {
                    'observations': obs_detached,
                    'actions': used_actions.detach(),
                    'next': {
                        'observations': next_obs_detached,
                        'rewards': rewards_eff.detach(),
                        'truncations': truncations.long(),
                        'dones': dones_eff.long(),
                    },
                },
                batch_size=(envs.num_envs,),
                device=device,
            )
            rb.extend(transition)

            # Append counterfactual rows to CF buffer (student-denied actions)
            if args.cf_buffer_enable and teacher_mask is not None and student_actions is not None and cf_obs is not None:
                denied_ids = torch.nonzero(teacher_mask, as_tuple=False).flatten()
                if denied_ids.numel() > 0:
                    # Store (s, a_student)
                    cf_s = obs_detached[denied_ids]
                    cf_a = student_actions[denied_ids]
                    # Initialize CF tensors on first use
                    cf_append(cf_s, cf_a)
            
            # We already folded denied penalties into rewards_eff; just expose count via logging

            # Book-keeping for episode stats
            cur_reward_sum += rewards
            cur_episode_length += 1
            done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                rewbuffer += cur_reward_sum[done_ids].tolist()
                lenbuffer += cur_episode_length[done_ids].tolist()
                cur_reward_sum[done_ids] = 0
                cur_episode_length[done_ids] = 0

            # Learn
            iteration_idx += 1
            total_env_steps += envs.num_envs

            # Optionally reset main replay buffer once at reward switch (curriculum)
            if (
                args.reset_replay_on_switch
                and not did_reset_replay
                and args.reward_switch_after_steps > 0
                and total_env_steps >= args.reward_switch_after_steps
            ):
                record_progress(f"[Replay] Resetting main replay buffer at step {total_env_steps}")
                rb = SimpleReplayBuffer(
                    n_env=args.num_envs,
                    buffer_size=args.buffer_size,
                    n_obs=n_obs,
                    n_act=n_act,
                    n_critic_obs=n_obs,
                    asymmetric_obs=False,
                    playground_mode=False,
                    n_steps=1,
                    gamma=args.gamma,
                    device=device,
                )
                did_reset_replay = True
                # Reapply warm-up: postpone learning by the same number of steps as initially
                next_learning_starts_at = total_env_steps + int(args.learning_starts)
                record_progress(f"[Replay] Post-reset warm-up: learning resumes at env_step >= {next_learning_starts_at}")

                if args.reset_critic_on_switch:
                    record_progress(f"[Critic] Resetting critic weights/optimizer at step {total_env_steps}")
                    qnet.load_state_dict(initial_qnet_state)
                    qnet_target.load_state_dict(initial_qnet_target_state)
                    q_optimizer = optim.AdamW(list(qnet.parameters()), lr=args.critic_learning_rate, weight_decay=0.1)

            if next_save_step is not None and total_env_steps >= next_save_step:
                tag_name = f"step{total_env_steps}"
                ckpt_path = save_checkpoint(tag_name, total_env_steps)
                maybe_render_policy_map(tag_name, total_env_steps, ckpt_path)
                next_save_step += args.save_interval

            # Only learn if we have enough data in replay (also after any reset)
            if total_env_steps >= next_learning_starts_at and getattr(rb, 'ptr', 0) > 0:
                batch_size = args.batch_size // max(1, args.num_envs)
                for i in range(args.num_updates):
                    data = rb.sample(batch_size)
                    # Normalize obs
                    data['observations'] = _normalize_obs(data['observations'])
                    data['next']['observations'] = _normalize_obs(data['next']['observations'])

                    # Critic update
                    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                        next_pi, next_log_pi, _ = actor(data['next']['observations'])
                        q1_t, q2_t = qnet_target(data['next']['observations'], next_pi)
                        min_q_t = torch.min(q1_t, q2_t) - log_alpha.exp() * next_log_pi
                        next_q = data['next']['rewards'].unsqueeze(-1) + (1.0 - data['next']['dones'].float()).unsqueeze(-1) * (args.gamma * min_q_t)
                        q1, q2 = qnet(data['observations'], data['actions'])
                        qf_loss = F.mse_loss(q1, next_q) + F.mse_loss(q2, next_q)
                        # Counterfactual critic penalty
                        if args.cf_buffer_enable and args.cf_q_weight > 0.0 and args.cf_sample_ratio > 0.0 and 'cf_size' in locals() and cf_size > 0:
                            cf_b = max(1, int((args.batch_size // max(1, args.num_envs)) * args.cf_sample_ratio))
                            s_cf, a_cf = cf_sample(cf_b)
                            if s_cf is not None:
                                q1_cf, q2_cf = qnet(s_cf, a_cf)
                                y_bad = torch.full_like(q1_cf, cf_penalty_target)
                                qf_loss = qf_loss + args.cf_q_weight * (F.mse_loss(q1_cf, y_bad) + F.mse_loss(q2_cf, y_bad))

                    q_optimizer.zero_grad(set_to_none=True)
                    scaler.scale(qf_loss).backward()
                    scaler.unscale_(q_optimizer)
                    torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float('inf'))
                    scaler.step(q_optimizer)
                    scaler.update()

                    # Actor update
                    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                        pi, log_pi, _ = actor(data['observations'])
                        q1_pi, q2_pi = qnet(data['observations'], pi)
                        q_pi = torch.min(q1_pi, q2_pi)
                        actor_loss = (log_alpha.exp().detach() * log_pi - q_pi).mean()

                    actor_optimizer.zero_grad(set_to_none=True)
                    scaler.scale(actor_loss).backward()
                    scaler.unscale_(actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float('inf'))
                    scaler.step(actor_optimizer)
                    scaler.update()

                    # Alpha update
                    alpha_optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        _, log_pi_curr, _ = actor(data['observations'])
                    alpha_loss = -log_alpha.exp() * (log_pi_curr + (-float(n_act))).detach().mean()
                    scaler.scale(alpha_loss).backward()
                    scaler.unscale_(alpha_optimizer)
                    alpha_optimizer.step()

                    # Soft update target
                    for p, tp in zip(qnet.parameters(), qnet_target.parameters()):
                        tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

            # Logging (RSL-RL style)
            should_log = False
            if next_log_step is not None and total_env_steps >= next_log_step:
                should_log = True
                next_log_step += args.log_interval
            if total_env_steps >= args.total_timesteps:
                should_log = True

            if should_log:
                collection_time = time.time() - start_time
                fps = int(total_env_steps / max(1e-6, collection_time))
                logs = {
                    'Perf/total_fps': fps,
                    'Perf/collection_time_sec': collection_time,
                    'Perf/env_steps': total_env_steps,
                    'Perf/iterations': iteration_idx,
                }
                if len(rewbuffer) > 0:
                    logs['Train/mean_reward'] = float(np.mean(rewbuffer[-100:]))
                    logs['Train/mean_episode_length'] = float(np.mean(lenbuffer[-100:]))
                if 'log' in infos and isinstance(infos['log'], dict):
                    for k, v in infos['log'].items():
                        try:
                            logs[k] = float(v.float().mean().item())
                        except Exception:
                            pass
                if args.store_denied_actions:
                    logs['/Teacher/denied_transition_samples'] = float(last_denied_samples)

                log_line_parts = [
                    f"env_steps {total_env_steps}/{args.total_timesteps}",
                    f"iter {iteration_idx}",
                    f"fps {fps}",
                ]
                if 'Train/mean_reward' in logs:
                    log_line_parts.append(f"mean_reward {logs['Train/mean_reward']:.2f}")
                if '/Episode/goal_success_rate' in logs:
                    log_line_parts.append(f"success {logs['/Episode/goal_success_rate']:.2f}")
                if '/Teacher/teacher_fraction_steps' in logs:
                    log_line_parts.append(f"teacher_frac {logs['/Teacher/teacher_fraction_steps']:.2f}")
                if args.store_denied_actions and last_denied_samples > 0:
                    log_line_parts.append(f"denied {last_denied_samples}")
                console_line = "[FastSAC] " + " | ".join(log_line_parts)
                print(console_line, flush=True)
                record_progress(console_line)

                if args.use_wandb:
                    import wandb
                    if wandb_run is None:
                        wandb_run = wandb.init(
                            project=args.project,
                            name=args.exp_name,
                            id=args.exp_name,
                            config=vars(args),
                            reinit=True,
                            resume="allow",
                        )
                    wandb_run.log(logs, step=total_env_steps)

            obs = next_obs
        # Save final
        final_ckpt = save_checkpoint('final', total_env_steps)
        maybe_render_policy_map('final', total_env_steps, final_ckpt)
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
        total_time = time.time() - start_time
        summary_line = (
            "✅ FastSAC training complete"
            f" env_steps={total_env_steps}"
            f" iterations={iteration_idx}"
            f" duration_sec={total_time:.1f}"
            f" models_dir={run_model_dir}"
        )
        print("=" * 80)
        print(summary_line)
        record_progress(summary_line)
    finally:
        try:
            progress_file.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
