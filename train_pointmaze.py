import os
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import ogbench

import sys
sys.path.append('fasttd3/fast_td3')

from fast_td3_utils import SimpleReplayBuffer
from hyperparams import get_args
from fast_td3 import Actor, Critic

def main():
    args = get_args()
    args.env_name = "pointmaze-medium-v0"  # Changed to OGBench environment
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_envs = 16  # More parallel environments for even faster training
    args.total_timesteps = 100_000
    args.learning_starts = 1_000
    args.batch_size = 256
    args.buffer_size = 1_000_000
    args.actor_learning_rate = 1e-3
    args.critic_learning_rate = 1e-3
    args.gamma = 0.99
    args.tau = 0.005
    args.policy_noise = 0.2
    args.noise_clip = 0.5
    args.policy_frequency = 2
    args.seed = 0
    
    print(f"Using device: {args.device}")
    print(f"Training with {args.num_envs} parallel environments")

    # OGBench environments are automatically registered on import
    
    env = gym.make(
        args.env_name, 
        render_mode=None,  # No rendering for training
        max_episode_steps=500
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(env.observation_space)
    print(env.action_space)

    # OGBench PointMaze has simple Box observation space (position only)
    obs_dim = env.observation_space.shape[0]  # 2D position
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    
    print(f"Observation dimensions: {obs_dim}")
    print(f"Creating replay buffer with n_obs={obs_dim}")

    def process_obs(obs):
        """Convert observation to tensor"""
        return torch.tensor(obs, dtype=torch.float32, device=args.device)

    actor = Actor(obs_dim, act_dim, num_envs=args.num_envs, device=args.device, init_scale=1.0, hidden_dim=256).to(args.device)
    actor_target = Actor(obs_dim, act_dim, num_envs=args.num_envs, device=args.device, init_scale=1.0, hidden_dim=256).to(args.device)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(obs_dim, act_dim, num_atoms=1, v_min=-100, v_max=100, hidden_dim=256, device=args.device).to(args.device)
    critic_target = Critic(obs_dim, act_dim, num_atoms=1, v_min=-100, v_max=100, hidden_dim=256, device=args.device).to(args.device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    replay = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=obs_dim,
        n_act=act_dim,
        n_critic_obs=obs_dim,
        asymmetric_obs=False,
        device=args.device,
    )

    obs, _ = env.reset(seed=args.seed)
    obs = process_obs(obs)
    episode_reward = 0
    global_step = 0

    while global_step < args.total_timesteps:
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = actor(obs.unsqueeze(0)).cpu().numpy()[0]  # Add batch dim for actor

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_obs = process_obs(next_obs)

        transition = {
            "observations": obs,
            "actions": torch.tensor(action, dtype=torch.float32, device=args.device),  # Remove .unsqueeze(0)
            "next": {
                "observations": next_obs,
                "rewards": torch.tensor([reward], dtype=torch.float32, device=args.device),  # Single bracket
                "truncations": torch.tensor([truncated], dtype=torch.int, device=args.device),  # Single bracket
                "dones": torch.tensor([terminated], dtype=torch.int, device=args.device),  # Single bracket
            },
        }
        replay.extend(transition)

        obs = next_obs
        global_step += 1
        episode_reward += reward

        if done:
            print(f"Step: {global_step}, Episode Reward: {episode_reward:.2f}")
            obs, _ = env.reset()
            obs = process_obs(obs)
            episode_reward = 0

        if global_step >= args.learning_starts:
            for _ in range(1):  # single update per step
                batch = replay.sample(args.batch_size)
                obs_b = batch["observations"]
                act_b = batch["actions"]
                next_obs_b = batch["next"]["observations"]
                rew_b = batch["next"]["rewards"]
                done_b = batch["next"]["dones"]

                with torch.no_grad():
                    noise = (torch.randn_like(act_b) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                    next_action = (actor_target(next_obs_b) + noise).clamp(-act_limit, act_limit)
                    target_q1, target_q2 = critic_target(next_obs_b, next_action)
                    target_q = torch.min(target_q1, target_q2)
                    target = rew_b + args.gamma * (1 - done_b) * target_q

                q1, q2 = critic(obs_b, act_b)
                critic_loss = ((q1 - target).pow(2).mean() + (q2 - target).pow(2).mean())

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                if global_step % args.policy_frequency == 0:
                    # Get Q-values for actor loss
                    q1_dist, q2_dist = critic(obs_b, actor(obs_b))
                    q1_values = critic.get_value(F.softmax(q1_dist, dim=1))
                    actor_loss = -q1_values.mean()
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

if __name__ == "__main__":
    main()