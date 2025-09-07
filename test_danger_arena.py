#!/usr/bin/env python3
"""
Quick interactive test for the Arena-Danger maze variants.

Usage examples:
  python scripts/test_danger_arena.py --mode floor
  python scripts/test_danger_arena.py --mode wall
  python scripts/test_danger_arena.py --mode sticky --episodes 3
  python scripts/test_danger_arena.py --mode lethal

This runs the environment with a human-control wrapper and a dedicated
keyboard control window to verify the dangerous tile marker and behavior.
"""

import argparse
import time
import gymnasium as gym
import ogbench  # ensure envs are registered
from ogbench.wrappers import DirectTeleopWrapper
from ogbench.teleop import ControlWindowTeleop


ENV_BY_MODE = {
    'floor': 'pointmaze-arena-danger-floor-v0',
    'wall': 'pointmaze-arena-danger-wall-v0',
    'sticky': 'pointmaze-arena-danger-sticky-v0',
    'lethal': 'pointmaze-arena-danger-lethal-v0',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='floor', choices=list(ENV_BY_MODE.keys()))
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    env_id = ENV_BY_MODE[args.mode]
    print(f"Creating env: {env_id}")
    env = gym.make(env_id, render_mode='human')

    # Create a dedicated control window for reliable keyboard focus
    teleop = ControlWindowTeleop(width=520, height=420, show_debug_info=True)
    env = DirectTeleopWrapper(env, teleop)

    print("Controls: Focus the 'Teleoperation Control Panel' window.")
    print("Arrow keys to move; ESC or closing the control window exits.")
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            ep_rew = 0.0
            teleop.update_state(obs, info, step_count=0)

            for t in range(args.max_steps):
                # DirectTeleopWrapper ignores the action argument and uses human input
                obs, rew, term, trunc, info = env.step(None)
                ep_rew += float(rew)
                teleop.update_state(obs, info, step_count=t + 1)
                # Simple frame pacing
                time.sleep(max(0.0, 1.0 / args.fps))

                # Allow exiting via the control window
                if hasattr(teleop, 'should_quit') and teleop.should_quit():
                    print("Quit requested from control window. Exiting...")
                    return
                if term or trunc:
                    print(f"Episode {ep+1} done at step {t+1}. reward={ep_rew:.2f}, info={info}")
                    break
        print("Done.")
    finally:
        # Close both env and teleop window
        try:
            teleop.close()
        except Exception:
            pass
        env.close()


if __name__ == '__main__':
    main()
