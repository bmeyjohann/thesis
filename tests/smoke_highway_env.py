#!/usr/bin/env python3
"""
Quick smoke test for HighwayEnv with continuous actions.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

HIGHWAY_PATH = Path(__file__).resolve().parents[1] / "HighwayEnv"
if HIGHWAY_PATH.exists():
    os.sys.path.append(str(HIGHWAY_PATH))

import highway_env  # noqa: F401  # Registers gymnasium environments.  # noqa: E402

try:  # noqa: E402
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required for HighwayEnv smoke test") from exc


def main() -> None:
    env = gym.make("highway-v0")
    env.unwrapped.configure(
        {
            "observation": {"type": "Kinematics", "vehicles_count": 10},
            "action": {"type": "ContinuousAction"},
        }
    )
    env = gym.wrappers.FlattenObservation(env)
    obs, info = env.reset(seed=0)
    print(f"obs shape={obs.shape} action_space={env.action_space}")
    total_reward = 0.0
    for step in range(25):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            print(f"terminated at step {step}")
            break
    print(f"total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
