from __future__ import annotations

import torch


UNITREE_HIGH_LEVEL_BASE_HZ = 20.0
DEFAULT_NAVIGATION_POLICY_HZ = 5.0


def navigation_action_repeat(policy_hz: float) -> int:
    """Return the integer 20 Hz control-step hold for one policy decision."""
    if policy_hz <= 0.0 or policy_hz > UNITREE_HIGH_LEVEL_BASE_HZ:
        raise ValueError(f"navigation_policy_hz must be in (0, {UNITREE_HIGH_LEVEL_BASE_HZ}], got {policy_hz}")
    repeat = int(round(UNITREE_HIGH_LEVEL_BASE_HZ / policy_hz))
    resolved_hz = UNITREE_HIGH_LEVEL_BASE_HZ / repeat
    if abs(resolved_hz - policy_hz) > 1e-6:
        raise ValueError(
            f"navigation_policy_hz={policy_hz} must divide {UNITREE_HIGH_LEVEL_BASE_HZ} Hz exactly; "
            f"nearest supported value is {resolved_hz:g} Hz"
        )
    return repeat


def project_navigation_action(action: torch.Tensor, max_norm: float = 1.0) -> torch.Tensor:
    """Project normalized [forward, lateral, yaw] commands onto a unit ball."""
    if max_norm <= 0.0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")
    norm = torch.linalg.vector_norm(action, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / norm.clamp_min(1e-8), max=1.0)
    return action * scale


class SingleEnvNavigationDecisionAdapter:
    """One transition per held command, stopping before an auto-reset boundary."""

    def __init__(self, env, *, policy_hz: float):
        if int(env.env.unwrapped.num_envs) != 1:
            raise ValueError("Decision aggregation requires num_envs=1; asynchronous resets need a separate vector design")
        self.base_env = env
        self.repeat = navigation_action_repeat(policy_hz)

    def __getattr__(self, name):
        return getattr(self.base_env, name)

    def step(self, action):
        reward_sum = None
        cost_sums = {}
        for tick in range(self.repeat):
            observation, reward, done, extras = self.base_env.step(action)
            reward_sum = reward.clone() if reward_sum is None else reward_sum + reward
            for key in ('cost', 'costs'):
                if key in extras and torch.is_tensor(extras[key]):
                    value = extras[key]
                    cost_sums[key] = value.clone() if key not in cost_sums else cost_sums[key] + value
            if bool(done.any()):
                break
        extras = dict(extras)
        extras.update(cost_sums)
        extras['navigation_ticks'] = tick + 1
        return observation, reward_sum, done, extras
