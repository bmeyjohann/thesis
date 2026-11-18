from types import SimpleNamespace
import pytest
import torch
from unitree_nav_control import SingleEnvNavigationDecisionAdapter


class FakeEnv:
    def __init__(self, terminal_at=99, num_envs=1):
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=num_envs))
        self.count, self.terminal_at = 0, terminal_at
        self.actions = []

    def step(self, action):
        self.count += 1
        self.actions.append(action.clone())
        return torch.tensor([[self.count]]), torch.tensor([2.]), torch.tensor([self.count == self.terminal_at]), {
            'costs': torch.tensor([[1., 2.]]), 'time_outs': torch.tensor([False])}


def test_exact_5hz_hold_and_reward_cost_aggregation():
    env = FakeEnv()
    wrapper = SingleEnvNavigationDecisionAdapter(env, policy_hz=5)
    action = torch.tensor([[0.2, 0., 0.]])
    obs, reward, done, extra = wrapper.step(action)
    assert env.count == 4 and reward.item() == 8 and obs.item() == 4
    assert extra['costs'].tolist() == [[4, 8]] and extra['navigation_ticks'] == 4
    assert not done.item() and all(torch.equal(x, action) for x in env.actions)


def test_terminal_stops_before_auto_reset_transitions_can_enter_replay():
    env = FakeEnv(terminal_at=2)
    wrapper = SingleEnvNavigationDecisionAdapter(env, policy_hz=5)
    _, reward, done, extra = wrapper.step(torch.zeros(1, 3))
    assert env.count == 2 and reward.item() == 4 and done.item()
    assert extra['navigation_ticks'] == 2


def test_vector_aggregation_rejected_instead_of_crossing_resets():
    with pytest.raises(ValueError, match='num_envs=1'):
        SingleEnvNavigationDecisionAdapter(FakeEnv(num_envs=2), policy_hz=5)
