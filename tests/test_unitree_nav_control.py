import pytest
import torch

from unitree_nav_control import navigation_action_repeat, project_navigation_action


def test_default_five_hz_repeats_four_twenty_hz_steps():
    assert navigation_action_repeat(5.0) == 4


def test_navigation_action_projection_preserves_direction_and_caps_norm():
    action = torch.tensor([[1.0, 1.0, 1.0], [0.2, -0.1, 0.3]])
    projected = project_navigation_action(action)
    assert torch.linalg.vector_norm(projected[0]).item() == pytest.approx(1.0)
    assert torch.equal(projected[1], action[1])
    assert torch.allclose(projected[0] / projected[0, 0], action[0] / action[0, 0])


def test_navigation_rate_must_divide_base_rate():
    with pytest.raises(ValueError):
        navigation_action_repeat(6.0)
