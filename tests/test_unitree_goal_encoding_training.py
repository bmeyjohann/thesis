import argparse
import torch
from train_unitree_nav_thesis import _goal_success


def test_normalized_goal_success_uses_metric_distance():
    args = argparse.Namespace(goal_encoding="distance_bearing", goal_distance_scale=14.0)
    obs = torch.zeros(2, 9)
    obs[:, 6] = torch.tensor([0.02, 0.10])
    assert _goal_success(obs, 0.4, args).tolist() == [True, False]