import torch

from unitree_nav_observation import UnitreeScanHistory, prepare_unitree_actor_obs, unitree_goal_distance, unitree_goal_xy


def _obs(scan_value: float, batch: int = 1) -> torch.Tensor:
    context = torch.zeros(batch, 9)
    scan = torch.full((batch, 4), scan_value)
    return torch.cat((context, scan), dim=1)


def test_sparse_scan_history_uses_requested_temporal_stride():
    history = UnitreeScanHistory(history=3, history_stride=2)
    stacked = history.reset(_obs(0.0))
    assert stacked[0, 9:].reshape(3, 4)[:, 0].tolist() == [0.0, 0.0, 0.0]

    stacked = history.step(_obs(1.0))
    assert stacked[0, 9:].reshape(3, 4)[:, 0].tolist() == [1.0, 0.0, 0.0]

    stacked = history.step(_obs(2.0))
    assert stacked[0, 9:].reshape(3, 4)[:, 0].tolist() == [2.0, 0.0, 0.0]

    history.step(_obs(3.0))
    stacked = history.step(_obs(4.0))
    assert stacked[0, 9:].reshape(3, 4)[:, 0].tolist() == [4.0, 2.0, 0.0]


def test_done_resets_sparse_scans_and_dense_action_history():
    history = UnitreeScanHistory(history=3, action_history=2, action_dim=2, history_stride=2)
    history.reset(_obs(0.0, batch=2))
    history.step(_obs(1.0, batch=2), action=torch.ones(2, 2))
    stacked = history.step(
        _obs(2.0, batch=2),
        done=torch.tensor([True, False]),
        action=torch.full((2, 2), 2.0),
    )

    scan_values = stacked[:, 9:21].reshape(2, 3, 4)[:, :, 0]
    assert scan_values[0].tolist() == [2.0, 2.0, 2.0]
    assert scan_values[1].tolist() == [2.0, 0.0, 0.0]
    assert stacked[0, 21:].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert stacked[1, 21:].tolist() == [2.0, 2.0, 1.0, 1.0]


def test_distance_bearing_goal_encoding_is_normalized_and_reversible():
    obs = torch.zeros(1, 13)
    obs[0, :3] = torch.tensor([2.0, -0.5, 0.0])
    obs[0, 6:9] = torch.tensor([3.0, 4.0, 2.5])
    encoded = prepare_unitree_actor_obs(
        obs, goal_encoding="distance_bearing", goal_distance_scale=10.0, velocity_scale=2.0
    )
    assert torch.allclose(encoded[0, :3], torch.tensor([1.0, -0.25, 0.0]))
    assert torch.allclose(encoded[0, 6:9], torch.tensor([0.5, 0.6, 0.8]))
    assert torch.allclose(
        unitree_goal_distance(encoded, goal_encoding="distance_bearing", goal_distance_scale=10.0),
        torch.tensor([5.0]),
    )
    assert torch.allclose(
        unitree_goal_xy(encoded, goal_encoding="distance_bearing", goal_distance_scale=10.0),
        torch.tensor([[3.0, 4.0]]),
    )