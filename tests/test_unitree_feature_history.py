"""Exact-lag fusion, reset isolation and checkpoint compatibility checks."""
import argparse
import torch
import pytest

from unitree_nav_observation import UnitreeScanHistory
from unitree_viser_training import BrowserTrainingSession
from unitree_nav_checkpoint import apply_unitree_checkpoint_config


def obs(step):
    return torch.full((2, 13), float(step))


def test_exact_lags_and_independent_reset():
    h = UnitreeScanHistory(3, scan_lags=[0, 1, 25])
    h.reset(obs(0))
    retained = None
    for step in range(1, 66):
        value = h.step(obs(step), done=torch.tensor([step == 30, False]))
        for env, start in [(0, 30 if step >= 30 else 0), (1, 0)]:
            expected = [max(start, step - lag) for lag in [0, 1, 25]]
            assert torch.equal(value[env, 9:].reshape(3, 4)[:, 0], torch.tensor(expected).float())
            assert (value[env, :9] == step).all()
        if step == 29:
            retained = value.clone()
    assert (retained[0, 9:].reshape(3, 4)[:, 0] == torch.tensor([29, 28, 4])).all()
    h.reset(obs(99), env_ids=torch.tensor([0]))
    value = h.step(obs(100))
    assert (value[0, 9:].reshape(3, 4)[:, 0] == torch.tensor([100, 99, 99])).all()


@pytest.mark.parametrize('lags', [[1, 2, 25], [0, 1, 1], [0, -1, 25], [0, 1.0, 25]])
def test_invalid_lags(lags):
    with pytest.raises(ValueError):
        UnitreeScanHistory(3, scan_lags=lags)


@pytest.mark.parametrize('feature_dim', [187 + 64, 32])
def test_fused_sensor_history_uses_current_context(feature_dim):
    session = BrowserTrainingSession.__new__(BrowserTrainingSession)
    session.timing_seconds = dict(physics=0.0, perception=0.0, learning=0.0)
    session.vision = None
    session.feature_history = UnitreeScanHistory(3, scan_lags=[0, 1, 25])
    frame = torch.zeros(1, 9 + feature_dim)
    session._policy_input(frame, reset=True)
    for step in range(1, 30):
        value = session._policy_input(frame + step)
    assert value.shape == (1, 9 + feature_dim * 3)
    assert (value[:, :9] == 29).all()
    assert (value[:, 9:].reshape(3, feature_dim)[:, 0] == torch.tensor([29, 28, 4])).all()
    reset = session._policy_input(frame + 90, reset=True)
    assert (reset == 90).all()


def test_checkpoint_lags_restore_and_legacy_clear(tmp_path):
    path = tmp_path / 'policy.pt'
    args = argparse.Namespace(model_path=str(path), controller='policy', navigation_feature_lags=[0, 1, 25])
    torch.save({'args': {'navigation_feature_lags': [0, 1, 25]}}, path)
    apply_unitree_checkpoint_config(args)
    assert args.navigation_feature_lags == [0, 1, 25]
    torch.save({'args': {}}, path)
    apply_unitree_checkpoint_config(args)
    assert args.navigation_feature_lags is None


def test_receipt_waits_for_raw_dataset_close(tmp_path):
    from types import SimpleNamespace
    session = BrowserTrainingSession.__new__(BrowserTrainingSession)
    session.timing_seconds = dict(physics=0.0, perception=0.0, learning=0.0)
    session.playback_events = []
    session.args = SimpleNamespace(online_train=True, online_total_steps=32, navigation_policy_hz=5,
        online_session_receipt=str(tmp_path / 'receipt.json'), online_objective_mode='pref_rl',
        navigation_observation_mode='php_depth', seed=7, online_run_name='test')
    session.steps, session.environment_ticks = 32, 128
    session.successes = session.interventions = session.cost_total = 0
    session.learner = SimpleNamespace(run_dir=tmp_path, close=lambda: (tmp_path / 'final.pt').touch())
    def failed_close():
        raise IOError('dataset write failed')
    session.writer = SimpleNamespace(close=failed_close)
    with pytest.raises(IOError):
        session.close()
    assert not (tmp_path / 'receipt.json').exists()
    session.writer = SimpleNamespace(close=lambda: None)
    session.close()
    assert (tmp_path / 'receipt.json').is_file()
