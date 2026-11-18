from types import SimpleNamespace
import pytest
from scripts.cluster.unitree_budget_bundle import run_bundle


def fake_study(tmp_path, fail=False):
    events, timeouts = [], []
    now = [0.0]

    def child(name, mode, args, timeout_s):
        timeouts.append(timeout_s)

    study = SimpleNamespace(run_child=child, RESULTS=tmp_path,
                            atomic_json=lambda *args: None, status=lambda **kw: events.append(kw))

    def phase(name):
        events.append(name)
        now[0] += 10
        study.run_child(name, 'train', None, 999999)
        if fail and name == 'pilot':
            raise RuntimeError('pilot failure')

    study.pilot = lambda: phase('pilot')
    study.screen = lambda: phase('screen')
    study.anti_screen = lambda: phase('anti')
    return study, events, timeouts, now, child


def test_all_phases_run_without_another_submission(tmp_path):
    study, events, timeouts, now, original = fake_study(tmp_path)
    run_bundle(study, clock=lambda: now[0])
    assert events == ['pilot', 'screen', 'anti']
    assert timeouts == [1790, 87890, 17990]
    assert study.run_child is original


def test_failed_pilot_never_runs_students(tmp_path):
    study, events, _, now, original = fake_study(tmp_path, fail=True)
    with pytest.raises(RuntimeError, match='pilot failure'):
        run_bundle(study, clock=lambda: now[0])
    assert 'screen' not in events and 'anti' not in events
    assert events[-1]['state'] == 'failed'
    assert study.run_child is original


def test_expired_phase_never_starts_another_child(tmp_path):
    study, events, timeouts, now, _ = fake_study(tmp_path)

    def expired():
        now[0] = 1801
        study.run_child('late', 'train', None, 10)

    study.pilot = expired
    with pytest.raises(RuntimeError, match='Phase time exhausted'):
        run_bundle(study, clock=lambda: now[0])
    assert not timeouts and 'screen' not in events
