"""Bounded GPU integration check for the real study inspection adapter."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import torch


def main():
    repo = Path('/home/benjamin/thesis')
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location('study_inspection',
        repo / 'scripts/inspect_unitree_study_policy.py')
    inspection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspection)
    import eval_interactive_unitree_nav as interactive
    from unitree_viser_training import BrowserTrainingSession

    sessions = []
    actions = []
    original_init = BrowserTrainingSession.__init__
    original_step = BrowserTrainingSession.step

    def initialize(session, *args, **kwargs):
        original_init(session, *args, **kwargs)
        sessions.append(session)
        assert session.learner is None and session.writer is None
        assert session.repeat == 4

    def step(session, executed, proposed, human, source, sample):
        assert not human and source == 0
        assert not sample.connected and not sample.intervening
        assert torch.allclose(executed, proposed)
        assert torch.linalg.vector_norm(executed) <= 1.00001
        actions.append(executed.detach().cpu().clone())
        return original_step(session, executed, proposed, human, source, sample)

    sys.argv.extend(['--no-start-paused', '--viser-max-steps', '12'])
    with patch.object(interactive, 'UnitreeGamepadController',
                      side_effect=AssertionError('Policy-only viewer opened a gamepad')), \
         patch.object(BrowserTrainingSession, '__init__', initialize), \
         patch.object(BrowserTrainingSession, 'step', step):
        result = inspection.main()
    assert result == 0 and len(sessions) == 1
    session = sessions[0]
    assert session.environment_ticks == 12
    assert session.steps == 3 and session.interventions == 0
    assert any(torch.linalg.vector_norm(action) > 0.01 for action in actions)
    for index in range(0, len(actions), 4):
        assert all(torch.allclose(actions[index], action) for action in actions[index:index + 4])
    print('PASS: Viser started, 12 ticks / 3 decisions, nonzero held policy actions, '
          'no gamepad construction, takeovers, training, or dataset writes.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
