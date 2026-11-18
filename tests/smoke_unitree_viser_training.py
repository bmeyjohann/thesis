"""GPU integration smoke: synthetic controller input through the real browser loop."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_interactive_unitree_nav as interactive


def main():
    name = f"browser_intervention_smoke_{time.time_ns()}"
    dataset = ROOT / 'local/unitree_human_interventions' / name
    low = ROOT / 'external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-09-03_01-25-36_g1_gentle_turn_curriculum_seed0_resume13000_20260903/model_15999.pt'
    sys.argv = ['smoke', '--viewer-backend', 'viser', '--viser-port', '8091',
                '--controller', 'policy', '--online-train', '--autopilot',
                '--online-run-name', name, '--online-total-steps', '12',
                '--online-learning-starts', '2', '--online-n-step', '1',
                '--online-batch-size', '2', '--online-checkpoint-interval', '6',
                '--online-log-interval', '1', '--online-wandb-mode', 'disabled',
                '--human-dataset-dir', str(dataset), '--num-envs', '1',
                '--human-input-device', 'gamepad', '--low-level-policy-path', str(low),
                '--target-terrain', '--target-terrain-seed', '7',
                '--target-terrain-preset', 'challenge', '--target-terrain-dynamic-obstacles',
                '--target-terrain-material-resolution', '0.2',
                '--target-terrain-ice-friction', '0.06', '--target-terrain-ice-contact-priority', '1',
                '--goal-distance-min', '4', '--goal-distance-max', '10',
                '--height-scan-resolution', '0.1', '--height-scan-forward-size', '1.6',
                '--height-scan-lateral-size', '1.0', '--use-layer-norm',
                '--no-start-paused']
    args = interactive.parse_args()
    calls = 0
    physics_ticks = 0
    from unitree_viser_training import BrowserTrainingSession
    original_step = BrowserTrainingSession.step

    def counted_step(session, *values):
        nonlocal physics_ticks
        physics_ticks += 1
        return original_step(session, *values)

    def sample(_):
        nonlocal calls
        calls += 1
        active = calls % 7 < 3
        return SimpleNamespace(action=np.array([0.2, 0.1, 0], dtype=np.float32),
            intervening=active, connected=True, stale=False, command_norm=0.224,
            state_age_s=0.0, state={'buttons': {}}, transport={'connected': True})

    with patch.object(interactive.UnitreeGamepadController, 'sample', sample), \
         patch.object(BrowserTrainingSession, 'step', counted_step):
        interactive.run(args)
    run_dir = Path(args.online_output_dir) / args.online_run_name
    metrics = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['step'] == 12
    assert metrics[-1]['updates'] > 0
    assert (run_dir / 'final.pt').is_file()
    summary = json.loads((run_dir / 'browser_summary.json').read_text())
    assert summary['limit_reached'] and summary['navigation_decisions'] == 12
    assert summary['environment_ticks'] == 48
    manifest = json.loads((dataset / 'manifest.json').read_text())
    assert manifest['summary']['rows'] == 12
    assert manifest['metadata']['transition_hz'] == 5
    assert physics_ticks == 48, physics_ticks
    assert 0 < manifest['summary']['intervention_rows'] < 12
    arrays = [np.load(path) for path in sorted((dataset / 'parts').glob('*.npz'))]
    actions = np.concatenate([part['executed_actions'] for part in arrays])
    human = np.concatenate([part['intervened'] for part in arrays])
    assert np.allclose(actions[human], [0.2, 0.1, 0])
    print(json.dumps({'status': 'PASS', 'run': str(run_dir), 'dataset': str(dataset),
                      'summary': manifest['summary'], 'updates': metrics[-1]['updates'],
                      'environment_steps': physics_ticks}))


if __name__ == '__main__':
    main()
