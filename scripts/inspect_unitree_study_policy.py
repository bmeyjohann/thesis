"""Inspect saved study policies without modifying their checkpoint metadata."""

import hashlib
import json
from pathlib import Path
import sys


REPO = Path('/home/benjamin/thesis')
STUDY = REPO / 'artifacts/unitree_budget25_20261009'
CONTROLLER = Path('external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/'
                  '2026-07-12_10-35-19_omni_finetune_model1499_20260712')


def main():
    check_only = '--check-only' in sys.argv
    if check_only:
        sys.argv.remove('--check-only')
    sys.path.insert(0, str(REPO))
    import eval_interactive_unitree_nav as interactive
    from unitree_nav_checkpoint import apply_unitree_checkpoint_config

    args = interactive.parse_args()
    apply_unitree_checkpoint_config(args)
    if args.controller != 'policy' or args.navigation_policy_hz != 5:
        raise ValueError('This study inspection requires policy-only control at 5 Hz.')
    if args.force_obstacles or args.runtime_navigation_episode_mode is not None:
        raise ValueError('Use the saved study environment without runtime profile overrides.')

    controller = Path(args.low_level_policy_path)
    if not controller.is_dir():
        desktop_repo = Path('/home/benjamin/unitree_priority_20261010_r1/repo')
        if controller != desktop_repo / CONTROLLER:
            raise FileNotFoundError(f'Unknown missing controller path: {controller}')
        controller = STUDY / 'local_core_20261010_r7/repo' / CONTROLLER
        print(f'[inspection] Remapped desktop controller to {controller}', flush=True)

    manifest = json.loads((STUDY / 'source_manifest_v1.json').read_text())
    for name in ('model_1528.pt', 'params/agent.yaml', 'params/env.yaml'):
        file = controller / name
        expected = manifest[str(CONTROLLER / name)]['sha256']
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Locomotion controller hash mismatch: {file}')
    args.low_level_policy_path = str(controller)
    print('[inspection] Checkpoint configuration restored; locomotion hashes verified.', flush=True)
    if check_only:
        return 0
    return interactive.run(args)


if __name__ == '__main__':
    raise SystemExit(main())
