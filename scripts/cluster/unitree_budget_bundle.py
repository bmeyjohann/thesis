#!/usr/bin/env python3
"""Run the existing study phases back-to-back in one Slurm allocation."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time


def run_bundle(study, *, clock=time.monotonic):
    original_child = study.run_child
    phase_deadline = 0.0

    def bounded_child(name, mode, args, timeout_s):
        remaining = phase_deadline - clock()
        if remaining <= 0:
            raise RuntimeError(f'Phase time exhausted before {name}; artifacts preserved')
        return original_child(name, mode, args, min(timeout_s, remaining))

    study.run_child = bounded_child
    try:
        for name, maximum_seconds in [('pilot', 1800), ('screen', 87900), ('anti', 18000)]:
            phase_deadline = clock() + maximum_seconds
            study.atomic_json(study.RESULTS / 'allocation_phase.json', {
                'job_id': os.environ.get('SLURM_JOB_ID'), 'phase': name,
                'phase_wall_seconds_max': maximum_seconds, 'updated_unix': time.time()})
            {'pilot': study.pilot, 'screen': study.screen, 'anti': study.anti_screen}[name]()
    except Exception as error:
        study.status(state='failed', error=str(error), allocation_mode='single_bundle')
        raise
    finally:
        study.run_child = original_child


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Bundle requires a Slurm allocation')
    root = Path(os.environ['UNITREE_STUDY_ROOT'])
    sys.path.insert(0, str(root / 'repo/scripts/cluster'))
    import unitree_budget_study as study
    run_bundle(study)


if __name__ == '__main__':
    main()
