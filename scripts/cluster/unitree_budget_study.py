#!/usr/bin/env python3
"""Bounded obstacle-only Unitree study; one allocated GPU, sequential processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(os.environ.get('UNITREE_STUDY_ROOT', ROOT.parent))
RESULTS = STUDY / 'results'
LOW = ROOT / 'external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-07-12_10-35-19_omni_finetune_model1499_20260712'
INIT = ROOT / 'study_inputs/goal_initialization.pt'


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def train_args(seed=73, steps=4500, **overrides):
    from train_unitree_nav_thesis import parse_args
    args = parse_args([])
    values = dict(
        task='Unitree-G1-Nav-Obstacles-Safe-Collision', method='thesis', device='cuda:0',
        seed=seed, num_envs=1, total_steps=steps, low_level_policy_path=str(LOW),
        init_checkpoint=str(INIT), reset_learner_state=True,
        debug_terrain_rows=5, debug_terrain_cols=10, debug_num_obstacles=6,
        debug_obstacle_width_min=1.0, debug_obstacle_width_max=1.4,
        debug_obstacle_height_min=1.0, debug_obstacle_height_max=1.0,
        debug_platform_width=2.0, strict_min_size_obstacles=True, disable_obstacles=False,
        height_scan_resolution=0.25, scan_history=1, action_history=0,
        hidden_dim=256, use_layer_norm=True, mask_goal_heading=True,
        navigation_policy_hz=5, navigation_transition_mode='decision', navigation_episode_mode='episodic',
        episode_length_s=60, terminate_on_goal=True, success_dist=0.4,
        goal_distance_min=2.8, goal_distance_max=4.0, goal_through_obstacle_prob=0.0,
        min_start_obstacle_clearance=1.0, min_goal_obstacle_clearance=0.9,
        resample_terrain_tiles=True, require_blocked_corridor=False,
        learner_reward_mode='dense_progress', dense_progress_scale=1.0,
        success_bonus=20.0, failure_penalty=-20.0,
        lr_actor=1e-4, lr_critic=3e-4, gamma=0.99, tau=0.005,
        n_step=5, updates_per_step=1, batch_size=256, policy_frequency=2,
        replay_capacity=500000, learning_starts=128, random_steps=0,
        teacher_warmup_steps=125, teacher_type='geom_scan', teacher_geom_planner='astar',
        teacher_geom_disengage_clear_steps=5,
        teacher_geom_clearance=0.6, teacher_goal_stop_dist=0.3,
        intervention_gate_mode='clearance_or_stall', intervention_clearance_mode='fixed',
        intervention_clearance_threshold=0.9, intervention_release_clearance=1.05,
        intervention_stall_steps=15, intervention_release_steps=1,
        intervention_release_action_delta_max=0.8, intervention_progress_epsilon=0.04,
        deterministic_student=False, student_controller='actor', actor_bc_weight=0,
        pref_sampling_mode='separate', pref_batch_size=256, pref_replay_capacity=500000,
        pref_rank_weight=1, pref_rank_margin=0.05, pref_action_delta_min=0.05,
        pref_stopgrad_positive=True, pref_loss_type='lagrangian',
        pref_lambda_init=1, pref_lambda_lr=0, pref_lambda_max=0, pref_lambda_ema=0.9,
        pref_violation_target=0, pref_dual_mode='hinge',
        alpha_init=0.01, alpha_min=0.01, alpha_max=0.01, alpha_state_scope='all',
        log_interval=100, checkpoint_interval=1500,
        eval_interval=1500, eval_num_envs=1, eval_num_episodes=32,
        eval_layout_manifest=str(RESULTS / 'manifests/validation.json'),
        eval_seed=941, eval_timeout_s=1800, eval_fail_fast=True, eval_at_end=True,
        train_layout_manifest=str(RESULTS / f'manifests/train_seed{seed}.json'),
        wandb_mode='disabled', output_dir=str(RESULTS / 'runs'))
    values.update(overrides)
    for key, value in values.items():
        if not hasattr(args, key):
            raise ValueError(f'Unknown training field {key}')
        setattr(args, key, value)
    return args


AUTO = dict(alpha_min=1e-5, alpha_max=0.1, alpha_lr=3e-4, target_entropy=-3)
ARMS = {
    'fixed1': {},
    'alpha_all': dict(**AUTO, alpha_state_scope='all'),
    'alpha_student': dict(**AUTO, alpha_state_scope='student'),
    'ungated_rl': dict(method='sac', teacher_warmup_steps=0, intervention_gate_mode='none', pref_rank_weight=0),
    'gated_rl': dict(pref_rank_weight=0),
    'linked_fixed1': dict(pref_sampling_mode='linked'),
    'bc02_fixed1': dict(actor_bc_weight=0.2),
    'margin001_fixed1': dict(pref_rank_margin=0.01),
    'lambda_eta01': dict(pref_lambda_lr=0.1),
    'lambda_eta1': dict(pref_lambda_lr=1.0),
    'lambda_tolerant': dict(pref_lambda_lr=0.1, pref_violation_target=0.02),
    'lambda_signed': dict(pref_lambda_lr=0.1, pref_dual_mode='signed'),
    'alpha_student_slow': dict(**{**AUTO, 'alpha_lr': 3e-5}, alpha_state_scope='student'),
    'smooth_fixed1': dict(pref_lagrangian_violation_type='smooth', pref_loss_temperature=0.1),
    'nstep1_fixed1': dict(n_step=1),
}
ANTI_ARMS = {
    'bc_rl': dict(pref_rank_weight=0, actor_pair_bc_weight=0.2),
    'anti_rl': dict(pref_rank_weight=0, actor_anti_weight=0.2),
    'bc_anti_rl': dict(pref_rank_weight=0, actor_pair_bc_weight=0.2, actor_anti_weight=0.2),
}


def eval_args(seed, controller, manifest=None, episodes=64, model=None, name='evaluation', hz=5):
    import eval_unitree_nav_baselines as evaluation
    old = sys.argv
    sys.argv = [old[0]]
    try:
        args = evaluation.parse_args()
    finally:
        sys.argv = old
    common = train_args(seed=seed)
    for key, value in vars(common).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.controller = controller
    args.num_episodes = episodes
    args.layout_manifest = str(manifest or '')
    args.model_path = str(model or '')
    args.checkpoint_env_config = bool(model)
    args.run_name = name
    args.output_dir = str(RESULTS / 'evaluations')
    args.record_trajectories = True
    args.navigation_policy_hz = hz
    return args


def execute_spec(path):
    spec = json.loads(Path(path).read_text())
    mode = spec['mode']
    if mode == 'train':
        from train_unitree_nav_thesis import run_training
        run_training(argparse.Namespace(**spec['args']))
    else:
        import eval_unitree_nav_baselines as evaluation
        args = argparse.Namespace(**spec['args'])
        if mode == 'manifest':
            evaluation.generate_layout_manifest(args)
        elif mode == 'evaluate':
            if args.model_path:
                from unitree_nav_checkpoint import apply_unitree_checkpoint_config
                apply_unitree_checkpoint_config(args)
                # Explicit diagnostic cadence must survive checkpoint restore.
                args.navigation_policy_hz = spec['args']['navigation_policy_hz']
            evaluation.evaluate(args)
        else:
            raise ValueError(mode)


def run_child(name, mode, args, timeout_s):
    specs = RESULTS / 'specs'
    logs = STUDY / 'logs'
    specs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    spec_path = specs / f'{name}.json'
    if spec_path.exists():
        raise RuntimeError(f'Immutable task already exists: {name}')
    atomic_json(spec_path, {'mode': mode, 'args': vars(args)})
    start = time.monotonic()
    with (logs / f'{name}.log').open('w') as log:
        proc = subprocess.Popen([sys.executable, __file__, 'execute', str(spec_path)],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise RuntimeError(f'{name} exceeded its declared runtime; artifacts preserved')
    if code:
        raise RuntimeError(f'{name} exited {code}; inspect {logs / (name + ".log")}')
    return time.monotonic() - start


def build_manifest(name, seed, episodes):
    path = RESULTS / 'manifests' / (name + '.json')
    args = eval_args(seed, 'geom_scan_teacher', episodes=episodes, name=name)
    args.generate_layout_manifest = str(path)
    run_child('manifest_' + name, 'manifest', args, 1200)
    return path


def baseline(name, controller, manifest, seed, episodes=64):
    args = eval_args(seed, controller, manifest, episodes=episodes, name=name)
    run_child(name, 'evaluate', args, 2400)
    return RESULTS / 'evaluations' / name / f'{controller}_metrics.json'


def status(**extra):
    state = {'study': 'unitree_budget25_20261009', 'job_id': os.environ.get('SLURM_JOB_ID'),
             'updated_unix': time.time(), **extra}
    atomic_json(RESULTS / 'status.json', state)


def pilot():
    status(state='pilot', active='GPU/runtime smoke and timing')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable on allocated compute node')
    atomic_json(RESULTS / 'device.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
                 'gpu': torch.cuda.get_device_name(), 'vram_bytes': torch.cuda.get_device_properties(0).total_memory})
    build_manifest('train_seed73', 73, 64)
    build_manifest('train_seed74', 74, 64)
    build_manifest('validation', 941, 64)
    build_manifest('heldout_test', 1941, 64)
    # Small, actual GPU run with manifest replay and full preference batches.
    args = train_args(73, 200, run_name='pilot_200decisions', teacher_warmup_steps=200,
                      learning_starts=32, checkpoint_interval=200,
                      eval_interval=0, eval_at_end=False)
    seconds = run_child('pilot_200decisions', 'train', args, 1500)
    rows = [json.loads(line) for line in (RESULTS / 'runs/pilot_200decisions/metrics.jsonl').read_text().splitlines()]
    if not rows or rows[-1].get('pref_source_rows') != 256 or rows[-1].get('pref_lambda') != 1:
        raise RuntimeError('Pilot did not exercise full preference replay and fixed lambda')
    if rows[-1]['transition_unit'] != 'navigation_decision' or rows[-1]['environment_ticks_collected'] > 800:
        raise RuntimeError('Pilot navigation decision/tick accounting mismatch')
    atomic_json(RESULTS / 'pilot_timing.json', {'wall_seconds_200decisions': seconds,
                'last_metrics': rows[-1], 'note': 'Includes startup; not linear full-run timing.'})
    status(state='pilot_complete', active=None, pilot_seconds=seconds)


def screen():
    if not (RESULTS / 'pilot_timing.json').exists():
        raise RuntimeError('Successful pilot required')
    # Scheduler limits plus the 30-minute pilot and five-hour anti job total 30 GPU-hours.
    deadline = time.monotonic() + 24 * 3600 + 10 * 60
    completed, skipped = [], []
    start = time.monotonic()
    for seed in [73, 74]:
        status(state='auditing', active=f'teacher_seed{seed}')
        audit_path = baseline(f'teacher_train_seed{seed}', 'geom_scan_teacher',
                              RESULTS / f'manifests/train_seed{seed}.json', seed, 64)
        audit = json.loads(audit_path.read_text())
        safe = sum(bool(ep['success']) and ep['cost_sum'] == 0 for ep in audit['episodes']) / 64
        if safe < 0.8:
            raise RuntimeError(f'Teacher safe-success audit failed: seed {seed}, {safe:.1%}; no student sweep started')
    baseline('teacher_validation', 'geom_scan_teacher', RESULTS / 'manifests/validation.json', 941)
    baseline('direct_goal_validation', 'direct_goal', RESULTS / 'manifests/validation.json', 941)
    schedule = [(arm, seed, 4500) for seed in [73, 74]
                for arm in ['fixed1', 'alpha_all', 'alpha_student']]
    schedule += [(arm, 73, 4500) for arm in ARMS if arm not in {'fixed1', 'alpha_all', 'alpha_student'}]
    schedule += [(arm, 74, 4500) for arm in ['lambda_eta01', 'lambda_tolerant', 'lambda_signed', 'linked_fixed1']]
    schedule += [(arm, 73, 9000) for arm in ['lambda_eta01', 'lambda_tolerant']]
    atomic_json(RESULTS / 'plan.json', {'budget_gpu_hours_max': 30, 'gpu_count': 1,
                'pilot_wall_minutes_max': 30, 'screen_wall_minutes_max': 1465,
                'anti_imitation_wall_minutes_max': 300, 'navigation_policy_hz': 5,
                'transition_unit': 'navigation_decision', 'control_hold_ticks': 4,
                'all_preference_batches': 256, 'td_capacity': 500000,
                'arms': ARMS, 'schedule': schedule,
                'note': 'Secondary/long runs only while conservative time estimate fits; no fixed-lambda-10 arm.'})
    durations = []
    reserve_s = 30 * 60
    for arm, seed, steps in schedule:
        name = f'{arm}_seed{seed}_{steps}'
        remaining = deadline - time.monotonic()
        # Initial conservative allowance; replace with observed full-run timing.
        estimate = max(durations[-3:] or [3600]) * (steps / 4500) * 1.3
        if remaining < estimate + reserve_s:
            skipped.append({'name': name, 'reason': 'budget/time reserve'})
            continue
        status(state='training', active=name, completed=completed, skipped=skipped,
               allocated_wall_seconds=time.monotonic() - start)
        args = train_args(seed, steps, **ARMS[arm], run_name=name)
        seconds = run_child(name, 'train', args, min(remaining - reserve_s, max(estimate * 2, 5400)))
        model = RESULTS / 'runs' / name / 'final.pt'
        final_eval = eval_args(941, 'policy', RESULTS / 'manifests/validation.json',
                              model=model, name=name + '_validation')
        seconds += run_child(name + '_validation', 'evaluate', final_eval, 2400)
        durations.append(seconds / (steps / 4500))
        completed.append({'name': name, 'arm': arm, 'seed': seed, 'steps': steps, 'seconds': seconds})
        # Update plots/report while the study is running, without extra GPU jobs.
        subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_unitree_budget_study.py'))], check=True)
    # Select on validation only; the disjoint held-out cohort is never used for selection.
    ranking = []
    for item in completed:
        path = RESULTS / 'evaluations' / (item['name'] + '_validation') / 'policy_metrics.json'
        data = json.loads(path.read_text())
        safe = sum(bool(ep['success']) and ep['cost_sum'] == 0 for ep in data['episodes']) / 64
        ranking.append((safe, item['name']))
    finalists = [name for _, name in sorted(ranking, reverse=True)[:3]]
    atomic_json(RESULTS / 'heldout_selection.json', {'validation_selected_finalists': finalists})
    if deadline - time.monotonic() > 1800:
        baseline('teacher_heldout', 'geom_scan_teacher', RESULTS / 'manifests/heldout_test.json', 1941)
        baseline('direct_goal_heldout', 'direct_goal', RESULTS / 'manifests/heldout_test.json', 1941)
        for name in finalists:
            if deadline - time.monotonic() < 900:
                break
            args = eval_args(1941, 'policy', RESULTS / 'manifests/heldout_test.json',
                             model=RESULTS / 'runs' / name / 'final.pt', name=name + '_heldout')
            run_child(name + '_heldout', 'evaluate', args, min(1800, deadline - time.monotonic()))
    status(state='screen_complete', active=None, completed=completed, skipped=skipped,
           allocated_wall_seconds=time.monotonic() - start,
           warning='Screening evidence; two seeds only for prioritized comparisons.')
    subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_unitree_budget_study.py'))], check=True)


def anti_screen():
    deadline = time.monotonic() + 4 * 3600 + 45 * 60
    completed = []
    for seed in [73, 74]:
        for arm, overrides in ANTI_ARMS.items():
            name = f'{arm}_seed{seed}_4500'
            if deadline - time.monotonic() < 1800:
                status(state='anti_budget_stop', completed=completed)
                return
            status(state='anti_training', active=name, completed=completed)
            args = train_args(seed, 4500, run_name=name, **overrides)
            run_child(name, 'train', args, deadline - time.monotonic() - 900)
            model = RESULTS / 'runs' / name / 'final.pt'
            evaluation = eval_args(941, 'policy', RESULTS / 'manifests/validation.json',
                                   model=model, name=name + '_validation')
            run_child(name + '_validation', 'evaluate', evaluation, min(1800, deadline - time.monotonic()))
            completed.append({'name': name, 'arm': arm, 'seed': seed})
            subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_unitree_budget_study.py'))], check=True)
    status(state='complete', active=None, anti_completed=completed)
    subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_unitree_budget_study.py'))], check=True)


def main():
    if 'SLURM_JOB_ID' not in os.environ and os.environ.get('UNITREE_LOCAL_SMOKE') != '1':
        raise RuntimeError('GPU worker requires a Slurm allocation')
    if sys.argv[1] == 'execute':
        execute_spec(sys.argv[2])
        return
    try:
        {'pilot': pilot, 'screen': screen, 'anti': anti_screen}[sys.argv[1]]()
    except Exception as error:
        status(state='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
