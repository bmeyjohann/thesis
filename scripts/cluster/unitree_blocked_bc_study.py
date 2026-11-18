#!/usr/bin/env python3
"""Bounded two-seed BC comparison with fail-closed, physically blocked cohorts."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT.parent
RESULTS = STUDY / 'results'
MIRROR = Path('/mnt/c/home/benjamin/thesis/artifacts') / STUDY.name
sys.path.insert(0, str(ROOT))
os.environ['UNITREE_STUDY_ROOT'] = str(STUDY)
os.environ['UNITREE_LOCAL_SMOKE'] = '1'
os.environ['WANDB_DISABLE_CODE'] = 'true'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
os.environ.setdefault('WARP_CACHE_PATH', '/tmp/warp-cache')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/unitree-cache')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def status(**fields):
    path = STUDY / 'monitor_status.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(updated_unix=time.time(), **fields)
    write_json(path, data)
    sync_compact()


def sync_compact():
    MIRROR.mkdir(parents=True, exist_ok=True)
    for name in ('monitor_status.json', 'plan.json', 'inputs.sha256.json'):
        p = STUDY / name
        if p.exists():
            shutil.copy2(p, MIRROR / name)
    for name in ('analysis', 'results/manifests'):
        p = STUDY / name
        if p.exists():
            shutil.copytree(p, MIRROR / name, dirs_exist_ok=True)


def study_module():
    spec = importlib.util.spec_from_file_location('frozen_budget', ROOT / 'scripts/cluster/unitree_budget_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hard_settings(args):
    args.debug_terrain_rows, args.debug_terrain_cols = 10, 20
    args.goal_distance_min, args.goal_distance_max = 4.0, 7.0
    args.require_blocked_corridor = True
    args.goal_through_obstacle_prob = 1.0
    return args


def eval_args(seed, controller, manifest, name, model=None, episodes=64):
    args = study_module().eval_args(seed, controller, manifest, episodes=episodes, model=model, name=name)
    return hard_settings(args)


def make_manifest(name, seed):
    import numpy as np
    import torch
    import eval_unitree_nav_baselines as evaluation
    from unitree_nav_eval_manifest import capture_layout_entries, save_layout_manifest
    from unitree_nav_layout import set_terrain_tile_indices
    from unitree_hard_cohort import CRITERIA, Geometry

    path = RESULTS / f'manifests/{name}.json'
    if path.exists():
        raise RuntimeError(f'Immutable cohort already exists: {path}')
    args = eval_args(seed, 'geom_scan_teacher', '', name)
    args.require_blocked_corridor = False
    args.goal_through_obstacle_prob = 0.0
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 81000)
    env = evaluation.make_env(args, num_envs=1, render=False)
    entries, banks = [], []
    try:
        env.reset()
        template = capture_layout_entries(env)[0]
        unwrapped = env.env.unwrapped
        sizes = evaluation._to_numpy(unwrapped.sim.model.hfield_size)
        ncols = evaluation._to_numpy(unwrapped.sim.model.hfield_ncol)
        candidates = {}
        for tile in rng.permutation(len(sizes)):
            tile = int(tile)
            set_terrain_tile_indices(env, torch.tensor([0], device=args.device), torch.tensor([tile], device=args.device))
            origin = evaluation._to_numpy(unwrapped.scene.terrain.env_origins[0])[:2]
            cells = evaluation._terrain_obstacle_cells_by_env(env)[0] - origin
            geometry = Geometry(cells, sizes[tile, :2], 2*sizes[tile, 0]/(int(ncols[tile])-1))
            banks.append(dict(tile_index=tile, cells=cells.tolist(), half_size=sizes[tile, :2].tolist(),
                              pitch=geometry.pitch, sha256=geometry.sha256))
            free = np.argwhere(geometry.distances >= 1.05)
            found = []
            for _ in range(12000):
                pair = free[rng.integers(len(free), size=2)]
                start, goal = np.array([[geometry.xs[i], geometry.ys[j]] for i, j in pair])
                # Avoid exact tile-margin equality becoming invalid in float32 replay.
                if np.any(abs(start) >= geometry.half_size-.801) or np.any(abs(goal) >= geometry.half_size-.801):
                    continue
                audit = geometry.audit(start, goal, require=False)
                if not audit['accepted']:
                    continue
                entry = copy.deepcopy(template)
                entry['tile_index'] = tile
                entry['root_pose_relative'][:2] = start.tolist()
                yaw = float(rng.uniform(-math.pi, math.pi))
                entry['root_pose_relative'][3:] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
                entry['root_velocity'] = [0.0]*6
                entry['joint_velocity'] = [0.0]*len(entry['joint_velocity'])
                entry['goal_xy_relative'] = goal.tolist()
                entry['goal_heading'] = float(rng.uniform(-math.pi, math.pi))
                entry['hard_audit'] = audit
                found.append(entry)
                if len(found) == 3:
                    break
            if found:
                candidates[tile] = found
            print(f'[hard-bank] {name} tile={tile} accepted={len(found)}', flush=True)
        # Balance tiles first; never silently replace difficult cases with easy ones.
        for k in range(3):
            for cases in candidates.values():
                if k < len(cases) and len(entries) < 64:
                    entries.append(cases[k])
        if len(entries) != 64 or len(candidates) < 24:
            write_json(STUDY / f'preparation/{name}_rejected_geometry.json', banks)
            raise RuntimeError(f'Cannot build diverse hard cohort: {len(entries)} cases/{len(candidates)} tiles')
        write_json(STUDY / f'preparation/{name}_candidates.json', entries)
        write_json(STUDY / f'preparation/{name}_geometry.json', banks)
        args.require_blocked_corridor = True
        for i, entry in enumerate(entries):
            evaluation._apply_manifest_layouts(args, env, [0], [entry])
            print(f'[hard-replay] {name} case={i} verified', flush=True)
        save_layout_manifest(path, entries=entries, metadata=dict(seed=seed, task=args.task,
            num_episodes=64, generator_num_envs=1, criteria=CRITERIA,
            require_blocked_corridor=True, unique_tile_ids=len({e['tile_index'] for e in entries}),
            benchmark='physical_two_blockers_clearance_detour_v1',
            route_note='Clearance-grid shortest path is a discretized diagnostic, not a safety certificate.'))
        write_json(RESULTS / f'manifests/{name}_geometry.json', banks)
        preview_manifest(name)
    finally:
        env.close()


def preview_manifest(name):
    import matplotlib.pyplot as plt
    import numpy as np
    data = json.loads((RESULTS / f'manifests/{name}.json').read_text())
    banks = {b['tile_index']: b for b in json.loads((RESULTS / f'manifests/{name}_geometry.json').read_text())}
    folder = STUDY / 'analysis'
    folder.mkdir(exist_ok=True)
    for page in range(4):
        fig, axes = plt.subplots(4, 4, figsize=(13, 13))
        for i, ax in enumerate(axes.flat, page*16):
            entry = data['episodes'][i]
            bank = banks[entry['tile_index']]
            cells = np.array(bank['cells'])
            ax.scatter(*cells.T, s=3, c='#bb4740', marker='s', linewidths=0)
            start, goal = np.array(entry['root_pose_relative'][:2]), np.array(entry['goal_xy_relative'])
            route = np.array(entry['hard_audit']['reference_route'])
            ax.plot(*np.array([start, goal]).T, '--', c='0.5', lw=1)
            ax.plot(*route.T, c='#24766f', lw=1.6)
            ax.scatter(*start, c='#24766f', s=25)
            ax.scatter(*goal, c='#d29716', marker='*', s=65)
            yaw = 2*math.atan2(entry['root_pose_relative'][6], entry['root_pose_relative'][3])
            ax.arrow(*start, .6*math.cos(yaw), .6*math.sin(yaw), width=.04, color='#24766f')
            a = entry['hard_audit']
            ax.set_title(f'{i:02d} tile{entry["tile_index"]}: {a["physical_blocking_components"]} blocks, {a["clearance_grid_route_ratio"]:.2f}x', fontsize=9)
            ax.set_aspect('equal')
            sx, sy = bank['half_size']
            ax.set_xlim(-sx, sx); ax.set_ylim(-sy, sy)
            ax.tick_params(labelsize=7)
        fig.suptitle(f'{name} | red: physical obstacles | dashed: blocked direct line | teal: feasible reference (NOT policy)', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(folder / f'{name}_layouts_{page+1}.png', dpi=125)
        plt.close(fig)
    sync_compact()


def child(name, mode, args, timeout_s):
    spec = RESULTS / f'specs/{name}.json'
    if spec.exists():
        raise RuntimeError(f'Immutable child already exists: {name}; do not repeat completed work')
    write_json(spec, dict(mode=mode, args=vars(args)))
    log_path = STUDY / f'logs/{name}.log'
    log_path.parent.mkdir(exist_ok=True)
    status(state='running', active=name, active_mode=mode, active_log=str(log_path))
    with log_path.open('w') as log:
        proc = subprocess.Popen([sys.executable, __file__, 'execute', str(spec)],
            cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status(child_pid=proc.pid)
        try:
            code = proc.wait(timeout_s)
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            raise
    if code:
        raise RuntimeError(f'{name} failed with {code}; see {log_path}')
    status(child_pid=None, last_completed=name)


def execute(path):
    spec = json.loads(Path(path).read_text())
    args = argparse.Namespace(**spec['args'])
    if spec['mode'] == 'train':
        from train_unitree_nav_thesis import run_training
        run_training(args)
    else:
        import eval_unitree_nav_baselines as evaluation
        if args.model_path:
            from unitree_nav_checkpoint import apply_unitree_checkpoint_config
            apply_unitree_checkpoint_config(args)
            hard_settings(args)
        evaluation.evaluate(args)


def check_inputs():
    hashes = json.loads((STUDY / 'inputs.sha256.json').read_text())
    for path, sha in hashes.items():
        if digest(STUDY / path) != sha:
            raise RuntimeError(f'Frozen input changed: {path}')


def summarize(path, manifest):
    import numpy as np
    data = json.loads(Path(path).read_text())
    cases = json.loads(Path(manifest).read_text())['episodes']
    episodes = data['episodes']
    indices = [e['manifest_episode_index'] for e in episodes]
    if indices != list(range(len(episodes))):
        raise ValueError('Evaluation case order/coverage mismatch')
    for e, c in zip(episodes, cases):
        if e['terrain_tile_id'] != c['tile_index']:
            raise ValueError('Evaluation tile mismatch')
        for actual, expected in [('start_local_xy', c['root_pose_relative'][:2]), ('goal_local_xy', c['goal_xy_relative']), ('start_quaternion', c['root_pose_relative'][3:])]:
            if not np.allclose(e[actual], expected, atol=3e-4, rtol=0):
                raise ValueError(f'Evaluation {actual} mismatch')
        if not e['blocked_corridor'] or e['blocking_component_count'] < 2:
            raise ValueError('Evaluation lost blockage')
    cost = np.array([e.get('collision_cost_sum', e['cost_sum']) for e in episodes])
    total = np.array([e['cost_sum'] for e in episodes])
    paths, progress = [], []
    for e in episodes:
        trajectory = np.asarray(e.get('trajectory_local_xy') or [e['start_local_xy']],dtype=float)
        paths.append(float(np.linalg.norm(np.diff(trajectory,axis=0),axis=1).sum()))
        initial = float(np.linalg.norm(np.subtract(e['goal_local_xy'],e['start_local_xy'])))
        final = float(np.linalg.norm(np.subtract(e['goal_local_xy'],trajectory[-1])))
        progress.append((initial-final)/max(initial,1e-9))
    return dict(episodes=len(episodes), costful_episodes=int((cost>0).sum()),
        accumulated_collision_cost=float(cost.sum()), mean_collision_cost=float(cost.mean()),
        p95_collision_cost=float(np.quantile(cost, .95)), max_collision_cost=float(cost.max()),
        conditional_mean_collision_cost=float(cost[cost>0].mean()) if (cost>0).any() else 0.0,
        success=int(sum(e['success'] for e in episodes)), falls=int(sum(e.get('fall_cost_sum',0)>0 for e in episodes)),
        zero_cost_non_success=int(sum(c==0 and not e['success'] for c,e in zip(total,episodes))),
        mean_duration_s=float(np.mean([e['episode_length_s'] for e in episodes])),
        mean_path_length=float(np.mean(paths)),mean_net_progress_fraction=float(np.mean(progress)),
        mean_min_goal_distance=float(np.mean([e['min_goal_distance'] for e in episodes])),
        manifest_sha256=digest(manifest), metrics_sha256=digest(path))


def plot_rollouts(path, manifest, name):
    import matplotlib.pyplot as plt
    import numpy as np
    episodes = json.loads(Path(path).read_text())['episodes']
    for page in range(math.ceil(len(episodes)/16)):
        fig, axes = plt.subplots(4, 4, figsize=(13, 13))
        for offset, ax in enumerate(axes.flat):
            i = page*16+offset
            if i >= len(episodes):
                ax.axis('off'); continue
            e = episodes[i]
            origin = np.array(e['env_origin_xy'])
            cells = np.array(e['obstacle_cells_xy']) - origin
            ax.scatter(*cells.T, s=3, c='#bb4740', marker='s', linewidths=0)
            trajectory = np.array(e['trajectory_local_xy'])
            ax.plot(*trajectory.T, c='#24766f', lw=1.4)
            start, goal = np.array(e['start_local_xy']), np.array(e['goal_local_xy'])
            ax.plot(*np.array([start,goal]).T, '--', color='0.6', lw=.8)
            ax.scatter(*start, c='#24766f', s=25)
            ax.scatter(*goal, c='#d29716', marker='*', s=65)
            if e['cost_points_local_xy']:
                ax.scatter(*np.array(e['cost_points_local_xy']).T, marker='x', c='black', s=24)
            ax.set_title(f'{i:02d}: goal={int(e["success"])} cost={e["cost_sum"]:.0f}', fontsize=9)
            ax.set_aspect('equal')
        fig.suptitle(name + ' | policy trajectory, black crosses = costs; contacts do not end episodes')
        fig.tight_layout(rect=(0,0,1,.97))
        fig.savefig(STUDY / f'analysis/{name}_trajectories_{page+1}.png', dpi=125)
        plt.close(fig)


def learning_curves(run_dir, name):
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in (Path(run_dir)/'metrics.jsonl').read_text().splitlines() if l.strip()]
    panels = [('alpha','Learned alpha'), ('teacher_fraction_cumulative','Cumulative teacher execution'),
              ('teacher_fraction_interval','Recent teacher execution'), ('batch_teacher_fraction','Sampled TD teacher fraction'),
              ('pref_source_rows','Sampled preference rows'), ('pref_linked_rows','Usable preference support'),
              ('actor_loss_bc','Teacher-only BC loss'), ('pref_violation_rate','Preference violation')]
    fig, axes = plt.subplots(2,4,figsize=(16,7))
    for ax,(key,title) in zip(axes.flat,panels):
        valid = [(r['step'],r[key]) for r in rows if key in r]
        if valid:
            ax.plot(*zip(*valid),color='#24766f')
        ax.set_title(title);ax.set_xlabel('5 Hz navigation decisions');ax.grid(alpha=.2)
    fig.suptitle(name+' | collection metrics, not teacher-free policy quality')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(STUDY/f'analysis/{name}_learning.png',dpi=130)
    plt.close(fig)


def paired_analysis():
    import numpy as np
    comparison = {}
    for cohort in ('validation','heldout_test'):
        comparison[cohort] = []
        for seed in (73,74):
            paths = [RESULTS/f'evaluations/alpha_student_separate_bc{bc:g}_seed{seed}_4500_{cohort}/policy_metrics.json'
                     for bc in (0,.2)]
            episodes = [json.loads(p.read_text())['episodes'] for p in paths]
            delta = [dict(case_index=i,collision_delta_bc_minus_no_bc=b['collision_cost_sum']-a['collision_cost_sum'],
                success_delta_bc_minus_no_bc=int(b['success'])-int(a['success']),
                duration_delta_bc_minus_no_bc=b['episode_length_s']-a['episode_length_s'])
                for i,(a,b) in enumerate(zip(*episodes))]
            comparison[cohort].append(dict(seed=seed,paired_cases=delta,
                summed_collision_delta=float(sum(d['collision_delta_bc_minus_no_bc'] for d in delta)),
                successful_episode_delta=sum(d['success_delta_bc_minus_no_bc'] for d in delta),
                mean_duration_delta=float(np.mean([d['duration_delta_bc_minus_no_bc'] for d in delta]))))
    write_json(STUDY/'analysis/paired_comparison.json',comparison)


def wandb_export(name, run_dir, finals):
    import wandb
    run_id = hashlib.sha256(('unitree-blocked-bc-20261012/'+name).encode()).hexdigest()[:16]
    plan = json.loads((STUDY / 'plan.json').read_text())
    config = {k:plan[k] for k in ('seeds','decisions_per_training','bc_weights','bc_semantics',
        'sampling','alpha_state_scope','fixed_lambda','train_seeds','validation_seed','heldout_seed',
        'cases_per_manifest','initialization','criteria','ranking')}
    config['cohort_role'] = 'training' if run_dir else ('preflight' if name.startswith('preflight_') else 'baseline')
    if run_dir:
        resolved = json.loads((Path(run_dir)/'args.json').read_text())
        config['resolved'] = {k:v for k,v in resolved.items() if isinstance(v,(int,float,bool,str))
            and not any(word in k for word in ('path','dir','checkpoint','manifest','wandb'))}
        config['bc_weight'] = resolved['actor_bc_weight']
        config['training_seed'] = resolved['seed']
    (STUDY/'wandb_runtime').mkdir(exist_ok=True)
    receipts = STUDY/'wandb_metrics_export'
    receipts.mkdir(exist_ok=True)
    receipt_path = receipts/(name+'.json')
    payload_hashes = {k:digest(p) for k,(p,_) in finals.items()}
    if run_dir:
        for name_part in ('metrics.jsonl','eval_metrics.jsonl'):
            source = Path(run_dir)/name_part
            if source.exists():
                payload_hashes[name_part] = digest(source)
    fingerprint = hashlib.sha256(json.dumps(payload_hashes,sort_keys=True).encode()).hexdigest()
    remote = None
    try:
        remote = wandb.Api().run(f'bmeyjohann/thesis-unitree-nav/{run_id}')
    except wandb.errors.CommError:
        pass
    if remote and remote.summary.get('export_fingerprint') == fingerprint:
        write_json(receipt_path,dict(run_id=run_id,fingerprint=fingerprint,url=remote.url,verified_unix=time.time()))
        return
    uploaded_rows = set()
    if remote:
        uploaded_rows = {int(r['train/source_row']) for r in remote.scan_history(keys=['train/source_row'])
                         if r.get('train/source_row') is not None}
    run = wandb.init(entity='bmeyjohann', project='thesis-unitree-nav', id=run_id,
        resume='allow', name=name, group='unitree-blocked-bc-20261012', job_type='blocked-bc-comparison',
        config=config, settings=wandb.Settings(disable_code=True,disable_git=True,console='off',
            x_disable_stats=True,x_disable_meta=True,x_save_requirements=False,init_timeout=60,quiet=True), mode='online',
        dir=str(STUDY / 'wandb_runtime'))
    if run_dir:
        rows = [json.loads(l) for l in (Path(run_dir)/'metrics.jsonl').read_text().splitlines() if l.strip()]
        numeric = [{k:v for k,v in row.items() if isinstance(v,(int,float,bool)) and (not isinstance(v,float) or math.isfinite(v))} for row in rows]
        keys = sorted(set().union(*(r.keys() for r in numeric)))
        run.log({'training_trace': wandb.Table(columns=keys, data=[[r.get(k) for k in keys] for r in numeric])})
        if uploaded_rows != set(range(len(uploaded_rows))):
            raise RuntimeError('Remote training history is not an exact prefix')
        for index,row in enumerate(numeric):
            if index not in uploaded_rows:
                run.log({**{'train/'+k:v for k,v in row.items()},'train/source_row':index})
        if rows:
            run.summary.update({'last/'+k:v for k,v in numeric[-1].items()})
        periodic = Path(run_dir)/'eval_metrics.jsonl'
        if periodic.exists():
            for line in periodic.read_text().splitlines():
                row = json.loads(line)
                run.log({'periodic/'+k:v for k,v in row.items() if isinstance(v,(int,float,bool))})
        for metrics in sorted((Path(run_dir)/'eval').glob('*/policy_metrics.json')):
            data = json.loads(metrics.read_text())
            episodes = data.get('episodes', [])
            if episodes:
                keys = sorted(k for k,v in episodes[0].items() if isinstance(v,(int,float,bool)) or v is None)
                run.log({f'periodic/{metrics.parent.name}/episodes': wandb.Table(columns=keys,
                    data=[[e.get(k) for k in keys] for e in episodes])})
    for cohort, (metrics, manifest) in finals.items():
        summary = summarize(metrics, manifest)
        run.summary.update({cohort+'/'+k:v for k,v in summary.items()})
        episodes = json.loads(Path(metrics).read_text())['episodes']
        keys = sorted(k for k,v in episodes[0].items() if isinstance(v,(int,float,bool)) or v is None)
        run.log({cohort+'/episode_table': wandb.Table(columns=keys, data=[[e.get(k) for k in keys] for e in episodes])})
    run.summary['export_fingerprint'] = fingerprint
    url = run.url
    run.finish()
    api_run = wandb.Api().run(f'bmeyjohann/thesis-unitree-nav/{run_id}')
    for cohort, (_, manifest) in finals.items():
        if api_run.summary.get(cohort+'/manifest_sha256') != digest(manifest):
            raise RuntimeError('W&B remote manifest fingerprint mismatch')
        expected_count = summarize(finals[cohort][0],manifest)['episodes']
        if api_run.summary.get(cohort+'/episodes') != expected_count:
            raise RuntimeError('W&B remote episode count mismatch')
    write_json(receipt_path,dict(run_id=run_id,fingerprint=fingerprint,url=url,verified_unix=time.time()))
    status(wandb_last_verified_url=url)


def reuse_completed_preflight(name, args, manifest):
    record = json.loads((STUDY/'plan.json').read_text()).get('reuse_preflight', {}).get(name)
    if not record:
        return False
    for key in ('metrics', 'spec', 'manifest'):
        if digest(record[key]) != record[key+'_sha256']:
            raise RuntimeError(f'Reused preflight input changed: {key}')
    if digest(manifest) != record['manifest_sha256']:
        raise RuntimeError('Reused preflight cohort differs')
    old = json.loads(Path(record['spec']).read_text())
    source = record['source_study']
    resolved = {k:v.replace(source, str(STUDY)) if isinstance(v,str) else v
                for k,v in old['args'].items()}
    if old['mode'] != 'evaluate' or resolved != vars(args):
        raise RuntimeError('Reused preflight resolved configuration differs')
    summary = summarize(record['metrics'], manifest)
    if summary['episodes'] != args.num_episodes:
        raise RuntimeError('Reused preflight coverage differs')
    target = RESULTS/f'evaluations/{name}/{args.controller}_metrics.json'
    spec = RESULTS/f'specs/{name}.json'
    if target.exists() or spec.exists():
        raise RuntimeError('Recovery output already exists; no automatic repetition')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record['metrics'], target)
    write_json(spec, dict(mode='evaluate', args=vars(args), reused_from=record))
    status(last_completed=name, reused_completed_preflight=name)
    return True


def run():
    check_inputs()
    plan = json.loads((STUDY/'plan.json').read_text())
    if not plan.get('all_layout_pages_inspected'):
        raise RuntimeError('All saved hard-layout pages must be inspected before training')
    # Teacher and scan-blind controls are explicit, not omitted from a method comparison.
    for cohort, seed in [('train_seed73',73), ('train_seed74',74), ('validation',2941)]:
        for controller in (['geom_scan_teacher','direct_goal'] if cohort=='validation' else ['geom_scan_teacher']):
            name = f'preflight_{cohort}_{controller}'
            manifest = RESULTS / f'manifests/{cohort}.json'
            args = eval_args(seed,controller,manifest,name,episodes=8)
            if not reuse_completed_preflight(name, args, manifest):
                child(name, 'evaluate', args, 2400)
            path = RESULTS / f'evaluations/{name}/{controller}_metrics.json'
            summary = summarize(path, manifest)
            if controller == 'geom_scan_teacher' and (summary['success'] < 4 or summary['falls'] > 2):
                raise RuntimeError('Teacher cannot reliably solve hard preflight; inspect before training, do not weaken cohort')
            write_json(STUDY / f'analysis/{name}.json', summary)
            plot_rollouts(path, manifest, name)
            try:
                wandb_export(name,None,{cohort:(path,manifest)})
            except Exception as error:
                status(wandb_pending_name=name,wandb_error=repr(error))
    completed_trainings = []
    for seed in (73,74):
        for bc in (0.0,0.2):
            name = f'alpha_student_separate_bc{bc:g}_seed{seed}_4500'
            args = study_module().train_args(seed=seed, steps=4500,
                **study_module().ARMS['alpha_student'], actor_bc_weight=bc, run_name=name,
                eval_seed=2941, eval_timeout_s=7200)
            hard_settings(args)
            child(name, 'train', args, 28800)
            run_dir = RESULTS / f'runs/{name}'
            checkpoints = list(run_dir.rglob('*.pt'))
            import torch
            final = [p for p in checkpoints if torch.load(p,map_location='cpu',weights_only=False).get('step') == 4500]
            if not final:
                raise RuntimeError('Missing final checkpoint; never restart completed training for evaluation failure')
            checkpoint = sorted(final, key=lambda p:p.name)[0]
            completed_trainings.append(dict(name=name,checkpoint=str(checkpoint),sha256=digest(checkpoint)))
            status(completed_trainings=completed_trainings)
            rows = [json.loads(l) for l in (run_dir/'metrics.jsonl').read_text().splitlines() if l.strip()]
            last = rows[-1]
            if last.get('transition_unit') != 'navigation_decision' or last.get('environment_ticks_collected', math.inf)>18000:
                raise RuntimeError('Navigation decision cadence invalid')
            if bc and not any(row.get('actor_bc_rows',0)>0 and row.get('actor_loss_bc',0)>0 for row in rows):
                raise RuntimeError('BC arm produced no verified BC updates')
            learning_curves(run_dir,name)
            finals = {}
            for cohort, eval_seed in [('validation',2941),('heldout_test',3941)]:
                manifest = RESULTS / f'manifests/{cohort}.json'
                eval_name = name + '_' + cohort
                child(eval_name, 'evaluate', eval_args(eval_seed,'policy',manifest,eval_name,checkpoint), 10800)
                path = RESULTS / f'evaluations/{eval_name}/policy_metrics.json'
                summary = summarize(path,manifest)
                if summary['episodes'] !=64:
                    raise RuntimeError('Final evaluation incomplete')
                write_json(STUDY / f'analysis/{eval_name}.json', summary)
                plot_rollouts(path,manifest,eval_name)
                finals[cohort] = path,manifest
            try:
                wandb_export(name,run_dir,finals)
            except Exception as error:
                status(wandb_pending_name=name,wandb_error=repr(error))
            status(last_training_completed=name, last_checkpoint=str(checkpoint))
    for controller in ('direct_goal','geom_scan_teacher'):
        manifest = RESULTS/'manifests/validation.json'
        name = 'baseline_validation_'+controller
        child(name,'evaluate',eval_args(2941,controller,manifest,name),10800)
        path = RESULTS/f'evaluations/{name}/{controller}_metrics.json'
        write_json(STUDY/f'analysis/{name}.json',summarize(path,manifest))
        plot_rollouts(path,manifest,name)
        try:
            wandb_export(name,None,{'validation':(path,manifest)})
        except Exception as error:
            status(wandb_pending_name=name,wandb_error=repr(error))
    paired_analysis()
    status(state='numerical_complete_visual_review_pending', active=None, child_pid=None)


def main():
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}; artifacts preserved')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    command = sys.argv[1]
    if command == 'execute':
        execute(sys.argv[2]); return
    with open('/tmp/unitree_core_gpu0.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status(controller_pid=os.getpid(),gpu_lock='/tmp/unitree_core_gpu0.lock')
        try:
            if command == 'prepare':
                for name,seed in [('train_seed73',73),('train_seed74',74),('validation',2941),('heldout_test',3941)]:
                    status(state='preparing_hard_cohorts',active=name)
                    make_manifest(name,seed)
                status(state='awaiting_layout_visual_review',active=None)
            elif command == 'run':
                run()
            else:
                raise ValueError(command)
        except BaseException as error:
            status(state='failed_preserved_no_automatic_retry',error=repr(error))
            raise


if __name__ == '__main__':
    main()
