"""CPU-only, explicit-opt-in W&B backfill of authoritative Unitree metrics."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np


GROUP = 'unitree-obstacle-study-20261009'
VERSION = 1


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if not re.search(r'api_key|password|secret|token|credential', k, re.I)}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Geometry diagnostics use infinity for an absent nearby obstacle.
        return 'nonfinite:' + str(value)
    if isinstance(value, str) and (value.startswith(('/', 'C:\\', 'D:\\'))):
        return 'local-file:' + Path(value.replace('\\', '/')).name
    return value


def jsonl(path):
    if not path.exists():
        return []
    # The live worker may be in the middle of appending its final line.
    raw = path.read_text()
    lines = raw.splitlines()
    if raw and not raw.endswith('\n'):
        lines = lines[:-1]
    return [clean(json.loads(line)) for line in lines if line.strip()]


def numeric(row):
    return {k: v for k, v in row.items() if isinstance(v, (int, float, bool))}


def table_data(episodes):
    columns = sorted({k for ep in episodes for k, v in ep.items()
                      if isinstance(v, (int, float, bool, str))})
    rows = []
    for ep in episodes:
        nonfinite = {k: v for k, v in ep.items()
                     if isinstance(v, str) and v.startswith('nonfinite:')}
        rows.append([None if k in nonfinite else ep.get(k) for k in columns] +
                    [json.dumps(nonfinite, sort_keys=True)])
    return columns + ['nonfinite_fields_json'], rows


def history_cursor(summary, rows, prefix):
    key = 'training' if prefix == 'train' else 'periodic'
    count = int(summary.get('import/'+key+'_rows', 0))
    assert count <= len(rows)
    if count:
        assert digest(rows[:count]) == summary['import/'+key+'_prefix_sha256']
    # Recover a completed SDK flush if an interrupted upload did not save its cursor.
    last_decision = summary.get(prefix+'/decision')
    if last_decision is not None:
        assert any(r['step'] == last_decision for r in rows), 'Remote history not in local source'
        count = max(count, sum(r['step'] <= last_decision for r in rows))
    return count


def run_id(host, name):
    return hashlib.sha256(f'{GROUP}/{host}/{name}'.encode()).hexdigest()[:16]


def check_episodes(data, manifest, partial=False):
    episodes = data['episodes']
    indices = sorted(e['manifest_episode_index'] for e in episodes)
    assert indices == list(range(len(episodes))), 'Missing, repeated or filtered cases'
    assert len(episodes) == 64 or (partial and len(episodes) == 32)
    for e in episodes:
        c = manifest['episodes'][e['manifest_episode_index']]
        for a, b in [(e['start_local_xy'], c['root_pose_relative'][:2]),
                     (e['goal_local_xy'], c['goal_xy_relative']),
                     (e['start_quaternion'], c['root_pose_relative'][3:7])]:
            assert np.shape(a) == np.shape(b) and np.allclose(a, b, atol=1e-4, rtol=0)
        assert e['terrain_tile_id'] == c['tile_index']
        assert all(math.isfinite(e[k]) for k in ['cost_sum', 'collision_cost_sum', 'fall_cost_sum'])
        assert e['cost_sum'] >= 0 and math.isclose(
            e['cost_sum'], e['collision_cost_sum'] + e['fall_cost_sum'], abs_tol=1e-6)


def episode_summary(episodes):
    cost = np.array([e['cost_sum'] for e in episodes])
    collision = np.array([e['collision_cost_sum'] for e in episodes])
    progress, lengths = [], []
    for e in episodes:
        path = np.asarray(e['trajectory_local_xy'])
        start, goal = np.asarray(e['start_local_xy']), np.asarray(e['goal_local_xy'])
        progress.append((np.linalg.norm(goal-start)-np.linalg.norm(goal-path[-1])) /
                        max(float(np.linalg.norm(goal-start)), 1e-6))
        lengths.append(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()))
    return dict(episodes=len(episodes), costful_episode_rate=float((cost > 0).mean()),
                collision_episode_rate=float((collision > 0).mean()),
                collision_cost_sum=float(collision.sum()), collision_cost_mean=float(collision.mean()),
                total_cost_sum=float(cost.sum()), total_cost_mean=float(cost.mean()),
                max_episode_cost=float(cost.max()), max_episode_collision_cost=float(collision.max()),
                p95_episode_cost=float(np.percentile(cost, 95)),
                p95_episode_collision_cost=float(np.percentile(collision, 95)),
                success_rate=sum(bool(e['success']) for e in episodes)/len(episodes),
                falls=sum(e['fall_cost_sum'] > 0 for e in episodes),
                zero_cost_non_success=sum(not e['success'] and e['cost_sum'] == 0 for e in episodes),
                mean_episode_seconds=float(np.mean([e['episode_length_s'] for e in episodes])),
                mean_net_progress=float(np.mean(progress)), mean_path_length=float(np.mean(lengths)))


def intervention_summary(rows):
    samples = [r for r in rows if 'teacher_fraction_cumulative' in r]
    if not samples:
        return {}
    x = np.array([r['step'] for r in samples], dtype=float)
    y = np.array([r['teacher_fraction_cumulative'] for r in samples], dtype=float)
    interval = [r for r in samples if 'teacher_fraction_interval' in r]
    # Interval fractions describe the covered logging windows, not independent episodes.
    total = sum(max(0, r['step'] - (interval[i-1]['step'] if i else 0)) *
                r['teacher_fraction_interval'] for i, r in enumerate(interval))
    duration = interval[-1]['step'] if interval else 0
    late = [r['teacher_fraction_interval'] for r in interval if r['step'] > x[-1]-1000]
    return dict(final_teacher_fraction=float(y[-1]),
                cumulative_teacher_fraction_auc=float(np.trapz(y, x)/(x[-1]-x[0])) if len(x)>1 else float(y[-1]),
                teacher_interval_time_average=total/duration if duration else float(y[-1]),
                late1000_teacher_fraction=float(np.mean(late)) if late else float(y[-1]),
                teacher_collection_cost=sum(r.get('teacher_executed_cost_sum_interval', 0) for r in rows),
                student_collection_cost=sum(r.get('student_executed_cost_sum_interval', 0) for r in rows))


def frontier(rows):
    """No arbitrary scalar weights or cost threshold; compare within host/cohort."""
    keys = ['collision_episode_rate', 'collision_cost_sum', 'max_episode_collision_cost']
    for row in rows:
        peers = [r for r in rows if r['host'] == row['host'] and
                 r['cohort'] == row['cohort'] and r['kind'] == 'policy']
        row['cost_frontier'] = row['kind'] == 'policy' and not any(
            all(p[k] <= row[k] for k in keys) and any(p[k] < row[k] for k in keys)
            for p in peers)
    return rows


def inventory(base):
    schedule = read(base/'priority_resource_schedule.json')
    core = Path(schedule['laptop']['root'])
    follow = Path(schedule['alpha_sampling_followup']['root'])
    desktop = base/'desktop_core_snapshot_20261010T1226'
    expected = read(follow/'control/followup.json')['manifest_sha256']
    entries = {}
    for host, root, names in [
        ('laptop', core, [r['name'] for r in read(core/'results/status.json')['completed']]),
        ('desktop', desktop, [r['name'] for r in read(desktop/'results/status.json')['completed']]),
        ('laptop', follow, [r['name'] for r in read(follow/'control/followup.json')['new_training']])]:
        manifests = {p.name: read(p) for p in (root/'results/manifests').glob('*.json')}
        for filename, checksum in expected.items():
            assert hashlib.sha256((root/'results/manifests'/filename).read_bytes()).hexdigest() == checksum
        for name in names:
            entries[(host, name)] = dict(host=host, name=name, kind='policy', root=root,
                                        directory=root/'results/runs'/name, manifests=manifests)
        for name in ['direct_goal', 'teacher']:
            # Prefer the core's original complete baseline, not its follow-up copies.
            entries.setdefault((host, name), dict(host=host, name=name, kind='baseline',
                               root=root, directory=None, manifests=manifests))
    return list(entries.values()), follow


def snapshot(entry, follow):
    root, name, directory = entry['root'], entry['name'], entry['directory']
    rows = jsonl(directory/'metrics.jsonl') if directory else []
    evaluations = {}
    # Follow-up may add heldouts for inherited core policies later.
    roots = [root] + ([follow] if entry['host']=='laptop' and root != follow else [])
    for source in roots:
        for p in (source/'results/evaluations').glob(name+'_*/*_metrics.json'):
            label = p.parent.name[len(name)+1:]
            data = read(p)
            if label in evaluations:
                assert clean(evaluations[label]) == clean(data), 'Conflicting duplicate evaluation'
                continue
            check_episodes(data, entry['manifests'][Path(data['layout_manifest']).name])
            evaluations[label] = data
    if directory:
        for p in directory.glob('eval/step_*/*_metrics.json'):
            data = read(p)
            check_episodes(data, entry['manifests'][Path(data['layout_manifest']).name], partial=True)
            evaluations['periodic_'+p.parent.name] = data
    args = clean(read(directory/'args.json')) if directory and (directory/'args.json').exists() else {}
    return dict(args=args, training=rows,
                periodic=jsonl(directory/'eval_metrics.jsonl') if directory else [],
                evaluations={k: clean(v) for k, v in evaluations.items()})


def upload_one(wandb, api, entry, data, cli, export):
    identifier = run_id(entry['host'], entry['name'])
    path = f'{cli.entity}/{cli.project}/{identifier}'
    fingerprint = digest(data)
    existing = None
    try:
        existing = api.run(path)
    except (wandb.errors.CommError, ValueError) as error:
        if 'not find' not in str(error).lower() and 'not found' not in str(error).lower():
            raise
    summary = dict(existing.summary) if existing else {}
    if summary.get('import/fingerprint') == fingerprint:
        return dict(id=identifier, url=existing.url, skipped=True, name=entry['name'], host=entry['host'])
    config = dict(data['args'], study=GROUP, original_host=entry['host'], kind=entry['kind'],
                  import_only=True, exporter_version=VERSION,
                  heldout_reused=True, comparison='Within-host/cohort; cross-host exploratory')
    config['intervention_comparable'] = entry['kind']=='policy' and not entry['name'].startswith('ungated_rl')
    settings = wandb.Settings(disable_code=True, disable_git=True, console='off',
                              x_disable_stats=True, x_disable_meta=True,
                              x_save_requirements=False, init_timeout=60, quiet=True)
    run = wandb.init(entity=cli.entity, project=cli.project, id=identifier, resume='allow',
                     name=f"{entry['host']}/{entry['name']}", group=GROUP, job_type='metrics-backfill',
                     config=config, tags=['obstacle-only', 'cost-first', entry['host'], entry['kind']],
                     dir=str(export), settings=settings, mode='online')
    try:
        for prefix, axis in [('train', 'train/decision'), ('periodic', 'periodic/decision')]:
            run.define_metric(axis)
            run.define_metric(prefix+'/*', step_metric=axis)
        old_train = history_cursor(summary, data['training'], 'train')
        old_periodic = history_cursor(summary, data['periodic'], 'periodic')
        for row in data['training'][old_train:]:
            run.log({'train/decision': row['step'], **{'train/'+k:v for k,v in numeric(row).items()}})
        for row in data['periodic'][old_periodic:]:
            run.log({'periodic/decision': row['step'], **{'periodic/'+k:v for k,v in numeric(row).items()}})
        for key, rows in [('training', data['training']), ('periodic', data['periodic'])]:
            run.summary['import/'+key+'_rows'] = len(rows)
            run.summary['import/'+key+'_prefix_sha256'] = digest(rows)
        for label, value in data['evaluations'].items():
            if label.startswith('periodic_'):
                continue
            scalars = {**numeric(value), **episode_summary(value['episodes'])}
            for k, v in scalars.items():
                run.summary[f'{label}/{k}'] = v
            columns, rows = table_data(value['episodes'])
            table = wandb.Table(columns=columns, data=rows)
            run.log({label+'/episodes': table})
        for k,v in intervention_summary(data['training']).items():
            run.summary['intervention/'+k] = v
        run.summary['training_complete'] = bool(data['training'] and data['training'][-1]['step']==4500)
        run.summary['import/training_rows'] = len(data['training'])
        run.summary['import/periodic_rows'] = len(data['periodic'])
        run.summary['import/training_prefix_sha256'] = digest(data['training'])
        run.summary['import/periodic_prefix_sha256'] = digest(data['periodic'])
        run.summary['import/evaluation_cohorts'] = list(data['evaluations'])
        run.summary['import/verified_matched_cases'] = True
        bundle = export/'payloads'/identifier/fingerprint
        bundle.mkdir(parents=True, exist_ok=True)
        atomic(bundle/'metrics.json', data)
        artifact = wandb.Artifact('unitree-metrics-'+identifier, type='study-metrics',
                                  metadata=dict(import_fingerprint=fingerprint, manifest_matched=True))
        artifact.add_file(str(bundle/'metrics.json'))
        run.log_artifact(artifact).wait()
        run.summary['import/fingerprint'] = fingerprint
    finally:
        run.finish()
    for attempt in range(3):
        # Public API Run.summary caches separately from Run.load(force=True).
        remote = wandb.Api(timeout=30).run(path)
        if remote.summary.get('import/fingerprint') == fingerprint:
            break
        time.sleep(2)
    assert remote.summary.get('import/fingerprint') == fingerprint, 'Remote fingerprint not yet verified'
    assert remote.summary.get('import/training_rows') == len(data['training'])
    return dict(id=identifier, url=remote.url, skipped=False, name=entry['name'], host=entry['host'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--entity', required=True)
    parser.add_argument('--project', default='thesis-unitree-nav')
    parser.add_argument('--upload', action='store_true', help='Explicit authorization gate for network writes')
    cli = parser.parse_args()
    export = cli.study_root/'wandb_metrics_export'
    export.mkdir(exist_ok=True)
    with (export/'export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries, follow = inventory(cli.study_root)
        payloads, comparison = [], []
        for entry in entries:
            data = snapshot(entry, follow)
            if not data['training'] and not data['evaluations']:
                continue
            payloads.append((entry, data))
            for cohort in ['validation', 'heldout']:
                if cohort in data['evaluations']:
                    comparison.append(dict(host=entry['host'], name=entry['name'], kind=entry['kind'],
                                           cohort=cohort, **episode_summary(data['evaluations'][cohort]['episodes']),
                                           **intervention_summary(data['training'])))
        comparison = frontier(comparison)
        atomic(export/'comparison.json', comparison)
        state = dict(group=GROUP, entity=cli.entity, project=cli.project,
                     url=f'https://wandb.ai/{cli.entity}/{cli.project}/groups/{GROUP}',
                     expected_configurations=31, available_runs=len(payloads),
                     ranking='Qualitative low-cost group; success second; intervention third; no scalar score',
                     no_safety_claim=True, state='validated', checked_unix=time.time())
        atomic(export/'status.json', state)
        if not cli.upload:
            print(json.dumps(state)); return
        import wandb
        api = wandb.Api(timeout=30)
        results = []
        try:
            for entry, data in payloads:
                result = upload_one(wandb, api, entry, data, cli, export)
                results.append(result)
                state.update(state='uploading', runs=results, checked_unix=time.time())
                atomic(export/'status.json', state)
                print(json.dumps(result), flush=True)
            state.update(state='uploaded_current_snapshot', checked_unix=time.time())
            atomic(export/'status.json', state)
        except BaseException as error:
            state.update(state='failed', error=str(error), checked_unix=time.time())
            atomic(export/'status.json', state)
            raise
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
