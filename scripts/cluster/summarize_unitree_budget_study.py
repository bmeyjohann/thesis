#!/usr/bin/env python3
"""CPU-only, partial-safe reporting for the bounded Unitree study."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(os.environ.get('UNITREE_STUDY_ROOT', ROOT.parent))
RESULTS = STUDY / 'results'
OUT = RESULTS / 'analysis'


def read_json(path):
    return json.loads(Path(path).read_text())


def cohort_check(data):
    path = Path(data['layout_manifest'])
    manifest = read_json(path)
    episodes = data['episodes']
    indices = [ep['manifest_episode_index'] for ep in episodes]
    errors = []
    if sorted(indices) != list(range(len(episodes))):
        errors.append('episode coverage/order')
    for ep in episodes:
        case = manifest['episodes'][ep['manifest_episode_index']]
        for actual, expected, label in [
            (ep['start_local_xy'], case['root_pose_relative'][:2], 'start'),
            (ep['goal_local_xy'], case['goal_xy_relative'], 'goal'),
            (ep.get('start_quaternion', []), case['root_pose_relative'][3:7], 'orientation')]:
            if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, atol=1e-4, rtol=0):
                errors.append(f"case {ep['manifest_episode_index']} {label}")
        if ep['terrain_tile_id'] != case['tile_index']:
            errors.append('tile')
    return {'episodes': len(episodes), 'matched': not errors, 'errors': errors}


def plot_curves(runs, keys, name):
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 3.2 * len(keys)), squeeze=False)
    for run, args, rows in runs:
        for axis, key in zip(axes[:, 0], keys):
            selected = [row for row in rows if key in row]
            if selected:
                axis.plot([r['step'] for r in selected], [r[key] for r in selected], label=run, lw=1.2)
            axis.set_ylabel(key)
            axis.grid(alpha=.2)
    axes[-1, 0].set_xlabel('5 Hz navigation decisions')
    if runs:
        axes[0, 0].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def trajectories(evaluations):
    policies = [(name, data) for name, data in evaluations if data['controller'] == 'policy' and name.endswith('_validation')]
    if not policies:
        return
    policies = policies[:4]
    fig, axes = plt.subplots(2, len(policies), figsize=(4 * len(policies), 8), squeeze=False)
    for column, (name, data) in enumerate(policies):
        for row, case in enumerate([0, 1]):
            ep = next(ep for ep in data['episodes'] if ep['manifest_episode_index'] == case)
            axis = axes[row, column]
            path = np.asarray(ep['trajectory_local_xy'])
            if len(path):
                assert np.allclose(path[0], ep['start_local_xy'], atol=1e-4)
                axis.plot(path[:, 0], path[:, 1], color='#246989')
            cells = np.asarray(ep.get('obstacle_cells_xy', []))
            if len(cells):
                cells = cells[:, :2] - np.asarray(ep['env_origin_xy'])
                axis.scatter(cells[:, 0], cells[:, 1], s=3, color='#b75848', alpha=.25)
            costs = np.asarray(ep.get('cost_points_local_xy', []))
            if len(costs):
                axis.scatter(costs[:, 0], costs[:, 1], s=18, marker='x', c='crimson')
            axis.scatter(*ep['start_local_xy'], marker='o', c='#29834e', label='start')
            axis.scatter(*ep['goal_local_xy'], marker='*', s=110, c='#e8a725', label='goal')
            axis.set_title(f'{name}\ncase {case}, success={ep["success"]}, cost={ep["cost_sum"]:g}', fontsize=8)
            axis.set_aspect('equal', adjustable='datalim')
            axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / 'matched_trajectories.png', dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evaluations, table, checks, runs = [], [], {}, []
    for path in sorted((RESULTS / 'evaluations').glob('*/*_metrics.json')):
        data = read_json(path)
        name = path.parent.name
        check = cohort_check(data)
        checks[name] = check
        episodes = data['episodes']
        table.append(dict(name=name, controller=data['controller'], episodes=len(episodes),
                          matched=check['matched'], success=sum(ep['success'] for ep in episodes) / len(episodes),
                          safe_success=sum(ep['success'] and ep['cost_sum'] == 0 for ep in episodes) / len(episodes),
                          fall_rate=sum(ep.get('fall_cost_sum', 0) > 0 for ep in episodes) / len(episodes),
                          costful_rate=sum(ep['cost_sum'] > 0 for ep in episodes) / len(episodes),
                          mean_cost=float(np.mean([ep['cost_sum'] for ep in episodes]))))
        evaluations.append((name, data))
    for path in sorted((RESULTS / 'runs').glob('*/metrics.jsonl')):
        args = read_json(path.parent / 'args.json')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert args['navigation_policy_hz'] == 5 and args['navigation_transition_mode'] == 'decision'
        if args['pref_rank_weight'] > 0 and args['pref_lambda_lr'] == 0:
            assert all(abs(r['pref_lambda'] - 1) < 1e-6 for r in rows if 'pref_lambda' in r)
        runs.append((path.parent.name, args, rows))
    (OUT / 'cohort_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    (OUT / 'comparison.json').write_text(json.dumps(table, indent=2) + '\n')
    if table:
        with (OUT / 'comparison.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        fig, axis = plt.subplots(figsize=(max(10, len(table) * .45), 5))
        x = np.arange(len(table))
        axis.bar(x - .2, [r['success'] for r in table], .2, label='Success', color='#246989')
        axis.bar(x, [r['safe_success'] for r in table], .2, label='Zero-cost success', color='#43876a')
        axis.bar(x + .2, [r['fall_rate'] for r in table], .2, label='Falls', color='#ba5145')
        axis.set_xticks(x, [r['name'] for r in table], rotation=65, ha='right', fontsize=7)
        axis.set_ylim(0, 1)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUT / 'teacher_free_outcomes.png', dpi=150)
        plt.close(fig)
    plot_curves(runs, ['pref_lambda', 'pref_violation', 'pref_violation_rate'], 'lambda_dynamics.png')
    plot_curves(runs, ['alpha', 'alpha_rows', 'policy_entropy_student', 'policy_entropy_teacher'], 'entropy.png')
    plot_curves(runs, ['teacher_fraction_cumulative', 'batch_teacher_fraction', 'pref_source_rows', 'pref_sample_reuse_mean'], 'intervention_replay.png')
    plot_curves([run for run in runs if run[1].get('actor_anti_weight', 0) or run[1].get('actor_pair_bc_weight', 0)],
                ['actor_pair_bc_loss', 'actor_anti_loss', 'actor_rejected_distance', 'actor_aux_rows'], 'anti_imitation.png')
    trajectories(evaluations)
    lines = ['# Unitree bounded hyperparameter study', '',
             'Partial results until all scheduled jobs finish. No missing arm is treated as a zero result.',
             'Navigation is 5 Hz, with four held 20 Hz ticks per decision; terminations end the hold early.',
             'Obstacle-only geometric teacher; no target-terrain sand or RGB claims. Training manifests are unfiltered by teacher success.',
             'All validation evaluations are deterministic and teacher-free; held-out results are separate.', '',
             '| Evaluation | N | Success | Zero-cost success | Falls | Costful | Matched |',
             '|---|---:|---:|---:|---:|---:|---|']
    for item in table:
        lines.append(f'| {item["name"]} | {item["episodes"]} | {item["success"]:.1%} | {item["safe_success"]:.1%} | {item["fall_rate"]:.1%} | {item["costful_rate"]:.1%} | {item["matched"]} |')
    lines += ['', '## Interpretation Constraints',
              '- A nonnegative hinge dual signal at zero tolerance makes lambda nondecreasing by construction; a plateau is empirical only if residuals and late slopes vanish.',
              '- Smaller margins, nonzero violation tolerance and signed dual updates test different constraint semantics; they are not numerical lambda caps.',
              '- Alpha measures latent tanh actor entropy, not the entropy of the noninvertible projected unit-ball action distribution.',
              '- Student-only alpha conditions on actual agent-controlled replay states; TD and actor updates still use the full TD batch.',
              '- Independent preference replay preserves the TD buffer and supplies 256 pairs with replacement; report unique rows and reuse for small buffers.',
              '- Anti-imitation is bounded kernel repulsion from rejected executed actions, not proof all other actions are safe. Warmup-only labels are excluded.',
              '- Most secondary arms have one seed; priority alpha comparisons and matched BC/anti comparisons target two seeds. Rank changes require replication.',
              '- Old HIL comparisons also differ in executed-action Q correction and decision aggregation; they cannot isolate a pure lambda effect.', '',
              '## Training Diagnostics',
              '| Run | Decisions | Lambda final | Alpha final | Takeover cumulative | Sampled teacher | Preference rows |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, args, rows in runs:
        if rows:
            r = rows[-1]
            lines.append(f'| {name} | {r["step"]} | {r.get("pref_lambda", float("nan")):.4g} | {r.get("alpha", float("nan")):.4g} | {r.get("teacher_fraction_cumulative", 0):.1%} | {r.get("batch_teacher_fraction", 0):.1%} | {r.get("pref_source_rows", 0):g} |')
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if any(not item['matched'] for item in checks.values()):
        raise RuntimeError('Evaluation cohort mismatch; do not interpret rankings')


if __name__ == '__main__':
    main()
