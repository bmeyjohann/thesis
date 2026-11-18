"""Summarize matched native online modality pilots separately from assisted metrics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('artifacts/target_modality_gate_20260914')
output=root/'comparison';output.mkdir(exist_ok=True)
results={}
figure,axes=plt.subplots(1,3,figsize=(15,4))
for mode in ('height_scan','rgb','rgb_height','depth_height'):
    summary=root/mode/'summary.json'
    evaluation=root/'evaluations'/mode/'results.json'
    metrics=root/mode/'training/metrics.jsonl'
    if metrics.exists():
        rows=[json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        for ax,key in zip(axes,('intervention_fraction','batch_teacher_fraction','critic_loss_total')):
            data=[r for r in rows if key in r]
            ax.plot([r['step'] for r in data],[r[key] for r in data],label=mode)
            ax.set(xlabel='Navigation decisions',title=key.replace('_',' '))
    if not(summary.exists() and evaluation.exists()):
        continue
    train=json.loads(summary.read_text())
    episodes=json.loads((root/mode/'episodes.json').read_text())
    train['visited_training_goals']=len({r['seed'] for r in episodes})
    train['episodes_with_planning_failure']=sum(r.get('gate',{}).get('planning_failures',0)>0 for r in episodes)
    rows=json.loads(evaluation.read_text())['episodes']
    groups={}
    for stratum in ('all','unfiltered','sand_crossing'):
        selected=[r for r in rows if stratum=='all' or r['stratum']==stratum]
        groups[stratum]=dict(n=len(selected),success=sum(r['success'] for r in selected),
            falls=sum(r['fall'] for r in selected),timeouts=sum(r['timeout'] for r in selected),
            sand_seconds=sum(r['sand_seconds'] for r in selected),
            success_mean_seconds=float(np.mean([r['seconds'] for r in selected if r['success']])) if any(r['success'] for r in selected) else None)
    results[mode]=dict(training=train,evaluation=groups,complete=len(rows)==24 and train['steps']==4500)
for ax in axes:
    if ax.lines:ax.legend(fontsize=8)
figure.suptitle('Training diagnostics only: fixed hazard/progress gate, frozen encoders')
figure.tight_layout();figure.savefig(output/'training_curves.png',dpi=160);plt.close(figure)
(output/'results.json').write_text(json.dumps(results,indent=2))
if results:
    names=list(results)
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key,total in ((axes[0],'all',24),(axes[1],'sand_crossing',16)):
        x=np.arange(len(names))
        ax.bar(x-.18,[results[n]['evaluation'][key]['success'] for n in names],.36,label='Goals reached',color='#167cad')
        ax.bar(x+.18,[results[n]['evaluation'][key]['falls'] for n in names],.36,label='Falls',color='crimson')
        ax.set(xticks=x,xticklabels=names,ylim=(0,total),title='All 24 goals' if key=='all' else '16 sand-crossing goals')
        ax.tick_params(axis='x',rotation=15);ax.legend()
    figure.suptitle('Teacher-free evaluation: one training seed, fixed gate during training')
    figure.tight_layout();figure.savefig(output/'policy_comparison.png',dpi=160);plt.close(figure)
lines=['# Fixed-Gate Observation Modality Sweep','','One training seed; 4,500 decisions; 24 held-out goals, 180 seconds each. Camera encoders remain pretrained and frozen.','',
    '| Observation | Complete | Goals | Falls | Sand goals | Sand seconds | Training takeovers |',
    '|---|---|---:|---:|---:|---:|---:|']
for mode,result in results.items():
    a=result['evaluation']['all'];s=result['evaluation']['sand_crossing'];t=result['training']
    lines.append(f"| {mode} | {result['complete']} | {a['success']}/{a['n']} | {a['falls']} | {s['success']}/{s['n']} | {s['sand_seconds']:.2f} | {t['interventions']/t['steps']:.1%} |")
lines+=['','## Interpretation','',
    '- Rank teacher-free success and falls first, then sand-route completion and exposure. Low exposure from early failures is not avoidance.',
    '- Same arena and held-out goals, not unseen-terrain generalization. Shared training bank/order is not an identical visited-state distribution.',
    '- These are actual preference + BC + RL runs. Training takeovers are scripted, not human effort estimates.',
    '- Falls are not a comprehensive collision metric. Sand exposure samples the root centre, not all foot contacts.',
    '- Compare to the frozen-manifest direct-goal baseline: 14/24 goals, six falls, 8/16 sand goals (180-second horizon).',
    '- Checkpoints: ../<modality>/training/final.pt. Actual trajectories: ../evaluations/<modality>/trajectories.png.',
    '- This sweep does not compare encoder pretraining or fine-tuning. Any modality recommendation is provisional until replicated.']
lines+=['','## Training Coverage','',
    '| Observation | Unique training goals visited | Episodes with planner failure | Updates |',
    '|---|---:|---:|---:|']
for mode,result in results.items():
    t=result['training']
    lines.append(f"| {mode} | {t['visited_training_goals']} | {t['episodes_with_planning_failure']} | {t['updates']} |")
lines+=['','The teacher/gate was intentionally held fixed. In the current training path a failed plan can produce zero-command takeovers until timeout; these runs are a pragmatic screening comparison, not a validated optimal teaching setup. Evaluation has no teacher and no gate.']
if len(results)==4 and all(r['complete'] for r in results.values()):
    lines+=['','## Manual Testing Order','',
        '1. RGB-only with the existing frozen encoder: first exploratory manual run. It tied height scan on total successes with fewer falls, but did not improve sand-goal completion over height scan.',
        '2. Retain your existing manual RGB+height policy as the practical reference. Its earlier 19/24 result is not a matched modality experiment, but it remains stronger than these automated policies.',
        '3. Height scan is a useful inexpensive control, especially for separating geometry navigation from visual material cues. Its 9/16 sand-goal successes do not establish sand recognition or avoidance.',
        '4. Defer depth+height manual training under this configuration. The observed 0/24 is a failed pilot, not proof that depth is intrinsically unusable.',
        '', 'No modality clearly beats the direct-goal baseline on both success and falls. These results prioritize experiments rather than establish a definitive winner.',
        '', '## Verified Scope','',
        'Four completed native preference + BC + RL runs, 18,000 total navigation decisions, 15,988 gradient updates, and 96 teacher-free evaluation episodes. No gate tuning, encoder fine-tuning, or new pretraining was included.',
        'Training plots show cumulative takeovers, sampled-batch teacher fraction, and critic loss. They are not intermediate teacher-free policy learning curves.',
        '', 'For a fresh manual RGB run, use the existing WSL launcher with the Windows gamepad sender connected:',
        '', '```bash', 'NAVIGATION_OBSERVATION_MODE=rgb \\', 'ONLINE_TOTAL_STEPS=4500 \\',
        'bash /home/benjamin/thesis/scripts/run_unitree_target_human_training_local.sh', '```']
(output/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
