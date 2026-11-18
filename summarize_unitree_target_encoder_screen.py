"""Decision-facing summaries; keep incomplete cohorts and unlike benchmarks separate."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def aggregate(rows):
    output={}
    for stratum in ('all','unfiltered','sand_crossing'):
        selected=[r for r in rows if stratum=='all' or r['stratum']==stratum]
        output[stratum]=dict(n=len(selected),success=sum(r['success'] for r in selected),
            falls=sum(r['fall'] for r in selected),timeouts=sum(r['timeout'] for r in selected),
            abstentions=sum(r.get('planning_failed',False) for r in selected),
            sand_seconds=sum(r['sand_seconds'] for r in selected),
            successful_mean_seconds=float(np.mean([r['seconds'] for r in selected if r['success']])) if any(r['success'] for r in selected) else None)
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('artifacts/target_teacher_screen_20260913'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    training={}
    for name,path in [('history',args.root/'screen/trained/results.json'),('no_history',args.root/'screen_no_history/results.json')]:
        training[name]=json.loads(path.read_text())
    variants=list(dict.fromkeys(r['variant'] for r in training['history']))
    colors={name:plt.cm.tab10(i) for i,name in enumerate(variants)}
    split=json.loads((args.root/'screen/trained/split.json').read_text())
    previous=[]
    for path in split['validation']:
        with np.load(path) as data:
            previous.append((data['base'][:,-15:-12]-data['action'])**2)
    repeat_mse=float(np.concatenate(previous).mean())
    figure,axes=plt.subplots(1,2,figsize=(14,5))
    offline={}
    for ax,(condition,rows) in zip(axes,training.items()):
        offline[condition]={}
        for variant in variants:
            selected=[r for r in rows if r['variant']==variant]
            if not selected:
                continue
            curves=np.array([[c['validation_mse'] for c in r['curve']] for r in selected])
            epochs=np.arange(1,curves.shape[1]+1)
            ax.plot(epochs,curves.mean(0),label=variant,color=colors[variant])
            ax.fill_between(epochs,curves.min(0),curves.max(0),color=colors[variant],alpha=.12)
            offline[condition][variant]=dict(seeds=len(selected),validation_mse=curves[:,-1].tolist(),
                mean_validation_mse=float(curves[:,-1].mean()),encoder_max_changes=[r['encoder_max_change'] for r in selected],
                gradient_steps=[r['gradient_steps'] for r in selected],training_seconds=[r['seconds'] for r in selected])
        ax.axhline(repeat_mse if condition=='history' else split['validation_mean_action_mse'],
            color='black',ls='--',lw=1,label='repeat previous action' if condition=='history' else 'constant mean action')
        ax.set(title=condition.replace('_',' '),xlabel='Epoch',ylabel='Held-out episode action MSE',yscale='log')
        ax.legend(fontsize=7)
    figure.suptitle('Offline BC: two seeds, identical demonstration episodes; shaded range is seed range, not CI')
    figure.tight_layout();figure.savefig(args.output/'learning_curves.png',dpi=160);plt.close(figure)
    pilot={}
    for condition,folder in [('history',args.root/'screen/evaluations'),('no_history',args.root/'no_history_evaluations'),('online',args.root/'online_evaluations')]:
        if folder.exists():
            for path in sorted(folder.glob('*/results.json')):
                data=json.loads(path.read_text())
                rows=data['episodes']
                pilot[condition+'/'+path.parent.name]=dict(complete=len(rows)==8,horizon=data['args']['episode_length_s'],
                    seeds=[r['seed'] for r in rows],metrics=aggregate(rows),source=str(path))
    full_teacher=json.loads((args.root/'full24/results.json').read_text())['episodes']
    manual=json.loads((args.root.parent/'rgb_height_extended_20260913/episodes.json').read_text())
    full=dict(teacher=aggregate(full_teacher),manual_rgb_height=aggregate([r for r in manual if r['controller']=='policy']),
              direct_goal=aggregate([r for r in manual if r['controller']=='direct_goal']))
    online={}
    figure,axes=plt.subplots(1,3,figsize=(15,4))
    for mode in ('rgb','rgb_height'):
        folder=args.root/('online_'+mode)
        path=folder/'training/metrics.jsonl'
        if not path.exists():
            continue
        metrics=[]
        for line in path.read_text().splitlines():
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        online[mode]=dict(complete=(folder/'summary.json').exists(),source=str(folder),
            summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else None)
        for ax,key in zip(axes,('intervention_fraction','batch_teacher_fraction','critic_loss_total')):
            selected=[r for r in metrics if key in r and 'step' in r]
            ax.plot([r['step'] for r in selected],[r[key] for r in selected],label=mode)
            ax.set(xlabel='Navigation decisions',title=key.replace('_',' '))
    for ax in axes:
        if ax.lines:
            ax.legend()
    figure.suptitle('Automated preference + BC + RL: training diagnostics, not policy quality')
    figure.tight_layout();figure.savefig(args.output/'online_training.png',dpi=160);plt.close(figure)
    report=dict(full24_180_seconds=full,offline=offline,online_training=online,repeat_previous_action_mse=repeat_mse,pilot8_60_seconds=pilot,
        limitations=['Offline BC is not online preference/RL training.','Same arena, held-out goals, not unseen-terrain generalization.',
            'Standard-history pilots use seed 0; selected no-history comparisons also test seed 1. Check each cohort separately.',
            'Automated online runs use one training seed and episodic teacher-selected training goals, not the manual continuous-goal distribution.',
            'Fall cost is not a comprehensive physical collision metric.','Original ResNet training times include avoidable repeated frozen encoding.',
            'Do not reward a policy for low sand exposure when it fails before reaching the material.'])
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    lines=['# Teacher And Encoder Screening','','## Full Audit: 24 Goals, 180 Seconds','',
        '| Controller | Goals | Falls | Timeouts | Abstentions | Sand-crossing goals |',
        '|---|---:|---:|---:|---:|---:|']
    for name,groups in full.items():
        a,s=groups['all'],groups['sand_crossing']
        lines.append(f"| {name} | {a['success']}/{a['n']} | {a['falls']} | {a['timeouts']} | {a['abstentions']} | {s['success']}/{s['n']} |")
    lines+=['','## Encoder Pilot: 8 Goals, 60 Seconds','','Partial rows are explicitly marked; do not rank incomplete cohorts.','',
        '| Variant | Complete | Goals | Falls | Unfiltered goals | Sand goals | Sand seconds |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,row in pilot.items():
        a,u,s=[row['metrics'][k] for k in ('all','unfiltered','sand_crossing')]
        lines.append(f"| {name} | {row['complete']} | {a['success']}/{a['n']} | {a['falls']} | {u['success']}/{u['n']} | {s['success']}/{s['n']} | {s['sand_seconds']:.2f} |")
    lines+=['','## Decision Rule','','Promote only a teacher-free navigation improvement without increased falls; require a second-seed confirmation before treating a small pilot advantage as an encoder recommendation.',
        'A lower offline action error alone is not sufficient. Retain the existing manual RGB+height policy as the reference, not an untested BC replacement.','','## Limitations','']
    lines += ['- '+line for line in report['limitations']]
    lines += ['','## Recommendation For Manual Work','',
        '1. Keep the existing pretrained frozen RGB+height configuration as the reference. The original manual policy achieved 19/24 goals, including 13/16 sand-crossing goals. Do not replace it with these automated checkpoints.',
        '2. If spending time on one modality comparison, test frozen RGB-only against a fresh matched RGB+height run. RGB-only is a reasonable simpler challenger, not a demonstrated winner: the BC pilot was better, but the actual online objective tied at 3/8 and both failed every sand-crossing goal.',
        '3. Defer unfreezing, scratch encoders, ImageNet ResNet18, and height-only manual runs. None produced a reproducible closed-loop advantage. This does not rule out other pretrained encoders or better-tuned fine-tuning.',
        '4. Prioritize training coverage and recovery behavior before another encoder sweep. Include safe detours before entering sand, recovery from learner-induced turns, and varied approach headings. Keep evaluation goals held out; do not teach directly on the reported test seeds.',
        '5. Change one factor at a time: identical 4,500-decision budgets, objective, action history, low-level controller, arena, and training-goal schedule. Continuous-goal human runs do not guarantee paired layouts just by sharing an RNG seed; predeclare/replay a training schedule for a causal comparison.',
        '6. Evaluate final policies without interventions on the frozen 24-goal/180-second manifest. Compare success and falls first, then successful-route time and sand exposure. A small 8-goal gain only screens a candidate; require another training seed before promotion.',
        '', '## What The Metrics Show','',
        f'- Repeating the last action gives validation MSE {repeat_mse:.6f}; most history-enabled BC models are only modestly better. Low imitation loss is not evidence of terrain understanding.',
        '- Removing action history worsened offline MSE, while some closed-loop results improved. Fine-tuning without history improved seed 0 (5/8 versus 2/8), but seed 1 tied (3/8 each) and increased falls (3 versus 1). No robust fine-tuning recommendation follows.',
        '- Both native online runs completed 4,500 decisions and 3,997 updates with zero training falls. Cumulative takeovers were 2,858/4,500 for RGB and 2,863/4,500 for RGB+height. Declining takeovers and critic loss did not transfer to independent sand avoidance.',
        '- Student trajectories show persistent turning and sand stalls, not simply near-zero commands. The automated teacher succeeds with privileged global planning; its full route information is not available to the student.',
        '- The current RGB encoder is already task-pretrained and frozen: the small CNN is transferred from the 100k-step reconstruction/imitation donor, but the donor GRU and action head are not transferred.',
        '', '## Teacher And Data Boundaries','',
        '- Teacher: 22/24 successes, zero falls, two planning abstentions retained in the denominator; all 16 sand-crossing goals succeed. Sand exposure on that stratum totals 1.35 seconds.',
        '- The teacher uses conservative obstacle and moving-sweep exclusion plus material costs. It is not a globally optimal traversability oracle: routes 275/276 take about 105/107 seconds versus about 30/24 for the manual policy.',
        '- Dataset: 32 attempted training goals, 24 successful demonstration episodes, eight planning abstentions, 3,112 samples. The whole-episode split is 19 train / five validation episodes (2,653 / 459 samples).',
        '- Twenty-eight BC checkpoints cover seven variants, two seeds, and two history conditions. All seven history-enabled variants have a seed-0 closed-loop pilot; selected no-history variants have additional checks.',
        '- Sand seconds are root-centre material samples at 20 Hz, not all foot contacts. Cost counts falls, not all obstacle collisions. Lower sand exposure caused by an early fall or failure is not safer navigation.',
        '', '## Artifacts And Reproduction','',
        '- `learning_curves.png`: same-split offline curves with two-seed ranges, not confidence intervals.',
        '- `online_training.png`: cumulative intervention fraction, sampled-batch teacher fraction, and critic loss.',
        '- `results.json`: machine-readable results, source paths, and cohort completeness.',
        '- `../full24/trajectories.png`: actual teacher trajectories and fall markers.',
        '- `../online_evaluations/{rgb,rgb_height}/trajectories.png`: actual native student trajectories.',
        '- Native online checkpoints: `../online_rgb/training/final.pt` and `../online_rgb_height/training/final.pt`.',
        '- BC checkpoints: `../screen/trained/` and `../screen_no_history/`. These are diagnostic ScreenPolicy models, not drop-in native manual-training checkpoints.',
        '', 'Run from the canonical WSL checkout. The manual launcher requires the existing Windows gamepad sender/endpoint and starts paused:',
        '', '```bash', 'cd /home/benjamin/thesis',
        'NAVIGATION_OBSERVATION_MODE=rgb \\',
        'ONLINE_TOTAL_STEPS=4500 \\',
        'bash /home/benjamin/thesis/scripts/run_unitree_target_human_training_local.sh',
        '```', '', 'Use `NAVIGATION_OBSERVATION_MODE=rgb_height` for the reference. The launcher generates a unique run name; keep outputs separate. Neither command enables encoder fine-tuning.',
        '', 'Example native checkpoint evaluation (use a new output directory; omit `--seeds` for all 24 goals):',
        '', '```bash', 'bash /home/benjamin/thesis/scripts/run_unitree_target_teacher_benchmark_local.sh \\',
        '  --nav-checkpoint /home/benjamin/thesis/artifacts/target_teacher_screen_20260913/online_rgb/training/final.pt \\',
        '  --output /home/benjamin/thesis/artifacts/target_teacher_screen_20260913/online_rgb_full24_new \\',
        '  --max-seconds 180', '```']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(full=full,pilot=pilot),indent=2))


if __name__=='__main__':
    main()
