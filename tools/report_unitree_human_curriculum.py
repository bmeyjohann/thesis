"""CPU-only report of immutable human runs and all matched evaluation cases."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABELS = {'r1_rgb_height_pref_rl': 'RGB+height / pref+RL',
          'r1_php_depth_pref_rl': 'PHP depth / pref+RL',
          'r1_rgb_height_pref_bc_rl': 'RGB+height / pref+BC+RL',
          'direct_goal': 'Direct-goal controller (not RL)'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def training(curriculum, output):
    summaries, episodes, hashes = [], [], []
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    support_fig, support_axes = plt.subplots(1, 3, figsize=(15,4), constrained_layout=True)
    colors = ['#007c91', '#d48200', '#9c3456']
    for index, run in enumerate(r for r in curriculum['Runs'] if r['Status'] == 'complete'):
        name = run['Id']
        model = ROOT/'models/unitree_mjlab_nav_human'/run['RunName']
        dataset = ROOT/'local/unitree_human_interventions'/run['RunName']
        manifest = json.loads((dataset/'manifest.json').read_text())
        keys = ['intervened','intervention_start','episode_index','done','terminal_success',
                'cost','wall_time_unix_s','goal_distance','next_goal_distance','gamepad_stale']
        columns = {k: [] for k in keys}
        for item in manifest['parts']:
            path = dataset/item['file']
            hashes.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            with np.load(path) as part:
                for key in keys:
                    columns[key].append(part[key].reshape(-1))
        data = {k: np.concatenate(v) for k,v in columns.items()}
        summary = json.loads((model/'browser_summary.json').read_text())
        episode_rows = []
        for eid in np.unique(data['episode_index']):
            mask = data['episode_index'] == eid
            end = bool(data['done'][mask][-1])
            row = dict(run=name, episode=int(eid), rows=int(mask.sum()), completed_boundary=end,
                       success=bool(data['terminal_success'][mask].any()),
                       native_cost=float(data['cost'][mask].sum()),
                       intervention_fraction=float(data['intervened'][mask].mean()),
                       initial_distance=float(data['goal_distance'][mask][0]),
                       last_recorded_distance=float(data['next_goal_distance'][mask][-1]))
            episode_rows.append(row)
        episodes.extend(episode_rows)
        complete = [r for r in episode_rows if r['completed_boundary']]
        human = int(data['intervened'].sum())
        row = dict(run=name, decisions=len(data['cost']), human_decisions=human,
                   human_fraction=human/len(data['cost']), intervention_bursts=int(data['intervention_start'].sum()),
                   goals=summary['goals'], native_fall_cost=summary['cost'],
                   environment_ticks=summary['environment_ticks'], simulation_seconds=summary['environment_ticks']*.05,
                   elapsed_recorded_wall_seconds=float(data['wall_time_unix_s'][-1]-data['wall_time_unix_s'][0]),
                   logged_segments=len(episode_rows), completed_segments=len(complete),
                   censored_segments=len(episode_rows)-len(complete),
                   success_fraction_completed_segments=sum(r['success'] for r in complete)/len(complete) if complete else None,
                   transport_stale_fraction=float(data['gamepad_stale'].mean()), checkpoint=str(model/'final.pt'))
        assert row['decisions'] == 4500 and human == summary['intervention_decisions']
        summaries.append(row)
        metric = [json.loads(line) for line in (model/'metrics.jsonl').read_text().splitlines()]
        x = np.array([r['step'] for r in metric])
        def series(key):
            return np.array([r.get(key, np.nan) for r in metric], dtype=float)
        color = colors[index]
        label = LABELS[name]
        axes[0,0].plot(x, series('alpha'), color=color, label=label)
        axes[0,1].plot(np.arange(len(data['cost']))+1, np.cumsum(data['intervened'])/(np.arange(len(data['cost']))+1), color=color, label=label)
        axes[1,0].plot(x, series('batch_teacher_fraction'), color=color, label=label, alpha=.8)
        axes[1,1].plot(x, series('pref_source_rows'), color=color, label=label)
        axes[2,0].plot(x, series('critic_loss_replay'), color=color, label=label)
        axes[2,1].plot(x, series('actor_loss_bc'), color=color, label=label)
        for ax,key in zip(support_axes,['pref_sample_unique_rows','pref_sample_age_ticks_mean','pref_sample_reuse_mean']):
            ax.plot(x,series(key),color=color,label=label)
        write_json(output/f'{name}_training_metrics.json', metric)
    titles = ['Learned alpha', 'Cumulative human intervention fraction', 'Sampled TD teacher fraction',
              'Preference source rows per logged update', 'TD critic loss', 'Actor BC loss (zero when disabled)']
    for ax,title in zip(axes.flat,titles):
        ax.set(title=title,xlabel='Navigation decisions')
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle('Human-assisted training: not teacher-free policy performance')
    fig.savefig(output/'training_curves.png', dpi=150)
    plt.close(fig)
    for ax,title in zip(support_axes,['Unique preference pairs per sample','Mean preference age (decision ticks)','Mean preference-pair sample reuse']):
        ax.set(title=title,xlabel='Navigation decisions'); ax.grid(alpha=.25); ax.legend(fontsize=6)
    support_fig.savefig(output/'preference_support.png',dpi=150);plt.close(support_fig)
    write_csv(output/'training_summary.csv', summaries)
    write_csv(output/'training_segments.csv', episodes)
    write_json(output/'training_summary.json', summaries)
    write_json(output/'training_source_hashes.json', hashes)
    return summaries


def summarize(rows, name, stratum='all'):
    n = len(rows)
    success_times = [r['seconds'] for r in rows if r['success']]
    contacts = [r['obstacle_contact_seconds'] for r in rows]
    return dict(run=name, stratum=stratum, n=n, successes=sum(r['success'] for r in rows),
                safe_successes=sum(r['safe_success'] for r in rows),
                material_safe_successes=sum(r['material_safe_success'] for r in rows),
                falls=sum(r['fall'] for r in rows), timeouts=sum(r['timeout'] for r in rows),
                other_terminations=sum(r['other_termination'] for r in rows),
                collision_episodes=sum(r['obstacle_contact_substeps']>0 for r in rows),
                collision_seconds_total=sum(contacts), collision_seconds_p95=float(np.percentile(contacts,95)),
                collision_seconds_max=max(contacts), max_penetration_m=max(r['max_penetration_m'] for r in rows),
                native_cost_total=sum(r['native_cost'] for r in rows),
                success_time_median_s=float(np.median(success_times)) if success_times else None,
                success_time_mean_s=float(np.mean(success_times)) if success_times else None,
                episode_time_mean_s=float(np.mean([r['seconds'] for r in rows])),
                zero_cost_failures=sum(not r['success'] and r['native_cost']==0 for r in rows),
                collision_free_failures=sum(not r['success'] and not r['fall'] and not r['obstacle_contact_substeps'] for r in rows),
                mean_progress_m=float(np.mean([r['progress_m'] for r in rows])),
                mean_final_distance_m=float(np.mean([r['final_distance'] for r in rows])),
                mean_action_norm=float(np.mean([np.linalg.norm(r['commands'],axis=1).mean() for r in rows])),
                ice_seconds_total=sum(r['ice_seconds'] for r in rows), sand_seconds_total=sum(r['sand_seconds'] for r in rows),
                evaluation_wall_seconds=sum(r['wall_seconds'] for r in rows))


def trajectory_pages(name, rows, layout, output, prefix=''):
    from eval_unitree_target_terrain_policy import _draw_layout
    from matplotlib.patches import Rectangle
    pages = []
    for page in range(0,len(rows),8):
        fig,axes = plt.subplots(2,4,figsize=(18,10),constrained_layout=True)
        for ax,row in zip(axes.flat,rows[page:page+8]):
            _draw_layout(ax,layout)
            half=layout.arena_size/2
            ax.add_patch(Rectangle((-half,-half),layout.arena_size,layout.arena_size,
                                   fill=False,edgecolor='black',linestyle='--',linewidth=1.2,zorder=7))
            path=np.array(row['trajectory'])
            assert np.linalg.norm(path[0]-row['start']) < 1e-5
            ax.plot(path[:,0],path[:,1],color='#101d33',lw=1.5,zorder=8)
            ax.scatter(*row['start'],c='white',edgecolor='black',s=40,zorder=10)
            ax.scatter(*row['goal'],c='#00d938',edgecolor='black',marker='*',s=100,zorder=10)
            if row['contact_points']:
                xy=np.array([v['xy'] for v in row['contact_points']])
                ax.scatter(xy[:,0],xy[:,1],c='crimson',s=13,zorder=11)
            ax.scatter(*path[-1],marker='X' if row['fall'] else 's',c='black',s=40,zorder=12)
            allxy=np.vstack([path,row['goal']]); lo=allxy.min(0)-1; hi=allxy.max(0)+1
            span=max((hi-lo).max(),5); center=(lo+hi)/2
            ax.set_xlim(center[0]-span/2,center[0]+span/2); ax.set_ylim(center[1]-span/2,center[1]+span/2)
            outcome='SAFE SUCCESS' if row['safe_success'] else 'SUCCESS' if row['success'] else 'FALL' if row['fall'] else 'TIMEOUT' if row['timeout'] else 'TERMINATED'
            ax.set_title(f"{row['seed']} | {row['stratum']}\n{outcome} | {row['seconds']:.1f}s | contact {row['obstacle_contact_seconds']:.2f}s",fontsize=8)
        for ax in axes.flat[len(rows[page:page+8]):]:
            ax.axis('off')
        scope='Additional completed cases, NOT a matched comparison' if prefix else 'All matched cases, no curation'
        fig.suptitle(LABELS[name]+' | '+scope+'\nWhite=start; green star=goal; crimson=contact; X=fall; square=end; dashed=arena edge. Magenta objects at nominal position.')
        filename=f'{prefix}{name}_trajectories_{page//8+1:02d}.png'
        fig.savefig(output/filename,dpi=140); plt.close(fig)
        pages.append(filename)
    return pages


def supplemental_pages(root, selected_root, output):
    from unitree_target_terrain import generate_target_terrain
    selected={c['seed'] for c in json.loads((selected_root/'matched_manifest.json').read_text())['cases']}
    source_manifest=root/'matched_manifest.json'
    manifest=json.loads(source_manifest.read_text())
    cases={c['seed']:c for c in manifest['cases']}
    pages, episodes, provenance=[], [], []
    for path in sorted(root.glob('*/results.json')):
        saved=json.loads(path.read_text())
        assert saved['manifest_sha256']==hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        assert saved['layout']==manifest['layout']
        rows=[r for r in saved['episodes'] if r['seed'] not in selected]
        for row in rows:
            case=cases[row['seed']]
            for key,target in [('start','start'),('goal','goal'),('state','reset_state')]:
                np.testing.assert_allclose(case[key],row[target],rtol=0,atol=1e-5)
            assert row['decisions']==math_ceil_quarter(row['ticks'])
        layout=generate_target_terrain(saved['layout']['seed'],preset=saved['layout']['preset'],arena_size=saved['layout']['arena_size'])
        pages+=trajectory_pages(saved['name'],rows,layout,output,'supplemental_')
        episodes.extend(dict(run=saved['name'],**r) for r in rows)
        provenance.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),additional_cases=len(rows)))
    write_json(output/'supplemental_episodes.json',episodes)
    write_json(output/'supplemental_sources.json',provenance)
    return pages,len(episodes)


def evaluations(root, output):
    from unitree_target_terrain import generate_target_terrain
    manifest = json.loads((root/'matched_manifest.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    names = list(plan['checkpoints'])+['direct_goal']
    collected, summary, flat, pages = {}, [], [], []
    for name in names:
        saved = json.loads((root/name/'results.json').read_text())
        rows = saved['episodes']
        assert len(rows) == plan['count']
        assert saved['layout'] == manifest['layout']
        assert saved['manifest_sha256'] == hashlib.sha256((root/'matched_manifest.json').read_bytes()).hexdigest()
        for case,row in zip(manifest['cases'],rows):
            assert case['seed'] == row['seed']
            for key, target in [('start','start'),('goal','goal'),('state','reset_state')]:
                np.testing.assert_allclose(case[key],row[target],rtol=0,atol=1e-5)
            assert row['decisions'] == math_ceil_quarter(row['ticks'])
        collected[name] = rows
        summary.append(summarize(rows, name))
        for stratum in ['unfiltered','blocked_detour']:
            summary.append(summarize([r for r in rows if r['stratum']==stratum],name,stratum))
        layout = generate_target_terrain(saved['layout']['seed'], preset=saved['layout']['preset'],arena_size=saved['layout']['arena_size'])
        pages+=trajectory_pages(name,rows,layout,output)
        for row in rows:
            flat.append(dict(run=name,**{k:v for k,v in row.items() if k not in ['trajectory','contact_points','cost_points','materials','commands','reset_state','start','goal']}))
    write_csv(output/'evaluation_summary.csv',summary)
    write_csv(output/'evaluation_episodes.csv',flat)
    write_json(output/'evaluation_summary.json',summary)
    paired=[]
    for i,case in enumerate(manifest['cases']):
        paired.append(dict(seed=case['seed'],stratum=case['stratum'],**{
            name+suffix:collected[name][i][key] for name in names for suffix,key in
            [('_success','success'),('_safe','safe_success'),('_fall','fall'),('_seconds','seconds')]}))
    write_csv(output/'paired_cases.csv',paired)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for i,name in enumerate(names):
        row=next(r for r in summary if r['run']==name and r['stratum']=='all')
        axes[0].bar(i-.16,row['successes']/row['n'],width=.3,color='#5298b9')
        axes[0].bar(i+.16,row['safe_successes']/row['n'],width=.3,color='#12634c')
        axes[1].bar(i,row['falls']/row['n'],color='#b94a48')
        axes[2].bar(i,row['collision_episodes']/row['n'],color='#b98232')
    for ax,title in zip(axes,['Total (blue) / safe (green) success','Falls','Obstacle-contact episodes']):
        ax.set(title=title,ylim=(0,1),xticks=range(4),xticklabels=['RGB pref','PHP pref','RGB pref+BC','Direct goal'])
        ax.tick_params(axis='x',rotation=20); ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'evaluation_overview.png',dpi=160);plt.close(fig)
    return summary,pages


def math_ceil_quarter(value):
    return (value+3)//4


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--curriculum',type=Path,required=True)
    p.add_argument('--evaluation',type=Path)
    p.add_argument('--supplemental-evaluation',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    curriculum=json.loads(args.curriculum.read_text(encoding='utf-8-sig'))
    train=training(curriculum,args.output)
    summary,pages=evaluations(args.evaluation,args.output) if args.evaluation else ([],[])
    extra_pages,extra_count=supplemental_pages(args.supplemental_evaluation,args.evaluation,args.output) if args.supplemental_evaluation else ([],0)
    text=['# Human curriculum evaluation','',
          'Three independently human-trained policies, one training seed each. Different human intervention exposure confounds pure BC/sensor comparisons. The fourth PHP+BC arm was stopped and is not included.',
          '', '## Human-assisted training','',
          '| Run | Decisions | Human decisions | Human % | Goals | Native fall cost | Sim seconds | Recorded wall minutes |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in train:
        text.append(f"| {LABELS[r['run']]} | {r['decisions']} | {r['human_decisions']} | {100*r['human_fraction']:.1f} | {r['goals']} | {r['native_fall_cost']:g} | {r['simulation_seconds']:.1f} | {r['elapsed_recorded_wall_seconds']/60:.1f} |")
    text+=['','Wall duration includes pauses between first/last recorded transitions. Final open/reset-abandoned segments are censored, not counted as failures. Native training cost is fall-only; obstacle contacts were not recorded. Training trajectories cannot be faithfully recovered because these datasets do not contain world poses. See full teacher-free evaluation trajectories instead.',
           '', '![Training curves](training_curves.png)', '', '![Preference support](preference_support.png)']
    if summary:
        text+=['','## Matched policy-only evaluation','',
               'No human gate or learning. Same arena geometry, low-level policy, initial robot states, goals, orientations and moving-obstacle phases; 5 Hz decisions, four 20 Hz ticks per decision, 180 s maximum per episode. Half the cases are unfiltered reset draws; half have verified static centerline blockage and a planner-feasible detour. The latter is a challenge stratum, not an unbiased deployment sample. Same training arena, new reset seeds; not held-out-terrain generalization.',
               '', '**Safe success:** goal reached without a fall or any penetrating/touching robot-obstacle contact sampled at every physics substep. Ice/sand avoidance is a separate stricter material-safe metric. Contacts do not terminate episodes and are not called falls. Native cost remains fall-only; collision duration and maximum penetration are separate diagnostics, not a weighted cost or safety certificate.',
               '', '| Run | Total success | Safe success | Material-safe success | Falls | Timeout | Contact episodes | Median successful time |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in summary:
            if r['stratum']!='all': continue
            t='n/a' if r['success_time_median_s'] is None else f"{r['success_time_median_s']:.1f}s"
            text.append(f"| {LABELS[r['run']]} | {r['successes']}/{r['n']} | {r['safe_successes']}/{r['n']} | {r['material_safe_successes']}/{r['n']} | {r['falls']} | {r['timeouts']} | {r['collision_episodes']} | {t} |")
        text+=['','### Ordinary versus blocked cases','',
               '| Run | Stratum | Total success | Safe success | Falls | Contact episodes |',
               '|---|---|---:|---:|---:|---:|']
        for r in summary:
            if r['stratum']=='all': continue
            text.append(f"| {LABELS[r['run']]} | {r['stratum']} | {r['successes']}/{r['n']} | {r['safe_successes']}/{r['n']} | {r['falls']}/{r['n']} | {r['collision_episodes']}/{r['n']} |")
        text+=['','### Contact severity and duration','',
               '| Run | Total contact seconds | P95 episode contact seconds | Worst episode contact seconds | Max penetration mm | Mean episode seconds | Zero native-cost failures |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for r in summary:
            if r['stratum']!='all': continue
            text.append(f"| {LABELS[r['run']]} | {r['collision_seconds_total']:.3f} | {r['collision_seconds_p95']:.3f} | {r['collision_seconds_max']:.3f} | {1000*r['max_penetration_m']:.1f} | {r['episode_time_mean_s']:.2f} | {r['zero_cost_failures']} |")
        text+=['','Contact duration counts physics steps with one or more robot-obstacle contacts, not the number of contact points. Penetration is a geometric proxy, not impact force. P95 is a descriptive interpolated sample quantile; with eight cases it is not a reliable population tail estimate. Early falls shorten exposure, so a lower accumulated contact duration is not sufficient evidence of a safer policy.',
               '', '### Interpretation boundaries','',
               'These are final saved policies, not performance during assisted training. There is one independently trained seed per configuration and unequal human supervision: PHP received far fewer intervention decisions, and its preference replay increasingly reused older examples. The comparison therefore does not isolate a causal BC or sensor effect. The geometric planner only screened blocked-case feasibility and never controlled an evaluated policy. No additional RL-only trained baseline was produced; direct-goal is explicitly a hand-coded reference.']
        text+=['','Successful-time medians exclude failed episodes; mean episode duration, final/minimum goal distance, progress, zero-cost failures, cost/contact tails, ice/sand exposure and paired-case outcomes are in the CSV/JSON tables. Material exposure uses the robot base center, not a complete footprint. Direct-goal is a scan-blind hand-coded reference, not an RL-only trained arm.',
               '', '![Evaluation overview](evaluation_overview.png)','', '## Interactive policy inspection', '',
               'Run one command at a time in Ubuntu WSL, then open http://localhost:8080 and press Pause / Run. These commands load the saved policy and perception settings without training or a gamepad gate. They inspect new interactive episodes, not an exact replay of the recorded matched manifest. The shared GPU lock prevents overlap with another study job.']
        for r in train:
            command = [
                'cd /home/benjamin/thesis',
                'flock -n /tmp/unitree_core_gpu0.lock env MUJOCO_GL=egl /home/benjamin/miniconda3/envs/fasttd3/bin/python /home/benjamin/thesis/eval_interactive_unitree_nav.py',
                f"  --model-path '{r['checkpoint']}'",
                '  --checkpoint-env-config --viewer-backend viser --controller policy',
                '  --human-input-device none --observation-only --autopilot',
                '  --start-paused --sim-fps 20 --auto-reset --seed 7']
            text += ['', '### '+LABELS[r['run']], '', '```bash', command[0],
                     (' '+chr(92)+'\n').join(command[1:]), '```']
        text+=['','## Entire trajectory gallery']
        text += [f'\n![{name}]({name})' for name in pages]
        if extra_pages:
            text+=['','## Additional completed cases','',f'{extra_count} additional episodes completed before switching to the requested eight-case comparison. Every additional trajectory is included below; these unequal per-policy samples are NOT pooled into the matched comparison. Selection for the main comparison was the first four cases per stratum in original manifest order, not outcome-based.']
            text += [f'\n![{name}]({name})' for name in extra_pages]
    else:
        text+=['','Policy-only evaluation is pending; the training counts above are not final policy-quality results.']
    (args.output/'REPORT.md').write_text('\n'.join(text)+'\n')
    body='<h1>Human curriculum results</h1><pre>'+html.escape('\n'.join(text[:text.index('![Training curves](training_curves.png)')]))+'</pre>'
    if summary:
        body+='<h2>Policy-only evaluation</h2><p>Safe success = goal reached without a fall or physical obstacle contact. Material-safe additionally excludes base-center ice/sand exposure. Same-arena, one-training-seed evidence; see the report for boundaries.</p><table><tr><th>Configuration</th><th>Total success</th><th>Safe success</th><th>Material-safe</th><th>Falls</th><th>Timeouts</th><th>Contact episodes</th><th>Median success time</th></tr>'
        for r in summary:
            if r['stratum']!='all': continue
            cells=[LABELS[r['run']],f"{r['successes']}/{r['n']}",f"{r['safe_successes']}/{r['n']}",f"{r['material_safe_successes']}/{r['n']}",r['falls'],r['timeouts'],r['collision_episodes'],r['success_time_median_s']]
            body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'</tr>'
        body+='</table>'
        body+='<img src="evaluation_overview.png"><p>All trajectory cases are shown below. See REPORT.md and CSV/JSON files for definitions and full metrics.</p>'
    body+='<img src="training_curves.png"><img src="preference_support.png">'+''.join(f'<h2>{html.escape(n)}</h2><a href="{n}"><img loading="lazy" src="{n}"></a>' for n in pages)
    if extra_pages:
        body+=f'<h2>All {extra_count} additional completed cases</h2><p>Not matched across all controllers; excluded from the comparison table. No outcome curation.</p>'
        body+=''.join(f'<h3>{html.escape(n)}</h3><a href="{n}"><img loading="lazy" src="{n}"></a>' for n in extra_pages)
    body+='<p><a href="REPORT.md">Full report</a> | <a href="evaluation_summary.csv">Summary CSV</a> | <a href="evaluation_episodes.csv">Every episode</a> | <a href="paired_cases.csv">Paired cases</a></p>'
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Human curriculum evaluation</title><style>body{margin:32px;background:#f4f1e8;color:#172331;font:16px Georgia}img{max-width:100%;height:auto}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #aaa;text-align:left}</style>'+body)
    print(json.dumps(dict(training=train,evaluation=summary)),flush=True)


if __name__=='__main__':
    main()
