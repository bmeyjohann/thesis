"""Bounded, resumable offline encoder screening after immutable teacher collection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-goals',type=int,default=32)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--eval-seeds',type=int,nargs='+',default=[201,202,203,204,209,217,222,226])
    p.add_argument('--max-seconds',type=float,default=60)
    p.add_argument('--wall-budget-seconds',type=int,default=10800)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    print('[screen-pipeline] '+str(vars(args)),flush=True)
    assert args.expected_goals>=8 and args.epochs>0 and args.max_seconds>0
    assert (ROOT/'unitree_target_encoder_screen.py').is_file()
    assert (ROOT/'run_unitree_target_teacher_benchmark.py').is_file()
    if args.plan_only:
        return
    start=time.monotonic()
    def remaining():
        budget=args.wall_budget_seconds-(time.monotonic()-start)
        if budget<=0:
            raise TimeoutError('Declared screening wall-time budget exhausted; completed outputs retained')
        return budget
    while True:
        remaining()
        try:
            collection=json.loads((args.data/'results.json').read_text())
            count=len(collection['episodes'])
        except (FileNotFoundError,json.JSONDecodeError):
            count=0
        if count==args.expected_goals:
            break
        if count>args.expected_goals:
            raise ValueError('Collection differs from expected episode quota')
        print(f'[screen-pipeline] waiting for complete collection: {count}/{args.expected_goals}',flush=True)
        time.sleep(min(60,remaining()))
    args.output.mkdir(parents=True,exist_ok=True)
    def run(script,*arguments):
        command=[sys.executable,str(ROOT/script),*[str(v) for v in arguments]]
        print('[screen-pipeline] RUN '+json.dumps(command),flush=True)
        subprocess.run(command,cwd=ROOT,check=True,timeout=remaining())
    trained=args.output/'trained'
    finished=json.loads((trained/'results.json').read_text()) if (trained/'results.json').exists() else []
    if len(finished)<14:
        run('unitree_target_encoder_screen.py','--data',args.data,'--output',trained,'--epochs',args.epochs,
            *(['--resume'] if trained.exists() else []))
    training=json.loads((trained/'results.json').read_text())
    assert len(training)==14,'Training incomplete; do not silently evaluate a partial variant set'
    evals=args.output/'evaluations'
    evals.mkdir(exist_ok=True)
    selected=[r for r in training if r['seed']==0]
    jobs=[('teacher',[]),('direct_goal',['--direct-goal'])]+[(r['variant'],['--student',r['checkpoint']]) for r in selected]
    summaries=[]
    for name,extra in jobs:
        directory=evals/name
        path=directory/'results.json'
        complete=False
        if path.exists():
            complete=len(json.loads(path.read_text())['episodes'])==len(args.eval_seeds)
        if not complete:
            run('run_unitree_target_teacher_benchmark.py','--output',directory,'--max-seconds',args.max_seconds,
                '--seeds',*args.eval_seeds,*extra,*(['--resume'] if directory.exists() else []))
        episodes=json.loads(path.read_text())['episodes']
        row=dict(variant=name,episodes=len(episodes),success=sum(r['success'] for r in episodes),
            falls=sum(r['fall'] for r in episodes),timeouts=sum(r['timeout'] for r in episodes),
            planning_failed=sum(r['planning_failed'] for r in episodes),sand_seconds=sum(r['sand_seconds'] for r in episodes),
            success_seconds=[r['seconds'] for r in episodes if r['success']])
        summaries.append(row)
        (args.output/'pilot_summary.json').write_text(json.dumps(dict(
            warning='Small same-arena closed-loop pilot, seed 0 only. Offline BC is not online HIL/RL. Do not infer a definitive encoder ranking.',
            goal_seeds=args.eval_seeds,horizon=args.max_seconds,results=summaries),indent=2))
        print('[screen-pipeline] RESULT '+json.dumps(row),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(15,6))
    for row in training:
        axes[0].plot([c['epoch'] for c in row['curve']],[c['validation_mse'] for c in row['curve']],
            label=row['variant'] if row['seed']==0 else None,alpha=.8 if row['seed']==0 else .35)
    axes[0].set(xlabel='Epoch',ylabel='Held-out episode action MSE',title='Offline teacher imitation, two training seeds')
    axes[0].legend(fontsize=7)
    axes[1].barh([r['variant'] for r in summaries],[r['success']/r['episodes'] for r in summaries])
    axes[1].set(xlim=(0,1),xlabel='Success fraction',title=f'Small closed-loop pilot: {len(args.eval_seeds)} goals, seed 0')
    figure.tight_layout();figure.savefig(args.output/'screening.png',dpi=160)
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,results=summaries),indent=2))
    print('[screen-pipeline] COMPLETE; provisional evidence saved, not a definitive ranking',flush=True)


if __name__=='__main__':
    main()
