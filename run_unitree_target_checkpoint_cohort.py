"""Bounded matched teacher-free evaluation for explicitly selected checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[201,202,203,204,209,217,222,226])
    p.add_argument('--max-seconds',type=float,default=60)
    p.add_argument('--wall-budget-seconds',type=int,default=7200)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    assert all(path.is_file() for path in args.checkpoints)
    assert len({path.stem for path in args.checkpoints})==len(args.checkpoints)
    print('[checkpoint-cohort] '+str(vars(args)),flush=True)
    if args.plan_only:
        return
    started=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    for checkpoint in args.checkpoints:
        output=args.output/checkpoint.stem
        results=output/'results.json'
        if results.exists():
            saved=json.loads(results.read_text())
            completed={r['seed'] for r in saved['episodes']}
            assert completed.issubset(set(args.seeds))
            assert all(r['controller']==str(checkpoint) for r in saved['episodes'])
            assert saved['args']['episode_length_s']==args.max_seconds
            if completed==set(args.seeds):
                continue
        remaining=args.wall_budget_seconds-(time.monotonic()-started)
        if remaining<=0:
            raise TimeoutError('Cohort wall-time budget exhausted; completed artifacts retained')
        subprocess.run([sys.executable,str(root/'run_unitree_target_teacher_benchmark.py'),
            '--student',str(checkpoint),'--output',str(output),'--max-seconds',str(args.max_seconds),
            '--seeds',*[str(seed) for seed in args.seeds],*(['--resume'] if output.exists() else [])],cwd=root,check=True,timeout=remaining)
    print('[checkpoint-cohort] COMPLETE',flush=True)


if __name__=='__main__':
    main()
