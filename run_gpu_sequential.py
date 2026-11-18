"""Run a finite JSON experiment list sequentially, without an MCP or daemon."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def process_identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--plan-only',action='store_true')
    options=parser.parse_args()
    manifest=options.manifest.resolve()
    spec=json.loads(manifest.read_text())
    root=Path(spec['cwd']).resolve()
    jobs=spec['jobs']
    assert root.is_dir() and jobs
    assert len({j['id'] for j in jobs})==len(jobs)
    for job in jobs:
        assert job['id'].replace('_','').replace('-','').isalnum()
        assert job['argv'] and Path(job['argv'][0]).is_absolute()
        assert Path(job['argv'][0]).is_file() and job.get('timeout_seconds',0)>0
    if options.plan_only:
        print(json.dumps(spec,indent=2));return
    directory=manifest.parent
    lock_path=Path(spec['lock']).resolve()
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=directory/'status.json'
        digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
        state=json.loads(state_path.read_text()) if state_path.exists() else dict(manifest_sha256=digest,jobs={})
        assert state['manifest_sha256']==digest,'Use a new manifest directory for changed experiments'
        if options.wait_pid:
            identity=process_identity(options.wait_pid)
            if identity:
                state.update(status='waiting_for_existing_job',wait_pid=options.wait_pid,runner_pid=os.getpid())
                atomic_json(state_path,state)
                while process_identity(options.wait_pid)==identity:
                    time.sleep(5)
        for job in jobs:
            if state['jobs'].get(job['id'],{}).get('status')=='completed':
                continue
            if job.get('gpu',True):
                probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                active=[line.strip() for line in probe.stdout.splitlines() if line.strip().isdigit()]
                if active:
                    state.update(status='blocked_external_gpu',gpu_pids=active)
                    atomic_json(state_path,state)
                    raise RuntimeError(f'GPU already owned by processes {active}; nothing was stopped')
            attempt=1+state['jobs'].get(job['id'],{}).get('attempt',0)
            log_path=directory/f"{job['id']}.attempt{attempt}.log"
            row=dict(status='running',attempt=attempt,started=time.time(),argv=job['argv'],log=str(log_path))
            state['jobs'][job['id']]=row
            state.update(status='running',runner_pid=os.getpid())
            atomic_json(state_path,state)
            env=os.environ.copy();env.update(spec.get('env',{}))
            with log_path.open('x') as log:
                process=subprocess.Popen(job['argv'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=process.pid;atomic_json(state_path,state)
                try:
                    code=process.wait(timeout=job['timeout_seconds'])
                except (subprocess.TimeoutExpired,KeyboardInterrupt):
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    row.update(status='interrupted',finished=time.time())
                    state['status']='stopped';atomic_json(state_path,state)
                    raise
            row.update(status='completed' if code==0 else 'failed',exit_code=code,finished=time.time())
            state['status']=row['status'];atomic_json(state_path,state)
            if code:
                raise RuntimeError(f"{job['id']} failed; later jobs were not started. See {log_path}")
        state['status']='completed';atomic_json(state_path,state)
        print(json.dumps(state,indent=2))


if __name__=='__main__':
    main()
