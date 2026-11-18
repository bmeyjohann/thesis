import json
from pathlib import Path
import subprocess
import sys


def test_sequence_resume_and_failure(tmp_path):
    runner=Path(__file__).resolve().parents[1]/'run_gpu_sequential.py'
    marker=tmp_path/'marker'
    def job(name,code):
        return dict(id=name,argv=[sys.executable,'-c',code],gpu=False,timeout_seconds=10)
    jobs=[job('first',f'from pathlib import Path; Path({str(marker)!r}).write_text("first")'),
          job('second',f'from pathlib import Path; p=Path({str(marker)!r}); assert p.read_text()=="first"; p.write_text("second")')]
    manifest=tmp_path/'jobs.json'
    manifest.write_text(json.dumps(dict(cwd=str(tmp_path),lock=str(tmp_path/'gpu.lock'),jobs=jobs)))
    for _ in range(2):
        subprocess.run([sys.executable,str(runner),str(manifest)],check=True,capture_output=True)
    state=json.loads((tmp_path/'status.json').read_text())
    assert state['status']=='completed' and marker.read_text()=='second'
    assert all(j['attempt']==1 for j in state['jobs'].values())
    other=tmp_path/'failure';other.mkdir()
    manifest=other/'jobs.json'
    manifest.write_text(json.dumps(dict(cwd=str(tmp_path),lock=str(tmp_path/'gpu.lock'),jobs=[
        job('fail','raise SystemExit(7)'),job('must_not_run','raise SystemExit(9)')])))
    result=subprocess.run([sys.executable,str(runner),str(manifest)],capture_output=True)
    state=json.loads((other/'status.json').read_text())
    assert result.returncode!=0 and state['jobs']['fail']['exit_code']==7
    assert 'must_not_run' not in state['jobs']
