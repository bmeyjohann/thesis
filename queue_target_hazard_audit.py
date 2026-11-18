"""Submit the bounded gate audit through the existing queue engine."""
import json
from pathlib import Path
from tools.experiment_queue_mcp.server import build_arg_parser,build_queue,build_daemon_command

root=Path('/home/benjamin/thesis')
control=json.loads((root/'experiment_queue/state/control.json').read_text())
args=build_arg_parser().parse_args(['--queue-root',str(root/'experiment_queue'),
    '--workspace-root',str(root),'--default-cwd',str(root),'--script-root',str(root/'scripts'),
    '--default-conda-env','fasttd3','--conda-sh-path','/home/benjamin/miniconda3/etc/profile.d/conda.sh',
    '--max-concurrent-jobs',str(control.get('max_concurrent_jobs',4))])
queue=build_queue(args)
print(queue.queue_status())
job=queue.enqueue_script(source_path=str(root/'scripts/generated/run_target_hazard_audit.sh'),
    name='target-hazard-progress-audit-v2',cwd=str(root),conda_env='fasttd3',
    owner_session_id='01a0253e-2795-7193-9ccd-554bbd45b104',owner_label='target-hazard-progress')
print(job)
print(queue.ensure_daemon_running(build_daemon_command(args)))
