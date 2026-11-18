"""Create an immutable, sequential four-modality experiment manifest."""
import json
from pathlib import Path

root=Path('/home/benjamin/thesis')
output=root/'artifacts/target_modality_gate_20260914'
queue=root/'local/sequential_gpu/modality_gate_20260914'
python='/home/benjamin/miniconda3/envs/fasttd3/bin/python'
assert not output.exists() and not queue.exists(),'Preserve earlier sweeps; choose a new run root'
queue.mkdir(parents=True)
output.mkdir(parents=True)
jobs=[]
for mode in ('height_scan','rgb','rgb_height','depth_height'):
    train=output/mode
    evaluate=output/'evaluations'/mode
    jobs.append(dict(id='train_'+mode,argv=['/bin/bash',str(root/'scripts/run_unitree_target_automated_intervention_local.sh'),
        '--gate','hazard_progress','--mode',mode,'--output',str(train),'--steps','4500','--seed','0',
        '--teacher-warmup','0','--max-falls','4500'],timeout_seconds=10800,gpu=True))
    jobs.append(dict(id='eval_'+mode,argv=['/bin/bash',str(root/'scripts/run_unitree_target_teacher_benchmark_local.sh'),
        '--nav-checkpoint',str(train/'training/final.pt'),'--output',str(evaluate),'--max-seconds','180'],
        timeout_seconds=7200,gpu=True))
    jobs.append(dict(id='plot_'+mode,argv=[python,str(root/'plot_unitree_target_teacher.py'),str(evaluate)],
        timeout_seconds=180,gpu=False))
jobs.append(dict(id='summarize',argv=[python,str(root/'summarize_target_modality_sweep.py')],timeout_seconds=180,gpu=False))
spec=dict(cwd=str(root),lock=str(root/'local/sequential_gpu/gpu.lock'),
    env=dict(MUJOCO_GL='egl',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1'),jobs=jobs)
(queue/'jobs.json').write_text(json.dumps(spec,indent=2)+'\n')
(output/'protocol.json').write_text(json.dumps(dict(modalities=['height_scan','rgb','rgb_height','depth_height'],
    decisions=4500,training_seed=0,gate='hazard_progress',encoder='pretrained_frozen',
    evaluation_manifest='artifacts/rgb_height_extended_20260913/manifest.json',evaluation_horizon=180,
    note='Screening study, one training seed. Same ordered training-goal bank; episode counts may differ by policy. No encoder or gate retuning.'),indent=2))
print(queue/'jobs.json')
