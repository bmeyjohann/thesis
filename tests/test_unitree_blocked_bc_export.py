"""No-network checks of the metrics-only W&B exporter."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def test_metrics_only_export_preserves_counts_and_is_idempotent(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('blocked_export_test', root/'scripts/cluster/unitree_blocked_bc_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STUDY = tmp_path
    module.status = lambda **kwargs: None
    fields = ('seeds','decisions_per_training','bc_weights','bc_semantics','sampling',
              'alpha_state_scope','fixed_lambda','train_seeds','validation_seed',
              'heldout_seed','cases_per_manifest','initialization','criteria','ranking')
    (tmp_path/'plan.json').write_text(json.dumps(dict.fromkeys(fields, 'test')))
    run_dir = tmp_path/'run'
    run_dir.mkdir()
    (run_dir/'args.json').write_text(json.dumps(dict(actor_bc_weight=.2,seed=73,
        output_dir='/private/runtime',init_checkpoint='/private/checkpoint',pref_sampling_mode='separate')))
    (run_dir/'metrics.jsonl').write_text('{"step":100,"actor_loss_bc":0.1}\n{"step":200,"actor_loss_bc":0.05}\n')
    case = dict(tile_index=7,root_pose_relative=[-3,0,1,1,0,0,0],goal_xy_relative=[3,0])
    episode = dict(manifest_episode_index=0,terrain_tile_id=7,start_local_xy=[-3,0],
        goal_local_xy=[3,0],start_quaternion=[1,0,0,0],blocked_corridor=True,
        blocking_component_count=2,collision_cost_sum=2,cost_sum=2,success=True,
        fall_cost_sum=0,episode_length_s=10,min_goal_distance=.3)
    manifest,metrics=tmp_path/'manifest.json',tmp_path/'metrics.json'
    manifest.write_text(json.dumps(dict(episodes=[case])))
    metrics.write_text(json.dumps(dict(episodes=[episode])))
    class CommError(Exception):
        pass
    state = dict(created=False,logs=[],init=[])
    remote = SimpleNamespace(summary={},url='https://wandb.ai/test/run')
    remote.scan_history = lambda **kwargs: []
    def get_run(path):
        if not state['created']:
            raise CommError('not found')
        return remote
    def init(**kwargs):
        state['created']=True
        state['init'].append(kwargs)
        remote.log=lambda row: state['logs'].append(row)
        remote.finish=lambda: None
        return remote
    fake = SimpleNamespace(Api=lambda:SimpleNamespace(run=get_run),init=init,
        Settings=lambda **kwargs:kwargs,Table=lambda **kwargs:dict(table=kwargs),
        errors=SimpleNamespace(CommError=CommError))
    monkeypatch.setitem(sys.modules,'wandb',fake)
    module.wandb_export('test',run_dir,{'validation':(metrics,manifest)})
    assert remote.summary['validation/episodes']==1
    assert remote.summary['validation/accumulated_collision_cost']==2
    config=state['init'][0]['config']
    assert config['bc_weight']==.2 and config['training_seed']==73
    assert 'output_dir' not in config['resolved'] and 'init_checkpoint' not in config['resolved']
    assert state['init'][0]['settings']['x_disable_meta']
    assert any('validation/episode_table' in row for row in state['logs'])
    assert [row['train/source_row'] for row in state['logs'] if 'train/source_row' in row]==[0,1]
    before=len(state['logs'])
    module.wandb_export('test',run_dir,{'validation':(metrics,manifest)})
    assert len(state['logs'])==before and len(state['init'])==1


def test_summary_json_handles_zero_cost_failure_and_separate_fall(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('blocked_summary_test', root/'scripts/cluster/unitree_blocked_bc_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = dict(tile_index=7,root_pose_relative=[-3,0,1,1,0,0,0],goal_xy_relative=[3,0])
    base = dict(terrain_tile_id=7,start_local_xy=[-3,0],goal_local_xy=[3,0],
        start_quaternion=[1,0,0,0],blocked_corridor=True,blocking_component_count=2,
        episode_length_s=60,min_goal_distance=2,success=False)
    episodes = [dict(base,manifest_episode_index=0,collision_cost_sum=0,cost_sum=0,fall_cost_sum=0),
                dict(base,manifest_episode_index=1,collision_cost_sum=1,cost_sum=2,fall_cost_sum=1)]
    manifest,metrics=tmp_path/'manifest.json',tmp_path/'metrics.json'
    manifest.write_text(json.dumps(dict(episodes=[case,case])))
    metrics.write_text(json.dumps(dict(episodes=episodes)))
    summary=module.summarize(metrics,manifest)
    module.write_json(tmp_path/'summary.json',summary)
    assert summary['zero_cost_non_success']==1 and summary['falls']==1
    assert summary['accumulated_collision_cost']==1
    assert type(summary['zero_cost_non_success']) is int


def test_completed_preflight_reuse_verifies_inputs_and_never_repeats(tmp_path):
    import argparse
    import pytest
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('blocked_reuse_test',root/'scripts/cluster/unitree_blocked_bc_study.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source,target=tmp_path/'old',tmp_path/'new'
    source.mkdir();target.mkdir()
    args=argparse.Namespace(controller='geom_scan_teacher',num_episodes=8,output_dir=str(target/'results'))
    paths={k:source/(k+'.json') for k in ('spec','manifest','metrics')}
    paths['spec'].write_text(json.dumps(dict(mode='evaluate',args=dict(vars(args),output_dir=str(source/'results')))))
    paths['manifest'].write_text('{}');paths['metrics'].write_text('{}')
    record={k:str(p) for k,p in paths.items()}
    record.update({k+'_sha256':module.digest(p) for k,p in paths.items()})
    record['source_study']=str(source)
    (target/'plan.json').write_text(json.dumps(dict(reuse_preflight={'preflight':record})))
    module.STUDY=target;module.RESULTS=target/'results';module.status=lambda **kw:None
    module.summarize=lambda *a:dict(episodes=8)
    assert not module.reuse_completed_preflight('unknown',args,paths['manifest'])
    assert module.reuse_completed_preflight('preflight',args,paths['manifest'])
    with pytest.raises(RuntimeError,match='already exists'):
        module.reuse_completed_preflight('preflight',args,paths['manifest'])
    paths['metrics'].write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='input changed'):
        module.reuse_completed_preflight('preflight',args,paths['manifest'])
