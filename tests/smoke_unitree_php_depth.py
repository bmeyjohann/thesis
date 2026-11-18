"""Bounded PHP camera -> online head -> checkpoint -> viewer reload verification."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import eval_interactive_unitree_nav as interactive
from unitree_viser_training import BrowserTrainingSession
from unitree_php_depth import range_to_axial


def main():
    sys.argv=['smoke']
    args=interactive.parse_args()
    src=ROOT/'models/unitree_mjlab_nav_human/target_human_rgb_height_20260925_175559/args.json'
    vars(args).update(json.loads(src.read_text()))
    out=ROOT/'artifacts/unitree_depth_transfer_20261013'
    args.model_path=''
    args.navigation_observation_mode='php_depth'
    args.navigation_encoder_checkpoint=str(out/'models/php_student/depth_backbone.onnx')
    args.online_train=True
    args.online_output_dir=str(out/'wiring_smoke')
    args.online_run_name='php_depth_8decisions'
    args.online_total_steps=8
    args.online_learning_starts=2
    args.online_batch_size=2
    args.online_n_step=1
    args.online_log_interval=1
    args.online_checkpoint_interval=4
    args.online_wandb_mode='disabled'
    args.online_objective_mode='pref_rl'
    args.human_dataset_dir=''
    args.human_input_device='none'
    args.viewer_backend='viser'
    args.viser_host='127.0.0.1'
    args.viser_port=8766
    args.viser_max_steps=32
    args.start_paused=False
    args.autopilot=True
    args.show_rgb=False
    args.controller='policy'
    before={}
    original_init=BrowserTrainingSession.__init__
    original_close=BrowserTrainingSession.close
    def initialize(s,*values):
        original_init(s,*values)
        assert s.obs.shape==(1,56),s.obs.shape
        stacked=s.history.reset(interactive._prepare_policy_obs(values[2],s.args))
        altered=stacked.clone();altered[:,9:-15]+=100
        assert torch.equal(s.vision.transform(stacked),s.vision.transform(altered)), 'privileged scan leaked'
        before.update({k:v.detach().cpu().clone() for k,v in s.vision.state_dict().items()})
    def close(s):
        assert all(torch.equal(v,s.vision.state_dict()[k].cpu()) for k,v in before.items())
        original_close(s)
    with patch.object(BrowserTrainingSession,'__init__',initialize),patch.object(BrowserTrainingSession,'close',close):
        interactive.run(args)
    run=Path(args.online_output_dir)/args.online_run_name
    saved=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
    summary=json.loads((run/'browser_summary.json').read_text())
    assert summary['navigation_decisions']==8 and summary['environment_ticks']==32,summary
    assert summary['intervention_decisions']==0
    metrics=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['updates']>0
    args.model_path=str(run/'final.pt');args.online_train=False
    args.navigation_encoder_checkpoint='/absent/donor.onnx'
    env=interactive.make_env(args,num_envs=1,render=False)
    try:
        raw,_,_,_,cells=interactive._reset_until_feasible(args,env)
        s=BrowserTrainingSession(args,env,raw,cells)
        assert s.obs.shape==(1,56) and torch.isfinite(s.proposal()).all()
        assert all(torch.equal(v,s.vision.state_dict()[k].cpu()) for k,v in before.items())
        assert all(torch.equal(v,s.actor.state_dict()[k].cpu()) for k,v in saved['actor_state_dict'].items())
        s.close()
    finally:env.close()
    result=dict(status='PASS',navigation_obs_dim=56,steps=8,ticks=32,
        updates=metrics[-1]['updates'],height_scan_leakage=False,frozen_encoder=True,
        donor_free_reload=True,human_hardware_tested=False,low_level_still_privileged=True,run=str(run))
    (out/'hil_wiring_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
