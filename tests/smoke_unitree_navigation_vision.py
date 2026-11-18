"""Real camera + frozen transfer + online replay + self-contained reload smoke."""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_interactive_unitree_nav as interactive
from unitree_viser_training import BrowserTrainingSession
from unitree_nav_camera_features import VARIANTS


def main():
    mode = sys.argv[1]
    name = f'vision_smoke_{mode}_{time.time_ns()}'
    dataset = ROOT/'local/unitree_human_interventions'/name
    sys.argv = ['smoke']
    args = interactive.parse_args()
    vars(args).update(json.loads((ROOT/'models/unitree_mjlab_nav_human/target_human_20260910_193717/args.json').read_text()))
    args.navigation_observation_mode = mode
    args.online_run_name = name
    args.human_dataset_dir = str(dataset)
    args.online_total_steps = 12
    args.online_learning_starts = 2
    args.online_n_step = 1
    args.online_batch_size = 2
    args.online_checkpoint_interval = 6
    args.online_log_interval = 1
    args.online_wandb_mode = 'disabled'
    args.viser_port = 8093
    args.start_paused = False
    args.gamepad_debug_console = False
    args.target_terrain_preset = 'challenge'
    args.target_terrain_dynamic_obstacles = True
    args.target_terrain_material_resolution = .2
    args.target_terrain_ice_friction = .06
    args.target_terrain_ice_contact_priority = 1
    baseline, after = {}, {}
    original_close = BrowserTrainingSession.close

    def close(session):
        if session.vision is not None:
            after.update({k:v.detach().cpu().clone() for k,v in session.vision.state_dict().items()})
        original_close(session)

    original_init = BrowserTrainingSession.__init__
    def initialize(session, *values):
        original_init(session, *values)
        if session.vision is not None:
            baseline.update({k:v.detach().cpu().clone() for k,v in session.vision.state_dict().items()})
            stacked = session.history.reset(interactive._prepare_policy_obs(values[2], session.args))
            original = session.vision.transform(stacked)
            perturbed = stacked.clone()
            perturbed[:,9:-15] += 10
            changed = session.vision.transform(perturbed)
            assert torch.equal(original,changed) == (not VARIANTS[mode][0]), 'Height-scan leakage or missing scan'
    count = 0
    def sample(_):
        nonlocal count
        count += 1
        return SimpleNamespace(action=np.array([.2,.1,0], dtype=np.float32), intervening=count%7<3,
            connected=True, stale=False, command_norm=.224, state_age_s=0., state={'buttons':{}},
            transport={'connected':True})
    started = time.perf_counter()
    with patch.object(interactive.UnitreeGamepadController, 'sample', sample), \
         patch.object(BrowserTrainingSession, '__init__', initialize), \
         patch.object(BrowserTrainingSession, 'close', close):
        interactive.run(args)
    run = Path(args.online_output_dir)/args.online_run_name
    saved = torch.load(run/'final.pt', map_location='cpu', weights_only=False)
    assert saved['step'] == 12
    summary = json.loads((run/'browser_summary.json').read_text())
    assert summary['limit_reached'] and summary['environment_ticks'] == 48
    metrics = [json.loads(r) for r in (run/'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['updates'] > 0
    include_scan, modalities = VARIANTS[mode]
    assert saved['obs_dim'] == 9+15+187*include_scan+64*len(modalities)
    assert baseline and all(torch.equal(v,after[k]) for k,v in baseline.items())
    part = np.load(next((dataset/'parts').glob('*.npz')))
    assert part['observations'].shape == (12,saved['obs_dim'])
    from PIL import Image
    for key in ['vision_rgb','vision_right_rgb','vision_depth']:
        if key in part:
            assert part[key].shape[0] == 12 and np.isfinite(part[key]).all()
            assert float(part[key].std()) > 0
            if 'depth' not in key:
                Image.fromarray(part[key][0]).save(run/f'{key}.png')
    # Rebuild a head and encoder exclusively from the navigation checkpoint,
    # with an intentionally unavailable donor path. No dependency on donor bytes.
    args.model_path = str(run/'final.pt')
    args.navigation_encoder_checkpoint = '/does/not/exist.pt'
    args.online_train = False
    args.human_dataset_dir = ''
    env = interactive.make_env(args, num_envs=1, render=False)
    raw,_,_,_,cells = interactive._reset_until_feasible(args,env)
    session = BrowserTrainingSession(args,env,raw,cells)
    assert session.obs.shape[1] == saved['obs_dim']
    assert torch.isfinite(session.proposal()).all()
    assert all(torch.equal(v,session.actor.state_dict()[k].cpu()) for k,v in saved['actor_state_dict'].items())
    assert all(torch.equal(v,session.vision.state_dict()[k].cpu()) for k,v in baseline.items())
    session.close()
    env.close()
    print(json.dumps(dict(status='PASS',mode=mode,obs_dim=saved['obs_dim'],decisions=12,
                         updates=metrics[-1]['updates'],frozen_encoder=True,self_contained_reload=True,
                         wall_seconds=time.perf_counter()-started,run=str(run))),flush=True)


if __name__ == '__main__':
    main()
