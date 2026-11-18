"""Finite GPU check of camera HIL, live speed actions and decision accounting."""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_interactive_unitree_nav as interactive
from unitree_viser_training import BrowserTrainingSession
from unitree_viewer_pacing import LatestFramePacing


def main():
    sys.argv = ['pacing-smoke']
    defaults = vars(interactive.parse_args())
    old = json.loads((ROOT / 'models/unitree_mjlab_nav_human/target_human_rgb_height_20260925_175559/args.json').read_text())
    output = ROOT / 'local' / f'viewer_pacing_smoke_{time.time_ns()}'
    results = []
    for index, sensor in enumerate(['rgb_height', 'php_depth']):
        args = SimpleNamespace(**{**defaults, **old})
        args.navigation_observation_mode = sensor
        args.navigation_image_profile = os.environ.get('UNITREE_TEST_IMAGE_PROFILE', 'legacy')
        if sensor == 'php_depth':
            args.navigation_encoder_checkpoint = str(ROOT / 'artifacts/unitree_depth_transfer_20261013/models/php_student/depth_backbone.onnx')
        args.model_path = ''
        args.navigation_feature_lags = [0, 1, 25]
        args.scan_history, args.action_history = 1, 0
        args.online_train, args.autopilot = True, True
        args.online_output_dir, args.online_run_name = str(output), sensor
        args.online_total_steps, args.online_learning_starts = 32, 8
        args.online_batch_size, args.online_pref_batch_size = 256, 256
        args.online_n_step, args.online_policy_frequency = 1, 2
        args.online_lr_actor, args.online_lr_critic = 1e-4, 3e-4
        args.online_actor_bc_weight = 0.1
        args.online_objective_mode = 'pref_bc_rl'
        args.online_pref_lambda_init, args.online_pref_lambda_lr = 1., 0.
        args.online_wandb_mode = 'disabled'
        args.online_log_interval = 1
        args.online_session_receipt = str(output / f'{sensor}_receipt.json')
        args.human_dataset_dir = str(output / f'{sensor}_raw')
        args.human_input_device, args.controller = 'gamepad', 'policy'
        args.viewer_backend, args.viser_host, args.viser_port = 'viser', '0.0.0.0', 8772 + index
        args.observation_only = os.environ.get('UNITREE_TEST_OBSERVATION_ONLY') == '1'
        args.start_paused = False
        current = {}
        initialize, tick, original_close = BrowserTrainingSession.__init__, LatestFramePacing.tick, BrowserTrainingSession.close

        def init(session, *values):
            initialize(session, *values)
            current['session'] = session
            if args.navigation_image_profile in ('stabilized_goal_v2', 'stabilized_goal_v3'):
                from check_unitree_scene_goal import audit_goal
                audit_goal(session, output)
            if args.navigation_image_profile in ('stabilized_goal_v1', 'stabilized_goal_v2', 'stabilized_goal_v3'):
                base = interactive._prepare_policy_obs(session.raw, args)
                first = session.vision.transform(base).clone()
                changed = base.clone()
                changed[:, 7:9] *= -1
                second = session.vision.transform(changed)
                start = 9 + (187 if sensor == 'rgb_height' else 0)
                same = torch.allclose(first[:, start:-16], second[:, start:-16])
                assert same == (args.navigation_image_profile == 'stabilized_goal_v3'), 'Unexpected compass strip in encoder input'
                session.vision.transform(base)
                key = 'php_depth' if sensor == 'php_depth' else 'mono_rgb'
                expected = session.vision.last_encoder_inputs[key][0].cpu().numpy()
                visible = next(iter(session.vision.display_frames.values()))
                assert np.array_equal(visible, expected[0]+.5 if sensor == 'php_depth' else expected.transpose(1,2,0))
                from PIL import Image
                Image.fromarray((np.clip(visible, 0, 1)*255).round().astype(np.uint8)).save(output / f'{sensor}_policy_image.png')

        def close(session):
            original_close(session)
            if args.navigation_image_profile in ('stabilized_goal_v1', 'stabilized_goal_v2', 'stabilized_goal_v3'):
                path = output / sensor / 'final.pt'
                saved = torch.load(path, map_location='cpu', weights_only=False)
                assert saved['navigation_vision']['image_profile'] == args.navigation_image_profile
                assert saved['obs_dim'] == (153 if sensor == 'php_depth' else 810)
                restored_args = SimpleNamespace(**vars(args))
                restored_args.model_path = str(path)
                restored_args.navigation_encoder_checkpoint = '/not/available.pt'
                restored_args.online_train = False
                restored_args.human_dataset_dir = ''
                restored = object.__new__(BrowserTrainingSession)
                initialize(restored, restored_args, session.env, session.raw, session.cells)
                assert torch.isfinite(restored.proposal()).all()
                assert restored.vision.payload['image_profile'] == args.navigation_image_profile
                assert all(torch.equal(v, restored.actor.state_dict()[k].cpu()) for k,v in saved['actor_state_dict'].items())
                original_close(restored)

        def sample(_):
            return SimpleNamespace(action=np.array([.2, .1, 0.], dtype=np.float32),
                intervening=current['session'].steps < 4, connected=True, stale=False,
                command_norm=.224, state_age_s=0., state={'buttons': {}}, transport={'connected': True})

        def paced(viewer):
            if 'started' not in current:
                current['started'] = time.perf_counter()
                if args.observation_only:
                    from urllib.request import urlopen
                    assert not hasattr(viewer, '_scene')
                    state = json.load(urlopen(f'http://127.0.0.1:{args.viser_port}/state'))
                    assert state['sensor'] == sensor and len(state['history']) == 3
                    assert (state['scan'] is not None) == (sensor == 'rgb_height')
                    extra = 12 if args.navigation_image_profile in ('stabilized_goal_v1', 'stabilized_goal_v2') else 0
                    expected_size = (87, 58+extra) if sensor == 'php_depth' else (80, 60+extra)
                    frame = state['history'][0]['frames'][0]
                    assert (frame['width'], frame['height']) == expected_size
                    hold = (float(os.environ.get('UNITREE_TEST_UI_HOLD_S', '0'))
                            if sensor == os.environ.get('UNITREE_TEST_UI_SENSOR', 'rgb_height') else 0)
                    if hold:
                        viewer.pause()
                        current['ui_until'] = time.perf_counter() + hold
                else:
                    labels = {getattr(h, 'label', '') for h in viewer._server.gui._gui_input_handle_from_uuid.values()}
                    assert {'Simulation speed', 'Display FPS cap', 'Adaptive display frame skipping'} <= labels
                viewer.request_action('CUSTOM', {'type': 'speed', 'value': 1.5})
            if current.get('ui_until') and time.perf_counter() >= current['ui_until']:
                current['ui_until'] = None
                current['started'] = time.perf_counter()
                viewer.resume()
            if current['session'].steps >= 16 and not current.get('changed'):
                viewer.request_action('CUSTOM', {'type': 'speed', 'value': 2.0})
                viewer.request_action('CUSTOM', {'type': 'display_fps', 'value': 10})
                viewer.request_action('CUSTOM', {'type': 'policy_preview', 'value': False})
                current['changed'] = True
            return tick(viewer)

        with patch.object(BrowserTrainingSession, '__init__', init), \
             patch.object(BrowserTrainingSession, 'close', close), \
             patch.object(interactive.UnitreeGamepadController, 'sample', sample), \
             patch.object(LatestFramePacing, 'tick', paced):
            interactive.run(args)
        elapsed = time.perf_counter() - current['started']
        summary = json.loads((output / sensor / 'browser_summary.json').read_text())
        assert summary['navigation_decisions'] == 32
        assert summary['environment_ticks'] == 128
        assert summary['intervention_decisions'] == 4
        assert [e['target_speed'] for e in summary['playback_events']] == [1.5, 2.0]
        manifest = json.loads((output / f'{sensor}_raw' / 'manifest.json').read_text())
        assert manifest['summary']['rows'] == 32
        if args.navigation_image_profile in ('stabilized_goal_v1', 'stabilized_goal_v2', 'stabilized_goal_v3'):
            part = np.load(next((output / f'{sensor}_raw' / 'parts').glob('*.npz')))
            key = 'policy_image_php_depth' if sensor == 'php_depth' else 'policy_image_mono_rgb'
            assert key in part and 'next_'+key in part and 'vision_valid_mask' in part
            extra = 0 if args.navigation_image_profile == 'stabilized_goal_v3' else 12
            assert part[key].shape[1:3] == ((58+extra,87) if sensor == 'php_depth' else (60+extra,80))
        receipt = json.loads((output / f'{sensor}_receipt.json').read_text())
        assert receipt['limit_reached'] and Path(receipt['checkpoint']).is_file()
        results.append(dict(sensor=sensor, status='PASS', elapsed_s=elapsed,
            achieved_including_save=6.4 / elapsed, summary=summary))
        print(json.dumps(results[-1]), flush=True)
    (output / 'verification.json').write_text(json.dumps(results, indent=2))
    print(f'VERIFICATION={output}', flush=True)


if __name__ == '__main__':
    main()
