"""Bounded live scene-goal assertions, called before the browser smoke starts."""
import json
import numpy as np
import torch
from PIL import Image


def audit_goal(session, output):
    import eval_interactive_unitree_nav as interactive
    manager = session.env.env.unwrapped
    vision, args = session.vision, session.args
    robot = manager.scene['robot']
    goal = manager.command_manager._terms['pose']._goal_pos_w
    original = goal.clone()
    base = interactive._prepare_policy_obs(session.raw, args)
    yaw = robot.data.heading_w
    forward = torch.stack((yaw.cos(), yaw.sin()), -1)
    left = torch.stack((-yaw.sin(), yaw.cos()), -1)
    scans = {name: sensor.data.distances.clone() for name, sensor in manager.scene.sensors.items()
             if hasattr(sensor.cfg, 'include_geom_groups')}
    records = []
    model = manager.sim.mj_model
    geometries = [model.geom(i) for i in range(model.ngeom) if 'navigation_goal_marker' in (model.geom(i).name or '')]
    assert geometries and all(int(g.group) == 5 and int(g.contype) == 0 and int(g.conaffinity) == 0 for g in geometries)
    try:
        for label, distance, lateral in [('front',1.8,0.), ('left',1.8,.5), ('right',1.8,-.5), ('far',5.,0.), ('behind',-2.,0.), ('scan',.4,0.)]:
            goal[:, :2] = robot.data.root_link_pos_w[:, :2]+distance*forward+lateral*left
            conditioned = base.clone()
            length = (distance**2+lateral**2)**.5
            conditioned[:, 6] = length/args.goal_distance_scale
            conditioned[:, 7] = distance/length
            conditioned[:, 8] = lateral/length
            vision.transform(conditioned)
            if args.navigation_observation_mode == 'rgb_height':
                raw = vision.last_frames['vision_rgb'].astype(float)
                mask = (raw[...,1] > 2*raw[...,0]) & (raw[...,1] > 2*raw[...,2]) & (raw[...,1] > 3)
                if label in ('front','left','right','far'):
                    assert vision.last_goal_mask.sum() > 3, 'Goal identity mask missed scene cylinder'
            else:
                mask = vision.last_goal_mask[0,0].cpu().numpy()
            count = int(mask.sum())
            visible = next(iter(vision.display_frames.values()))
            Image.fromarray((np.clip(visible,0,1)*255).round().astype(np.uint8)).save(output/f'{args.navigation_observation_mode}_goal_{label}.png')
            print('goal-check', label, count, 'root', robot.data.root_link_pos_w.tolist(), 'goal', goal.tolist(), flush=True)
            if label in ('front','left','right','far'):
                assert count > 3, (args.navigation_observation_mode, label, count)
            if label == 'behind':
                assert count == 0, 'Marker drawn through the back of the camera'
            for name, before in scans.items():
                sensor = manager.scene.sensors[name]
                sensor.update(0.)
                assert torch.allclose(before, sensor.data.distances, atol=1e-5), 'Goal marker leaked into terrain scan'
            records.append(dict(case=label, pixels=count, x=float(np.nonzero(mask)[1].mean()) if count else None))
        assert records[1]['x'] < records[0]['x'] < records[2]['x']
        (output/f'{args.navigation_observation_mode}_goal_checks.json').write_text(json.dumps(records, indent=2))
    finally:
        goal.copy_(original)
        vision.transform(base)
