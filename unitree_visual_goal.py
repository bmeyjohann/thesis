"""Camera-visible, non-contact navigation goal geometry and depth annotation."""
import functools
import math

import torch
from torch.nn import functional as F

MARKER = dict(radius=.3, height=2.4, geom_group=5)


def marker_spec(radius=.3, height=2.4, geom_group=5):
    import mujoco
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='goal', mocap=True)
    body.add_geom(name='visual', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=(radius, height/2, 0), rgba=(.03, 1., .08, 1.),
                  contype=0, conaffinity=0, group=geom_group)
    return spec


def reset_marker_cache(env, env_ids):
    del env_ids
    env._visual_goal_previous = None


def update_marker(env, env_ids=None, *, height=2.4):
    del env_ids
    goal = env.command_manager._terms['pose']._goal_pos_w
    previous = getattr(env, '_visual_goal_previous', None)
    if previous is not None and torch.equal(goal, previous):
        return False
    pose = torch.zeros((env.num_envs, 7), device=env.device)
    pose[:, :2] = goal[:, :2]
    pose[:, 2] = env.scene.env_origins[:, 2]+height/2
    pose[:, 3] = 1
    env.scene['navigation_goal_marker'].write_mocap_pose_to_sim(pose)
    env._visual_goal_previous = goal.clone()
    env.sim.forward()
    return True


def configure_marker(env_cfg, parameters):
    from mjlab.entity import EntityCfg
    from mjlab.managers import EventTermCfg
    group = parameters['geom_group']
    env_cfg.scene.entities['navigation_goal_marker'] = EntityCfg(
        spec_fn=functools.partial(marker_spec, **parameters),
        init_state=EntityCfg.InitialStateCfg(pos=(0., 0., -100.)))
    env_cfg.events['navigation_goal_marker_reset'] = EventTermCfg(func=reset_marker_cache, mode='reset')
    env_cfg.events['navigation_goal_marker_step'] = EventTermCfg(func=update_marker, mode='step',
                                                               params={'height': parameters['height']})
    for sensor in env_cfg.scene.sensors:
        if hasattr(sensor, 'enabled_geom_groups'):
            sensor.enabled_geom_groups = tuple(sorted(set(sensor.enabled_geom_groups) | {group}))
        if hasattr(sensor, 'include_geom_groups'):
            groups = sensor.include_geom_groups
            sensor.include_geom_groups = tuple(g for g in (range(6) if groups is None else groups) if g != group)


def visible_cylinder_mask(ray_range, camera_position, camera_rotation, center, *, fovy, radius, height):
    """Identify actual cylinder hits, rejecting occlusion via measured ray range.

    Only known synthetic goal geometry is used; no obstacle map or RGB is exposed
    to the PHP policy. Camera axes follow MuJoCo: forward=-Z, up=Y.
    """
    batch, rows, cols = ray_range.shape
    y, x = torch.meshgrid(torch.arange(rows, device=ray_range.device)+.5,
                          torch.arange(cols, device=ray_range.device)+.5, indexing='ij')
    focal = rows/(2*math.tan(math.radians(fovy/2)))
    rays = torch.stack(((x-cols/2)/focal, -(y-rows/2)/focal, -torch.ones_like(x)), -1)
    direction = F.normalize(torch.einsum('bij,hwj->bhwi', camera_rotation, rays), dim=-1)
    origin = camera_position[:, None, None, :] - center[:, None, None, :]
    a = direction[..., :2].square().sum(-1).clamp_min(1e-12)
    b = 2*(origin[..., :2]*direction[..., :2]).sum(-1)
    c = origin[..., :2].square().sum(-1)-radius**2
    disc = b*b-4*a*c
    roots = [(-b-disc.clamp_min(0).sqrt())/(2*a), (-b+disc.clamp_min(0).sqrt())/(2*a)]
    candidates = []
    for t in roots:
        z = origin[..., 2]+t*direction[..., 2]
        valid = (disc >= 0) & (t > 0) & (z.abs() <= height/2)
        candidates.append(torch.where(valid, t, torch.full_like(t, float('inf'))))
    dz = direction[..., 2]
    for cap in (-height/2, height/2):
        t = (cap-origin[..., 2])/torch.where(dz.abs() > 1e-9, dz, torch.ones_like(dz))
        xy = origin[..., :2]+t[..., None]*direction[..., :2]
        valid = (dz.abs() > 1e-9) & (t > 0) & (xy.square().sum(-1) <= radius**2)
        candidates.append(torch.where(valid, t, torch.full_like(t, float('inf'))))
    distance = torch.stack(candidates).amin(0)
    return (torch.isfinite(distance) & ((ray_range-distance).abs() < .03+distance*.002))[:, None].float()


def annotate_php_goal(normalized_depth, stabilized_mask):
    # Use the same crop as PHP, then nearest-neighbor resize to retain mask identity.
    mask = F.interpolate(stabilized_mask[:, :, 2:, 4:-4], (58, 87), mode='nearest') > .5
    rows = torch.arange(58, device=normalized_depth.device)[None, None, :, None]
    stripes = torch.where((rows//3) % 2 == 0, .5, -.5).to(normalized_depth.dtype)
    return torch.where(mask, stripes, normalized_depth), mask
