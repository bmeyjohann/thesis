"""Versioned, shared policy/UI camera preprocessing (no additional rendering)."""
import math

import torch
from torch.nn import functional as F

PROFILE = 'stabilized_goal_v1'


def quaternion_matrix(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def stabilize(image, camera_rotation, *, fovy, pitch_down, depth=False, axial=False, far=3.):
    """Rotation-only rectification; depth is re-expressed in the virtual optical frame.

    camera_rotation maps MuJoCo camera axes into world coordinates. In simulation
    it is an ideal attitude measurement, not an estimated/noisy physical IMU.
    """
    batch, _, height, width = image.shape
    forward = -camera_rotation[:, :, 2]
    yaw = torch.atan2(forward[:, 1], forward[:, 0])
    pitch = math.radians(pitch_down)
    zero = torch.zeros_like(yaw)
    right = torch.stack((yaw.sin(), -yaw.cos(), zero), -1)
    direction = torch.stack((yaw.cos()*math.cos(pitch), yaw.sin()*math.cos(pitch),
                             zero-math.sin(pitch)), -1)
    back = -direction
    up = torch.linalg.cross(back, right)
    target = torch.stack((right, up, back), -1)
    flip = torch.diag(image.new_tensor([1., -1., -1.]))
    rotation = flip @ camera_rotation.transpose(1, 2) @ target @ flip
    focal = height / (2 * math.tan(math.radians(fovy/2)))
    y, x = torch.meshgrid(torch.arange(height, device=image.device, dtype=image.dtype)+.5,
                          torch.arange(width, device=image.device, dtype=image.dtype)+.5, indexing='ij')
    rays = torch.stack(((x-width/2)/focal, (y-height/2)/focal, torch.ones_like(x)), -1)
    source = torch.einsum('bij,hwj->bhwi', rotation, rays)
    z = source[..., 2]
    sx = focal*source[..., 0]/z.clamp_min(1e-6)+width/2
    sy = focal*source[..., 1]/z.clamp_min(1e-6)+height/2
    valid = ((z > 0) & (sx >= .5-1e-4) & (sx <= width-.5+1e-4)
             & (sy >= .5-1e-4) & (sy <= height-.5+1e-4))[:, None]
    grid = torch.stack((2*sx/width-1, 2*sy/height-1), -1)
    result = F.grid_sample(image, grid, mode='nearest' if depth else 'bilinear',
                           padding_mode='zeros', align_corners=False)
    if axial:
        result = result / z[:, None].clamp_min(1e-6)
    result = torch.where(valid, result, torch.full_like(result, far if depth else 0.))
    return result, valid.float()


def goal_image(image, context, *, encoding, distance_scale, low=0., high=1.):
    """Append a synthetic goal compass/distance strip BEFORE the image encoder.

    No scene pixels are covered. This is a goal annotation, not a depth return
    or a physical cylinder. Up is body-forward, left is body-left, full 360 deg.
    """
    batch, channels, _, width = image.shape
    if encoding == 'distance_bearing':
        forward, left = context[:, 7], context[:, 8]
        distance = context[:, 6].clamp(0, 1)
    else:
        raise ValueError('Visual goal profile requires distance_bearing goal encoding')
    hud = torch.full((batch, channels, 12, width), low, device=image.device, dtype=image.dtype)
    y, x = torch.meshgrid(torch.arange(12, device=image.device),
                          torch.arange(width, device=image.device), indexing='ij')
    dx, dy = x-7, y-6
    circle = ((dx*dx+dy*dy >= 19) & (dx*dx+dy*dy <= 29))[None]
    along = dx[None]*(-left[:, None, None]) + dy[None]*(-forward[:, None, None])
    cross = dx[None]*forward[:, None, None] - dy[None]*left[:, None, None]
    has_direction = (forward.square()+left.square() > 1e-8)[:, None, None]
    arrow = (along >= 0) & (along <= 5) & (cross.abs() < .8) & has_direction
    bar = ((x[None] >= 20) & (x[None] < 20+distance[:, None, None]*(width-22))
           & (y[None] >= 5) & (y[None] <= 8))
    mask = circle | arrow | bar
    color = image.new_full((1, channels, 1, 1), high)
    if channels == 3:
        color = image.new_tensor([.1, 1., .2]).reshape(1, 3, 1, 1)*(high-low)+low
    hud = torch.where(mask[:, None], color, hud)
    return torch.cat((image, hud), -2)
