"""CPU checks for shared visual goal conditioning and rotation rectification."""
import math
import sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unitree_camera_observation import goal_image, quaternion_matrix, stabilize


def nominal(pitch=25., yaw=0.):
    p, y = math.radians(pitch), math.radians(yaw)
    right = torch.tensor([math.sin(y), -math.cos(y), 0.])
    back = torch.tensor([-math.cos(y)*math.cos(p), -math.sin(y)*math.cos(p), math.sin(p)])
    return torch.stack((right, torch.linalg.cross(back, right), back), 1)[None]


def test_identity_and_yaw_preserved():
    image = torch.rand(1, 3, 60, 80)
    for yaw in (0, 90, -80, 180):
        result, valid = stabilize(image, nominal(yaw=yaw), fovy=70, pitch_down=25)
        assert torch.allclose(image, result, atol=2e-5)
        assert valid.min() == 1


def test_invalid_borders_exposed_and_no_new_pixels():
    image = torch.ones(1, 3, 60, 80)
    result, mask = stabilize(image, nominal(pitch=48), fovy=70, pitch_down=25)
    assert 0 < mask.mean() < 1
    assert torch.allclose(result[:, :1], mask, atol=1e-6)
    depth, _ = stabilize(image[:, :1]*2, nominal(pitch=48), fovy=70, pitch_down=25,
                         depth=True, axial=True, far=3)
    assert torch.all(depth[mask == 0] == 3)
    assert torch.isfinite(depth).all()


def test_axial_plane_is_reexpressed_not_only_warped():
    rotation = nominal(pitch=37)
    flip = torch.diag(torch.tensor([1., -1., -1.]))
    source_to_target = (flip @ nominal().transpose(1, 2) @ rotation @ flip)[0]
    f = 60/(2*math.tan(math.radians(35)))
    y, x = torch.meshgrid(torch.arange(60)+.5, torch.arange(80)+.5, indexing='ij')
    rays = torch.stack(((x-40)/f, (y-30)/f, torch.ones_like(x)), -1)
    depth = (2 / (rays @ source_to_target[2]))[None, None]
    result, valid = stabilize(depth, rotation, fovy=70, pitch_down=25, depth=True, axial=True)
    assert (result[valid.bool()]-2).abs().max() < .025


def test_goal_is_in_encoder_image_without_covering_scene():
    image = torch.rand(2, 3, 60, 80)
    context = torch.zeros(2, 9)
    context[:, 6] = .5
    context[:, 8] = torch.tensor([1., -1.])
    output = goal_image(image, context, encoding='distance_bearing', distance_scale=14)
    assert output.shape == (2, 3, 72, 80)
    assert torch.equal(output[:, :, :60], image)
    assert not torch.equal(output[0, :, 60:], output[1, :, 60:])
    assert output[0, 1, 66, 4] == 1 and output[1, 1, 66, 10] == 1


def test_depth_annotation_range_and_distance_signal():
    image = torch.zeros(2, 1, 58, 87)
    context = torch.zeros(2, 9)
    context[:, 7] = 1
    context[:, 6] = torch.tensor([.1, .9])
    output = goal_image(image, context, encoding='distance_bearing', distance_scale=14, low=-.5, high=.5)
    assert output.shape == (2, 1, 70, 87)
    assert output.min() == -.5 and output.max() == .5
    assert not torch.equal(output[0], output[1])


def test_quaternion_convention():
    assert torch.allclose(quaternion_matrix(torch.tensor([[1., 0., 0., 0.]]))[0], torch.eye(3))


def test_zero_distance_does_not_fill_the_entire_goal_strip():
    result = goal_image(torch.zeros(1,1,58,87), torch.zeros(1,9),
                        encoding='distance_bearing', distance_scale=14)
    assert (result[:, :, 58:] > 0).float().mean() < .1


def test_legacy_checkpoint_cannot_silently_adopt_new_images(monkeypatch):
    from types import SimpleNamespace
    from unitree_nav_checkpoint import apply_unitree_checkpoint_config
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: {'args': {}})
    args = SimpleNamespace(model_path='old.pt', controller='policy', navigation_image_profile='stabilized_goal_v1')
    apply_unitree_checkpoint_config(args)
    assert args.navigation_image_profile == 'legacy'


def test_new_checkpoint_restores_visual_profile(monkeypatch):
    from types import SimpleNamespace
    from unitree_nav_checkpoint import apply_unitree_checkpoint_config
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: {'args': {'navigation_image_profile': 'stabilized_goal_v1'}})
    args = SimpleNamespace(model_path='new.pt', controller='policy', navigation_image_profile='legacy')
    apply_unitree_checkpoint_config(args)
    assert args.navigation_image_profile == 'stabilized_goal_v1'
