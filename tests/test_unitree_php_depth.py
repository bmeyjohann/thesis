"""CPU checks for the released PHP geometry and preprocessing contract."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unitree_php_depth import PHPDepthEncoder, prepare_php_depth, range_to_axial


def test_planar_range_conversion():
    h, w, fovy = 60, 106, 58.6
    focal = h / (2 * math.tan(math.radians(fovy / 2)))
    x = (torch.arange(w) + .5 - w / 2) / focal
    y = (torch.arange(h) + .5 - h / 2) / focal
    ray_range = 2 * torch.sqrt(1 + x[None, :]**2 + y[:, None]**2)
    assert torch.allclose(range_to_axial(ray_range[None], fovy), torch.full((1, h, w), 2.))


def test_preprocessing_bounds_and_missing_depth():
    near = prepare_php_depth(torch.full((1, 60, 106), .3))
    far = prepare_php_depth(torch.full((1, 60, 106), 3.))
    missing = prepare_php_depth(torch.full((1, 60, 106), float('nan')))
    assert near.shape == (1, 1, 58, 87)
    assert torch.allclose(near, torch.full_like(near, -.5), atol=1e-6)
    assert torch.allclose(far, torch.full_like(far, .5), atol=1e-6)
    assert torch.equal(missing, far)


def test_released_architecture_shapes():
    model = PHPDepthEncoder()
    assert sum(p.numel() for p in model.parameters()) == 25632
    x = torch.zeros(2, 1, 58, 87)
    assert model.spatial(x).shape == (2, 64, 8, 11)
    assert model(x).shape == (2, 32)
