"""PHP student-assets-v1 depth encoder and its explicit preprocessing contract."""
from pathlib import Path
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

PHP_SHA256 = "5b22466a218944e3b0ab0f4f82a910e3eec35375ee03e7fb94eb65fa88431b56"


def range_to_axial(depth_range: torch.Tensor, fovy_deg: float) -> torch.Tensor:
    """MJLab/Warp returns Euclidean ray range; PHP and stereo use optical-axis depth."""
    import math
    h, w = depth_range.shape[-2:]
    f = h / (2 * math.tan(math.radians(fovy_deg / 2)))
    u = (torch.arange(w, device=depth_range.device, dtype=depth_range.dtype) + .5 - w/2) / f
    v = (torch.arange(h, device=depth_range.device, dtype=depth_range.dtype) + .5 - h/2) / f
    return depth_range / torch.sqrt(1 + v[:, None]**2 + u[None, :]**2)


def prepare_php_depth(depth_m: torch.Tensor) -> torch.Tensor:
    """Axial depth [B,H,W] in metres, with PHP's uncropped 89.5-degree HFOV."""
    if depth_m.ndim != 3:
        raise ValueError("Expected [batch,height,width] metric camera depth")
    x = torch.nan_to_num(depth_m.float(), nan=3., posinf=3., neginf=.3)
    x = x.clamp(.3, 3.)[:, None]
    if x.shape[-2:] != (60, 106):
        x = F.interpolate(x, (60, 106), mode="bicubic", align_corners=False, antialias=True)
        x = x.clamp(.3, 3.)
    x = x[:, :, 2:, 4:-4]
    x = F.interpolate(x, (58, 87), mode="bicubic", align_corners=False, antialias=True)
    x = x.clamp(max=3.)
    x = torch.where(x < .15, torch.full_like(x, 3.), x)
    return (x - .3) / 2.7 - .5


class PHPDepthEncoder(nn.Module):
    """Released feed-forward 25,632-parameter CNN; no recurrent state."""
    def __init__(self):
        super().__init__()
        self.image_compression = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 32))
        self.output_activation = nn.ELU()

    @classmethod
    def from_release(cls, path):
        import onnx
        from onnx.numpy_helper import to_array
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != PHP_SHA256:
            raise ValueError("This adapter requires the verified student-assets-v1 backbone")
        model = cls()
        graph = onnx.load(str(path))
        model.load_state_dict({t.name: torch.from_numpy(to_array(t).copy())
                               for t in graph.graph.initializer}, strict=True)
        return model.eval()

    def spatial(self, normalized):
        return self.image_compression[:6](normalized)

    def forward(self, normalized):
        return self.output_activation(self.image_compression(normalized))

    def from_metres(self, depth_m):
        return self(prepare_php_depth(depth_m))


def navigation_input(stacked, features, action_history):
    """Retain nine navigation context values and action history; remove all scan values."""
    history_dim = 3 * int(action_history)
    blocks = [stacked[:, :9], features]
    if history_dim:
        blocks.append(stacked[:, -history_dim:])
    return torch.cat(blocks, dim=1)
