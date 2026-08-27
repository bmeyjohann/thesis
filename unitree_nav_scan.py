"""Height-scan footprint definitions shared by Unitree train and eval."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ForwardFrustumGridPatternCfg:
    """Vertical ground probes arranged like a camera's forward field of view."""

    near_distance: float = 0.25
    far_distance: float = 4.0
    horizontal_fov_deg: float = 70.0
    side: int = 17
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def generate_rays(self, mj_model, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        del mj_model
        if self.side < 2:
            raise ValueError("Forward frustum scanner requires side >= 2")
        if self.near_distance < 0.0 or self.far_distance <= self.near_distance:
            raise ValueError("Forward frustum scanner requires 0 <= near < far")
        if not 0.0 < self.horizontal_fov_deg < 180.0:
            raise ValueError("Forward frustum FOV must be between 0 and 180 degrees")

        forward = torch.linspace(
            self.near_distance, self.far_distance, self.side, device=device, dtype=torch.float32
        )
        lateral_unit = torch.linspace(-1.0, 1.0, self.side, device=device, dtype=torch.float32)
        lateral_norm, forward_grid = torch.meshgrid(lateral_unit, forward, indexing="ij")
        half_width = forward_grid * math.tan(math.radians(self.horizontal_fov_deg) / 2.0)

        offsets = torch.zeros((self.side * self.side, 3), device=device, dtype=torch.float32)
        offsets[:, 0] = forward_grid.flatten()
        offsets[:, 1] = (lateral_norm * half_width).flatten()

        direction = torch.as_tensor(self.direction, device=device, dtype=torch.float32)
        direction = direction / direction.norm()
        directions = direction.unsqueeze(0).expand(offsets.shape[0], 3).clone()
        return offsets, directions


def scan_lateral_forward_coordinates(
    *,
    pattern: str,
    side: int,
    forward_size: float,
    lateral_size: float,
    frustum_near: float,
    frustum_far: float,
    frustum_fov_deg: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [lateral-row, forward-column] coordinates for a square scan."""
    if pattern == "forward_frustum":
        forward_axis = torch.linspace(frustum_near, frustum_far, side, device=device)
        lateral_unit = torch.linspace(-1.0, 1.0, side, device=device)
        lateral_norm, forward = torch.meshgrid(lateral_unit, forward_axis, indexing="ij")
        lateral = lateral_norm * forward * math.tan(math.radians(frustum_fov_deg) / 2.0)
        return lateral, forward

    forward_axis = torch.linspace(-forward_size / 2.0, forward_size / 2.0, side, device=device)
    lateral_axis = torch.linspace(-lateral_size / 2.0, lateral_size / 2.0, side, device=device)
    lateral, forward = torch.meshgrid(lateral_axis, forward_axis, indexing="ij")
    return lateral, forward


def scan_points_numpy(**kwargs) -> np.ndarray:
    lateral, forward = scan_lateral_forward_coordinates(device="cpu", **kwargs)
    return np.stack([forward.numpy().reshape(-1), lateral.numpy().reshape(-1)], axis=1)
