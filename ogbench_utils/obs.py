from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


POLICY_OBS_KEYS: Sequence[str] = ("policy", "pixels", "image", "observation")


def infer_pixel_shape(obs_input: Any) -> tuple[int, int, int]:
    """Infer (C, H, W) from a sample observation."""
    tensor = _extract_sample(obs_input)
    shape = tuple(int(s) for s in tensor.shape)
    if len(shape) == 4:
        shape = shape[1:]
    if len(shape) != 3:
        raise RuntimeError(f"Unable to infer pixel shape from {shape}")
    channels_first = shape[0] in (1, 3, 4)
    channels_last = shape[-1] in (1, 3, 4)
    if channels_first and not channels_last:
        return shape
    if channels_last and not channels_first:
        return (shape[-1], shape[0], shape[1])
    return shape


def prepare_observation(
    obs_input: Any,
    *,
    device: torch.device,
    obs_mode: str,
    pixel_shape: tuple[int, int, int] | None,
    flatten: bool = True,
) -> torch.Tensor:
    """Convert environment output to a torch tensor matching the requested mode."""
    tensor = _to_tensor(obs_input, device=device)
    if obs_mode == "pixels":
        tensor = _ensure_channels_first(tensor, pixel_shape)
        tensor = _ensure_float_pixels(tensor)
        if flatten:
            tensor = tensor.view(tensor.shape[0], -1)
    else:
        if flatten:
            tensor = tensor.view(tensor.shape[0], -1)
    return tensor


def reshape_observation(
    obs_flat: torch.Tensor, *, obs_mode: str, pixel_shape: tuple[int, int, int] | None
) -> torch.Tensor:
    """Reshape flattened observations back to CNN input shape when needed."""
    if obs_mode != "pixels":
        return obs_flat
    if pixel_shape is None:
        raise ValueError("pixel_shape must be provided for pixel observations")
    return obs_flat.view(obs_flat.shape[0], *pixel_shape)


def _extract_sample(obs_input: Any) -> torch.Tensor:
    data = obs_input
    if isinstance(data, tuple):
        data = data[0]
    if isinstance(data, Mapping):
        for key in POLICY_OBS_KEYS:
            if key in data:
                data = data[key]
                break
        else:
            raise KeyError(f"Unknown observation keys {list(data.keys())}")
    if isinstance(data, torch.Tensor):
        sample = data[0]
    elif isinstance(data, np.ndarray):
        sample = data[0] if data.ndim >= 1 else data
        sample = torch.from_numpy(sample)
    else:
        sample = torch.as_tensor(data)[0]
    return sample


def _to_tensor(obs_input: Any, *, device: torch.device) -> torch.Tensor:
    data = obs_input
    if isinstance(data, tuple):
        data = data[0]
    if isinstance(data, Mapping):
        for key in POLICY_OBS_KEYS:
            if key in data:
                data = data[key]
                break
        else:
            raise KeyError(f"Unknown observation keys {list(data.keys())}")
    tensor: torch.Tensor
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.as_tensor(data)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def _ensure_channels_first(tensor: torch.Tensor, pixel_shape: Sequence[int] | None) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Pixel observations must have 4 dims, got {tensor.shape}")
    if tensor.shape[1] in (1, 3, 4):
        return tensor
    if tensor.shape[-1] in (1, 3, 4):
        return tensor.permute(0, 3, 1, 2).contiguous()
    if pixel_shape is not None:
        return tensor.view(tensor.shape[0], *pixel_shape)
    raise RuntimeError(f"Unable to infer channel axis for pixels: {tensor.shape}")


def _ensure_float_pixels(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if torch.numel(tensor) == 0:
        return tensor
    if torch.any(torch.isfinite(tensor)) and tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor
