"""Shared Unitree torso-camera configuration and inspection utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnitreeVisionCfg:
    mode: str = "rgbd"
    width: int = 320
    height: int = 240
    fovy: float = 70.0
    pitch_down_deg: float = 25.0
    yaw_deg: float = 0.0
    stereo_baseline_m: float = 0.12
    max_depth_m: float = 5.0
    camera_x_m: float = 0.08
    camera_z_m: float = 0.42

    def validate(self) -> None:
        if self.mode not in {"mono_rgb", "stereo_rgb", "rgbd", "stereo_rgbd"}:
            raise ValueError(f"Unsupported vision mode: {self.mode}")
        if min(self.width, self.height) < 1 or self.max_depth_m <= 0:
            raise ValueError("Camera dimensions and max depth must be positive")
        if self.stereo_baseline_m <= 0:
            raise ValueError("Stereo baseline must be positive")


def camera_quat(pitch_down_deg: float, yaw_deg: float = 0.0) -> tuple[float, ...]:
    """Return MuJoCo camera quaternion for forward-facing torso mounting."""
    pitch = math.radians(float(pitch_down_deg))
    yaw = math.radians(float(yaw_deg))
    q_base = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float64)
    q_pitch = np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])
    q_yaw = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])

    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    quat = mul(q_yaw, mul(q_pitch, q_base))
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)


def attach_vision_sensors(env_cfg, cfg: UnitreeVisionCfg) -> tuple[str, ...]:
    """Attach selectable mono/stereo RGB-D sensors to a navigation config."""
    from mjlab.sensor import CameraSensorCfg

    cfg.validate()
    stereo = cfg.mode.startswith("stereo")
    depth = cfg.mode in {"rgbd", "stereo_rgbd"}
    names = ("nav_camera_left", "nav_camera_right") if stereo else ("nav_camera",)
    offsets = (-cfg.stereo_baseline_m / 2, cfg.stereo_baseline_m / 2) if stereo else (0.0,)
    sensors = list(env_cfg.scene.sensors or ())
    for name, lateral in zip(names, offsets):
        data_types = ("rgb", "depth") if depth and name != "nav_camera_right" else ("rgb",)
        sensors.append(CameraSensorCfg(
            name=name,
            parent_body="robot/torso_link",
            pos=(cfg.camera_x_m, lateral, cfg.camera_z_m),
            quat=camera_quat(cfg.pitch_down_deg, cfg.yaw_deg),
            fovy=cfg.fovy,
            width=cfg.width,
            height=cfg.height,
            data_types=data_types,
            use_textures=True,
            use_shadows=False,
            clone_data=True,
        ))
    env_cfg.scene.sensors = tuple(sensors)
    return names


def rgb_numpy(tensor) -> np.ndarray:
    return tensor[0].detach().cpu().numpy().astype(np.uint8)


def depth_colormap(tensor, max_depth_m: float) -> np.ndarray:
    depth = tensor[0, ..., 0].detach().cpu().numpy()
    depth = np.nan_to_num(depth, nan=max_depth_m, posinf=max_depth_m, neginf=0.0)
    proximity = 1.0 - np.clip(depth / max_depth_m, 0.0, 1.0)
    return np.stack(
        (255 * proximity, 255 * np.sqrt(proximity), 255 * (1.0 - proximity)), axis=-1
    ).astype(np.uint8)


def compose_vision_frame(scene, names: tuple[str, ...], cfg: UnitreeVisionCfg) -> np.ndarray:
    panels: list[np.ndarray] = []
    for name in names:
        data = scene.sensors.get(name).data
        if data.rgb is None:
            raise RuntimeError(f"Camera {name} did not produce RGB")
        panels.append(rgb_numpy(data.rgb))
        if data.depth is not None:
            panels.append(depth_colormap(data.depth, cfg.max_depth_m))
    return np.concatenate(panels, axis=1)