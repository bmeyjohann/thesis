#!/usr/bin/env python3
"""Render fixed side views of the controlled bidirectional ramp and stairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import imageio.v3 as iio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unitree_target_terrain import BidirectionalTraversalTerrainCfg


def render(kind: str, output: Path) -> None:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1200
    spec.visual.global_.offheight = 500
    body = spec.worldbody.add_body(name="terrain")
    del body
    spec.worldbody.add_light(
        name="key",
        pos=(6.3, 1.5, 8.0),
        dir=(0.0, 0.2, -1.0),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    BidirectionalTraversalTerrainCfg(
        proportion=1.0,
        size=(16.0, 8.0),
        kind=kind,
        rise=0.4,
        side_length=2.0,
        plateau_length=0.7,
        width=2.0,
        steps_per_side=5,
    ).function(0.0, spec, np.random.default_rng(0))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (6.35, 4.0, 0.25)
    camera.distance = 8.0
    camera.azimuth = 90.0
    camera.elevation = -8.0
    with mujoco.Renderer(model, height=500, width=1200) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, pixels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations/unitree_geometry_calibration_fixed"),
    )
    args = parser.parse_args()
    for kind in ("ramp", "stairs"):
        render(kind, args.output_dir / f"{kind}_side.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
