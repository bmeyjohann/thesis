#!/usr/bin/env python3
"""Render controlled Unitree navigation terrain concepts for experiment selection."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


@dataclass(frozen=True)
class TerrainSpec:
    key: str
    title: str
    question: str
    observation: str
    implementation: str
    recommendation: str


SPECS = {
    "geometry_fork": TerrainSpec(
        key="geometry_fork",
        title="A. Embodiment-dependent geometry",
        question="Can interventions teach route choice when one route requires stepping over gaps?",
        observation="Current height scan is sufficient.",
        implementation="MjLab boxes/stepping stones; no new observation channel.",
        recommendation="Low-risk core benchmark after validating the low-level locomotion policy.",
    ),
    "friction_fork": TerrainSpec(
        key="friction_fork",
        title="B. Geometry-identical friction fork",
        question="Can interventions teach that a visually marked, geometrically flat route is slippery?",
        observation="Requires RGB/material features; height alone cannot distinguish the routes.",
        implementation="Per-geom MuJoCo friction plus a visible material channel or camera encoder.",
        recommendation="Best next scientific benchmark once a material observation is wired in.",
    ),
    "soft_fork": TerrainSpec(
        key="soft_fork",
        title="C. Compliance / drag proxy fork",
        question="Can interventions avoid a route whose dynamics are poor despite flat geometry?",
        observation="Requires RGB/material features or post-contact proprioceptive memory.",
        implementation="Approximate softness using drag, reduced support, or stochastic foothold disturbance.",
        recommendation="High-value extension, but defer until friction works because semantics are harder to defend.",
    ),
    "mixed_fork": TerrainSpec(
        key="mixed_fork",
        title="D. Geometry versus surface trade-off",
        question="Does the learned route reflect controller-specific trade-offs rather than a fixed terrain label?",
        observation="Requires both height/depth and material appearance.",
        implementation="Combine stable stepping stones with a short slippery flat lane.",
        recommendation="Use as the final ablation after separate geometry and friction experiments.",
    ),
}

MATERIALS = {
    0: ("rigid / high friction", "#d9d3c7"),
    1: ("slippery", "#e9b44c"),
    2: ("soft / drag", "#c17cba"),
    3: ("stepping stones", "#61a5c2"),
    4: ("obstacle / void", "#30343f"),
}


def _base_grid(resolution: float = 0.10):
    x = np.arange(-7.0, 7.0 + resolution, resolution)
    y = np.arange(-4.5, 4.5 + resolution, resolution)
    xx, yy = np.meshgrid(x, y)
    height = np.zeros_like(xx)
    material = np.zeros_like(xx, dtype=np.uint8)
    # Central island forces a visible fork while preserving two comparable routes.
    island = (np.abs(xx) <= 2.0) & (np.abs(yy) <= 0.75)
    height[island] = 1.1
    material[island] = 4
    return x, y, xx, yy, height, material


def _lane_mask(xx, yy, upper: bool):
    side = (yy >= 0.85) & (yy <= 3.25) if upper else (yy <= -0.85) & (yy >= -3.25)
    return (np.abs(xx) <= 3.0) & side


def build_terrain(key: str, resolution: float = 0.10):
    x, y, xx, yy, height, material = _base_grid(resolution)
    upper = _lane_mask(xx, yy, True)
    lower = _lane_mask(xx, yy, False)

    if key == "geometry_fork":
        # A chain of safe footholds in the upper route; the lower route stays flat.
        band = upper & (np.abs(yy - 2.0) <= 0.75) & (np.abs(xx) <= 2.8)
        material[band] = 4
        height[band] = -0.35
        for cx in np.linspace(-2.5, 2.5, 7):
            stone = (np.abs(xx - cx) <= 0.34) & (np.abs(yy - 2.0) <= 0.62)
            height[stone] = 0.12
            material[stone] = 3
    elif key == "friction_fork":
        material[lower] = 1
    elif key == "soft_fork":
        material[lower] = 2
    elif key == "mixed_fork":
        material[lower] = 1
        band = upper & (np.abs(yy - 2.0) <= 0.75) & (np.abs(xx) <= 2.8)
        material[band] = 4
        height[band] = -0.25
        for cx in np.linspace(-2.5, 2.5, 8):
            stone = (np.abs(xx - cx) <= 0.38) & (np.abs(yy - 2.0) <= 0.66)
            height[stone] = 0.08
            material[stone] = 3
    else:
        raise KeyError(key)

    start = np.array([-5.8, 0.0])
    goal = np.array([5.8, 0.0])
    upper_path = np.array([start, [-3.0, 0.0], [-2.4, 2.0], [2.4, 2.0], [3.0, 0.0], goal])
    lower_path = upper_path.copy()
    lower_path[:, 1] *= -1
    return x, y, height, material, start, goal, upper_path, lower_path


def _topdown(ax, spec, x, y, height, material, start, goal, upper_path, lower_path):
    cmap = ListedColormap([MATERIALS[i][1] for i in range(len(MATERIALS))])
    ax.imshow(material, origin="lower", extent=(x.min(), x.max(), y.min(), y.max()), cmap=cmap,
              vmin=-0.5, vmax=len(MATERIALS)-0.5, interpolation="nearest")
    ax.contour(x, y, height, levels=[0.05, 0.5], colors=["#111827"], linewidths=0.8)
    ax.plot(upper_path[:, 0], upper_path[:, 1], "--", color="#2563eb", lw=2, label="candidate route A")
    ax.plot(lower_path[:, 0], lower_path[:, 1], "--", color="#dc2626", lw=2, label="candidate route B")
    ax.scatter(*start, s=110, c="#111827", marker="o", zorder=6)
    ax.scatter(*goal, s=170, c="#22c55e", marker="*", edgecolors="#14532d", zorder=6)
    ax.text(start[0], start[1]-0.35, "START", ha="center", va="top", weight="bold")
    ax.text(goal[0], goal[1]-0.35, "GOAL", ha="center", va="top", weight="bold")
    ax.set_title(spec.title, loc="left", weight="bold")
    ax.set(xlabel="world x [m]", ylabel="world y [m]", aspect="equal")
    ax.grid(alpha=0.15)


def _surface(ax, spec, x, y, height, material, start, goal, azim=235.0):
    xx, yy = np.meshgrid(x, y)
    colors = np.empty(height.shape + (4,), dtype=float)
    for idx, (_, color) in MATERIALS.items():
        colors[material == idx] = matplotlib.colors.to_rgba(color)
    z = np.maximum(height, 0.0)
    ax.plot_surface(xx, yy, z, facecolors=colors, linewidth=0, antialiased=False, shade=True)
    ax.scatter(start[0], start[1], 0.18, s=70, c="#111827")
    ax.scatter(goal[0], goal[1], 0.18, s=120, c="#22c55e", marker="*")
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="height [m]", zlim=(0, 1.5))
    ax.set_title(spec.title, loc="left", weight="bold")
    ax.view_init(elev=31, azim=azim)
    try:
        ax.set_box_aspect((14, 9, 3.5))
    except AttributeError:
        pass


def render_one(key: str, out_dir: Path, fps: int, video_seconds: float, make_video: bool):
    spec = SPECS[key]
    data = build_terrain(key)
    x, y, height, material, start, goal, upper_path, lower_path = data
    variant_dir = out_dir / key
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(variant_dir / "terrain_arrays.npz", x=x, y=y, height=height, material=material,
                        start=start, goal=goal, upper_path=upper_path, lower_path=lower_path)
    (variant_dir / "metadata.json").write_text(json.dumps(asdict(spec), indent=2) + "\n", encoding="utf-8")

    fig = plt.figure(figsize=(14, 6.6), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _topdown(ax1, spec, *data)
    _surface(ax2, spec, x, y, height, material, start, goal)
    handles = [Patch(facecolor=color, label=label) for label, color in MATERIALS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(spec.question, fontsize=12)
    fig.savefig(variant_dir / "overview.png", dpi=180)
    plt.close(fig)

    if make_video:
        frames = []
        count = max(24, int(fps * video_seconds))
        for i in range(count):
            fig = plt.figure(figsize=(9.6, 6.0), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")
            _surface(ax, spec, x, y, height, material, start, goal, azim=220 + 360 * i / count)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(frame)
            plt.close(fig)
        imageio.mimsave(variant_dir / "orbit.mp4", frames, fps=fps, macro_block_size=2)
    return variant_dir / "overview.png"


def render_contact_sheet(paths: list[Path], out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for ax, path in zip(axes.flat, paths, strict=True):
        ax.imshow(imageio.imread(path))
        ax.axis("off")
    fig.suptitle("Candidate Unitree terrain benchmarks", fontsize=18, weight="bold")
    fig.savefig(out_dir / "terrain_benchmark_contact_sheet.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=["all", *SPECS], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/unitree_terrain_benchmarks"))
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--video-seconds", type=float, default=4.0)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = list(SPECS) if args.terrain == "all" else [args.terrain]
    paths = [render_one(k, args.output_dir, args.fps, args.video_seconds, not args.no_video) for k in keys]
    if len(paths) == len(SPECS):
        render_contact_sheet(paths, args.output_dir)
    manifest = {"terrains": [asdict(SPECS[k]) for k in keys]}
    (args.output_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
