"""Render seeded Unitree target terrains with MuJoCo's real renderer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unitree_target_terrain import (  # noqa: E402
    build_target_terrain_spec,
    generate_target_terrain,
    layout_statistics,
    set_moving_obstacles,
)


def _camera(layout, *, topdown: bool) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.0)
    if topdown:
        camera.distance = layout.arena_size * 1.08
        camera.azimuth = 90.0
        camera.elevation = -90.0
    else:
        camera.distance = layout.arena_size * 0.70
        camera.azimuth = 132.0
        camera.elevation = -42.0
    return camera


def _annotate(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((18, 18, 650, 92), radius=12, fill=(10, 14, 16, 205))
    draw.text((34, 29), title, fill=(255, 255, 255, 255))
    draw.text((34, 57), subtitle, fill=(220, 226, 230, 255))
    return np.asarray(canvas)


def _render(renderer: mujoco.Renderer, data: mujoco.MjData, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def render_variant(seed: int, preset: str, arena_size: float, output_dir: Path, *, width: int, height: int, video_seconds: float) -> dict[str, object]:
    layout = generate_target_terrain(seed, preset=preset, arena_size=arena_size)
    spec = build_target_terrain_spec(layout)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    variant_dir = output_dir / f"{preset}_seed{seed}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        top = _annotate(
            _render(renderer, data, _camera(layout, topdown=True)),
            f"Target terrain: {preset}, seed {seed}",
            "cyan=start  green=goal  blue=ice  yellow=sand  red=static  magenta=moving",
        )
        perspective = _annotate(
            _render(renderer, data, _camera(layout, topdown=False)),
            f"MuJoCo perspective: {preset}, seed {seed}",
            f"{len(layout.materials)} material + {len(layout.geometry)} geometry brushes; {len(layout.obstacles)} obstacles",
        )
        top_path = variant_dir / "topdown.png"
        perspective_path = variant_dir / "perspective.png"
        imageio.imwrite(top_path, top)
        imageio.imwrite(perspective_path, perspective)

        fps = 20
        frame_count = max(1, int(video_seconds * fps))
        video_path = variant_dir / "moving_obstacles.mp4"
        with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=7) as writer:
            for frame in range(frame_count):
                time_s = frame / fps
                set_moving_obstacles(model, data, layout, time_s)
                rendered = _annotate(
                    _render(renderer, data, _camera(layout, topdown=False)),
                    f"Moving-obstacle preview: {preset}, seed {seed}",
                    f"simulation time {time_s:4.1f}s",
                )
                writer.append_data(rendered)

    metadata_path = variant_dir / "layout.json"
    metadata_path.write_text(json.dumps(layout.to_dict(), indent=2), encoding="utf-8")
    return {
        "seed": seed,
        "preset": preset,
        "topdown": str(top_path),
        "perspective": str(perspective_path),
        "video": str(video_path),
        "metadata": str(metadata_path),
        "model": {"ngeom": int(model.ngeom), "nbody": int(model.nbody), "nmocap": int(model.nmocap)},
        "coverage": layout_statistics(layout),
    }


def _contact_sheet(results: list[dict[str, object]], output_path: Path) -> None:
    images = [Image.open(item["topdown"]).convert("RGB") for item in results]
    thumb_w = 640
    thumb_h = int(images[0].height * thumb_w / images[0].width)
    cols = 2
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (18, 21, 22))
    for index, image in enumerate(images):
        image.thumbnail((thumb_w, thumb_h))
        sheet.paste(image, ((index % cols) * thumb_w, (index // cols) * thumb_h))
    sheet.save(output_path)
    for image in images:
        image.close()


def _write_gallery(results: list[dict[str, object]], output_dir: Path) -> None:
    cards = []
    for item in results:
        variant = f"{item['preset']}_seed{item['seed']}"
        coverage = item["coverage"]
        cards.append(
            f"""
            <article class="card">
              <header><h2>{item['preset']} <span>seed {item['seed']}</span></h2></header>
              <img src="{variant}/topdown.png" alt="Top-down MuJoCo render for {variant}">
              <video controls preload="metadata" src="{variant}/moving_obstacles.mp4"></video>
              <dl>
                <dt>ice</dt><dd>{100 * coverage['ice_fraction']:.1f}%</dd>
                <dt>sand</dt><dd>{100 * coverage['sand_fraction']:.1f}%</dd>
                <dt>geometry</dt><dd>{100 * coverage['geometry_fraction']:.1f}%</dd>
                <dt>crossed layers</dt><dd>{100 * coverage['material_geometry_overlap_fraction']:.1f}%</dd>
                <dt>obstacles</dt><dd>{coverage['static_obstacles']} static + {coverage['moving_obstacles']} moving</dd>
              </dl>
              <nav><a href="{variant}/perspective.png">perspective</a><a href="{variant}/layout.json">layout JSON</a></nav>
            </article>
            """
        )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Unitree target terrain variants</title>
<style>
:root{{--ink:#18211e;--paper:#ede9dd;--card:#faf7ed;--accent:#d84b2f;--line:#c5bfaf}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(135deg,#d8e2d4,var(--paper) 45%,#e9d5b8);color:var(--ink);font-family:"Aptos","Trebuchet MS",sans-serif}}
main{{max-width:1500px;margin:auto;padding:42px}} h1{{font:700 clamp(34px,5vw,72px)/.95 Georgia,serif;margin:0 0 12px;letter-spacing:-.04em}} .intro{{max-width:850px;font-size:18px;margin:0 0 34px}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:30px}} .legend b{{padding:8px 12px;border:1px solid var(--line);border-radius:999px;background:#ffffff99}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px}} .card{{background:var(--card);border:1px solid var(--line);box-shadow:0 12px 40px #2f3b3218;padding:16px}}
h2{{font:700 28px Georgia,serif;margin:3px 0 14px;text-transform:capitalize}} h2 span{{font:500 14px "Aptos",sans-serif;color:#68706b}}
img,video{{display:block;width:100%;background:#101312}} video{{margin-top:10px}} dl{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:15px 0}} dt{{font-size:11px;text-transform:uppercase;color:#6e746f}} dd{{margin:2px 0 0;font-weight:700}} nav{{display:flex;gap:18px}} a{{color:var(--accent);font-weight:700}}
@media(max-width:600px){{main{{padding:22px}}.grid{{grid-template-columns:1fr}}dl{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main><h1>One arena. Crossed terrain semantics.</h1>
<p class="intro">Actual MuJoCo renders of deterministic continuous target terrains. Geometry and material brushes are independent, enabling combinations such as rough ice and sandy stairs.</p>
<div class="legend"><b>cyan start</b><b>green goal</b><b>blue ice</b><b>yellow sand</b><b>red static obstacles</b><b>magenta moving obstacles</b></div>
<section class="grid">{''.join(cards)}</section></main></body></html>"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 29])
    parser.add_argument("--presets", nargs="+", default=["balanced", "traversal", "navigation"])
    parser.add_argument("--arena-size", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--video-seconds", type=float, default=4.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "visualizations" / "unitree_target_terrain_variants",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for preset in args.presets:
        for seed in args.seeds:
            results.append(
                render_variant(
                    seed,
                    preset,
                    args.arena_size,
                    args.output_dir,
                    width=args.width,
                    height=args.height,
                    video_seconds=args.video_seconds,
                )
            )
    _contact_sheet(results, args.output_dir / "target_terrain_contact_sheet.png")
    _write_gallery(results, args.output_dir)
    manifest = {"arena_size": args.arena_size, "variants": results}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
