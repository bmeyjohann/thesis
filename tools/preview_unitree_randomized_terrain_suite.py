#!/usr/bin/env python3
"""Generate seedable geometry/material terrain distributions for Unitree navigation."""

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
class Material:
    name: str
    color: str
    friction: float
    drag: float = 0.0


MATERIALS = {
    0: Material("rigid", "#d8d2c4", 1.00),
    1: Material("sand proxy", "#d9a441", 0.45, 0.25),
    2: Material("slippery", "#4ea8de", 0.12),
    3: Material("soft / drag", "#c77dba", 0.65, 0.70),
    4: Material("gravel proxy", "#8d99ae", 0.70, 0.15),
}

FAMILIES = {
    "isaac_geometry_mix": "Isaac-style geometry mix",
    "material_mosaic": "Visual material mosaic",
    "combined_world": "Combined geometry and material world",
}


def _segment_distance(xx, yy, a, b):
    vx, vy = b - a
    denom = max(float(vx * vx + vy * vy), 1e-9)
    t = np.clip(((xx - a[0]) * vx + (yy - a[1]) * vy) / denom, 0.0, 1.0)
    return np.hypot(xx - (a[0] + t * vx), yy - (a[1] + t * vy))


def _route_mask(xx, yy, route, radius):
    out = np.zeros_like(xx, dtype=bool)
    for a, b in zip(route[:-1], route[1:]):
        out |= _segment_distance(xx, yy, a, b) <= radius
    return out


def _rect(xx, yy, cx, cy, sx, sy):
    return (np.abs(xx - cx) <= sx / 2) & (np.abs(yy - cy) <= sy / 2)


def _ellipse(xx, yy, cx, cy, rx, ry):
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def generate(family: str, seed: int, resolution: float = 0.10):
    rng = np.random.default_rng(seed)
    x = np.arange(-8.0, 8.0 + resolution, resolution)
    y = np.arange(-6.0, 6.0 + resolution, resolution)
    xx, yy = np.meshgrid(x, y)
    height = np.zeros_like(xx)
    material = np.zeros_like(xx, dtype=np.uint8)
    feature = np.zeros_like(xx, dtype=np.uint8)

    start = np.array([-7.0, rng.uniform(-3.0, 3.0)])
    goal = np.array([7.0, rng.uniform(-3.0, 3.0)])
    route = np.array([
        start,
        [-3.5, rng.uniform(-4.2, 4.2)],
        [0.0, rng.uniform(-4.2, 4.2)],
        [3.5, rng.uniform(-4.2, 4.2)],
        goal,
    ], dtype=float)
    route_clear = _route_mask(xx, yy, route, radius=0.55)
    endpoint_clear = (np.hypot(xx-start[0], yy-start[1]) < 1.1) | (np.hypot(xx-goal[0], yy-goal[1]) < 1.1)

    include_geometry = family in {"isaac_geometry_mix", "combined_world"}
    include_material = family in {"material_mosaic", "combined_world"}

    if include_geometry:
        # Discrete boxes: equivalent experimental role to Isaac Lab discrete-obstacle/random-box terrains.
        for _ in range(rng.integers(10, 18)):
            cx, cy = rng.uniform(-6.2, 6.2), rng.uniform(-4.7, 4.7)
            sx, sy = rng.uniform(0.7, 2.0), rng.uniform(0.6, 1.7)
            mask = _rect(xx, yy, cx, cy, sx, sy) & ~endpoint_clear
            height[mask] = np.maximum(height[mask], rng.uniform(0.25, 0.85))
            feature[mask] = 1

        # Rough heightfield patches.
        for _ in range(rng.integers(2, 5)):
            cx, cy = rng.uniform(-5.5, 5.5), rng.uniform(-4.2, 4.2)
            rx, ry = rng.uniform(0.9, 2.0), rng.uniform(0.8, 1.7)
            mask = _ellipse(xx, yy, cx, cy, rx, ry) & ~endpoint_clear
            noise = rng.normal(0.05, 0.025, size=height.shape)
            height[mask] = np.maximum(height[mask], np.clip(noise[mask], 0.01, 0.12))
            feature[mask] = np.maximum(feature[mask], 2)

        # One randomized stairs or stepping-stone/gap section per world.
        cx, cy = rng.uniform(-3.5, 3.5), rng.uniform(-3.4, 3.4)
        if rng.random() < 0.5:
            band = _rect(xx, yy, cx, cy, rng.uniform(2.5, 4.5), rng.uniform(1.3, 2.2)) & ~endpoint_clear
            levels = np.floor((xx - (cx - 2.0)) / rng.uniform(0.35, 0.6))
            height[band] = np.maximum(height[band], np.clip(levels[band], 0, 5) * rng.uniform(0.04, 0.10))
            feature[band] = 3
        else:
            sx, sy = rng.uniform(3.0, 4.5), rng.uniform(1.4, 2.3)
            band = _rect(xx, yy, cx, cy, sx, sy) & ~endpoint_clear
            height[band] = -0.35
            feature[band] = 4
            spacing = rng.uniform(0.65, 0.95)
            for px in np.arange(cx-sx/2+0.35, cx+sx/2, spacing):
                for py in np.arange(cy-sy/2+0.35, cy+sy/2, spacing):
                    stone = _rect(xx, yy, px+rng.uniform(-0.10,0.10), py+rng.uniform(-0.10,0.10), 0.48, 0.48)
                    height[stone & band] = rng.uniform(0.02, 0.12)
                    feature[stone & band] = 5

        # Force one obstacle across the direct start-goal corridor so each world is relevant.
        direct_t = rng.uniform(0.38, 0.62)
        direct_xy = start + direct_t * (goal - start)
        forced = _rect(
            xx,
            yy,
            direct_xy[0] + rng.uniform(-0.15, 0.15),
            direct_xy[1] + rng.uniform(-0.15, 0.15),
            rng.uniform(1.2, 2.0),
            rng.uniform(1.4, 2.4),
        ) & ~endpoint_clear
        height[forced] = np.maximum(height[forced], rng.uniform(0.45, 0.85))
        feature[forced] = 1

        # Guarantee at least one feasible route without making it visible to the policy.
        height[route_clear] = np.minimum(height[route_clear], 0.06)
        feature[route_clear & (height >= 0)] = np.minimum(feature[route_clear & (height >= 0)], 2)

    if include_material:
        for _ in range(rng.integers(6, 11)):
            mat_id = int(rng.choice([1, 2, 3, 4], p=[0.30, 0.25, 0.20, 0.25]))
            cx, cy = rng.uniform(-6.0, 6.0), rng.uniform(-4.6, 4.6)
            if rng.random() < 0.5:
                mask = _ellipse(xx, yy, cx, cy, rng.uniform(0.8, 2.2), rng.uniform(0.7, 1.8))
            else:
                mask = _rect(xx, yy, cx, cy, rng.uniform(1.4, 4.0), rng.uniform(1.2, 3.0))
            mask &= ~endpoint_clear
            material[mask] = mat_id

        # Force one visually marked physical-property region onto the direct route.
        direct_t = rng.uniform(0.35, 0.65)
        direct_xy = start + direct_t * (goal - start)
        forced_material = _ellipse(
            xx,
            yy,
            direct_xy[0],
            direct_xy[1],
            rng.uniform(1.0, 1.8),
            rng.uniform(1.0, 1.8),
        ) & ~endpoint_clear
        material[forced_material] = int(rng.choice([1, 2, 3]))

    # Start and goal are always rigid, flat, and clear.
    height[endpoint_clear] = 0.0
    material[endpoint_clear] = 0
    feature[endpoint_clear] = 0
    return dict(family=family, seed=seed, x=x, y=y, height=height, material=material,
                feature=feature, start=start, goal=goal, feasible_route=route)


def _material_colors(material):
    rgba = np.empty(material.shape + (4,), dtype=float)
    for idx, cfg in MATERIALS.items():
        rgba[material == idx] = matplotlib.colors.to_rgba(cfg.color)
    return rgba


def draw_topdown(ax, world, show_route=False):
    x, y = world["x"], world["y"]
    mat_cmap = ListedColormap([MATERIALS[i].color for i in MATERIALS])
    ax.imshow(world["material"], origin="lower", extent=(x.min(),x.max(),y.min(),y.max()),
              cmap=mat_cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax.contourf(x, y, np.maximum(world["height"], 0), levels=[0.08,0.2,0.45,1.2],
                colors=["#a8a29e","#78716c","#292524"], alpha=0.9)
    ax.contourf(x, y, world["height"], levels=[-1,-0.05], colors=["#111827"], alpha=0.95)
    if show_route:
        r=world["feasible_route"]
        ax.plot(r[:,0],r[:,1],"--",color="#16a34a",lw=1.3,alpha=0.75)
    ax.scatter(*world["start"],c="#111827",s=45,zorder=5)
    ax.scatter(*world["goal"],c="#22c55e",edgecolors="#14532d",marker="*",s=90,zorder=5)
    ax.set_title(f"seed {world['seed']}",loc="left",weight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def draw_surface(ax, world, azim=225):
    xx, yy=np.meshgrid(world["x"],world["y"])
    z=np.maximum(world["height"],-0.3)
    colors=_material_colors(world["material"])
    obstacle=world["height"]>0.08
    colors[obstacle]=matplotlib.colors.to_rgba("#343a40")
    colors[world["height"]<0]=matplotlib.colors.to_rgba("#111827")
    ax.plot_surface(xx,yy,z,facecolors=colors,linewidth=0,antialiased=False,shade=True)
    ax.scatter(*world["start"],0.15,c="#111827",s=50)
    ax.scatter(*world["goal"],0.15,c="#22c55e",marker="*",s=90)
    ax.set(zlim=(-0.4,1.1),xlabel="x [m]",ylabel="y [m]",zlabel="z [m]")
    ax.view_init(elev=34,azim=azim)
    try: ax.set_box_aspect((16,12,3))
    except AttributeError: pass


def save_world(world, root: Path):
    out=root/world["family"]/f"seed_{world['seed']}"
    out.mkdir(parents=True,exist_ok=True)
    arrays={k:v for k,v in world.items() if isinstance(v,np.ndarray)}
    np.savez_compressed(out/"terrain_arrays.npz",**arrays)
    stats={
        "family":world["family"],"seed":world["seed"],
        "height_min":float(world["height"].min()),"height_max":float(world["height"].max()),
        "material_fraction":{MATERIALS[i].name:float(np.mean(world["material"]==i)) for i in MATERIALS},
        "materials":{MATERIALS[i].name:asdict(MATERIALS[i]) for i in MATERIALS},
    }
    (out/"metadata.json").write_text(json.dumps(stats,indent=2)+"\n")
    fig=plt.figure(figsize=(13,5.8),constrained_layout=True)
    a=fig.add_subplot(1,2,1); b=fig.add_subplot(1,2,2,projection="3d")
    draw_topdown(a,world,show_route=True); draw_surface(b,world)
    fig.suptitle(f"{FAMILIES[world['family']]} | randomized seed {world['seed']}",weight="bold")
    fig.savefig(out/"overview.png",dpi=170); plt.close(fig)
    return out/"overview.png"


def save_contact_sheet(worlds, root):
    fig,axes=plt.subplots(len(FAMILIES),len(set(w["seed"] for w in worlds)),figsize=(16,11),constrained_layout=True)
    for row,(family,title) in enumerate(FAMILIES.items()):
        subset=[w for w in worlds if w["family"]==family]
        for col,w in enumerate(subset):
            draw_topdown(axes[row,col],w,show_route=False)
            if col==0: axes[row,col].set_ylabel(title,fontsize=11,weight="bold")
    handles=[Patch(facecolor=m.color,label=f"{m.name} (mu={m.friction:g})") for m in MATERIALS.values()]
    handles += [Patch(facecolor="#343a40",label="geometry obstacle"),Patch(facecolor="#111827",label="gap/void")]
    fig.legend(handles=handles,loc="lower center",ncol=4,frameon=False)
    fig.suptitle("Randomized Unitree terrain distributions",fontsize=18,weight="bold")
    fig.savefig(root/"randomized_terrain_contact_sheet.png",dpi=170); plt.close(fig)


def save_family_video(family, worlds, root, fps, seconds_per_seed):
    frames=[]
    count=max(12,int(fps*seconds_per_seed))
    for w in worlds:
        for i in range(count):
            fig=plt.figure(figsize=(10,6),constrained_layout=True)
            ax=fig.add_subplot(111,projection="3d")
            draw_surface(ax,w,azim=215+100*i/max(1,count-1))
            fig.suptitle(f"{FAMILIES[family]} | seed {w['seed']}",weight="bold")
            fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()); plt.close(fig)
    imageio.mimsave(root/f"{family}_randomized_seeds.mp4",frames,fps=fps,macro_block_size=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=Path("visualizations/unitree_randomized_terrain_suite"))
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,37,51])
    p.add_argument("--family",choices=["all",*FAMILIES],default="all")
    p.add_argument("--fps",type=int,default=10)
    p.add_argument("--seconds-per-seed",type=float,default=1.5)
    p.add_argument("--no-video",action="store_true")
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    families=list(FAMILIES) if args.family=="all" else [args.family]
    worlds=[generate(f,s) for f in families for s in args.seeds]
    for w in worlds: save_world(w,args.output_dir)
    if len(families)==len(FAMILIES): save_contact_sheet(worlds,args.output_dir)
    if not args.no_video:
        for family in families: save_family_video(family,[w for w in worlds if w["family"]==family],args.output_dir,args.fps,args.seconds_per_seed)
    manifest={"source_inspiration":{
        "Isaac Lab terrains":"https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab/isaaclab/terrains",
        "MjLab adapted primitives":"/home/benjamin/miniconda3/envs/fasttd3/lib/python3.10/site-packages/mjlab/terrains",
    },"families":FAMILIES,"seeds":args.seeds,"materials":{i:asdict(m) for i,m in MATERIALS.items()}}
    (args.output_dir/"suite_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(args.output_dir.resolve())


if __name__=="__main__": main()
