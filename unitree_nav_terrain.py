from __future__ import annotations

import types
import uuid

import mujoco
import numpy as np


def _strict_discrete_obstacles_function(self, difficulty, spec, rng):
    """Generate obstacles without clipping their dimensions at boundaries/platforms."""
    from mjlab.terrains.heightfield_terrains import _compute_flat_patches, color_by_height
    from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

    body = spec.body("terrain")
    if self.border_width > 0 and self.border_width < self.horizontal_scale:
        raise ValueError(
            f"Border width ({self.border_width}) must be >= horizontal scale "
            f"({self.horizontal_scale})"
        )

    obs_height = self.obstacle_height_range[0] + difficulty * (
        self.obstacle_height_range[1] - self.obstacle_height_range[0]
    )
    border_pixels = int(self.border_width / self.horizontal_scale)
    width_pixels = int(self.size[0] / self.horizontal_scale)
    length_pixels = int(self.size[1] / self.horizontal_scale)
    inner_width = width_pixels - 2 * border_pixels
    inner_length = length_pixels - 2 * border_pixels
    obs_h = int(obs_height / self.vertical_scale)
    obs_width_min = max(1, int(np.ceil(self.obstacle_width_range[0] / self.horizontal_scale)))
    obs_width_max = max(obs_width_min, int(self.obstacle_width_range[1] / self.horizontal_scale))
    platform_pixels = int(self.platform_width / self.horizontal_scale)
    noise = np.zeros((inner_width, inner_length), dtype=np.int16)

    width_choices = np.arange(obs_width_min, obs_width_max + 1, 4)
    if width_choices.size == 0:
        width_choices = np.array([obs_width_min])
    cx, cy = inner_width // 2, inner_length // 2
    half_pf = platform_pixels // 2
    platform = (max(cx - half_pf, 0), min(cx + half_pf, inner_width), max(cy - half_pf, 0), min(cy + half_pf, inner_length))

    for _ in range(self.num_obstacles):
        h = (
            rng.choice(np.array([-obs_h, -obs_h // 2, obs_h // 2, obs_h]))
            if self.obstacle_height_mode == "choice"
            else obs_h
        )
        placed = False
        for _attempt in range(64):
            w = int(rng.choice(width_choices))
            length = w if self.square_obstacles else int(rng.choice(width_choices))
            max_x = inner_width - w
            max_y = inner_length - length
            if max_x < 0 or max_y < 0:
                break
            x_candidates = np.arange(0, max_x + 1, 4)
            y_candidates = np.arange(0, max_y + 1, 4)
            if x_candidates.size == 0 or y_candidates.size == 0:
                break
            x, y = int(rng.choice(x_candidates)), int(rng.choice(y_candidates))
            x1, y1 = x + w, y + length
            px0, px1, py0, py1 = platform
            intersects_platform = x < px1 and x1 > px0 and y < py1 and y1 > py0
            if intersects_platform:
                continue
            noise[x:x1, y:y1] = h
            placed = True
            break
        if not placed:
            continue

    if border_pixels > 0:
        outer = np.zeros((width_pixels, length_pixels), dtype=np.int16)
        outer[border_pixels : border_pixels + inner_width, border_pixels : border_pixels + inner_length] = noise
        noise = outer

    elevation_min = np.min(noise)
    elevation_max = np.max(noise)
    elevation_range = elevation_max - elevation_min if elevation_max != elevation_min else 1
    max_physical_height = elevation_range * self.vertical_scale
    base_thickness = max_physical_height * self.base_thickness_ratio
    normalized = (noise - elevation_min) / elevation_range
    unique_id = uuid.uuid4().hex
    field = spec.add_hfield(
        name=f"hfield_{unique_id}",
        size=[self.size[0] / 2, self.size[1] / 2, max_physical_height, base_thickness],
        nrow=noise.shape[0],
        ncol=noise.shape[1],
        userdata=normalized.flatten().astype(np.float32).tolist(),
    )
    hfield_z_offset = elevation_min * self.vertical_scale if self.obstacle_height_mode == "choice" else 0
    material_name = color_by_height(spec, noise, unique_id, normalized)
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=field.name,
        pos=[self.size[0] / 2, self.size[1] / 2, hfield_z_offset],
        material=material_name,
    )
    origin = np.array([self.size[0] / 2, self.size[1] / 2, self.origin_z_offset])
    flat_patches = _compute_flat_patches(
        noise,
        self.vertical_scale,
        self.horizontal_scale,
        hfield_z_offset,
        self.flat_patch_sampling,
        rng,
    )
    return TerrainOutput(
        origin=origin,
        geometries=[TerrainGeometry(geom=geom, hfield=field)],
        flat_patches=flat_patches,
    )


def enable_strict_min_size_obstacles(obstacle_cfg) -> None:
    obstacle_cfg.function = types.MethodType(_strict_discrete_obstacles_function, obstacle_cfg)

