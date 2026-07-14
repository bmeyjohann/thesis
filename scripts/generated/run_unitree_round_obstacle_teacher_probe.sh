#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

python - <<'PY'
import copy
import sys
import time

import eval_unitree_nav_baselines as eval_mod


_original_apply_debug = eval_mod._apply_debug_obstacle_overrides


def _make_round_cfg(obstacle_cfg):
    import mjlab.terrains as terrain_gen

    class HfRoundDiscreteObstaclesTerrainCfg(terrain_gen.HfDiscreteObstaclesTerrainCfg):
        """Heightfield obstacle generator with disk masks instead of rectangles."""

        def function(self, difficulty, spec, rng):
            g = terrain_gen.HfDiscreteObstaclesTerrainCfg.function.__globals__
            np = g["np"]
            uuid = g["uuid"]
            mujoco = g["mujoco"]
            color_by_height = g["color_by_height"]
            _compute_flat_patches = g["_compute_flat_patches"]
            TerrainGeometry = g["TerrainGeometry"]
            TerrainOutput = g["TerrainOutput"]

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
            obs_h = int(obs_height / self.vertical_scale)
            obs_width_min = int(self.obstacle_width_range[0] / self.horizontal_scale)
            obs_width_max = int(self.obstacle_width_range[1] / self.horizontal_scale)
            platform_pixels = int(self.platform_width / self.horizontal_scale)

            if border_pixels > 0:
                inner_width_pixels = width_pixels - 2 * border_pixels
                inner_length_pixels = length_pixels - 2 * border_pixels
            else:
                inner_width_pixels = width_pixels
                inner_length_pixels = length_pixels

            noise = np.zeros((inner_width_pixels, inner_length_pixels), dtype=np.int16)
            obs_width_range = np.arange(obs_width_min, obs_width_max + 1, 4)
            if len(obs_width_range) == 0:
                obs_width_range = np.array([obs_width_min])
            x_range = np.arange(0, inner_width_pixels, 4)
            y_range = np.arange(0, inner_length_pixels, 4)

            for _ in range(self.num_obstacles):
                if self.obstacle_height_mode == "choice":
                    h = rng.choice(np.array([-obs_h, -obs_h // 2, obs_h // 2, obs_h]))
                else:
                    h = obs_h
                diameter = int(rng.choice(obs_width_range))
                radius = max(1, diameter // 2)
                if len(x_range) == 0 or len(y_range) == 0:
                    continue
                cx = int(rng.choice(x_range))
                cy = int(rng.choice(y_range))
                x0 = max(cx - radius, 0)
                x1 = min(cx + radius + 1, inner_width_pixels)
                y0 = max(cy - radius, 0)
                y1 = min(cy + radius + 1, inner_length_pixels)
                xx, yy = np.ogrid[x0:x1, y0:y1]
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
                patch = noise[x0:x1, y0:y1]
                patch[mask] = h

            cx = inner_width_pixels // 2
            cy = inner_length_pixels // 2
            half_pf = platform_pixels // 2
            x0 = max(cx - half_pf, 0)
            x1 = min(cx + half_pf, inner_width_pixels)
            y0 = max(cy - half_pf, 0)
            y1 = min(cy + half_pf, inner_length_pixels)
            noise[x0:x1, y0:y1] = 0

            if border_pixels > 0:
                outer_noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)
                outer_noise[
                    border_pixels : border_pixels + inner_width_pixels,
                    border_pixels : border_pixels + inner_length_pixels,
                ] = noise
                noise = outer_noise

            elevation_min = np.min(noise)
            elevation_max = np.max(noise)
            elevation_range = elevation_max - elevation_min if elevation_max != elevation_min else 1
            max_physical_height = elevation_range * self.vertical_scale
            base_thickness = max_physical_height * self.base_thickness_ratio
            if elevation_range > 0:
                normalized_elevation = (noise - elevation_min) / elevation_range
            else:
                normalized_elevation = np.zeros_like(noise)

            unique_id = uuid.uuid4().hex
            field = spec.add_hfield(
                name=f"hfield_{unique_id}",
                size=[self.size[0] / 2, self.size[1] / 2, max_physical_height, base_thickness],
                nrow=noise.shape[0],
                ncol=noise.shape[1],
                userdata=normalized_elevation.flatten().astype(np.float32).tolist(),
            )
            hfield_z_offset = elevation_min * self.vertical_scale if self.obstacle_height_mode == "choice" else 0
            material_name = color_by_height(spec, noise, unique_id, normalized_elevation)
            hfield_geom = body.add_geom(
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
            geom = TerrainGeometry(geom=hfield_geom, hfield=field)
            return TerrainOutput(origin=origin, geometries=[geom], flat_patches=flat_patches)

    return HfRoundDiscreteObstaclesTerrainCfg(
        proportion=obstacle_cfg.proportion,
        size=obstacle_cfg.size,
        flat_patch_sampling=obstacle_cfg.flat_patch_sampling,
        obstacle_height_mode=obstacle_cfg.obstacle_height_mode,
        obstacle_width_range=obstacle_cfg.obstacle_width_range,
        obstacle_height_range=obstacle_cfg.obstacle_height_range,
        num_obstacles=obstacle_cfg.num_obstacles,
        platform_width=obstacle_cfg.platform_width,
        horizontal_scale=obstacle_cfg.horizontal_scale,
        vertical_scale=obstacle_cfg.vertical_scale,
        base_thickness_ratio=obstacle_cfg.base_thickness_ratio,
        border_width=obstacle_cfg.border_width,
        square_obstacles=True,
        origin_z_offset=obstacle_cfg.origin_z_offset,
    )


def _round_apply_debug(args, env_cfg):
    _original_apply_debug(args, env_cfg)
    terrain = getattr(env_cfg.scene, "terrain", None)
    generator = getattr(terrain, "terrain_generator", None) if terrain is not None else None
    if generator is None or "discrete_obstacles" not in getattr(generator, "sub_terrains", {}):
        raise ValueError("round obstacle probe requires discrete_obstacles heightfield terrain")
    env_cfg.scene.terrain.terrain_generator = copy.deepcopy(generator)
    obstacle_cfg = env_cfg.scene.terrain.terrain_generator.sub_terrains["discrete_obstacles"]
    env_cfg.scene.terrain.terrain_generator.sub_terrains["discrete_obstacles"] = _make_round_cfg(obstacle_cfg)


eval_mod._apply_debug_obstacle_overrides = _round_apply_debug

run_name = f"unitree_round_scan_teacher_probe_{time.strftime('%Y%m%d_%H%M%S')}"
sys.argv = [
    "eval_unitree_nav_baselines.py",
    "--controller", "scan_teacher",
    "--task", "Unitree-G1-Nav-Obstacles-Safe-Collision",
    "--device", "cuda:0",
    "--num-envs", "1",
    "--num-episodes", "10",
    "--episode-length-s", "20",
    "--run-name", run_name,
    "--debug-obstacle-width-min", "0.9",
    "--debug-obstacle-width-max", "1.2",
    "--debug-obstacle-height-min", "0.45",
    "--debug-obstacle-height-max", "0.55",
    "--debug-num-obstacles", "12",
    "--debug-platform-width", "2.0",
    "--debug-obstacle-border-width", "0.5",
    "--debug-goal-through-obstacle",
    "--debug-goal-distance", "3.4",
    "--debug-goal-obstacle-min-dist", "0.9",
    "--debug-goal-obstacle-max-dist", "2.2",
    "--min-start-obstacle-clearance", "0.8",
    "--min-goal-obstacle-clearance", "0.9",
    "--require-blocked-corridor",
    "--blocked-corridor-radius", "0.45",
    "--blocked-corridor-ignore-end-radius", "0.75",
    "--blocked-corridor-min-cells", "1",
    "--teacher-scan-block-threshold", "0.12",
    "--teacher-sector-half-width", "0.45",
    "--teacher-align-angle", "0.8",
    "--teacher-max-vx", "0.55",
    "--teacher-max-vy", "0.0",
    "--teacher-yaw-gain", "1.2",
    "--teacher-clearance-weight", "6.0",
    "--teacher-clearance-power", "2.0",
    "--teacher-speed-clearance-scale", "8.0",
    "--teacher-num-sectors", "15",
    "--teacher-min-forward-scale", "0.15",
    "--teacher-rollout-horizon", "1.6",
    "--teacher-rollout-clearance", "0.75",
    "--teacher-rollout-samples", "8",
    "--teacher-rollout-clearance-weight", "25.0",
    "--teacher-rollout-forward-bias", "0.3",
]
raise SystemExit(eval_mod.main())
PY
