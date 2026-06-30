from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from .env import (
    clip_action_to_space,
    extract_agent_forward_xy,
    extract_agent_velocity_xy,
    extract_agent_xy,
    extract_goal_distance,
    extract_goal_xy,
    extract_min_constrained_clearance,
    unwrap_env,
    _task_constrained_object_specs,
)


_FIXED_LAYOUT_PRESETS: dict[str, dict[str, object]] = {
    "car_center_block": {
        "agent_xy": (-1.2, 0.0),
        "agent_yaw": 0.0,
        "goal_xy": (1.2, 0.0),
        "body_xy": {
            "hazard0": (0.0, 0.0),
            "hazard1": (-1.25, 1.25),
            "hazard2": (-1.25, -1.25),
            "hazard3": (1.25, 1.25),
            "hazard4": (1.25, -1.25),
            "hazard5": (0.0, 1.25),
            "hazard6": (0.0, -1.25),
            "hazard7": (1.35, 0.95),
            "vase0": (1.35, 1.35),
        },
    },
    "car_single_block": {
        "agent_xy": (-1.2, 0.0),
        "agent_yaw": 0.0,
        "goal_xy": (1.2, 0.0),
        "body_xy": {
            "hazard0": (0.0, 0.0),
            "hazard1": (-8.0, 8.0),
            "hazard2": (-8.0, -8.0),
            "hazard3": (8.0, 8.0),
            "hazard4": (8.0, -8.0),
            "hazard5": (-9.0, 8.0),
            "hazard6": (-9.0, -8.0),
            "hazard7": (9.0, 8.0),
            "vase0": (9.0, -8.0),
        },
    },
    "point_center_block": {
        "agent_xy": (-1.2, 0.0),
        "agent_yaw": 0.0,
        "goal_xy": (1.2, 0.0),
        "body_xy": {
            "hazard0": (0.0, 0.0),
            "hazard1": (-1.25, 1.25),
            "hazard2": (-1.25, -1.25),
            "hazard3": (1.25, 1.25),
            "hazard4": (1.25, -1.25),
            "hazard5": (0.0, 1.25),
            "hazard6": (0.0, -1.25),
            "hazard7": (1.35, 0.95),
            "vase0": (1.35, 1.35),
        },
    },
    "point_wall_gap": {
        "agent_xy": (-1.2, 0.0),
        "agent_yaw": 0.0,
        "goal_xy": (1.2, 0.0),
        "body_xy": {
            "hazard0": (0.0, -1.05),
            "hazard1": (0.0, -0.75),
            "hazard2": (0.0, -0.45),
            "hazard3": (0.0, -0.15),
            "hazard4": (0.0, 0.15),
            "hazard5": (0.0, 0.45),
            "hazard6": (0.0, 0.75),
            "hazard7": (0.0, 1.05),
            "vase0": (1.35, 1.35),
        },
    },
}


def fixed_layout_preset_names() -> tuple[str, ...]:
    return tuple(sorted(_FIXED_LAYOUT_PRESETS.keys()))


def layout_curriculum_names() -> tuple[str, ...]:
    return ("car_block_bridge", "car_block_progression", "car_random_blocked_filter")


class FrameStackObservationWrapper(gym.Wrapper):
    """Concatenate the last k flat observations for memory-light SAC experiments."""

    def __init__(self, env: gym.Env, *, num_frames: int):
        super().__init__(env)
        self.num_frames = int(max(1, num_frames))
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("FrameStackObservationWrapper requires a Box observation space")
        if len(space.shape) != 1:
            raise ValueError(
                "FrameStackObservationWrapper only supports flat 1D observations; "
                f"got shape={space.shape!r}"
            )
        self._frames: list[np.ndarray] = []
        low = np.tile(np.asarray(space.low, dtype=np.float32), self.num_frames)
        high = np.tile(np.asarray(space.high, dtype=np.float32), self.num_frames)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _stack(self) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("Frame stack is empty; reset must be called before step")
        return np.concatenate(self._frames, axis=0).astype(np.float32, copy=False)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._frames = [obs_arr.copy() for _ in range(self.num_frames)]
        return self._stack(), info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._frames.append(obs_arr.copy())
        if len(self._frames) > self.num_frames:
            self._frames = self._frames[-self.num_frames :]
        return self._stack(), reward, cost, terminated, truncated, info


@dataclass(frozen=True)
class _VisualGeom2D:
    kind: str
    center: np.ndarray
    radius: float = 0.0
    half_extents: np.ndarray | None = None
    axes: np.ndarray | None = None
    name: str = ""


class FootprintCostWrapper(gym.Wrapper):
    """Override cost when the agent footprint reaches constrained obstacles.

    ``visual`` mode uses rendered MuJoCo geom primitives projected into XY.
    ``keepout`` mode uses Safety-Gym's larger semantic keepout radii.
    """

    def __init__(self, env: gym.Env, *, margin: float = 0.0, cost_value: float = 1.0, mode: str = "visual"):
        super().__init__(env)
        self.margin = float(margin)
        self.cost_value = float(max(0.0, cost_value))
        self.mode = str(mode or "visual").strip().lower()
        if self.mode not in {"visual", "keepout"}:
            raise ValueError(f"Unsupported footprint cost mode: {mode!r}")

    def _footprint_cost(self) -> tuple[float, float]:
        if self.mode == "keepout":
            clearance = float(extract_min_constrained_clearance(self.env))
        else:
            clearance = float(self._visual_min_clearance())
        if not np.isfinite(clearance):
            return 0.0, clearance
        active = clearance <= self.margin
        return (self.cost_value if active else 0.0), clearance

    @staticmethod
    def _geom_name(model, geom_id: int) -> str:
        try:
            return str(model.geom(geom_id).name or "")
        except Exception:
            return ""

    @staticmethod
    def _body_name(model, body_id: int) -> str:
        try:
            return str(model.body(body_id).name or "")
        except Exception:
            return ""

    @staticmethod
    def _geom_to_shape(model, data, geom_id: int) -> _VisualGeom2D | None:
        xpos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(-1)
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64).reshape(-1)
        if xpos.size < 2 or size.size < 1 or not np.isfinite(xpos[:2]).all():
            return None

        geom_type = int(model.geom_type[geom_id])
        name = FootprintCostWrapper._geom_name(model, geom_id)
        # MuJoCo types used here: 2=sphere, 5=cylinder, 6=box.
        if geom_type in {2, 5}:
            return _VisualGeom2D(kind="circle", center=xpos[:2].copy(), radius=float(max(size[0], 0.0)), name=name)
        if geom_type == 6:
            half = np.asarray(
                [
                    float(max(size[0], 0.0)),
                    float(max(size[1] if size.size > 1 else size[0], 0.0)),
                ],
                dtype=np.float64,
            )
            xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            axes = np.stack([xmat[:2, 0], xmat[:2, 1]], axis=0)
            for idx in range(2):
                norm = float(np.linalg.norm(axes[idx]))
                axes[idx] = axes[idx] / max(norm, 1e-9)
            return _VisualGeom2D(kind="rect", center=xpos[:2].copy(), half_extents=half, axes=axes, name=name)
        return None

    @staticmethod
    def _circle_circle_clearance(a: _VisualGeom2D, b: _VisualGeom2D) -> float:
        return float(np.linalg.norm(a.center - b.center) - (a.radius + b.radius))

    @staticmethod
    def _rect_circle_clearance(rect: _VisualGeom2D, circ: _VisualGeom2D) -> float:
        assert rect.axes is not None and rect.half_extents is not None
        local = rect.axes @ (circ.center - rect.center)
        delta = np.abs(local) - rect.half_extents
        outside = np.maximum(delta, 0.0)
        outside_dist = float(np.linalg.norm(outside))
        if outside_dist > 0.0:
            return outside_dist - circ.radius
        inside_depth = float(np.min(rect.half_extents - np.abs(local)))
        return -inside_depth - circ.radius

    @staticmethod
    def _rect_rect_clearance(a: _VisualGeom2D, b: _VisualGeom2D) -> float:
        assert a.axes is not None and b.axes is not None and a.half_extents is not None and b.half_extents is not None
        min_overlap = float("inf")
        max_gap = -float("inf")
        for axis in (a.axes[0], a.axes[1], b.axes[0], b.axes[1]):
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-9:
                continue
            u = axis / norm
            ra = float(a.half_extents[0] * abs(np.dot(u, a.axes[0])) + a.half_extents[1] * abs(np.dot(u, a.axes[1])))
            rb = float(b.half_extents[0] * abs(np.dot(u, b.axes[0])) + b.half_extents[1] * abs(np.dot(u, b.axes[1])))
            sep = abs(float(np.dot(u, b.center - a.center)))
            gap = sep - (ra + rb)
            max_gap = max(max_gap, gap)
            min_overlap = min(min_overlap, -gap)
            if gap > 0.0:
                return gap
        if np.isfinite(min_overlap):
            return -min_overlap
        return max_gap

    @staticmethod
    def _shape_clearance(a: _VisualGeom2D, b: _VisualGeom2D) -> float:
        if a.kind == "circle" and b.kind == "circle":
            return FootprintCostWrapper._circle_circle_clearance(a, b)
        if a.kind == "rect" and b.kind == "circle":
            return FootprintCostWrapper._rect_circle_clearance(a, b)
        if a.kind == "circle" and b.kind == "rect":
            return FootprintCostWrapper._rect_circle_clearance(b, a)
        if a.kind == "rect" and b.kind == "rect":
            return FootprintCostWrapper._rect_rect_clearance(a, b)
        return float("inf")

    def _visual_min_clearance(self) -> float:
        base = unwrap_env(self.env)
        task = getattr(base, "task", None)
        model = getattr(task, "model", None)
        data = getattr(task, "data", None)
        if task is None or model is None or data is None:
            return float("nan")

        constrained_bodies = {str(spec.get("body_name", "")) for spec in _task_constrained_object_specs(task)}
        constrained_bodies.discard("")
        agent_shapes: list[_VisualGeom2D] = []
        obstacle_shapes: list[_VisualGeom2D] = []

        for geom_id in range(int(getattr(model, "ngeom", 0))):
            body_id = int(model.geom_bodyid[geom_id])
            body_name = self._body_name(model, body_id)
            geom_name = self._geom_name(model, geom_id)
            shape = self._geom_to_shape(model, data, geom_id)
            if shape is None:
                continue
            if body_name in {"agent", "left", "right", "rear"} or geom_name in {
                "agent",
                "back_bumper",
                "back_connector",
                "front_bumper",
                "front_connector",
            }:
                agent_shapes.append(shape)
            elif body_name in constrained_bodies:
                obstacle_shapes.append(shape)

        if not agent_shapes or not obstacle_shapes:
            return float("nan")

        min_clearance = float("inf")
        for agent_shape in agent_shapes:
            for obstacle_shape in obstacle_shapes:
                clearance = self._shape_clearance(agent_shape, obstacle_shape)
                if clearance < min_clearance:
                    min_clearance = clearance
        return float(min_clearance) if np.isfinite(min_clearance) else float("nan")

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        native_cost = float(cost)
        footprint_cost, clearance = self._footprint_cost()
        effective_cost = max(native_cost, float(footprint_cost))
        info["native_cost"] = native_cost
        info["footprint_cost"] = float(footprint_cost)
        info["footprint_cost_active"] = 1.0 if footprint_cost > 0.0 else 0.0
        info["footprint_cost_mode"] = self.mode
        info["footprint_cost_margin"] = float(self.margin)
        info["footprint_clearance"] = float(clearance)
        return obs, reward, effective_cost, terminated, truncated, info


class FixedSafetyLayoutWrapper(gym.Wrapper):
    """Apply a deterministic Safety-Gym layout after every reset."""

    def __init__(self, env: gym.Env, *, preset: str = "none"):
        super().__init__(env)
        self.preset = str(preset or "none").strip().lower()
        if self.preset in {"", "none"}:
            self._spec: dict[str, object] | None = None
        else:
            if self.preset not in _FIXED_LAYOUT_PRESETS:
                known = ", ".join(fixed_layout_preset_names())
                raise ValueError(f"Unknown fixed layout preset: {preset!r}. Known presets: {known}")
            self._spec = _FIXED_LAYOUT_PRESETS[self.preset]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._spec is None:
            return obs, info
        task = self.unwrapped.task
        self._apply_layout(task)
        obs = np.asarray(task.obs(), dtype=np.float32).reshape(-1)
        info = dict(info)
        info["fixed_layout_preset"] = self.preset
        return obs, info

    def _apply_layout(self, task) -> None:
        import mujoco

        body_xy = dict(self._spec.get("body_xy", {})) if self._spec is not None else {}
        goal_xy = self._spec.get("goal_xy") if self._spec is not None else None
        if goal_xy is not None:
            body_xy["goal"] = tuple(goal_xy)
        for name, xy_raw in body_xy.items():
            xy = np.asarray(xy_raw, dtype=np.float64).reshape(2)
            if name in task.world_info.layout:
                task.world_info.layout[name] = xy.copy()
            try:
                qposadr = int(task.model.joint(name).qposadr)
                if qposadr >= 0 and qposadr + 2 <= int(task.data.qpos.shape[0]):
                    task.data.qpos[qposadr : qposadr + 2] = xy[:2]
            except KeyError:
                pass
            try:
                task.model.body(name).pos[:2] = xy[:2]
            except KeyError:
                continue
            if name == "goal":
                task.goal.pos[:2] = xy[:2]
                task.world_info.world_config_dict["geoms"]["goal"]["pos"][:2] = xy[:2]

        agent_xy = self._spec.get("agent_xy") if self._spec is not None else None
        if agent_xy is not None:
            xy = np.asarray(agent_xy, dtype=np.float64).reshape(2)
            task.world_info.layout["agent"] = xy.copy()
            yaw = float(self._spec.get("agent_yaw", 0.0))
            try:
                qposadr = int(task.model.joint("agent").qposadr)
                if qposadr >= 0 and qposadr + 7 <= int(task.data.qpos.shape[0]):
                    z = float(task.data.qpos[qposadr + 2])
                    task.data.qpos[qposadr : qposadr + 3] = (float(xy[0]), float(xy[1]), z)
                    task.data.qpos[qposadr + 3 : qposadr + 7] = (
                        math.cos(0.5 * yaw),
                        0.0,
                        0.0,
                        math.sin(0.5 * yaw),
                    )
                    task.data.qvel[:] = 0.0
                    mujoco.mj_forward(task.model, task.data)
                    return
            except KeyError:
                pass
            try:
                task.model.body("agent").pos[:2] = xy[:2]
                # SafetyPoint uses slide joints named x/y and a z hinge. Keep qpos x/y at
                # zero because body("agent").pos carries the fixed reset location.
                if task.model.joint("x").qposadr >= 0 and task.model.joint("y").qposadr >= 0:
                    task.data.qpos[int(task.model.joint("x").qposadr)] = 0.0
                    task.data.qpos[int(task.model.joint("y").qposadr)] = 0.0
                task.data.qpos[int(task.model.joint("z").qposadr)] = yaw
                task.data.qvel[:] = 0.0
            except KeyError:
                pass

        mujoco.mj_forward(task.model, task.data)


class SafetyLayoutCurriculumWrapper(gym.Wrapper):
    """Sample controlled Safety-Gym layouts for staged obstacle-avoidance curricula."""

    def __init__(self, env: gym.Env, *, curriculum: str, level: int = 0):
        super().__init__(env)
        self.curriculum = str(curriculum or "none").strip().lower()
        self.level = int(max(0, level))
        if self.curriculum not in layout_curriculum_names():
            known = ", ".join(layout_curriculum_names())
            raise ValueError(f"Unknown layout curriculum: {curriculum!r}. Known curricula: {known}")
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        if self.curriculum == "car_random_blocked_filter":
            return self._reset_random_blocked_filter(**kwargs)
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(int(seed) + 9973 * (self.level + 1))
        obs, info = self.env.reset(**kwargs)
        task = self.unwrapped.task
        if self.curriculum == "car_block_bridge":
            spec = self._sample_car_block_bridge()
        else:
            spec = self._sample_car_block_progression()
        FixedSafetyLayoutWrapper._apply_layout_from_spec(task, spec)
        obs = np.asarray(task.obs(), dtype=np.float32).reshape(-1)
        info = dict(info)
        info["layout_curriculum"] = self.curriculum
        info["layout_curriculum_level"] = int(self.level)
        info["layout_agent_xy"] = tuple(float(x) for x in spec["agent_xy"])
        info["layout_goal_xy"] = tuple(float(x) for x in spec["goal_xy"])
        return obs, info

    def _reset_random_blocked_filter(self, **kwargs):
        seed = kwargs.get("seed")
        best = None
        best_score = -float("inf")
        max_attempts = 20
        base_seed = int(seed) if seed is not None else None
        for attempt in range(max_attempts):
            reset_kwargs = dict(kwargs)
            if base_seed is not None:
                # Keep filter attempts for adjacent eval episodes from overlapping.
                # `seed + attempt` made episode S attempt 1 identical to episode S+1
                # attempt 0, which inflated repeated-layout eval metrics.
                reset_kwargs["seed"] = int(base_seed * max_attempts + attempt)
            obs, info = self.env.reset(**reset_kwargs)
            task = self.unwrapped.task
            diag = self._random_layout_diagnostics(task)
            score = float(diag.get("challenge_obstacles_near_corridor", 0.0)) - max(
                0.0, float(diag.get("challenge_min_line_clearance", 1.0))
            )
            if score > best_score:
                best = (obs, dict(info), diag, attempt)
                best_score = score
            if (
                float(diag.get("challenge_direct_path_blocked", 0.0)) >= 1.0
                and float(diag.get("challenge_obstacles_near_corridor", 0.0)) >= 1.0
            ):
                info = dict(info)
                info.update(diag)
                info["layout_curriculum"] = self.curriculum
                info["layout_curriculum_level"] = int(self.level)
                info["layout_filter_attempts"] = int(attempt + 1)
                if base_seed is not None:
                    info["layout_filter_base_seed"] = int(base_seed)
                    info["layout_filter_selected_seed"] = int(base_seed * max_attempts + attempt)
                return obs, info
        if best is None:
            return self.env.reset(**kwargs)
        obs, info, diag, attempt = best
        info.update(diag)
        info["layout_curriculum"] = self.curriculum
        info["layout_curriculum_level"] = int(self.level)
        info["layout_filter_attempts"] = int(attempt + 1)
        if base_seed is not None:
            info["layout_filter_base_seed"] = int(base_seed)
            info["layout_filter_selected_seed"] = int(base_seed * max_attempts + attempt)
        return obs, info

    def _random_layout_diagnostics(self, task) -> dict[str, float]:
        agent_xy = extract_agent_xy(self.env)
        goal_xy = extract_goal_xy(self.env)
        if agent_xy is None or goal_xy is None:
            return {
                "challenge_start_goal_distance": float("nan"),
                "challenge_min_line_clearance": float("nan"),
                "challenge_obstacles_near_corridor": 0.0,
                "challenge_direct_path_blocked": 0.0,
            }
        agent = np.asarray(agent_xy, dtype=np.float64).reshape(2)
        goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        seg = goal - agent
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6 or not np.isfinite(seg_len):
            return {
                "challenge_start_goal_distance": float(seg_len),
                "challenge_min_line_clearance": float("nan"),
                "challenge_obstacles_near_corridor": 0.0,
                "challenge_direct_path_blocked": 0.0,
            }
        unit = seg / seg_len
        world_cfg = getattr(getattr(task, "world_info", None), "world_config_dict", None)
        min_clearance = float("inf")
        near_count = 0
        if isinstance(world_cfg, dict):
            for section in ("geoms", "free_geoms"):
                items = world_cfg.get(section, {})
                if not isinstance(items, dict):
                    continue
                for name, cfg in items.items():
                    label = str(name).lower()
                    if not any(key in label for key in ("hazard", "vase", "pillar", "gremlin", "wall")):
                        continue
                    try:
                        pos = np.asarray(task.data.body(str(name)).xpos[:2], dtype=np.float64).reshape(2)
                    except Exception:
                        continue
                    geom_list = cfg.get("geoms", []) if isinstance(cfg, dict) else []
                    geom = geom_list[0] if geom_list else {}
                    size = np.asarray(geom.get("size", [0.1]), dtype=np.float64).reshape(-1)
                    radius = float(np.linalg.norm(size[:2])) if str(geom.get("type", "")).lower() == "box" and size.size >= 2 else float(size[0] if size.size else 0.1)
                    rel = pos - agent
                    t = float(np.clip(np.dot(rel, unit) / seg_len, 0.0, 1.0))
                    closest = agent + t * seg
                    clearance = float(np.linalg.norm(pos - closest) - radius)
                    min_clearance = min(min_clearance, clearance)
                    if clearance < 0.25 and 0.05 < t < 0.95:
                        near_count += 1
        if not np.isfinite(min_clearance):
            min_clearance = float("nan")
        return {
            "challenge_start_goal_distance": float(seg_len),
            "challenge_min_line_clearance": float(min_clearance),
            "challenge_obstacles_near_corridor": float(near_count),
            "challenge_direct_path_blocked": 1.0 if np.isfinite(min_clearance) and min_clearance < 0.05 else 0.0,
        }

    def _uniform(self, lo: float, hi: float) -> float:
        return float(self._rng.uniform(float(lo), float(hi)))

    def _sample_car_block_progression(self) -> dict[str, object]:
        level = int(self.level)
        agent_x = -1.2
        goal_x = 1.2
        agent_y = 0.0
        goal_y = 0.0
        agent_yaw = 0.0
        main_hazard = np.asarray((0.0, 0.0), dtype=np.float64)

        if level >= 1:
            yaw_span = math.pi if level >= 2 else math.radians(45.0)
            agent_yaw = self._uniform(-yaw_span, yaw_span)
        if level >= 2:
            jitter = 0.25 if level == 2 else 0.45
            agent_y += self._uniform(-jitter, jitter)
        if level >= 3:
            goal_jitter = 0.35 if level == 3 else 0.65
            goal_y += self._uniform(-goal_jitter, goal_jitter)
            main_hazard[1] += self._uniform(-0.25, 0.25)
        if level >= 5:
            agent_x = self._uniform(-1.55, -0.95)
            goal_x = self._uniform(0.95, 1.55)
            agent_y = self._uniform(-0.8, 0.8)
            goal_y = self._uniform(-0.8, 0.8)
            main_hazard = self._sample_blocking_hazard((agent_x, agent_y), (goal_x, goal_y))

        body_xy: dict[str, tuple[float, float]] = {
            "hazard0": (float(main_hazard[0]), float(main_hazard[1])),
            "hazard1": (-1.35, 1.35),
            "hazard2": (-1.35, -1.35),
            "hazard3": (1.35, 1.35),
            "hazard4": (1.35, -1.35),
            "hazard5": (0.0, 1.35),
            "hazard6": (0.0, -1.35),
            "hazard7": (1.45, 0.95),
            "vase0": (1.45, 1.45),
        }
        if level >= 4:
            # Add a second near-path obstacle on the opposite side to force a slalom,
            # while leaving a visible route around the pair.
            sign = -1.0 if float(main_hazard[1]) >= 0.0 else 1.0
            body_xy["hazard1"] = (0.45, float(sign * self._uniform(0.22, 0.45)))
        if level >= 5:
            for name in ("hazard2", "hazard3"):
                body_xy[name] = (self._uniform(-1.1, 1.1), self._uniform(-1.1, 1.1))

        return {
            "agent_xy": (float(agent_x), float(agent_y)),
            "agent_yaw": float(agent_yaw),
            "goal_xy": (float(goal_x), float(goal_y)),
            "body_xy": body_xy,
        }

    def _sample_car_block_bridge(self) -> dict[str, object]:
        spec = self._sample_car_block_progression()
        body_xy = dict(spec["body_xy"])
        main_y = float(body_xy.get("hazard0", (0.0, 0.0))[1])
        # Milder than progression level 4: the second obstacle is present often
        # enough to force route adaptation but leaves a wider corridor.
        if self._rng.random() < 0.7:
            sign = -1.0 if main_y >= 0.0 else 1.0
            body_xy["hazard1"] = (
                self._uniform(0.32, 0.62),
                float(sign * self._uniform(0.38, 0.62)),
            )
        spec["body_xy"] = body_xy
        return spec

    def _sample_blocking_hazard(self, agent_xy: tuple[float, float], goal_xy: tuple[float, float]) -> np.ndarray:
        a = np.asarray(agent_xy, dtype=np.float64)
        g = np.asarray(goal_xy, dtype=np.float64)
        seg = g - a
        norm = float(np.linalg.norm(seg))
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float64)
        unit = seg / norm
        perp = np.asarray((-unit[1], unit[0]), dtype=np.float64)
        t = self._uniform(0.38, 0.62)
        offset = self._uniform(-0.25, 0.25)
        return a + t * seg + offset * perp


# Reuse the robust MuJoCo layout mutation from FixedSafetyLayoutWrapper without
# forcing curriculum users through a named fixed preset.
def _apply_layout_from_spec(task, spec: dict[str, object]) -> None:
    shim = object.__new__(FixedSafetyLayoutWrapper)
    shim._spec = spec
    FixedSafetyLayoutWrapper._apply_layout(shim, task)


FixedSafetyLayoutWrapper._apply_layout_from_spec = staticmethod(_apply_layout_from_spec)  # type: ignore[attr-defined]


class SafetyLayoutSeedReplayWrapper(gym.Wrapper):
    """Occasionally replace reset seeds with a curated hard-layout seed list."""

    def __init__(
        self,
        env: gym.Env,
        *,
        seeds: list[int] | tuple[int, ...],
        replay_prob: float = 1.0,
        mode: str = "cycle",
        rng_seed: int = 0,
    ):
        super().__init__(env)
        parsed = [int(s) for s in seeds]
        if not parsed:
            raise ValueError("SafetyLayoutSeedReplayWrapper requires at least one seed")
        self.seeds = tuple(parsed)
        self.replay_prob = float(np.clip(float(replay_prob), 0.0, 1.0))
        self.mode = str(mode or "cycle").strip().lower()
        if self.mode not in {"cycle", "random"}:
            raise ValueError(f"Unknown layout seed replay mode: {mode!r}")
        self._rng = np.random.default_rng(int(rng_seed))
        self._idx = 0

    def reset(self, **kwargs):
        original_seed = kwargs.get("seed")
        use_replay = self.replay_prob >= 1.0 or bool(self._rng.random() < self.replay_prob)
        replay_seed = None
        if use_replay:
            if self.mode == "random":
                replay_seed = int(self.seeds[int(self._rng.integers(0, len(self.seeds)))])
            else:
                replay_seed = int(self.seeds[self._idx % len(self.seeds)])
                self._idx += 1
            kwargs = dict(kwargs)
            kwargs["seed"] = replay_seed
        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        info["layout_seed_replay_enabled"] = float(use_replay)
        info["layout_seed_replay_prob"] = float(self.replay_prob)
        if replay_seed is not None:
            info["layout_seed_replay_seed"] = int(replay_seed)
            if original_seed is not None:
                info["layout_seed_replay_original_seed"] = int(original_seed)
        return obs, info


def parse_layout_seed_replay(value: str | None) -> tuple[int, ...]:
    """Parse comma/newline separated reset seeds or a path containing them."""

    raw = str(value or "").strip()
    if raw in {"", "none"}:
        return ()
    path = Path(raw).expanduser()
    if path.exists():
        raw = path.read_text()
    seeds: list[int] = []
    for token in raw.replace(",", "\n").splitlines():
        token = token.strip()
        if not token or token.startswith("#"):
            continue
        token = token.split("#", 1)[0].strip()
        if token:
            seeds.append(int(token))
    return tuple(seeds)


@dataclass
class InterventionEpisodeStats:
    episode_steps: int = 0
    intervention_steps: int = 0
    num_interventions: int = 0
    burst_count: int = 0
    burst_steps_sum: int = 0
    _in_burst: bool = False
    _cur_burst_steps: int = 0

    def on_step(self, intervened: bool) -> None:
        self.episode_steps += 1
        if intervened:
            self.intervention_steps += 1
            if not self._in_burst:
                self._in_burst = True
                self._cur_burst_steps = 0
                self.burst_count += 1
                self.num_interventions += 1
            self._cur_burst_steps += 1
        else:
            self._close_burst()

    def _close_burst(self) -> None:
        if self._in_burst:
            self._in_burst = False
            self.burst_steps_sum += self._cur_burst_steps
            self._cur_burst_steps = 0

    def finalize(self) -> dict:
        self._close_burst()
        frac = float(self.intervention_steps) / max(1, int(self.episode_steps))
        avg_burst = float(self.burst_steps_sum) / max(1, int(self.burst_count))
        return {
            "teacher_episode_steps": int(self.episode_steps),
            "teacher_intervention_steps": int(self.intervention_steps),
            "teacher_fraction_steps": float(frac),
            "teacher_num_interventions": int(self.num_interventions),
            "teacher_num_bursts": int(self.burst_count),
            "teacher_avg_burst_len": float(avg_burst),
        }


class HumanInterventionWrapper(gym.Wrapper):
    """Overlay human teleop intervention on top of student policy actions."""

    def __init__(
        self,
        env: gym.Env,
        *,
        controller,
        threshold: float = 0.1,
        hold_seconds: float = 0.25,
        clearance_override_threshold: float = -1.0,
        clearance_override_exit_threshold: float = -1.0,
        clearance_override_mode: str = "clearance",
        teacher_goal_progress_steps: int = 3,
        teacher_goal_progress_epsilon: float = 1e-3,
        teacher_progress_bad_steps: int = 3,
        teacher_progress_good_steps: int = 5,
        teacher_progress_epsilon: float = 1e-4,
        teacher_progress_trigger_mode: str = "worse",
        teacher_progress_release_mode: str = "improve",
        teacher_progress_score_mode: str = "reward_wrapper",
        teacher_progress_dense_scale: float = 1.0,
        teacher_progress_clearance_scale: float = -1.0,
        teacher_progress_clearance_margin: float = 0.0,
        teacher_progress_clearance_mode: str = "softplus",
        teacher_progress_clearance_temperature: float = 0.001,
        teacher_clearance_source: str = "keepout",
        debug_console: bool = False,
    ):
        super().__init__(env)
        self.controller = controller
        self.threshold = float(threshold)
        self.hold_seconds = float(hold_seconds)
        self.clearance_override_threshold = float(clearance_override_threshold)
        self.clearance_override_exit_threshold = (
            float(clearance_override_exit_threshold)
            if float(clearance_override_exit_threshold) >= 0.0
            else float(clearance_override_threshold)
        )
        self.clearance_override_mode = str(clearance_override_mode).strip().lower()
        if self.clearance_override_mode not in {
            "clearance",
            "clearance_projected_release",
            "clearance_projected_release_or_progress",
            "teacher_goal_progress",
            "student_forward_clearance",
            "student_projected_clearance",
            "reward_progress",
            "clearance_or_progress",
            "pcpo_value_progress",
            "pcpo_cost_value_progress",
        }:
            raise ValueError(f"Unsupported clearance_override_mode: {clearance_override_mode}")
        self.teacher_goal_progress_steps = int(max(1, teacher_goal_progress_steps))
        self.teacher_goal_progress_epsilon = float(max(0.0, teacher_goal_progress_epsilon))
        self.teacher_progress_bad_steps = int(max(1, teacher_progress_bad_steps))
        self.teacher_progress_good_steps = int(max(1, teacher_progress_good_steps))
        self.teacher_progress_epsilon = float(max(0.0, teacher_progress_epsilon))
        self.teacher_progress_trigger_mode = str(teacher_progress_trigger_mode).strip().lower()
        if self.teacher_progress_trigger_mode in {"no_progress", "not_progressing", "no-improve"}:
            self.teacher_progress_trigger_mode = "not_improving"
        if self.teacher_progress_trigger_mode not in {"worse", "not_improving"}:
            raise ValueError(f"Unsupported teacher_progress_trigger_mode: {teacher_progress_trigger_mode}")
        self.teacher_progress_release_mode = str(teacher_progress_release_mode).strip().lower()
        if self.teacher_progress_release_mode in {"not_worse", "not-worse", "stable", "plateau"}:
            self.teacher_progress_release_mode = "non_worse"
        if self.teacher_progress_release_mode not in {"improve", "non_worse"}:
            raise ValueError(f"Unsupported teacher_progress_release_mode: {teacher_progress_release_mode}")
        self.teacher_progress_score_mode = str(teacher_progress_score_mode).strip().lower()
        if self.teacher_progress_score_mode in {"reward", "wrapper"}:
            self.teacher_progress_score_mode = "reward_wrapper"
        if self.teacher_progress_score_mode in {"potential", "clearance", "clearance_potential"}:
            self.teacher_progress_score_mode = "potential_field"
        if self.teacher_progress_score_mode not in {"reward_wrapper", "euclidean", "potential_field"}:
            raise ValueError(f"Unsupported teacher_progress_score_mode: {teacher_progress_score_mode}")
        self.teacher_progress_dense_scale = float(teacher_progress_dense_scale)
        self.teacher_progress_clearance_scale = float(teacher_progress_clearance_scale)
        self.teacher_progress_clearance_margin = float(max(0.0, teacher_progress_clearance_margin))
        self.teacher_progress_clearance_mode = str(teacher_progress_clearance_mode).strip().lower()
        if self.teacher_progress_clearance_mode == "hinge_power":
            self.teacher_progress_clearance_mode = "quadratic_hinge"
        if self.teacher_progress_clearance_mode not in {"softplus", "hinge", "quadratic_hinge", "exp_soft"}:
            raise ValueError(f"Unsupported teacher_progress_clearance_mode: {teacher_progress_clearance_mode}")
        self.teacher_progress_clearance_temperature = float(max(1e-6, teacher_progress_clearance_temperature))
        self.teacher_clearance_source = str(teacher_clearance_source or "keepout").strip().lower()
        if self.teacher_clearance_source in {"visual", "footprint", "visual_footprint", "footprint_cost"}:
            self.teacher_clearance_source = "visual_footprint"
        if self.teacher_clearance_source not in {"keepout", "visual_footprint"}:
            raise ValueError(f"Unsupported teacher_clearance_source: {teacher_clearance_source}")
        self.debug_console = bool(debug_console)

        self._last_override_ts = -1e9
        self._last_override_action: Optional[np.ndarray] = None
        self._stats = InterventionEpisodeStats()
        self._last_obs: Optional[np.ndarray] = None
        self._last_teacher_reason: Optional[str] = None
        self._last_trigger_clearance: Optional[float] = None
        self._last_human_norm: float = 0.0
        self._last_human_above_threshold: bool = False
        self._last_human_action: Optional[np.ndarray] = None
        self._was_intervening_prev_step: bool = False
        self._teacher_gate_active: bool = False
        self._teacher_clearance_gate_active: bool = False
        self._teacher_progress_gate_active: bool = False
        self._teacher_goal_progress_count: int = 0
        self._teacher_goal_prev_distance: float = float("nan")
        self._teacher_progress_bad_count: int = 0
        self._teacher_progress_good_count: int = 0
        self._teacher_progress_prev_score: float = float("nan")
        self._teacher_progress_last_score: float = float("nan")
        self._teacher_progress_last_clearance_potential: float = float("nan")
        self._last_gate_release_ready: bool = False
        self._last_student_projected_clearance: float = float("nan")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_override_ts = -1e9
        self._last_override_action = None
        self._stats = InterventionEpisodeStats()
        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._last_teacher_reason = None
        self._last_trigger_clearance = None
        self._last_human_norm = 0.0
        self._last_human_above_threshold = False
        self._last_human_action = None
        self._was_intervening_prev_step = False
        if self.clearance_override_mode in {
            "reward_progress",
            "clearance_or_progress",
            "clearance_projected_release_or_progress",
            "pcpo_value_progress",
            "pcpo_cost_value_progress",
        }:
            self._teacher_gate_active = False
        else:
            self._teacher_gate_active = self.clearance_override_threshold < 0.0
        self._teacher_clearance_gate_active = False
        self._teacher_progress_gate_active = False
        self._teacher_goal_progress_count = 0
        self._teacher_goal_prev_distance = float("nan")
        self._teacher_progress_bad_count = 0
        self._teacher_progress_good_count = 0
        self._teacher_progress_prev_score = self._current_progress_score()
        self._teacher_progress_last_score = self._teacher_progress_prev_score
        self._teacher_progress_last_clearance_potential = float("nan")
        self._last_gate_release_ready = False
        self._last_student_projected_clearance = float("nan")
        if hasattr(self.controller, "reset"):
            try:
                self.controller.reset()
            except TypeError:
                pass
        return obs, info

    def _iter_inner_wrappers(self):
        cur = self.env
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            yield cur
            cur = getattr(cur, "env", None)

    def _teacher_progress_uses_decoupled_score(self) -> bool:
        return self.teacher_progress_score_mode != "reward_wrapper"

    def _teacher_clearance(self) -> float:
        if self.teacher_clearance_source == "visual_footprint":
            for wrapper in self._iter_inner_wrappers():
                if hasattr(wrapper, "_visual_min_clearance"):
                    try:
                        clearance = wrapper._visual_min_clearance()
                    except TypeError:
                        continue
                    if clearance is not None and np.isfinite(clearance):
                        return float(clearance)
        return float(extract_min_constrained_clearance(self.env))

    def _teacher_progress_clearance_potential(self, clearance: float) -> float:
        if self.teacher_progress_clearance_scale < 0.0 or not np.isfinite(clearance):
            return 0.0
        gap = float(self.teacher_progress_clearance_margin - float(clearance))
        scale = float(self.teacher_progress_clearance_scale)
        mode = str(self.teacher_progress_clearance_mode)
        if mode == "softplus":
            temp = float(self.teacher_progress_clearance_temperature)
            return float(-scale * temp * np.logaddexp(gap / temp, 0.0))
        if mode == "hinge":
            return float(-scale * max(0.0, gap))
        if mode == "quadratic_hinge":
            violation = max(0.0, gap)
            return float(-scale * violation * violation)
        if mode == "exp_soft":
            temp = float(self.teacher_progress_clearance_temperature)
            return float(-scale * temp * math.exp(min(gap / temp, 50.0)))
        return 0.0

    def _reward_progress_score(self) -> float:
        if self._teacher_progress_uses_decoupled_score():
            dist = extract_goal_distance(self.env)
            score = 0.0
            if np.isfinite(dist):
                score += float(-self.teacher_progress_dense_scale * float(dist))
            if self.teacher_progress_score_mode == "potential_field":
                clearance = self._teacher_clearance()
                clearance_potential = self._teacher_progress_clearance_potential(float(clearance))
                self._teacher_progress_last_clearance_potential = float(clearance_potential)
                if np.isfinite(clearance_potential):
                    score += float(clearance_potential)
            else:
                self._teacher_progress_last_clearance_potential = 0.0
            return float(score)
        for wrapper in self._iter_inner_wrappers():
            if hasattr(wrapper, "current_shaped_state_score"):
                try:
                    score = wrapper.current_shaped_state_score()
                except TypeError:
                    continue
                if score is not None and np.isfinite(score):
                    return float(score)
        dist = extract_goal_distance(self.env)
        if np.isfinite(dist):
            return float(-dist)
        return float("nan")

    def _pcpo_value_score(self) -> float:
        if self._last_obs is None or not hasattr(self.controller, "evaluate_value"):
            return float("nan")
        try:
            score = self.controller.evaluate_value(obs=self._last_obs, env=self.env)
        except TypeError:
            score = self.controller.evaluate_value(self._last_obs)
        if score is None or not np.isfinite(score):
            return float("nan")
        return float(score)

    def _pcpo_cost_value_score(self) -> float:
        if self._last_obs is None or not hasattr(self.controller, "evaluate_cost_value"):
            return float("nan")
        try:
            score = self.controller.evaluate_cost_value(obs=self._last_obs, env=self.env)
        except TypeError:
            score = self.controller.evaluate_cost_value(self._last_obs)
        if score is None or not np.isfinite(score):
            return float("nan")
        return float(-score)

    def _current_progress_score(self) -> float:
        if self.clearance_override_mode == "pcpo_value_progress":
            return self._pcpo_value_score()
        if self.clearance_override_mode == "pcpo_cost_value_progress":
            return self._pcpo_cost_value_score()
        return self._reward_progress_score()

    def _update_progress_delta_counts(self, score: float) -> None:
        prev = float(self._teacher_progress_prev_score)
        if np.isfinite(prev):
            delta = float(score - prev)
            if (
                (self.teacher_progress_trigger_mode == "worse" and delta < -self.teacher_progress_epsilon)
                or (
                    self.teacher_progress_trigger_mode == "not_improving"
                    and delta <= self.teacher_progress_epsilon
                )
            ):
                self._teacher_progress_bad_count += 1
                self._teacher_progress_good_count = 0
            elif (
                delta > self.teacher_progress_epsilon
                or (self.teacher_progress_release_mode == "non_worse" and delta >= -self.teacher_progress_epsilon)
            ):
                self._teacher_progress_good_count += 1
                self._teacher_progress_bad_count = 0
            else:
                self._teacher_progress_bad_count = 0
                self._teacher_progress_good_count = 0
        self._teacher_progress_prev_score = float(score)

    def _progress_override_active(self, student_action: Optional[np.ndarray] = None) -> tuple[bool, Optional[float]]:
        score = self._current_progress_score()
        self._teacher_progress_last_score = float(score)
        if not np.isfinite(score):
            self._last_gate_release_ready = False
            return False, None

        clearance = self._teacher_clearance()
        clearance_trigger = (
            self.clearance_override_threshold >= 0.0
            and np.isfinite(clearance)
            and float(clearance) <= self.clearance_override_threshold
        )
        clearance_release_ready = (
            self.clearance_override_exit_threshold < 0.0
            or not np.isfinite(clearance)
            or float(clearance) >= self.clearance_override_exit_threshold
        )

        self._update_progress_delta_counts(float(score))

        if self.clearance_override_mode == "clearance_projected_release_or_progress":
            release_thr = float(
                self.clearance_override_exit_threshold
                if self.clearance_override_exit_threshold >= 0.0
                else self.clearance_override_threshold
            )
            projected = self._student_projected_clearance(student_action)
            self._last_student_projected_clearance = float(projected)
            projected_trigger = np.isfinite(projected) and float(projected) <= release_thr
            current_trigger = (
                self.clearance_override_threshold >= 0.0
                and np.isfinite(clearance)
                and float(clearance) <= self.clearance_override_threshold
            )
            clearance_release_ready = (
                (not np.isfinite(clearance) or float(clearance) > release_thr)
                and (not np.isfinite(projected) or float(projected) > release_thr)
            )
            if not self._teacher_clearance_gate_active and (current_trigger or projected_trigger):
                self._teacher_clearance_gate_active = True
            elif self._teacher_clearance_gate_active and clearance_release_ready:
                self._teacher_clearance_gate_active = False

            if (
                not self._teacher_progress_gate_active
                and self._teacher_progress_bad_count >= self.teacher_progress_bad_steps
            ):
                self._teacher_progress_gate_active = True
                self._teacher_progress_good_count = 0
            elif (
                self._teacher_progress_gate_active
                and self._teacher_progress_good_count >= self.teacher_progress_good_steps
            ):
                self._teacher_progress_gate_active = False
                self._teacher_progress_bad_count = 0

            self._teacher_gate_active = bool(
                self._teacher_clearance_gate_active or self._teacher_progress_gate_active
            )
            self._last_gate_release_ready = not self._teacher_gate_active
            return bool(self._teacher_gate_active), float(score)

        if self.clearance_override_mode == "clearance_or_progress":
            if self.clearance_override_threshold >= 0.0:
                if not self._teacher_clearance_gate_active and clearance_trigger:
                    self._teacher_clearance_gate_active = True
                elif self._teacher_clearance_gate_active and clearance_release_ready:
                    self._teacher_clearance_gate_active = False
            if (
                not self._teacher_progress_gate_active
                and self._teacher_progress_bad_count >= self.teacher_progress_bad_steps
            ):
                self._teacher_progress_gate_active = True
                self._teacher_progress_good_count = 0
            elif (
                self._teacher_progress_gate_active
                and self._teacher_progress_good_count >= self.teacher_progress_good_steps
            ):
                self._teacher_progress_gate_active = False
                self._teacher_progress_bad_count = 0
            self._teacher_gate_active = bool(
                self._teacher_clearance_gate_active or self._teacher_progress_gate_active
            )
            self._last_gate_release_ready = not self._teacher_gate_active
            return bool(self._teacher_gate_active), float(score)

        if not self._teacher_gate_active:
            if clearance_trigger or self._teacher_progress_bad_count >= self.teacher_progress_bad_steps:
                self._teacher_gate_active = True
                self._teacher_progress_good_count = 0
                self._last_gate_release_ready = False
        else:
            release_ready = (
                self._teacher_progress_good_count >= self.teacher_progress_good_steps
                and clearance_release_ready
            )
            self._last_gate_release_ready = bool(release_ready)
            if release_ready:
                self._teacher_gate_active = False
                self._teacher_progress_bad_count = 0

        return bool(self._teacher_gate_active), float(score)

    def _student_forward_command(self, student_action: Optional[np.ndarray]) -> float:
        if student_action is None:
            return 0.0
        arr = np.asarray(student_action, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0
        cur = self.env
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            mode = str(getattr(cur, "action_mode", "")).strip().lower()
            if mode == "throttle_turn":
                return float(arr[0])
            if mode == "cardinal":
                return float(arr[1] if arr.size > 1 else 0.0)
            cur = getattr(cur, "env", None)
        if arr.size >= 2:
            scale = float(np.max(np.abs(np.concatenate([np.asarray(self.action_space.low).reshape(-1)[:2], np.asarray(self.action_space.high).reshape(-1)[:2]]))))
            return float(0.5 * (arr[0] + arr[1]) / max(scale, 1e-6))
        return float(arr[0])

    def _student_projected_clearance(self, student_action: Optional[np.ndarray]) -> float:
        """Predict the minimum visual footprint clearance after the proposed action.

        This is intentionally used only as an intervention gate diagnostic for the
        raw SafetyCar action path. It restores MuJoCo state after the probe.
        """
        if student_action is None or self.teacher_clearance_source != "visual_footprint":
            return float("nan")
        base = unwrap_env(self.env)
        task = getattr(base, "task", None)
        model = getattr(task, "model", None)
        data = getattr(task, "data", None)
        agent = getattr(task, "agent", None)
        if task is None or model is None or data is None or agent is None:
            return float("nan")
        action = np.asarray(student_action, dtype=np.float32).reshape(-1)
        if int(getattr(model, "nu", 0)) != int(action.size):
            return float("nan")
        try:
            import mujoco
        except Exception:
            return float("nan")

        qpos = np.asarray(data.qpos).copy()
        qvel = np.asarray(data.qvel).copy()
        ctrl = np.asarray(data.ctrl).copy()
        time_before = float(data.time)
        act = np.asarray(data.act).copy() if getattr(data, "act", None) is not None and np.asarray(data.act).size else None
        mocap_pos = np.asarray(data.mocap_pos).copy() if getattr(data, "mocap_pos", None) is not None and np.asarray(data.mocap_pos).size else None
        mocap_quat = np.asarray(data.mocap_quat).copy() if getattr(data, "mocap_quat", None) is not None and np.asarray(data.mocap_quat).size else None

        min_clearance = self._teacher_clearance()
        try:
            sim_conf = getattr(task, "sim_conf", None)
            frames = int(max(1, getattr(sim_conf, "frameskip_binom_n", 1)))
        except Exception:
            frames = 1
        try:
            agent.apply_action(action)
            for _ in range(frames):
                mujoco.mj_step(model, data)
                clearance = self._teacher_clearance()
                if np.isfinite(clearance) and (
                    not np.isfinite(min_clearance) or float(clearance) < float(min_clearance)
                ):
                    min_clearance = float(clearance)
        except Exception:
            return float("nan")
        finally:
            try:
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                data.ctrl[:] = ctrl
                data.time = time_before
                if act is not None:
                    data.act[:] = act
                if mocap_pos is not None:
                    data.mocap_pos[:] = mocap_pos
                if mocap_quat is not None:
                    data.mocap_quat[:] = mocap_quat
                mujoco.mj_forward(model, data)
            except Exception:
                pass
        return float(min_clearance) if np.isfinite(min_clearance) else float("nan")

    def _clearance_override_active(self, student_action: Optional[np.ndarray] = None) -> tuple[bool, Optional[float]]:
        if self.clearance_override_mode in {
            "reward_progress",
            "clearance_or_progress",
            "clearance_projected_release_or_progress",
            "pcpo_value_progress",
            "pcpo_cost_value_progress",
        }:
            return self._progress_override_active(student_action=student_action)
        if self.clearance_override_threshold < 0.0:
            return True, None
        clearance = self._teacher_clearance()
        if clearance is None or not np.isfinite(clearance):
            return False, None
        clearance_f = float(clearance)
        enter = float(self.clearance_override_threshold)
        exit_thr = float(max(enter, self.clearance_override_exit_threshold))

        if self.clearance_override_mode == "clearance_projected_release":
            # Enter on a conservative current-clearance margin, but release once
            # the immediate projected student action no longer moves the visual
            # footprint into the recovery band. This avoids long sticky takeover
            # labels after the teacher has already steered out of danger.
            release_thr = float(
                self.clearance_override_exit_threshold
                if self.clearance_override_exit_threshold >= 0.0
                else enter
            )
            projected = self._student_projected_clearance(student_action)
            self._last_student_projected_clearance = float(projected)
            projected_trigger = np.isfinite(projected) and float(projected) <= release_thr
            current_trigger = clearance_f <= enter
            if not self._teacher_gate_active:
                if current_trigger or projected_trigger:
                    self._teacher_gate_active = True
                    self._teacher_goal_progress_count = 0
                    self._last_gate_release_ready = False
                else:
                    self._last_gate_release_ready = True
                return bool(self._teacher_gate_active), float(
                    min(clearance_f, float(projected)) if np.isfinite(projected) else clearance_f
                )
            release_ready = clearance_f > release_thr and (
                not np.isfinite(projected) or float(projected) > release_thr
            )
            self._last_gate_release_ready = bool(release_ready)
            if release_ready:
                self._teacher_gate_active = False
            return bool(self._teacher_gate_active), float(
                min(clearance_f, float(projected)) if np.isfinite(projected) else clearance_f
            )

        if self.clearance_override_mode == "student_forward_clearance":
            self._teacher_gate_active = bool(clearance_f <= enter and self._student_forward_command(student_action) > 0.05)
            self._last_gate_release_ready = not self._teacher_gate_active
            if not self._teacher_gate_active:
                self._teacher_goal_progress_count = 0
            return bool(self._teacher_gate_active), clearance_f

        if self.clearance_override_mode == "student_projected_clearance":
            projected = self._student_projected_clearance(student_action)
            self._last_student_projected_clearance = float(projected)
            projected_trigger = np.isfinite(projected) and float(projected) <= enter
            current_trigger = clearance_f <= enter
            if not self._teacher_gate_active:
                if current_trigger or projected_trigger:
                    self._teacher_gate_active = True
                    self._teacher_goal_progress_count = 0
                    self._last_gate_release_ready = False
                else:
                    self._last_gate_release_ready = True
                return bool(self._teacher_gate_active), float(
                    min(clearance_f, float(projected)) if np.isfinite(projected) else clearance_f
                )
            release_ready = clearance_f > exit_thr and (
                not np.isfinite(projected) or float(projected) > enter
            )
            self._last_gate_release_ready = bool(release_ready)
            if release_ready:
                self._teacher_gate_active = False
            return bool(self._teacher_gate_active), float(
                min(clearance_f, float(projected)) if np.isfinite(projected) else clearance_f
            )

        if not self._teacher_gate_active:
            if clearance_f <= enter:
                self._teacher_gate_active = True
                self._teacher_goal_progress_count = 0
                self._last_gate_release_ready = False
            return bool(self._teacher_gate_active), clearance_f

        if self.clearance_override_mode == "teacher_goal_progress":
            release_ready = clearance_f > exit_thr and self._teacher_goal_progress_count >= self.teacher_goal_progress_steps
            self._last_gate_release_ready = bool(release_ready)
            if release_ready:
                self._teacher_gate_active = False
                self._teacher_goal_progress_count = 0
        else:
            self._last_gate_release_ready = bool(clearance_f > exit_thr)
            if clearance_f > exit_thr:
                self._teacher_gate_active = False

        return bool(self._teacher_gate_active), clearance_f

    def _current_teacher_action(self, student_action: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if getattr(self.controller, "always_active", False):
            active, clearance = self._clearance_override_active(student_action=student_action)
            self._last_trigger_clearance = clearance
            if not active:
                self._last_teacher_reason = None
                return None
            if self._last_obs is None:
                self._last_teacher_reason = None
                return None
            self._teacher_goal_prev_distance = extract_goal_distance(self.env)
            try:
                action = self.controller.get_action(obs=self._last_obs, env=self.env, student_action=student_action)
            except TypeError:
                try:
                    action = self.controller.get_action(obs=self._last_obs, env=self.env)
                except TypeError:
                    action = self.controller.get_action(obs=self._last_obs)
            if action is None:
                self._last_teacher_reason = None
                return None
            if self.clearance_override_mode in {"reward_progress", "pcpo_value_progress", "pcpo_cost_value_progress"}:
                self._last_teacher_reason = f"expert_{self.clearance_override_mode}"
            else:
                self._last_teacher_reason = (
                    "expert_clearance_override"
                    if clearance is not None and self.clearance_override_threshold >= 0.0
                    else "expert"
                )
            return np.asarray(action, dtype=np.float32)

        try:
            human_raw = self.controller.get_action(obs=self._last_obs)
        except TypeError:
            human_raw = self.controller.get_action()
        human = np.asarray(human_raw, dtype=np.float32)
        self._last_human_action = np.asarray(human, dtype=np.float32)
        self._last_human_norm = float(np.linalg.norm(human))
        now = time.perf_counter()
        if self._last_human_norm > self.threshold:
            self._last_override_ts = now
            self._last_override_action = human
            self._last_human_above_threshold = True
        else:
            self._last_human_above_threshold = False
        active = (now - self._last_override_ts) <= self.hold_seconds
        if active and self._last_override_action is not None:
            self._last_teacher_reason = "human"
            self._last_trigger_clearance = None
            return np.asarray(self._last_override_action, dtype=np.float32)
        self._last_teacher_reason = None
        self._last_trigger_clearance = None
        return None

    def step(self, student_action):
        student_action = np.asarray(student_action, dtype=np.float32)
        t_ctrl0 = time.perf_counter()
        teacher_action = self._current_teacher_action(student_action=student_action)
        controller_ms = (time.perf_counter() - t_ctrl0) * 1e3
        if teacher_action is not None:
            applied_action = clip_action_to_space(teacher_action, self.action_space)
            intervened = True
        else:
            applied_action = clip_action_to_space(student_action, self.action_space)
            intervened = False

        obs, reward, cost, terminated, truncated, info = self.env.step(applied_action)
        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        post_step_clearance = self._teacher_clearance()
        post_step_goal_distance = extract_goal_distance(self.env)
        if (
            intervened
            and self.clearance_override_mode == "teacher_goal_progress"
            and self.clearance_override_threshold >= 0.0
        ):
            exit_thr = float(max(self.clearance_override_threshold, self.clearance_override_exit_threshold))
            prev_dist = float(self._teacher_goal_prev_distance)
            cur_dist = float(post_step_goal_distance)
            if (
                post_step_clearance is not None
                and np.isfinite(post_step_clearance)
                and float(post_step_clearance) > exit_thr
                and np.isfinite(prev_dist)
                and np.isfinite(cur_dist)
                and (prev_dist - cur_dist) > self.teacher_goal_progress_epsilon
            ):
                self._teacher_goal_progress_count += 1
            else:
                self._teacher_goal_progress_count = 0

        self._stats.on_step(intervened)
        info = dict(info)
        info["teacher_intervened"] = bool(intervened)
        info["teacher_reason"] = self._last_teacher_reason if intervened else None
        info["human_input_norm"] = float(self._last_human_norm)
        info["human_input_above_threshold"] = bool(self._last_human_above_threshold)
        info["teacher_trigger_clearance"] = (
            float(self._last_trigger_clearance) if intervened and self._last_trigger_clearance is not None else None
        )
        info["teacher_action"] = np.asarray(teacher_action if teacher_action is not None else applied_action, dtype=np.float32)
        info["student_action"] = np.asarray(student_action, dtype=np.float32)
        info["teacher_delta_l2"] = float(np.linalg.norm(info["teacher_action"] - info["student_action"]))
        info["teacher_controller_ms"] = float(controller_ms)
        if hasattr(self.controller, "last_intervention_probability"):
            info["teacher_intervention_probability"] = float(getattr(self.controller, "last_intervention_probability"))
        if hasattr(self.controller, "set_intervention_status"):
            try:
                self.controller.set_intervention_status(
                    active=bool(intervened),
                    probability=(
                        float(info["teacher_intervention_probability"])
                        if "teacher_intervention_probability" in info
                        else None
                    ),
                )
            except TypeError:
                pass
        info["teacher_gate_active"] = bool(self._teacher_gate_active)
        info["teacher_clearance_gate_active"] = bool(self._teacher_clearance_gate_active)
        info["teacher_progress_gate_active"] = bool(self._teacher_progress_gate_active)
        info["teacher_gate_mode"] = str(self.clearance_override_mode)
        info["teacher_gate_release_ready"] = bool(self._last_gate_release_ready)
        info["teacher_goal_progress_count"] = float(self._teacher_goal_progress_count)
        info["teacher_progress_bad_count"] = float(self._teacher_progress_bad_count)
        info["teacher_progress_good_count"] = float(self._teacher_progress_good_count)
        info["teacher_progress_score_decoupled"] = 1.0 if self._teacher_progress_uses_decoupled_score() else 0.0
        info["teacher_progress_trigger_mode"] = str(self.teacher_progress_trigger_mode)
        info["teacher_progress_release_mode"] = str(self.teacher_progress_release_mode)
        info["teacher_progress_score_mode"] = str(self.teacher_progress_score_mode)
        info["teacher_progress_dense_scale"] = float(self.teacher_progress_dense_scale)
        if np.isfinite(self._last_student_projected_clearance):
            info["teacher_student_projected_clearance"] = float(self._last_student_projected_clearance)
        info["teacher_progress_clearance_scale"] = float(self.teacher_progress_clearance_scale)
        info["teacher_progress_clearance_margin"] = float(self.teacher_progress_clearance_margin)
        info["teacher_progress_clearance_temperature"] = float(self.teacher_progress_clearance_temperature)
        info["teacher_progress_clearance_mode"] = str(self.teacher_progress_clearance_mode)
        info["teacher_clearance_source"] = str(self.teacher_clearance_source)
        if np.isfinite(self._teacher_progress_last_clearance_potential):
            info["teacher_progress_clearance_potential"] = float(self._teacher_progress_last_clearance_potential)
        if np.isfinite(self._teacher_progress_last_score):
            info["teacher_progress_score"] = float(self._teacher_progress_last_score)
        if np.isfinite(post_step_goal_distance):
            info["teacher_gate_goal_distance"] = float(post_step_goal_distance)
        if post_step_clearance is not None and np.isfinite(post_step_clearance):
            info["teacher_gate_clearance"] = float(post_step_clearance)

        if self.debug_console:
            if intervened and not self._was_intervening_prev_step:
                print(
                    "[Intervention] START "
                    f"reason={info.get('teacher_reason')} "
                    f"mode={info.get('teacher_gate_mode')} "
                    f"score={info.get('teacher_progress_score', float('nan'))} "
                    f"bad={int(float(info.get('teacher_progress_bad_count', 0.0)))} "
                    f"good={int(float(info.get('teacher_progress_good_count', 0.0)))} "
                    f"clearance={info.get('teacher_gate_clearance', None)} "
                    f"norm={self._last_human_norm:.4f} "
                    f"threshold={self.threshold:.4f} "
                    f"hold={self.hold_seconds:.3f}s "
                    f"action={np.array2string(np.asarray(info['teacher_action']), precision=3)}",
                    flush=True,
                )
            elif (not intervened) and self._was_intervening_prev_step:
                print(
                    "[Intervention] END "
                    f"mode={info.get('teacher_gate_mode')} "
                    f"score={info.get('teacher_progress_score', float('nan'))} "
                    f"bad={int(float(info.get('teacher_progress_bad_count', 0.0)))} "
                    f"good={int(float(info.get('teacher_progress_good_count', 0.0)))} "
                    f"clearance={info.get('teacher_gate_clearance', None)} "
                    f"norm={self._last_human_norm:.4f} "
                    f"above_threshold={int(self._last_human_above_threshold)}",
                    flush=True,
                )
        self._was_intervening_prev_step = bool(intervened)

        if terminated or truncated:
            info.update(self._stats.finalize())

        return obs, reward, cost, terminated, truncated, info


class RewardModeWrapper(gym.Wrapper):
    """Apply sparse/dense reward modes for Safety-Gymnasium."""

    def __init__(
        self,
        env: gym.Env,
        *,
        reward_mode: str,
        dense_reward_scale: float = 1.0,
        success_reward_scale: float = 1.0,
        step_penalty: float = 0.0,
        cost_penalty: float = 0.0,
        cost_penalty_warmup_steps: int = 0,
        cost_penalty_ramp_steps: int = 0,
        clearance_penalty_scale: float = 0.0,
        clearance_margin: float = 0.0,
        clearance_penalty_power: float = 1.0,
        clearance_penalty_mode: str = "hinge_power",
        clearance_penalty_temperature: float = 0.08,
        clearance_penalty_warmup_steps: int = 0,
        clearance_penalty_ramp_steps: int = 0,
        forward_reward_scale: float = 0.0,
        backward_penalty_scale: float = 0.0,
        heading_reward_scale: float = 0.0,
        heading_positive_only: bool = True,
        adaptive_safety_curriculum: bool = False,
        adaptive_safety_goal_target: float = 1.0,
        adaptive_safety_window_episodes: int = 10,
        adaptive_safety_step: float = 0.05,
        adaptive_safety_init: float = 0.0,
        adaptive_safety_min: float = 0.0,
        adaptive_safety_max: float = 1.0,
    ):
        super().__init__(env)
        mode = str(reward_mode).lower()
        if mode == "dual":
            mode = "dense_plus_sparse"
        if mode not in {"sparse", "dense", "dense_plus_sparse", "potential_diff", "native", "none"}:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")
        self.reward_mode = mode
        self.dense_reward_scale = float(dense_reward_scale)
        self.success_reward_scale = float(success_reward_scale)
        self.step_penalty = float(step_penalty)
        self.cost_penalty = float(cost_penalty)
        self.cost_penalty_warmup_steps = int(max(0, cost_penalty_warmup_steps))
        self.cost_penalty_ramp_steps = int(max(0, cost_penalty_ramp_steps))
        self.clearance_penalty_scale = float(clearance_penalty_scale)
        self.clearance_margin = float(max(0.0, clearance_margin))
        self.clearance_penalty_power = float(max(1.0, clearance_penalty_power))
        self.clearance_penalty_mode = str(clearance_penalty_mode).strip().lower()
        if self.clearance_penalty_mode not in {"hinge_power", "softplus"}:
            raise ValueError(f"Unsupported clearance_penalty_mode: {clearance_penalty_mode}")
        self.clearance_penalty_temperature = float(max(1e-6, clearance_penalty_temperature))
        self.clearance_penalty_warmup_steps = int(max(0, clearance_penalty_warmup_steps))
        self.clearance_penalty_ramp_steps = int(max(0, clearance_penalty_ramp_steps))
        self.forward_reward_scale = float(forward_reward_scale)
        self.backward_penalty_scale = float(backward_penalty_scale)
        self.heading_reward_scale = float(heading_reward_scale)
        self.heading_positive_only = bool(heading_positive_only)
        self.adaptive_safety_curriculum = bool(adaptive_safety_curriculum)
        self.adaptive_safety_goal_target = float(max(0.0, adaptive_safety_goal_target))
        self.adaptive_safety_window_episodes = int(max(1, adaptive_safety_window_episodes))
        self.adaptive_safety_step = float(max(0.0, adaptive_safety_step))
        self.adaptive_safety_min = float(adaptive_safety_min)
        self.adaptive_safety_max = float(max(adaptive_safety_min, adaptive_safety_max))
        self._adaptive_safety_scale = float(
            min(self.adaptive_safety_max, max(self.adaptive_safety_min, adaptive_safety_init))
        )
        self._adaptive_goal_history: list[float] = []
        self._adaptive_cost_history: list[float] = []
        self._episode_goal_hits = 0
        self._episode_cost_sum = 0.0
        self._prev_goal_distance = float("nan")
        self._prev_clearance_potential = float("nan")
        self._total_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_goal_distance = extract_goal_distance(self.env)
        self._prev_clearance_potential = self._clearance_potential(extract_min_constrained_clearance(self.env))
        self._episode_goal_hits = 0
        self._episode_cost_sum = 0.0
        return obs, info

    def set_total_steps(self, total_steps: int) -> None:
        self._total_steps = int(max(0, total_steps))

    def _curriculum_scale(self, warmup_steps: int, ramp_steps: int) -> float:
        if warmup_steps <= 0 and ramp_steps <= 0:
            return 1.0
        steps = int(max(0, self._total_steps))
        if steps <= warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return 1.0
        return float(
            min(
                1.0,
                max(0.0, (steps - warmup_steps) / max(1, ramp_steps)),
            )
        )

    def _cost_penalty_scale(self) -> float:
        scale = self._curriculum_scale(self.cost_penalty_warmup_steps, self.cost_penalty_ramp_steps)
        if self.adaptive_safety_curriculum:
            scale *= self._adaptive_safety_scale
        return float(scale)

    def _clearance_penalty_scale(self) -> float:
        scale = self._curriculum_scale(self.clearance_penalty_warmup_steps, self.clearance_penalty_ramp_steps)
        if self.adaptive_safety_curriculum:
            scale *= self._adaptive_safety_scale
        return float(scale)

    def _clearance_potential(self, min_constrained_clearance: float) -> float:
        if (
            self.clearance_penalty_scale == 0.0
            or self.clearance_margin < 0.0
            or not np.isfinite(min_constrained_clearance)
        ):
            return 0.0
        clearance_scale = self._clearance_penalty_scale()
        if self.clearance_penalty_mode == "softplus":
            temp = self.clearance_penalty_temperature
            x = (self.clearance_margin - float(min_constrained_clearance)) / temp
            return float(-self.clearance_penalty_scale * clearance_scale * temp * np.logaddexp(x, 0.0))
        clearance_violation = max(0.0, self.clearance_margin - float(min_constrained_clearance))
        if clearance_violation <= 0.0:
            return 0.0
        return float(
            -self.clearance_penalty_scale
            * clearance_scale
            * (clearance_violation ** self.clearance_penalty_power)
        )

    def current_shaped_state_score(self) -> float:
        """State potential used by intervention gates; higher means closer/safer."""
        dist = extract_goal_distance(self.env)
        score = 0.0
        if np.isfinite(dist):
            score += float(-self.dense_reward_scale * float(dist))
        clearance_potential = self._clearance_potential(extract_min_constrained_clearance(self.env))
        if np.isfinite(clearance_potential):
            score += float(clearance_potential)
        return float(score)

    def _adaptive_goal_mean(self) -> float:
        if not self._adaptive_goal_history:
            return float("nan")
        return float(np.mean(self._adaptive_goal_history))

    def _adaptive_cost_mean(self) -> float:
        if not self._adaptive_cost_history:
            return float("nan")
        return float(np.mean(self._adaptive_cost_history))

    def _update_adaptive_safety_scale_on_episode_end(self) -> None:
        if not self.adaptive_safety_curriculum:
            return
        self._adaptive_goal_history.append(float(self._episode_goal_hits))
        self._adaptive_cost_history.append(float(self._episode_cost_sum))
        if len(self._adaptive_goal_history) > self.adaptive_safety_window_episodes:
            self._adaptive_goal_history.pop(0)
            self._adaptive_cost_history.pop(0)
        if len(self._adaptive_goal_history) < self.adaptive_safety_window_episodes:
            return
        if self._adaptive_goal_mean() >= self.adaptive_safety_goal_target:
            self._adaptive_safety_scale = min(
                self.adaptive_safety_max,
                self._adaptive_safety_scale + self.adaptive_safety_step,
            )
        else:
            self._adaptive_safety_scale = max(
                self.adaptive_safety_min,
                self._adaptive_safety_scale - self.adaptive_safety_step,
            )

    def step(self, action):
        obs, env_reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        self._total_steps += 1

        goal_met = bool(info.get("goal_met", False))
        if goal_met:
            self._episode_goal_hits += 1
        self._episode_cost_sum += float(cost)
        cur_dist = extract_goal_distance(self.env)
        prev_dist = self._prev_goal_distance
        rew = 0.0
        dense_component = 0.0
        sparse_component = 0.0
        cost_penalty_component = 0.0
        clearance_penalty_component = 0.0
        forward_reward_component = 0.0
        backward_penalty_component = 0.0
        heading_reward_component = 0.0
        dense_goal_resample_skip = False

        if self.reward_mode == "sparse":
            sparse_component = self.success_reward_scale if goal_met else 0.0
            rew = sparse_component
        elif self.reward_mode == "dense":
            if goal_met:
                dense_goal_resample_skip = True
                dense_component = 0.0
                rew = 0.0
            elif np.isfinite(prev_dist) and np.isfinite(cur_dist):
                dense_component = self.dense_reward_scale * float(prev_dist - cur_dist)
                rew = dense_component
            else:
                rew = 0.0
        elif self.reward_mode in {"dense_plus_sparse", "potential_diff"}:
            if goal_met:
                dense_goal_resample_skip = True
                dense_component = 0.0
                rew = 0.0
            elif np.isfinite(prev_dist) and np.isfinite(cur_dist):
                dense_component = self.dense_reward_scale * float(prev_dist - cur_dist)
                rew = dense_component
            else:
                rew = 0.0
            if goal_met:
                sparse_component = self.success_reward_scale
                rew += sparse_component
        elif self.reward_mode == "native":
            rew = float(env_reward)
        elif self.reward_mode == "none":
            rew = 0.0

        rew += self.step_penalty
        if self.cost_penalty != 0.0:
            cost_penalty_component = float(self.cost_penalty * self._cost_penalty_scale() * float(cost))
            rew += cost_penalty_component

        min_constrained_clearance = extract_min_constrained_clearance(self.env)
        clearance_potential = self._clearance_potential(min_constrained_clearance)
        if self.clearance_penalty_scale != 0.0 and self.clearance_margin >= 0.0:
            if self.reward_mode == "potential_diff":
                prev_clearance_potential = self._prev_clearance_potential
                if np.isfinite(prev_clearance_potential):
                    clearance_penalty_component = float(clearance_potential - prev_clearance_potential)
                    rew += clearance_penalty_component
            else:
                clearance_penalty_component = float(clearance_potential)
                rew += clearance_penalty_component

        agent_xy = extract_agent_xy(self.env)
        goal_xy = extract_goal_xy(self.env)
        forward_xy = extract_agent_forward_xy(self.env)
        vel_xy = extract_agent_velocity_xy(self.env)
        forward_speed = float("nan")
        goal_heading_alignment = float("nan")
        goal_direction_speed = float("nan")
        if vel_xy is not None and forward_xy is not None:
            forward_speed = float(np.dot(vel_xy, forward_xy))
            if self.forward_reward_scale != 0.0 and forward_speed > 0.0:
                forward_reward_component = float(self.forward_reward_scale * forward_speed)
                rew += forward_reward_component
            if self.backward_penalty_scale != 0.0 and forward_speed < 0.0:
                backward_penalty_component = float(-self.backward_penalty_scale * (-forward_speed))
                rew += backward_penalty_component
        if agent_xy is not None and goal_xy is not None and forward_xy is not None:
            goal_vec = np.asarray(goal_xy - agent_xy, dtype=np.float64).reshape(-1)
            goal_norm = float(np.linalg.norm(goal_vec))
            if goal_norm > 1e-6:
                goal_dir = goal_vec / goal_norm
                goal_heading_alignment = float(np.dot(forward_xy, goal_dir))
                if vel_xy is not None:
                    goal_direction_speed = float(np.dot(vel_xy, goal_dir))
                if self.heading_reward_scale != 0.0 and not goal_met:
                    heading_signal = goal_heading_alignment
                    if self.heading_positive_only:
                        heading_signal = max(0.0, heading_signal)
                    heading_reward_component = float(self.heading_reward_scale * heading_signal)
                    rew += heading_reward_component

        info["reward_mode"] = self.reward_mode
        info["reward_raw_env"] = float(env_reward)
        info["reward_shaped"] = float(rew)
        info["reward_dense_component"] = float(dense_component)
        info["reward_sparse_component"] = float(sparse_component)
        info["reward_success_scale"] = float(self.success_reward_scale)
        info["reward_step_penalty_component"] = float(self.step_penalty)
        info["reward_cost_penalty_component"] = float(cost_penalty_component)
        info["reward_clearance_penalty_component"] = float(clearance_penalty_component)
        info["reward_cost_penalty_scale"] = float(self._cost_penalty_scale())
        info["reward_clearance_penalty_scale"] = float(self._clearance_penalty_scale())
        info["reward_clearance_penalty_mode_softplus"] = 1.0 if self.clearance_penalty_mode == "softplus" else 0.0
        info["reward_clearance_penalty_temperature"] = float(self.clearance_penalty_temperature)
        info["reward_adaptive_safety_enabled"] = 1.0 if self.adaptive_safety_curriculum else 0.0
        info["reward_adaptive_safety_scale"] = float(self._adaptive_safety_scale)
        info["reward_adaptive_goal_window_mean"] = float(self._adaptive_goal_mean())
        info["reward_adaptive_cost_window_mean"] = float(self._adaptive_cost_mean())
        info["reward_forward_component"] = float(forward_reward_component)
        info["reward_backward_penalty_component"] = float(backward_penalty_component)
        info["reward_heading_component"] = float(heading_reward_component)
        info["reward_dense_goal_resample_skip"] = 1.0 if dense_goal_resample_skip else 0.0
        if np.isfinite(forward_speed):
            info["agent_forward_speed"] = float(forward_speed)
        if np.isfinite(goal_heading_alignment):
            info["goal_heading_alignment"] = float(goal_heading_alignment)
        if np.isfinite(goal_direction_speed):
            info["goal_direction_speed"] = float(goal_direction_speed)
        if np.isfinite(prev_dist):
            info["goal_distance_prev"] = float(prev_dist)
        if np.isfinite(cur_dist) and np.isfinite(prev_dist):
            raw_delta = float(cur_dist - prev_dist)
            info["goal_distance_delta_raw"] = raw_delta
            info["goal_distance_delta"] = 0.0 if dense_goal_resample_skip else raw_delta
        if np.isfinite(cur_dist):
            info["goal_distance"] = float(cur_dist)
        if np.isfinite(min_constrained_clearance):
            info["min_constrained_clearance"] = float(min_constrained_clearance)
        if np.isfinite(clearance_potential):
            info["reward_clearance_potential"] = float(clearance_potential)

        if terminated or truncated:
            self._update_adaptive_safety_scale_on_episode_end()

        self._prev_goal_distance = cur_dist
        self._prev_clearance_potential = clearance_potential
        return obs, float(rew), cost, terminated, truncated, info


class TerminateOnGoalWrapper(gym.Wrapper):
    """End the episode immediately after the first goal hit."""

    def set_total_steps(self, total_steps: int) -> None:
        if hasattr(self.env, "set_total_steps"):
            self.env.set_total_steps(total_steps)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        if bool(info.get("goal_met", False)):
            terminated = True
            info["terminated_on_goal"] = True
        else:
            info["terminated_on_goal"] = False
        return obs, reward, cost, bool(terminated), bool(truncated), info


class TerminateOnCostWrapper(gym.Wrapper):
    """End the episode immediately after a positive environment cost."""

    def set_total_steps(self, total_steps: int) -> None:
        if hasattr(self.env, "set_total_steps"):
            self.env.set_total_steps(total_steps)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        if float(cost) > 0.0:
            terminated = True
            info["terminated_on_cost"] = True
        else:
            info["terminated_on_cost"] = False
        return obs, reward, cost, bool(terminated), bool(truncated), info
