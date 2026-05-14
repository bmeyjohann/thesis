from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np


class SurfaceConfigWrapper(gym.Wrapper):
    """Apply deterministic surface physics tweaks after every reset."""

    def __init__(self, env: gym.Env, *, mode: str):
        super().__init__(env)
        mode_l = str(mode).lower()
        if mode_l not in {"default", "grippy"}:
            raise ValueError(f"Unsupported surface mode: {mode}")
        self.mode = mode_l

    def _apply_surface_config(self) -> None:
        if self.mode == "default":
            return

        base = unwrap_env(self.env)
        task = getattr(base, "task", None)
        model = getattr(task, "model", None)
        if model is None:
            return

        # Increase floor torsional/rolling friction modestly.
        try:
            for geom_id in range(int(model.ngeom)):
                if model.geom(geom_id).name == "floor":
                    model.geom_friction[geom_id] = np.asarray([1.0, 0.02, 0.001], dtype=np.float64)
        except Exception:
            pass

        # Point-like agents expose x/y/z joints; add mild damping to reduce glide
        # without freezing motion.
        try:
            for joint_id in range(int(model.njnt)):
                jname = model.joint(joint_id).name
                dof_adr = int(model.jnt_dofadr[joint_id])
                if dof_adr < 0:
                    continue
                if jname in {"x", "y"}:
                    model.dof_damping[dof_adr] = max(float(model.dof_damping[dof_adr]), 0.015)
                    model.dof_frictionloss[dof_adr] = max(float(model.dof_frictionloss[dof_adr]), 0.0005)
                elif jname == "z":
                    model.dof_damping[dof_adr] = max(float(model.dof_damping[dof_adr]), 0.0075)
                    model.dof_frictionloss[dof_adr] = max(float(model.dof_frictionloss[dof_adr]), 0.00025)
        except Exception:
            pass

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_surface_config()
        return obs, info


class CarActuationWrapper(gym.Wrapper):
    """Allow stronger wheel commands for SafetyCar while preserving action dimensionality."""

    def __init__(self, env: gym.Env, *, wheel_command_limit: float = 2.0, force_scale: float = 2.0):
        super().__init__(env)
        self.wheel_command_limit = float(max(1.0, wheel_command_limit))
        self.force_scale = float(max(0.0, force_scale))
        self._base_ctrlrange = None
        self._base_forcerange = None

        if isinstance(getattr(env, "action_space", None), gym.spaces.Box):
            shape = env.action_space.shape
            self.action_space = gym.spaces.Box(
                low=np.full(shape, -self.wheel_command_limit, dtype=np.float32),
                high=np.full(shape, self.wheel_command_limit, dtype=np.float32),
                dtype=np.float32,
            )

    def _apply_car_actuation(self) -> None:
        base = unwrap_env(self.env)
        task = getattr(base, "task", None)
        model = getattr(task, "model", None)
        if model is None:
            return
        if int(getattr(model, "nu", 0)) != 2:
            return

        names = []
        try:
            for i in range(int(model.nu)):
                names.append(str(model.actuator(i).name))
        except Exception:
            return
        if set(names) != {"left", "right"}:
            return

        if self._base_ctrlrange is None:
            self._base_ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64).copy()
        if self._base_forcerange is None:
            self._base_forcerange = np.asarray(model.actuator_forcerange, dtype=np.float64).copy()

        ctrl = np.asarray(self._base_ctrlrange, dtype=np.float64).copy()
        ctrl[:, 0] = -self.wheel_command_limit
        ctrl[:, 1] = self.wheel_command_limit
        model.actuator_ctrlrange[:] = ctrl

        if self.force_scale > 0.0:
            model.actuator_forcerange[:] = np.asarray(self._base_forcerange, dtype=np.float64) * self.force_scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_car_actuation()
        return obs, info


class CarThrottleTurnActionWrapper(gym.ActionWrapper):
    """Expose SafetyCar actions as [throttle, turn] instead of raw wheel commands."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base_space = getattr(env, "action_space", None)
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("CarThrottleTurnActionWrapper expects a Box action space.")
        low = np.asarray(base_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_space.high, dtype=np.float32).reshape(-1)
        if low.shape[0] < 2:
            raise ValueError("CarThrottleTurnActionWrapper expects at least 2 action dims.")
        self._base_low = low.copy()
        self._base_high = high.copy()
        self._wheel_limit = float(max(np.max(np.abs(low[:2])), np.max(np.abs(high[:2])), 1.0))
        self.action_mode = "throttle_turn"
        self.action_space = gym.spaces.Box(
            low=np.full((2,), -1.0, dtype=np.float32),
            high=np.full((2,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 2:
            arr = np.pad(arr, (0, 2 - arr.shape[0]), mode="constant")
        throttle = float(np.clip(arr[0], -1.0, 1.0))
        turn = float(np.clip(arr[1], -1.0, 1.0))
        # Preserve the same max wheel authority as the raw-wheel policy:
        # throttle=1 -> [wheel_limit, wheel_limit], turn=1 -> [-wheel_limit, wheel_limit].
        left = self._wheel_limit * np.clip(throttle - turn, -1.0, 1.0)
        right = self._wheel_limit * np.clip(throttle + turn, -1.0, 1.0)
        wheel_action = np.asarray([left, right], dtype=np.float32)
        return np.clip(wheel_action, self._base_low[:2], self._base_high[:2]).astype(np.float32, copy=False)

    def reverse_action(self, action):
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 2:
            arr = np.pad(arr, (0, 2 - arr.shape[0]), mode="constant")
        wheel_scale = max(1e-6, float(self._wheel_limit))
        throttle = 0.5 * (float(arr[0]) + float(arr[1])) / wheel_scale
        turn = 0.5 * (float(arr[1]) - float(arr[0])) / wheel_scale
        return np.clip(np.asarray([throttle, turn], dtype=np.float32), -1.0, 1.0)


class CarCardinalActionWrapper(gym.ActionWrapper):
    """Expose SafetyCar actions as 2D inputs snapped to 4 cardinal motion primitives."""

    def __init__(self, env: gym.Env, *, deadzone: float = 0.05):
        super().__init__(env)
        base_space = getattr(env, "action_space", None)
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("CarCardinalActionWrapper expects a Box action space.")
        low = np.asarray(base_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_space.high, dtype=np.float32).reshape(-1)
        if low.shape[0] < 2:
            raise ValueError("CarCardinalActionWrapper expects at least 2 action dims.")
        self._base_low = low.copy()
        self._base_high = high.copy()
        self._wheel_limit = float(max(np.max(np.abs(low[:2])), np.max(np.abs(high[:2])), 1.0))
        self._deadzone = float(max(0.0, deadzone))
        self.action_mode = "cardinal"
        self.action_space = gym.spaces.Box(
            low=np.full((2,), -1.0, dtype=np.float32),
            high=np.full((2,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 2:
            arr = np.pad(arr, (0, 2 - arr.shape[0]), mode="constant")
        throttle = float(np.clip(arr[0], -1.0, 1.0))
        turn = float(np.clip(arr[1], -1.0, 1.0))
        mag = max(abs(throttle), abs(turn))
        if mag <= self._deadzone:
            wheel_action = np.zeros((2,), dtype=np.float32)
        elif abs(throttle) >= abs(turn):
            sign = 1.0 if throttle >= 0.0 else -1.0
            wheel_action = np.asarray([sign * self._wheel_limit, sign * self._wheel_limit], dtype=np.float32)
        else:
            sign = 1.0 if turn >= 0.0 else -1.0
            wheel_action = np.asarray([-sign * self._wheel_limit, sign * self._wheel_limit], dtype=np.float32)
        return np.clip(wheel_action, self._base_low[:2], self._base_high[:2]).astype(np.float32, copy=False)

    def reverse_action(self, action):
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 2:
            arr = np.pad(arr, (0, 2 - arr.shape[0]), mode="constant")
        wheel_scale = max(1e-6, float(self._wheel_limit))
        throttle = 0.5 * (float(arr[0]) + float(arr[1])) / wheel_scale
        turn = 0.5 * (float(arr[1]) - float(arr[0])) / wheel_scale
        mag = max(abs(throttle), abs(turn))
        if mag <= self._deadzone:
            return np.zeros((2,), dtype=np.float32)
        if abs(throttle) >= abs(turn):
            return np.asarray([1.0 if throttle >= 0.0 else -1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0 if turn >= 0.0 else -1.0], dtype=np.float32)


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PointWorldVelocityActionWrapper(gym.ActionWrapper):
    """Expose SafetyPoint actions as desired world-frame velocity [vx, vy]."""

    def __init__(
        self,
        env: gym.Env,
        *,
        turn_gain: float = 2.5,
        alignment_power: float = 1.0,
        allow_backward: bool = False,
    ):
        super().__init__(env)
        base_space = getattr(env, "action_space", None)
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("PointWorldVelocityActionWrapper expects a Box action space.")
        low = np.asarray(base_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_space.high, dtype=np.float32).reshape(-1)
        if low.shape[0] < 2:
            raise ValueError("PointWorldVelocityActionWrapper expects at least 2 native action dims.")
        self._base_low = low.copy()
        self._base_high = high.copy()
        self._turn_gain = float(max(0.0, turn_gain))
        self._alignment_power = float(max(0.0, alignment_power))
        self._allow_backward = bool(allow_backward)
        self.action_mode = "world_velocity"
        self.action_space = gym.spaces.Box(
            low=np.full((2,), -1.0, dtype=np.float32),
            high=np.full((2,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        desired = np.asarray(action, dtype=np.float32).reshape(-1)
        if desired.shape[0] < 2:
            desired = np.pad(desired, (0, 2 - desired.shape[0]), mode="constant")
        vx = float(np.clip(desired[0], -1.0, 1.0))
        vy = float(np.clip(desired[1], -1.0, 1.0))
        speed = float(min(1.0, np.hypot(vx, vy)))
        if speed <= 1e-6:
            return np.zeros((2,), dtype=np.float32)

        forward_xy = extract_agent_forward_xy(self.env)
        if forward_xy is None:
            # SafetyPoint native action order is [forward, turn].
            return np.asarray([speed, 0.0], dtype=np.float32)

        desired_heading = float(np.arctan2(vy, vx))
        current_heading = float(np.arctan2(float(forward_xy[1]), float(forward_xy[0])))
        err = _wrap_angle(desired_heading - current_heading)
        turn = float(np.clip(self._turn_gain * err / np.pi, -1.0, 1.0))

        alignment = float(np.cos(err))
        if self._allow_backward:
            forward = speed * np.sign(alignment) * (abs(alignment) ** self._alignment_power)
        else:
            forward = speed * (max(0.0, alignment) ** self._alignment_power)
        native = np.asarray([forward, turn], dtype=np.float32)
        return np.clip(native, self._base_low[:2], self._base_high[:2]).astype(np.float32, copy=False)


class GoalOnlyLidarObservationWrapper(gym.Wrapper):
    """Zero all non-goal lidar channels while preserving agent proprio and goal lidar."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base = unwrap_env(env)
        task = getattr(base, "task", None)
        obs_space_dict = getattr(getattr(task, "obs_info", None), "obs_space_dict", None)
        if not isinstance(obs_space_dict, gym.spaces.Dict):
            raise TypeError("GoalOnlyLidarObservationWrapper requires a Dict-backed Safety-Gym observation space.")
        self._masked_keys = tuple(
            key
            for key in obs_space_dict.spaces.keys()
            if key.endswith("_lidar") and key != "goal_lidar"
        )
        self._flattened = bool(getattr(task, "observation_flatten", True))
        self._flat_slices: dict[str, slice] = {}
        offset = 0
        for key, space in obs_space_dict.spaces.items():
            width = int(gym.spaces.utils.flatdim(space))
            self._flat_slices[key] = slice(offset, offset + width)
            offset += width
        self.observation_space = env.observation_space

    def _mask_observation(self, observation):
        if not self._masked_keys:
            return observation
        if self._flattened:
            arr = np.asarray(observation).copy()
            for key in self._masked_keys:
                sl = self._flat_slices.get(key)
                if sl is not None:
                    arr[sl] = 0
            return arr
        masked = dict(observation)
        for key in self._masked_keys:
            if key in masked:
                masked[key] = np.zeros_like(masked[key])
        return masked

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._mask_observation(observation), info

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 6:
            observation, reward, cost, terminated, truncated, info = out
            return self._mask_observation(observation), reward, cost, terminated, truncated, info
        if len(out) == 5:
            observation, reward, terminated, truncated, info = out
            return self._mask_observation(observation), reward, terminated, truncated, info
        raise ValueError(f"Unexpected env.step() output length for GoalOnlyLidarObservationWrapper: {len(out)}")


def ensure_safety_gymnasium_importable() -> None:
    """Add the local safety-gymnasium checkout to sys.path when present."""
    repo_root = Path(__file__).resolve().parent.parent
    local_pkg = repo_root / "safety-gymnasium"
    if local_pkg.exists():
        local_pkg_str = str(local_pkg)
        if local_pkg_str not in sys.path:
            sys.path.insert(0, local_pkg_str)


def make_safety_env(
    env_name: str,
    *,
    render_mode: Optional[str] = None,
    max_episode_steps: int = 0,
    surface_mode: str = "default",
    car_wheel_command_limit: float = 2.0,
    car_force_scale: float = 2.0,
    car_action_mode: str = "raw_wheels",
    point_action_mode: str = "native",
    point_turn_gain: float = 2.5,
    point_alignment_power: float = 1.0,
    point_allow_backward: bool = False,
    obs_mask_mode: str = "none",
    seed: Optional[int] = None,
):
    ensure_safety_gymnasium_importable()
    import safety_gymnasium  # type: ignore

    kwargs = {}
    if render_mode is not None and render_mode != "none":
        kwargs["render_mode"] = render_mode
    if max_episode_steps > 0:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    env = safety_gymnasium.make(env_name, **kwargs)
    if "car" in str(env_name).lower():
        env = CarActuationWrapper(
            env,
            wheel_command_limit=car_wheel_command_limit,
            force_scale=car_force_scale,
        )
        action_mode = str(car_action_mode).strip().lower()
        if action_mode not in {"raw_wheels", "throttle_turn", "cardinal"}:
            raise ValueError(f"Unsupported car_action_mode: {car_action_mode}")
        if action_mode == "throttle_turn":
            env = CarThrottleTurnActionWrapper(env)
        elif action_mode == "cardinal":
            env = CarCardinalActionWrapper(env)
    elif "point" in str(env_name).lower():
        action_mode = str(point_action_mode).strip().lower()
        if action_mode not in {"native", "world_velocity"}:
            raise ValueError(f"Unsupported point_action_mode: {point_action_mode}")
        if action_mode == "world_velocity":
            env = PointWorldVelocityActionWrapper(
                env,
                turn_gain=float(point_turn_gain),
                alignment_power=float(point_alignment_power),
                allow_backward=bool(point_allow_backward),
            )
    obs_mask_mode_l = str(obs_mask_mode).strip().lower()
    if obs_mask_mode_l not in {"none", "goal_only_lidar"}:
        raise ValueError(f"Unsupported obs_mask_mode: {obs_mask_mode}")
    if obs_mask_mode_l == "goal_only_lidar":
        env = GoalOnlyLidarObservationWrapper(env)
    env = SurfaceConfigWrapper(env, mode=surface_mode)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def resolve_control_scheme(
    env_name: str,
    *,
    car_action_mode: str = "raw_wheels",
    point_action_mode: str = "native",
) -> str:
    name = str(env_name).lower()
    if "car" in name:
        return (
            "planar_velocity"
            if str(car_action_mode).strip().lower() in {"throttle_turn", "cardinal"}
            else "differential_wheels"
        )
    if "point" in name and str(point_action_mode).strip().lower() == "world_velocity":
        return "world_velocity"
    return "planar_velocity"


def unwrap_env(env):
    base = env
    visited = set()
    while hasattr(base, "unwrapped") and getattr(base, "unwrapped") is not base and id(base) not in visited:
        visited.add(id(base))
        base = base.unwrapped
    return base


def extract_goal_distance(env) -> float:
    """Best-effort goal distance read; returns NaN when unavailable."""
    base = unwrap_env(env)
    try:
        task = getattr(base, "task", None)
        if task is not None and hasattr(task, "dist_goal"):
            return float(task.dist_goal())
    except Exception:
        pass
    return float("nan")


def extract_goal_xy(env) -> Optional[np.ndarray]:
    """Best-effort current goal xy position; returns None when unavailable."""
    base = unwrap_env(env)
    try:
        task = getattr(base, "task", None)
        goal = getattr(task, "goal", None)
        pos = np.asarray(getattr(goal, "pos", None), dtype=np.float64).reshape(-1)
        if pos.size >= 2 and np.isfinite(pos[:2]).all():
            return pos[:2].astype(np.float64, copy=True)
    except Exception:
        pass
    return None


def extract_agent_xy(env) -> Optional[np.ndarray]:
    """Best-effort agent xy position; returns None when unavailable."""
    base = unwrap_env(env)
    try:
        task = getattr(base, "task", None)
        agent = getattr(task, "agent", None)
        pos = np.asarray(getattr(agent, "pos", None), dtype=np.float64).reshape(-1)
        if pos.size >= 2 and np.isfinite(pos[:2]).all():
            return pos[:2].astype(np.float64, copy=True)
    except Exception:
        pass
    return None


def extract_agent_velocity_xy(env) -> Optional[np.ndarray]:
    """Best-effort agent xy velocity; returns None when unavailable."""
    base = unwrap_env(env)
    try:
        task = getattr(base, "task", None)
        agent = getattr(task, "agent", None)
        vel = np.asarray(getattr(agent, "vel", None), dtype=np.float64).reshape(-1)
        if vel.size >= 2 and np.isfinite(vel[:2]).all():
            return vel[:2].astype(np.float64, copy=True)
    except Exception:
        pass
    return None


def extract_agent_forward_xy(env) -> Optional[np.ndarray]:
    """Best-effort normalized forward xy direction from the agent body pose."""
    base = unwrap_env(env)
    try:
        task = getattr(base, "task", None)
        agent = getattr(task, "agent", None)
        mat = np.asarray(getattr(agent, "mat", None), dtype=np.float64).reshape(3, 3)
        if np.isfinite(mat).all():
            agent_class = type(agent).__name__.lower()
            if agent_class == "point":
                forward = mat[:2, 0].astype(np.float64, copy=False)
            else:
                # Matches the existing topdown renderer convention for SafetyCar.
                forward = (-mat[:2, 1]).astype(np.float64, copy=False)
            norm = float(np.linalg.norm(forward))
            if norm > 1e-6:
                return (forward / norm).astype(np.float64, copy=False)
    except Exception:
        pass
    return None


def _strip_numeric_suffix(name: str) -> str:
    out = str(name)
    while out and out[-1].isdigit():
        out = out[:-1]
    for suffix in ("obj", "mocap"):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out or str(name)


def _task_constrained_object_specs(task) -> list[dict]:
    cached = getattr(task, "_codex_constrained_object_specs", None)
    if isinstance(cached, list):
        return cached

    cfg = getattr(getattr(task, "world_info", None), "world_config_dict", None)
    if not isinstance(cfg, dict):
        setattr(task, "_codex_constrained_object_specs", [])
        return []

    plural_map = {
        "hazard": "hazards",
        "vase": "vases",
        "pillar": "pillars",
        "gremlin": "gremlins",
        "button": "buttons",
    }
    specs: list[dict] = []
    for section in ("geoms", "free_geoms", "mocaps"):
        items = cfg.get(section, {})
        if not isinstance(items, dict):
            continue
        for body_name, body_cfg in items.items():
            if not isinstance(body_cfg, dict):
                continue
            geoms = body_cfg.get("geoms", [])
            if not geoms:
                continue
            geom = geoms[0]
            label = _strip_numeric_suffix(str(body_name)).lower()
            task_attr = plural_map.get(label, f"{label}s")
            obj = getattr(task, task_attr, None)
            if obj is None or not bool(getattr(obj, "is_constrained", False)):
                continue
            size_arr = np.asarray(geom.get("size", [0.1, 0.1, 0.1]), dtype=np.float64).reshape(-1)
            size = float(size_arr[0]) if size_arr.size else 0.1
            keepout = float(getattr(obj, "keepout", size))
            specs.append(
                {
                    "body_name": str(body_name),
                    "label": str(label),
                    "size": float(size),
                    "keepout": float(keepout),
                }
            )

    setattr(task, "_codex_constrained_object_specs", specs)
    return specs


def extract_min_constrained_clearance(env) -> float:
    """Best-effort keepout-based clearance to the nearest constrained obstacle."""
    base = unwrap_env(env)
    task = getattr(base, "task", None)
    if task is None:
        return float("nan")

    agent_xy = extract_agent_xy(env)
    if agent_xy is None:
        return float("nan")

    agent = getattr(task, "agent", None)
    agent_keepout = float(getattr(agent, "keepout", 0.0) or 0.0)
    specs = _task_constrained_object_specs(task)
    if not specs:
        return float("nan")

    min_clearance = float("inf")
    for spec in specs:
        try:
            body = task.data.body(str(spec["body_name"]))
            pos = np.asarray(body.xpos, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if pos.size < 2 or not np.isfinite(pos[:2]).all():
            continue
        dist = float(np.linalg.norm(pos[:2] - agent_xy))
        clearance = dist - (agent_keepout + float(spec["keepout"]))
        if clearance < min_clearance:
            min_clearance = clearance

    if np.isfinite(min_clearance):
        return float(min_clearance)
    return float("nan")


def extract_step_limit(env) -> int:
    """Return effective per-episode step limit."""
    values: list[int] = []

    try:
        spec = getattr(env, "spec", None)
        max_steps = getattr(spec, "max_episode_steps", None)
        if max_steps is not None:
            values.append(int(max_steps))
    except Exception:
        pass

    try:
        base = unwrap_env(env)
        task = getattr(base, "task", None)
        num_steps = getattr(task, "num_steps", None)
        if num_steps is not None:
            values.append(int(num_steps))
    except Exception:
        pass

    clean = [v for v in values if v > 0]
    if not clean:
        return 1000
    return min(clean)


def clip_action_to_space(action: np.ndarray, action_space) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    if arr.shape[0] != low.shape[0]:
        if arr.shape[0] < low.shape[0]:
            arr = np.pad(arr, (0, low.shape[0] - arr.shape[0]), mode="constant")
        else:
            arr = arr[: low.shape[0]]
    return np.clip(arr, low, high).astype(np.float32, copy=False)


def scale_action_np(action: np.ndarray, action_space) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    if arr.shape[0] != low.shape[0]:
        if arr.shape[0] < low.shape[0]:
            arr = np.pad(arr, (0, low.shape[0] - arr.shape[0]), mode="constant")
        else:
            arr = arr[: low.shape[0]]
    center = 0.5 * (high + low)
    half = 0.5 * (high - low)
    return (center + arr * half).astype(np.float32, copy=False)
