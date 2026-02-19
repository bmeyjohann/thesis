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
    env = SurfaceConfigWrapper(env, mode=surface_mode)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


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
