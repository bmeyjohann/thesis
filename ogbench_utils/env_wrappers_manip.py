from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
import warnings

from .env_wrappers_common import FixedResetSeedWrapper, maybe_wrap_intervention

_WRAPPER_STACK_PRINTED: set[tuple] = set()

CUBE_REWARD_MODE_CHOICES = (
    "sparse_final",
    "sparse_intermediate",
    "dense",
)

def canonicalize_cube_reward_mode(mode: str) -> str:
    mode_key = str(mode or "dense").strip()
    if mode_key not in CUBE_REWARD_MODE_CHOICES:
        raise ValueError(
            f"Unknown cube_reward_mode={mode_key!r}. "
            f"Expected one of: {', '.join(CUBE_REWARD_MODE_CHOICES)}"
        )
    return mode_key


class ManipDetailedRewardWrapper(gym.Wrapper):
    """Manip-only reward wrapper with sparse/dense/combined/none modes."""

    def __init__(
        self,
        env: gym.Env,
        *,
        reward_type: str = "sparse",
        dense_reward_scale: float = 1.0,
        goal_reward: float = 1.0,
        step_penalty: float = 0.0,
        switch_reward_to_sparse_after_steps_per_env: int = 0,
        normalize_success_reward: bool = True,
    ):
        super().__init__(env)
        self.reward_type = str(reward_type)
        self.dense_reward_scale = float(dense_reward_scale)
        self.goal_reward = float(goal_reward)
        self.step_penalty = float(step_penalty)
        self.switch_reward_to_sparse_after_steps_per_env = int(switch_reward_to_sparse_after_steps_per_env or 0)
        self.normalize_success_reward = bool(normalize_success_reward)
        self._global_step_env = 0
        self._goal_reached = False
        self._episode_steps = 0
        self._episode_sparse_reward = 0.0
        self._episode_dense_reward = 0.0
        self._last_sparse_reward_step = 0.0
        self._last_dense_reward_step = 0.0

    def _effective_reward_type(self) -> str:
        if self.reward_type == "none":
            return "none"
        if (
            self.switch_reward_to_sparse_after_steps_per_env > 0
            and self._global_step_env >= self.switch_reward_to_sparse_after_steps_per_env
        ):
            return "sparse"
        return self.reward_type

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._goal_reached = False
        self._episode_steps = 0
        self._episode_sparse_reward = 0.0
        self._episode_dense_reward = 0.0
        self._last_sparse_reward_step = 0.0
        self._last_dense_reward_step = 0.0
        if isinstance(info, dict):
            info.update(
                {
                    "episode_sparse_reward": 0.0,
                    "episode_dense_reward": 0.0,
                    "episode_steps": 0,
                    "goal_reached": False,
                    "reward_type": self._effective_reward_type(),
                }
            )
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if isinstance(info, dict) else {}

        raw_success = info.get("success", False)
        if isinstance(raw_success, (np.ndarray, list, tuple)):
            raw_arr = np.asarray(raw_success).reshape(-1)
            success_flag = bool(raw_arr[0]) if raw_arr.size else False
        else:
            success_flag = bool(raw_success)
        success_flag = bool(success_flag or info.get("goal_reached", False))

        sparse_reward_raw = self.goal_reward if success_flag else 0.0
        # Manip dense signal follows environment-native scalar reward, scaled if requested.
        dense_reward_raw = float(env_reward) * self.dense_reward_scale
        sparse_reward = float(sparse_reward_raw)
        dense_reward = float(dense_reward_raw)

        effective_type = self._effective_reward_type()
        if effective_type == "sparse":
            dense_reward = 0.0
        elif effective_type == "dense":
            sparse_reward = 0.0
        elif effective_type == "none":
            sparse_reward = 0.0
            dense_reward = 0.0
        elif effective_type != "combined":
            raise ValueError(f"Unknown reward_type: {effective_type}")

        # Optional success normalization for sparse-like task rewards.
        if self.normalize_success_reward and effective_type in ("sparse", "combined"):
            sparse_reward = self.goal_reward if success_flag else 0.0

        total_reward_raw = float(sparse_reward_raw + dense_reward_raw - self.step_penalty)
        total_reward = float(sparse_reward + dense_reward - self.step_penalty)

        self._last_sparse_reward_step = float(sparse_reward)
        self._last_dense_reward_step = float(dense_reward)
        self._episode_sparse_reward += float(sparse_reward)
        self._episode_dense_reward += float(dense_reward)
        self._episode_steps += 1
        self._global_step_env += 1
        if success_flag:
            self._goal_reached = True

        info.update(
            {
                "sparse_reward": float(self._last_sparse_reward_step),
                "dense_reward": float(self._last_dense_reward_step),
                "total_reward": float(total_reward),
                "sparse_reward_raw": float(sparse_reward_raw),
                "dense_reward_raw": float(dense_reward_raw),
                "total_reward_raw": float(total_reward_raw),
                "episode_sparse_reward": float(self._episode_sparse_reward),
                "episode_dense_reward": float(self._episode_dense_reward),
                "episode_steps": int(self._episode_steps),
                "goal_reached": bool(self._goal_reached),
                "goal_reached_from_success": bool(success_flag),
                "reward_type": effective_type,
            }
        )
        return obs, total_reward, terminated, truncated, info


class ManipGoalConditionedObsWrapper(gym.Wrapper):
    """Append goal state to manipulation state observations."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_goal: Optional[np.ndarray] = None
        self._goal_dim: Optional[int] = None
        self._warned_goal_mismatch = False

        obs_space = getattr(env, "observation_space", None)
        if isinstance(obs_space, gym.spaces.Box):
            obs_dim = int(np.prod(obs_space.shape))
            self._goal_dim = obs_dim
            low = np.asarray(obs_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(obs_space.high, dtype=np.float32).reshape(-1)
            goal_low = np.full((obs_dim,), -np.inf, dtype=np.float32)
            goal_high = np.full((obs_dim,), np.inf, dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                low=np.concatenate([low, goal_low], axis=0),
                high=np.concatenate([high, goal_high], axis=0),
                dtype=np.float32,
            )

    def _extract_goal(self, info: Optional[dict]) -> Optional[np.ndarray]:
        if not isinstance(info, dict):
            return None
        goal = info.get("goal", None)
        if goal is None:
            return None
        try:
            return np.asarray(goal, dtype=np.float32).reshape(-1)
        except Exception:
            return None

    def _compose_obs(self, obs: np.ndarray) -> np.ndarray:
        obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self._goal_dim is None:
            self._goal_dim = int(obs_vec.shape[0])

        goal_vec = self._last_goal
        if goal_vec is None:
            goal_vec = np.zeros((self._goal_dim,), dtype=np.float32)
        else:
            goal_vec = np.asarray(goal_vec, dtype=np.float32).reshape(-1)
            if goal_vec.shape[0] != self._goal_dim:
                if not self._warned_goal_mismatch:
                    warnings.warn(
                        "ManipGoalConditionedObsWrapper: goal dim does not match obs dim; "
                        "falling back to zero goal vector for this episode."
                    )
                    self._warned_goal_mismatch = True
                goal_vec = np.zeros((self._goal_dim,), dtype=np.float32)
        return np.concatenate([obs_vec, goal_vec], axis=0).astype(np.float32, copy=False)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_goal = self._extract_goal(info)
        return self._compose_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        goal = self._extract_goal(info)
        if goal is not None:
            self._last_goal = goal
        return self._compose_obs(obs), reward, terminated, truncated, info


class ManipRelativeCubeFeaturesWrapper(gym.Wrapper):
    """Append target-relative cube features to state observations."""

    FEATURE_DIM = 10

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = getattr(env, "observation_space", None)
        if isinstance(obs_space, gym.spaces.Box):
            low = np.asarray(obs_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(obs_space.high, dtype=np.float32).reshape(-1)
            feat_low = np.full((self.FEATURE_DIM,), -np.inf, dtype=np.float32)
            feat_high = np.full((self.FEATURE_DIM,), np.inf, dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                low=np.concatenate([low, feat_low], axis=0),
                high=np.concatenate([high, feat_high], axis=0),
                dtype=np.float32,
            )

    @staticmethod
    def _vec3(value) -> np.ndarray:
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size >= 3:
                return arr[:3].astype(np.float32, copy=False)
        except Exception:
            pass
        return np.zeros((3,), dtype=np.float32)

    @staticmethod
    def _scalar(value, default: float = 0.0) -> float:
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size > 0 and np.isfinite(arr[0]):
                return float(arr[0])
        except Exception:
            pass
        return float(default)

    @staticmethod
    def _wrap_angle(x: float) -> float:
        return float((x + np.pi) % (2.0 * np.pi) - np.pi)

    def _compute_features(self, info: Optional[dict]) -> np.ndarray:
        if not isinstance(info, dict):
            return np.zeros((self.FEATURE_DIM,), dtype=np.float32)
        target_block = int(self._scalar(info.get("privileged/target_block", 0.0), default=0.0))
        eff_pos = self._vec3(info.get("proprio/effector_pos"))
        eff_yaw = self._scalar(info.get("proprio/effector_yaw"), default=0.0)
        cube_pos = self._vec3(info.get(f"privileged/block_{target_block}_pos"))
        cube_yaw = self._scalar(info.get(f"privileged/block_{target_block}_yaw"), default=eff_yaw)
        goal_pos = self._vec3(info.get("privileged/target_block_pos"))
        goal_yaw = self._scalar(info.get("privileged/target_block_yaw"), default=cube_yaw)

        eff_to_cube = cube_pos - eff_pos
        cube_to_goal = goal_pos - cube_pos
        eff_to_cube_dist = float(np.linalg.norm(eff_to_cube))
        cube_to_goal_dist = float(np.linalg.norm(cube_to_goal))
        eff_cube_yaw_delta = self._wrap_angle(cube_yaw - eff_yaw)
        cube_goal_yaw_delta = self._wrap_angle(goal_yaw - cube_yaw)
        features = np.concatenate(
            [
                eff_to_cube.astype(np.float32, copy=False),
                np.asarray([eff_to_cube_dist, eff_cube_yaw_delta], dtype=np.float32),
                cube_to_goal.astype(np.float32, copy=False),
                np.asarray([cube_to_goal_dist, cube_goal_yaw_delta], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        return features

    def _augment(self, obs, info: Optional[dict]):
        obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
        feat = self._compute_features(info)
        if isinstance(info, dict):
            info["diag/obs_rel_eff_to_cube_dist"] = float(feat[3])
            info["diag/obs_rel_eff_to_cube_yaw_delta"] = float(feat[4])
            info["diag/obs_rel_cube_to_goal_dist"] = float(feat[8])
            info["diag/obs_rel_cube_to_goal_yaw_delta"] = float(feat[9])
            info["diag/obs_relative_feature_dim"] = int(self.FEATURE_DIM)
        return np.concatenate([obs_vec, feat], axis=0).astype(np.float32, copy=False), info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_aug, info = self._augment(obs, info)
        return obs_aug, reward, terminated, truncated, info


class ManipRelativeOnlyObsWrapper(gym.Wrapper):
    """Keep only compact proprioception and optional relative cube features."""

    PROPRIO_DIM = 7  # eff_xyz(3), eff_yaw_cos_sin(2), gripper_open/contact(2)

    def __init__(
        self,
        env: gym.Env,
        *,
        include_relative_cube_features: bool,
        relative_feature_dim: int = ManipRelativeCubeFeaturesWrapper.FEATURE_DIM,
    ):
        super().__init__(env)
        self.include_relative_cube_features = bool(include_relative_cube_features)
        self.relative_feature_dim = int(max(0, relative_feature_dim))
        obs_dim = self.PROPRIO_DIM + (self.relative_feature_dim if self.include_relative_cube_features else 0)
        self.observation_space = gym.spaces.Box(
            low=np.full((obs_dim,), -np.inf, dtype=np.float32),
            high=np.full((obs_dim,), np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    @staticmethod
    def _infer_num_cubes(base_dim: int) -> Optional[int]:
        for n_cubes in range(1, 9):
            rem = int(base_dim) - (7 + 9 * n_cubes)
            if rem >= 0 and rem % 2 == 0:
                return int(n_cubes)
        return None

    def _split_obs(self, obs_vec: np.ndarray, info: Optional[dict]) -> tuple[np.ndarray, np.ndarray]:
        core = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
        rel = np.zeros((0,), dtype=np.float32)
        if self.include_relative_cube_features and core.size >= self.relative_feature_dim:
            rel = core[-self.relative_feature_dim :].astype(np.float32, copy=False)
            core = core[: -self.relative_feature_dim]

        if isinstance(info, dict):
            goal = info.get("goal")
            if goal is not None:
                try:
                    goal_vec = np.asarray(goal, dtype=np.float32).reshape(-1)
                    if goal_vec.size > 0 and core.size == 2 * goal_vec.size:
                        core = core[: goal_vec.size]
                except Exception:
                    pass

        return core, rel

    def _extract_proprio(self, core_obs: np.ndarray) -> np.ndarray:
        base_dim = int(core_obs.size)
        n_cubes = self._infer_num_cubes(base_dim)
        if n_cubes is None:
            out = np.zeros((self.PROPRIO_DIM,), dtype=np.float32)
            n_take = min(self.PROPRIO_DIM, base_dim)
            if n_take > 0:
                out[:n_take] = core_obs[:n_take]
            return out

        rem = int(base_dim) - (7 + 9 * n_cubes)
        n_joint = rem // 2
        i = 2 * n_joint
        eff_xyz = core_obs[i : i + 3]
        i += 3
        eff_yaw = core_obs[i : i + 2]
        i += 2
        grip_open = core_obs[i : i + 1]
        i += 1
        grip_contact = core_obs[i : i + 1]
        return np.concatenate([eff_xyz, eff_yaw, grip_open, grip_contact], axis=0).astype(np.float32, copy=False)

    def _compose(self, obs, info: Optional[dict]) -> tuple[np.ndarray, Optional[dict]]:
        obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
        core, rel = self._split_obs(obs_vec, info)
        proprio = self._extract_proprio(core)
        if self.include_relative_cube_features:
            rel_safe = rel
            if rel_safe.size != self.relative_feature_dim:
                rel_safe = np.zeros((self.relative_feature_dim,), dtype=np.float32)
            out = np.concatenate([proprio, rel_safe], axis=0).astype(np.float32, copy=False)
        else:
            out = proprio

        if isinstance(info, dict):
            info["diag/obs_relative_only_active"] = 1
            info["diag/obs_relative_only_dim"] = int(out.size)
            info["diag/obs_relative_only_proprio_dim"] = int(self.PROPRIO_DIM)
            info["diag/obs_relative_only_rel_dim"] = int(self.relative_feature_dim if self.include_relative_cube_features else 0)
        return out, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._compose(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_new, info = self._compose(obs, info)
        return obs_new, reward, terminated, truncated, info


class ManipDisableRotationActionWrapper(gym.ActionWrapper):
    """Expose manipulation control as 4D xyz + gripper while fixing yaw internally."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-np.ones((4,), dtype=np.float32),
            high=np.ones((4,), dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

    @staticmethod
    def _project_action_like(value):
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            return value
        if arr.ndim == 0 or arr.shape[-1] < 5:
            return value
        return np.concatenate([arr[..., :3], arr[..., 4:5]], axis=-1).astype(np.float32, copy=False)

    def action(self, action):
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 4:
            raise ValueError(f"Expected 4D xyz+gripper action, got shape={tuple(arr.shape)}")
        full = np.zeros((5,), dtype=np.float32)
        full[:3] = arr[:3]
        full[4] = arr[3]
        return full

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info = dict(info) if isinstance(info, dict) else {}
        for key in ("teacher_action", "student_action", "teacher_actions", "student_actions", "applied_action", "applied_actions"):
            if key in info:
                info[key] = self._project_action_like(info.get(key))
        return obs, reward, terminated, truncated, info


def is_cube_env_name(env_name: Optional[str]) -> bool:
    return "cube" in str(env_name or "").lower()


def cube_reward_mode_active(*, env_name: Optional[str], obs_mode: str, reward_mode: str) -> bool:
    mode = canonicalize_cube_reward_mode(reward_mode)
    return mode in CUBE_REWARD_MODE_CHOICES and obs_mode == "state" and is_cube_env_name(env_name)


class CubeRewardModeTracker:
    """Stateful reward recomposition for cube state runs."""

    def __init__(
        self,
        *,
        mode: str,
        num_envs: int,
        device: torch.device,
        success_reward: float = 1.0,
        grasp_reward: float = 0.25,
        place_reward: float = 1.0,
        drop_penalty: float = 0.0,
        grasp_error_threshold: float = 0.08,
        progress_scale: float = 5.0,
        progress_clip: float = 0.05,
    ):
        self.mode = canonicalize_cube_reward_mode(mode)
        if self.mode not in CUBE_REWARD_MODE_CHOICES:
            raise ValueError(f"Unknown cube reward mode: {self.mode}")
        self.num_envs = int(max(1, num_envs))
        self.device = device
        self.success_reward = float(success_reward)
        self.grasp_reward = float(grasp_reward)
        self.place_reward = float(place_reward)
        self.drop_penalty = float(drop_penalty)
        self.grasp_error_threshold = float(grasp_error_threshold)
        self.progress_scale = float(progress_scale)
        self.progress_clip = float(max(0.0, progress_clip))
        self._reach_tolerance = 0.06
        self._grasp_contact_threshold = 0.30
        self._release_contact_threshold = 0.15
        self._dense_gripper_bonus = 0.05
        self._dense_release_bonus = 0.10
        self._dense_movement_penalty_scale = 0.05

        self._prev_cubes_solved: Optional[torch.Tensor] = None
        self._prev_max_error: Optional[torch.Tensor] = None
        self._prev_eff_block_dist: Optional[torch.Tensor] = None
        self._prev_block_target_dist: Optional[torch.Tensor] = None
        self._prev_eff_pos: Optional[torch.Tensor] = None
        self._prev_target_cube_pos: Optional[torch.Tensor] = None
        self._prev_target_block: Optional[torch.Tensor] = None
        self._phase: Optional[torch.Tensor] = None
        self._dense_phase_cumulative: Optional[torch.Tensor] = None
        self._obs_layout_cache: dict[tuple[int, int], Optional[dict[str, object]]] = {}
        self._success_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._num_cubes: Optional[int] = None

    def _as_batch(self, value, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if torch.is_tensor(value):
            out = value.to(device=self.device, dtype=dtype)
        else:
            out = torch.as_tensor(value, device=self.device, dtype=dtype)
        if out.ndim == 0:
            out = out.repeat(self.num_envs)
        elif out.shape[0] == 1 and self.num_envs > 1:
            reps = [1] * out.ndim
            reps[0] = self.num_envs
            out = out.repeat(*reps)
        return out.view(self.num_envs)

    def _extract_scalar_batch(self, infos, key: str, *, default: float = 0.0) -> torch.Tensor:
        if not isinstance(infos, dict) or key not in infos:
            return torch.full((self.num_envs,), float(default), dtype=torch.float32, device=self.device)
        try:
            raw = infos[key]
            out = torch.as_tensor(raw, device=self.device, dtype=torch.float32)
            if out.ndim == 0:
                return out.repeat(self.num_envs)
            if out.ndim == 1:
                if out.numel() == self.num_envs:
                    return out
                if out.numel() == 1:
                    return out.repeat(self.num_envs)
            if out.ndim >= 2 and out.shape[0] == self.num_envs:
                return out.reshape(self.num_envs, -1)[:, 0]
            if out.numel() == self.num_envs:
                return out.reshape(self.num_envs)
        except Exception:
            pass
        return torch.full((self.num_envs,), float(default), dtype=torch.float32, device=self.device)

    def _extract_vec3_batch(self, infos, key: str) -> Optional[torch.Tensor]:
        if not isinstance(infos, dict) or key not in infos:
            return None
        try:
            out = torch.as_tensor(infos[key], device=self.device, dtype=torch.float32)
            if out.ndim == 1 and out.numel() == 3:
                return out.view(1, 3).repeat(self.num_envs, 1)
            if out.ndim >= 2 and out.shape[0] == self.num_envs:
                flat = out.reshape(self.num_envs, -1)
                if flat.shape[1] >= 3:
                    return flat[:, :3]
            if out.numel() == self.num_envs * 3:
                return out.reshape(self.num_envs, 3)
        except Exception:
            return None
        return None

    def _extract_success_batch(self, infos, cubes_solved: torch.Tensor, cubes_total: torch.Tensor) -> torch.Tensor:
        success = self._extract_scalar_batch(infos, "success", default=0.0)
        if torch.all(success <= 0.0):
            success = ((cubes_total > 0.0) & (cubes_solved >= cubes_total - 1e-6)).float()
        return success

    def _extract_log_scalar_batch(self, infos, key: str, *, default: float = 0.0) -> torch.Tensor:
        if not isinstance(infos, dict):
            return torch.full((self.num_envs,), float(default), dtype=torch.float32, device=self.device)
        log = infos.get("log", None)
        if not isinstance(log, dict) or key not in log:
            return torch.full((self.num_envs,), float(default), dtype=torch.float32, device=self.device)
        try:
            out = torch.as_tensor(log[key], device=self.device, dtype=torch.float32)
            if out.ndim == 0:
                return out.repeat(self.num_envs)
            if out.ndim == 1:
                if out.numel() == self.num_envs:
                    return out
                if out.numel() == 1:
                    return out.repeat(self.num_envs)
            if out.ndim >= 2 and out.shape[0] == self.num_envs:
                return out.reshape(self.num_envs, -1)[:, 0]
            if out.numel() == self.num_envs:
                return out.reshape(self.num_envs)
        except Exception:
            pass
        return torch.full((self.num_envs,), float(default), dtype=torch.float32, device=self.device)

    def _extract_obs_tensor_from_infos(self, infos) -> Optional[torch.Tensor]:
        if not isinstance(infos, dict) or "observations" not in infos:
            return None
        raw = infos["observations"]
        # Vec-env often wraps observations as {"raw": {"obs": tensor(...)}}.
        for _ in range(4):
            if not isinstance(raw, dict):
                break
            next_raw = None
            for key in ("obs", "observation", "state", "policy", "raw"):
                if key in raw:
                    next_raw = raw[key]
                    break
            if next_raw is None:
                break
            raw = next_raw
        try:
            out = torch.as_tensor(raw, device=self.device, dtype=torch.float32)
        except Exception:
            return None
        if out.ndim == 0:
            return None
        if out.ndim == 1:
            if self.num_envs == 1:
                return out.view(1, -1)
            if out.numel() % self.num_envs == 0:
                return out.view(self.num_envs, -1)
            return None
        if out.ndim >= 2 and out.shape[0] == self.num_envs:
            return out.reshape(self.num_envs, -1)
        if out.numel() % self.num_envs == 0:
            return out.reshape(self.num_envs, -1)
        return None

    def _split_obs_goal(self, obs_full: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        obs_full = obs_full.reshape(self.num_envs, -1)
        dim = int(obs_full.shape[1])
        if dim % 2 != 0:
            return obs_full, None
        half = dim // 2
        first = obs_full[:, :half]
        second = obs_full[:, half:]
        # Goal-conditioned wrapper outputs [obs, goal]; halves differ in normal operation.
        if torch.max(torch.abs(first - second)).item() > 1e-6:
            return first, second
        return obs_full, None

    def _infer_num_cubes_from_obs_dim(self, obs_dim: int) -> Optional[int]:
        for n in range(1, 8):
            base = int(obs_dim) - 9 * n
            rem = base - 7
            if rem >= 0 and rem % 2 == 0:
                return n
        return None

    def _infer_obs_layout(self, obs_dim: int, num_cubes: int) -> Optional[dict[str, object]]:
        key = (int(obs_dim), int(num_cubes))
        if key in self._obs_layout_cache:
            return self._obs_layout_cache[key]
        block_width = 9
        base = int(obs_dim) - int(num_cubes) * block_width
        rem = base - 7
        if rem < 0 or rem % 2 != 0:
            self._obs_layout_cache[key] = None
            return None
        n_joint = rem // 2
        i = 2 * n_joint
        eff_pos = (i, i + 3)
        i += 3
        i += 2  # cos/sin yaw
        gripper_open_idx = i
        i += 1
        gripper_contact_idx = i
        i += 1
        block_pos = []
        for _ in range(num_cubes):
            block_pos.append((i, i + 3))
            i += block_width
        if i != int(obs_dim):
            self._obs_layout_cache[key] = None
            return None
        layout = {
            "eff_pos": eff_pos,
            "gripper_open_idx": gripper_open_idx,
            "gripper_contact_idx": gripper_contact_idx,
            "block_pos": block_pos,
        }
        self._obs_layout_cache[key] = layout
        return layout

    def _infer_num_cubes(self, infos) -> int:
        if self._num_cubes is not None and self._num_cubes > 0:
            return self._num_cubes
        total = int(round(float(self._extract_scalar_batch(infos, "diag/cubes_total", default=0.0).max().item())))
        if total > 0:
            self._num_cubes = total
            return total
        num_from_keys = 0
        if isinstance(infos, dict):
            for key in infos.keys():
                if isinstance(key, str) and key.startswith("privileged/block_") and key.endswith("_pos"):
                    try:
                        idx = int(key.split("_")[1])
                        num_from_keys = max(num_from_keys, idx + 1)
                    except Exception:
                        pass
        self._num_cubes = max(1, num_from_keys)
        return self._num_cubes

    def reset(self, done_mask: Optional[torch.Tensor] = None) -> None:
        if done_mask is None:
            self._prev_cubes_solved = None
            self._prev_max_error = None
            self._prev_eff_block_dist = None
            self._prev_block_target_dist = None
            self._prev_eff_pos = None
            self._prev_target_cube_pos = None
            self._prev_target_block = None
            self._phase = None
            self._dense_phase_cumulative = None
            self._success_awarded.zero_()
            return
        mask = self._as_batch(done_mask, dtype=torch.bool)
        self._success_awarded[mask] = False
        if self._prev_cubes_solved is not None:
            self._prev_cubes_solved[mask] = torch.nan
        if self._prev_max_error is not None:
            self._prev_max_error[mask] = torch.nan
        if self._prev_eff_block_dist is not None:
            self._prev_eff_block_dist[mask] = torch.nan
        if self._prev_block_target_dist is not None:
            self._prev_block_target_dist[mask] = torch.nan
        if self._prev_eff_pos is not None:
            self._prev_eff_pos[mask] = torch.nan
        if self._prev_target_cube_pos is not None:
            self._prev_target_cube_pos[mask] = torch.nan
        if self._prev_target_block is not None:
            self._prev_target_block[mask] = -1
        if self._phase is not None:
            self._phase[mask] = False
        if self._dense_phase_cumulative is not None:
            self._dense_phase_cumulative[mask] = 0.0

    def compute(
        self,
        *,
        base_rewards: torch.Tensor,
        infos,
        dones: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base = self._as_batch(base_rewards, dtype=torch.float32)
        zeros = torch.zeros_like(base)
        obs_full = self._extract_obs_tensor_from_infos(infos)
        obs_core: Optional[torch.Tensor] = None
        goal_core: Optional[torch.Tensor] = None
        if obs_full is not None:
            obs_core, goal_core = self._split_obs_goal(obs_full)
            if self._num_cubes is None:
                inferred = self._infer_num_cubes_from_obs_dim(int(obs_core.shape[1]))
                if inferred is not None:
                    self._num_cubes = int(inferred)
        num_cubes = self._infer_num_cubes(infos)
        cubes_solved = self._extract_scalar_batch(infos, "diag/cubes_solved", default=0.0)
        if not (isinstance(infos, dict) and "diag/cubes_solved" in infos):
            cubes_solved = self._extract_log_scalar_batch(infos, "/Teacher/diag_cubes_solved", default=0.0)
        cubes_total = self._extract_scalar_batch(infos, "diag/cubes_total", default=float(num_cubes))
        if not (isinstance(infos, dict) and "diag/cubes_total" in infos):
            cubes_total = torch.full((self.num_envs,), float(max(1, num_cubes)), dtype=torch.float32, device=self.device)
        cube_max_error = self._extract_scalar_batch(infos, "diag/cube_max_target_error", default=0.0).clamp_min(0.0)
        if not (isinstance(infos, dict) and "diag/cube_max_target_error" in infos):
            cube_max_error = self._extract_log_scalar_batch(
                infos, "/Teacher/diag_cube_max_target_error", default=0.0
            ).clamp_min(0.0)
        target_block = self._extract_scalar_batch(infos, "privileged/target_block", default=0.0).long()
        if not (isinstance(infos, dict) and "privileged/target_block" in infos):
            target_block = self._extract_log_scalar_batch(infos, "/Teacher/diag_target_block", default=0.0).long()
        target_block = torch.clamp(target_block, min=0, max=max(0, num_cubes - 1))
        if self._prev_target_block is None:
            self._prev_target_block = target_block.clone()
        target_changed = self._prev_target_block != target_block
        target_error = cube_max_error.clone()
        if isinstance(infos, dict) and "diag/cube_target_errors" in infos:
            try:
                errs = torch.as_tensor(infos["diag/cube_target_errors"], device=self.device, dtype=torch.float32)
                if errs.ndim == 1:
                    if self.num_envs == 1 and errs.numel() > 0:
                        idx = int(torch.clamp(target_block[0], min=0, max=max(0, errs.numel() - 1)).item())
                        target_error = errs[idx].view(1).repeat(self.num_envs)
                elif errs.ndim >= 2 and errs.shape[0] == self.num_envs:
                    flat = errs.reshape(self.num_envs, -1)
                    if flat.shape[1] > 0:
                        idx = torch.clamp(target_block, min=0, max=flat.shape[1] - 1)
                        target_error = flat.gather(1, idx.view(-1, 1)).squeeze(1)
            except Exception:
                pass
        target_error = torch.where(torch.isfinite(target_error), target_error.clamp_min(0.0), cube_max_error)
        layout = None
        if obs_core is not None:
            layout = self._infer_obs_layout(int(obs_core.shape[1]), num_cubes)
            if layout is not None and goal_core is not None and goal_core.shape[1] == obs_core.shape[1]:
                try:
                    block_pos = layout["block_pos"]
                    block_obs = torch.stack([obs_core[:, s:e] for (s, e) in block_pos], dim=1)
                    block_goal = torch.stack([goal_core[:, s:e] for (s, e) in block_pos], dim=1)
                    block_errs = torch.linalg.norm(block_obs - block_goal, dim=2)
                    idx = torch.clamp(target_block, min=0, max=block_errs.shape[1] - 1).view(-1, 1)
                    target_error = block_errs.gather(1, idx).squeeze(1).clamp_min(0.0)
                    cube_max_error = torch.max(block_errs, dim=1).values.clamp_min(0.0)
                except Exception:
                    pass

        eff_pos = self._extract_vec3_batch(infos, "proprio/effector_pos")
        if eff_pos is None and obs_core is not None and layout is not None:
            s, e = layout["eff_pos"]
            eff_pos = obs_core[:, s:e]
        if eff_pos is None:
            eff_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        target_cube_pos = None
        if isinstance(infos, dict):
            block_positions = []
            for i in range(num_cubes):
                block_i = self._extract_vec3_batch(infos, f"privileged/block_{i}_pos")
                if block_i is None:
                    block_i = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
                block_positions.append(block_i)
            if block_positions:
                block_stack = torch.stack(block_positions, dim=1)
                batch_idx = torch.arange(self.num_envs, device=self.device)
                target_cube_pos = block_stack[batch_idx, target_block]
        if (
            (target_cube_pos is None or not torch.isfinite(target_cube_pos).all())
            and obs_core is not None
            and layout is not None
        ):
            try:
                block_pos = layout["block_pos"]
                block_obs = torch.stack([obs_core[:, s:e] for (s, e) in block_pos], dim=1)
                batch_idx = torch.arange(self.num_envs, device=self.device)
                target_cube_pos = block_obs[batch_idx, target_block]
            except Exception:
                pass
        if target_cube_pos is None:
            target_cube_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        target_effector_dist = torch.linalg.norm(eff_pos - target_cube_pos, dim=1)
        target_tcp_xy_dist = torch.linalg.norm((eff_pos - target_cube_pos)[:, :2], dim=1)
        target_tcp_z_gap = torch.abs((eff_pos - target_cube_pos)[:, 2])
        target_cube_z = target_cube_pos[:, 2]
        gripper_contact = self._extract_scalar_batch(infos, "proprio/gripper_contact", default=0.0)
        gripper_opening = self._extract_scalar_batch(infos, "proprio/gripper_opening", default=1.0)
        if obs_core is not None and layout is not None:
            contact_idx = int(layout["gripper_contact_idx"])
            opening_idx = int(layout["gripper_open_idx"])
            if not (isinstance(infos, dict) and "proprio/gripper_contact" in infos):
                gripper_contact = obs_core[:, contact_idx].clamp(0.0, 1.0)
            if not (isinstance(infos, dict) and "proprio/gripper_opening" in infos):
                gripper_opening = (obs_core[:, opening_idx] / 3.0).clamp(0.0, 1.0)
        gripper_closure = torch.clamp(1.0 - gripper_opening, min=0.0, max=1.0)
        if self._prev_target_cube_pos is None:
            target_cube_speed = torch.zeros_like(target_effector_dist)
        else:
            target_cube_speed = torch.linalg.norm(target_cube_pos - self._prev_target_cube_pos, dim=1)
        self._prev_target_cube_pos = target_cube_pos.clone()
        target_grasp_detected = (
            (target_effector_dist <= self.grasp_error_threshold)
            & ((gripper_contact >= self._grasp_contact_threshold) | (gripper_opening < 0.45))
        ).float()
        # Stricter grasp proxy for debugging: near target + closed/contact + target cube moving.
        target_grasp_detected_strict = (
            (target_effector_dist <= self.grasp_error_threshold)
            & ((gripper_contact >= self._grasp_contact_threshold) | (gripper_closure >= 0.55))
            & (target_cube_speed >= 5e-4)
        ).float()

        if self._phase is None:
            self._phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        holding_prev = self._phase.clone()
        grasp_candidate = (
            (target_effector_dist <= self.grasp_error_threshold)
            & (gripper_contact >= self._grasp_contact_threshold)
        )
        grasp_event = (~holding_prev) & grasp_candidate
        release_candidate = (
            (gripper_contact < self._release_contact_threshold)
            & (gripper_closure < 0.55)
            & (target_effector_dist > (self.grasp_error_threshold * 1.25))
        )
        phase_is_carry = (holding_prev | grasp_candidate) & (~release_candidate) & (~target_changed)

        if self._prev_cubes_solved is None:
            self._prev_cubes_solved = cubes_solved.clone()
        prev_cubes_solved = self._prev_cubes_solved.clone()
        prev_cubes_solved_valid = torch.isfinite(prev_cubes_solved)
        solved_delta_raw = cubes_solved - prev_cubes_solved
        solved_delta = torch.where(
            prev_cubes_solved_valid,
            torch.clamp(solved_delta_raw, min=0.0),
            torch.zeros_like(cubes_solved),
        )
        drop_event = prev_cubes_solved_valid & (cubes_solved < (prev_cubes_solved - 1e-6))
        drop_penalty = self.drop_penalty * drop_event.float()

        if self._prev_eff_block_dist is None:
            reach_progress = torch.zeros_like(target_effector_dist)
        else:
            reach_delta = self._prev_eff_block_dist - target_effector_dist
            reach_progress = torch.where(torch.isfinite(reach_delta), reach_delta, torch.zeros_like(reach_delta))

        if self._prev_max_error is None:
            carry_progress = torch.zeros_like(target_error)
        else:
            carry_delta = self._prev_max_error - target_error
            carry_progress = torch.where(torch.isfinite(carry_delta), carry_delta, torch.zeros_like(carry_delta))

        reach_progress = torch.where(target_changed, torch.zeros_like(reach_progress), reach_progress)
        carry_progress = torch.where(target_changed, torch.zeros_like(carry_progress), carry_progress)
        phase_progress = torch.where(phase_is_carry, carry_progress, reach_progress)
        if self.progress_clip > 0.0:
            phase_progress = torch.clamp(phase_progress, min=-self.progress_clip, max=self.progress_clip)
        dense_phase_reward = (
            (self.progress_scale * phase_progress)
            + (self.grasp_reward * grasp_event.float())
            + (self.place_reward * solved_delta)
            + drop_penalty
        )
        if self._dense_phase_cumulative is None:
            self._dense_phase_cumulative = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._dense_phase_cumulative = self._dense_phase_cumulative + dense_phase_reward

        success = self._extract_success_batch(infos, cubes_solved=cubes_solved, cubes_total=cubes_total)
        success_event = (success > 0.5) & (~self._success_awarded)
        success_sparse = self.success_reward * success_event.float()
        self._success_awarded = self._success_awarded | success_event

        if self.mode == "sparse_final":
            recomposed = success_sparse
        elif self.mode == "sparse_intermediate":
            recomposed = solved_delta + drop_penalty
        else:
            recomposed = dense_phase_reward
        self._prev_cubes_solved = cubes_solved.clone()
        self._prev_eff_block_dist = target_effector_dist.clone()
        self._prev_max_error = target_error.clone()
        self._prev_target_block = target_block.clone()
        self._phase = phase_is_carry & (solved_delta <= 0.0)

        components = {
            "base_env_reward": base,
            "success_sparse": success_sparse,
            "solved_delta_sparse": solved_delta,
            "drop_event": drop_event.float(),
            "dense_target_error": target_error,
            "dense_target_distance_reward": -self.progress_scale * target_error,
            "dense_phase_reach_progress": reach_progress,
            "dense_phase_carry_progress": carry_progress,
            "dense_phase_progress": phase_progress,
            "dense_phase_grasp_event": grasp_event.float(),
            "dense_phase_place_event": solved_delta,
            "dense_phase_is_carry": phase_is_carry.float(),
            "dense_phase_reward": dense_phase_reward,
            "dense_phase_cumulative": self._dense_phase_cumulative.clone(),
            "target_block": target_block.float(),
            "target_changed": target_changed.float(),
            "cubes_solved": cubes_solved,
            "cubes_total": cubes_total,
            "target_effector_dist": target_effector_dist,
            "target_tcp_xy_dist": target_tcp_xy_dist,
            "target_tcp_z_gap": target_tcp_z_gap,
            "target_cube_z": target_cube_z,
            "target_cube_speed": target_cube_speed,
            "target_grasp_detected": target_grasp_detected,
            "target_grasp_detected_strict": target_grasp_detected_strict,
            "gripper_contact_raw": gripper_contact,
            "gripper_opening_raw": gripper_opening,
            "gripper_closure": gripper_closure,
            "drop_penalty": drop_penalty,
            # Legacy component keys retained for easier downstream logging compatibility.
            "reach_sparse": zeros,
            "grasp_sparse": zeros,
            "place_sparse": zeros,
            "release_sparse": zeros,
            "progress_dense": zeros,
            "dense_gripper": zeros,
            "dense_release": zeros,
            "dense_movement_penalty": zeros,
            "mode_total_pre_intervention": recomposed,
        }
        if dones is not None:
            self.reset(dones)
        return recomposed, components


def _yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class CubeTeacherInfoAdapter(gym.Wrapper):
    """Reconstruct cube target-block info in task-mode environments."""

    def __init__(self, env: gym.Env, *, target_mode: str = "sequential", success_tolerance: float = 0.04):
        super().__init__(env)
        if target_mode not in {"fixed", "sequential"}:
            raise ValueError(f"Unknown target_mode={target_mode}")
        self.target_mode = target_mode
        self.success_tolerance = float(success_tolerance)

    def _cube_target_errors(self, out: dict, unwrapped) -> np.ndarray:
        num_cubes = int(getattr(unwrapped, "_num_cubes", 0))
        if num_cubes <= 0:
            return np.zeros(0, dtype=np.float32)
        errs: list[float] = []
        for i in range(num_cubes):
            try:
                obj = np.asarray(out[f"privileged/block_{i}_pos"], dtype=np.float32)
            except Exception:
                try:
                    obj = np.asarray(unwrapped._data.joint(f"object_joint_{i}").qpos[:3], dtype=np.float32)
                except Exception:
                    obj = np.zeros(3, dtype=np.float32)
            try:
                mocap_id = int(unwrapped._cube_target_mocap_ids[i])
                tar = np.asarray(unwrapped._data.mocap_pos[mocap_id], dtype=np.float32)
            except Exception:
                tar = obj
            errs.append(float(np.linalg.norm(obj - tar)))
        return np.asarray(errs, dtype=np.float32)

    def _select_target_block(self, unwrapped, errs: np.ndarray) -> int:
        base_target = int(getattr(unwrapped, "_target_block", 0))
        if self.target_mode != "sequential" or errs.size == 0:
            return base_target
        unresolved = np.where(errs > self.success_tolerance)[0]
        if unresolved.size == 0:
            return base_target
        return int(unresolved[0])

    def _augment_info(self, info):
        if not isinstance(info, dict):
            return info
        out = dict(info)
        unwrapped = self.unwrapped
        errs = self._cube_target_errors(out, unwrapped)
        out["diag/cube_success_tolerance"] = float(self.success_tolerance)
        out["diag/cube_target_errors"] = errs
        cubes_total = int(errs.size) if errs.size else 0
        cubes_solved = int(np.sum(errs <= self.success_tolerance)) if errs.size else 0
        out["diag/cubes_total"] = cubes_total
        out["diag/cubes_solved"] = cubes_solved
        out["diag/cubes_solved_fraction"] = float(cubes_solved / cubes_total) if cubes_total > 0 else 0.0
        out["diag/cube_max_target_error"] = float(np.max(errs)) if errs.size else 0.0

        target_idx = self._select_target_block(unwrapped, errs)
        out["privileged/target_block"] = int(target_idx)
        out["diag/target_block_dynamic"] = int(target_idx)

        try:
            target_mocap_ids = getattr(unwrapped, "_cube_target_mocap_ids", None)
            if target_mocap_ids is not None:
                mocap_id = int(target_mocap_ids[target_idx])
                mocap_pos = np.asarray(unwrapped._data.mocap_pos, dtype=np.float32)
                if 0 <= mocap_id < mocap_pos.shape[0]:
                    out["privileged/target_block_pos"] = mocap_pos[mocap_id].copy()
        except Exception:
            pass

        try:
            target_mocap_ids = getattr(unwrapped, "_cube_target_mocap_ids", None)
            if target_mocap_ids is not None:
                mocap_id = int(target_mocap_ids[target_idx])
                mocap_quat = np.asarray(unwrapped._data.mocap_quat, dtype=np.float32)
                if 0 <= mocap_id < mocap_quat.shape[0]:
                    yaw = _yaw_from_quat_wxyz(mocap_quat[mocap_id])
                    out["privileged/target_block_yaw"] = np.asarray([yaw], dtype=np.float32)
        except Exception:
            pass
        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._augment_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._augment_info(info)


def build_ogbench_manip_wrapper(
    *,
    env_name: Optional[str] = None,
    obs_mode: str,
    include_goal: bool = True,
    include_distance: bool = False,
    include_direction: bool = False,
    include_velocity: bool = False,
    include_relative_cube_features: bool = False,
    relative_only_obs: bool = False,
    reward_type: str,
    dense_reward_scale: float,
    step_penalty: float,
    disable_rotation: bool = False,
    reward_switch_after_steps: int = 0,
    cube_reward_mode: str = "dense",
    intervention_mode: str = "none",
    teacher_type: str = "bfs",
    teacher_action_noise_std: float = 0.0,
    tolerance_type: str = "angle",
    tolerance_value: float = 30.0,
    tolerance_channel_weights: Optional[str] = None,
    tolerance_xyz_value: float = -1.0,
    tolerance_yaw_value: float = -1.0,
    tolerance_gripper_value: float = -1.0,
    tolerance_adaptive_enable: bool = True,
    tolerance_adaptive_near_distance: float = 0.08,
    tolerance_adaptive_far_distance: float = 0.30,
    tolerance_adaptive_near_scale: float = 0.35,
    binary_gripper_actions: bool = False,
    binary_gripper_threshold: float = 0.0,
    hard_gripper_intervention: bool = False,
    gripper_intervene_pick_radius: float = 0.06,
    gripper_intervene_place_radius: float = 0.06,
    gripper_intervene_contact_threshold: float = 0.3,
    hard_block_lethal: bool = True,
    intervention_enable_after_steps: int = 0,
    intervention_agent_mode: str = "divergence",
    intervention_safety_margin_frac: float = 0.0,
    intervention_release_steps: int = 3,
    intervention_reward_patience_steps: int = 5,
    intervention_reward_improvement_epsilon: float = 1e-6,
    intervention_episode_prob: float = 1.0,
    intervention_episode_prob_min: float = 0.0,
    intervention_episode_prob_decay_steps: int = 0,
    intervention_episode_prob_decay_start: int = 0,
    intervention_episode_prob_seed: Optional[int] = None,
    human_intervention_threshold: float = 0.1,
    human_intervention_hold_time: float = 0.5,
    teacher_target_mode: str = "sequential",
    cube_success_tolerance: float = 0.04,
    static_reset_seed: Optional[int] = None,
    teleop_interface: Optional[object] = None,
) -> Callable:
    """Build a manipulation-only wrapper stack."""

    def _apply(env: gym.Env):
        wrapper_chain: list[str] = []
        cube_mode = canonicalize_cube_reward_mode(cube_reward_mode)
        effective_include_goal = bool(include_goal)
        if relative_only_obs and effective_include_goal:
            warnings.warn(
                "relative_only_obs requested; forcing include_goal=False to avoid absolute goal-state leakage."
            )
            effective_include_goal = False
        if static_reset_seed is not None:
            env = FixedResetSeedWrapper(env, reset_seed=int(static_reset_seed))
            wrapper_chain.append(f"FixedResetSeed({int(static_reset_seed)})")

        if obs_mode == "state":
            if include_distance or include_direction or include_velocity:
                warnings.warn(
                    "Manip state mode ignores include_distance/include_direction/include_velocity and "
                    "uses native environment state observations."
                )
            wrapper_chain.append("NativeManipState")
            if effective_include_goal:
                env = ManipGoalConditionedObsWrapper(env)
                wrapper_chain.append("ManipGoalConditionedObsWrapper")

        env_id = env_name or str(getattr(getattr(env, "spec", None), "id", "") or "")
        wrapper_reward_type = str(reward_type)
        if cube_reward_mode_active(env_name=env_id, obs_mode=obs_mode, reward_mode=cube_mode):
            if wrapper_reward_type != "sparse":
                warnings.warn(
                    "Cube reward tracker is active; forcing ManipDetailedRewardWrapper reward_type='sparse' "
                    "to avoid multiple dense reward implementations."
                )
            wrapper_reward_type = "sparse"

        env = ManipDetailedRewardWrapper(
            env,
            reward_type=wrapper_reward_type,
            dense_reward_scale=dense_reward_scale,
            step_penalty=step_penalty,
            switch_reward_to_sparse_after_steps_per_env=reward_switch_after_steps,
        )
        wrapper_chain.append("ManipDetailedRewardWrapper")
        env_id_l = env_id.lower()
        if "cube" in env_id_l:
            env = CubeTeacherInfoAdapter(
                env,
                target_mode=teacher_target_mode,
                success_tolerance=cube_success_tolerance,
            )
            wrapper_chain.append("CubeTeacherInfoAdapter")
        if include_relative_cube_features and obs_mode == "state":
            env = ManipRelativeCubeFeaturesWrapper(env)
            wrapper_chain.append("ManipRelativeCubeFeaturesWrapper")
        if relative_only_obs and obs_mode == "state":
            env = ManipRelativeOnlyObsWrapper(
                env,
                include_relative_cube_features=include_relative_cube_features,
                relative_feature_dim=ManipRelativeCubeFeaturesWrapper.FEATURE_DIM,
            )
            wrapper_chain.append("ManipRelativeOnlyObsWrapper")

        env, intervention_name = maybe_wrap_intervention(
            env,
            intervention_mode=intervention_mode,
            teacher_type=teacher_type,
            teacher_action_noise_std=teacher_action_noise_std,
            tolerance_type=tolerance_type,
            tolerance_value=tolerance_value,
            tolerance_channel_weights=tolerance_channel_weights,
            tolerance_xyz_value=tolerance_xyz_value,
            tolerance_yaw_value=tolerance_yaw_value,
            tolerance_gripper_value=tolerance_gripper_value,
            tolerance_adaptive_enable=tolerance_adaptive_enable,
            tolerance_adaptive_near_distance=tolerance_adaptive_near_distance,
            tolerance_adaptive_far_distance=tolerance_adaptive_far_distance,
            tolerance_adaptive_near_scale=tolerance_adaptive_near_scale,
            binary_gripper_actions=binary_gripper_actions,
            binary_gripper_threshold=binary_gripper_threshold,
            hard_gripper_intervention=hard_gripper_intervention,
            gripper_intervene_pick_radius=gripper_intervene_pick_radius,
            gripper_intervene_place_radius=gripper_intervene_place_radius,
            gripper_intervene_contact_threshold=gripper_intervene_contact_threshold,
            hard_block_lethal=hard_block_lethal,
            intervention_enable_after_steps=intervention_enable_after_steps,
            intervention_agent_mode=intervention_agent_mode,
            intervention_safety_margin_frac=intervention_safety_margin_frac,
            intervention_release_steps=intervention_release_steps,
            intervention_reward_patience_steps=intervention_reward_patience_steps,
            intervention_reward_improvement_epsilon=intervention_reward_improvement_epsilon,
            intervention_episode_prob=intervention_episode_prob,
            intervention_episode_prob_min=intervention_episode_prob_min,
            intervention_episode_prob_decay_steps=intervention_episode_prob_decay_steps,
            intervention_episode_prob_decay_start=intervention_episode_prob_decay_start,
            intervention_episode_prob_seed=intervention_episode_prob_seed,
            human_threshold=human_intervention_threshold,
            human_hold_time=human_intervention_hold_time,
            teleop_interface=teleop_interface,
        )
        if intervention_name is not None:
            wrapper_chain.append(intervention_name)
        if disable_rotation:
            env = ManipDisableRotationActionWrapper(env)
            wrapper_chain.append("ManipDisableRotationActionWrapper")

        stack_key = (
            env_id,
            obs_mode,
            intervention_mode,
            teacher_type,
            wrapper_reward_type,
            cube_mode,
            bool(disable_rotation),
            tuple(wrapper_chain),
        )
        if stack_key not in _WRAPPER_STACK_PRINTED:
            _WRAPPER_STACK_PRINTED.add(stack_key)
            print(f"[WrapperStack] env={env_id or '<unknown>'} chain=" + " -> ".join(wrapper_chain))
        return env

    return _apply
