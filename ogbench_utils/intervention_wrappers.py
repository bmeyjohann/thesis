import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym


@dataclass
class _InterventionStats:
    episode_steps: int = 0
    intervention_steps: int = 0
    num_interventions: int = 0
    num_safety: int = 0
    num_divergence: int = 0
    num_progress: int = 0
    num_gripper: int = 0
    num_manual: int = 0
    num_component_movement_xyz: int = 0
    num_component_yaw: int = 0
    num_component_gripper: int = 0
    num_component_mixed: int = 0
    episode_gate_enabled: int = 1
    episode_gate_prob: float = 1.0
    _in_burst: bool = False
    _current_burst_len: int = 0
    _burst_lens_sum: int = 0
    _burst_count: int = 0

    def start_burst(self):
        if not self._in_burst:
            self._in_burst = True
            self._current_burst_len = 0
            self._burst_count += 1

    def step_burst(self):
        if self._in_burst:
            self._current_burst_len += 1

    def end_burst_if_needed(self):
        if self._in_burst:
            self._in_burst = False
            self._burst_lens_sum += self._current_burst_len
            self._current_burst_len = 0

    def flush_metrics(self) -> dict:
        # Ensure an open burst is closed at episode end
        self.end_burst_if_needed()
        frac = (self.intervention_steps / max(1, self.episode_steps)) if self.episode_steps > 0 else 0.0
        avg_burst = (self._burst_lens_sum / max(1, self._burst_count)) if self._burst_count > 0 else 0.0
        return dict(
            teacher_num_interventions=int(self.num_interventions),
            teacher_intervention_steps=int(self.intervention_steps),
            teacher_fraction_steps=float(frac),
            teacher_avg_burst_len=float(avg_burst),
            teacher_num_safety_interventions=int(self.num_safety),
            teacher_num_divergence_interventions=int(self.num_divergence),
            teacher_num_progress_interventions=int(self.num_progress),
            teacher_num_gripper_interventions=int(self.num_gripper),
            teacher_num_manual_interventions=int(self.num_manual),
            teacher_num_component_movement_xyz_interventions=int(self.num_component_movement_xyz),
            teacher_num_component_yaw_interventions=int(self.num_component_yaw),
            teacher_num_component_gripper_interventions=int(self.num_component_gripper),
            teacher_num_component_mixed_interventions=int(self.num_component_mixed),
            teacher_episode_steps=int(self.episode_steps),
            teacher_episode_gate_enabled=int(self.episode_gate_enabled),
            teacher_episode_gate_prob=float(self.episode_gate_prob),
        )


class InterventionWrapper(gym.Wrapper):
    """
    Generic intervention wrapper supporting human teleop and agent (teacher) modes.

    Modes:
    - mode='human': override with teleop input above threshold for a hold duration
    - mode='agent': override based on teacher policy (e.g., BFS) using safety and tolerance rules
    """
    _manual_gate_ui_initialized = False
    _manual_gate_ui_available = False
    _manual_gate_owns_window = False
    _manual_gate_screen = None
    _manual_gate_font = None
    _manual_gate_last_state: Optional[bool] = None
    _manual_gate_warned = False
    _manual_gate_last_fps: float = 20.0
    _manual_gate_last_q_min: float = float("nan")
    _manual_gate_last_q_dis: float = float("nan")
    _manual_gate_last_rewards = {
        "sparse": 0.0,
        "dense": 0.0,
        "total": 0.0,
    }
    _manual_gate_reward_scales = {
        "sparse": 1.0,
        "dense": 0.1,
        "total": 0.5,
    }

    def __init__(
        self,
        env: gym.Env,
        teleop_interface=None,
        *,
        mode: str = 'human',  # 'human' or 'agent'
        teacher_type: str = 'bfs',
        # Human params
        threshold: float = 0.1,
        hold_time: float = 0.5,
        # Agent/teacher params
        tolerance_type: str = 'angle',  # 'angle', 'l2', or 'component'
        tolerance_value: float = 30.0,  # degrees for angle, absolute for l2
        tolerance_channel_weights: Optional[object] = None,  # optional per-action weights for l2 metric
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
        hard_block_lethal: bool = True,  # intervene if student's step enters lethal/danger cell
        enable_after_steps: int = 0,     # warmup steps per env before enabling interventions
        agent_mode: str = 'divergence',  # 'always', 'divergence', 'safety_align', 'safety_progress', 'reward_progress', 'manual_gripper'
        safety_margin_frac: float = 0.0,  # fraction of maze cell size for safety margin
        release_steps: int = 3,  # consecutive steps to release intervention
        reward_patience_steps: int = 5,  # reward-progress mode trigger/release horizon
        reward_improvement_epsilon: float = 1e-6,  # minimum signal delta considered improvement
        episode_intervention_prob: float = 1.0,
        episode_intervention_prob_min: float = 0.0,
        episode_intervention_prob_decay_steps: int = 0,
        episode_intervention_prob_decay_start: int = 0,
        episode_intervention_seed: Optional[int] = None,
    ):
        super().__init__(env)

        assert mode in ('human', 'agent')
        assert tolerance_type in ('angle', 'l2', 'component')
        assert agent_mode in ('always', 'divergence', 'safety_align', 'safety_progress', 'reward_progress', 'manual_gripper')
        self.mode = mode
        self.teacher_type = teacher_type
        self.teleop = teleop_interface
        self.threshold = float(threshold)
        self.hold_time = float(hold_time)
        self.tolerance_type = tolerance_type
        self.tolerance_value = float(tolerance_value)
        self.tolerance_xyz_value = float(tolerance_xyz_value)
        self.tolerance_yaw_value = float(tolerance_yaw_value)
        self.tolerance_gripper_value = float(tolerance_gripper_value)
        self.tolerance_adaptive_enable = bool(tolerance_adaptive_enable)
        self.tolerance_adaptive_near_distance = float(tolerance_adaptive_near_distance)
        self.tolerance_adaptive_far_distance = float(tolerance_adaptive_far_distance)
        self.tolerance_adaptive_near_scale = float(tolerance_adaptive_near_scale)
        self._action_dim = int(np.prod(getattr(getattr(self.env, "action_space", None), "shape", (0,))))
        self.tolerance_channel_weights = self._parse_tolerance_channel_weights(
            tolerance_channel_weights,
            self._action_dim,
        )
        self.binary_gripper_actions = bool(binary_gripper_actions)
        self.binary_gripper_threshold = float(binary_gripper_threshold)
        self.hard_gripper_intervention = bool(hard_gripper_intervention)
        self.gripper_intervene_pick_radius = float(gripper_intervene_pick_radius)
        self.gripper_intervene_place_radius = float(gripper_intervene_place_radius)
        self.gripper_intervene_contact_threshold = float(gripper_intervene_contact_threshold)
        self.hard_block_lethal = bool(hard_block_lethal)
        self.enable_after_steps = int(enable_after_steps)
        self.agent_mode = agent_mode
        self.safety_margin_frac = float(safety_margin_frac)
        self.release_steps = int(release_steps)
        self.reward_patience_steps = int(max(1, reward_patience_steps))
        self.reward_improvement_epsilon = float(reward_improvement_epsilon)
        self.episode_intervention_prob = float(episode_intervention_prob)
        self.episode_intervention_prob_min = float(episode_intervention_prob_min)
        self.episode_intervention_prob_decay_steps = int(episode_intervention_prob_decay_steps)
        self.episode_intervention_prob_decay_start = int(episode_intervention_prob_decay_start)
        self._rng = np.random.default_rng(episode_intervention_seed)
        self._manual_gate_enabled = bool(self.mode == "agent" and self.agent_mode == "manual_gripper")
        self._manual_gate_target_fps = 20.0
        self._manual_gate_min_fps = 1.0
        self._manual_gate_max_fps = 120.0


        # Internal timers (human)
        self._last_override_ts = 0.0

        # Episode stats
        self._stats = _InterventionStats()

        # Teacher cache (for future types); BFS uses env oracle per-step
        # Avoid flooding stdout when vectorized envs create many wrapper instances.
        cls = type(self)
        if not getattr(cls, "_init_logged_once", False):
            print(
                "[InterventionWrapper] Initialized."
                f" mode={self.mode}, teacher={self.teacher_type},"
                f" agent_mode={self.agent_mode}"
            )
            cls._init_logged_once = True
        # Global per-env step counter across episodes
        self._global_step_env = 0
        self._danger_centers: Optional[np.ndarray] = None
        self._maze_unit: Optional[float] = None
        self._align_count = 0
        self._progress_good_count = 0
        self._progress_violation = False
        self._last_distance: Optional[float] = None
        self._safety_align_active = False
        self._progress_active = False
        self._reward_progress_active = False
        self._reward_non_improve_count = 0
        self._reward_improve_count = 0
        self._reward_cooldown_remaining = 0
        self._last_reward_progress_signal: Optional[float] = None
        self._reward_delta_window = deque(maxlen=self.reward_patience_steps)
        self._reward_window_sum = 0.0
        self._reward_window_ready = False
        self._episode_interventions_enabled = True
        self._oracle = None
        self._oracle_type: Optional[str] = None
        self._oracle_target_block: Optional[int] = None
        self._last_obs = None
        self._last_info: Optional[dict] = None

    def _parse_tolerance_channel_weights(self, raw: Optional[object], action_dim: int) -> Optional[np.ndarray]:
        if raw is None:
            return None
        if isinstance(raw, str):
            raw = raw.strip()
            if raw == "":
                return None
            parts = [p for p in re.split(r"[,\s;]+", raw) if p]
            values = np.asarray([float(p) for p in parts], dtype=np.float32)
        elif np.isscalar(raw):
            values = np.asarray([float(raw)], dtype=np.float32)
        else:
            values = np.asarray(raw, dtype=np.float32).reshape(-1)

        if np.any(~np.isfinite(values)):
            raise ValueError("tolerance_channel_weights must be finite")
        if np.any(values <= 0.0):
            raise ValueError("tolerance_channel_weights must be > 0")

        if action_dim <= 0:
            return values
        if values.size == 1:
            return np.full((action_dim,), float(values[0]), dtype=np.float32)
        if values.size != action_dim:
            raise ValueError(
                f"tolerance_channel_weights has {values.size} entries but action_dim is {action_dim}. "
                "Provide one value or exactly one per action channel."
            )
        return values.astype(np.float32)

    def _l2_delta(self, a: np.ndarray, b: np.ndarray, *, weighted: bool) -> float:
        delta = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        if weighted and self.tolerance_channel_weights is not None and delta.shape[-1] == self.tolerance_channel_weights.shape[0]:
            delta = delta * self.tolerance_channel_weights
        return float(np.linalg.norm(delta))

    @staticmethod
    def _project_5d_to_4d(action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 5:
            return arr
        return np.concatenate([arr[:3], arr[4:5]], axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _expand_4d_to_5d(action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 4:
            return arr
        out = np.zeros((5,), dtype=np.float32)
        out[:3] = arr[:3]
        out[4] = arr[3]
        return out

    def _coerce_action_dim(self, action: Optional[np.ndarray], target_dim: int) -> Optional[np.ndarray]:
        if action is None:
            return None
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if target_dim <= 0 or arr.size == target_dim:
            return arr
        if arr.size == 5 and target_dim == 4:
            return self._project_5d_to_4d(arr)
        if arr.size == 4 and target_dim == 5:
            return self._expand_4d_to_5d(arr)
        return arr

    @staticmethod
    def _gripper_index_for_size(size: int) -> Optional[int]:
        return (size - 1) if size >= 4 else None

    @staticmethod
    def _yaw_index_for_size(size: int) -> Optional[int]:
        return 3 if size >= 5 else None

    def _apply_gripper_binary(self, action: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if action is None:
            return None
        out = np.asarray(action, dtype=np.float32).copy()
        if not self.binary_gripper_actions or out.shape[-1] < 4:
            return out
        gripper_idx = out.shape[-1] - 1
        out[..., gripper_idx] = 1.0 if out[..., gripper_idx] >= self.binary_gripper_threshold else -1.0
        return out

    def _gripper_sign(self, action: Optional[np.ndarray]) -> int:
        if action is None:
            return 0
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < 4:
            return 0
        return 1 if float(arr[-1]) >= self.binary_gripper_threshold else -1

    def _force_gripper_to_teacher(
        self,
        policy_action: Optional[np.ndarray],
        teacher_action: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], dict]:
        diag = {
            "teacher_gripper_sync_applied": 0.0,
            "teacher_gripper_sync_policy_sign": 0.0,
            "teacher_gripper_sync_teacher_sign": 0.0,
        }
        if policy_action is None or teacher_action is None:
            return None, diag
        p = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        t = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
        if p.size < 4 or t.size != p.size:
            return None, diag
        policy_sign = self._gripper_sign(p)
        teacher_sign = self._gripper_sign(t)
        diag["teacher_gripper_sync_policy_sign"] = float(policy_sign)
        diag["teacher_gripper_sync_teacher_sign"] = float(teacher_sign)
        if teacher_sign == 0 or policy_sign == teacher_sign:
            return None, diag
        forced = p.copy()
        gripper_idx = self._gripper_index_for_size(forced.size)
        if gripper_idx is None:
            return None, diag
        forced[gripper_idx] = 1.0 if teacher_sign > 0 else -1.0
        forced = self._apply_gripper_binary(forced)
        diag["teacher_gripper_sync_applied"] = 1.0
        return forced, diag

    def _force_gripper_channel_to_teacher(
        self,
        policy_action: Optional[np.ndarray],
        teacher_action: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], dict]:
        diag = {
            "teacher_gripper_sync_applied": 0.0,
            "teacher_gripper_sync_policy_sign": 0.0,
            "teacher_gripper_sync_teacher_sign": 0.0,
        }
        if policy_action is None or teacher_action is None:
            return None, diag
        p = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        t = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
        if p.size < 4 or t.size != p.size:
            return None, diag
        policy_sign = self._gripper_sign(p)
        teacher_sign = self._gripper_sign(t)
        diag["teacher_gripper_sync_policy_sign"] = float(policy_sign)
        diag["teacher_gripper_sync_teacher_sign"] = float(teacher_sign)
        forced = p.copy()
        gripper_idx = self._gripper_index_for_size(forced.size)
        if gripper_idx is None:
            return None, diag
        forced[gripper_idx] = t[gripper_idx]
        forced = self._apply_gripper_binary(forced)
        if abs(float(forced[gripper_idx]) - float(p[gripper_idx])) <= 1e-6:
            return None, diag
        diag["teacher_gripper_sync_applied"] = 1.0
        return forced, diag

    def _extract_scalar(self, value, default: float = 0.0) -> float:
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return float(default)
            return float(arr[0])
        except Exception:
            return float(default)

    def _hard_gripper_violation(self, info: Optional[dict], policy_action: Optional[np.ndarray], teacher_action: Optional[np.ndarray]):
        if (
            not self.hard_gripper_intervention
            or self.teacher_type not in {"cube_plan", "cube_markov"}
            or info is None
            or policy_action is None
            or teacher_action is None
        ):
            return False, {}
        try:
            target_block = int(info.get("privileged/target_block", 0))
        except Exception:
            target_block = 0
        try:
            eff_pos = np.asarray(info.get("proprio/effector_pos"), dtype=np.float32).reshape(-1)
            block_pos = np.asarray(info.get(f"privileged/block_{target_block}_pos"), dtype=np.float32).reshape(-1)
            target_pos = np.asarray(info.get("privileged/target_block_pos"), dtype=np.float32).reshape(-1)
        except Exception:
            return False, {}
        if eff_pos.size < 3 or block_pos.size < 3 or target_pos.size < 3:
            return False, {}

        eff_block_dist = float(np.linalg.norm(eff_pos[:3] - block_pos[:3]))
        block_target_dist = float(np.linalg.norm(block_pos[:3] - target_pos[:3]))
        gripper_contact = self._extract_scalar(info.get("proprio/gripper_contact"), default=0.0)
        holding = gripper_contact >= self.gripper_intervene_contact_threshold
        near_pick = eff_block_dist <= self.gripper_intervene_pick_radius
        near_place = block_target_dist <= self.gripper_intervene_place_radius
        critical = bool(near_pick or near_place or holding)

        policy_sign = self._gripper_sign(policy_action)
        teacher_sign = self._gripper_sign(teacher_action)
        mismatch = policy_sign != teacher_sign and policy_sign != 0 and teacher_sign != 0
        violation = bool(critical and mismatch)
        diag = {
            "teacher_gripper_critical": float(1.0 if critical else 0.0),
            "teacher_gripper_mismatch": float(1.0 if mismatch else 0.0),
            "teacher_gripper_hard_violation": float(1.0 if violation else 0.0),
            "teacher_gripper_policy_sign": float(policy_sign),
            "teacher_gripper_teacher_sign": float(teacher_sign),
            "teacher_gripper_eff_block_dist": float(eff_block_dist),
            "teacher_gripper_block_target_dist": float(block_target_dist),
            "teacher_gripper_contact": float(gripper_contact),
        }
        return violation, diag

    def _force_gripper_closed_action(self, policy_action: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if policy_action is None:
            return None
        out = np.asarray(policy_action, dtype=np.float32).copy()
        out_flat = out.reshape(-1)
        gripper_idx = self._gripper_index_for_size(out_flat.size)
        if gripper_idx is None:
            return None
        # Keep agent movement/yaw, only force gripper close.
        out_flat[gripper_idx] = 1.0
        return self._apply_gripper_binary(out)

    def _gripper_lock_violation(self, info: Optional[dict], policy_action: Optional[np.ndarray]):
        """Detect unsafe release and return (violation, forced_action, diagnostics).

        Unsafe release means the student wants to open gripper while the target cube
        is effectively being carried and not yet place-ready. In this case we apply a
        local constraint: preserve student action except force gripper close.
        """
        if (
            not self.hard_gripper_intervention
            or self.teacher_type not in {"cube_plan", "cube_markov"}
            or info is None
            or policy_action is None
        ):
            return False, None, {}
        forced_action = self._force_gripper_closed_action(policy_action)
        if forced_action is None:
            return False, None, {}
        try:
            target_block = int(info.get("privileged/target_block", 0))
        except Exception:
            target_block = 0
        try:
            eff_pos = np.asarray(info.get("proprio/effector_pos"), dtype=np.float32).reshape(-1)
            block_pos = np.asarray(info.get(f"privileged/block_{target_block}_pos"), dtype=np.float32).reshape(-1)
            target_pos = np.asarray(info.get("privileged/target_block_pos"), dtype=np.float32).reshape(-1)
        except Exception:
            return False, None, {}
        if eff_pos.size < 3 or block_pos.size < 3 or target_pos.size < 3:
            return False, None, {}

        eff_block_dist = float(np.linalg.norm(eff_pos[:3] - block_pos[:3]))
        block_target_dist = float(np.linalg.norm(block_pos[:3] - target_pos[:3]))
        gripper_contact = self._extract_scalar(info.get("proprio/gripper_contact"), default=0.0)
        gripper_opening = self._extract_scalar(info.get("proprio/gripper_opening"), default=1.0)
        cube_max_error = self._extract_scalar(info.get("diag/cube_max_target_error"), default=np.inf)
        cube_success_tol = self._extract_scalar(info.get("diag/cube_success_tolerance"), default=0.04)

        holding = gripper_contact >= self.gripper_intervene_contact_threshold
        near_pick = eff_block_dist <= self.gripper_intervene_pick_radius
        near_place = block_target_dist <= self.gripper_intervene_place_radius
        place_ready = near_place or (cube_max_error <= cube_success_tol)
        # Ignore likely finger-to-finger self-contact when gripper is fully closed.
        # Treat "holding target" as contact while still near the target cube.
        not_fully_closed = gripper_opening > 0.05
        holding_target = holding and near_pick and not_fully_closed
        must_hold = (holding_target or near_pick) and (not place_ready)

        policy_sign = self._gripper_sign(policy_action)
        wants_release = policy_sign < 0
        violation = bool(must_hold and wants_release)
        diag = {
            "teacher_gripper_lock_violation": float(1.0 if violation else 0.0),
            "teacher_gripper_lock_must_hold": float(1.0 if must_hold else 0.0),
            "teacher_gripper_lock_place_ready": float(1.0 if place_ready else 0.0),
            "teacher_gripper_lock_wants_release": float(1.0 if wants_release else 0.0),
            "teacher_gripper_lock_policy_sign": float(policy_sign),
            "teacher_gripper_lock_holding_target": float(1.0 if holding_target else 0.0),
            "teacher_gripper_lock_not_fully_closed": float(1.0 if not_fully_closed else 0.0),
            "teacher_gripper_opening": float(gripper_opening),
            "teacher_gripper_eff_block_dist": float(eff_block_dist),
            "teacher_gripper_block_target_dist": float(block_target_dist),
            "teacher_gripper_contact": float(gripper_contact),
            "teacher_gripper_cube_max_error": float(cube_max_error),
            "teacher_gripper_cube_success_tol": float(cube_success_tol),
        }
        return violation, forced_action, diag

    def _classify_intervention_component(
        self,
        policy_action: Optional[np.ndarray],
        teacher_action: Optional[np.ndarray],
        reason: Optional[str],
    ):
        diag = {
            "teacher_delta_xyz_l2": 0.0,
            "teacher_delta_yaw_abs": 0.0,
            "teacher_delta_gripper_abs": 0.0,
            "teacher_reason_component": "none",
            "teacher_reason_component_is_movement_xyz": 0.0,
            "teacher_reason_component_is_yaw": 0.0,
            "teacher_reason_component_is_gripper": 0.0,
            "teacher_reason_component_is_mixed": 0.0,
        }
        if policy_action is None or teacher_action is None:
            return "none", diag
        p = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        t = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
        if p.size == 0 or t.size == 0:
            return "none", diag
        n = min(p.size, t.size)
        d = np.abs(p[:n] - t[:n]).astype(np.float32)

        xyz_l2 = float(np.linalg.norm(d[:3])) if n >= 3 else 0.0
        yaw_idx = self._yaw_index_for_size(n)
        gripper_idx = self._gripper_index_for_size(n)
        yaw_abs = float(d[yaw_idx]) if yaw_idx is not None else 0.0
        gripper_abs = float(d[gripper_idx]) if gripper_idx is not None else 0.0
        eps = 1e-6
        has_xyz = xyz_l2 > eps
        has_yaw = yaw_idx is not None and yaw_abs > eps
        has_gripper = gripper_idx is not None and gripper_abs > eps

        # If hard gripper branch decided intervention, keep label stable.
        if reason in {"gripper", "gripper_lock", "gripper_sync"}:
            component = "gripper"
        else:
            active_count = int(has_xyz) + int(has_yaw) + int(has_gripper)
            if active_count >= 2:
                component = "mixed"
            elif has_xyz:
                component = "movement_xyz"
            elif has_yaw:
                component = "yaw"
            elif has_gripper:
                component = "gripper"
            else:
                component = "none"

        diag["teacher_delta_xyz_l2"] = xyz_l2
        diag["teacher_delta_yaw_abs"] = yaw_abs
        diag["teacher_delta_gripper_abs"] = gripper_abs
        diag["teacher_reason_component"] = component
        diag["teacher_reason_component_is_movement_xyz"] = float(1.0 if component == "movement_xyz" else 0.0)
        diag["teacher_reason_component_is_yaw"] = float(1.0 if component == "yaw" else 0.0)
        diag["teacher_reason_component_is_gripper"] = float(1.0 if component == "gripper" else 0.0)
        diag["teacher_reason_component_is_mixed"] = float(1.0 if component == "mixed" else 0.0)
        return component, diag

    def _component_tolerance_scale(self, info: Optional[dict]) -> float:
        if not self.tolerance_adaptive_enable:
            return 1.0
        if info is None or not isinstance(info, dict):
            return 1.0
        try:
            target_block = int(info.get("privileged/target_block", 0))
            eff_pos = np.asarray(info.get("proprio/effector_pos"), dtype=np.float32).reshape(-1)
            block_pos = np.asarray(info.get(f"privileged/block_{target_block}_pos"), dtype=np.float32).reshape(-1)
            target_pos = np.asarray(info.get("privileged/target_block_pos"), dtype=np.float32).reshape(-1)
            if eff_pos.size < 3 or block_pos.size < 3 or target_pos.size < 3:
                return 1.0
            eff_block_dist = float(np.linalg.norm(eff_pos[:3] - block_pos[:3]))
            block_target_dist = float(np.linalg.norm(block_pos[:3] - target_pos[:3]))
            d = float(min(eff_block_dist, block_target_dist))
            near = max(0.0, self.tolerance_adaptive_near_distance)
            far = max(near + 1e-6, self.tolerance_adaptive_far_distance)
            near_scale = float(np.clip(self.tolerance_adaptive_near_scale, 1e-3, 1.0))
            if d <= near:
                return near_scale
            if d >= far:
                return 1.0
            alpha = (d - near) / (far - near)
            return float(near_scale + alpha * (1.0 - near_scale))
        except Exception:
            return 1.0

    def _component_thresholds(self, info: Optional[dict]) -> tuple[float, float, float, float]:
        xyz_base = self.tolerance_xyz_value if self.tolerance_xyz_value > 0.0 else self.tolerance_value
        yaw_base = self.tolerance_yaw_value if self.tolerance_yaw_value > 0.0 else self.tolerance_value
        grip_base = self.tolerance_gripper_value if self.tolerance_gripper_value > 0.0 else self.tolerance_value
        scale = self._component_tolerance_scale(info)
        return float(xyz_base * scale), float(yaw_base * scale), float(grip_base * scale), float(scale)

    def _component_diverged(
        self,
        policy_action: Optional[np.ndarray],
        teacher_action: Optional[np.ndarray],
        info: Optional[dict],
    ) -> tuple[bool, dict]:
        diag = {
            "teacher_component_threshold_xyz": 0.0,
            "teacher_component_threshold_yaw": 0.0,
            "teacher_component_threshold_gripper": 0.0,
            "teacher_component_threshold_scale": 1.0,
            "teacher_component_diverged_xyz": 0.0,
            "teacher_component_diverged_yaw": 0.0,
            "teacher_component_diverged_gripper": 0.0,
            "teacher_component_diverged_any": 0.0,
        }
        if policy_action is None or teacher_action is None:
            return False, diag
        p = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        t = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
        n = min(p.size, t.size)
        if n <= 0:
            return False, diag
        d = np.abs(p[:n] - t[:n]).astype(np.float32)
        xyz_l2 = float(np.linalg.norm(d[:3])) if n >= 3 else 0.0
        yaw_idx = self._yaw_index_for_size(n)
        gripper_idx = self._gripper_index_for_size(n)
        yaw_abs = float(d[yaw_idx]) if yaw_idx is not None else 0.0
        gripper_abs = float(d[gripper_idx]) if gripper_idx is not None else 0.0

        thr_xyz, thr_yaw, thr_grip, scale = self._component_thresholds(info)
        div_xyz = bool(xyz_l2 > thr_xyz) if n >= 3 else False
        div_yaw = bool(yaw_abs > thr_yaw) if yaw_idx is not None else False
        div_grip = bool(gripper_abs > thr_grip) if gripper_idx is not None else False
        diverged = bool(div_xyz or div_yaw or div_grip)

        diag["teacher_component_threshold_xyz"] = float(thr_xyz)
        diag["teacher_component_threshold_yaw"] = float(thr_yaw)
        diag["teacher_component_threshold_gripper"] = float(thr_grip)
        diag["teacher_component_threshold_scale"] = float(scale)
        diag["teacher_component_diverged_xyz"] = float(1.0 if div_xyz else 0.0)
        diag["teacher_component_diverged_yaw"] = float(1.0 if div_yaw else 0.0)
        diag["teacher_component_diverged_gripper"] = float(1.0 if div_grip else 0.0)
        diag["teacher_component_diverged_any"] = float(1.0 if diverged else 0.0)
        return diverged, diag

    def _current_episode_prob(self) -> float:
        if self.episode_intervention_prob_decay_steps <= 0:
            return self.episode_intervention_prob
        progress = max(0, self._global_step_env - self.episode_intervention_prob_decay_start)
        frac = min(1.0, progress / float(self.episode_intervention_prob_decay_steps))
        return self.episode_intervention_prob + (self.episode_intervention_prob_min - self.episode_intervention_prob) * frac

    def _manual_gate_draw(self, active: bool) -> None:
        cls = type(self)
        if not cls._manual_gate_ui_available or not cls._manual_gate_owns_window or cls._manual_gate_screen is None:
            return
        bg = (120, 20, 20) if active else (20, 20, 20)
        cls._manual_gate_screen.fill(bg)
        lines = [
            "Manual teacher gate: hold I for full teacher takeover.",
            "Release I for student movement + teacher gripper sync only.",
            "Speed: -/[ slower, =/] faster",
        ]
        y = 18
        for line in lines:
            surf = cls._manual_gate_font.render(line, True, (230, 230, 230))
            cls._manual_gate_screen.blit(surf, (10, y))
            y += 30
        status = cls._manual_gate_font.render(
            f"FPS: {float(cls._manual_gate_last_fps):.1f} | Teacher: {'ACTIVE' if active else 'idle'}",
            True,
            (230, 230, 230),
        )
        cls._manual_gate_screen.blit(status, (10, 108))
        q_line = cls._manual_gate_font.render(
            f"Q_min: {float(cls._manual_gate_last_q_min):+.3f} | Q_dis: {float(cls._manual_gate_last_q_dis):+.3f}",
            True,
            (230, 230, 230),
        )
        cls._manual_gate_screen.blit(q_line, (10, 132))

        def _dot_color(value: float, scale: float) -> tuple[int, int, int]:
            s = max(1e-6, float(scale))
            x = np.tanh(float(value) / s)
            t = 0.5 * (x + 1.0)
            red = int(max(0.0, min(255.0, 255.0 * (1.0 - t))))
            green = int(max(0.0, min(255.0, 255.0 * t)))
            return red, green, 40

        try:
            import pygame

            reward_items = [("Sparse", "sparse"), ("Dense", "dense"), ("Total", "total")]
            x = 16
            y_dot = 166
            for label, key in reward_items:
                value = float(cls._manual_gate_last_rewards.get(key, 0.0))
                scale = float(cls._manual_gate_reward_scales.get(key, 1.0))
                color = _dot_color(value, scale)
                pygame.draw.circle(cls._manual_gate_screen, color, (x, y_dot), 10)
                text = cls._manual_gate_font.render(f"{label}: {value:+.3f}", True, (230, 230, 230))
                cls._manual_gate_screen.blit(text, (x + 16, y_dot - 10))
                x += 230
        except Exception:
            pass
        try:
            import pygame
            pygame.display.flip()
        except Exception:
            pass
        cls._manual_gate_last_state = bool(active)

    def _manual_gate_update_status(self, *, active: bool, info: Optional[dict], reward_step: float) -> None:
        cls = type(self)
        cls._manual_gate_last_fps = float(self._manual_gate_target_fps)
        sparse = self._extract_scalar((info or {}).get("sparse_reward"), default=0.0) if isinstance(info, dict) else 0.0
        dense = self._extract_scalar((info or {}).get("dense_reward"), default=0.0) if isinstance(info, dict) else 0.0
        total = self._extract_scalar((info or {}).get("total_reward"), default=float(reward_step)) if isinstance(info, dict) else float(reward_step)
        cls._manual_gate_last_rewards = {
            "sparse": float(sparse),
            "dense": float(dense),
            "total": float(total),
        }
        cls._manual_gate_reward_scales["sparse"] = max(float(cls._manual_gate_reward_scales.get("sparse", 1.0)), abs(float(sparse)), 1e-3)
        cls._manual_gate_reward_scales["dense"] = max(float(cls._manual_gate_reward_scales.get("dense", 0.1)), abs(float(dense)), 1e-3)
        cls._manual_gate_reward_scales["total"] = max(float(cls._manual_gate_reward_scales.get("total", 0.5)), abs(float(total)), 1e-3)
        if cls._manual_gate_owns_window:
            self._manual_gate_draw(bool(active))

    @classmethod
    def set_manual_gate_q_stats(cls, *, q_min: float, q_disagreement: float) -> None:
        cls._manual_gate_last_q_min = float(q_min)
        cls._manual_gate_last_q_dis = float(q_disagreement)

    def _manual_gate_setup(self) -> None:
        if not self._manual_gate_enabled:
            return
        cls = type(self)
        if cls._manual_gate_ui_initialized:
            return
        cls._manual_gate_ui_initialized = True
        try:
            import pygame

            if not pygame.get_init():
                pygame.init()
            screen = pygame.display.get_surface()
            if screen is None:
                screen = pygame.display.set_mode((760, 190))
                pygame.display.set_caption("Intervention Manual Gate")
                cls._manual_gate_owns_window = True
            cls._manual_gate_screen = screen
            cls._manual_gate_font = pygame.font.SysFont("Arial", 18)
            cls._manual_gate_ui_available = True
            cls._manual_gate_last_state = None
            cls._manual_gate_last_fps = float(self._manual_gate_target_fps)
            self._manual_gate_draw(False)
        except Exception as exc:
            cls._manual_gate_ui_available = False
            if not cls._manual_gate_warned:
                print(f"[InterventionWrapper] manual gate UI unavailable: {exc}")
                cls._manual_gate_warned = True

    def _manual_gate_active(self) -> bool:
        if not self._manual_gate_enabled:
            return False
        self._manual_gate_setup()
        cls = type(self)
        if not cls._manual_gate_ui_available:
            return False
        active = False
        try:
            import pygame

            # Consume events in this dedicated window to support runtime FPS hotkeys.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_MINUS, pygame.K_LEFTBRACKET):
                        self._manual_gate_target_fps = max(self._manual_gate_min_fps, self._manual_gate_target_fps - 2.0)
                        if cls._manual_gate_owns_window:
                            self._manual_gate_draw(bool(cls._manual_gate_last_state))
                    elif event.key in (pygame.K_EQUALS, pygame.K_RIGHTBRACKET, pygame.K_PLUS):
                        self._manual_gate_target_fps = min(self._manual_gate_max_fps, self._manual_gate_target_fps + 2.0)
                        if cls._manual_gate_owns_window:
                            self._manual_gate_draw(bool(cls._manual_gate_last_state))
            keys = pygame.key.get_pressed()
            active = bool(keys[pygame.K_i])
        except Exception:
            active = False
        if cls._manual_gate_owns_window and cls._manual_gate_last_state != bool(active):
            self._manual_gate_draw(bool(active))
        return bool(active)

    # ---------------
    # Human utilities
    # ---------------
    def _human_action(self) -> Optional[np.ndarray]:
        if self.teleop is None or not hasattr(self.teleop, 'get_action'):
            return None
        human_action = self.teleop.get_action()
        if human_action is None:
            return None
        if np.linalg.norm(human_action) > self.threshold:
            self._last_override_ts = time.perf_counter()
        is_active = (time.perf_counter() - self._last_override_ts) < self.hold_time
        return human_action if is_active else None

    # ----------------
    # Agent utilities
    # ----------------
    def _bfs_teacher_action(self) -> Optional[np.ndarray]:
        """Use env oracle subgoal to compute a direction action towards the next waypoint."""
        try:
            agent_xy = np.array(self.unwrapped.get_xy(), dtype=np.float32)
            # Goal from env state; fallback to observation if needed
            goal_xy = np.array(getattr(self.unwrapped, 'cur_goal_xy', None), dtype=np.float32)
            if goal_xy is None or goal_xy.shape != (2,):
                # Fallback: best-effort from observation
                obs = getattr(self.unwrapped, 'get_ob', lambda: None)()
                if isinstance(obs, np.ndarray) and obs.shape[0] >= 4:
                    goal_xy = obs[2:4].astype(np.float32)
                else:
                    return None

            # Query env for oracle subgoal (BFS one-step waypoint)
            subgoal_out = self.unwrapped.get_oracle_subgoal(agent_xy, goal_xy)
            if isinstance(subgoal_out, (list, tuple)):
                subgoal_xy = np.array(subgoal_out[0], dtype=np.float32)
            else:
                subgoal_xy = np.array(subgoal_out, dtype=np.float32)
            if subgoal_xy is None or subgoal_xy.shape != (2,):
                return None

            direction = subgoal_xy - agent_xy
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                return np.zeros_like(direction)
            unit = direction / norm
            # Scale to action space range
            if hasattr(self.env.action_space, 'high'):
                max_mag = float(np.min(self.env.action_space.high))
                max_mag = 1.0 if not np.isfinite(max_mag) or max_mag <= 0 else max_mag
            else:
                max_mag = 1.0
            return unit * max_mag
        except Exception:
            return None

    def _teacher_action(self, obs, info) -> Optional[np.ndarray]:
        if self.teacher_type == 'bfs':
            return self._bfs_teacher_action()
        if self.teacher_type in {"cube_plan", "cube_markov"} and self._oracle is not None:
            try:
                current_target_block = info.get('privileged/target_block') if isinstance(info, dict) else None
                try:
                    current_target_block = int(current_target_block)
                except Exception:
                    current_target_block = None

                oracle_done = bool(getattr(self._oracle, 'done', False))
                target_switched = (
                    current_target_block is not None
                    and self._oracle_target_block is not None
                    and current_target_block != self._oracle_target_block
                )
                if oracle_done or target_switched:
                    self._oracle.reset(obs, info or {})
                    if current_target_block is not None:
                        self._oracle_target_block = current_target_block

                return self._oracle.select_action(obs, info or {})
            except Exception:
                return None
        return None

    def _angle_deg(self, a: np.ndarray, b: np.ndarray) -> float:
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an < 1e-8 or bn < 1e-8:
            return 180.0
        cos = float(np.clip(np.dot(a, b) / (an * bn), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    def _predict_next_xy(self, action: np.ndarray) -> Optional[Tuple[int, int]]:
        """Predict next grid cell if we applied this action (PointEnv dynamics)."""
        try:
            cur_xy = np.array(self.unwrapped.get_xy(), dtype=np.float32)
            # PointEnv applies action scaled by 0.2 per step
            next_xy = cur_xy + 0.2 * action
            i, j = self.unwrapped.xy_to_ij(next_xy)
            return i, j
        except Exception:
            return None

    def _is_traversable_cell(self, ij: Tuple[int, int]) -> bool:
        """Delegate to env.is_traversable(i, j) when available; default True if unknown."""
        try:
            i, j = ij
            return bool(self.unwrapped.is_traversable(i, j))
        except Exception:
            # Fallback: be permissive if API missing
            return True

    def _danger_centers_xy(self) -> Optional[np.ndarray]:
        """Cache and return dangerous tile centers as an (N, 2) array."""
        if self._danger_centers is not None:
            return self._danger_centers
        base = self.unwrapped
        maze_map = getattr(base, 'maze_map', None)
        dangerous_id = getattr(base, '_dangerous_tile_id', None)
        ij_to_xy = getattr(base, 'ij_to_xy', None)
        maze_unit = getattr(base, '_maze_unit', None)
        if maze_map is None or dangerous_id is None or ij_to_xy is None:
            return None
        centers = []
        for i in range(maze_map.shape[0]):
            for j in range(maze_map.shape[1]):
                if maze_map[i, j] == dangerous_id:
                    centers.append(ij_to_xy((i, j)))
        if not centers:
            self._danger_centers = None
            return None
        self._danger_centers = np.asarray(centers, dtype=np.float32)
        self._maze_unit = float(maze_unit) if maze_unit is not None else 1.0
        return self._danger_centers

    def _near_danger_margin(self) -> bool:
        if self.safety_margin_frac <= 0:
            return False
        centers = self._danger_centers_xy()
        if centers is None:
            return False
        try:
            agent_xy = np.array(self.unwrapped.get_xy(), dtype=np.float32)
        except Exception:
            return False
        maze_unit = self._maze_unit if self._maze_unit is not None else 1.0
        margin = self.safety_margin_frac * maze_unit
        dists = np.linalg.norm(centers - agent_xy[None, :], axis=1)
        min_dist = float(np.min(dists))
        dist_to_boundary = max(0.0, min_dist - 0.5 * maze_unit)
        return dist_to_boundary <= margin

    def _goal_distance(self) -> Optional[float]:
        try:
            agent_xy = np.array(self.unwrapped.get_xy(), dtype=np.float32)
            goal_xy = np.array(getattr(self.unwrapped, 'cur_goal_xy', None), dtype=np.float32)
        except Exception:
            return None
        if goal_xy is None or goal_xy.shape != (2,):
            return None
        return float(np.linalg.norm(goal_xy - agent_xy))

    def _reward_progress_signal(self, info: Optional[dict], reward_hint: Optional[float]) -> Optional[float]:
        """Return a scalar signal that should increase when behavior improves."""
        if isinstance(info, dict):
            # Cube adapter diagnostic: lower error is better, so negate it.
            # This must take priority over wrapper-level dense fields because manip can run
            # with reward_type=sparse (dense_reward exists but is identically zero), which
            # otherwise causes reward-progress interventions to latch on permanently.
            if "diag/cube_max_target_error" in info:
                err = self._extract_scalar(info.get("diag/cube_max_target_error"), default=0.0)
                return -float(err)
            # Prefer explicit dense metrics when available.
            for key in ("mode_dense_pre", "mode_dense", "dense_phase_reward"):
                if key in info:
                    return self._extract_scalar(info.get(key), default=0.0)
            reward_type = str(info.get("reward_type", "")).strip().lower()
            if reward_type != "sparse":
                for key in ("dense_reward", "dense_reward_raw"):
                    if key in info:
                        return self._extract_scalar(info.get(key), default=0.0)
        if reward_hint is not None:
            try:
                return float(reward_hint)
            except Exception:
                return None
        return None

    # --------------
    # Gym overrides
    # --------------
    def reset(self, **kwargs):
        if self._manual_gate_enabled:
            self._manual_gate_setup()
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_info = info
        if self.mode == 'human' and self.teleop is not None and hasattr(self.teleop, "reset"):
            try:
                self.teleop.reset()
            except Exception:
                pass
        if self.teacher_type in {"cube_plan", "cube_markov"}:
            if self._oracle is None or self._oracle_type != self.teacher_type:
                if self.teacher_type == "cube_plan":
                    from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
                    self._oracle = CubePlanOracle(env=self.unwrapped)
                else:
                    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
                    self._oracle = CubeMarkovOracle(env=self.unwrapped)
                self._oracle_type = self.teacher_type
            if self._oracle is not None:
                try:
                    self._oracle.reset(obs, info)
                    try:
                        self._oracle_target_block = int(info.get('privileged/target_block'))
                    except Exception:
                        self._oracle_target_block = None
                except Exception:
                    pass
        # reset stats
        self._stats = _InterventionStats()
        # reset human timer
        self._last_override_ts = 0.0
        self._align_count = 0
        self._progress_good_count = 0
        self._progress_violation = False
        self._safety_align_active = False
        self._progress_active = False
        self._reward_progress_active = False
        self._reward_non_improve_count = 0
        self._reward_improve_count = 0
        self._reward_cooldown_remaining = 0
        self._reward_delta_window.clear()
        self._reward_window_sum = 0.0
        self._reward_window_ready = False
        self._last_distance = self._goal_distance()
        self._last_reward_progress_signal = self._reward_progress_signal(info, reward_hint=None)
        prob = float(np.clip(self._current_episode_prob(), 0.0, 1.0))
        self._episode_interventions_enabled = bool(self._rng.random() < prob)
        self._stats.episode_gate_enabled = 1 if self._episode_interventions_enabled else 0
        self._stats.episode_gate_prob = prob
        return obs, info

    def step(self, policy_action: np.ndarray):
        step_start = time.perf_counter()
        # Decide override
        policy_action = self._coerce_action_dim(policy_action, self._action_dim)
        policy_action = self._apply_gripper_binary(policy_action)
        teacher_action = None
        teacher_candidate_action = None
        reason = None
        teacher_delta_l2 = 0.0
        teacher_delta_l2_raw = 0.0
        teacher_delta_angle_deg = 0.0
        teacher_candidate_available = False
        gripper_diag = {}
        gripper_sync_diag = {
            "teacher_gripper_sync_applied": 0.0,
            "teacher_gripper_sync_policy_sign": 0.0,
            "teacher_gripper_sync_teacher_sign": 0.0,
        }
        manual_gate_active = False
        component_reason = "none"
        component_diag = {}
        component_diag_for_logging = {}
        component_threshold_diag = {
            "teacher_component_threshold_xyz": 0.0,
            "teacher_component_threshold_yaw": 0.0,
            "teacher_component_threshold_gripper": 0.0,
            "teacher_component_threshold_scale": 1.0,
            "teacher_component_diverged_xyz": 0.0,
            "teacher_component_diverged_yaw": 0.0,
            "teacher_component_diverged_gripper": 0.0,
            "teacher_component_diverged_any": 0.0,
        }

        if self.mode == 'human':
            human = self._human_action()
            if human is not None:
                teacher_candidate_action = self._coerce_action_dim(human, self._action_dim)
                teacher_candidate_action = self._apply_gripper_binary(teacher_candidate_action)
                teacher_candidate_available = teacher_candidate_action is not None
                teleop_diag = {}
                if self.teleop is not None and hasattr(self.teleop, "get_last_diag"):
                    try:
                        teleop_diag = dict(self.teleop.get_last_diag() or {})
                    except Exception:
                        teleop_diag = {}
                gate_pressed = bool(teleop_diag.get("gate_pressed", False))
                reset_gate_latched = bool(teleop_diag.get("reset_gate_latched", False))
                if gate_pressed and not reset_gate_latched:
                    teacher_action = teacher_candidate_action
                    reason = 'human'
                else:
                    teacher_action, gripper_sync_diag = self._force_gripper_channel_to_teacher(
                        policy_action,
                        teacher_candidate_action,
                    )
                    if teacher_action is not None:
                        reason = 'gripper_sync'
        else:  # agent mode
            if not self._episode_interventions_enabled:
                teacher_action = None
            else:
                # Respect warmup schedule
                if self._global_step_env < self.enable_after_steps:
                    teacher_action = None
                else:
                    teacher_action = self._teacher_action(self._last_obs, self._last_info or {})
                    teacher_action = self._coerce_action_dim(teacher_action, self._action_dim)
                    teacher_action = self._apply_gripper_binary(teacher_action)
                manual_gate_active = self._manual_gate_active()
                teacher_candidate_action = teacher_action
                teacher_candidate_available = teacher_candidate_action is not None
                if teacher_candidate_available and policy_action is not None:
                    teacher_delta_l2_raw = self._l2_delta(policy_action, teacher_candidate_action, weighted=False)
                    teacher_delta_l2 = self._l2_delta(policy_action, teacher_candidate_action, weighted=True)
                    teacher_delta_angle_deg = float(self._angle_deg(policy_action, teacher_candidate_action))
                    _, component_diag_for_logging = self._classify_intervention_component(
                        policy_action, teacher_candidate_action, None
                    )
                    if self.tolerance_type == "component":
                        _, component_threshold_diag = self._component_diverged(
                            policy_action,
                            teacher_candidate_action,
                            self._last_info or {},
                        )
                gripper_lock_violation, gripper_lock_action, gripper_lock_diag = self._gripper_lock_violation(
                    self._last_info or {},
                    policy_action,
                )
                hard_gripper_violation, gripper_diag = self._hard_gripper_violation(
                    self._last_info or {},
                    policy_action,
                    teacher_action,
                )
                if gripper_lock_diag:
                    gripper_diag.update(gripper_lock_diag)
                # Safety check (maze-only)
                safety_violation = False
                safety_margin = False
                if self.teacher_type == "bfs":
                    if self.hard_block_lethal and teacher_action is not None and policy_action is not None:
                        predicted = self._predict_next_xy(policy_action)
                        if predicted is not None and (not self._is_traversable_cell(predicted)):
                            safety_violation = True
                    safety_margin = self._near_danger_margin()

                if self.agent_mode == 'always':
                    if gripper_lock_violation:
                        teacher_action = gripper_lock_action
                        reason = 'gripper_lock'
                    elif hard_gripper_violation:
                        reason = 'gripper'
                    elif teacher_action is not None:
                        reason = 'always'
                    else:
                        teacher_action = None
                elif self.agent_mode == 'divergence':
                    diverged = False
                    if teacher_action is not None and policy_action is not None and not safety_violation:
                        if self.tolerance_type == 'angle':
                            ang = self._angle_deg(policy_action, teacher_action)
                            diverged = ang > self.tolerance_value
                        elif self.tolerance_type == 'component':
                            diverged, component_threshold_diag = self._component_diverged(
                                policy_action, teacher_action, self._last_info or {}
                            )
                        else:
                            diverged = self._l2_delta(policy_action, teacher_action, weighted=True) > self.tolerance_value
                    if gripper_lock_violation:
                        teacher_action = gripper_lock_action
                        reason = 'gripper_lock'
                    elif hard_gripper_violation:
                        reason = 'gripper'
                    elif safety_violation:
                        reason = 'safety'
                    elif diverged:
                        reason = 'divergence'
                    else:
                        teacher_action = None
                elif self.agent_mode == 'safety_align':
                    if gripper_lock_violation:
                        teacher_action = gripper_lock_action
                        reason = 'gripper_lock'
                    elif hard_gripper_violation:
                        reason = 'gripper'
                    else:
                        aligned = False
                        if teacher_action is not None and policy_action is not None:
                            if self.tolerance_type == 'angle':
                                aligned = self._angle_deg(policy_action, teacher_action) <= self.tolerance_value
                            elif self.tolerance_type == 'component':
                                diverged_component, component_threshold_diag = self._component_diverged(
                                    policy_action, teacher_action, self._last_info or {}
                                )
                                aligned = not diverged_component
                            else:
                                aligned = self._l2_delta(policy_action, teacher_action, weighted=True) <= self.tolerance_value
                        if aligned:
                            self._align_count += 1
                        else:
                            self._align_count = 0

                        if safety_violation or safety_margin:
                            self._safety_align_active = True

                        if self._safety_align_active:
                            if (not safety_violation and not safety_margin and self._align_count >= self.release_steps):
                                self._safety_align_active = False
                            else:
                                reason = 'safety'
                        if not self._safety_align_active:
                            teacher_action = None
                elif self.agent_mode == 'safety_progress':
                    if gripper_lock_violation:
                        teacher_action = gripper_lock_action
                        reason = 'gripper_lock'
                    elif hard_gripper_violation:
                        reason = 'gripper'
                    else:
                        if safety_violation or safety_margin or self._progress_violation:
                            self._progress_active = True

                        if self._progress_active:
                            if (not safety_violation and not safety_margin and self._progress_good_count >= self.release_steps):
                                self._progress_active = False
                            else:
                                reason = 'safety' if (safety_violation or safety_margin) else 'progress'
                        if not self._progress_active:
                            teacher_action = None
                elif self.agent_mode == 'reward_progress':
                    reward_signal_available = self._last_reward_progress_signal is not None
                    window_bad = self._reward_window_ready and (self._reward_window_sum <= self.reward_improvement_epsilon)
                    window_good = self._reward_window_ready and (self._reward_window_sum > self.reward_improvement_epsilon)
                    cooldown_active = (
                        (self._reward_cooldown_remaining > 0)
                        and (not safety_violation)
                        and (not safety_margin)
                    )
                    if gripper_lock_violation:
                        teacher_action = gripper_lock_action
                        reason = 'gripper_lock'
                    elif hard_gripper_violation:
                        reason = 'gripper'
                    else:
                        if safety_violation or safety_margin:
                            self._reward_progress_active = True
                        elif (
                            (not cooldown_active)
                            and reward_signal_available
                            and window_bad
                        ):
                            self._reward_progress_active = True

                        if self._reward_progress_active:
                            if (
                                not safety_violation
                                and not safety_margin
                                and (
                                    (not reward_signal_available)
                                    or window_good
                                )
                            ):
                                self._reward_progress_active = False
                                self._reward_non_improve_count = 0
                                self._reward_improve_count = 0
                                # +1 because countdown decrement happens at end-of-step.
                                self._reward_cooldown_remaining = int(self.reward_patience_steps) + 1
                            else:
                                reason = 'safety' if (safety_violation or safety_margin) else 'progress'
                        if not self._reward_progress_active:
                            teacher_action = None
                else:  # manual_gripper
                    if manual_gate_active and teacher_action is not None:
                        reason = "manual_gate"
                    else:
                        forced_gripper_action, gripper_sync_diag = self._force_gripper_to_teacher(
                            policy_action,
                            teacher_action,
                        )
                        if forced_gripper_action is not None:
                            teacher_action = forced_gripper_action
                            reason = "gripper_sync"
                        else:
                            teacher_action = None

        # Apply action
        intervened = teacher_action is not None
        action_to_take = teacher_action if intervened else policy_action
        obs, reward, terminated, truncated, info = self.env.step(action_to_take)
        self._last_obs = obs
        self._last_info = info

        # Update progress tracking based on actual next state
        new_distance = self._goal_distance()
        if new_distance is not None and self._last_distance is not None:
            delta = new_distance - self._last_distance
            self._progress_violation = delta > 0.0
            if self._progress_active:
                if delta < 0.0:
                    self._progress_good_count += 1
                else:
                    self._progress_good_count = 0
            else:
                self._progress_good_count = 0
        self._last_distance = new_distance

        # Reward-progress tracking (increase = improvement).
        reward_signal = self._reward_progress_signal(info, reward_hint=reward)
        reward_signal_delta = 0.0
        reward_improved = False
        if reward_signal is not None and self._last_reward_progress_signal is not None:
            reward_signal_delta = float(reward_signal - self._last_reward_progress_signal)
            self._reward_delta_window.append(reward_signal_delta)
            if len(self._reward_delta_window) >= self.reward_patience_steps:
                self._reward_window_ready = True
                self._reward_window_sum = float(sum(self._reward_delta_window))
            else:
                self._reward_window_ready = False
                self._reward_window_sum = 0.0

            reward_improved = self._reward_window_ready and (self._reward_window_sum > self.reward_improvement_epsilon)
            if reward_improved:
                self._reward_improve_count += 1
                self._reward_non_improve_count = 0
            else:
                self._reward_non_improve_count += 1
                self._reward_improve_count = 0
        elif reward_signal is None:
            self._reward_non_improve_count = 0
            self._reward_improve_count = 0
            self._reward_delta_window.clear()
            self._reward_window_sum = 0.0
            self._reward_window_ready = False
        self._last_reward_progress_signal = reward_signal
        if (not self._reward_progress_active) and self._reward_cooldown_remaining > 0:
            self._reward_non_improve_count = 0
            self._reward_improve_count = 0
            self._reward_cooldown_remaining = max(0, int(self._reward_cooldown_remaining) - 1)

        # Stats update
        self._stats.episode_steps += 1
        if intervened:
            component_reason, component_diag = self._classify_intervention_component(policy_action, teacher_action, reason)
            self._stats.intervention_steps += 1
            self._stats.start_burst()
            self._stats.step_burst()
            if reason == 'safety':
                self._stats.num_safety += 1
            elif reason in {'divergence', 'always', 'human'}:
                self._stats.num_divergence += 1
            elif reason == 'progress':
                self._stats.num_progress += 1
            elif reason in {'gripper', 'gripper_lock', 'gripper_sync'}:
                self._stats.num_gripper += 1
            elif reason == 'manual_gate':
                self._stats.num_manual += 1
            if component_reason == 'movement_xyz':
                self._stats.num_component_movement_xyz += 1
            elif component_reason == 'yaw':
                self._stats.num_component_yaw += 1
            elif component_reason == 'gripper':
                self._stats.num_component_gripper += 1
            elif component_reason == 'mixed':
                self._stats.num_component_mixed += 1
            # Count new interventions when a burst starts at this step
            if self._stats._current_burst_len == 1:
                self._stats.num_interventions += 1
        else:
            self._stats.end_burst_if_needed()
        # Increment global step counter regardless of intervention
        self._global_step_env += 1

        # Annotate info
        info['teacher_intervened'] = bool(intervened)
        info['teacher_candidate_available'] = bool(teacher_candidate_available)
        info['teacher_delta_l2'] = float(teacher_delta_l2)
        info['teacher_delta_l2_raw'] = float(teacher_delta_l2_raw)
        info['teacher_delta_angle_deg'] = float(teacher_delta_angle_deg)
        info['teacher_tolerance_value'] = float(self.tolerance_value)
        info['teacher_reward_progress_signal'] = float(reward_signal) if reward_signal is not None else 0.0
        info['teacher_reward_progress_signal_available'] = float(1.0 if reward_signal is not None else 0.0)
        info['teacher_reward_progress_signal_delta'] = float(reward_signal_delta)
        info['teacher_reward_progress_improved'] = float(1.0 if reward_improved else 0.0)
        info['teacher_reward_non_improve_count'] = int(self._reward_non_improve_count)
        info['teacher_reward_improve_count'] = int(self._reward_improve_count)
        info['teacher_reward_window_ready'] = float(1.0 if self._reward_window_ready else 0.0)
        info['teacher_reward_window_sum'] = float(self._reward_window_sum)
        info['teacher_reward_cooldown_remaining'] = int(self._reward_cooldown_remaining)
        info['teacher_reward_progress_active'] = float(1.0 if self._reward_progress_active else 0.0)
        info['teacher_reward_patience_steps'] = int(self.reward_patience_steps)
        info['teacher_manual_gate_active'] = float(1.0 if manual_gate_active else 0.0)
        if self.tolerance_channel_weights is not None:
            info['teacher_tolerance_channel_weights'] = np.array(self.tolerance_channel_weights, dtype=np.float32)
        if gripper_diag:
            info.update(gripper_diag)
        if gripper_sync_diag:
            info.update(gripper_sync_diag)
        if component_diag_for_logging:
            info.update(component_diag_for_logging)
        if component_threshold_diag:
            info.update(component_threshold_diag)
        if component_diag:
            info.update(component_diag)
        if intervened:
            info['teacher_reason'] = reason
            info['teacher_reason_component'] = component_reason
            info['teacher_action'] = np.array(teacher_action, dtype=np.float32)
            info['student_action'] = np.array(policy_action, dtype=np.float32)
        else:
            info['teacher_reason'] = None
            info['teacher_reason_component'] = 'none'
            if teacher_candidate_available and teacher_candidate_action is not None:
                info['teacher_action'] = np.array(teacher_candidate_action, dtype=np.float32)
            info['student_action'] = np.array(policy_action, dtype=np.float32)
        info['applied_action'] = np.array(action_to_take, dtype=np.float32)

        # On episode end, flush metrics for logging
        if terminated or truncated:
            ep_metrics = self._stats.flush_metrics()
            info.update(ep_metrics)

        if self._manual_gate_enabled:
            self._manual_gate_update_status(active=manual_gate_active, info=info if isinstance(info, dict) else None, reward_step=float(reward))
            target_dt = 1.0 / max(1.0, float(self._manual_gate_target_fps))
            elapsed = time.perf_counter() - step_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

        return obs, reward, terminated, truncated, info

    def close(self):
        cls = type(self)
        if self._manual_gate_enabled and cls._manual_gate_owns_window and cls._manual_gate_ui_available:
            try:
                import pygame

                pygame.quit()
            except Exception:
                pass
            cls._manual_gate_ui_initialized = False
            cls._manual_gate_ui_available = False
            cls._manual_gate_owns_window = False
            cls._manual_gate_screen = None
            cls._manual_gate_font = None
            cls._manual_gate_last_state = None
        return self.env.close()


# Backwards-compatible alias for existing imports/usages
HumanInterventionWrapper = InterventionWrapper


class DirectTeleopWrapper(gym.Wrapper):
    """
    A wrapper that allows direct human control of the environment.
    Unlike HumanInterventionWrapper, this provides complete human control
    without any autonomous policy.
    
    This is useful for:
    - Manual environment exploration.
    - Collecting human demonstrations.
    - Testing environment mechanics.
    """

    def __init__(self, env: gym.Env, teleop_interface):
        """
        Initializes the wrapper.

        Args:
            env (gym.Env): The Gymnasium environment to wrap.
            teleop_interface: An object with a `get_action()` method that returns
                              a NumPy array of the same shape as the env's action space.
        """
        super().__init__(env)
        
        if not hasattr(teleop_interface, "get_action"):
            raise TypeError("teleop_interface must have a 'get_action' method.")
        
        self.teleop = teleop_interface
        
        print("[DirectTeleopWrapper] Initialized for direct human control.")

    def reset(self, **kwargs):
        """
        Resets the environment.
        """
        obs, info = self.env.reset(**kwargs)
        
        if hasattr(self.teleop, "reset"):
            self.teleop.reset()
            
        return obs, info

    def step(self, policy_action=None):
        """
        Executes a step in the environment using only human input.
        The policy_action parameter is ignored.

        Args:
            policy_action: Ignored. Kept for compatibility.

        Returns:
            The standard (obs, reward, terminated, truncated, info) tuple.
        """
        # Always use human action
        human_action = self.teleop.get_action()
        
        # Step the wrapped environment
        obs, reward, terminated, truncated, info = self.env.step(human_action)
        self._last_obs = obs
        self._last_info = info

        # Annotate info with action source
        info["human_action"] = human_action
        info["control_mode"] = "human"
        
        return obs, reward, terminated, truncated, info
