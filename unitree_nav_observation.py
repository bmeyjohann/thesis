from __future__ import annotations

import torch


def prepare_unitree_actor_obs(
    obs: torch.Tensor,
    *,
    mask_proprioception: bool = False,
    mask_goal_heading: bool = False,
    mask_height_scan: bool = False,
    goal_encoding: str = "cartesian",
    goal_distance_scale: float = 14.0,
    velocity_scale: float = 1.0,
) -> torch.Tensor:
    """Apply policy-facing observation masks without changing checkpoint dimensions."""
    if (
        not mask_proprioception
        and not mask_goal_heading
        and not mask_height_scan
        and goal_encoding == "cartesian"
        and velocity_scale == 1.0
    ):
        return obs
    if obs.shape[-1] < 9:
        raise ValueError(f"Expected at least 9 Unitree observation values, got {obs.shape[-1]}")
    obs = obs.clone()
    if velocity_scale <= 0.0:
        raise ValueError("velocity_scale must be positive")
    obs[..., :3] = (obs[..., :3] / float(velocity_scale)).clamp(-1.0, 1.0)
    if goal_encoding == "distance_bearing":
        goal_xy = obs[..., 6:8].clone()
        distance = torch.linalg.norm(goal_xy, dim=-1)
        safe_distance = distance.clamp_min(1e-6)
        obs[..., 6] = (distance / float(goal_distance_scale)).clamp(0.0, 1.0)
        obs[..., 7] = torch.where(distance > 1e-6, goal_xy[..., 0] / safe_distance, 0.0)
        obs[..., 8] = torch.where(distance > 1e-6, goal_xy[..., 1] / safe_distance, 0.0)
    elif goal_encoding != "cartesian":
        raise ValueError(f"Unsupported goal encoding: {goal_encoding}")
    if mask_proprioception:
        obs[..., :6] = 0.0
    if mask_goal_heading and goal_encoding == "cartesian":
        obs[..., 8] = 0.0
    if mask_height_scan and obs.shape[-1] > 9:
        obs[..., 9:] = 0.0
    return obs


def unitree_goal_distance(
    obs: torch.Tensor, *, goal_encoding: str, goal_distance_scale: float
) -> torch.Tensor:
    if goal_encoding == "distance_bearing":
        return obs[..., 6] * float(goal_distance_scale)
    return torch.linalg.norm(obs[..., 6:8], dim=-1)

def unitree_goal_xy(
    obs: torch.Tensor, *, goal_encoding: str, goal_distance_scale: float
) -> torch.Tensor:
    if goal_encoding == "distance_bearing":
        distance = obs[..., 6] * float(goal_distance_scale)
        return torch.stack((distance * obs[..., 7], distance * obs[..., 8]), dim=-1)
    return obs[..., 6:8]


class UnitreeScanHistory:
    """Keep current context plus reset-aware, optionally sparse height scans."""

    def __init__(
        self,
        history: int = 1,
        action_history: int = 0,
        action_dim: int = 3,
        history_stride: int = 1,
    ):
        self.history = max(1, int(history))
        self.action_history = max(0, int(action_history))
        self.action_dim = int(action_dim)
        self.history_stride = max(1, int(history_stride))
        self._scans: torch.Tensor | None = None
        self._scan_anchors: torch.Tensor | None = None
        self._scan_steps: torch.Tensor | None = None
        self._actions: torch.Tensor | None = None
        self.scan_dim: int | None = None

    def reset(self, obs: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        context, scan = self._split(obs)
        if self._scans is None or self._scans.shape[0] != obs.shape[0]:
            self._scans = scan[:, None, :].repeat(1, self.history, 1)
            self._scan_anchors = scan.clone()
            self._scan_steps = torch.zeros(obs.shape[0], device=obs.device, dtype=torch.long)
            self._actions = torch.zeros(
                obs.shape[0], self.action_history, self.action_dim, device=obs.device, dtype=obs.dtype
            )
        elif env_ids is None:
            self._scans[:] = scan[:, None, :]
            assert self._scan_anchors is not None and self._scan_steps is not None
            self._scan_anchors[:] = scan
            self._scan_steps.zero_()
            if self._actions is not None:
                self._actions.zero_()
        else:
            self._scans[env_ids] = scan[env_ids, None, :]
            assert self._scan_anchors is not None and self._scan_steps is not None
            self._scan_anchors[env_ids] = scan[env_ids]
            self._scan_steps[env_ids] = 0
            if self._actions is not None:
                self._actions[env_ids] = 0.0
        return self._join(context)

    def step(
        self,
        obs: torch.Tensor,
        done: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context, scan = self._split(obs)
        if self._scans is None:
            return self.reset(obs)
        assert self._scan_anchors is not None and self._scan_steps is not None
        self._scan_steps += 1
        capture = self._scan_steps >= self.history_stride
        if self.history > 1 and capture.any():
            captured_history = self._scans[capture].clone()
            if self.history > 2:
                self._scans[capture, 2:, :] = captured_history[:, 1:-1, :]
            self._scans[capture, 1, :] = self._scan_anchors[capture]
            self._scan_anchors[capture] = scan[capture]
            self._scan_steps[capture] = 0
        self._scans[:, 0, :] = scan
        if self.action_history:
            if action is None:
                raise ValueError("Unitree action history requires the executed action on every step")
            assert self._actions is not None
            self._actions = torch.roll(self._actions, shifts=1, dims=1)
            self._actions[:, 0, :] = action
        if done is not None and done.any():
            self._scans[done] = scan[done, None, :]
            self._scan_anchors[done] = scan[done]
            self._scan_steps[done] = 0
            if self._actions is not None:
                self._actions[done] = 0.0
        return self._join(context)

    def _split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim != 2 or obs.shape[1] <= 9:
            raise ValueError(f"Expected Unitree observations [N, 9 + scan], got {tuple(obs.shape)}")
        scan = obs[:, 9:]
        self.scan_dim = int(scan.shape[1])
        return obs[:, :9], scan

    def _join(self, context: torch.Tensor) -> torch.Tensor:
        assert self._scans is not None
        parts = [context, self._scans.flatten(start_dim=1)]
        if self._actions is not None:
            parts.append(self._actions.flatten(start_dim=1))
        return torch.cat(parts, dim=1)


def current_unitree_scan_obs(
    obs: torch.Tensor,
    *,
    scan_history: int | None = None,
    action_history: int = 0,
    action_dim: int = 3,
) -> torch.Tensor:
    """Return the current 9D context + first scan frame for scripted teachers."""
    scan_values = obs.shape[1] - 9 - max(0, int(action_history)) * int(action_dim)
    if scan_values <= 0:
        return obs
    # For history-stacked observations the first frame is current. When the
    # history is explicit, rectangular and square scans are both unambiguous.
    if scan_history is not None:
        history = max(1, int(scan_history))
        if scan_values % history != 0:
            raise ValueError(
                f"Cannot split {scan_values} scan values across history={history}"
            )
        return obs[:, : 9 + scan_values // history]

    # Legacy callers without shape metadata can only be inferred safely for
    # square scans.
    histories = (8, 7, 6, 5, 4, 3, 2, 1)
    for history in histories:
        if scan_values % history == 0:
            per_frame = scan_values // history
            side = int(round(per_frame ** 0.5))
            if side * side == per_frame:
                return obs[:, : 9 + per_frame]
    raise ValueError(f"Cannot infer Unitree scan history layout from obs_dim={obs.shape[1]}")
