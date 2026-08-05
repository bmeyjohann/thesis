from __future__ import annotations

import torch


def prepare_unitree_actor_obs(
    obs: torch.Tensor,
    *,
    mask_proprioception: bool = False,
    mask_goal_heading: bool = False,
    mask_height_scan: bool = False,
) -> torch.Tensor:
    """Apply policy-facing observation masks without changing checkpoint dimensions."""
    if not mask_proprioception and not mask_goal_heading and not mask_height_scan:
        return obs
    if obs.shape[-1] < 9:
        raise ValueError(f"Expected at least 9 Unitree observation values, got {obs.shape[-1]}")
    obs = obs.clone()
    if mask_proprioception:
        obs[..., :6] = 0.0
    if mask_goal_heading:
        obs[..., 8] = 0.0
    if mask_height_scan and obs.shape[-1] > 9:
        obs[..., 9:] = 0.0
    return obs


class UnitreeScanHistory:
    """Keep current context plus a reset-aware history of flattened height scans."""

    def __init__(self, history: int = 1, action_history: int = 0, action_dim: int = 3):
        self.history = max(1, int(history))
        self.action_history = max(0, int(action_history))
        self.action_dim = int(action_dim)
        self._scans: torch.Tensor | None = None
        self._actions: torch.Tensor | None = None
        self.scan_dim: int | None = None

    def reset(self, obs: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        context, scan = self._split(obs)
        if self._scans is None or self._scans.shape[0] != obs.shape[0]:
            self._scans = scan[:, None, :].repeat(1, self.history, 1)
            self._actions = torch.zeros(
                obs.shape[0], self.action_history, self.action_dim, device=obs.device, dtype=obs.dtype
            )
        elif env_ids is None:
            self._scans[:] = scan[:, None, :]
        else:
            self._scans[env_ids] = scan[env_ids, None, :]
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
        self._scans = torch.roll(self._scans, shifts=1, dims=1)
        self._scans[:, 0, :] = scan
        if self.action_history:
            if action is None:
                raise ValueError("Unitree action history requires the executed action on every step")
            assert self._actions is not None
            self._actions = torch.roll(self._actions, shifts=1, dims=1)
            self._actions[:, 0, :] = action
        if done is not None and done.any():
            self._scans[done] = scan[done, None, :]
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
    histories = (int(scan_history),) if scan_history is not None else (8, 7, 6, 5, 4, 3, 2, 1)
    for history in histories:
        if scan_values % history == 0:
            per_frame = scan_values // history
            side = int(round(per_frame ** 0.5))
            if side * side == per_frame:
                return obs[:, : 9 + per_frame]
    raise ValueError(f"Cannot infer Unitree scan history layout from obs_dim={obs.shape[1]}")
