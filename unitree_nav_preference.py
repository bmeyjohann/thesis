"""Independent preference replay and explicit Unitree learner initialization."""
from __future__ import annotations

import math

import torch


class UnitreePreferenceBuffer:
    """Fixed-capacity pair replay, independent of TD/n-step replay routing."""

    def __init__(self, *, capacity: int, obs_dim: int, act_dim: int, device):
        if capacity <= 0:
            raise ValueError("Preference capacity must be positive")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = torch.empty((capacity, obs_dim), device=self.device)
        self.teacher = torch.empty((capacity, act_dim), device=self.device)
        self.student = torch.empty((capacity, act_dim), device=self.device)
        self.qualified_bad = torch.empty(capacity, dtype=torch.bool, device=self.device)
        self.step = torch.empty(capacity, dtype=torch.long, device=self.device)
        self.sample_count = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.pos = self.size = self.inserted = 0

    def add(self, obs, teacher_actions, student_actions, *, step: int, qualified_bad=None):
        if obs.ndim != 2 or teacher_actions.shape != student_actions.shape or obs.shape[0] != teacher_actions.shape[0]:
            raise ValueError("Preference observations and action pairs must align")
        count = int(obs.shape[0])
        if count == 0:
            return
        dropped = max(0, count - self.capacity)
        idx = (torch.arange(count - dropped, device=self.device) + self.pos + dropped) % self.capacity
        self.obs[idx] = obs[dropped:].detach().to(self.device)
        self.teacher[idx] = teacher_actions[dropped:].detach().to(self.device)
        self.student[idx] = student_actions[dropped:].detach().to(self.device)
        self.qualified_bad[idx] = True if qualified_bad is None else qualified_bad[dropped:].detach().to(self.device)
        self.step[idx] = int(step)
        self.sample_count[idx] = 0
        self.pos = (self.pos + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.inserted += count

    def sample(self, batch_size: int, *, step: int):
        if self.size == 0:
            return None
        if batch_size <= 0:
            raise ValueError("Preference batch size must be positive")
        idx = torch.randint(self.size, (int(batch_size),), device=self.device)
        self.sample_count.scatter_add_(0, idx, torch.ones_like(idx))
        return {
            "obs": self.obs[idx], "teacher_actions": self.teacher[idx],
            "student_actions": self.student[idx],
            "qualified_bad": self.qualified_bad[idx],
            "unique_rows": float(idx.unique().numel()),
            "age_ticks_mean": float((int(step) - self.step[idx]).float().mean().item()),
            "reuse_mean": float(self.sample_count[idx].float().mean().item()),
        }


def configure_learner_state(sac, args):
    """Apply explicit overrides after a warm start; never restore replay/optimizers."""
    if args.reset_learner_state:
        with torch.no_grad():
            sac.log_alpha.fill_(math.log(args.alpha_init))
        sac.pref_lambda = 0.0
        sac.pref_violation_ema = 0.0
    if args.pref_lambda_init is not None and (
        not args.init_checkpoint or args.reset_learner_state or args.pref_lambda_lr == 0
    ):
        sac.pref_lambda = float(args.pref_lambda_init)
        sac.pref_violation_ema = 0.0
    if args.pref_lambda_max > 0:
        sac.pref_lambda = min(args.pref_lambda_max, max(0.0, sac.pref_lambda))
    if args.alpha_min > 0 and args.alpha_min == args.alpha_max:
        with torch.no_grad():
            sac.log_alpha.fill_(math.log(args.alpha_min))
    if args.target_entropy is not None:
        sac.target_entropy = float(args.target_entropy)
    if args.alpha_lr is not None:
        for group in sac.alpha_optimizer.param_groups:
            group["lr"] = float(args.alpha_lr)
