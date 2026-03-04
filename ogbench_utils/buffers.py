from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

NormalizeFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class PreferencePairSample:
    states: torch.Tensor
    teacher_actions: torch.Tensor
    student_actions: torch.Tensor


class PreferencePairBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        normalize_fn: NormalizeFn | None = None,
    ):
        self.capacity = int(max(1, capacity))
        self.device = device
        self.normalize_fn = normalize_fn
        self.states = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.teacher_actions = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.student_actions = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    def append(self, states: torch.Tensor, teacher_actions: torch.Tensor, student_actions: torch.Tensor) -> None:
        if states is None or teacher_actions is None or student_actions is None:
            return
        b = int(states.shape[0])
        if b <= 0:
            return
        if b > self.capacity:
            states = states[-self.capacity :]
            teacher_actions = teacher_actions[-self.capacity :]
            student_actions = student_actions[-self.capacity :]
            b = self.capacity
        end = self.ptr + b
        if end <= self.capacity:
            self.states[self.ptr:end].copy_(states)
            self.teacher_actions[self.ptr:end].copy_(teacher_actions)
            self.student_actions[self.ptr:end].copy_(student_actions)
        else:
            first = self.capacity - self.ptr
            self.states[self.ptr:].copy_(states[:first])
            self.teacher_actions[self.ptr:].copy_(teacher_actions[:first])
            self.student_actions[self.ptr:].copy_(student_actions[:first])
            remain = b - first
            self.states[:remain].copy_(states[first:])
            self.teacher_actions[:remain].copy_(teacher_actions[first:])
            self.student_actions[:remain].copy_(student_actions[first:])
        self.ptr = (self.ptr + b) % self.capacity
        self.size = min(self.size + b, self.capacity)

    def sample(self, batch_size: int) -> PreferencePairSample | None:
        if self.size <= 0:
            return None
        idx = torch.randint(0, self.size, (int(batch_size),), device=self.device)
        states = self.states[idx]
        if self.normalize_fn is not None:
            states = self.normalize_fn(states)
        return PreferencePairSample(
            states=states,
            teacher_actions=self.teacher_actions[idx],
            student_actions=self.student_actions[idx],
        )
