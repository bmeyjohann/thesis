from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

NormalizeFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class CounterfactualSample:
    states: torch.Tensor
    actions: torch.Tensor


class CounterfactualBuffer:
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
        self.actions = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    def append(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        if states is None or actions is None:
            return
        b = int(states.shape[0])
        if b <= 0:
            return
        if b > self.capacity:
            states = states[-self.capacity :]
            actions = actions[-self.capacity :]
            b = self.capacity
        end = self.ptr + b
        if end <= self.capacity:
            self.states[self.ptr:end].copy_(states)
            self.actions[self.ptr:end].copy_(actions)
        else:
            first = self.capacity - self.ptr
            self.states[self.ptr:].copy_(states[:first])
            self.actions[self.ptr:].copy_(actions[:first])
            remain = b - first
            self.states[:remain].copy_(states[first:])
            self.actions[:remain].copy_(actions[first:])
        self.ptr = (self.ptr + b) % self.capacity
        self.size = min(self.size + b, self.capacity)

    def sample(self, batch_size: int) -> CounterfactualSample | None:
        if self.size <= 0:
            return None
        idx = torch.randint(0, self.size, (int(batch_size),), device=self.device)
        states = self.states[idx]
        actions = self.actions[idx]
        if self.normalize_fn is not None:
            states = self.normalize_fn(states)
        return CounterfactualSample(states=states, actions=actions)


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


class PreferenceTDBuffer:
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

        self.teacher_states = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.teacher_actions = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.teacher_rewards = torch.empty((self.capacity, 1), dtype=torch.float32, device=device)
        self.teacher_next_states = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.teacher_dones = torch.empty((self.capacity, 1), dtype=torch.float32, device=device)
        self.teacher_ptr = 0
        self.teacher_size = 0

        self.student_states = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.student_actions = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.student_rewards = torch.empty((self.capacity, 1), dtype=torch.float32, device=device)
        self.student_ptr = 0
        self.student_size = 0

    def append_teacher(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self._append_ring(
            states,
            actions,
            rewards.view(-1, 1),
            buffer_states=self.teacher_states,
            buffer_actions=self.teacher_actions,
            buffer_rewards=self.teacher_rewards,
            buffer_next_states=self.teacher_next_states,
            buffer_dones=self.teacher_dones,
            ptr_attr="teacher_ptr",
            size_attr="teacher_size",
            next_states_src=next_states,
            dones_src=dones.view(-1, 1),
        )

    def append_student(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> None:
        self._append_ring(
            states,
            actions,
            rewards.view(-1, 1),
            buffer_states=self.student_states,
            buffer_actions=self.student_actions,
            buffer_rewards=self.student_rewards,
            ptr_attr="student_ptr",
            size_attr="student_size",
        )

    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        if self.teacher_size <= 0 or self.student_size <= 0:
            return None
        b_teacher = max(1, int(batch_size // 2))
        b_student = max(1, batch_size - b_teacher)
        teacher_idx = torch.randint(0, self.teacher_size, (b_teacher,), device=self.device)
        student_idx = torch.randint(0, self.student_size, (b_student,), device=self.device)
        teacher_states = self.teacher_states[teacher_idx]
        teacher_next_states = self.teacher_next_states[teacher_idx]
        student_states = self.student_states[student_idx]

        if self.normalize_fn is not None:
            teacher_states = self.normalize_fn(teacher_states)
            teacher_next_states = self.normalize_fn(teacher_next_states)
            student_states = self.normalize_fn(student_states)

        return {
            "t_states": teacher_states,
            "t_actions": self.teacher_actions[teacher_idx],
            "t_rewards": self.teacher_rewards[teacher_idx],
            "t_next_states": teacher_next_states,
            "t_dones": self.teacher_dones[teacher_idx],
            "s_states": student_states,
            "s_actions": self.student_actions[student_idx],
            "s_rewards": self.student_rewards[student_idx],
        }

    def _append_ring(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        *,
        buffer_states: torch.Tensor,
        buffer_actions: torch.Tensor,
        buffer_rewards: torch.Tensor,
        ptr_attr: str,
        size_attr: str,
        buffer_next_states: torch.Tensor | None = None,
        next_states_src: torch.Tensor | None = None,
        buffer_dones: torch.Tensor | None = None,
        dones_src: torch.Tensor | None = None,
    ) -> None:
        if states is None or actions is None:
            return
        b = int(states.shape[0])
        if b <= 0:
            return
        ptr = getattr(self, ptr_attr)
        size = getattr(self, size_attr)
        capacity = self.capacity
        if b > capacity:
            states = states[-capacity:]
            actions = actions[-capacity:]
            rewards = rewards[-capacity:]
            if next_states_src is not None:
                next_states_src = next_states_src[-capacity:]
            if dones_src is not None:
                dones_src = dones_src[-capacity:]
            b = capacity
        end = ptr + b
        if end <= capacity:
            buffer_states[ptr:end].copy_(states)
            buffer_actions[ptr:end].copy_(actions)
            buffer_rewards[ptr:end].copy_(rewards)
            if buffer_next_states is not None and next_states_src is not None:
                buffer_next_states[ptr:end].copy_(next_states_src)
            if buffer_dones is not None and dones_src is not None:
                buffer_dones[ptr:end].copy_(dones_src)
        else:
            first = capacity - ptr
            buffer_states[ptr:].copy_(states[:first])
            buffer_actions[ptr:].copy_(actions[:first])
            buffer_rewards[ptr:].copy_(rewards[:first])
            if buffer_next_states is not None and next_states_src is not None:
                buffer_next_states[ptr:].copy_(next_states_src[:first])
            if buffer_dones is not None and dones_src is not None:
                buffer_dones[ptr:].copy_(dones_src[:first])
            remain = b - first
            buffer_states[:remain].copy_(states[first:])
            buffer_actions[:remain].copy_(actions[first:])
            buffer_rewards[:remain].copy_(rewards[first:])
            if buffer_next_states is not None and next_states_src is not None:
                buffer_next_states[:remain].copy_(next_states_src[first:])
            if buffer_dones is not None and dones_src is not None:
                buffer_dones[:remain].copy_(dones_src[first:])
        new_ptr = (ptr + b) % capacity
        setattr(self, ptr_attr, new_ptr)
        setattr(self, size_attr, min(size + b, capacity))
