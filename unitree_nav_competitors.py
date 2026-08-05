"""Method-specific learners for matched Unitree intervention comparisons."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetygym_utils.sac import SafetyActor, SACTensors


def concat_replay_batches(*batches: dict[str, Any]) -> dict[str, Any]:
    batches = tuple(batch for batch in batches if batch)
    if not batches:
        raise ValueError("At least one replay batch is required")
    return {
        "observations": torch.cat([batch["observations"] for batch in batches], dim=0),
        "actions": torch.cat([batch["actions"] for batch in batches], dim=0),
        "student_actions": torch.cat([batch["student_actions"] for batch in batches], dim=0),
        "teacher_intervened": torch.cat([batch["teacher_intervened"] for batch in batches], dim=0),
        "intervention_start": torch.cat([batch["intervention_start"] for batch in batches], dim=0),
        "eil_good": torch.cat([batch["eil_good"] for batch in batches], dim=0),
        "eil_bad": torch.cat([batch["eil_bad"] for batch in batches], dim=0),
        "next": {
            key: torch.cat([batch["next"][key] for batch in batches], dim=0)
            for key in batches[0]["next"]
        },
    }


class UnitreeExpertBuffer:
    def __init__(self, *, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.pos = 0
        self.size = 0

    def add(self, obs: torch.Tensor, actions: torch.Tensor) -> None:
        if obs.numel() == 0:
            return
        n = int(obs.shape[0])
        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        self.obs[idx] = obs.detach()
        self.actions[idx] = actions.detach()
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size <= 0:
            raise RuntimeError("Cannot sample an empty expert buffer")
        idx = torch.randint(0, self.size, (int(batch_size),), device=self.device)
        return self.obs[idx], self.actions[idx]


class HGDaggerActorEnsemble(nn.Module):
    """Deterministic actor ensemble matching the OGBench HG-DAgger baseline."""

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        num_envs: int,
        hidden_dim: int,
        ensemble_size: int,
        use_layer_norm: bool,
        policy_encoder: str,
        scan_history: int,
        action_history: int,
        device: torch.device,
    ):
        super().__init__()
        temporal_encoder = "unitree_scan_cnn" if policy_encoder == "scan_cnn" else "none"
        self.members = nn.ModuleList(
            [
                SafetyActor(
                    n_obs=obs_dim,
                    n_act=act_dim,
                    num_envs=num_envs,
                    init_scale=0.01,
                    hidden_dim=hidden_dim,
                    use_layer_norm=use_layer_norm,
                    temporal_encoder=temporal_encoder,
                    obs_frame_stack=scan_history,
                    unitree_action_history=action_history,
                    device=device,
                )
                for _ in range(max(1, int(ensemble_size)))
            ]
        )

    def member_means(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(obs)[2] for member in self.members], dim=0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.member_means(obs).mean(dim=0)
        log_prob = torch.zeros((obs.shape[0], 1), dtype=obs.dtype, device=obs.device)
        return mean, log_prob, mean


@dataclass
class HGDaggerState:
    actor: HGDaggerActorEnsemble
    optimizer: torch.optim.Optimizer


def build_hg_dagger(
    *,
    obs_dim: int,
    act_dim: int,
    num_envs: int,
    hidden_dim: int,
    ensemble_size: int,
    use_layer_norm: bool,
    policy_encoder: str,
    scan_history: int,
    action_history: int,
    lr: float,
    device: torch.device,
) -> HGDaggerState:
    actor = HGDaggerActorEnsemble(
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_envs=num_envs,
        hidden_dim=hidden_dim,
        ensemble_size=ensemble_size,
        use_layer_norm=use_layer_norm,
        policy_encoder=policy_encoder,
        scan_history=scan_history,
        action_history=action_history,
        device=device,
    )
    return HGDaggerState(actor=actor, optimizer=torch.optim.AdamW(actor.parameters(), lr=float(lr), weight_decay=1e-5))


def hg_dagger_update(
    state: HGDaggerState,
    expert_buffer: UnitreeExpertBuffer,
    *,
    batch_size: int,
    max_grad_norm: float,
) -> dict[str, float]:
    # Independent bootstrap batches keep warm-started ensemble members from
    # collapsing to the same deterministic policy.
    member_losses = []
    for member in state.actor.members:
        obs, actions = expert_buffer.sample(batch_size)
        prediction = member(obs)[2]
        member_losses.append(F.mse_loss(prediction, actions))
    loss = torch.stack(member_losses).mean()
    state.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(state.actor.parameters(), float(max_grad_norm))
    state.optimizer.step()
    with torch.no_grad():
        doubt_obs, _ = expert_buffer.sample(batch_size)
        member_means = state.actor.member_means(doubt_obs)
        doubt = torch.linalg.vector_norm(torch.var(member_means, dim=0, unbiased=False), dim=-1).mean()
    return {
        "actor_loss": float(loss.detach().cpu().item()),
        "actor_loss_bc": float(loss.detach().cpu().item()),
        "hg_doubt_mean": float(doubt.detach().cpu().item()),
        "hg_expert_buffer_size": float(expert_buffer.size),
    }


@dataclass
class PVPState:
    actor_target: SafetyActor
    update_index: int = 0


def build_pvp_state(sac: SACTensors) -> PVPState:
    actor_target = copy.deepcopy(sac.actor)
    actor_target.eval()
    return PVPState(actor_target=actor_target)


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.mul_(1.0 - float(tau)).add_(source_param, alpha=float(tau))


def pvp_update_step(
    *,
    sac: SACTensors,
    state: PVPState,
    batch: dict[str, Any],
    gamma: float,
    tau: float,
    max_grad_norm: float,
    proxy_value_bound: float,
    cql_coefficient: float,
    policy_delay: int,
    target_policy_noise: float,
    target_noise_clip: float,
    include_env_reward: bool,
    stop_td_on_intervention_start: bool,
) -> dict[str, float]:
    obs = batch["observations"]
    next_obs = batch["next"]["observations"]
    behavior_actions = batch["actions"]
    novice_actions = batch["student_actions"]
    interventions = batch["teacher_intervened"].float().reshape(-1, 1)
    starts = batch["intervention_start"].float().reshape(-1, 1)
    rewards = batch["next"]["rewards"].reshape(-1, 1)
    dones = batch["next"]["dones"].float().reshape(-1, 1)
    n_steps = batch["next"]["effective_n_steps"].reshape(-1, 1)

    with torch.no_grad():
        next_action = state.actor_target(next_obs)[2]
        noise = (torch.randn_like(next_action) * float(target_policy_noise)).clamp(
            -float(target_noise_clip), float(target_noise_clip)
        )
        next_action = (next_action + noise).clamp(-1.0, 1.0)
        target_q = torch.min(torch.stack(sac.critic_target(next_obs, next_action), dim=0), dim=0).values
        td_reward = rewards if include_env_reward else torch.zeros_like(rewards)
        target_q = td_reward + (1.0 - dones) * torch.pow(float(gamma), n_steps) * target_q

    q_behavior = torch.stack(sac.critic(obs, behavior_actions), dim=0)
    q_novice = torch.stack(sac.critic(obs, novice_actions), dim=0)
    td_mask = 1.0 - starts if stop_td_on_intervention_start else torch.ones_like(starts)
    td_loss = ((td_mask.unsqueeze(0) * (q_behavior - target_q.unsqueeze(0))) ** 2).mean()
    positive = float(proxy_value_bound) * torch.ones_like(q_behavior)
    negative = -float(proxy_value_bound) * torch.ones_like(q_novice)
    mask = interventions.unsqueeze(0)
    # Sum the two critic proxy losses, matching the dedicated OGBench PVP
    # implementation rather than averaging away half their configured weight.
    proxy_teacher = float(cql_coefficient) * (mask * (q_behavior - positive).pow(2)).sum(dim=0).mean()
    proxy_student = float(cql_coefficient) * (mask * (q_novice - negative).pow(2)).sum(dim=0).mean()
    critic_loss = td_loss + proxy_teacher + proxy_student
    sac.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    if max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(sac.critic.parameters(), float(max_grad_norm))
    sac.critic_optimizer.step()

    actor_loss_value = 0.0
    actor_updated = state.update_index % max(1, int(policy_delay)) == 0
    if actor_updated:
        actor_actions = sac.actor(obs)[2]
        actor_q = torch.min(torch.stack(sac.critic(obs, actor_actions), dim=0), dim=0).values
        actor_loss = -actor_q.mean()
        sac.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(sac.actor.parameters(), float(max_grad_norm))
        sac.actor_optimizer.step()
        actor_loss_value = float(actor_loss.detach().cpu().item())
        _soft_update(sac.actor, state.actor_target, tau)
        _soft_update(sac.critic, sac.critic_target, tau)
    state.update_index += 1

    return {
        "critic_loss": float(critic_loss.detach().cpu().item()),
        "critic_loss_replay": float(td_loss.detach().cpu().item()),
        "critic_loss_total": float(critic_loss.detach().cpu().item()),
        "actor_loss": actor_loss_value,
        "pvp_proxy_teacher_loss": float(proxy_teacher.detach().cpu().item()),
        "pvp_proxy_student_loss": float(proxy_student.detach().cpu().item()),
        "pvp_intervened_batch_fraction": float(interventions.mean().detach().cpu().item()),
        "target_q_mean": float(target_q.mean().detach().cpu().item()),
        "q_min_data_mean": float(q_behavior.min(dim=0).values.mean().detach().cpu().item()),
        "actor_updates": float(actor_updated),
    }
