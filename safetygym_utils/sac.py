from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Ensure local FastSAC package is importable without installation.
_FAST_SAC_PATH = Path(__file__).resolve().parent.parent / "fasttd3" / "fast_sac"
if _FAST_SAC_PATH.exists():
    _fast_sac_path_str = str(_FAST_SAC_PATH)
    if _fast_sac_path_str not in sys.path:
        sys.path.insert(0, _fast_sac_path_str)

from fast_sac import Actor, Critic


@dataclass
class SACTensors:
    actor: Actor
    critic: Critic
    critic_target: Critic
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    alpha_optimizer: torch.optim.Optimizer
    log_alpha: torch.Tensor
    target_entropy: float
    pref_lambda: float = 0.0
    pref_violation_ema: float = 0.0


@dataclass
class SACUpdateMetrics:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    target_q_mean: float
    critic_loss_pref: float = 0.0
    critic_loss_pref_weighted: float = 0.0
    pref_q_delta: float = 0.0
    pref_lambda: float = 0.0
    pref_lambda_delta: float = 0.0
    pref_dual_violation: float = 0.0
    pref_dual_signal: float = 0.0
    pref_violation: float = 0.0
    pref_violation_ema: float = 0.0
    pref_lagrangian_loss: float = 0.0


@dataclass
class QDisagreementMetrics:
    abs_diff_mean: float
    abs_diff_max: float
    q1_mean: float
    q2_mean: float
    q_min_mean: float
    q_max_mean: float


def build_sac(
    *,
    obs_dim: int,
    act_dim: int,
    hidden_actor: int,
    hidden_critic: int,
    init_scale: float,
    lr_actor: float,
    lr_critic: float,
    weight_decay: float,
    num_envs: int,
    device: torch.device,
) -> SACTensors:
    actor = Actor(
        n_obs=obs_dim,
        n_act=act_dim,
        num_envs=num_envs,
        init_scale=init_scale,
        hidden_dim=hidden_actor,
        device=device,
    )
    critic = Critic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=hidden_critic,
        device=device,
    )
    critic_target = Critic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=hidden_critic,
        device=device,
    )
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

    log_alpha = torch.ones(1, requires_grad=True, device=device)
    log_alpha.data.copy_(torch.tensor([np.log(1e-3)], device=device))
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr_critic)
    target_entropy = -float(act_dim)

    return SACTensors(
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        log_alpha=log_alpha,
        target_entropy=target_entropy,
    )


def reset_critic(
    *,
    sac: SACTensors,
    obs_dim: int,
    act_dim: int,
    hidden_critic: int,
    lr_critic: float,
    weight_decay: float,
    device: torch.device,
) -> None:
    """Reinitialize critic, target critic, and critic optimizer state."""
    critic = Critic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=hidden_critic,
        device=device,
    )
    critic_target = Critic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=hidden_critic,
        device=device,
    )
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=lr_critic, weight_decay=weight_decay)
    sac.critic = critic
    sac.critic_target = critic_target
    sac.critic_optimizer = critic_optimizer


def compute_q_disagreement(
    *,
    sac: SACTensors,
    obs: torch.Tensor,
    actions: torch.Tensor,
) -> QDisagreementMetrics:
    """Return scalar critic disagreement statistics for the given batch."""
    with torch.no_grad():
        q1, q2 = sac.critic(obs, actions)
        abs_diff = torch.abs(q1 - q2)
        q_min = torch.min(q1, q2)
        q_max = torch.max(q1, q2)
    return QDisagreementMetrics(
        abs_diff_mean=float(abs_diff.mean().detach().cpu().item()),
        abs_diff_max=float(abs_diff.max().detach().cpu().item()),
        q1_mean=float(q1.mean().detach().cpu().item()),
        q2_mean=float(q2.mean().detach().cpu().item()),
        q_min_mean=float(q_min.mean().detach().cpu().item()),
        q_max_mean=float(q_max.mean().detach().cpu().item()),
    )


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.lerp_(p.data, tau)


def sac_update_step(
    *,
    sac: SACTensors,
    batch,
    gamma: float,
    tau: float,
    max_grad_norm: float,
    pref_batch=None,
    pref_rank_weight: float = 0.0,
    pref_rank_margin: float = 0.1,
    pref_loss_type: str = "margin",
    pref_stopgrad_positive: bool = False,
    pref_lambda_lr: float = 1e-3,
    pref_lambda_max: float = 10.0,
    pref_lambda_ema: float = 0.9,
    pref_violation_clip: float = 10.0,
    pref_violation_target: float = 0.0,
    pref_lagrangian_violation_type: str = "hinge",
) -> SACUpdateMetrics:
    obs = batch["observations"]
    actions = batch["actions"]
    next_obs = batch["next"]["observations"]
    rewards = batch["next"]["rewards"].unsqueeze(-1)
    dones = batch["next"]["dones"].bool().unsqueeze(-1)
    trunc = batch["next"]["truncations"].bool().unsqueeze(-1)
    discount = torch.as_tensor(gamma, device=obs.device, dtype=torch.float32)
    bootstrap = (trunc | ~dones).float()

    with torch.no_grad():
        next_actions, next_log_pi, _ = sac.actor(next_obs)
        q1_next, q2_next = sac.critic_target(next_obs, next_actions)
        min_q_next = torch.min(q1_next, q2_next) - sac.log_alpha.exp() * next_log_pi
        target_q = rewards + bootstrap * discount * min_q_next

    q1, q2 = sac.critic(obs, actions)
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    critic_loss_pref = torch.tensor(0.0, device=obs.device)
    critic_loss_pref_weighted = torch.tensor(0.0, device=obs.device)
    pref_q_delta = 0.0
    pref_lambda_value = float(sac.pref_lambda)
    pref_lambda_delta = 0.0
    pref_dual_violation = float(sac.pref_violation_ema)
    pref_dual_signal = 0.0
    pref_violation = 0.0
    pref_violation_ema_value = float(sac.pref_violation_ema)
    pref_lagrangian_loss = 0.0

    if pref_batch is not None and float(pref_rank_weight) > 0.0:
        pref_obs = pref_batch["obs"]
        pref_teacher = pref_batch["teacher_actions"]
        pref_student = pref_batch["student_actions"]
        tq1, tq2 = sac.critic(pref_obs, pref_teacher)
        sq1, sq2 = sac.critic(pref_obs, pref_student)
        q_teacher = torch.stack((tq1, tq2), dim=0)
        q_student = torch.stack((sq1, sq2), dim=0)
        q_teacher_term = q_teacher.detach() if bool(pref_stopgrad_positive) else q_teacher
        delta = q_teacher_term - q_student
        pref_q_delta = float(delta.detach().mean().cpu().item())
        pref_loss_mode = str(pref_loss_type).strip().lower()
        if pref_loss_mode == "bradley_terry":
            critic_loss_pref = F.softplus(-delta).mean()
            critic_loss_pref_weighted = float(pref_rank_weight) * critic_loss_pref
        elif pref_loss_mode == "lagrangian":
            prev_pref_lambda = float(sac.pref_lambda)
            violation_mode = str(pref_lagrangian_violation_type).strip().lower()
            margin_gap = float(pref_rank_margin) - delta
            if violation_mode == "smooth":
                violation_tensor = F.softplus(margin_gap)
            else:
                violation_tensor = torch.clamp(margin_gap, min=0.0)
            pref_violation = float(violation_tensor.detach().mean().cpu().item())
            if float(pref_violation_clip) > 0.0:
                violation_tensor = torch.clamp(violation_tensor, max=float(pref_violation_clip))
            if float(pref_lambda_ema) > 0.0:
                sac.pref_violation_ema = (
                    float(pref_lambda_ema) * float(sac.pref_violation_ema)
                    + (1.0 - float(pref_lambda_ema)) * float(pref_violation)
                )
                pref_dual_violation = float(sac.pref_violation_ema)
            else:
                pref_dual_violation = float(pref_violation)
            pref_dual_signal = float(pref_dual_violation - float(pref_violation_target))
            if float(pref_lambda_lr) > 0.0:
                sac.pref_lambda = max(0.0, float(sac.pref_lambda) + float(pref_lambda_lr) * pref_dual_signal)
                if float(pref_lambda_max) > 0.0:
                    sac.pref_lambda = min(float(sac.pref_lambda), float(pref_lambda_max))
            pref_lambda_value = float(sac.pref_lambda)
            pref_lambda_delta = float(pref_lambda_value - prev_pref_lambda)
            pref_violation_ema_value = float(sac.pref_violation_ema)
            critic_loss_pref = (pref_lambda_value * violation_tensor).mean()
            critic_loss_pref_weighted = critic_loss_pref
            pref_lagrangian_loss = float(critic_loss_pref_weighted.detach().cpu().item())
        else:
            critic_loss_pref = F.softplus(float(pref_rank_margin) - delta).mean()
            critic_loss_pref_weighted = float(pref_rank_weight) * critic_loss_pref
        critic_loss = critic_loss + critic_loss_pref_weighted

    sac.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(sac.critic.parameters(), max_grad_norm)
    sac.critic_optimizer.step()

    pi_actions, log_pi, _ = sac.actor(obs)
    q1_pi, q2_pi = sac.critic(obs, pi_actions)
    q_pi = torch.min(q1_pi, q2_pi)
    actor_loss = ((sac.log_alpha.exp().detach() * log_pi) - q_pi).mean()

    sac.actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(sac.actor.parameters(), max_grad_norm)
    sac.actor_optimizer.step()

    alpha_loss = -sac.log_alpha.exp() * (log_pi.detach() + sac.target_entropy).mean()
    sac.alpha_optimizer.zero_grad(set_to_none=True)
    alpha_loss.backward()
    sac.alpha_optimizer.step()

    soft_update(sac.critic, sac.critic_target, tau)

    return SACUpdateMetrics(
        critic_loss=float(critic_loss.detach().cpu().item()),
        actor_loss=float(actor_loss.detach().cpu().item()),
        alpha_loss=float(alpha_loss.detach().cpu().item()),
        alpha=float(sac.log_alpha.exp().detach().cpu().item()),
        target_q_mean=float(target_q.detach().mean().cpu().item()),
        critic_loss_pref=float(critic_loss_pref.detach().cpu().item()),
        critic_loss_pref_weighted=float(critic_loss_pref_weighted.detach().cpu().item()),
        pref_q_delta=float(pref_q_delta),
        pref_lambda=float(pref_lambda_value),
        pref_lambda_delta=float(pref_lambda_delta),
        pref_dual_violation=float(pref_dual_violation),
        pref_dual_signal=float(pref_dual_signal),
        pref_violation=float(pref_violation),
        pref_violation_ema=float(pref_violation_ema_value),
        pref_lagrangian_loss=float(pref_lagrangian_loss),
    )
