from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -5


def _mlp_hidden_dims(hidden_dim: int) -> tuple[int, int, int]:
    h1 = max(1, int(hidden_dim))
    h2 = max(1, h1 // 2)
    h3 = max(1, h2 // 2)
    return h1, h2, h3


def _linear_block(
    in_dim: int,
    out_dim: int,
    *,
    use_layer_norm: bool,
    layer_norm_eps: float,
    device: torch.device | None,
) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim, device=device)]
    if use_layer_norm:
        layers.append(nn.LayerNorm(out_dim, eps=float(layer_norm_eps), device=device))
    layers.append(nn.ReLU())
    return layers


class SafetyActor(nn.Module):
    def __init__(
        self,
        *,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        use_layer_norm: bool = False,
        layer_norm_eps: float = 1e-5,
        device: torch.device | None = None,
    ):
        super().__init__()
        h1, h2, h3 = _mlp_hidden_dims(hidden_dim)
        layers: list[nn.Module] = []
        layers.extend(
            _linear_block(
                n_obs,
                h1,
                use_layer_norm=use_layer_norm,
                layer_norm_eps=layer_norm_eps,
                device=device,
            )
        )
        layers.extend(
            _linear_block(
                h1,
                h2,
                use_layer_norm=use_layer_norm,
                layer_norm_eps=layer_norm_eps,
                device=device,
            )
        )
        layers.extend(
            _linear_block(
                h2,
                h3,
                use_layer_norm=use_layer_norm,
                layer_norm_eps=layer_norm_eps,
                device=device,
            )
        )
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(h3, n_act, device=device)
        self.fc_logstd = nn.Linear(h3, n_act, device=device)
        nn.init.normal_(self.fc_mu.weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        self.n_envs = int(num_envs)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        latent = normal.rsample()
        action = torch.tanh(latent)
        log_prob = normal.log_prob(latent)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


class _QNetwork(nn.Module):
    def __init__(
        self,
        *,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        use_layer_norm: bool = False,
        layer_norm_eps: float = 1e-5,
        device: torch.device | None = None,
    ):
        super().__init__()
        h1, h2, h3 = _mlp_hidden_dims(hidden_dim)
        layers: list[nn.Module] = []
        layers.extend(
            _linear_block(
                n_obs + n_act,
                h1,
                use_layer_norm=use_layer_norm,
                layer_norm_eps=layer_norm_eps,
                device=device,
            )
        )
        layers.extend(
            _linear_block(
                h1,
                h2,
                use_layer_norm=use_layer_norm,
                layer_norm_eps=layer_norm_eps,
                device=device,
            )
        )
        layers.extend(
            _linear_block(
                h2,
                h3,
                use_layer_norm=use_layer_norm,
                layer_norm_eps=layer_norm_eps,
                device=device,
            )
        )
        layers.append(nn.Linear(h3, 1, device=device))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=1)
        return self.net(x)


class SafetyCritic(nn.Module):
    def __init__(
        self,
        *,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        num_critics: int,
        use_layer_norm: bool = False,
        layer_norm_eps: float = 1e-5,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_critics = max(1, int(num_critics))
        self.qnet1 = _QNetwork(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
            device=device,
        )
        self.qnet2 = (
            _QNetwork(
                n_obs=n_obs,
                n_act=n_act,
                hidden_dim=hidden_dim,
                use_layer_norm=use_layer_norm,
                layer_norm_eps=layer_norm_eps,
                device=device,
            )
            if self.num_critics >= 2
            else None
        )
        self.extra_qnets = nn.ModuleList(
            [
                _QNetwork(
                    n_obs=n_obs,
                    n_act=n_act,
                    hidden_dim=hidden_dim,
                    use_layer_norm=use_layer_norm,
                    layer_norm_eps=layer_norm_eps,
                    device=device,
                )
                for _ in range(max(0, self.num_critics - 2))
            ]
        )

    def _iter_qnets(self) -> list[_QNetwork]:
        qnets: list[_QNetwork] = [self.qnet1]
        if self.qnet2 is not None:
            qnets.append(self.qnet2)
        qnets.extend(list(self.extra_qnets))
        return qnets

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> list[torch.Tensor]:
        return [qnet(obs, actions) for qnet in self._iter_qnets()]


@dataclass
class SACTensors:
    actor: SafetyActor
    critic: SafetyCritic
    critic_target: SafetyCritic
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    alpha_optimizer: torch.optim.Optimizer
    log_alpha: torch.Tensor
    target_entropy: float
    pref_lambda: float = 0.0
    pref_violation_ema: float = 0.0


@dataclass
class SACUpdateMetrics:
    critic_loss: float = 0.0
    critic_loss_replay: float = 0.0
    critic_loss_total: float = 0.0
    actor_loss: float = 0.0
    actor_loss_sac: float = 0.0
    alpha_loss: float = 0.0
    alpha: float = 0.0
    target_q_mean: float = 0.0
    policy_entropy: float = 0.0
    log_pi_mean: float = 0.0
    action_l2: float = 0.0
    q_min_pi_mean: float = 0.0
    q_min_data_mean: float = 0.0
    q_disagreement_data_mean: float = 0.0
    q_disagreement_pi_mean: float = 0.0
    replay_reward_mean: float = 0.0
    replay_reward_abs_mean: float = 0.0
    actor_updates: float = 1.0
    alpha_updates: float = 1.0
    critic_loss_pref: float = 0.0
    critic_loss_pref_weighted: float = 0.0
    pref_q_delta: float = 0.0
    pref_q_teacher_mean: float = 0.0
    pref_q_student_mean: float = 0.0
    pref_action_delta_l2: float = 0.0
    pref_linked_rows: float = 0.0
    pref_lambda: float = 0.0
    pref_lambda_delta: float = 0.0
    pref_dual_violation: float = 0.0
    pref_dual_signal: float = 0.0
    pref_violation: float = 0.0
    pref_violation_ema: float = 0.0
    pref_lagrangian_loss: float = 0.0
    q_min_teacher_action_mean: float = 0.0
    q_min_teacher_action_count: float = 0.0
    q_min_non_teacher_action_mean: float = 0.0
    q_min_non_teacher_action_count: float = 0.0
    batch_teacher_fraction: float = 0.0


@dataclass
class QDisagreementMetrics:
    abs_diff_mean: float
    abs_diff_max: float
    q1_mean: float
    q2_mean: float
    q_min_mean: float
    q_max_mean: float


def _as_q_list(q_out) -> list[torch.Tensor]:
    if isinstance(q_out, list):
        return q_out
    if isinstance(q_out, tuple):
        return list(q_out)
    raise TypeError(f"Unsupported critic output type: {type(q_out)!r}")


def _batch_has_key(batch, key: str) -> bool:
    try:
        return key in batch.keys(include_nested=False)
    except TypeError:
        return key in batch.keys()
    except Exception:
        return key in batch


def build_sac(
    *,
    obs_dim: int,
    act_dim: int,
    hidden_actor: int,
    hidden_critic: int,
    num_critics: int,
    use_layer_norm: bool,
    layer_norm_eps: float,
    init_scale: float,
    lr_actor: float,
    lr_critic: float,
    weight_decay: float,
    num_envs: int,
    device: torch.device,
    alpha_init: float = 1e-3,
) -> SACTensors:
    actor = SafetyActor(
        n_obs=obs_dim,
        n_act=act_dim,
        num_envs=num_envs,
        init_scale=init_scale,
        hidden_dim=hidden_actor,
        use_layer_norm=use_layer_norm,
        layer_norm_eps=layer_norm_eps,
        device=device,
    )
    critic = SafetyCritic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=hidden_critic,
        num_critics=num_critics,
        use_layer_norm=use_layer_norm,
        layer_norm_eps=layer_norm_eps,
        device=device,
    )
    critic_target = SafetyCritic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=hidden_critic,
        num_critics=num_critics,
        use_layer_norm=use_layer_norm,
        layer_norm_eps=layer_norm_eps,
        device=device,
    )
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

    log_alpha = torch.ones(1, requires_grad=True, device=device)
    log_alpha.data.copy_(torch.tensor([np.log(max(1e-6, float(alpha_init)))], device=device))
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
    num_critics: int,
    use_layer_norm: bool,
    layer_norm_eps: float,
    lr_critic: float,
    weight_decay: float,
    device: torch.device,
) -> None:
    """Reinitialize critic, target critic, and critic optimizer state."""
    critic = SafetyCritic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=hidden_critic,
        num_critics=num_critics,
        use_layer_norm=use_layer_norm,
        layer_norm_eps=layer_norm_eps,
        device=device,
    )
    critic_target = SafetyCritic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=hidden_critic,
        num_critics=num_critics,
        use_layer_norm=use_layer_norm,
        layer_norm_eps=layer_norm_eps,
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
        q_list = _as_q_list(sac.critic(obs, actions))
        stacked = torch.stack(q_list, dim=0)
        q_min = torch.min(stacked, dim=0).values
        q_max = torch.max(stacked, dim=0).values
        abs_diff = q_max - q_min
        q1 = q_list[0]
        q2 = q_list[1] if len(q_list) > 1 else q_list[0]
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
    obs_preprocess=None,
    pref_batch=None,
    pref_sampling_mode: str = "separate",
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
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    scale_actor_to_env_bounds: bool = False,
    action_low: Optional[torch.Tensor] = None,
    action_high: Optional[torch.Tensor] = None,
    update_actor: bool = True,
    critic_loss_reduction: str = "mean",
) -> SACUpdateMetrics:
    obs = batch["observations"]
    actions = batch["actions"]
    next_obs = batch["next"]["observations"]
    if obs_preprocess is not None:
        obs = obs_preprocess(obs)
        next_obs = obs_preprocess(next_obs)
    rewards = batch["next"]["rewards"].unsqueeze(-1)
    dones = batch["next"]["dones"].bool().unsqueeze(-1)
    trunc = batch["next"]["truncations"].bool().unsqueeze(-1)
    discount = torch.as_tensor(gamma, device=obs.device, dtype=torch.float32)
    bootstrap = (trunc | ~dones).float()

    def _scale_actions(actions: torch.Tensor) -> torch.Tensor:
        if not bool(scale_actor_to_env_bounds):
            return actions
        if action_low is None or action_high is None:
            raise ValueError("action bounds are required when scale_actor_to_env_bounds is enabled")
        center = 0.5 * (action_high + action_low)
        half = 0.5 * (action_high - action_low)
        return center + actions * half

    with torch.no_grad():
        next_actions_norm, next_log_pi, _ = sac.actor(next_obs)
        next_actions = _scale_actions(next_actions_norm)
        q_next_list = _as_q_list(sac.critic_target(next_obs, next_actions))
        min_q_next = torch.min(torch.stack(q_next_list, dim=0), dim=0).values - sac.log_alpha.exp() * next_log_pi
        target_q = rewards + bootstrap * discount * min_q_next

    q_list = _as_q_list(sac.critic(obs, actions))
    q_stack_data = torch.stack(q_list, dim=0)
    q_min_data = torch.min(q_stack_data, dim=0).values
    q_max_data = torch.max(q_stack_data, dim=0).values
    q_disagreement_data = q_max_data - q_min_data
    q_losses = torch.stack([F.mse_loss(q, target_q) for q in q_list], dim=0)
    critic_loss_replay_value = float(q_losses.detach().mean().cpu().item())
    if str(critic_loss_reduction).strip().lower() == "sum":
        critic_loss = q_losses.sum()
    else:
        critic_loss = q_losses.mean()
    critic_loss_pref = torch.tensor(0.0, device=obs.device)
    critic_loss_pref_weighted = torch.tensor(0.0, device=obs.device)
    pref_q_delta = 0.0
    pref_q_teacher_mean = 0.0
    pref_q_student_mean = 0.0
    pref_action_delta_l2 = 0.0
    pref_linked_rows = 0.0
    pref_lambda_value = float(sac.pref_lambda)
    pref_lambda_delta = 0.0
    pref_dual_violation = float(sac.pref_violation_ema)
    pref_dual_signal = 0.0
    pref_violation = 0.0
    pref_violation_ema_value = float(sac.pref_violation_ema)
    pref_lagrangian_loss = 0.0
    q_min_teacher_action_mean = 0.0
    q_min_teacher_action_count = 0.0
    q_min_non_teacher_action_mean = 0.0
    q_min_non_teacher_action_count = 0.0
    batch_teacher_fraction = 0.0

    teacher_mask = None
    pref_sampling_mode_value = str(pref_sampling_mode).strip().lower()
    if _batch_has_key(batch, "teacher_intervened"):
        teacher_mask = batch["teacher_intervened"].to(torch.bool).reshape(-1)
        batch_teacher_fraction = float(teacher_mask.float().mean().detach().cpu().item())
        q_min_data_flat = q_min_data.reshape(-1)
        if bool(teacher_mask.any().item()):
            q_min_teacher_action_mean = float(q_min_data_flat[teacher_mask].detach().mean().cpu().item())
            q_min_teacher_action_count = float(int(teacher_mask.sum().item()))
        non_teacher_mask = ~teacher_mask
        if bool(non_teacher_mask.any().item()):
            q_min_non_teacher_action_mean = float(q_min_data_flat[non_teacher_mask].detach().mean().cpu().item())
            q_min_non_teacher_action_count = float(int(non_teacher_mask.sum().item()))

    if pref_batch is not None and float(pref_rank_weight) > 0.0:
        pref_obs = pref_batch["obs"]
        pref_teacher = pref_batch["teacher_actions"]
        pref_student = pref_batch["student_actions"]
    elif float(pref_rank_weight) > 0.0 and pref_sampling_mode_value == "linked":
        pref_obs = None
        pref_teacher = None
        pref_student = None
        if teacher_mask is not None and _batch_has_key(batch, "student_actions"):
            linked_rows = int(teacher_mask.sum().item())
            pref_linked_rows = float(linked_rows)
            if linked_rows > 0:
                pref_obs = obs[teacher_mask]
                pref_teacher = actions[teacher_mask]
                pref_student = batch["student_actions"][teacher_mask].to(device=obs.device, dtype=torch.float32)
    else:
        pref_obs = None
        pref_teacher = None
        pref_student = None

    if (
        float(pref_rank_weight) > 0.0
        and pref_obs is not None
        and pref_teacher is not None
        and pref_student is not None
        and pref_obs.shape[0] > 0
    ):
        if obs_preprocess is not None and pref_obs is not obs:
            pref_obs = obs_preprocess(pref_obs)
        q_teacher = torch.stack(sac.critic(pref_obs, pref_teacher), dim=0)
        q_student = torch.stack(sac.critic(pref_obs, pref_student), dim=0)
        pref_q_teacher_mean = float(q_teacher.detach().mean().cpu().item())
        pref_q_student_mean = float(q_student.detach().mean().cpu().item())
        pref_action_delta_l2 = float((pref_teacher - pref_student).detach().norm(dim=-1).mean().cpu().item())
        pref_linked_rows = float(pref_obs.shape[0])
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

    pi_actions_norm, log_pi, _ = sac.actor(obs)
    pi_actions = _scale_actions(pi_actions_norm)
    q_pi_list = _as_q_list(sac.critic(obs, pi_actions))
    q_stack_pi = torch.stack(q_pi_list, dim=0)
    q_pi = torch.min(q_stack_pi, dim=0).values
    q_disagreement_pi = torch.max(q_stack_pi, dim=0).values - torch.min(q_stack_pi, dim=0).values

    actor_loss = torch.tensor(0.0, device=obs.device)
    alpha_loss = torch.tensor(0.0, device=obs.device)
    actor_updates = 0.0
    alpha_updates = 0.0

    if bool(update_actor):
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
        with torch.no_grad():
            min_log_alpha = None
            if float(alpha_min) > 0.0:
                min_log_alpha = float(np.log(max(1e-6, float(alpha_min))))
            max_log_alpha = None
            if float(alpha_max) > 0.0:
                max_log_alpha = float(np.log(float(alpha_max)))
            lower = min_log_alpha if min_log_alpha is not None else -torch.inf
            upper = max_log_alpha if max_log_alpha is not None else torch.inf
            sac.log_alpha.clamp_(min=lower, max=upper)
        actor_updates = 1.0
        alpha_updates = 1.0

    soft_update(sac.critic, sac.critic_target, tau)
    actor_loss_sac_value = float(actor_loss.detach().cpu().item())

    return SACUpdateMetrics(
        critic_loss=float(critic_loss.detach().cpu().item()),
        critic_loss_replay=float(critic_loss_replay_value),
        critic_loss_total=float(critic_loss_replay_value + float(critic_loss_pref_weighted.detach().cpu().item())),
        actor_loss=float(actor_loss.detach().cpu().item()),
        actor_loss_sac=float(actor_loss_sac_value),
        alpha_loss=float(alpha_loss.detach().cpu().item()),
        alpha=float(sac.log_alpha.exp().detach().cpu().item()),
        target_q_mean=float(target_q.detach().mean().cpu().item()),
        policy_entropy=float((-log_pi).detach().mean().cpu().item()),
        log_pi_mean=float(log_pi.detach().mean().cpu().item()),
        action_l2=float(pi_actions.detach().norm(dim=-1).mean().cpu().item()),
        q_min_pi_mean=float(q_pi.detach().mean().cpu().item()),
        q_min_data_mean=float(q_min_data.detach().mean().cpu().item()),
        q_disagreement_data_mean=float(q_disagreement_data.detach().mean().cpu().item()),
        q_disagreement_pi_mean=float(q_disagreement_pi.detach().mean().cpu().item()),
        replay_reward_mean=float(rewards.detach().mean().cpu().item()),
        replay_reward_abs_mean=float(rewards.detach().abs().mean().cpu().item()),
        actor_updates=float(actor_updates),
        alpha_updates=float(alpha_updates),
        critic_loss_pref=float(critic_loss_pref.detach().cpu().item()),
        critic_loss_pref_weighted=float(critic_loss_pref_weighted.detach().cpu().item()),
        pref_q_delta=float(pref_q_delta),
        pref_q_teacher_mean=float(pref_q_teacher_mean),
        pref_q_student_mean=float(pref_q_student_mean),
        pref_action_delta_l2=float(pref_action_delta_l2),
        pref_linked_rows=float(pref_linked_rows),
        pref_lambda=float(pref_lambda_value),
        pref_lambda_delta=float(pref_lambda_delta),
        pref_dual_violation=float(pref_dual_violation),
        pref_dual_signal=float(pref_dual_signal),
        pref_violation=float(pref_violation),
        pref_violation_ema=float(pref_violation_ema_value),
        pref_lagrangian_loss=float(pref_lagrangian_loss),
        q_min_teacher_action_mean=float(q_min_teacher_action_mean),
        q_min_teacher_action_count=float(q_min_teacher_action_count),
        q_min_non_teacher_action_mean=float(q_min_non_teacher_action_mean),
        q_min_non_teacher_action_count=float(q_min_non_teacher_action_count),
        batch_teacher_fraction=float(batch_teacher_fraction),
    )
