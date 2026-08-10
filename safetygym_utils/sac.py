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


class _TemporalFrameEncoder(nn.Module):
    """Encode a stacked observation as a short frame sequence."""

    def __init__(
        self,
        *,
        n_obs: int,
        num_frames: int,
        action_history: int,
        embed_dim: int,
        use_layer_norm: bool,
        layer_norm_eps: float,
        device: torch.device | None,
    ):
        super().__init__()
        self.num_frames = max(1, int(num_frames))
        if self.num_frames <= 1 or int(n_obs) % self.num_frames != 0:
            raise ValueError(
                f"temporal attention requires stacked observations: n_obs={n_obs}, num_frames={num_frames}"
            )
        self.frame_dim = int(n_obs) // self.num_frames
        self.proj = nn.Linear(self.frame_dim, embed_dim, device=device)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_frames, embed_dim, device=device))
        num_heads = 4 if embed_dim % 4 == 0 else 1
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, device=device)
        self.norm = nn.LayerNorm(embed_dim, eps=float(layer_norm_eps), device=device) if use_layer_norm else nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.reshape(obs.shape[0], self.num_frames, self.frame_dim)
        x = self.proj(x) + self.pos_embed
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn_out)
        return F.relu(x[:, -1, :])


class _UnitreeScanEncoder(nn.Module):
    """Fuse current Unitree context with one or more square height scans."""

    def __init__(
        self,
        *,
        n_obs: int,
        num_frames: int,
        action_history: int,
        embed_dim: int,
        use_layer_norm: bool,
        layer_norm_eps: float,
        device: torch.device | None,
    ):
        super().__init__()
        self.num_frames = max(1, int(num_frames))
        self.action_history = max(0, int(action_history))
        self.context_dim = 9 + 3 * self.action_history
        scan_values = int(n_obs) - self.context_dim
        if scan_values <= 0 or scan_values % self.num_frames != 0:
            raise ValueError(f"unitree_scan_cnn cannot split n_obs={n_obs} into {self.num_frames} scans")
        self.scan_dim = scan_values // self.num_frames
        self.scan_side = int(round(self.scan_dim ** 0.5))
        if self.scan_side * self.scan_side != self.scan_dim:
            raise ValueError(f"unitree_scan_cnn requires square scans, got {self.scan_dim} values")
        scan_channels = max(8, int(embed_dim) // 16)
        self.scan_net = nn.Sequential(
            nn.Conv2d(self.num_frames, scan_channels, kernel_size=3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(scan_channels, scan_channels, kernel_size=3, padding=1, device=device),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(scan_channels * self.scan_side * self.scan_side, int(embed_dim), device=device),
        )
        self.context_proj = nn.Linear(self.context_dim, int(embed_dim), device=device)
        self.norm = (
            nn.LayerNorm(int(embed_dim), eps=float(layer_norm_eps), device=device)
            if use_layer_norm
            else nn.Identity()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        context = self.context_proj(torch.cat((obs[:, :9], obs[:, 9 + self.num_frames * self.scan_dim :]), dim=-1))
        scan = obs[:, 9 : 9 + self.num_frames * self.scan_dim].reshape(
            obs.shape[0], self.num_frames, self.scan_side, self.scan_side
        )
        return F.relu(self.norm(context + self.scan_net(scan)))


def _maybe_temporal_encoder(
    *,
    temporal_encoder: str,
    n_obs: int,
    obs_frame_stack: int,
    unitree_action_history: int,
    embed_dim: int,
    use_layer_norm: bool,
    layer_norm_eps: float,
    device: torch.device | None,
) -> tuple[nn.Module | None, int]:
    encoder = str(temporal_encoder or "none").strip().lower()
    if encoder in {"", "none", "mlp"}:
        return None, int(n_obs)
    if encoder == "unitree_scan_cnn":
        module = _UnitreeScanEncoder(
            n_obs=int(n_obs),
            num_frames=int(obs_frame_stack),
            action_history=int(unitree_action_history),
            embed_dim=int(embed_dim),
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
            device=device,
        )
        return module, int(embed_dim)
    if encoder != "attention":
        raise ValueError(f"Unsupported temporal_encoder: {temporal_encoder!r}")
    module = _TemporalFrameEncoder(
        n_obs=int(n_obs),
        num_frames=int(obs_frame_stack),
        embed_dim=int(embed_dim),
        use_layer_norm=use_layer_norm,
        layer_norm_eps=layer_norm_eps,
        device=device,
    )
    return module, int(embed_dim)


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
        temporal_encoder: str = "none",
        obs_frame_stack: int = 1,
        unitree_action_history: int = 0,
        intervention_aux_head: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()
        h1, h2, h3 = _mlp_hidden_dims(hidden_dim)
        self.obs_encoder, first_in = _maybe_temporal_encoder(
            temporal_encoder=temporal_encoder,
            n_obs=n_obs,
            obs_frame_stack=obs_frame_stack,
            unitree_action_history=unitree_action_history,
            embed_dim=h1,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
            device=device,
        )
        layers: list[nn.Module] = []
        layers.extend(
            _linear_block(
                first_in,
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
        self.intervention_aux_head = (
            nn.Linear(h3, 1, device=device) if bool(intervention_aux_head) else None
        )
        nn.init.normal_(self.fc_mu.weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        self.n_envs = int(num_envs)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.obs_encoder(obs) if self.obs_encoder is not None else obs
        return self.net(x)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._features(obs)
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

    def intervention_logits(self, obs: torch.Tensor) -> torch.Tensor:
        if self.intervention_aux_head is None:
            raise RuntimeError("SafetyActor was created without intervention_aux_head=True")
        return self.intervention_aux_head(self._features(obs)).squeeze(-1)


class _QNetwork(nn.Module):
    def __init__(
        self,
        *,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        use_layer_norm: bool = False,
        layer_norm_eps: float = 1e-5,
        temporal_encoder: str = "none",
        obs_frame_stack: int = 1,
        unitree_action_history: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        h1, h2, h3 = _mlp_hidden_dims(hidden_dim)
        self.obs_encoder, encoded_obs_dim = _maybe_temporal_encoder(
            temporal_encoder=temporal_encoder,
            n_obs=n_obs,
            obs_frame_stack=obs_frame_stack,
            unitree_action_history=unitree_action_history,
            embed_dim=h1,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
            device=device,
        )
        layers: list[nn.Module] = []
        layers.extend(
            _linear_block(
                encoded_obs_dim + n_act,
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
        obs = self.obs_encoder(obs) if self.obs_encoder is not None else obs
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
        temporal_encoder: str = "none",
        obs_frame_stack: int = 1,
        unitree_action_history: int = 0,
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
            temporal_encoder=temporal_encoder,
            obs_frame_stack=obs_frame_stack,
            unitree_action_history=unitree_action_history,
            device=device,
        )
        self.qnet2 = (
            _QNetwork(
                n_obs=n_obs,
                n_act=n_act,
                hidden_dim=hidden_dim,
                use_layer_norm=use_layer_norm,
                layer_norm_eps=layer_norm_eps,
                temporal_encoder=temporal_encoder,
                obs_frame_stack=obs_frame_stack,
                unitree_action_history=unitree_action_history,
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
                    temporal_encoder=temporal_encoder,
                    obs_frame_stack=obs_frame_stack,
                    unitree_action_history=unitree_action_history,
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
    actor_loss_bc: float = 0.0
    actor_loss_ref: float = 0.0
    actor_bc_rows: float = 0.0
    actor_bc_weight_mean: float = 0.0
    actor_bc_only_updates: float = 0.0
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
    pvp_proxy_teacher_loss: float = 0.0
    pvp_proxy_student_loss: float = 0.0
    eil_good_loss: float = 0.0
    eil_bad_loss: float = 0.0
    eil_pair_loss: float = 0.0
    eil_good_batch_fraction: float = 0.0
    eil_bad_batch_fraction: float = 0.0
    pref_q_delta: float = 0.0
    pref_q_teacher_mean: float = 0.0
    pref_q_student_mean: float = 0.0
    pref_action_delta_l2: float = 0.0
    pref_action_delta_kept_fraction: float = 1.0
    pref_action_weight_mean: float = 0.0
    pref_linked_rows: float = 0.0
    pref_augmented_rows: float = 0.0
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
    intervention_aux_loss: float = 0.0
    intervention_aux_acc: float = 0.0
    intervention_aux_label_rate: float = 0.0
    intervention_aux_pred_rate: float = 0.0


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
    temporal_encoder: str = "none",
    obs_frame_stack: int = 1,
    unitree_action_history: int = 0,
    intervention_aux_head: bool = False,
) -> SACTensors:
    actor = SafetyActor(
        n_obs=obs_dim,
        n_act=act_dim,
        num_envs=num_envs,
        init_scale=init_scale,
        hidden_dim=hidden_actor,
        use_layer_norm=use_layer_norm,
        layer_norm_eps=layer_norm_eps,
        temporal_encoder=temporal_encoder,
        obs_frame_stack=obs_frame_stack,
        unitree_action_history=unitree_action_history,
        intervention_aux_head=intervention_aux_head,
        device=device,
    )
    critic = SafetyCritic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=hidden_critic,
        num_critics=num_critics,
        use_layer_norm=use_layer_norm,
        layer_norm_eps=layer_norm_eps,
        temporal_encoder=temporal_encoder,
        obs_frame_stack=obs_frame_stack,
        unitree_action_history=unitree_action_history,
        device=device,
    )
    critic_target = SafetyCritic(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=hidden_critic,
        num_critics=num_critics,
        use_layer_norm=use_layer_norm,
        layer_norm_eps=layer_norm_eps,
        temporal_encoder=temporal_encoder,
        obs_frame_stack=obs_frame_stack,
        unitree_action_history=unitree_action_history,
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
    pref_obs_noise_std: float = 0.0,
    pref_action_noise_std: float = 0.0,
    pref_action_noise_copies: int = 1,
    pref_action_delta_min: float = 0.0,
    pref_action_delta_weight_scale: float = 0.0,
    pref_action_delta_weight_max: float = 10.0,
    algo_variant: str = "plain",
    pvp_proxy_value_bound: float = 1.0,
    eil_threshold: float = 0.0,
    eil_good_margin: float = 0.0,
    eil_bad_margin: float = 0.01,
    eil_pair_margin: float = 0.01,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    scale_actor_to_env_bounds: bool = False,
    action_low: Optional[torch.Tensor] = None,
    action_high: Optional[torch.Tensor] = None,
    update_actor: bool = True,
    critic_loss_reduction: str = "mean",
    critic_td_weight: float = 1.0,
    actor_q_only: bool = False,
    actor_bc_weight: float = 0.0,
    actor_bc_teacher_only: bool = True,
    actor_bc_only: bool = False,
    actor_bc_reward_weight_scale: float = 0.0,
    actor_bc_reward_weight_max: float = 10.0,
    actor_bc_obstacle_lidar_weight_scale: float = 0.0,
    actor_bc_obstacle_lidar_weight_max: float = 10.0,
    actor_bc_goal_block_weight_scale: float = 0.0,
    actor_bc_goal_block_weight_max: float = 10.0,
    actor_reference=None,
    actor_reference_distill_weight: float = 0.0,
    intervention_aux_weight: float = 0.0,
    intervention_aux_pos_weight: float = 0.0,
) -> SACUpdateMetrics:
    obs_raw = batch["observations"]
    obs = obs_raw
    actions = batch["actions"]
    next_obs_raw = batch["next"]["observations"]
    next_obs = next_obs_raw
    if obs_preprocess is not None:
        obs = obs_preprocess(obs)
        next_obs = obs_preprocess(next_obs)
    rewards = batch["next"]["rewards"].unsqueeze(-1)
    dones = batch["next"]["dones"].bool().unsqueeze(-1)
    trunc = batch["next"]["truncations"].bool().unsqueeze(-1)
    effective_n_steps = batch["next"].get("effective_n_steps")
    if effective_n_steps is None:
        discount = torch.as_tensor(gamma, device=obs.device, dtype=torch.float32)
    else:
        discount = torch.pow(
            torch.as_tensor(gamma, device=obs.device, dtype=torch.float32),
            effective_n_steps.to(device=obs.device, dtype=torch.float32).unsqueeze(-1),
        )
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
        critic_loss = float(critic_td_weight) * q_losses.sum()
    else:
        critic_loss = float(critic_td_weight) * q_losses.mean()
    critic_loss_pref = torch.tensor(0.0, device=obs.device)
    critic_loss_pref_weighted = torch.tensor(0.0, device=obs.device)
    pvp_proxy_teacher_loss = torch.tensor(0.0, device=obs.device)
    pvp_proxy_student_loss = torch.tensor(0.0, device=obs.device)
    eil_good_loss = torch.tensor(0.0, device=obs.device)
    eil_bad_loss = torch.tensor(0.0, device=obs.device)
    eil_pair_loss = torch.tensor(0.0, device=obs.device)
    eil_good_batch_fraction = 0.0
    eil_bad_batch_fraction = 0.0
    pref_q_delta = 0.0
    pref_q_teacher_mean = 0.0
    pref_q_student_mean = 0.0
    pref_action_delta_l2 = 0.0
    pref_action_weight_mean = 0.0
    pref_linked_rows = 0.0
    pref_augmented_rows = 0.0
    pref_action_delta_kept_fraction = 1.0
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
                # Linked preference rows must reuse the same raw observations and
                # preprocessing path as the TD batch. Slicing from the already
                # normalized `obs` tensor here would apply obs_preprocess twice.
                pref_obs = obs_raw[teacher_mask]
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
        base_pref_rows = float(pref_obs.shape[0])
        aug_copies = int(max(1, pref_action_noise_copies))
        obs_noise_std = float(max(0.0, pref_obs_noise_std))
        noise_std = float(max(0.0, pref_action_noise_std))
        if aug_copies > 1:
            pref_obs = pref_obs.repeat_interleave(aug_copies, dim=0)
            pref_teacher = pref_teacher.repeat_interleave(aug_copies, dim=0)
            pref_student = pref_student.repeat_interleave(aug_copies, dim=0)
        if obs_noise_std > 0.0:
            # This is applied after obs_preprocess so the scale is normalized-observation units.
            pref_obs = pref_obs + torch.randn_like(pref_obs) * obs_noise_std
        if noise_std > 0.0:
            pref_teacher = pref_teacher + torch.randn_like(pref_teacher) * noise_std
            pref_student = pref_student + torch.randn_like(pref_student) * noise_std
            if action_low is not None and action_high is not None:
                pref_teacher = torch.max(torch.min(pref_teacher, action_high), action_low)
                pref_student = torch.max(torch.min(pref_student, action_high), action_low)
            elif bool(scale_actor_to_env_bounds):
                pref_teacher = pref_teacher.clamp(-1.0, 1.0)
                pref_student = pref_student.clamp(-1.0, 1.0)
        action_delta_l2 = (pref_teacher - pref_student).detach().norm(dim=-1)
        rows_before_delta_filter = int(action_delta_l2.numel())
        min_action_delta = float(max(0.0, pref_action_delta_min))
        if min_action_delta > 0.0:
            keep = action_delta_l2 >= min_action_delta
            kept_rows = int(keep.detach().sum().cpu().item())
            pref_action_delta_kept_fraction = float(kept_rows / max(1, rows_before_delta_filter))
            if bool(keep.any().item()):
                pref_obs = pref_obs[keep]
                pref_teacher = pref_teacher[keep]
                pref_student = pref_student[keep]
                action_delta_l2 = action_delta_l2[keep]
            else:
                pref_obs = pref_obs[:0]
                pref_teacher = pref_teacher[:0]
                pref_student = pref_student[:0]
                action_delta_l2 = action_delta_l2[:0]
        if pref_obs.shape[0] == 0:
            q_teacher = None
            q_student = None
            pref_linked_rows = 0.0
            pref_augmented_rows = 0.0
        else:
            q_teacher = torch.stack(sac.critic(pref_obs, pref_teacher), dim=0)
            q_student = torch.stack(sac.critic(pref_obs, pref_student), dim=0)
            pref_q_teacher_mean = float(q_teacher.detach().mean().cpu().item())
            pref_q_student_mean = float(q_student.detach().mean().cpu().item())
        pref_action_delta_l2 = float(action_delta_l2.mean().cpu().item()) if action_delta_l2.numel() > 0 else 0.0
        pref_weight_tensor = None
        if pref_obs.shape[0] > 0 and float(pref_action_delta_weight_scale) > 0.0:
            pref_weight_tensor = 1.0 + float(pref_action_delta_weight_scale) * action_delta_l2
            if float(pref_action_delta_weight_max) > 0.0:
                pref_weight_tensor = torch.clamp(pref_weight_tensor, max=float(pref_action_delta_weight_max))
            pref_action_weight_mean = float(pref_weight_tensor.mean().cpu().item())
            pref_weight_tensor = pref_weight_tensor.view(1, -1, 1)
        pref_linked_rows = float(pref_obs.shape[0])
        pref_augmented_rows = max(0.0, float(pref_obs.shape[0]) - base_pref_rows)
        if pref_obs.shape[0] > 0:
            q_teacher_term = q_teacher.detach() if bool(pref_stopgrad_positive) else q_teacher
            delta = q_teacher_term - q_student
            pref_q_delta = float(delta.detach().mean().cpu().item())
            pref_loss_mode = str(pref_loss_type).strip().lower()
            if pref_loss_mode == "bradley_terry":
                loss_tensor = F.softplus(-delta)
                if pref_weight_tensor is not None:
                    loss_tensor = loss_tensor * pref_weight_tensor
                critic_loss_pref = loss_tensor.mean()
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
                if pref_weight_tensor is not None:
                    violation_tensor = violation_tensor * pref_weight_tensor
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
                loss_tensor = F.softplus(float(pref_rank_margin) - delta)
                if pref_weight_tensor is not None:
                    loss_tensor = loss_tensor * pref_weight_tensor
                critic_loss_pref = loss_tensor.mean()
                critic_loss_pref_weighted = float(pref_rank_weight) * critic_loss_pref
            critic_loss = critic_loss + critic_loss_pref_weighted

    variant_value = str(algo_variant or "plain").strip().lower()
    if variant_value == "pvp":
        has_student_actions = _batch_has_key(batch, "student_actions")
        student_actions_batch = (
            batch["student_actions"].to(device=obs.device, dtype=torch.float32) if has_student_actions else actions
        )
        if teacher_mask is not None and bool(teacher_mask.any().item()):
            q_student_proxy = torch.stack(sac.critic(obs, student_actions_batch), dim=0)
            teacher_mask_f = teacher_mask.to(device=obs.device, dtype=torch.float32).view(1, -1, 1)
            bound = float(pvp_proxy_value_bound)
            pvp_proxy_teacher_loss = (teacher_mask_f * F.mse_loss(
                q_stack_data,
                bound * torch.ones_like(q_stack_data),
                reduction="none",
            )).mean()
            pvp_proxy_student_loss = (teacher_mask_f * F.mse_loss(
                q_student_proxy,
                -bound * torch.ones_like(q_student_proxy),
                reduction="none",
            )).mean()
            critic_loss = critic_loss + pvp_proxy_teacher_loss + pvp_proxy_student_loss

    elif variant_value == "eil":
        has_student_actions = _batch_has_key(batch, "student_actions")
        student_actions_batch = (
            batch["student_actions"].to(device=obs.device, dtype=torch.float32) if has_student_actions else actions
        )
        if _batch_has_key(batch, "eil_good"):
            eil_good_mask = batch["eil_good"].to(device=obs.device, dtype=torch.bool).reshape(-1)
        else:
            eil_good_mask = torch.zeros(actions.shape[0], dtype=torch.bool, device=obs.device)
        if _batch_has_key(batch, "eil_bad"):
            eil_bad_mask = batch["eil_bad"].to(device=obs.device, dtype=torch.bool).reshape(-1)
        else:
            eil_bad_mask = torch.zeros(actions.shape[0], dtype=torch.bool, device=obs.device)
        teacher_mask_local = (
            teacher_mask
            if teacher_mask is not None
            else torch.zeros(actions.shape[0], dtype=torch.bool, device=obs.device)
        )
        good_mask = eil_good_mask | teacher_mask_local
        bad_exec_mask = eil_bad_mask
        bad_student_mask = teacher_mask_local
        eil_good_batch_fraction = float(good_mask.float().mean().detach().cpu().item())
        eil_bad_batch_fraction = float((bad_exec_mask | bad_student_mask).float().mean().detach().cpu().item())

        q_student_eil = torch.stack(sac.critic(obs, student_actions_batch), dim=0)
        threshold = float(eil_threshold)
        good_target = threshold + float(eil_good_margin)
        bad_target = threshold - float(eil_bad_margin)
        pair_margin = float(eil_pair_margin)
        good_mask_f = good_mask.to(dtype=torch.float32).view(1, -1, 1)
        bad_exec_mask_f = bad_exec_mask.to(dtype=torch.float32).view(1, -1, 1)
        bad_student_mask_f = bad_student_mask.to(dtype=torch.float32).view(1, -1, 1)
        eil_good_loss = (good_mask_f * torch.clamp(good_target - q_stack_data, min=0.0)).mean()
        eil_bad_loss = (
            (bad_exec_mask_f * torch.clamp(q_stack_data - bad_target, min=0.0)).mean()
            + (bad_student_mask_f * torch.clamp(q_student_eil - bad_target, min=0.0)).mean()
        )
        eil_pair_loss = (
            bad_student_mask_f * torch.clamp(float(pair_margin) - (q_stack_data - q_student_eil), min=0.0)
        ).mean()
        critic_loss = critic_loss + eil_good_loss + eil_bad_loss + eil_pair_loss

    sac.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(sac.critic.parameters(), max_grad_norm)
    sac.critic_optimizer.step()

    pi_actions_norm, log_pi, pi_mean_norm = sac.actor(obs)
    pi_actions = _scale_actions(pi_actions_norm)
    pi_mean_actions = _scale_actions(pi_mean_norm)
    q_pi_list = _as_q_list(sac.critic(obs, pi_actions))
    q_stack_pi = torch.stack(q_pi_list, dim=0)
    q_pi = torch.min(q_stack_pi, dim=0).values
    q_disagreement_pi = torch.max(q_stack_pi, dim=0).values - torch.min(q_stack_pi, dim=0).values

    actor_loss = torch.tensor(0.0, device=obs.device)
    actor_loss_sac = torch.tensor(0.0, device=obs.device)
    actor_loss_bc = torch.tensor(0.0, device=obs.device)
    actor_loss_ref = torch.tensor(0.0, device=obs.device)
    actor_bc_rows = 0.0
    actor_bc_weight_mean = 0.0
    alpha_loss = torch.tensor(0.0, device=obs.device)
    intervention_aux_loss = torch.tensor(0.0, device=obs.device)
    intervention_aux_acc = 0.0
    intervention_aux_label_rate = 0.0
    intervention_aux_pred_rate = 0.0
    actor_updates = 0.0
    alpha_updates = 0.0

    if bool(update_actor):
        actor_loss_sac = (-q_pi).mean() if bool(actor_q_only) else ((sac.log_alpha.exp().detach() * log_pi) - q_pi).mean()
        actor_loss = torch.tensor(0.0, device=obs.device) if bool(actor_bc_only) else actor_loss_sac
        effective_actor_bc_weight = float(actor_bc_weight)
        if bool(actor_bc_only) and effective_actor_bc_weight <= 0.0:
            effective_actor_bc_weight = 1.0
        if effective_actor_bc_weight > 0.0:
            bc_mask = None
            if bool(actor_bc_teacher_only):
                bc_mask = teacher_mask
            if bc_mask is not None:
                bc_mask = bc_mask.to(torch.bool).reshape(-1)
                actor_bc_rows = float(int(bc_mask.sum().item()))
            else:
                actor_bc_rows = float(int(actions.shape[0]))
            if bc_mask is None:
                per_row_bc = F.mse_loss(pi_mean_actions, actions.detach(), reduction="none").mean(dim=-1)
            elif bool(bc_mask.any().item()):
                per_row_bc = F.mse_loss(pi_mean_actions[bc_mask], actions.detach()[bc_mask], reduction="none").mean(dim=-1)
            else:
                per_row_bc = None
            if per_row_bc is not None:
                row_weights = torch.ones_like(per_row_bc)
                reward_scale = float(actor_bc_reward_weight_scale)
                if reward_scale > 0.0:
                    reward_rows = rewards.detach().reshape(-1)
                    if bc_mask is not None:
                        reward_rows = reward_rows[bc_mask]
                    row_weights = row_weights + reward_scale * torch.clamp(-reward_rows, min=0.0)
                    max_weight = float(actor_bc_reward_weight_max)
                    if max_weight > 0.0:
                        row_weights = torch.clamp(row_weights, max=max_weight)
                lidar_scale = float(actor_bc_obstacle_lidar_weight_scale)
                if lidar_scale > 0.0 and obs_raw.shape[-1] >= 72:
                    obstacle_lidar = obs_raw.detach()[..., 40:72].reshape(obs_raw.shape[0], -1)
                    if bc_mask is not None:
                        obstacle_lidar = obstacle_lidar[bc_mask]
                    obstacle_intensity = torch.clamp(obstacle_lidar.max(dim=-1).values, min=0.0, max=1.0)
                    row_weights = row_weights + lidar_scale * obstacle_intensity
                    max_weight = float(actor_bc_obstacle_lidar_weight_max)
                    if max_weight > 0.0:
                        row_weights = torch.clamp(row_weights, max=max_weight)
                goal_block_scale = float(actor_bc_goal_block_weight_scale)
                if goal_block_scale > 0.0 and obs_raw.shape[-1] >= 72:
                    goal_lidar = obs_raw.detach()[..., 24:40].reshape(obs_raw.shape[0], 16)
                    hazards_lidar = obs_raw.detach()[..., 40:56].reshape(obs_raw.shape[0], 16)
                    vases_lidar = obs_raw.detach()[..., 56:72].reshape(obs_raw.shape[0], 16)
                    obstacle_lidar_16 = torch.maximum(hazards_lidar, vases_lidar)
                    if bc_mask is not None:
                        goal_lidar = goal_lidar[bc_mask]
                        obstacle_lidar_16 = obstacle_lidar_16[bc_mask]
                    goal_idx = torch.argmax(goal_lidar, dim=-1, keepdim=True)
                    goal_peak = torch.gather(goal_lidar, dim=-1, index=goal_idx).squeeze(-1)
                    obstacle_at_goal = torch.gather(obstacle_lidar_16, dim=-1, index=goal_idx).squeeze(-1)
                    blocked_signal = torch.clamp(obstacle_at_goal - goal_peak, min=0.0, max=1.0)
                    row_weights = row_weights + goal_block_scale * blocked_signal
                    max_weight = float(actor_bc_goal_block_weight_max)
                    if max_weight > 0.0:
                        row_weights = torch.clamp(row_weights, max=max_weight)
                actor_bc_weight_mean = float(row_weights.detach().mean().cpu().item())
                actor_loss_bc = (per_row_bc * row_weights).sum() / torch.clamp(
                    row_weights.sum(), min=torch.finfo(row_weights.dtype).eps
                )
            actor_loss = actor_loss + effective_actor_bc_weight * actor_loss_bc

        if actor_reference is not None and float(actor_reference_distill_weight) > 0.0:
            with torch.no_grad():
                _, _, ref_mean_norm = actor_reference(obs)
                ref_mean_actions = _scale_actions(ref_mean_norm)
            actor_loss_ref = F.mse_loss(pi_mean_actions, ref_mean_actions)
            actor_loss = actor_loss + float(actor_reference_distill_weight) * actor_loss_ref

        if (
            float(intervention_aux_weight) > 0.0
            and teacher_mask is not None
            and hasattr(sac.actor, "intervention_aux_head")
            and getattr(sac.actor, "intervention_aux_head") is not None
        ):
            targets = teacher_mask.to(device=obs.device, dtype=torch.float32).reshape(-1)
            logits = sac.actor.intervention_logits(obs)
            pos_weight = None
            if float(intervention_aux_pos_weight) > 0.0:
                pos_weight = torch.as_tensor(float(intervention_aux_pos_weight), device=obs.device)
            intervention_aux_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            actor_loss = actor_loss + float(intervention_aux_weight) * intervention_aux_loss
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = probs >= 0.5
                labels = targets >= 0.5
                intervention_aux_acc = float((preds == labels).float().mean().detach().cpu().item())
                intervention_aux_label_rate = float(labels.float().mean().detach().cpu().item())
                intervention_aux_pred_rate = float(preds.float().mean().detach().cpu().item())

        if bool(actor_loss.requires_grad):
            sac.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(sac.actor.parameters(), max_grad_norm)
            sac.actor_optimizer.step()
        else:
            actor_updates = 0.0

        if not bool(actor_bc_only) and not bool(actor_q_only):
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
        if bool(actor_loss.requires_grad):
            actor_updates = 1.0
        alpha_updates = 0.0 if bool(actor_bc_only or actor_q_only) else 1.0

    soft_update(sac.critic, sac.critic_target, tau)
    actor_loss_sac_value = float(actor_loss_sac.detach().cpu().item())

    return SACUpdateMetrics(
        critic_loss=float(critic_loss.detach().cpu().item()),
        critic_loss_replay=float(critic_loss_replay_value),
        critic_loss_total=float(
            float(critic_td_weight) * critic_loss_replay_value
            + float(critic_loss_pref_weighted.detach().cpu().item())
            + float(pvp_proxy_teacher_loss.detach().cpu().item())
            + float(pvp_proxy_student_loss.detach().cpu().item())
            + float(eil_good_loss.detach().cpu().item())
            + float(eil_bad_loss.detach().cpu().item())
            + float(eil_pair_loss.detach().cpu().item())
        ),
        actor_loss=float(actor_loss.detach().cpu().item()),
        actor_loss_sac=float(actor_loss_sac_value),
        actor_loss_bc=float(actor_loss_bc.detach().cpu().item()),
        actor_loss_ref=float(actor_loss_ref.detach().cpu().item()),
        actor_bc_rows=float(actor_bc_rows),
        actor_bc_weight_mean=float(actor_bc_weight_mean),
        actor_bc_only_updates=float(1.0 if bool(update_actor and actor_bc_only) else 0.0),
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
        pvp_proxy_teacher_loss=float(pvp_proxy_teacher_loss.detach().cpu().item()),
        pvp_proxy_student_loss=float(pvp_proxy_student_loss.detach().cpu().item()),
        eil_good_loss=float(eil_good_loss.detach().cpu().item()),
        eil_bad_loss=float(eil_bad_loss.detach().cpu().item()),
        eil_pair_loss=float(eil_pair_loss.detach().cpu().item()),
        eil_good_batch_fraction=float(eil_good_batch_fraction),
        eil_bad_batch_fraction=float(eil_bad_batch_fraction),
        pref_q_delta=float(pref_q_delta),
        pref_q_teacher_mean=float(pref_q_teacher_mean),
        pref_q_student_mean=float(pref_q_student_mean),
        pref_action_delta_l2=float(pref_action_delta_l2),
        pref_action_delta_kept_fraction=float(pref_action_delta_kept_fraction),
        pref_action_weight_mean=float(pref_action_weight_mean),
        pref_linked_rows=float(pref_linked_rows),
        pref_augmented_rows=float(pref_augmented_rows),
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
        intervention_aux_loss=float(intervention_aux_loss.detach().cpu().item()),
        intervention_aux_acc=float(intervention_aux_acc),
        intervention_aux_label_rate=float(intervention_aux_label_rate),
        intervention_aux_pred_rate=float(intervention_aux_pred_rate),
    )
