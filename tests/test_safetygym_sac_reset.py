import torch

from safetygym_utils.sac import build_sac, compute_q_disagreement, reset_critic


def test_compute_q_disagreement_and_reset_critic():
    torch.manual_seed(0)
    device = torch.device("cpu")
    sac = build_sac(
        obs_dim=5,
        act_dim=2,
        hidden_actor=32,
        hidden_critic=32,
        init_scale=0.01,
        lr_actor=3e-4,
        lr_critic=3e-4,
        weight_decay=0.0,
        num_envs=1,
        device=device,
    )

    obs = torch.randn(4, 5, device=device)
    act = torch.randn(4, 2, device=device)
    qd = compute_q_disagreement(sac=sac, obs=obs, actions=act)
    assert qd.abs_diff_mean >= 0.0
    assert qd.abs_diff_max >= 0.0
    assert torch.isfinite(torch.tensor([qd.q1_mean, qd.q2_mean, qd.q_min_mean, qd.q_max_mean])).all().item()

    actor_before = [p.detach().clone() for p in sac.actor.parameters()]
    critic_before = [p.detach().clone() for p in sac.critic.parameters()]
    reset_critic(
        sac=sac,
        obs_dim=5,
        act_dim=2,
        hidden_critic=32,
        lr_critic=3e-4,
        weight_decay=0.0,
        device=device,
    )

    actor_after = [p.detach() for p in sac.actor.parameters()]
    critic_after = [p.detach() for p in sac.critic.parameters()]
    assert all(torch.allclose(b, a) for b, a in zip(actor_before, actor_after))
    assert any(not torch.allclose(b, a) for b, a in zip(critic_before, critic_after))
