import torch
import pytest
from tests.test_unitree_preference_learning import learner, batch, update
from unitree_nav_control import project_navigation_action


def pairs(qualified=True):
    return {'obs': torch.randn(32, 5), 'teacher_actions': torch.ones(32, 3) * 0.4,
            'student_actions': torch.zeros(32, 3),
            'qualified_bad': torch.full((32,), qualified, dtype=torch.bool)}


def test_anti_rl_does_not_add_a_critic_preference_loss():
    result = update(learner(), batch(), pref_batch=pairs(), pref_rank_weight=0,
                    actor_anti_weight=0.2, actor_action_transform=project_navigation_action)
    assert result.critic_loss_pref == 0 and result.actor_aux_rows == 32
    assert 0 <= result.actor_anti_loss <= 1


def test_warmup_only_pairs_are_not_negative_labels():
    result = update(learner(), pref_batch=pairs(False), actor_anti_weight=0.2,
                    actor_pair_bc_weight=0.2)
    assert result.actor_aux_rows == 0 and result.actor_anti_loss == 0 and result.actor_pair_bc_loss == 0


def test_matched_bc_plus_anti_both_have_finite_active_losses():
    result = update(learner(), pref_batch=pairs(), actor_anti_weight=0.2,
                    actor_pair_bc_weight=0.2, pref_rank_weight=0)
    assert result.actor_anti_loss > 0 and result.actor_pair_bc_loss > 0
    assert result.critic_loss_pref == 0


def test_repulsion_moves_probability_samples_away_from_rejected_command():
    mean = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    bad = torch.tensor([[0.2, 0.0, 0.0]])
    loss = torch.exp(-(mean - bad).square().sum(-1) / (2 * 0.35 ** 2)).mean()
    loss.backward()
    after = mean.detach() - 0.01 * mean.grad
    assert (after - bad).norm() > (mean.detach() - bad).norm()


def test_anti_without_qualified_labels_fails_instead_of_mislabeling_warmup():
    data = pairs()
    del data['qualified_bad']
    with pytest.raises(ValueError, match='qualified'):
        update(learner(), pref_batch=data, actor_anti_weight=0.2)
