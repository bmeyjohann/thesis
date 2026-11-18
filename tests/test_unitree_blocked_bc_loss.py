import math
import torch

from tests.test_unitree_preference_learning import learner, batch, update
from unitree_nav_control import project_navigation_action


def test_bc_is_teacher_only_with_preferences_and_rl_retained():
    data = batch()
    data['actions'][:] = .4
    pairs = dict(obs=torch.randn(256, 5), teacher_actions=torch.ones(256, 3)*.4,
                 student_actions=torch.zeros(256, 3))
    sac = learner()
    sac.pref_lambda = 1
    result = update(sac, data, actor_bc_weight=.2, actor_bc_teacher_only=True,
                    pref_sampling_mode='separate', pref_batch=pairs, pref_rank_weight=1,
                    pref_loss_type='lagrangian', pref_lambda_lr=0,
                    alpha_state_scope='student', actor_action_transform=project_navigation_action)
    assert result.actor_bc_rows == 2 and result.actor_loss_bc > 0
    assert result.pref_source_rows == 256 and result.pref_lambda == 1
    assert math.isfinite(result.critic_loss_replay) and result.critic_loss_pref > 0
    assert result.alpha_rows == 2


def test_no_bc_arm_does_not_clone_teacher_or_student_actions():
    result = update(learner(), batch(), actor_bc_weight=0, actor_bc_teacher_only=True)
    assert result.actor_loss_bc == 0 and result.actor_bc_rows == 0


def test_bc_skips_no_teacher_td_batch_without_disabling_preference():
    pairs = dict(obs=torch.randn(256,5), teacher_actions=torch.ones(256,3)*.4,
                 student_actions=torch.zeros(256,3))
    result = update(learner(), batch((False,)*4), actor_bc_weight=.2,
                    actor_bc_teacher_only=True, pref_sampling_mode='separate',
                    pref_batch=pairs, pref_rank_weight=1)
    assert result.actor_bc_rows == 0 and result.actor_loss_bc == 0
    assert result.pref_source_rows == 256
