import copy
import math

import pytest
import torch

from safetygym_utils.sac import build_sac, sac_update_step
from train_unitree_nav_thesis import parse_args
from unitree_nav_control import project_navigation_action
from unitree_nav_preference import UnitreePreferenceBuffer, configure_learner_state


def learner():
    torch.manual_seed(4)
    return build_sac(obs_dim=5, act_dim=3, hidden_actor=16, hidden_critic=16,
                     num_critics=2, use_layer_norm=False, layer_norm_eps=1e-5,
                     init_scale=0.01, lr_actor=3e-4, lr_critic=3e-4,
                     weight_decay=0, num_envs=1, device=torch.device('cpu'), alpha_init=0.01)


def batch(mask=(True, False, False, True)):
    count = len(mask)
    return {'observations': torch.randn(count, 5), 'actions': torch.zeros(count, 3),
            'student_actions': torch.zeros(count, 3), 'teacher_intervened': torch.tensor(mask),
            'next': {'observations': torch.randn(count, 5), 'rewards': torch.zeros(count),
                     'dones': torch.zeros(count, dtype=torch.bool),
                     'truncations': torch.zeros(count, dtype=torch.bool)}}


def update(sac, data=None, **kwargs):
    return sac_update_step(sac=sac, batch=batch() if data is None else data,
                           gamma=0.99, tau=0.005, max_grad_norm=10, **kwargs)


def test_independent_replay_fixed_size_alignment_and_wrap():
    buf = UnitreePreferenceBuffer(capacity=3, obs_dim=1, act_dim=1, device='cpu')
    assert buf.sample(8, step=0) is None
    obs = torch.arange(5.).reshape(-1, 1)
    buf.add(obs, obs + 10, obs + 20, step=7)
    assert buf.size == 3 and buf.inserted == 5
    assert set(buf.obs.flatten().tolist()) == {2, 3, 4}
    obs.fill_(-100)
    sample = buf.sample(256, step=10)
    assert sample['obs'].shape == (256, 1)
    assert torch.equal(sample['teacher_actions'], sample['obs'] + 10)
    assert torch.equal(sample['student_actions'], sample['obs'] + 20)
    assert sample['unique_rows'] <= 3 and sample['age_ticks_mean'] == 3
    assert buf.sample_count.sum().item() == 256


def test_separate_preferences_train_with_zero_teacher_td_rows():
    sac = learner()
    preference = {'obs': torch.randn(256, 5), 'teacher_actions': torch.ones(256, 3),
                  'student_actions': torch.zeros(256, 3)}
    result = update(sac, batch((False,) * 4), pref_sampling_mode='separate',
                    pref_batch=preference, pref_rank_weight=1,
                    pref_loss_type='hinge', actor_action_transform=project_navigation_action)
    assert result.batch_teacher_fraction == 0
    assert result.pref_source_rows == 256 and result.pref_linked_rows == 256
    assert math.isfinite(result.critic_loss_pref)


def test_empty_separate_store_never_uses_linked_teacher_rows():
    result = update(learner(), batch((True,) * 4), pref_sampling_mode='separate', pref_rank_weight=1)
    assert result.pref_source_rows == 0 and result.critic_loss_pref == 0


def test_student_temperature_is_conditional_mean():
    sac = learner()
    result = update(sac, alpha_state_scope='student')
    assert result.alpha_rows == 2 and result.alpha_updates == 1
    expected = -0.01 * (-result.policy_entropy_student + sac.target_entropy)
    assert result.alpha_loss == pytest.approx(expected, rel=1e-5)


def test_all_teacher_batch_skips_alpha_including_adam_momentum():
    sac = learner()
    update(sac, batch((False,) * 4), alpha_state_scope='student')
    before = sac.log_alpha.detach().clone()
    state = copy.deepcopy(sac.alpha_optimizer.state_dict())
    result = update(sac, batch((True,) * 4), alpha_state_scope='student')
    assert result.alpha_rows == 0 and result.alpha_updates == 0
    assert torch.equal(before, sac.log_alpha)
    for key, value in state['state'][0].items():
        assert torch.equal(value, sac.alpha_optimizer.state_dict()['state'][0][key])


def test_fixed_alpha_is_exact_and_does_not_step_optimizer():
    sac = learner()
    with torch.no_grad():
        sac.log_alpha.fill_(1)
    result = update(sac, alpha_min=0.01, alpha_max=0.01)
    assert result.alpha == pytest.approx(0.01) and result.alpha_updates == 0
    assert not sac.alpha_optimizer.state


def test_student_scope_requires_actual_execution_labels():
    data = batch()
    del data['teacher_intervened']
    with pytest.raises(ValueError, match='labels'):
        update(learner(), data, alpha_state_scope='student')


def test_actor_target_and_noisy_preferences_stay_in_executed_domain():
    sac = learner()
    recorded = []
    handles = [critic.register_forward_pre_hook(lambda module, args: recorded.append(args[1].detach().clone()))
               for critic in [sac.critic, sac.critic_target]]
    preference = {'obs': torch.randn(8, 5), 'teacher_actions': torch.ones(8, 3) * 2,
                  'student_actions': torch.ones(8, 3) * -2}
    update(sac, pref_batch=preference, pref_rank_weight=1, pref_action_noise_std=0.3,
           actor_action_transform=project_navigation_action, actor_bc_weight=0.2)
    for handle in handles:
        handle.remove()
    assert len(recorded) >= 5
    assert all(torch.all(actions.norm(dim=-1) <= 1.00001) for actions in recorded)


def test_noise_cannot_create_preference_for_identical_original_actions():
    preference = {'obs': torch.randn(8, 5), 'teacher_actions': torch.zeros(8, 3),
                  'student_actions': torch.zeros(8, 3)}
    result = update(learner(), pref_batch=preference, pref_rank_weight=1,
                    pref_action_delta_min=0.05, pref_action_noise_std=1)
    assert result.critic_loss_pref == 0 and result.pref_linked_rows == 0


def test_tolerant_dual_can_decrease_and_fixed_lambda_stays_fixed():
    sac = learner()
    sac.pref_lambda = 1
    preference = {'obs': torch.randn(8, 5), 'teacher_actions': torch.zeros(8, 3),
                  'student_actions': torch.zeros(8, 3)}
    result = update(sac, pref_batch=preference, pref_rank_weight=1,
                    pref_loss_type='lagrangian', pref_rank_margin=0,
                    pref_lambda_ema=0, pref_violation_target=0.02, pref_lambda_lr=0.01)
    assert result.pref_lambda == pytest.approx(0.9998)
    fixed = update(sac, pref_batch=preference, pref_rank_weight=1,
                   pref_loss_type='lagrangian', pref_lambda_lr=0)
    assert fixed.pref_lambda_delta == 0


def test_hinge_and_softplus_are_explicitly_different():
    preference = {'obs': torch.randn(8, 5), 'teacher_actions': torch.zeros(8, 3),
                  'student_actions': torch.zeros(8, 3)}
    hinge = update(learner(), pref_batch=preference, pref_rank_weight=1, pref_loss_type='hinge', pref_rank_margin=0)
    soft = update(learner(), pref_batch=preference, pref_rank_weight=1, pref_loss_type='softplus', pref_rank_margin=0)
    assert hinge.critic_loss_pref == 0
    assert soft.critic_loss_pref == pytest.approx(math.log(2), rel=1e-5)


def test_fixed_warm_start_override_and_adaptive_restore():
    sac = learner()
    args = parse_args(['--init-checkpoint', 'example.pt', '--pref-lambda-init', '1', '--pref-lambda-lr', '0'])
    sac.pref_lambda = 7
    configure_learner_state(sac, args)
    assert sac.pref_lambda == 1
    args.pref_lambda_lr = 0.01
    sac.pref_lambda = 7
    configure_learner_state(sac, args)
    assert sac.pref_lambda == 7
    args.reset_learner_state = True
    configure_learner_state(sac, args)
    assert sac.pref_lambda == 1


@pytest.mark.parametrize('flags', [
    ['--pref-replay-capacity', '0'], ['--pref-batch-size', '0'],
    ['--pref-lambda-init', '11', '--pref-lambda-max', '10'],
    ['--pref-lambda-ema', '1'], ['--alpha-min', '0.1', '--alpha-max', '0.01'],
    ['--target-entropy', 'nan'], ['--pref-lambda-lr', '-1'],
    ['--pref-violation-target', '-0.1'], ['--alpha-max', 'nan'],
])
def test_parser_rejects_invalid_learning_contracts(flags):
    with pytest.raises(SystemExit):
        parse_args(flags)
