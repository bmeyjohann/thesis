"""Regression checks for the human trainer's selected alpha/replay settings."""
import copy
import json
import math
import sys

import pytest
import torch

from eval_interactive_unitree_nav import parse_args
from unitree_nav_online_human import UnitreeOnlineHumanLearner


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["online-test"])
    a = parse_args()
    a.device = "cpu"
    a.num_envs = 1
    a.hidden_dim = 16
    a.policy_encoder = "mlp"
    a.use_layer_norm = False
    a.scan_history = 1
    a.action_history = 0
    a.model_path = ""
    a.online_output_dir = str(tmp_path)
    a.online_run_name = "test"
    a.online_wandb_mode = "disabled"
    a.online_objective_mode = "pref_rl"
    a.online_replay_capacity = 16
    a.online_pref_replay_capacity = 16
    a.online_learning_starts = 1
    a.online_batch_size = 2
    a.online_pref_batch_size = 7
    a.online_n_step = 1
    a.online_policy_frequency = 1
    a.online_log_interval = 1
    a.online_checkpoint_interval = 1000
    return a


def observe(learner, human, *, same_action=False):
    obs = torch.arange(5, dtype=torch.float32).reshape(1, 5)
    learner.observe(obs=obs, student_action=torch.zeros(1, 3),
                    executed_action=torch.tensor([[.2, .1, 0.]]) if not same_action else torch.zeros(1, 3),
                    next_obs=obs + .1, done=False, terminal_success=False,
                    intervened=human, intervention_start=human,
                    goal_distance=2., next_goal_distance=1.9, cost=0.)
    rows = learner.metrics_path.read_text().splitlines()
    return json.loads(rows[-1])


def test_parser_selected_profile(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["online-test"])
    a = parse_args()
    assert (a.online_alpha, a.online_alpha_min, a.online_alpha_max) == (.01, 1e-5, .1)
    assert a.online_alpha_lr == 3e-4 and a.online_target_entropy == -3
    assert a.online_alpha_state_scope == "student"
    assert a.online_pref_sampling_mode == "separate" and a.online_pref_batch_size == 256
    assert a.online_pref_lambda_init == 1 and a.online_pref_lambda_lr == 0


def test_raw_intervention_pair_is_independent_of_nstep_replay(args):
    args.online_n_step = 3
    l = UnitreeOnlineHumanLearner(args, obs_dim=5, act_dim=3)
    observe(l, True)
    assert l.buffer.size == 0 and l.preference_buffer.size == 1
    pair = l.preference_buffer.sample(11, step=1)
    assert pair["obs"].shape == (11, 5)
    assert torch.equal(pair["obs"], torch.arange(5.).repeat(11, 1))
    assert torch.allclose(pair["teacher_actions"], torch.tensor([.2, .1, 0.]).repeat(11, 1))
    assert not pair["student_actions"].any()
    observe(l, False)
    observe(l, True, same_action=True)
    assert l.preference_buffer.size == 1


def test_alpha_skips_human_batches_and_preferences_keep_full_support(args, monkeypatch):
    l = UnitreeOnlineHumanLearner(args, obs_dim=5, act_dim=3)
    initial = l.sac.log_alpha.detach().clone()
    student = observe(l, False)
    assert student["alpha_rows"] == 2 and student["alpha_updates"] == 1
    assert not torch.equal(initial, l.sac.log_alpha)
    before = l.sac.log_alpha.detach().clone()
    optimizer = copy.deepcopy(l.sac.alpha_optimizer.state_dict())
    sample = l.buffer.sample

    def human_batch(n):
        batch = sample(n)
        batch["teacher_intervened"].fill_(True)
        return batch

    monkeypatch.setattr(l.buffer, "sample", human_batch)
    human = observe(l, True)
    assert human["alpha_rows"] == 0 and human["alpha_updates"] == 0
    assert torch.equal(before, l.sac.log_alpha)
    for key, value in optimizer["state"][0].items():
        assert torch.equal(value, l.sac.alpha_optimizer.state_dict()["state"][0][key])
    assert human["pref_source_rows"] == 7 and human["batch_teacher_fraction"] == 1
    assert human["pref_lambda"] == 1 and human["pref_lambda_delta"] == 0

    def student_batch(n):
        batch = sample(n)
        batch["teacher_intervened"].fill_(False)
        return batch

    monkeypatch.setattr(l.buffer, "sample", student_batch)
    student = observe(l, False)
    assert student["batch_teacher_fraction"] == 0 and student["pref_source_rows"] == 7


def test_empty_pair_store_does_not_fall_back_to_teacher_td(args):
    l = UnitreeOnlineHumanLearner(args, obs_dim=5, act_dim=3)
    row = observe(l, True, same_action=True)
    assert row["batch_teacher_fraction"] == 1 and row["preference_replay_size"] == 0
    assert row["pref_source_rows"] == 0 and row["critic_loss_pref"] == 0


def test_fixed_alpha_and_linked_replay_are_explicit_overrides(args):
    args.online_alpha_min = args.online_alpha_max = .01
    args.online_pref_sampling_mode = "linked"
    l = UnitreeOnlineHumanLearner(args, obs_dim=5, act_dim=3)
    row = observe(l, True)
    assert l.preference_buffer is None and row["pref_source_rows"] == 2
    assert row["alpha_updates"] == 0 and not l.sac.alpha_optimizer.state
    assert l.sac.log_alpha.exp().item() == pytest.approx(.01)


def test_checkpoint_retains_settings_and_learned_alpha(args):
    args.online_alpha_lr = 1e-3
    l = UnitreeOnlineHumanLearner(args, obs_dim=5, act_dim=3)
    observe(l, False)
    checkpoint = l.save()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["args"]["online_alpha_state_scope"] == "student"
    assert saved["args"]["online_pref_sampling_mode"] == "separate"
    assert saved["target_entropy"] == -3
    restored_args = copy.deepcopy(args)
    restored_args.model_path = str(checkpoint)
    restored_args.online_restore_full_state = True
    restored = UnitreeOnlineHumanLearner(restored_args, obs_dim=5, act_dim=3)
    assert torch.equal(restored.sac.log_alpha, l.sac.log_alpha)
    assert restored.sac.alpha_optimizer.param_groups[0]["lr"] == 1e-3
    assert restored.sac.target_entropy == -3 and restored.sac.pref_lambda == 1
    assert restored.buffer.size == 0 and restored.preference_buffer.size == 0


@pytest.mark.parametrize("field,value", [
    ("online_alpha", float("nan")), ("online_alpha_min", .2),
    ("online_alpha_max", .001), ("online_alpha_lr", 0),
    ("online_target_entropy", math.inf), ("online_pref_batch_size", 0),
    ("online_pref_replay_capacity", 0), ("online_alpha_state_scope", "invalid"),
])
def test_invalid_settings_rejected_before_creating_run(args, field, value):
    setattr(args, field, value)
    with pytest.raises(ValueError):
        UnitreeOnlineHumanLearner(args, obs_dim=5, act_dim=3)
    from pathlib import Path
    assert not (Path(args.online_output_dir) / args.online_run_name).exists()
