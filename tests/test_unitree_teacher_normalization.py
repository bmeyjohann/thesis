import torch

from train_unitree_modality_student import (
    RSL_RL_NORMALIZATION_EPS,
    _rsl_normalization_denominator,
)


def test_rsl_normalizer_adds_epsilon_instead_of_clamping_std():
    std = torch.tensor([0.0, 0.005, 0.5])
    denominator = _rsl_normalization_denominator(std)
    assert RSL_RL_NORMALIZATION_EPS == 0.01
    assert torch.allclose(denominator, torch.tensor([0.01, 0.015, 0.51]))
