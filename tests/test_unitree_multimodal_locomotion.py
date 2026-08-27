import torch

from unitree_multimodal_locomotion import GEOMETRIES, MATERIALS, MODALITIES, MultimodalGruActor, MultimodalGruReconstructionActor, MultimodalNoMemoryActor, balanced_assignments, modality_mask, preprocess_student_input
from train_unitree_shared_multimodal_student import balanced_modality_names, masked_reconstruction_loss


def test_balanced_assignments_cover_cartesian_product():
    count = len(GEOMETRIES) * len(MATERIALS) * len(MODALITIES)
    assert len(set(balanced_assignments(count))) == count


def test_actor_supports_every_single_modality():
    batch = 2
    actor = MultimodalNoMemoryActor(proprio_dim=12, action_dim=29)
    shapes = {
        "height_scan": (batch, 1, 17, 11),
        "depth": (batch, 1, 120, 160),
        "mono_rgb": (batch, 3, 120, 160),
        "stereo_rgb": (batch, 6, 120, 160),
    }
    proprio = torch.randn(batch, 12)
    for name, shape in shapes.items():
        action = actor(proprio, {name: torch.randn(shape)}, modality_mask([name] * batch))
        assert action.shape == (batch, 29)
        assert torch.isfinite(action).all()


def test_actor_supports_different_modalities_within_one_batch():
    names = list(MODALITIES)
    actor = MultimodalNoMemoryActor(proprio_dim=12, action_dim=4)
    observations = {
        "height_scan": torch.randn(4, 1, 11, 17),
        "depth": torch.randn(4, 1, 16, 16),
        "mono_rgb": torch.randn(4, 3, 16, 16),
        "stereo_rgb": torch.randn(4, 6, 16, 16),
    }
    action = actor(torch.randn(4, 12), observations, modality_mask(names))
    assert action.shape == (4, 4)
    action.square().mean().backward()
    assert all(encoder.network[0].weight.grad is not None for encoder in actor.encoders.values())


def test_balanced_modality_names_rotate_without_imbalance():
    assert balanced_modality_names(8, 0).count("height_scan") == 2
    assert balanced_modality_names(4, 499) == list(MODALITIES)
    assert balanced_modality_names(4, 500) == ["depth", "mono_rgb", "stereo_rgb", "height_scan"]


def test_reconstruction_loss_ignores_height_scan_rows():
    prediction = torch.tensor([[100.0], [2.0], [3.0], [4.0]])
    target = torch.zeros_like(prediction)
    availability = modality_mask(list(MODALITIES))
    loss = masked_reconstruction_loss(prediction, target, availability)
    assert torch.isclose(loss, torch.tensor((4.0 + 9.0 + 16.0) / 3.0))


def test_student_preprocessing_replays_checkpoint_normalization():
    actor_obs = torch.cat((torch.full((2, 98), 3.0), torch.full((2, 187), 5.0)), dim=1)
    terrain = torch.full((2, 1, 11, 17), 5.0)
    checkpoint = {"input_normalization": {
        "proprio_mean": torch.ones(1, 98),
        "proprio_denominator": torch.full((1, 98), 2.0),
        "height_scan_mean": torch.ones(1, 187),
        "height_scan_denominator": torch.full((1, 187), 4.0),
    }}
    proprio, normalized_terrain = preprocess_student_input(
        actor_obs, terrain, "height_scan", checkpoint
    )
    assert torch.allclose(proprio, torch.ones_like(proprio))
    assert torch.allclose(normalized_terrain, torch.ones_like(normalized_terrain))


def test_actor_rejects_implicit_missing_observation():
    actor = MultimodalNoMemoryActor(proprio_dim=12, action_dim=29)
    try:
        actor(torch.randn(2, 12), {}, torch.zeros(2, len(MODALITIES)))
    except ValueError:
        return
    raise AssertionError("missing modalities should be explicit")


def test_gru_actor_preserves_batch_hidden_state():
    actor = MultimodalGruActor(proprio_dim=12, action_dim=4, hidden_dim=16)
    availability = modality_mask(["depth", "depth"])
    action, hidden = actor(torch.randn(2, 12), {"depth": torch.randn(2, 1, 16, 16)}, availability)
    assert action.shape == (2, 4)
    assert hidden.shape == (2, 16)
    action_2, hidden_2 = actor(torch.randn(2, 12), {"depth": torch.randn(2, 1, 16, 16)}, availability, hidden)
    assert action_2.shape == action.shape
    assert not torch.equal(hidden, hidden_2)


def test_reconstruction_actor_returns_privileged_target_shape():
    actor = MultimodalGruReconstructionActor(proprio_dim=12, action_dim=4, hidden_dim=16, reconstruction_dim=187)
    availability = modality_mask(["mono_rgb"])
    action, reconstruction, hidden = actor(torch.randn(1, 12), {"mono_rgb": torch.randn(1, 3, 16, 16)}, availability)
    assert action.shape == (1, 4)
    assert reconstruction.shape == (1, 187)
    assert hidden.shape == (1, 16)
