"""No-memory multimodal locomotion policy components and balanced sampling."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import nn

MODALITIES = ("height_scan", "depth", "mono_rgb", "stereo_rgb")
GEOMETRIES = ("flat", "random_rough", "cobblestone", "stairs", "stepping_stones")
MATERIALS = ("rigid", "slippery", "sand_drag")


def _fuse_modalities(
    encoders: nn.ModuleDict,
    observations: dict[str, torch.Tensor],
    availability: torch.Tensor,
    latent_dim: int,
) -> torch.Tensor:
    """Encode only the rows assigned to each available modality."""
    if availability.ndim != 2 or availability.shape[1] != len(MODALITIES):
        raise ValueError("availability must have shape [batch, num_modalities]")
    if torch.any(availability.sum(dim=1) < 1):
        raise ValueError("Every sample needs at least one terrain modality")
    fused = availability.new_zeros((availability.shape[0], latent_dim))
    for index, name in enumerate(MODALITIES):
        rows = torch.nonzero(availability[:, index] > 0, as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            continue
        if name not in observations:
            raise ValueError(f"Missing observation for available modality: {name}")
        encoded = encoders[name](observations[name].index_select(0, rows))
        weights = availability.index_select(0, rows)[:, index:index + 1]
        fused = fused.index_add(0, rows, encoded * weights)
    return fused / availability.sum(dim=1, keepdim=True).clamp_min(1.0)


class ImageEncoder(nn.Module):
    def __init__(self, channels: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, 16, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim), nn.ELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class MultimodalNoMemoryActor(nn.Module):
    """One actor accepting any explicitly identified non-empty modality subset."""

    def __init__(self, proprio_dim: int, action_dim: int, terrain_latent_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.terrain_latent_dim = terrain_latent_dim
        self.encoders = nn.ModuleDict({
            "height_scan": ImageEncoder(1, terrain_latent_dim),
            "depth": ImageEncoder(1, terrain_latent_dim),
            "mono_rgb": ImageEncoder(3, terrain_latent_dim),
            "stereo_rgb": ImageEncoder(6, terrain_latent_dim),
        })
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + terrain_latent_dim + len(MODALITIES), hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, proprio: torch.Tensor, observations: dict[str, torch.Tensor], availability: torch.Tensor) -> torch.Tensor:
        if availability.shape[0] != proprio.shape[0]:
            raise ValueError("availability must have shape [batch, num_modalities]")
        fused = _fuse_modalities(
            self.encoders, observations, availability, self.terrain_latent_dim
        )
        return self.actor(torch.cat((self.proprio_encoder(proprio), fused, availability), dim=-1))


class MultimodalGruActor(nn.Module):
    """Recurrent modality actor with explicit hidden-state ownership."""

    def __init__(self, proprio_dim: int, action_dim: int, terrain_latent_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.terrain_latent_dim = terrain_latent_dim
        self.encoders = nn.ModuleDict({
            "height_scan": ImageEncoder(1, terrain_latent_dim),
            "depth": ImageEncoder(1, terrain_latent_dim),
            "mono_rgb": ImageEncoder(3, terrain_latent_dim),
            "stereo_rgb": ImageEncoder(6, terrain_latent_dim),
        })
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU()
        )
        self.gru = nn.GRUCell(hidden_dim + terrain_latent_dim + len(MODALITIES), hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def initial_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden_dim), device=device)

    def encode(self, proprio: torch.Tensor, observations: dict[str, torch.Tensor], availability: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if availability.shape[0] != proprio.shape[0]:
            raise ValueError("availability must have shape [batch, num_modalities]")
        fused = _fuse_modalities(
            self.encoders, observations, availability, self.terrain_latent_dim
        )
        if hidden is None:
            hidden = self.initial_hidden(proprio.shape[0], proprio.device)
        recurrent_input = torch.cat((self.proprio_encoder(proprio), fused, availability), dim=-1)
        return self.gru(recurrent_input, hidden)

    def forward(self, proprio: torch.Tensor, observations: dict[str, torch.Tensor], availability: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        next_hidden = self.encode(proprio, observations, availability, hidden)
        return self.actor(next_hidden), next_hidden


class MultimodalGruReconstructionActor(MultimodalGruActor):
    """GRU actor with an auxiliary privileged local-height reconstruction head."""

    def __init__(self, *args, reconstruction_dim: int = 187, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reconstruction_dim = reconstruction_dim
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ELU(),
            nn.Linear(self.hidden_dim, reconstruction_dim),
        )

    def forward(self, proprio: torch.Tensor, observations: dict[str, torch.Tensor], availability: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_hidden = self.encode(proprio, observations, availability, hidden)
        return self.actor(next_hidden), self.reconstruction_head(next_hidden), next_hidden


@dataclass(frozen=True)
class Assignment:
    geometry: str
    material: str
    modality: str


def balanced_assignments(num_envs: int, epoch: int = 0) -> list[Assignment]:
    cells = [Assignment(*values) for values in product(GEOMETRIES, MATERIALS, MODALITIES)]
    offset = (int(epoch) * max(num_envs, 1)) % len(cells)
    return [cells[(offset + index) % len(cells)] for index in range(num_envs)]


def modality_mask(names: list[str], device: torch.device | str = "cpu") -> torch.Tensor:
    mask = torch.zeros((len(names), len(MODALITIES)), device=device)
    for row, name in enumerate(names):
        mask[row, MODALITIES.index(name)] = 1.0
    return mask


def preprocess_student_input(
    actor_obs: torch.Tensor,
    terrain_obs: torch.Tensor,
    modality: str,
    checkpoint: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the fixed expert-derived normalization stored with a student."""
    normalization = checkpoint.get("input_normalization")
    if normalization is None:
        return actor_obs[:, :98], terrain_obs
    proprio_mean = normalization["proprio_mean"].to(actor_obs.device)
    proprio_denominator = normalization["proprio_denominator"].to(actor_obs.device)
    proprio = (actor_obs[:, :98] - proprio_mean) / proprio_denominator
    if modality == "height_scan":
        scan_mean = normalization["height_scan_mean"].to(terrain_obs.device).reshape(
            1, 1, *terrain_obs.shape[-2:]
        )
        scan_denominator = normalization["height_scan_denominator"].to(
            terrain_obs.device
        ).reshape(1, 1, *terrain_obs.shape[-2:])
        terrain_obs = (terrain_obs - scan_mean) / scan_denominator
    return proprio, terrain_obs
