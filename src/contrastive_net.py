from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """MLP encoder that maps raw feature vectors to an L2-normalised embedding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ProjectionHead(nn.Module):
    """Non-linear projection head used only during contrastive pretraining (SimCLR style)."""

    def __init__(self, embed_dim: int = 64, proj_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=-1)


class Decoder(nn.Module):
    """Symmetric decoder that reconstructs the input from the embedding."""

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 68,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ContrastiveAutoencoder(nn.Module):
    """Full contrastive autoencoder (encoder, projection head, decoder).

    Training loss = contrastive_weight * NT-Xent + recon_weight * MSE
    Anomaly score = (1 - recon_weight) * centroid_dist + recon_weight * recon_error
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        proj_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, embed_dim, dropout)
        self.projection = ProjectionHead(embed_dim, proj_dim)
        self.decoder = Decoder(embed_dim, hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (embedding h, projection z, reconstruction x_hat)."""
        h = self.encode(x)
        z = self.projection(h)
        x_hat = self.decoder(h)
        return h, z, x_hat


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent (InfoNCE) contrastive loss as used in SimCLR.

    z1, z2 : L2-normalised projections of shape (N, proj_dim).
    Positive pairs are (z1[i], z2[i]); all other cross-pairs are negatives.
    """
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                        # (2N, proj_dim)
    sim = torch.mm(z, z.T) / temperature                  # (2N, 2N)

    # Mask diagonal (self-similarity) to -inf so softmax ignores it.
    eye = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))

    # For row i in [0, N): the positive is at index i+N.
    # For row i in [N, 2N): the positive is at index i-N.
    labels = torch.cat(
        [torch.arange(N, 2 * N), torch.arange(0, N)], dim=0
    ).to(z.device)

    return F.cross_entropy(sim, labels)
