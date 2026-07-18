from __future__ import annotations

import torch


class TabularAugmentation:
    """Three augmentation strategies for tabular network-flow features.

    Each call to the instance returns two independently augmented views of
    the same batch — the paired positive samples required by NT-Xent.

    Strategies
    ----------
    noise     : additive Gaussian noise
    mask      : random feature zeroing (simulates missing / dropped fields)
    jitter    : per-feature random scaling
    combined  : all three applied sequentially (default)
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        mask_prob: float = 0.15,
        jitter_range: tuple[float, float] = (0.8, 1.2),
        strategy: str = "combined",
    ) -> None:
        allowed = {"noise", "mask", "jitter", "combined"}
        if strategy not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got '{strategy}'")

        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.jitter_lo, self.jitter_hi = jitter_range
        self.strategy = strategy

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two augmented views of batch x (each independently augmented)."""
        return self._augment(x), self._augment(x)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()

        if self.strategy in ("noise", "combined"):
            out = out + torch.randn_like(out) * self.noise_std

        if self.strategy in ("mask", "combined"):
            mask = torch.bernoulli(torch.full_like(out, self.mask_prob)).bool()
            out = out.masked_fill(mask, 0.0)

        if self.strategy in ("jitter", "combined"):
            scale = torch.empty_like(out).uniform_(self.jitter_lo, self.jitter_hi)
            out = out * scale

        return out
