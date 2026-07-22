"""Building blocks for velocity fields."""

import torch
import torch.nn as nn


class VelocityBlock(nn.Module):
    """
    Residual block: f(x) = W3·(W2·act(W1·x + b1) + b2).
    """

    def __init__(self, in_dim: int, width: int, activation: str = "relu"):
        super().__init__()
        self.lift = nn.Linear(in_dim, width)  # W1, b1
        self.mix = nn.Linear(width, width)  # W2, b2
        self.proj = nn.Linear(width, 3, bias=False)  # W3 — no bias
        self.act = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }[activation]
        # Zero-init proj.weight so v ≡ 0 at init ⇒ flow == identity
        nn.init.zeros_(self.proj.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[B, N, in_dim] -> [B, N, 3]."""
        return self.proj(self.mix(self.act(self.lift(h))))


class FourierFeatures(nn.Module):
    """Positional encoding: γ(h) = [h, sin(2⁰πh), cos(2⁰πh), …, sin(2^{Ne−1}πh), cos(2^{Ne−1}πh)].

    Base-2 log-spaced (AD-SVFD Methods); identity passthrough included.
    Output dim = (2·n_e + 1)·d where d is the input feature dimension.
    """

    def __init__(self, n_e: int = 3):
        super().__init__()
        # Register buffer: frequencies [2^0, 2^1, ..., 2^(n_e-1)] * π
        self.register_buffer("freqs", torch.pi * 2.0 ** torch.arange(n_e))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[..., d] -> [..., (2*n_e + 1)*d]."""
        # h.unsqueeze(-1) broadcasts [..., d] to [..., d, 1]
        # freqs shape: [n_e]
        # angles shape: [..., d, n_e]
        angles = h.unsqueeze(-1) * self.freqs
        # sin/cos of angles: [..., d, n_e] each
        # flatten to [..., d*n_e] each
        sin_parts = angles.sin().flatten(-2)
        cos_parts = angles.cos().flatten(-2)
        # Concatenate: [h, sin, cos] -> [..., d + d*n_e + d*n_e] = [..., (2*n_e + 1)*d]
        return torch.cat([h, sin_parts, cos_parts], dim=-1)
