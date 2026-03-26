from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class PerLayerFusionMLP(nn.Module):
    """
    Lightweight MLP for learned per-injection fusion.

    For each ControlNet injection point j, the model learns two logits:
        [logit_canny_j, logit_depth_j]

    A softmax converts them into normalized mixing weights:
        alpha_canny_j + alpha_depth_j = 1

    This follows the project proposal where the MLP consumes an embedding of
    the injection index and predicts one pair of weights per injection point.
    """

    def __init__(
        self,
        num_injection_points: int,
        index_emb_dim: int = 32,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_injection_points <= 0:
            raise ValueError("num_injection_points must be > 0")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be > 0")

        self.num_injection_points = num_injection_points
        self.index_emb_dim = index_emb_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout

        self.index_embedding = nn.Embedding(num_injection_points, index_emb_dim)

        layers: list[nn.Module] = []
        in_dim = index_emb_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.out_proj = nn.Linear(in_dim, 2)
        nn.init.zeros_(self.out_proj.weight) # to make initial fusion = simple average of canny/depth (after softmax)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        injection_indices: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            injection_indices:
                Optional LongTensor of shape [J]. If omitted, uses all indices
                [0, 1, ..., num_injection_points - 1].
            temperature:
                Softmax temperature. 1.0 is standard.
            return_logits:
                If True, returns (weights, logits).

        Returns:
            weights of shape [J, 2], where [:, 0] is canny and [:, 1] is depth.
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        if injection_indices is None:
            injection_indices = torch.arange(
                self.num_injection_points,
                device=self.index_embedding.weight.device,
                dtype=torch.long,
            )
        else:
            injection_indices = injection_indices.to(
                device=self.index_embedding.weight.device,
                dtype=torch.long,
            )

        x = self.index_embedding(injection_indices)
        x = self.mlp(x)
        logits = self.out_proj(x)
        weights = torch.softmax(logits / temperature, dim=-1)

        if return_logits:
            return weights, logits
        return weights

    def get_all_fusion_weights(
        self,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Convenience wrapper returning [num_injection_points, 2]."""
        return self.forward(injection_indices=None, temperature=temperature)

    @torch.no_grad()
    def pretty_print(self, temperature: float = 1.0) -> None:
        weights = self.get_all_fusion_weights(temperature=temperature).detach().cpu()
        for j, pair in enumerate(weights):
            canny_w, depth_w = pair.tolist()
            print(f"layer {j:02d}: canny={canny_w:.4f}, depth={depth_w:.4f}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "state_dict": self.state_dict(),
            "config": {
                "num_injection_points": self.num_injection_points,
                "index_emb_dim": self.index_emb_dim,
                "hidden_dim": self.hidden_dim,
                "num_hidden_layers": self.num_hidden_layers,
                "dropout": self.dropout,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "PerLayerFusionMLP":
        payload = torch.load(path, map_location=map_location)
        model = cls(**payload["config"])
        model.load_state_dict(payload["state_dict"])
        return model
