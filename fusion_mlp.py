from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal timestep embedding.

    Args:
        timesteps: [B] long tensor of diffusion timesteps.
        dim: embedding dimension (must be even).

    Returns:
        [B, dim] float32 tensor.
    """
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / (half - 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    return torch.cat([args.sin(), args.cos()], dim=-1)          # [B, dim]


# ---------------------------------------------------------------------------
# Context encoder
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """
    Encodes sample-dependent signals into a fixed-size context vector.

    Four branches (each projected to proj_dim with a SiLU activation):

        timestep   – sinusoidal embedding of the diffusion timestep
        text       – mean-pooled CLIP encoder hidden states
        canny_feat – globally-pooled first ControlNet down-block feature
        depth_feat – globally-pooled first ControlNet down-block feature

    Output: [B, context_dim] where context_dim = 4 * proj_dim = 256 by default.

    The module is dtype-agnostic: inputs are cast to the dtype of the internal
    linear weights before every projection.
    """

    def __init__(
        self,
        text_dim: int = 768,
        canny_feat_dim: int = 320,  # channels in first ControlNet down block (SD v1.5)
        depth_feat_dim: int = 320,
        ts_emb_dim: int = 256,      # sinusoidal embedding width before projection
        proj_dim: int = 64,         # per-branch output width
    ) -> None:
        super().__init__()
        self.ts_emb_dim = ts_emb_dim
        self.proj_dim = proj_dim
        self.context_dim = 4 * proj_dim

        self.ts_proj    = nn.Sequential(nn.Linear(ts_emb_dim, proj_dim),     nn.SiLU())
        self.text_proj  = nn.Sequential(nn.Linear(text_dim, proj_dim),       nn.SiLU())
        self.canny_proj = nn.Sequential(nn.Linear(canny_feat_dim, proj_dim), nn.SiLU())
        self.depth_proj = nn.Sequential(nn.Linear(depth_feat_dim, proj_dim), nn.SiLU())

    def forward(
        self,
        timestep: torch.Tensor,    # [B] long
        text_emb: torch.Tensor,    # [B, seq_len, text_dim]
        canny_feat: torch.Tensor,  # [B, canny_feat_dim]  (globally pooled)
        depth_feat: torch.Tensor,  # [B, depth_feat_dim]  (globally pooled)
    ) -> torch.Tensor:             # [B, context_dim]
        proj_dtype = self.ts_proj[0].weight.dtype
        device     = self.ts_proj[0].weight.device

        ts_emb = sinusoidal_embedding(timestep, self.ts_emb_dim).to(device=device, dtype=proj_dtype)
        ts_ctx    = self.ts_proj(ts_emb)
        text_ctx  = self.text_proj(text_emb.mean(dim=1).to(device=device, dtype=proj_dtype))
        canny_ctx = self.canny_proj(canny_feat.to(device=device, dtype=proj_dtype))
        depth_ctx = self.depth_proj(depth_feat.to(device=device, dtype=proj_dtype))
        return torch.cat([ts_ctx, text_ctx, canny_ctx, depth_ctx], dim=-1)  # [B, 4*proj_dim]

    @property
    def config(self) -> dict:
        return {
            "text_dim":       self.text_proj[0].in_features,
            "canny_feat_dim": self.canny_proj[0].in_features,
            "depth_feat_dim": self.depth_proj[0].in_features,
            "ts_emb_dim":     self.ts_emb_dim,
            "proj_dim":       self.proj_dim,
        }


# ---------------------------------------------------------------------------
# Fusion MLP
# ---------------------------------------------------------------------------

class PerLayerFusionMLP(nn.Module):
    """
    Lightweight MLP for learned per-injection-point fusion.

    Static mode (context_dim == 0, default)
    ----------------------------------------
    Input:  injection index embedding  [J, index_emb_dim]
    Output: fusion weights             [J, 2, Gh, Gw]  (canny, depth)

    Image-dependent mode (context_dim > 0)
    ----------------------------------------
    Input:  index embedding + context  [B, J, index_emb_dim + context_dim]
    Output: per-sample fusion weights  [B, J, 2, Gh, Gw]

    When context is not supplied at inference time but context_dim > 0, a
    zero context vector is used so that get_all_fusion_weights / pretty_print
    still work for diagnostics (they reflect the bias of the MLP at zero
    conditioning, i.e. the learned "default" weights).
    """

    def __init__(
        self,
        num_injection_points: int,
        index_emb_dim: int = 32,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
        context_dim: int = 0,
        weight_grid_h: int = 8,
        weight_grid_w: int = 8,
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
        self.context_dim = context_dim
        self.weight_grid_h = weight_grid_h
        self.weight_grid_w = weight_grid_w
        if self.weight_grid_h <= 0 or self.weight_grid_w <= 0:
            raise ValueError("weight_grid_h and weight_grid_w must be > 0")

        self.index_embedding = nn.Embedding(num_injection_points, index_emb_dim)

        layers: list[nn.Module] = []
        in_dim = index_emb_dim + context_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.out_proj = nn.Linear(in_dim, 2 * self.weight_grid_h * self.weight_grid_w)
        nn.init.zeros_(self.out_proj.weight) # to make initial fusion = simple average of canny/depth (after sigmoid)
        nn.init.constant_(self.out_proj.bias, 0.0)  
        # nn.init.zeros_(self.out_proj.weight) # to make initial fusion = simple average of canny/depth (after softmax)
        # nn.init.zeros_(self.out_proj.bias)


    def forward(
        self,
        injection_indices: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            injection_indices:
                Optional [J] long tensor. Defaults to all indices.
            context:
                Optional [B, context_dim] tensor. When supplied the MLP is run
                once per sample and returns [B, J, 2, Gh, Gw] weights.
                When omitted and context_dim > 0, a zero vector is used and
                the output is squeezed back to [J, 2, Gh, Gw] for compatibility.
            temperature: sigmoid temperature (> 0).
            return_logits: if True returns (weights, logits).

        Returns:
            weights [J, 2, Gh, Gw] or [B, J, 2, Gh, Gw].
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        emb_device = self.index_embedding.weight.device
        emb_dtype  = self.index_embedding.weight.dtype

        if injection_indices is None:
            injection_indices = torch.arange(
                self.num_injection_points, device=emb_device, dtype=torch.long
            )
        else:
            injection_indices = injection_indices.to(device=emb_device, dtype=torch.long)

        idx_emb = self.index_embedding(injection_indices)  # [J, index_emb_dim]

        # If context_dim > 0 but no context given, use zeros (diagnostic / compat path)
        squeeze = False
        if context is None and self.context_dim > 0:
            context = torch.zeros(1, self.context_dim, device=emb_device, dtype=emb_dtype)
            squeeze = True

        if context is not None:
            B = context.shape[0]
            J = idx_emb.shape[0]
            idx = idx_emb.unsqueeze(0).expand(B, -1, -1)           # [B, J, index_emb_dim]
            ctx = context.unsqueeze(1).expand(-1, J, -1)            # [B, J, context_dim]
            x = torch.cat([idx, ctx.to(dtype=emb_dtype)], dim=-1)  # [B, J, total_in]
        else:
            # Static path: context_dim == 0
            x = idx_emb  # [J, index_emb_dim]

        # nn.Linear applies to last dim, works for both [J, d] and [B, J, d]
        x = self.mlp(x)
        logits = self.out_proj(x)
        logits = logits.view(*logits.shape[:-1], 2, self.weight_grid_h, self.weight_grid_w)
        weights = torch.sigmoid(logits / temperature)

        if squeeze:
            weights = weights.squeeze(0)
            logits  = logits.squeeze(0)

        if return_logits:
            return weights, logits
        return weights

    def get_all_fusion_weights(self, temperature: float = 1.5) -> torch.Tensor:
        """Returns [num_injection_points, 2, Gh, Gw] (at zero context when context_dim > 0)."""
        return self.forward(injection_indices=None, context=None, temperature=temperature)

    @torch.no_grad()
    def pretty_print(self, temperature: float = 1.0) -> None:
        weights = self.get_all_fusion_weights(temperature=temperature).detach().cpu()
        for j in range(weights.shape[0]):
            canny_w = float(weights[j, 0].mean().item())
            depth_w = float(weights[j, 1].mean().item())
            print(f"layer {j:02d}: canny={canny_w:.4f}, depth={depth_w:.4f}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "state_dict": self.state_dict(),
            "config": {
                "num_injection_points": self.num_injection_points,
                "index_emb_dim":        self.index_emb_dim,
                "hidden_dim":           self.hidden_dim,
                "num_hidden_layers":    self.num_hidden_layers,
                "dropout":              self.dropout,
                "context_dim":          self.context_dim,
                "weight_grid_h":        self.weight_grid_h,
                "weight_grid_w":        self.weight_grid_w,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "PerLayerFusionMLP":
        payload = torch.load(path, map_location=map_location)
        cfg = dict(payload["config"])
        # Old checkpoints saved before spatial grids existed.
        cfg.setdefault("weight_grid_h", 1)
        cfg.setdefault("weight_grid_w", 1)
        model = cls(**cfg)
        model.load_state_dict(payload["state_dict"])
        return model


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[PerLayerFusionMLP, ContextEncoder | None]:
    """
    Load a checkpoint saved by save_checkpoint() in train_fusion_mlp.py.

    Handles both:
      - new combined format (fusion_mlp + context_encoder)
      - old format (fusion_mlp only, via PerLayerFusionMLP.load)

    Returns (fusion_mlp, context_encoder_or_None).
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)

    def _with_legacy_grid_defaults(cfg: dict) -> dict:
        # Old checkpoints saved before spatial weight maps existed.
        # Preserve old behavior by loading them as 1x1 (global) weights.
        cfg = dict(cfg)
        cfg.setdefault("weight_grid_h", 1)
        cfg.setdefault("weight_grid_w", 1)
        return cfg

    # New combined format written by save_checkpoint()
    if "fusion_mlp_config" in payload:
        fusion_mlp = PerLayerFusionMLP(**_with_legacy_grid_defaults(payload["fusion_mlp_config"]))
        fusion_mlp.load_state_dict(payload["fusion_mlp_state_dict"])
        context_encoder = None
        if "context_encoder_config" in payload:
            context_encoder = ContextEncoder(**payload["context_encoder_config"])
            context_encoder.load_state_dict(payload["context_encoder_state_dict"])
        return fusion_mlp, context_encoder

    # Old format written by PerLayerFusionMLP.save()
    fusion_mlp = PerLayerFusionMLP(**_with_legacy_grid_defaults(payload["config"]))
    fusion_mlp.load_state_dict(payload["state_dict"])
    return fusion_mlp, None
