from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from fusion_mlp import ContextEncoder, PerLayerFusionMLP, load_checkpoint


@dataclass
class FusedControlOutput:
    """
    Fused residuals to pass into the SD U-Net.

    These correspond to:
      - down_block_additional_residuals
      - mid_block_additional_residual
    """
    down_block_res_samples: Tuple[torch.Tensor, ...]
    mid_block_res_sample: torch.Tensor
    fusion_weights: Optional[torch.Tensor] = None  # [J, 2] or [B, J, 2]


class LearnedWeightMultiControlFusion(nn.Module):
    """
    Run two frozen ControlNets and fuse their outputs with either:
      - learned per-sample weights from a trained MLP + ContextEncoder, or
      - static learned weights (MLP only, no context encoder), or
      - fixed fallback weights.

    When a ContextEncoder is present the weights become image-, text-, and
    timestep-dependent, producing different fusion decisions for every sample.

    Injection-point ordering:
      [all down blocks in order..., mid block]
    so if there are N down residuals, the MLP must output N + 1 weight pairs.
    """

    def __init__(
        self,
        canny_controlnet: nn.Module,
        depth_controlnet: nn.Module,
        fusion_mlp: Optional[PerLayerFusionMLP] = None,
        context_encoder: Optional[ContextEncoder] = None,
        fusion_mlp_path: Optional[str | Path] = None,
        map_location: str | torch.device = "cpu",
        temperature: float = 1.0,
        fallback_canny_weight: float = 0.5,
        fallback_depth_weight: float = 0.5,
        validate_shapes_once: bool = True,
    ) -> None:
        super().__init__()
        self.canny_controlnet = canny_controlnet
        self.depth_controlnet = depth_controlnet
        self.temperature = float(temperature)
        self.fallback_canny_weight = float(fallback_canny_weight)
        self.fallback_depth_weight = float(fallback_depth_weight)
        self.validate_shapes_once = bool(validate_shapes_once)
        self._shapes_validated = False

        if fusion_mlp is not None and fusion_mlp_path is not None:
            raise ValueError("Provide either fusion_mlp or fusion_mlp_path, not both")

        if fusion_mlp_path is not None:
            fusion_mlp, context_encoder = load_checkpoint(fusion_mlp_path, map_location=map_location)

        self.fusion_mlp = fusion_mlp
        self.context_encoder = context_encoder

        for module in (self.fusion_mlp, self.context_encoder):
            if module is not None:
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)

    def has_learned_fusion(self) -> bool:
        return self.fusion_mlp is not None

    def _run_controlnets(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        encoder_hidden_states: torch.Tensor,
        canny_cond: torch.Tensor,
        depth_cond: torch.Tensor,
    ):
        canny_out = self.canny_controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=canny_cond,
            conditioning_scale=1.0,
            return_dict=True,
        )
        depth_out = self.depth_controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=depth_cond,
            conditioning_scale=1.0,
            return_dict=True,
        )
        return canny_out, depth_out

    @staticmethod
    def _validate_shapes(canny_out, depth_out) -> None:
        if len(canny_out.down_block_res_samples) != len(depth_out.down_block_res_samples):
            raise ValueError(
                "Mismatch in number of down block residuals: "
                f"{len(canny_out.down_block_res_samples)} vs "
                f"{len(depth_out.down_block_res_samples)}"
            )
        for i, (c_res, d_res) in enumerate(
            zip(canny_out.down_block_res_samples, depth_out.down_block_res_samples)
        ):
            if c_res.shape != d_res.shape:
                raise ValueError(
                    f"Residual shape mismatch at down block {i}: "
                    f"{tuple(c_res.shape)} vs {tuple(d_res.shape)}"
                )
        if canny_out.mid_block_res_sample.shape != depth_out.mid_block_res_sample.shape:
            raise ValueError(
                "Mid block residual shape mismatch: "
                f"{tuple(canny_out.mid_block_res_sample.shape)} vs "
                f"{tuple(depth_out.mid_block_res_sample.shape)}"
            )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        encoder_hidden_states: torch.Tensor,
        canny_cond: torch.Tensor,
        depth_cond: torch.Tensor,
    ) -> FusedControlOutput:
        canny_out, depth_out = self._run_controlnets(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            canny_cond=canny_cond,
            depth_cond=depth_cond,
        )

        if not self.validate_shapes_once or not self._shapes_validated:
            self._validate_shapes(canny_out, depth_out)
            self._shapes_validated = True

        num_down = len(canny_out.down_block_res_samples)
        B = sample.shape[0]

        # ── Get fusion weights ───────────────────────────────────────────────
        if self.fusion_mlp is None:
            # Fixed fallback weights: [J, 2]
            num_inj = num_down + 1
            weights = torch.tensor(
                [[self.fallback_canny_weight, self.fallback_depth_weight]] * num_inj,
                device=sample.device,
                dtype=canny_out.mid_block_res_sample.dtype,
            )
        elif self.context_encoder is not None:
            # Image-dependent weights via ContextEncoder → [B, J, 2]
            canny_feat = canny_out.down_block_res_samples[0].float().mean(dim=(-2, -1))  # [B, C]
            depth_feat = depth_out.down_block_res_samples[0].float().mean(dim=(-2, -1))  # [B, C]

            t = timestep
            if isinstance(t, (int, float)):
                t = torch.tensor([t], device=sample.device, dtype=torch.long).expand(B)
            elif not isinstance(t, torch.Tensor):
                t = torch.tensor([t], device=sample.device, dtype=torch.long).expand(B)
            elif t.dim() == 0:
                t = t.unsqueeze(0).expand(B)
            elif t.shape[0] == 1 and B > 1:
                t = t.expand(B)

            context = self.context_encoder(
                timestep=t,
                text_emb=encoder_hidden_states,
                canny_feat=canny_feat,
                depth_feat=depth_feat,
            )  # [B, context_dim]
            weights = self.fusion_mlp(context=context)  # [B, J, 2]
        else:
            # Static learned weights (old checkpoint without context encoder): [J, 2]
            weights = self.fusion_mlp.get_all_fusion_weights(temperature=self.temperature)
            weights = weights.to(device=sample.device, dtype=canny_out.mid_block_res_sample.dtype)

        batched = weights.dim() == 3  # True when context encoder was used

        # ── Apply weights ────────────────────────────────────────────────────
        fused_down = []
        for j, (c_res, d_res) in enumerate(
            zip(canny_out.down_block_res_samples, depth_out.down_block_res_samples)
        ):
            if batched:
                canny_w = weights[:, j, 0].to(c_res.dtype).view(B, 1, 1, 1)
                depth_w = weights[:, j, 1].to(d_res.dtype).view(B, 1, 1, 1)
            else:
                canny_w = weights[j, 0].to(c_res.dtype)
                depth_w = weights[j, 1].to(d_res.dtype)
            fused_down.append(canny_w * c_res + depth_w * d_res)

        mid_c = canny_out.mid_block_res_sample
        mid_d = depth_out.mid_block_res_sample
        if batched:
            canny_mid = weights[:, num_down, 0].to(mid_c.dtype).view(B, 1, 1, 1)
            depth_mid = weights[:, num_down, 1].to(mid_d.dtype).view(B, 1, 1, 1)
        else:
            canny_mid = weights[num_down, 0].to(mid_c.dtype)
            depth_mid = weights[num_down, 1].to(mid_d.dtype)
        fused_mid = canny_mid * mid_c + depth_mid * mid_d

        return FusedControlOutput(
            down_block_res_samples=tuple(fused_down),
            mid_block_res_sample=fused_mid,
            fusion_weights=weights,  # [J, 2] or [B, J, 2]
        )


# Backward-compatible alias so older imports do not break.
EqualWeightMultiControlFusion = LearnedWeightMultiControlFusion
