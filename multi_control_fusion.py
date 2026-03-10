from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from fusion_mlp import PerLayerFusionMLP


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
    fusion_weights: Optional[torch.Tensor] = None  # [J, 2], canny/depth


class LearnedWeightMultiControlFusion(nn.Module):
    """
    Run two frozen ControlNets and fuse their outputs with either:
      - learned per-layer weights from a trained MLP, or
      - fixed fallback weights.

    Injection-point ordering is:
      [all down blocks in order..., mid block]
    so if there are N down residuals, the MLP must output N + 1 weight pairs.
    """

    def __init__(
        self,
        canny_controlnet: nn.Module,
        depth_controlnet: nn.Module,
        fusion_mlp: Optional[PerLayerFusionMLP] = None,
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
            fusion_mlp = PerLayerFusionMLP.load(fusion_mlp_path, map_location=map_location)

        self.fusion_mlp = fusion_mlp
        if self.fusion_mlp is not None:
            self.fusion_mlp.eval()
            for p in self.fusion_mlp.parameters():
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
        """Call both ControlNets and return their raw outputs."""
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
        """Ensure both ControlNets return matching residual structures."""
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
                f"{tuple(canny_out.mid_block_res_sample.shape)} vs {tuple(depth_out.mid_block_res_sample.shape)}"
            )

    def _get_fusion_weights(
        self,
        num_down_blocks: int,
        sample_device: torch.device,
        sample_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns weights with shape [J, 2], where J = num_down_blocks + 1.
        [:, 0] is canny weight and [:, 1] is depth weight.
        """
        num_injection_points = num_down_blocks + 1

        if self.fusion_mlp is None:
            weights = torch.tensor(
                [[self.fallback_canny_weight, self.fallback_depth_weight]] * num_injection_points,
                device=sample_device,
                dtype=sample_dtype,
            )
            return weights

        expected = self.fusion_mlp.num_injection_points
        if expected != num_injection_points:
            raise ValueError(
                "Fusion MLP injection-point count does not match ControlNet outputs: "
                f"MLP expects {expected}, but ControlNets produced {num_injection_points} "
                f"({num_down_blocks} down + 1 mid)"
            )

        mlp_device = next(self.fusion_mlp.parameters()).device
        if mlp_device != sample_device:
            self.fusion_mlp = self.fusion_mlp.to(sample_device)

        weights = self.fusion_mlp.get_all_fusion_weights(temperature=self.temperature)
        return weights.to(device=sample_device, dtype=sample_dtype)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        encoder_hidden_states: torch.Tensor,
        canny_cond: torch.Tensor,
        depth_cond: torch.Tensor,
    ) -> FusedControlOutput:
        """
        1. Run both ControlNets
        2. Fuse residuals with learned per-layer weights, if available
        3. Otherwise fall back to fixed weights
        4. Return fused residuals
        """
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

        fusion_weights = self._get_fusion_weights(
            num_down_blocks=len(canny_out.down_block_res_samples),
            sample_device=sample.device,
            sample_dtype=canny_out.mid_block_res_sample.dtype,
        )

        fused_down = []
        for j, (c_res, d_res) in enumerate(
            zip(canny_out.down_block_res_samples, depth_out.down_block_res_samples)
        ):
            canny_w = fusion_weights[j, 0]
            depth_w = fusion_weights[j, 1]
            fused_down.append(canny_w * c_res + depth_w * d_res)

        mid_idx = len(canny_out.down_block_res_samples)
        fused_mid = (
            fusion_weights[mid_idx, 0] * canny_out.mid_block_res_sample
            + fusion_weights[mid_idx, 1] * depth_out.mid_block_res_sample
        )

        return FusedControlOutput(
            down_block_res_samples=tuple(fused_down),
            mid_block_res_sample=fused_mid,
            fusion_weights=fusion_weights,
        )


# Backward-compatible alias so older imports do not break.
EqualWeightMultiControlFusion = LearnedWeightMultiControlFusion
