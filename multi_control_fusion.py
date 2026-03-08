from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


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


class EqualWeightMultiControlFusion(nn.Module):
    """
    Run two frozen ControlNets and fuse their outputs with fixed weights.

    Current project milestone:
    - Canny ControlNet
    - Depth ControlNet
    - equal-weight fusion

    Later you will replace the fixed weights with an MLP.
    """

    def __init__(
        self,
        canny_controlnet: nn.Module,
        depth_controlnet: nn.Module,
        canny_weight: float = 0.5,
        depth_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.canny_controlnet = canny_controlnet
        self.depth_controlnet = depth_controlnet
        self.canny_weight = canny_weight
        self.depth_weight = depth_weight

    def _run_controlnets(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        encoder_hidden_states: torch.Tensor,
        canny_cond: torch.Tensor,
        depth_cond: torch.Tensor,
    ):
        """
        Call both ControlNets and return their raw outputs.
        """
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
        """
        Ensure both ControlNets return matching residual structures.
        """
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
        """
        1. Run both ControlNets
        2. Fuse residuals with equal weights
        3. Return fused residuals
        """
        canny_out, depth_out = self._run_controlnets(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            canny_cond=canny_cond,
            depth_cond=depth_cond,
        )

        self._validate_shapes(canny_out, depth_out)

        fused_down = []
        for c_res, d_res in zip(
            canny_out.down_block_res_samples,
            depth_out.down_block_res_samples,
        ):
            fused = self.canny_weight * c_res + self.depth_weight * d_res
            fused_down.append(fused)

        fused_mid = (
            self.canny_weight * canny_out.mid_block_res_sample
            + self.depth_weight * depth_out.mid_block_res_sample
        )

        return FusedControlOutput(
            down_block_res_samples=tuple(fused_down),
            mid_block_res_sample=fused_mid,
        )