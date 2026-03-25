from __future__ import annotations

import random

import numpy as np
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fusion_mlp import PerLayerFusionMLP
from models import load_training_models


# -----------------------------
# Dataset (pre-computed tensors)
# -----------------------------
class PrecomputedDataset(Dataset):
    """
    Scans pt_dir for all *_latent.pt files and expects matching:
        pt/{stem}_canny.pt
        pt/{stem}_depth.pt
        pt/{stem}_latent.pt
        pt/{stem}_text_emb.pt
    No CSV needed.
    """

    def __init__(self, pt_dir: str | Path = "pt") -> None:
        pt_dir = Path(pt_dir)
        self.samples: list[tuple[Path, Path, Path, Path]] = []

        for latent_path in sorted(pt_dir.glob("*_latent.pt")):
            stem = latent_path.stem[: -len("_latent")]
            canny    = pt_dir / f"{stem}_canny.pt"
            depth    = pt_dir / f"{stem}_depth.pt"
            text_emb = pt_dir / f"{stem}_text_emb.pt"
            missing  = [p for p in (canny, depth, text_emb) if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing for stem '{stem}': {missing}")
            self.samples.append((canny, depth, latent_path, text_emb))

        if not self.samples:
            raise ValueError(f"No *_latent.pt files found in {pt_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        canny_path, depth_path, latent_path, emb_path = self.samples[idx]
        return (
            torch.load(canny_path, weights_only=True),    # [3, H, W]
            torch.load(depth_path, weights_only=True),    # [3, H, W]
            torch.load(latent_path, weights_only=True),   # [4, 64, 64]
            torch.load(emb_path, weights_only=True),      # [77, 768]
        )


def collate_fn(batch):
    cannys, depths, latents, text_embs = zip(*batch)
    return torch.stack(cannys), torch.stack(depths), torch.stack(latents), torch.stack(text_embs)


# -----------------------------
# Fusion helpers
# -----------------------------
@dataclass
class FusedControlOutput:
    down_block_res_samples: tuple[torch.Tensor, ...]
    mid_block_res_sample: torch.Tensor


class LearnedPerLayerFusion(nn.Module):
    """
    Small helper that uses the frozen ControlNets and the trainable MLP.

    We keep ControlNet outputs detached because the project proposal freezes
    those backbones and only optimizes the MLP. The fused tensors still carry
    gradients to the MLP because the learned alpha weights require gradients.
    """

    def __init__(
        self,
        canny_controlnet: nn.Module,
        depth_controlnet: nn.Module,
        fusion_mlp: PerLayerFusionMLP,
    ) -> None:
        super().__init__()
        self.canny_controlnet = canny_controlnet
        self.depth_controlnet = depth_controlnet
        self.fusion_mlp = fusion_mlp

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        encoder_hidden_states: torch.Tensor,
        canny_cond: torch.Tensor,
        depth_cond: torch.Tensor,
    ) -> FusedControlOutput:
        with torch.no_grad():
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

        num_down = len(canny_out.down_block_res_samples)
        weights = self.fusion_mlp.get_all_fusion_weights()  # [J, 2]
        expected_points = num_down + 1
        if weights.shape[0] != expected_points:
            raise ValueError(
                f"Fusion MLP expects {weights.shape[0]} injection points, "
                f"but ControlNet produced {expected_points}"
            )

        fused_down = []
        for j, (c_res, d_res) in enumerate(zip(canny_out.down_block_res_samples, depth_out.down_block_res_samples)):
            alpha_canny = weights[j, 0].to(dtype=c_res.dtype)
            alpha_depth = weights[j, 1].to(dtype=d_res.dtype)
            fused = alpha_canny * c_res.detach() + alpha_depth * d_res.detach()
            fused_down.append(fused)

        alpha_canny_mid = weights[num_down, 0].to(dtype=canny_out.mid_block_res_sample.dtype)
        alpha_depth_mid = weights[num_down, 1].to(dtype=depth_out.mid_block_res_sample.dtype)
        fused_mid = (
            alpha_canny_mid * canny_out.mid_block_res_sample.detach()
            + alpha_depth_mid * depth_out.mid_block_res_sample.detach()
        )

        return FusedControlOutput(
            down_block_res_samples=tuple(fused_down),
            mid_block_res_sample=fused_mid,
        )


# -----------------------------
# Train config
# -----------------------------
@dataclass
class TrainConfig:
    output_dir: str = "fusion_mlp_ckpts"
    batch_size: int = 1          # 32 will OOM; use 1 for <8 GB VRAM, 2-4 for 12-16 GB
    gradient_accumulation_steps: int = 8   # effective batch = batch_size × accum_steps
    epochs: int = 1
    lr: float = 1e-3             # ok since only the tiny fusion MLP is trained
    weight_decay: float = 1e-4
    pt_dir: str = "pt"
    # num_inference_train_timesteps: int = 1000   # only 1 random t sampled per step — cost is fine
    max_grad_norm: float = 1.0
    save_every_steps: int = 500   # counted in optimizer steps (post-accumulation)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# -----------------------------
# Main train loop
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_num_injection_points(parts, device: str, dtype: torch.dtype) -> int:
    with torch.no_grad():
        sample = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
        cond = torch.randn(1, 3, 512, 512, device=device, dtype=dtype)
        tokens = torch.randn(1, 77, 768, device=device, dtype=dtype)
        out = parts["canny_controlnet"](
            sample=sample,
            timestep=torch.tensor(1, device=device),
            encoder_hidden_states=tokens,
            controlnet_cond=cond,
            conditioning_scale=1.0,
            return_dict=True,
        )
    return len(out.down_block_res_samples) + 1


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parts = load_training_models(device=cfg.device, dtype=cfg.dtype)
    num_points = discover_num_injection_points(parts, cfg.device, cfg.dtype)

    fusion_mlp = PerLayerFusionMLP(
        num_injection_points=num_points,
        index_emb_dim=32,
        hidden_dim=64,
        num_hidden_layers=2,
        dropout=0.0,
    ).to(cfg.device)

    fusion = LearnedPerLayerFusion(
        canny_controlnet=parts["canny_controlnet"],
        depth_controlnet=parts["depth_controlnet"],
        fusion_mlp=fusion_mlp,
    ).to(cfg.device)

    print("Initial weights (before any training):")
    fusion_mlp.pretty_print()

    optimizer = torch.optim.AdamW(
        fusion_mlp.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = DDIMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler",
    )
    # scheduler.set_timesteps(cfg.num_inference_train_timesteps, device=cfg.device)

    dataset = PrecomputedDataset(pt_dir=cfg.pt_dir)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    global_step = 0          # counts optimizer steps (post-accumulation)
    accum_step = 0           # counts micro-batches within the current accumulation window
    accum_loss = 0.0         # running sum for logging
    best_loss = float("inf")

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        fusion_mlp.train()
        epoch_loss_sum = 0.0
        epoch_optimizer_steps = 0

        for canny_cond, depth_cond, latents, text_embs in loader:
            batch_size = latents.shape[0]

            canny_cond            = canny_cond.to(device=cfg.device, dtype=cfg.dtype)
            depth_cond            = depth_cond.to(device=cfg.device, dtype=cfg.dtype)
            latents               = latents.to(device=cfg.device, dtype=cfg.dtype)
            encoder_hidden_states = text_embs.to(device=cfg.device, dtype=cfg.dtype)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                low=0,
                high=scheduler.config.num_train_timesteps,
                size=(batch_size,),
                device=cfg.device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            fused = fusion(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                canny_cond=canny_cond,
                depth_cond=depth_cond,
            )

            noise_pred = parts["unet"](
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=fused.down_block_res_samples,
                mid_block_additional_residual=fused.mid_block_res_sample,
                return_dict=True,
            ).sample

            # Divide loss so that the sum over accumulation steps ≈ a single
            # large-batch loss (same scale regardless of accum window size).
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            (loss / cfg.gradient_accumulation_steps).backward()
            accum_loss += loss.item()
            accum_step += 1

            if accum_step % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(fusion_mlp.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                avg_loss = accum_loss / cfg.gradient_accumulation_steps
                epoch_loss_sum += avg_loss
                epoch_optimizer_steps += 1
                accum_loss = 0.0

                print(
                    f"epoch={epoch + 1}/{cfg.epochs} "
                    f"opt_step={global_step} "
                    f"loss={avg_loss:.6f}"
                )

                # if avg_loss < best_loss:
                #     best_loss = avg_loss
                #     fusion_mlp.save(output_dir / "fusion_mlp_best.pth")

                if global_step % cfg.save_every_steps == 0:
                    fusion_mlp.save(output_dir / f"fusion_mlp_step_{global_step}.pth")

        print(f"\n--- epoch {epoch + 1} summary ---")
        print(f"avg loss: {epoch_loss_sum / max(epoch_optimizer_steps, 1):.6f}")
        print("weights after this epoch:")
        fusion_mlp.pretty_print()
        fusion_mlp.save(output_dir / f"fusion_mlp_epoch_{epoch + 1}.pth")

    fusion_mlp.save(output_dir / "fusion_mlp_last.pth")

    print("Training finished.")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Saved checkpoints to: {output_dir}")
    print("Final learned weights:")
    fusion_mlp.pretty_print()


if __name__ == "__main__":
    train(TrainConfig())
