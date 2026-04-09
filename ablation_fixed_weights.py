"""
Ablation: does depth actually help denoising?

Runs the eval loop three times with fixed fusion weights:
  - canny-only  (1.0, 0.0)
  - depth-only  (0.0, 1.0)
  - equal       (0.5, 0.5)

Uses the same deterministic noise + fixed timestep as the training evaluator
so the numbers are directly comparable across runs.

Usage:
    python ablation_fixed_weights.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader

from train_fusion_mlp import PrecomputedDataset, TrainConfig, collate_fn, set_seed
from models import load_training_models


# ── helpers ──────────────────────────────────────────────────────────────────

def run_fixed_weight_eval(
    canny_w: float,
    depth_w: float,
    loader: DataLoader,
    parts: dict,
    scheduler: DDIMScheduler,
    cfg: TrainConfig,
    max_batches: int | None = None,
) -> float:
    """
    Evaluate denoising MSE using constant per-layer fusion weights
    (same weight applied to every injection point / layer).
    No MLP involved — pure fixed-weight fusion.
    """
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for stems, canny_cond, depth_cond, latents, text_embs in loader:
            canny_cond            = canny_cond.to(device=cfg.device, dtype=cfg.dtype)
            depth_cond            = depth_cond.to(device=cfg.device, dtype=cfg.dtype)
            latents               = latents.to(device=cfg.device, dtype=cfg.dtype)
            encoder_hidden_states = text_embs.to(device=cfg.device, dtype=cfg.dtype)

            # Same deterministic noise as training eval
            generator = torch.Generator(device=cfg.device)
            generator.manual_seed(cfg.eval_seed + total_steps)
            noise = torch.randn(latents.shape, generator=generator,
                                device=cfg.device, dtype=cfg.dtype)
            timesteps = torch.full(
                (latents.shape[0],),
                fill_value=cfg.eval_timestep,
                device=cfg.device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            canny_out = parts["canny_controlnet"](
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=canny_cond,
                conditioning_scale=1.0,
                return_dict=True,
            )
            depth_out = parts["depth_controlnet"](
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=depth_cond,
                conditioning_scale=1.0,
                return_dict=True,
            )

            fused_down = [
                canny_w * c + depth_w * d
                for c, d in zip(canny_out.down_block_res_samples,
                                depth_out.down_block_res_samples)
            ]
            fused_mid = (
                canny_w * canny_out.mid_block_res_sample
                + depth_w * depth_out.mid_block_res_sample
            )

            noise_pred = parts["unet"](
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=fused_down,
                mid_block_additional_residual=fused_mid,
                return_dict=True,
            ).sample

            total_loss += F.mse_loss(noise_pred.float(), noise.float(),
                                     reduction="mean").item()
            total_steps += 1

            if max_batches is not None and total_steps >= max_batches:
                break

    return total_loss / max(total_steps, 1)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    print(f"Device: {cfg.device}  |  dtype: {cfg.dtype}")
    print("Loading models (no VAE / text encoder needed)...")
    parts = load_training_models(device=cfg.device, dtype=cfg.dtype)

    scheduler = DDIMScheduler.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", subfolder="scheduler"
    )

    dataset = PrecomputedDataset(pt_dir=cfg.pt_dir)
    # Use the full dataset for the ablation — or cap with max_batches below
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # Cap to first N batches to keep it fast; set to None for full eval
    max_batches = 50

    configs = [
        ("canny-only  (1.0, 0.0)", 1.0, 0.0),
        ("depth-only  (0.0, 1.0)", 0.0, 1.0),
        ("equal       (0.5, 0.5)", 0.5, 0.5),
    ]

    results: list[tuple[str, float]] = []
    for label, cw, dw in configs:
        print(f"\nRunning: {label} ...")
        loss = run_fixed_weight_eval(cw, dw, loader, parts, scheduler, cfg,
                                     max_batches=max_batches)
        results.append((label, loss))
        print(f"  MSE loss = {loss:.6f}")

    print("\n" + "─" * 45)
    print("  Ablation results")
    print("─" * 45)
    best_loss = min(l for _, l in results)
    for label, loss in results:
        marker = "  ← best" if loss == best_loss else ""
        print(f"  {label}:  {loss:.6f}{marker}")
    print("─" * 45)

    losses = {label: loss for label, loss in results}
    canny_loss = losses["canny-only  (1.0, 0.0)"]
    depth_loss = losses["depth-only  (0.0, 1.0)"]
    equal_loss = losses["equal       (0.5, 0.5)"]

    print("\nInterpretation:")
    improvement = (canny_loss - equal_loss) / canny_loss * 100
    if equal_loss < canny_loss and improvement > 0.5:
        print(f"  Equal weights beat canny-only by {improvement:.1f}% → depth IS contributing.")
    elif equal_loss > canny_loss:
        print(f"  Canny-only beats equal weights → depth adds noise for this dataset.")
    else:
        print(f"  Difference is negligible (<0.5%) → depth has minimal effect.")

    depth_vs_canny = (canny_loss - depth_loss) / canny_loss * 100
    if depth_loss < canny_loss:
        print(f"  Depth-only is {abs(depth_vs_canny):.1f}% better than canny-only.")
    else:
        print(f"  Depth-only is {abs(depth_vs_canny):.1f}% worse than canny-only.")


if __name__ == "__main__":
    main()
