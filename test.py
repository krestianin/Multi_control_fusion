from __future__ import annotations

import torch

from models import load_models
from multi_control_fusion import EqualWeightMultiControlFusion


def freeze_info(parts) -> None:
    print("\nLoaded models:")
    for name, module in parts.items():
        if name == "tokenizer":
            print(f"  {name}: tokenizer loaded")
        else:
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            print(f"  {name}: total={total:,}, trainable={trainable:,}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device}")
    print(f"Using dtype:  {dtype}")

    # Load all pretrained parts once
    parts = load_models(device=device, dtype=dtype)
    freeze_info(parts)

    # Build equal-weight fusion wrapper
    fusion = EqualWeightMultiControlFusion(
        canny_controlnet=parts["canny_controlnet"],
        depth_controlnet=parts["depth_controlnet"],
        canny_weight=0.5,
        depth_weight=0.5,
    ).to(device)

    # Fake test tensors just to verify the forward pass wiring
    # SD1.5 latent input shape is usually [B, 4, 64, 64] for 512x512 images
    batch_size = 1
    sample = torch.randn(batch_size, 4, 64, 64, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=torch.long)

    # CLIP text encoder hidden size for SD1.5 is 768, seq len usually 77
    encoder_hidden_states = torch.randn(batch_size, 77, 768, device=device, dtype=dtype)

    # Control images are usually 3-channel image-space tensors
    canny_cond = torch.randn(batch_size, 3, 512, 512, device=device, dtype=dtype)
    depth_cond = torch.randn(batch_size, 3, 512, 512, device=device, dtype=dtype)

    print("\nRunning equal-weight multi-control forward pass...")
    with torch.no_grad():
        fused = fusion(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            canny_cond=canny_cond,
            depth_cond=depth_cond,
        )

    print("\nFusion output:")
    print("Number of fused down block residuals:", len(fused.down_block_res_samples))
    for i, tensor in enumerate(fused.down_block_res_samples):
        print(f"  down[{i}] shape = {tuple(tensor.shape)}")
    print("  mid shape =", tuple(fused.mid_block_res_sample.shape))

    # Optional: verify that fused residuals can be fed into the real U-Net
    print("\nRunning one U-Net forward pass with fused residuals...")
    with torch.no_grad():
        unet_out = parts["unet"](
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=fused.down_block_res_samples,
            mid_block_additional_residual=fused.mid_block_res_sample,
            return_dict=True,
        )

    print("U-Net output sample shape:", tuple(unet_out.sample.shape))
    print("\nSuccess: your fusion module is wired correctly.")


if __name__ == "__main__":
    main()