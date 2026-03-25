# models.py

from __future__ import annotations

import torch
from pathlib import Path
from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer


def freeze_module(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def load_models(device: str = "cuda", dtype: torch.dtype = torch.float16, cache_dir: str | None = None):
    # Use local models folder in project if not specified
    if cache_dir is None:
        cache_dir = str(Path(__file__).parent / "models")
        Path(cache_dir).mkdir(exist_ok=True)
        print(f"[DEBUG] Using local cache directory: {cache_dir}")
    
    base_model = "runwayml/stable-diffusion-v1-5"
    canny_model = "lllyasviel/sd-controlnet-canny"
    depth_model = "lllyasviel/sd-controlnet-depth"

    print(f"[DEBUG] Loading Canny ControlNet from {canny_model}...")
    canny_controlnet = ControlNetModel.from_pretrained(
        canny_model,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    print("[DEBUG] Canny ControlNet downloaded, moving to device...")
    canny_controlnet = canny_controlnet.to(device)
    print("[DEBUG] Canny ControlNet loaded and on device")

    print(f"[DEBUG] Loading Depth ControlNet from {depth_model}...")
    depth_controlnet = ControlNetModel.from_pretrained(
        depth_model,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    print("[DEBUG] Depth ControlNet downloaded, moving to device...")
    depth_controlnet = depth_controlnet.to(device)
    print("[DEBUG] Depth ControlNet loaded and on device")

    print(f"[DEBUG] Loading UNet from {base_model}...")
    unet = UNet2DConditionModel.from_pretrained(
        base_model,
        subfolder="unet",
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    print("[DEBUG] UNet downloaded, moving to device...")
    unet = unet.to(device)
    print("[DEBUG] UNet loaded and on device")

    print(f"[DEBUG] Loading VAE from {base_model}...")
    vae = AutoencoderKL.from_pretrained(
        base_model,
        subfolder="vae",
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    print("[DEBUG] VAE downloaded, moving to device...")
    vae = vae.to(device)
    print("[DEBUG] VAE loaded and on device")

    print(f"[DEBUG] Loading Text Encoder from {base_model}...")
    text_encoder = CLIPTextModel.from_pretrained(
        base_model,
        subfolder="text_encoder",
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    print("[DEBUG] Text Encoder downloaded, moving to device...")
    text_encoder = text_encoder.to(device)
    print("[DEBUG] Text Encoder loaded and on device")

    print(f"[DEBUG] Loading Tokenizer from {base_model}...")
    tokenizer = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer",
        cache_dir=cache_dir,
    )
    print("[DEBUG] Tokenizer loaded")

    # freeze all pretrained parts
    print("[DEBUG] Freezing modules...")
    freeze_module(canny_controlnet)
    freeze_module(depth_controlnet)
    freeze_module(unet)
    freeze_module(vae)
    freeze_module(text_encoder)
    print("[DEBUG] All modules frozen")

    return {
        "canny_controlnet": canny_controlnet,
        "depth_controlnet": depth_controlnet,
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
    }


def load_training_models(device: str = "cuda", dtype: torch.dtype = torch.float16, cache_dir: str | None = None):
    """Like load_models but skips VAE and text encoder.

    Use this when latents and text embeddings have already been pre-computed
    with precompute_latents.py — saves ~2 GB of VRAM during training.
    """
    if cache_dir is None:
        cache_dir = str(Path(__file__).parent / "models")
        Path(cache_dir).mkdir(exist_ok=True)

    base_model = "runwayml/stable-diffusion-v1-5"
    canny_model = "lllyasviel/sd-controlnet-canny"
    depth_model = "lllyasviel/sd-controlnet-depth"

    print("[DEBUG] Loading Canny ControlNet…")
    canny_controlnet = ControlNetModel.from_pretrained(
        canny_model, torch_dtype=dtype, cache_dir=cache_dir
    ).to(device)

    print("[DEBUG] Loading Depth ControlNet…")
    depth_controlnet = ControlNetModel.from_pretrained(
        depth_model, torch_dtype=dtype, cache_dir=cache_dir
    ).to(device)

    print("[DEBUG] Loading UNet…")
    unet = UNet2DConditionModel.from_pretrained(
        base_model, subfolder="unet", torch_dtype=dtype, cache_dir=cache_dir
    ).to(device)

    freeze_module(canny_controlnet)
    freeze_module(depth_controlnet)
    freeze_module(unet)
    print("[DEBUG] All training modules loaded and frozen.")

    return {
        "canny_controlnet": canny_controlnet,
        "depth_controlnet": depth_controlnet,
        "unet": unet,
    }