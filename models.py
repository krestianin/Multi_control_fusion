# models.py

from __future__ import annotations

import torch
from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer


def freeze_module(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def load_models(device: str = "cuda", dtype: torch.dtype = torch.float16):
    base_model = "runwayml/stable-diffusion-v1-5"
    canny_model = "lllyasviel/sd-controlnet-canny"
    depth_model = "lllyasviel/sd-controlnet-depth"

    canny_controlnet = ControlNetModel.from_pretrained(
        canny_model,
        torch_dtype=dtype,
    ).to(device)

    depth_controlnet = ControlNetModel.from_pretrained(
        depth_model,
        torch_dtype=dtype,
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        base_model,
        subfolder="unet",
        torch_dtype=dtype,
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        base_model,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)

    text_encoder = CLIPTextModel.from_pretrained(
        base_model,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer",
    )

    # freeze all pretrained parts
    freeze_module(canny_controlnet)
    freeze_module(depth_controlnet)
    freeze_module(unet)
    freeze_module(vae)
    freeze_module(text_encoder)

    return {
        "canny_controlnet": canny_controlnet,
        "depth_controlnet": depth_controlnet,
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
    }