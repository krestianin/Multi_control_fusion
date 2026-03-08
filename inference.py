from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from transformers import pipeline

from models import load_models
from multi_control_fusion import EqualWeightMultiControlFusion


def load_rgb_image(image_path: str, size: int = 512) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((size, size))
    return image


def make_canny_control(image: Image.Image) -> Image.Image:
    np_img = np.array(image)
    edges = cv2.Canny(np_img, 100, 200)
    edges_3ch = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_3ch)


def make_depth_control(image: Image.Image, depth_pipe) -> Image.Image:
    depth_result = depth_pipe(image)
    depth_img = depth_result["depth"]

    if not isinstance(depth_img, Image.Image):
        depth_img = Image.fromarray(np.array(depth_img))

    depth_img = depth_img.convert("L")
    depth_img = depth_img.resize(image.size)
    depth_np = np.array(depth_img)
    depth_3ch = np.stack([depth_np, depth_np, depth_np], axis=-1)
    return Image.fromarray(depth_3ch.astype(np.uint8))


def pil_to_tensor(image: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=dtype)
    return tensor


def encode_prompt(prompt: str, tokenizer, text_encoder, device: str):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids)[0]

    return encoder_hidden_states


def encode_image_to_latent(image: Image.Image, vae, device: str, dtype: torch.dtype) -> torch.Tensor:
    img = np.array(image).astype(np.float32) / 255.0
    img = (img * 2.0) - 1.0  # [0,1] -> [-1,1]
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    return latents


def main():
    image_path = "input.jpg"   # replace with your image
    prompt = "a realistic street scene"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    parts = load_models(device=device, dtype=dtype)

    fusion = EqualWeightMultiControlFusion(
        canny_controlnet=parts["canny_controlnet"],
        depth_controlnet=parts["depth_controlnet"],
        canny_weight=0.5,
        depth_weight=0.5,
    ).to(device)

    # Load real input image
    image = load_rgb_image(image_path, size=512)

    # Build real control maps
    canny_image = make_canny_control(image)

    depth_pipe = pipeline(
        task="depth-estimation",
        model="Intel/dpt-large",
        device=0 if device == "cuda" else -1,
    )
    depth_image = make_depth_control(image, depth_pipe)

    # Convert control maps to tensors
    canny_cond = pil_to_tensor(canny_image, device=device, dtype=dtype)
    depth_cond = pil_to_tensor(depth_image, device=device, dtype=dtype)

    # Real prompt encoding
    encoder_hidden_states = encode_prompt(
        prompt,
        tokenizer=parts["tokenizer"],
        text_encoder=parts["text_encoder"],
        device=device,
    )

    # Real latent from real image
    latents = encode_image_to_latent(
        image=image,
        vae=parts["vae"],
        device=device,
        dtype=dtype,
    )

    # Add one diffusion step of noise
    noise = torch.randn_like(latents)
    timestep = torch.tensor([500], device=device, dtype=torch.long)
    alpha = 0.5
    noisy_latents = alpha * latents + (1 - alpha) * noise

    # Run your fusion module
    with torch.no_grad():
        fused = fusion(
            sample=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            canny_cond=canny_cond,
            depth_cond=depth_cond,
        )

    print("Number of fused down residuals:", len(fused.down_block_res_samples))
    print("Mid residual shape:", fused.mid_block_res_sample.shape)

    # Run real U-Net pass
    with torch.no_grad():
        unet_out = parts["unet"](
            sample=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=fused.down_block_res_samples,
            mid_block_additional_residual=fused.mid_block_res_sample,
            return_dict=True,
        )

    print("U-Net output shape:", unet_out.sample.shape)
    print("Real single-step forward pass completed successfully.")


if __name__ == "__main__":
    main()