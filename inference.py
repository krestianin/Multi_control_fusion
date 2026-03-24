from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler
from transformers import pipeline

from models import load_models
from multi_control_fusion import LearnedWeightMultiControlFusion


def load_rgb_image(image_path: str, size: int = 512) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((size, size))
    return image


def make_canny_control(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    np_img = np.array(image)
    edges = cv2.Canny(np_img, low_threshold, high_threshold)
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


def pil_to_tensor_01(image: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Converts PIL image to tensor in [0, 1], shape [1, 3, H, W]
    """
    arr = np.array(image).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=dtype)
    return tensor


def encode_prompt(prompt: str, tokenizer, text_encoder, device: str, do_cfg: bool = True):
    """
    Returns encoder_hidden_states.
    If do_cfg=True, returns concatenated unconditional + conditional embeddings.
    """
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        cond_embeds = text_encoder(input_ids)[0]

    if not do_cfg:
        return cond_embeds

    uncond_inputs = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_ids = uncond_inputs.input_ids.to(device)

    with torch.no_grad():
        uncond_embeds = text_encoder(uncond_ids)[0]

    return torch.cat([uncond_embeds, cond_embeds], dim=0)


def decode_latents(latents: torch.Tensor, vae) -> Image.Image:
    """
    Decode final latents to PIL image.
    """
    with torch.no_grad():
        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).round().astype(np.uint8)
    return Image.fromarray(image)


def main():
    # -----------------------------
    # User settings
    # -----------------------------
    image_path = "village.png"   # replace with your control/source image
    prompt = "a realistic cinematic street scene"
    output_path = "generated_learned_fusion.png"
    fusion_mlp_path = "fusion_mlp_ckpts/fusion_mlp_best.pth"  # set to None to fall back to fixed weights

    num_inference_steps = 30
    guidance_scale = 7.5
    height = 512
    width = 512

    # -----------------------------
    # Device / dtype
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device}")
    print(f"Using dtype:  {dtype}")

    # -----------------------------
    # Load base models
    # -----------------------------
    parts = load_models(device=device, dtype=dtype)

    fusion = LearnedWeightMultiControlFusion(
        canny_controlnet=parts["canny_controlnet"],
        depth_controlnet=parts["depth_controlnet"],
        fusion_mlp_path=fusion_mlp_path,
        map_location=device,
        temperature=1.0,
        fallback_canny_weight=0.5,
        fallback_depth_weight=0.5,
        validate_shapes_once=True,
    ).to(device)

    print(
        "Fusion mode:",
        "learned MLP" if fusion.has_learned_fusion() else "fixed fallback weights",
    )

    # -----------------------------
    # Load scheduler
    # -----------------------------
    scheduler = DDIMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler",
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    # -----------------------------
    # Load image and build controls
    # -----------------------------
    image = load_rgb_image(image_path, size=512)

    print("Computing Canny control...")
    canny_image = make_canny_control(image)

    print("Computing Depth control...")
    depth_pipe = pipeline(
        task="depth-estimation",
        model="Intel/dpt-large",
        device=0 if device == "cuda" else -1,
    )
    depth_image = make_depth_control(image, depth_pipe)

    canny_cond = pil_to_tensor_01(canny_image, device=device, dtype=dtype)
    depth_cond = pil_to_tensor_01(depth_image, device=device, dtype=dtype)

    # classifier-free guidance duplicates the batch
    canny_cond = torch.cat([canny_cond, canny_cond], dim=0)
    depth_cond = torch.cat([depth_cond, depth_cond], dim=0)

    # -----------------------------
    # Encode prompt
    # -----------------------------
    encoder_hidden_states = encode_prompt(
        prompt=prompt,
        tokenizer=parts["tokenizer"],
        text_encoder=parts["text_encoder"],
        device=device,
        do_cfg=True,
    )

    # -----------------------------
    # Initialize random latents
    # -----------------------------
    latent_h = height // 8
    latent_w = width // 8
    latents = torch.randn((1, 4, latent_h, latent_w), device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    # -----------------------------
    # Denoising loop
    # -----------------------------
    print("Running denoising loop...")
    for step_idx, t in enumerate(scheduler.timesteps):
        latent_model_input = torch.cat([latents, latents], dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            fused = fusion(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                canny_cond=canny_cond,
                depth_cond=depth_cond,
            )

            noise_pred = parts["unet"](
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=fused.down_block_res_samples,
                mid_block_additional_residual=fused.mid_block_res_sample,
                return_dict=True,
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if step_idx == 0 and fused.fusion_weights is not None:
            weights_cpu = fused.fusion_weights.detach().float().cpu()
            print("Learned per-layer fusion weights:")
            for j, pair in enumerate(weights_cpu):
                print(f"  layer {j:02d}: canny={pair[0]:.4f}, depth={pair[1]:.4f}")

        print(f"Step {step_idx + 1}/{num_inference_steps} done")

    # -----------------------------
    # Decode final latents
    # -----------------------------
    print("Decoding final image...")
    result = decode_latents(latents, parts["vae"])
    result.save(output_path)

    canny_image.save("debug_canny.png")
    depth_image.save("debug_depth.png")

    print(f"Saved generated image to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
