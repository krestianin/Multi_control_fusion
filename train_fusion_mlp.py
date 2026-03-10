from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import pipeline

from fusion_mlp import PerLayerFusionMLP
from models import load_models


# -----------------------------
# Simple dataset
# -----------------------------
class ImageCaptionDataset(Dataset):
    """
    CSV format:
        image_path,caption

    Keep it simple first. You can swap this for a Hugging Face dataset later.
    """

    def __init__(self, csv_path: str | Path, image_size: int = 512) -> None:
        self.image_size = image_size
        self.samples: list[tuple[str, str]] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image_path"], row["caption"]))

        if not self.samples:
            raise ValueError("Dataset is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return image, caption


def collate_pil(batch):
    images, captions = zip(*batch)
    return list(images), list(captions)


# -----------------------------
# Controls / text / image utils
# -----------------------------
def make_canny_control(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
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
    depth_img = depth_img.resize(image.size, Image.Resampling.BILINEAR)
    depth_np = np.array(depth_img)
    depth_3ch = np.stack([depth_np, depth_np, depth_np], axis=-1)
    return Image.fromarray(depth_3ch.astype(np.uint8))


def pil_to_tensor_01(images: Sequence[Image.Image], device: str, dtype: torch.dtype) -> torch.Tensor:
    arrs = []
    for image in images:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arrs.append(arr)
    tensor = torch.from_numpy(np.stack(arrs, axis=0)).to(device=device, dtype=dtype)
    return tensor


def pil_to_latent_tensor(images: Sequence[Image.Image], device: str, dtype: torch.dtype) -> torch.Tensor:
    arrs = []
    for image in images:
        arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        arrs.append(arr)
    return torch.from_numpy(np.stack(arrs, axis=0)).to(device=device, dtype=dtype)


def encode_prompt_batch(
    prompts: Sequence[str],
    tokenizer,
    text_encoder,
    device: str,
) -> torch.Tensor:
    text_inputs = tokenizer(
        list(prompts),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    return text_encoder(input_ids)[0]


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
    train_csv: str = "train.csv"
    output_dir: str = "fusion_mlp_ckpts"
    image_size: int = 512
    batch_size: int = 1
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_inference_train_timesteps: int = 1000
    max_grad_norm: float = 1.0
    save_every_steps: int = 200
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

    parts = load_models(device=cfg.device, dtype=cfg.dtype)
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

    optimizer = torch.optim.AdamW(
        fusion_mlp.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = DDIMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler",
    )
    scheduler.set_timesteps(cfg.num_inference_train_timesteps, device=cfg.device)

    dataset = ImageCaptionDataset(cfg.train_csv, image_size=cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pil,
    )

    depth_pipe = pipeline(
        task="depth-estimation",
        model="Intel/dpt-large",
        device=0 if cfg.device == "cuda" else -1,
    )

    global_step = 0
    best_loss = float("inf")

    for epoch in range(cfg.epochs):
        fusion_mlp.train()

        for images, captions in loader:
            batch_size = len(images)

            # Build controls on the fly for now.
            canny_images = [make_canny_control(img) for img in images]
            depth_images = [make_depth_control(img, depth_pipe) for img in images]

            pixel_values = pil_to_latent_tensor(images, device=cfg.device, dtype=cfg.dtype)
            canny_cond = pil_to_tensor_01(canny_images, device=cfg.device, dtype=cfg.dtype)
            depth_cond = pil_to_tensor_01(depth_images, device=cfg.device, dtype=cfg.dtype)

            encoder_hidden_states = encode_prompt_batch(
                prompts=captions,
                tokenizer=parts["tokenizer"],
                text_encoder=parts["text_encoder"],
                device=cfg.device,
            )

            # Encode image to latent z0.
            with torch.no_grad():
                latents = parts["vae"].encode(pixel_values).latent_dist.sample()
                latents = latents * parts["vae"].config.scaling_factor

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

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_mlp.parameters(), cfg.max_grad_norm)
            optimizer.step()

            global_step += 1

            print(
                f"epoch={epoch + 1}/{cfg.epochs} "
                f"step={global_step} "
                f"loss={loss.item():.6f}"
            )

            if loss.item() < best_loss:
                best_loss = loss.item()
                fusion_mlp.save(output_dir / "fusion_mlp_best.pth")

            if global_step % cfg.save_every_steps == 0:
                fusion_mlp.save(output_dir / f"fusion_mlp_step_{global_step}.pth")

        fusion_mlp.save(output_dir / f"fusion_mlp_epoch_{epoch + 1}.pth")

    fusion_mlp.save(output_dir / "fusion_mlp_last.pth")

    print("Training finished.")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Saved checkpoints to: {output_dir}")
    print("Final learned weights:")
    fusion_mlp.pretty_print()


if __name__ == "__main__":
    train(TrainConfig())
