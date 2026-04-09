"""
Run ONCE to pre-compute VAE latents and CLIP text embeddings.

    python precompute_latents.py

Saves into a pt/ directory (created automatically):
    pt/{stem}_latent.pt      — VAE latent   [4, 64, 64]  float16
    pt/{stem}_text_emb.pt    — CLIP hidden  [77, 768]    float16

After this script finishes you can unload the VAE and text-encoder from
training (load_training_models does exactly that), recovering ~2 GB VRAM.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

CACHE_DIR  = str(Path(__file__).parent / "models")
CSV_PATH   = "train.csv"
#for ai dataset
# PT_DIR     = Path("pt")
#for real dataset
PT_DIR     = Path("pt_flickr")
IMAGE_SIZE = 512
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.float16 if torch.cuda.is_available() else torch.float32


def main() -> None:
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Pre-computing latents + text embeddings for {len(rows)} samples…")
    print(f"Device: {DEVICE}  Dtype: {DTYPE}")

    print("Loading tokenizer…")
    tokenizer = CLIPTokenizer.from_pretrained(
        BASE_MODEL, subfolder="tokenizer", cache_dir=CACHE_DIR
    )

    print("Loading text encoder…")
    text_encoder = CLIPTextModel.from_pretrained(
        BASE_MODEL,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
        cache_dir=CACHE_DIR,
    ).to(DEVICE).eval()

    print("Loading VAE…")
    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL,
        subfolder="vae",
        torch_dtype=DTYPE,
        cache_dir=CACHE_DIR,
    ).to(DEVICE).eval()

    PT_DIR.mkdir(exist_ok=True)

    skipped = 0
    with torch.no_grad():
        for row in tqdm(rows):
            img_path   = Path(row["image_path"])
            caption    = row["caption"]
            stem       = img_path.stem
            latent_out = PT_DIR / f"{stem}_latent.pt"
            emb_out    = PT_DIR / f"{stem}_text_emb.pt"

            if latent_out.exists() and emb_out.exists():
                skipped += 1
                continue

            # ── text embedding ──────────────────────────────────────────────
            if not emb_out.exists():
                tokens = tokenizer(
                    caption,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                emb = text_encoder(tokens.input_ids.to(DEVICE))[0]  # [1, 77, 768]
                torch.save(emb.squeeze(0).cpu(), emb_out)            # [77, 768]

            # ── VAE latent ──────────────────────────────────────────────────
            if not latent_out.exists():
                img = Image.open(img_path).convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
                )
                arr = torch.from_numpy(
                    np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                ).unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
                pixel_values = arr * 2.0 - 1.0
                latent = vae.encode(pixel_values).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
                torch.save(latent.squeeze(0).cpu(), latent_out)  # [4, 64, 64]

    total = len(rows)
    print(f"Done. {total - skipped} processed, {skipped} skipped (already existed).")


if __name__ == "__main__":
    main()
