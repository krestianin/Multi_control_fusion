"""
Run ONCE before training to pre-compute canny edges and Depth Anything V2 depth maps.

    python precompute_controls.py

Saves into a pt/ directory (created automatically):
    pt/{stem}_canny.pt   — Canny edge map  [3, 512, 512] float32 in [0, 1]
    pt/{stem}_depth.pt   — Depth Anything V2 map [3, 512, 512] float32 in [0, 1]

After this script finishes you can delete depth-anything/Depth-Anything-V2-Small-hf from your cache
if you want to recover disk space — it is never needed during training.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

CACHE_DIR  = str(Path(__file__).parent / "models")
CSV_PATH   = "train.csv"
PT_DIR     = Path("pt_combined")
IMAGE_SIZE = 512


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL Image (RGB, [0,255]) → float32 tensor [3, H, W] in [0, 1]."""
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def make_canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    edges = cv2.Canny(np.array(image), low, high)
    return Image.fromarray(np.stack([edges, edges, edges], axis=-1))


def main() -> None:
    PT_DIR.mkdir(exist_ok=True)

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        image_paths = [Path(row["image_path"]) for row in csv.DictReader(f)]

    print(f"Pre-computing controls for {len(image_paths)} images…")
    depth_model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"Loading {depth_model_id} (cached in {CACHE_DIR})…")

    depth_pipe = pipeline(
        task="depth-estimation",
        model=depth_model_id,
        device=0,           # GPU only for speed; change to -1 for CPU
        model_kwargs={"cache_dir": CACHE_DIR},
    )

    skipped = 0
    for img_path in tqdm(image_paths):
        stem = img_path.stem
        canny_out = PT_DIR / f"{stem}_canny.pt"
        depth_out = PT_DIR / f"{stem}_depth.pt"

        if canny_out.exists() and depth_out.exists():
            skipped += 1
            continue

        image = Image.open(img_path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
        )

        if not canny_out.exists():
            torch.save(image_to_tensor(make_canny(image)), canny_out)

        if not depth_out.exists():
            depth_result = depth_pipe(image)
            depth_img = depth_result["depth"]
            if not isinstance(depth_img, Image.Image):
                depth_img = Image.fromarray(np.array(depth_img))
            depth_np = np.array(depth_img.convert("L").resize(image.size, Image.Resampling.BILINEAR))
            depth_rgb = Image.fromarray(np.stack([depth_np, depth_np, depth_np], axis=-1).astype(np.uint8))
            torch.save(image_to_tensor(depth_rgb), depth_out)

    print(f"Done. {len(image_paths) - skipped} images processed, {skipped} skipped (already existed).")


if __name__ == "__main__":
    main()
