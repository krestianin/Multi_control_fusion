# Learning Per-Layer Fusion for Multi-Control Diffusion Models

This project explores learned fusion for multi-control text-to-image diffusion. Instead of manually choosing fixed global weights for multiple ControlNet branches, the system learns how strongly to combine Canny edge control and monocular depth control at each ControlNet residual injection point.

The project is built around Stable Diffusion v1.5 with two frozen ControlNet branches: one for Canny edges and one for depth. A lightweight fusion network predicts Canny/depth mixing weights for the residuals that are injected into the frozen Stable Diffusion U-Net. The goal is to make multi-control generation more adaptive than standard fixed-weight ControlNet fusion.

## Motivation

Canny edges and depth maps provide complementary guidance:

- Canny edges preserve local contours, object boundaries, and fine structure.
- Depth maps preserve coarse layout, geometry, and spatial relationships.

In normal multi-ControlNet workflows, users usually tune global control weights by hand. This can work, but it treats all layers, prompts, timesteps, and images the same. This project tests whether a small learned module can choose better control balances automatically during the denoising process.

## Method

The pipeline uses:

- Stable Diffusion v1.5 as the frozen image-generation backbone
- ControlNet Canny for edge-based conditioning
- ControlNet Depth for depth-based conditioning
- A frozen VAE for latent encoding/decoding
- A lightweight fusion MLP for combining ControlNet residuals
- A diffusion denoising MSE objective for training the fusion module

During training, the Stable Diffusion U-Net and both ControlNets remain frozen. Only the fusion network is trained. For each image, the model samples a timestep, adds Gaussian noise to the latent, runs both ControlNet branches, fuses their residual outputs, and sends the fused residuals into the U-Net. The loss is the standard denoising MSE between predicted noise and sampled noise.

The fusion module predicts one Canny/depth weight pair for every ControlNet injection point, including the down-block residuals and the mid-block residual.

## Architecture

The code loads Stable Diffusion v1.5, ControlNet-Canny, ControlNet-Depth, the VAE, tokenizer, and text encoder through Hugging Face Diffusers, then freezes the pretrained modules so only the fusion component is optimized.

## Training Data

The final experiment used a combined dataset of approximately 20,000 image-caption pairs:

1) 10,000 samples from DiffusionDB
2) 10,000 samples from Flickr30k

Latents, Canny tensors, depth tensors, and text embeddings were precomputed and loaded from .pt files during training to reduce repeated preprocessing overhead.

Expected tensor files per sample:

- sample_canny.pt
- sample_depth.pt
- sample_latent.pt
- sample_text_emb.pt

## Results

The learned fusion model was most useful in visually ambiguous cases where one control signal alone was incomplete. Examples include:

- shadows
- mirrors
- glass or transparent objects
- weak-edge regions
- low-contrast boundaries

In simpler scenes with clear edges and depth layout, learned fusion was usually comparable to Canny-only or Depth-only baselines. This suggests that adaptive fusion can improve difficult cases without strongly harming easier ones.

The project also showed a compositional use case: using depth from one image and Canny edges from another image to guide a new generated output. This works best when the source images are already roughly spatially aligned.

## Report

The full project report is included in:

    docs/Final_Report.pdf


## Setup
Setup the environment
1. python -m venv .venv
2. source .venv/bin/activate ( .venv\\Scripts\\activate on Windows)
3. 
    pip install --upgrade pip
    pip install torch torchvision torchaudio
    pip install diffusers transformers accelerate safetensors huggingface_hub pillow opencv-python

## Model Storage
Models are stored locally in the `models/` folder in this project directory (on E: drive).
This avoids filling up your C: drive cache. The first run will download models (~10GB) to this folder.

## Clear cache
    Remove-Item -Path "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -Force

## Clear local models
    Remove-Item -Path "models" -Recurse -Force


## Datasets
1. Place this dataset in your data/ folder (we took 10k from this): 
    - https://huggingface.co/datasets/poloclub/diffusiondb 
    - Use the instruction and the code provided on their website
2. Download the following dataset (we took 10k from here as well):
    - https://huggingface.co/datasets/nlphuji/flickr30k
    - This should create two folders - flickr30k-images/ and flickr30k-descriptions/
    - Use the instruction and the code provided on their website
3. All folder (data, and both flicker ones should be in the root directory)
4. To produce csv files, run: 
    - prepare_dataset.py

2. Then run the following to produce final data folders used for training:

    - python precompute_controls.py   # canny + depth (if not done yet)
    - python precompute_latents.py    # latents + text embeddings

## Training
1. Adjust hyperparameters as needed (use comments in TrainConfig class) in the train_fusion_mlp.py
2. Run python train_fusion_mlp.py

## Inferencing
1. Adjust user settings in main of the inference.py
2. Run python inference.py

## What Is Not Included

This repository does not include:

Stable Diffusion model weights, 
ControlNet model weights, 
DiffusionDB or Flickr30k datasets, 
Large .pt precomputed tensor datasets, 
Large training checkpoints

These files should be downloaded or generated locally.