from __future__ import annotations

import random
import time

import numpy as np
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fusion_mlp import ContextEncoder, PerLayerFusionMLP
from models import load_training_models


# -----------------------------
# Dataset (pre-computed tensors)
# -----------------------------
class PrecomputedDataset(Dataset):
    """
    Scans pt_dir for all *_latent.pt files and expects matching:
        pt/{stem}_canny.pt
        pt/{stem}_depth.pt
        pt/{stem}_latent.pt
        pt/{stem}_text_emb.pt
    No CSV needed.
    """

    def __init__(self, pt_dir: str | Path = "pt") -> None:
        pt_dir = Path(pt_dir)
        self.samples: list[tuple[str, Path, Path, Path, Path]] = []

        for latent_path in sorted(pt_dir.glob("*_latent.pt")):
            stem = latent_path.stem[: -len("_latent")]
            canny    = pt_dir / f"{stem}_canny.pt"
            depth    = pt_dir / f"{stem}_depth.pt"
            text_emb = pt_dir / f"{stem}_text_emb.pt"
            missing  = [p for p in (canny, depth, text_emb) if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing for stem '{stem}': {missing}")
            self.samples.append((stem, canny, depth, latent_path, text_emb))

        if not self.samples:
            n_canny = sum(1 for _ in pt_dir.glob("*_canny.pt"))
            n_depth = sum(1 for _ in pt_dir.glob("*_depth.pt"))
            n_text  = sum(1 for _ in pt_dir.glob("*_text_emb.pt"))
            raise ValueError(
                f"No *_latent.pt files found in {pt_dir}. "
                f"Found canny={n_canny}, depth={n_depth}, text_emb={n_text}. "
                "Run preprocessing first: "
                "`python prepare_dataset.py` (if train.csv is not ready), then "
                "`python precompute_latents.py`."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        stem, canny_path, depth_path, latent_path, emb_path = self.samples[idx]
        return (
            stem,
            torch.load(canny_path, weights_only=True),    # [3, H, W]
            torch.load(depth_path, weights_only=True),    # [3, H, W]
            torch.load(latent_path, weights_only=True),   # [4, 64, 64]
            torch.load(emb_path, weights_only=True),      # [77, 768]
        )


def collate_fn(batch):
    stems, cannys, depths, latents, text_embs = zip(*batch)
    return list(stems), torch.stack(cannys), torch.stack(depths), torch.stack(latents), torch.stack(text_embs)


# -----------------------------
# Fusion helpers
# -----------------------------
@dataclass
class FusedControlOutput:
    down_block_res_samples: tuple[torch.Tensor, ...]
    mid_block_res_sample: torch.Tensor
    fusion_weights: torch.Tensor | None = None


class LearnedPerLayerFusion(nn.Module):
    """
    Small helper that uses the frozen ControlNets and the trainable MLP.

    When a ContextEncoder is provided the MLP receives a per-sample context
    vector built from:
        - sinusoidal timestep embedding
        - mean-pooled CLIP text embeddings
        - globally-pooled first down-block features from the canny ControlNet
        - globally-pooled first down-block features from the depth ControlNet

    This makes the fusion weights image- and prompt-dependent rather than
    static per-layer scalars, turning the model into a real conditional router.

    ControlNet outputs are always detached (frozen backbones); the fused
    tensors still carry gradients to the MLP / context encoder.
    """

    def __init__(
        self,
        canny_controlnet: nn.Module,
        depth_controlnet: nn.Module,
        fusion_mlp: PerLayerFusionMLP,
        context_encoder: ContextEncoder | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.canny_controlnet = canny_controlnet
        self.depth_controlnet = depth_controlnet
        self.fusion_mlp = fusion_mlp
        self.context_encoder = context_encoder  # None → static (original) mode
        self.temperature = float(temperature)

    @staticmethod
    def _get_weight_maps(
        weights: torch.Tensor,
        layer_idx: int,
        batch_size: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert scalar or grid weights into spatial maps [B, 1, H, W].

        Supported input layouts:
          - [J, 2]
          - [B, J, 2]
          - [J, 2, Gh, Gw]
          - [B, J, 2, Gh, Gw]
        """
        if weights.dim() == 5:  # [B, J, 2, Gh, Gw]
            canny_w = weights[:, layer_idx, 0]  # [B, Gh, Gw]
            depth_w = weights[:, layer_idx, 1]  # [B, Gh, Gw]
        elif weights.dim() == 4:  # [J, 2, Gh, Gw]
            canny_w = weights[layer_idx, 0].unsqueeze(0).expand(batch_size, -1, -1)  # [B, Gh, Gw]
            depth_w = weights[layer_idx, 1].unsqueeze(0).expand(batch_size, -1, -1)  # [B, Gh, Gw]
        elif weights.dim() == 3:  # [B, J, 2]
            canny_w = weights[:, layer_idx, 0].view(batch_size, 1, 1).expand(batch_size, height, width)  # [B, H, W]
            depth_w = weights[:, layer_idx, 1].view(batch_size, 1, 1).expand(batch_size, height, width)  # [B, H, W]
        elif weights.dim() == 2:  # [J, 2]
            canny_w = weights[layer_idx, 0].view(1, 1, 1).expand(batch_size, height, width)  # [B, H, W]
            depth_w = weights[layer_idx, 1].view(1, 1, 1).expand(batch_size, height, width)  # [B, H, W]
        else:
            raise ValueError(f"Unsupported fusion weight shape: {tuple(weights.shape)}")

        if canny_w.shape[-2:] != (height, width):
            canny_w = F.interpolate(canny_w.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
            depth_w = F.interpolate(depth_w.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)

        return canny_w.unsqueeze(1), depth_w.unsqueeze(1)  # [B, 1, H, W]

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

        # ── Build context and get weights ────────────────────────────────────
        context = None
        if self.context_encoder is not None:
            B = sample.shape[0]

            # Globally pool the first (shallowest) ControlNet down-block feature
            # shape: [B, C, H, W] → [B, C]
            canny_feat = canny_out.down_block_res_samples[0].detach().float().mean(dim=(-2, -1))
            depth_feat = depth_out.down_block_res_samples[0].detach().float().mean(dim=(-2, -1))

            # Ensure timestep is [B]
            t = timestep
            if isinstance(t, (int, float)):
                t = torch.tensor([t], device=sample.device, dtype=torch.long).expand(B)
            elif not isinstance(t, torch.Tensor):
                t = torch.tensor([t], device=sample.device, dtype=torch.long).expand(B)
            elif t.dim() == 0:
                t = t.unsqueeze(0).expand(B)
            elif t.shape[0] == 1 and B > 1:
                t = t.expand(B)

            context = self.context_encoder(
                timestep=t,
                text_emb=encoder_hidden_states,
                canny_feat=canny_feat,
                depth_feat=depth_feat,
            )  # [B, context_dim]

        if context is not None:
            weights = self.fusion_mlp(context=context, temperature=self.temperature)  # [B, J, 2, Gh, Gw]
        else:
            weights = self.fusion_mlp.get_all_fusion_weights(temperature=self.temperature)  # [J, 2, Gh, Gw]

        if weights.dim() in (5, 3):
            num_weights = weights.shape[1]
        elif weights.dim() in (4, 2):
            num_weights = weights.shape[0]
        else:
            raise ValueError(f"Unsupported fusion weight shape: {tuple(weights.shape)}")

        expected_points = num_down + 1
        if num_weights != expected_points:
            raise ValueError(
                f"Fusion MLP expects {num_weights} injection points, "
                f"but ControlNet produced {expected_points}"
            )

        # ── Apply fusion weights ─────────────────────────────────────────────
        B = sample.shape[0]

        fused_down = []
        for j, (c_res, d_res) in enumerate(zip(
            canny_out.down_block_res_samples, depth_out.down_block_res_samples
        )):
            alpha_c, alpha_d = self._get_weight_maps(weights, j, B, c_res.shape[-2], c_res.shape[-1])
            fused_down.append(alpha_c.to(c_res.dtype) * c_res.detach() + alpha_d.to(d_res.dtype) * d_res.detach())

        mid_c = canny_out.mid_block_res_sample
        mid_d = depth_out.mid_block_res_sample
        alpha_c_mid, alpha_d_mid = self._get_weight_maps(weights, num_down, B, mid_c.shape[-2], mid_c.shape[-1])
        fused_mid = alpha_c_mid.to(mid_c.dtype) * mid_c.detach() + alpha_d_mid.to(mid_d.dtype) * mid_d.detach()

        return FusedControlOutput(
            down_block_res_samples=tuple(fused_down),
            mid_block_res_sample=fused_mid,
            fusion_weights=weights,
        )


# -----------------------------
# Train config
# -----------------------------
@dataclass
class TrainConfig:
    output_dir: str = "fusion_mlp_ckpts"
    batch_size: int = 8          # 32 will OOM; use 1 for <8 GB VRAM, 2-4 for 12-16 GB
    gradient_accumulation_steps: int = 1   # effective batch = batch_size × accum_steps
    epochs: int = 2
    lr: float = 1e-3             # ok since only the tiny fusion MLP is trained
    weight_decay: float = 1e-4
    # for ai dataset
    # pt_dir: str = "pt"
    # for real dataset
    pt_dir: str = "pt_flickr"
    # num_inference_train_timesteps: int = 1000   # only 1 random t sampled per step — cost is fine
    max_grad_norm: float = 1.0
    # save_every_steps: int = 500   # counted in optimizer steps (post-accumulation)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    max_batches: int | None = None  # set to None to disable; 50 for a quick profiling run
    fusion_temperature: float = 1.0
    fusion_weight_grid_h: int = 8
    fusion_weight_grid_w: int = 8
    deterministic_eval: bool = True
    eval_seed: int = 1234
    eval_timestep: int = 500
    log_sample_weights: bool = True


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


# -----------------------------
# Checkpoint helpers
# -----------------------------

def save_checkpoint(
    path: str | Path,
    fusion_mlp: PerLayerFusionMLP,
    context_encoder: ContextEncoder | None,
) -> None:
    """Save fusion_mlp + optional context_encoder into a single file."""
    payload: dict = {
        "fusion_mlp_state_dict": fusion_mlp.state_dict(),
        "fusion_mlp_config": {
            "num_injection_points": fusion_mlp.num_injection_points,
            "index_emb_dim":        fusion_mlp.index_emb_dim,
            "hidden_dim":           fusion_mlp.hidden_dim,
            "num_hidden_layers":    fusion_mlp.num_hidden_layers,
            "dropout":              fusion_mlp.dropout,
            "context_dim":          fusion_mlp.context_dim,
            "weight_grid_h":        fusion_mlp.weight_grid_h,
            "weight_grid_w":        fusion_mlp.weight_grid_w,
        },
    }
    if context_encoder is not None:
        payload["context_encoder_state_dict"] = context_encoder.state_dict()
        payload["context_encoder_config"]     = context_encoder.config
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[PerLayerFusionMLP, ContextEncoder | None]:
    """Load and return (fusion_mlp, context_encoder_or_None)."""
    payload = torch.load(Path(path), map_location=map_location)
    fusion_cfg = dict(payload["fusion_mlp_config"])
    # Old checkpoints saved before spatial grids existed.
    fusion_cfg.setdefault("weight_grid_h", 1)
    fusion_cfg.setdefault("weight_grid_w", 1)
    fusion_mlp = PerLayerFusionMLP(**fusion_cfg)
    fusion_mlp.load_state_dict(payload["fusion_mlp_state_dict"])

    context_encoder = None
    if "context_encoder_config" in payload:
        context_encoder = ContextEncoder(**payload["context_encoder_config"])
        context_encoder.load_state_dict(payload["context_encoder_state_dict"])

    return fusion_mlp, context_encoder


# -----------------------------
# Evaluation
# -----------------------------

def evaluate(fusion: LearnedPerLayerFusion, fusion_mlp: PerLayerFusionMLP,
             context_encoder: ContextEncoder | None,
             loader: DataLoader, parts: dict, scheduler: DDIMScheduler,
             cfg: TrainConfig) -> float:
    """Run one pass over loader with no gradient updates and return average MSE loss."""
    fusion_mlp.eval()
    if context_encoder is not None:
        context_encoder.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for stems, canny_cond, depth_cond, latents, text_embs in loader:
            canny_cond            = canny_cond.to(device=cfg.device, dtype=cfg.dtype)
            depth_cond            = depth_cond.to(device=cfg.device, dtype=cfg.dtype)
            latents               = latents.to(device=cfg.device, dtype=cfg.dtype)
            encoder_hidden_states = text_embs.to(device=cfg.device, dtype=cfg.dtype)

            if cfg.deterministic_eval:
                generator = torch.Generator(device=cfg.device)
                generator.manual_seed(cfg.eval_seed + total_steps)
                noise = torch.randn(
                    latents.shape,
                    generator=generator,
                    device=cfg.device,
                    dtype=cfg.dtype,
                )
                timesteps = torch.full(
                    (latents.shape[0],),
                    fill_value=cfg.eval_timestep,
                    device=cfg.device,
                    dtype=torch.long,
                )
            else:
                noise = torch.randn_like(latents)
                timesteps = torch.full(
                    (latents.shape[0],),
                    fill_value=cfg.eval_timestep,
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

        
            total_loss += F.mse_loss(noise_pred.float(), noise.float(), reduction="mean").item()
            total_steps += 1

            print(f"eval_step={total_steps} stem={stems[0]} timestep={timesteps[0].item()}")

            # if cfg.log_sample_weights and fused.fusion_weights is not None:
            #     weights_to_print = fused.fusion_weights[0] if fused.fusion_weights.dim() == 3 else fused.fusion_weights
                # print("sample-conditioned fusion weights:")
                # for j, pair in enumerate(weights_to_print.detach().cpu()):
                #     canny_w, depth_w = pair.tolist()
                #     print(f"layer {j:02d}: canny={canny_w:.4f}, depth={depth_w:.4f}")


            if cfg.max_batches is not None and total_steps >= cfg.max_batches:
                break

    fusion_mlp.train()
    if context_encoder is not None:
        context_encoder.train()
    return total_loss / max(total_steps, 1)


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parts = load_training_models(device=cfg.device, dtype=cfg.dtype)

    num_points = discover_num_injection_points(parts, cfg.device, cfg.dtype)

    # ContextEncoder: 4×64 = 256-dim context from timestep, text, canny, depth
    context_encoder = ContextEncoder(
        text_dim=768,
        canny_feat_dim=320,  # first down-block channels for SD v1.5 ControlNet
        depth_feat_dim=320,
        ts_emb_dim=256,
        proj_dim=64,
    ).to(device=cfg.device, dtype=cfg.dtype)

    fusion_mlp = PerLayerFusionMLP(
        num_injection_points=num_points,
        index_emb_dim=32,
        hidden_dim=64,
        num_hidden_layers=2,
        dropout=0.0,
        context_dim=context_encoder.context_dim,  # 256
        weight_grid_h=cfg.fusion_weight_grid_h,
        weight_grid_w=cfg.fusion_weight_grid_w,
    ).to(cfg.device)

    fusion = LearnedPerLayerFusion(
        canny_controlnet=parts["canny_controlnet"],
        depth_controlnet=parts["depth_controlnet"],
        fusion_mlp=fusion_mlp,
        context_encoder=context_encoder,
        temperature=cfg.fusion_temperature,
    ).to(cfg.device)

    print("Initial weights (before any training):")
    fusion_mlp.pretty_print(temperature=cfg.fusion_temperature)

    scheduler = DDIMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler",
    )

    dataset = PrecomputedDataset(pt_dir=cfg.pt_dir)
    n_eval = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_eval
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_eval],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    print(f"Dataset split: {n_train} train / {n_eval} eval samples")
    print(f"Eval timestep is fixed at t={cfg.eval_timestep}\n")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    baseline_loss = evaluate(fusion, fusion_mlp, context_encoder, eval_loader, parts, scheduler, cfg)
    print(f"Baseline loss (0.5/0.5 weights, eval set): {baseline_loss:.6f}\n")

    trainable_params = list(fusion_mlp.parameters()) + list(context_encoder.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    global_step = 0          # counts optimizer steps (post-accumulation)
    accum_step = 0           # counts micro-batches within the current accumulation window
    accum_loss = 0.0         # running sum for logging
    best_loss = float("inf")

    # ----- profiling timers (seconds, accumulated across batches) -----
    use_cuda = cfg.device == "cuda"
    _T = {"dataloader": 0.0, "to_device": 0.0, "fusion": 0.0, "unet": 0.0, "backward_optim": 0.0, "backward": 0.0, "optim": 0.0}
    batches_timed = 0

    def _sync():
        if use_cuda:
            torch.cuda.synchronize()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        fusion_mlp.train()
        context_encoder.train()
        epoch_loss_sum = 0.0
        epoch_optimizer_steps = 0

        loader_iter = iter(loader)
        while True:
            # ── 1. DataLoader fetch ──────────────────────────────────────
            _sync()
            t0 = time.perf_counter()
            try:
                stems, canny_cond, depth_cond, latents, text_embs = next(loader_iter)
            except StopIteration:
                break
            _sync()
            _T["dataloader"] += time.perf_counter() - t0

            batch_size = latents.shape[0]
            batches_timed += 1

            # ── 2. .to(device) ───────────────────────────────────────────
            _sync()
            t0 = time.perf_counter()
            canny_cond            = canny_cond.to(device=cfg.device, dtype=cfg.dtype)
            depth_cond            = depth_cond.to(device=cfg.device, dtype=cfg.dtype)
            latents               = latents.to(device=cfg.device, dtype=cfg.dtype)
            encoder_hidden_states = text_embs.to(device=cfg.device, dtype=cfg.dtype)
            _sync()
            _T["to_device"] += time.perf_counter() - t0

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                low=0,
                high=scheduler.config.num_train_timesteps,
                size=(batch_size,),
                device=cfg.device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # ── 3. Fusion ControlNet calls ───────────────────────────────
            _sync()
            t0 = time.perf_counter()
            fused = fusion(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                canny_cond=canny_cond,
                depth_cond=depth_cond,
            )
            _sync()
            _T["fusion"] += time.perf_counter() - t0

            # ── 4. U-Net forward ─────────────────────────────────────────
            _sync()
            t0 = time.perf_counter()
            noise_pred = parts["unet"](
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=fused.down_block_res_samples,
                mid_block_additional_residual=fused.mid_block_res_sample,
                return_dict=True,
            ).sample
            _sync()
            _T["unet"] += time.perf_counter() - t0

            # ── 5. Backward + optimizer step ─────────────────────────────
            # _sync()
            # t0 = time.perf_counter()
            # Divide loss so that the sum over accumulation steps ≈ a single
            # large-batch loss (same scale regardless of accum window size).
            _sync()
            t1 = time.perf_counter()
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            (loss / cfg.gradient_accumulation_steps).backward()
            _sync()
            _T["backward"] += time.perf_counter() - t1

            accum_loss += loss.item()
            accum_step += 1

            if accum_step % cfg.gradient_accumulation_steps == 0:
                _sync()
                t1 = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _sync()
                _T["optim"] += time.perf_counter() - t1

                global_step += 1
                avg_loss = accum_loss / cfg.gradient_accumulation_steps
                epoch_loss_sum += avg_loss
                epoch_optimizer_steps += 1
                accum_loss = 0.0

                print(
                    f"epoch={epoch + 1}/{cfg.epochs} "
                    f"opt_step={global_step} "
                    f"loss={avg_loss:.6f} "
                    f"stem={stems[0]}"
                )

                # if avg_loss < best_loss:
                #     best_loss = avg_loss
                #     fusion_mlp.save(output_dir / "fusion_mlp_best.pth")

                # if global_step % cfg.save_every_steps == 0:
                #     fusion_mlp.save(output_dir / f"fusion_mlp_step_{global_step}.pth")
            # _sync()
            # _T["backward_optim"] += time.perf_counter() - t0

            if cfg.max_batches is not None and batches_timed >= cfg.max_batches:
                print(f"[profiling] reached max_batches={cfg.max_batches}, stopping early")
                break

        epoch_eval_loss = evaluate(fusion, fusion_mlp, context_encoder, eval_loader, parts, scheduler, cfg)
        print(f"\n--- epoch {epoch + 1} summary ---")
        print(f"eval loss: {epoch_eval_loss:.6f}" f"(baseline was {baseline_loss:.6f})")
        print("weights after this epoch (at zero context):")
        fusion_mlp.pretty_print(temperature=cfg.fusion_temperature)
        save_checkpoint(output_dir / f"fusion_mlp_epoch_{epoch + 1}.pth", fusion_mlp, context_encoder)

    # ── Timing summary ───────────────────────────────────────────────────────
    n = max(batches_timed, 1)
    total = sum(_T.values())
    print(f"\n{'─'*55}")
    print(f"  Timing summary  ({batches_timed} batches × bs={cfg.batch_size})")
    print(f"{'─'*55}")
    for name, secs in _T.items():
        avg_ms = secs / n * 1000
        pct = secs / total * 100 if total > 0 else 0
        print(f"  {name:<20s}  {avg_ms:7.2f} ms/batch  ({pct:5.1f}%)")
    print(f"  {'total (sum)':<20s}  {total / n * 1000:7.2f} ms/batch")
    print(f"{'─'*55}\n")

    save_checkpoint(output_dir / "fusion_mlp_last.pth", fusion_mlp, context_encoder)

    print("Training finished.")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Saved checkpoints to: {output_dir}")
    print("Final learned weights:")
    fusion_mlp.pretty_print(temperature=cfg.fusion_temperature)


if __name__ == "__main__":
    train(TrainConfig(
        batch_size=1,
        lr=1e-3,
    ))
