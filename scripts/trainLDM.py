#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from _common import load_local_sit_flow_model

SiTFlowModel = load_local_sit_flow_model()


class EMA:
    """Minimal EMA wrapper to avoid external dependencies."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_params = dict(self.module.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

        ema_buffers = dict(self.module.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, buf in model_buffers.items():
            ema_buffers[name].copy_(buf)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_latents_tensor(obj: Any) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj.float()
    if isinstance(obj, dict):
        for key in ("posterior_mean", "z", "latents"):
            if key in obj and torch.is_tensor(obj[key]):
                return obj[key].float()
        raise KeyError("Latents .pt dict must contain one of: posterior_mean, z, latents")
    raise TypeError("Expected latents data as torch.Tensor or dict containing tensor.")


def load_latents(path: Path, expected_tokens: int, expected_dim: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x = _extract_latents_tensor(payload)

    if x.ndim == 2 and x.shape[1] == expected_tokens * expected_dim:
        x = x.view(x.shape[0], expected_tokens, expected_dim)

    if x.ndim != 3:
        raise ValueError(f"Expected latent tensor with shape (N, T, D), got {tuple(x.shape)}")
    if x.shape[1] != expected_tokens or x.shape[2] != expected_dim:
        raise ValueError(
            f"Latent shape mismatch. Expected (N, {expected_tokens}, {expected_dim}), got {tuple(x.shape)}"
        )
    return x.contiguous()


def compute_feature_stats(latents: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    # For token latents (N,T,D), compute per-feature stats over N and T.
    mean = latents.mean(dim=(0, 1))
    std = latents.std(dim=(0, 1), unbiased=False).clamp_min(float(eps))
    return mean, std


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but CUDA is not available.")
        device = torch.device("cuda")

    set_seed(args.seed)
    os.makedirs(Path(args.model_output_path).parent, exist_ok=True)

    latents = load_latents(
        path=Path(args.train_latents_path),
        expected_tokens=args.latent_length,
        expected_dim=args.token_dim,
    )
    dataset = TensorDataset(latents)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    if len(loader) == 0:
        raise ValueError(
            f"No training batches produced. batch_size={args.batch_size} is larger than dataset size {len(dataset)}."
        )

    mean, std = compute_feature_stats(latents, eps=args.norm_eps)
    mean = mean.view(1, 1, -1).to(device)
    std = std.view(1, 1, -1).to(device)

    model = SiTFlowModel(
        token_dim=args.token_dim,
        latent_length=args.latent_length,
        hidden_dim=args.hidden_dim,
        cond_dim=None,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,
        qkv_bias=args.qkv_bias,
        use_udit=True,  # Force USiT backbone as requested.
        source_token_ids=None,
        num_sources=None,
    ).to(device)

    ema = EMA(model, decay=args.ema_decay)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    total_steps = args.num_epochs * len(loader)
    warmup_steps = int(args.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_bf16 = bool(args.use_bf16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for (x,) in pbar:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            x = (x - mean) / std

            bsz = x.shape[0]
            t = torch.rand(bsz, device=device, dtype=x.dtype)
            t_view = t.view(bsz, 1, 1)

            base = torch.randn_like(x)
            z_t = (1.0 - t_view) * base + t_view * x
            v_target = x - base

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                pred = model(z_t, t)
                loss = F.mse_loss(pred, v_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            running_loss += float(loss.item())
            steps += 1
            global_step += 1
            pbar.set_postfix(loss=f"{running_loss / steps:.6f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        epoch_loss = running_loss / max(steps, 1)
        print(f"[Epoch {epoch}] loss={epoch_loss:.6f}")

        if args.save_every > 0 and (epoch % args.save_every == 0):
            ckpt_epoch = Path(args.model_output_path).with_suffix("").as_posix() + f"_epoch{epoch}.pth"
            save_checkpoint(
                ckpt_path=Path(ckpt_epoch),
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                feature_mean=mean.detach().cpu().view(-1),
                feature_std=std.detach().cpu().view(-1),
                global_step=global_step,
                epoch=epoch,
            )

    save_checkpoint(
        ckpt_path=Path(args.model_output_path),
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        feature_mean=mean.detach().cpu().view(-1),
        feature_std=std.detach().cpu().view(-1),
        global_step=global_step,
        epoch=args.num_epochs,
    )
    print(f"[Done] Saved final checkpoint to {args.model_output_path}")


def save_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    args: argparse.Namespace,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    global_step: int,
    epoch: int,
) -> None:
    ckpt = {
        "weights": model.state_dict(),
        "ema_weights": ema.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "args": vars(args),
        "model_type": "usit",
        "flow_matching": True,
        "conditional": False,
        "latent_shape": (args.latent_length, args.token_dim),
        "channel_mean": feature_mean,
        "channel_std": feature_std,
        "epoch": int(epoch),
        "global_step": int(global_step),
    }
    torch.save(ckpt, ckpt_path)
    print(f"[Saved] {ckpt_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train unconditional USiT latent diffusion model (flow matching).")
    parser.add_argument("--train-latents-path", type=str, required=True, help="Path to latent .pt tensor/dict.")
    parser.add_argument("--model-output-path", type=str, required=True, help="Output checkpoint path (.pth).")

    parser.add_argument("--latent-length", type=int, default=256, help="Number of tokens per sample.")
    parser.add_argument("--token-dim", type=int, default=32, help="Token embedding dimension.")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10, help="Save epoch checkpoints every N epochs.")

    parser.add_argument("--num-layers", type=int, default=9, help="USiT layers (must be odd, >=3).")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=None, help="Internal transformer width.")
    parser.add_argument("--qkv-bias", action="store_true")

    parser.add_argument("--norm-eps", type=float, default=1e-6, help="Std clamp for latent normalization.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--use-bf16", action="store_true", help="Use bfloat16 autocast on CUDA.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
