#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from _common import load_local_sit_flow_model, repo_root

SiTFlowModel = load_local_sit_flow_model()


def _default_vae_ckpt() -> Path:
    return repo_root() / "models" / "vae_proteomicsUKB_XL_last.ckpt"


def _default_vae_config() -> Path:
    return repo_root() / "models" / "vae_proteomicsUKB_XL_config.yaml"


def _default_metadata() -> Path:
    return repo_root() / "models" / "vae_proteomicsUKB_XL_metadata.json"


def _default_decode_script() -> Path:
    return repo_root() / "scripts" / "decode_proteomics_vae.py"


def _normalize_state_dict(maybe_sd: Any) -> dict[str, torch.Tensor]:
    if not isinstance(maybe_sd, dict):
        raise TypeError(f"Expected state_dict dictionary, got {type(maybe_sd)}")
    if "state_dict" in maybe_sd and isinstance(maybe_sd["state_dict"], dict):
        maybe_sd = maybe_sd["state_dict"]
    if "module" in maybe_sd and isinstance(maybe_sd["module"], dict):
        maybe_sd = maybe_sd["module"]

    state_dict = dict(maybe_sd)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.split("module.", 1)[1] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
    return state_dict


def _get_checkpoint_weights(payload: dict[str, Any], use_ema: bool) -> dict[str, torch.Tensor]:
    if use_ema:
        for key in ("ema_weights", "ema"):
            if key in payload and payload[key] is not None:
                return _normalize_state_dict(payload[key])
    for key in ("weights", "model", "state_dict"):
        if key in payload and payload[key] is not None:
            return _normalize_state_dict(payload[key])
    raise KeyError("Could not find model weights in checkpoint.")


def _infer_model_config(payload: dict[str, Any]) -> tuple[int, int, int | None, int, int, int, bool]:
    latent_shape = payload.get("latent_shape")
    if latent_shape is None or len(latent_shape) != 2:
        raise ValueError(f"Checkpoint must include latent_shape=(L,D). Got: {latent_shape}")
    latent_length, token_dim = int(latent_shape[0]), int(latent_shape[1])

    sit_cfg = payload.get("sit_config", {}) or {}
    args_cfg = payload.get("args", {}) or {}

    hidden_dim = sit_cfg.get("hidden_dim", args_cfg.get("hidden_dim", args_cfg.get("sit_hidden_dim", None)))
    hidden_dim = None if hidden_dim in (None, "None") else int(hidden_dim)
    num_layers = int(sit_cfg.get("num_layers", args_cfg.get("num_layers", args_cfg.get("sit_num_layers", 9))))
    num_heads = int(sit_cfg.get("num_heads", args_cfg.get("num_heads", args_cfg.get("sit_num_heads", 8))))
    mlp_ratio = int(sit_cfg.get("mlp_ratio", args_cfg.get("mlp_ratio", args_cfg.get("sit_mlp_ratio", 4))))
    qkv_bias = bool(sit_cfg.get("qkv_bias", args_cfg.get("qkv_bias", args_cfg.get("sit_qkv_bias", False))))
    return latent_length, token_dim, hidden_dim, num_layers, num_heads, mlp_ratio, qkv_bias


def _get_norm_stats(payload: dict[str, Any], token_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = payload.get("channel_mean")
    std = payload.get("channel_std")

    mean_t = torch.as_tensor(mean if mean is not None else torch.zeros(token_dim), dtype=torch.float32).view(-1)
    std_t = torch.as_tensor(std if std is not None else torch.ones(token_dim), dtype=torch.float32).view(-1)
    if mean_t.numel() != token_dim or std_t.numel() != token_dim:
        raise ValueError(
            f"Normalization stats mismatch with token_dim={token_dim}: mean={mean_t.shape}, std={std_t.shape}"
        )
    return mean_t, std_t.clamp_min(1e-6)


@torch.no_grad()
def _sample_flow(
    model: torch.nn.Module,
    n_samples: int,
    latent_length: int,
    token_dim: int,
    steps: int,
    batch_size: int,
    solver: str,
    device: torch.device,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, n_samples, batch_size)):
        current_batch = min(batch_size, n_samples - start)
        x = torch.randn(current_batch, latent_length, token_dim, device=device, dtype=torch.float32)
        dt = 1.0 / float(steps)
        for index in range(steps):
            t0 = float(index) / float(steps)
            t1 = float(index + 1) / float(steps)
            t0_batch = torch.full((current_batch,), t0, device=device, dtype=x.dtype)
            if solver == "heun":
                v0 = model(x, t0_batch)
                x_euler = x + dt * v0
                t1_batch = torch.full((current_batch,), t1, device=device, dtype=x.dtype)
                v1 = model(x_euler, t1_batch)
                x = x + 0.5 * dt * (v0 + v1)
            else:
                x = x + dt * model(x, t0_batch)
        chunks.append(x.cpu())
    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic proteomic profiles from a trained USiT checkpoint and decode via VAE."
    )
    parser.add_argument("--usit-ckpt", type=Path, required=True)
    parser.add_argument("--n-syn", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vae-ckpt", type=Path, default=_default_vae_ckpt())
    parser.add_argument("--vae-config", type=Path, default=_default_vae_config())
    parser.add_argument("--metadata-json", type=Path, default=_default_metadata())
    parser.add_argument("--decode-script", type=Path, default=_default_decode_script())
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--solver", type=str, choices=["euler", "heun"], default="heun")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--decode-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--use-model-weights", action="store_true")
    parser.add_argument("--library-size-value", type=float, default=1.0)
    args = parser.parse_args()

    if args.n_syn <= 0:
        raise ValueError("--n-syn must be > 0")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latents_out = args.output_dir / f"synthetic_latents_n{args.n_syn}.pt"
    decoded_tsv_out = args.output_dir / f"synthetic_proteomics_n{args.n_syn}.tsv"
    decoded_pt_out = args.output_dir / f"synthetic_decoded_n{args.n_syn}.pt"

    payload = torch.load(args.usit_ckpt.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint payload type: {type(payload)}")

    latent_length, token_dim, hidden_dim, num_layers, num_heads, mlp_ratio, qkv_bias = _infer_model_config(payload)

    model = SiTFlowModel(
        token_dim=token_dim,
        latent_length=latent_length,
        hidden_dim=hidden_dim,
        cond_dim=None,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=0.0,
        qkv_bias=qkv_bias,
        use_udit=True,
        source_token_ids=None,
        num_sources=None,
    ).to(device)

    state_dict = _get_checkpoint_weights(payload, use_ema=(not args.use_model_weights))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    mean, std = _get_norm_stats(payload, token_dim=token_dim)
    z_norm = _sample_flow(
        model=model,
        n_samples=args.n_syn,
        latent_length=latent_length,
        token_dim=token_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        solver=args.solver,
        device=device,
    )
    z = z_norm * std.view(1, 1, -1) + mean.view(1, 1, -1)

    obs_names = [f"synthetic_{index}" for index in range(args.n_syn)]
    torch.save(
        {
            "posterior_mean": z.contiguous(),
            "z": z.contiguous(),
            "obs_names": obs_names,
            "usit_ckpt": args.usit_ckpt.resolve().as_posix(),
            "solver": args.solver,
            "steps": int(args.steps),
        },
        latents_out,
    )
    print(f"[OK] Saved synthetic latents: {latents_out} with shape {tuple(z.shape)}")

    decode_cmd = [
        sys.executable,
        args.decode_script.resolve().as_posix(),
        "--ckpt-path",
        args.vae_ckpt.resolve().as_posix(),
        "--config-path",
        args.vae_config.resolve().as_posix(),
        "--latents-pt",
        latents_out.resolve().as_posix(),
        "--output-tsv",
        decoded_tsv_out.resolve().as_posix(),
        "--output-pt",
        decoded_pt_out.resolve().as_posix(),
        "--batch-size",
        str(int(args.decode_batch_size)),
        "--device",
        args.device,
        "--library-size-value",
        str(float(args.library_size_value)),
    ]
    if args.metadata_json is not None:
        decode_cmd.extend(["--metadata-json", args.metadata_json.resolve().as_posix()])

    print("[INFO] Running decoder:")
    print(" ".join(decode_cmd))
    subprocess.run(decode_cmd, check=True)
    print(f"[OK] Saved decoded synthetic proteomics TSV: {decoded_tsv_out}")


if __name__ == "__main__":
    main()
