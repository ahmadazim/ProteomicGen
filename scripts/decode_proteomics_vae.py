#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import prepend_project_pythonpath, repo_root

prepend_project_pythonpath()

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from scldm._utils import remap_pickle
from scldm.encoder import VocabularyEncoderSimplified


def _default_ckpt_path() -> Path:
    return repo_root() / "models" / "vae_proteomicsUKB_XL_last.ckpt"


def _default_config_path() -> Path:
    return repo_root() / "models" / "vae_proteomicsUKB_XL_config.yaml"


def _default_metadata_path() -> Path:
    return repo_root() / "models" / "vae_proteomicsUKB_XL_metadata.json"


def _resolve_metadata_json(
    cfg: DictConfig,
    config_path: Path,
    metadata_json_override: Path | None,
) -> Path:
    if metadata_json_override is not None:
        if not metadata_json_override.exists():
            raise FileNotFoundError(f"metadata_json override not found: {metadata_json_override}")
        return metadata_json_override

    cfg_path = Path(str(cfg.datamodule.vocabulary_encoder.metadata_json))
    if cfg_path.exists():
        return cfg_path

    fallback = config_path.parents[2] / "data" / "proteomics_metadata.json"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Could not resolve metadata_json path from config or fallback. Please provide --metadata-json explicitly."
    )


def _build_vocabulary_encoder(cfg: DictConfig, metadata_json: Path) -> VocabularyEncoderSimplified:
    vocab_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datamodule.vocabulary_encoder, resolve=False))
    vocab_cfg.metadata_json = metadata_json.as_posix()
    vocab_cfg.adata_path = None
    return hydra.utils.instantiate(vocab_cfg)


def _load_module_from_checkpoint(
    ckpt_path: Path,
    config_path: Path,
    device: torch.device,
) -> Any:
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass

    module_cfg = OmegaConf.load(config_path)
    module = hydra.utils.instantiate(module_cfg.model.module)

    checkpoint = torch.load(ckpt_path, map_location="cpu", pickle_module=remap_pickle, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    module_keys = set(module.state_dict().keys())
    filtered = {key: value for key, value in state_dict.items() if key in module_keys}
    if len(filtered) != len(module_keys):
        missing = sorted(module_keys - set(filtered.keys()))
        raise RuntimeError(
            f"Checkpoint/config mismatch: loaded {len(filtered)} keys, expected {len(module_keys)}. "
            f"Missing examples: {missing[:10]}"
        )
    module.load_state_dict(filtered, strict=True)
    module.eval().to(device)
    return module, module_cfg


def _unpack_latents(latents_obj: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(latents_obj, dict):
        if "posterior_mean" in latents_obj:
            z = latents_obj["posterior_mean"]
        elif "z" in latents_obj:
            z = latents_obj["z"]
        else:
            raise KeyError("Latent .pt dictionary must contain key 'posterior_mean' or 'z'.")
        if not torch.is_tensor(z):
            raise TypeError("latents['posterior_mean' or 'z'] must be a torch.Tensor.")
        meta = {key: value for key, value in latents_obj.items() if key not in {"z", "posterior_mean"}}
        return z, meta
    if torch.is_tensor(latents_obj):
        return latents_obj, {}
    raise TypeError("Latent .pt must be either a torch.Tensor or a dict containing {'z': tensor, ...}.")


def _reshape_if_flat(z: torch.Tensor, latent_dim: int, latent_embedding: int) -> torch.Tensor:
    if z.ndim == 3:
        return z
    if z.ndim == 2 and z.shape[1] == latent_dim * latent_embedding:
        return z.view(z.shape[0], latent_dim, latent_embedding)
    raise ValueError(
        f"Unexpected latent shape {tuple(z.shape)}. Expected (N, {latent_dim}, {latent_embedding}) or "
        f"(N, {latent_dim * latent_embedding})."
    )


@torch.no_grad()
def _decode_in_batches(
    module: Any,
    z: torch.Tensor,
    genes: torch.Tensor,
    library_size: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    n = z.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        likelihood = module.vae_model.decode(
            z=z[start:end],
            genes=genes[start:end],
            library_size=library_size[start:end],
        )
        decoded = likelihood.mean if hasattr(likelihood, "mean") else likelihood.sample()
        out.append(decoded.cpu())
    return torch.cat(out, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode VAE latents (.pt) into reconstructed proteomics TSV.")
    parser.add_argument("--ckpt-path", type=Path, default=_default_ckpt_path())
    parser.add_argument("--config-path", type=Path, default=_default_config_path())
    parser.add_argument("--latents-pt", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-pt", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--metadata-json", type=Path, default=_default_metadata_path())
    parser.add_argument(
        "--library-size-value",
        type=float,
        default=1.0,
        help="Fallback library size when not provided in latents file.",
    )
    args = parser.parse_args()

    ckpt_path = args.ckpt_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    config_path = args.config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    latents_path = args.latents_pt.resolve()
    if not latents_path.exists():
        raise FileNotFoundError(f"Latents file not found: {latents_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")

    module, module_cfg = _load_module_from_checkpoint(ckpt_path=ckpt_path, config_path=config_path, device=device)
    metadata_json = _resolve_metadata_json(module_cfg, config_path, args.metadata_json)
    vocab = _build_vocabulary_encoder(module_cfg, metadata_json=metadata_json)

    latents_obj = torch.load(latents_path, map_location="cpu")
    z, meta = _unpack_latents(latents_obj)
    z = z.float()

    latent_dim = int(module.vae_model.encoder.latent_dim)
    latent_embedding = int(module.vae_model.encoder.latent_embedding)
    z = _reshape_if_flat(z, latent_dim=latent_dim, latent_embedding=latent_embedding)

    n_samples = z.shape[0]
    var_names = [str(gene) for gene in vocab.genes]
    genes_1d = vocab.encode_genes(var_names)

    obs_names = meta.get("obs_names")
    if obs_names is None:
        obs_names = [str(i) for i in range(n_samples)]
    else:
        obs_names = [str(value) for value in obs_names]
        if len(obs_names) != n_samples:
            raise ValueError(f"obs_names length {len(obs_names)} does not match latent samples {n_samples}.")

    if "genes" in meta and torch.is_tensor(meta["genes"]):
        genes_meta = meta["genes"].long()
        if genes_meta.ndim == 1:
            genes = genes_meta[None, :].repeat(n_samples, 1)
        elif genes_meta.ndim == 2:
            if genes_meta.shape[0] != n_samples:
                raise ValueError(
                    f"genes tensor in latents has shape {tuple(genes_meta.shape)} but expected first dim {n_samples}."
                )
            genes = genes_meta
        else:
            raise ValueError(f"Unsupported genes tensor shape in latents: {tuple(genes_meta.shape)}")
    else:
        genes = torch.from_numpy(genes_1d)[None, :].repeat(n_samples, 1)

    if "library_size" in meta and torch.is_tensor(meta["library_size"]):
        library_size = meta["library_size"].float()
        if library_size.ndim == 1:
            library_size = library_size.view(-1, 1)
        if library_size.shape != (n_samples, 1):
            raise ValueError(
                f"library_size shape {tuple(library_size.shape)} must be (N, 1)=({n_samples}, 1) for decoding."
            )
    else:
        library_size = torch.full((n_samples, 1), float(args.library_size_value), dtype=torch.float32)

    z = z.to(device)
    genes = genes.long().to(device)
    library_size = library_size.to(device)

    decoded = _decode_in_batches(
        module=module,
        z=z,
        genes=genes,
        library_size=library_size,
        batch_size=int(args.batch_size),
    )
    decoded_np = decoded.numpy()

    df_decoded = pd.DataFrame(decoded_np, index=obs_names, columns=var_names)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    df_decoded.to_csv(args.output_tsv.resolve(), sep="\t")

    if args.output_pt is not None:
        payload = {
            "decoded": decoded,
            "obs_names": obs_names,
            "var_names": var_names,
            "latents_path": latents_path.as_posix(),
            "ckpt_path": ckpt_path.as_posix(),
            "config_path": config_path.as_posix(),
            "metadata_json": metadata_json.as_posix(),
        }
        args.output_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.output_pt.resolve())

    print(f"[OK] Loaded latents from: {latents_path}")
    print(f"[OK] Latent shape used for decode: {tuple(z.shape)}")
    print(f"[OK] Decoded matrix shape: {decoded_np.shape}")
    print(f"[OK] Saved decoded TSV to: {args.output_tsv.resolve()}")
    if args.output_pt is not None:
        print(f"[OK] Saved decoded payload to: {args.output_pt.resolve()}")


if __name__ == "__main__":
    main()
