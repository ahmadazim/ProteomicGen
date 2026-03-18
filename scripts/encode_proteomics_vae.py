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


def _load_tsv(path: Path, transpose: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()[:10]
        raise ValueError(f"Found non-numeric values in TSV after parsing. Example columns: {bad_cols}")
    if transpose:
        df = df.T
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


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


@torch.no_grad()
def _encode_posterior_mean_in_batches(
    module: Any,
    counts: torch.Tensor,
    genes: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    n = counts.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        z_mean = module.vae_model.encode(counts[start:end], genes[start:end])
        out.append(z_mean.cpu())
    return torch.cat(out, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode all rows of a proteomics TSV into VAE posterior-mean latents (.pt)."
    )
    parser.add_argument("--ckpt-path", type=Path, default=_default_ckpt_path())
    parser.add_argument("--config-path", type=Path, default=_default_config_path())
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--output-pt", type=Path, required=True)
    parser.add_argument("--transpose", action="store_true", help="Transpose input TSV before encoding.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=_default_metadata_path(),
        help="Override metadata JSON that defines training feature order.",
    )
    args = parser.parse_args()

    ckpt_path = args.ckpt_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    config_path = args.config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")

    module, module_cfg = _load_module_from_checkpoint(ckpt_path=ckpt_path, config_path=config_path, device=device)
    metadata_json = _resolve_metadata_json(module_cfg, config_path, args.metadata_json)
    vocab = _build_vocabulary_encoder(module_cfg, metadata_json=metadata_json)

    df = _load_tsv(args.input_tsv.resolve(), transpose=args.transpose)
    target_genes = [str(gene) for gene in vocab.genes]
    target_gene_set = set(target_genes)
    input_gene_set = set(df.columns)

    missing_genes = [gene for gene in target_genes if gene not in input_gene_set]
    dropped_genes = [gene for gene in df.columns if gene not in target_gene_set]

    df_aligned = df.reindex(columns=target_genes, fill_value=0.0)

    counts_np = df_aligned.to_numpy(dtype="float32", copy=True)
    genes_1d = vocab.encode_genes(target_genes)
    genes_np = genes_1d[None, :].repeat(counts_np.shape[0], axis=0)
    library_size_np = counts_np.sum(axis=1, keepdims=True)

    counts_t = torch.from_numpy(counts_np).to(device)
    genes_t = torch.from_numpy(genes_np).long().to(device)
    z_mean = _encode_posterior_mean_in_batches(
        module=module,
        counts=counts_t,
        genes=genes_t,
        batch_size=int(args.batch_size),
    )

    payload = {
        "posterior_mean": z_mean.contiguous(),
        "z": z_mean.contiguous(),
        "obs_names": df_aligned.index.tolist(),
        "var_names": target_genes,
        "genes": torch.from_numpy(genes_1d).long(),
        "library_size": torch.from_numpy(library_size_np).float(),
        "input_tsv": args.input_tsv.resolve().as_posix(),
        "ckpt_path": ckpt_path.as_posix(),
        "config_path": config_path.as_posix(),
        "metadata_json": metadata_json.as_posix(),
        "missing_genes_filled_with_zero": missing_genes,
        "dropped_input_genes_not_in_model": dropped_genes,
    }
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_pt.resolve())

    print(f"[OK] Encoded {counts_np.shape[0]} samples")
    print(f"[OK] Input features: {df.shape[1]} -> model features: {len(target_genes)}")
    print(f"[OK] Posterior-mean latent shape: {tuple(z_mean.shape)}")
    print(f"[OK] Saved latents to: {args.output_pt.resolve()}")
    if missing_genes:
        print(f"[WARN] Missing model genes filled with 0: {len(missing_genes)}")
    if dropped_genes:
        print(f"[WARN] Input genes not used by model: {len(dropped_genes)}")


if __name__ == "__main__":
    main()
