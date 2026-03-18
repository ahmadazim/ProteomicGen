#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from _common import build_pythonpath_env, external_scldm_root


def _load_tsv_as_anndata(tsv_path: Path, transpose: bool) -> ad.AnnData:
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()[:10]
        raise ValueError(f"Found non-numeric values in TSV after parsing. Example columns: {bad_cols}")

    if transpose:
        df = df.T

    # Store as CSR so upstream scLDM dataloading paths that call `.toarray()` remain compatible.
    x = sparse.csr_matrix(df.to_numpy(dtype=np.float32, copy=True))
    adata = ad.AnnData(X=x)
    adata.obs_names = pd.Index(df.index.astype(str))
    adata.var_names = pd.Index(df.columns.astype(str))
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def _split_train_test(adata: ad.AnnData, test_frac: float, seed: int) -> tuple[ad.AnnData, ad.AnnData]:
    n = adata.n_obs
    if n < 2:
        raise ValueError(f"Need at least 2 samples to split train/test, got n_obs={n}")
    n_test = int(round(n * test_frac))
    n_test = max(1, min(n - 1, n_test))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return adata[train_idx].copy(), adata[test_idx].copy()


def _write_metadata_json(path: Path, var_names: pd.Index) -> None:
    payload = {
        "genes": [str(v) for v in var_names.tolist()],
        "labels": {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare proteomics TSV and launch scLDM VAE training.")
    parser.add_argument("--tsv-path", type=Path, required=True, help="TSV matrix path, samples x proteins by default.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Run directory for prepared data and checkpoints.")
    parser.add_argument("--experiment-name", type=str, default="vae_proteomicsUKB_XL")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--n-embed", type=int, default=192, help="Transformer hidden width.")
    parser.add_argument("--n-layer", type=int, default=8, help="Encoder/decoder transformer depth.")
    parser.add_argument("--n-head", type=int, default=8, help="Self-attention heads.")
    parser.add_argument("--n-head-cross", type=int, default=4, help="Cross-attention heads.")
    parser.add_argument("--n-inducing-points", type=int, default=256, help="Number of latent tokens.")
    parser.add_argument("--n-embed-latent", type=int, default=32, help="Latent token dimension.")
    parser.add_argument("--transpose", action="store_true", help="Transpose TSV before training.")
    parser.add_argument(
        "--sample-genes",
        type=str,
        default="none",
        choices=["none", "random", "weighted", "expressed", "expressed_zero"],
        help="Subset strategy for stage-1 VAE training.",
    )
    parser.add_argument("--enable-wandb", action="store_true", help="Enable wandb logging.")
    args = parser.parse_args()

    if args.n_embed % args.n_head != 0:
        raise ValueError(f"--n-embed ({args.n_embed}) must be divisible by --n-head ({args.n_head})")
    if args.n_embed % args.n_head_cross != 0:
        raise ValueError(
            f"--n-embed ({args.n_embed}) must be divisible by --n-head-cross ({args.n_head_cross})"
        )

    scldm_root = external_scldm_root()
    if not scldm_root.exists():
        raise FileNotFoundError(f"Upstream scLDM repo not found: {scldm_root}")

    work_dir = args.work_dir.resolve()
    data_dir = work_dir / "data"
    exp_dir = work_dir / "experiments"
    data_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    adata = _load_tsv_as_anndata(args.tsv_path.resolve(), transpose=args.transpose)
    train_adata, test_adata = _split_train_test(adata, test_frac=float(args.test_frac), seed=int(args.seed))
    n_features = int(train_adata.n_vars)

    train_path = data_dir / "proteomics_train.h5ad"
    test_path = data_dir / "proteomics_test.h5ad"
    metadata_json = data_dir / "proteomics_metadata.json"
    train_adata.write_h5ad(train_path)
    test_adata.write_h5ad(test_path)
    _write_metadata_json(metadata_json, train_adata.var_names)

    train_args = [
        "experiments/scripts/train.py",
        f"experiment_name={args.experiment_name}",
        f"seed={int(args.seed)}",
        f"paths.base_data_path={data_dir.as_posix()}",
        f"paths.base_experiment_path={exp_dir.as_posix()}",
        "datamodule.dataset=dentate_gyrus",
        "datamodule.datamodule._target_=genprot_scldm.datamodule.DataModule",
        f"datamodule.datamodule.train_adata_path={train_path.as_posix()}",
        f"datamodule.datamodule.test_adata_path={test_path.as_posix()}",
        "datamodule.datamodule.adata_attr=X",
        "datamodule.datamodule.adata_key=null",
        f"datamodule.datamodule.sample_genes={args.sample_genes}",
        f"datamodule.datamodule.genes_seq_len={n_features}",
        f"datamodule.datamodule.num_workers={int(args.num_workers)}",
        "datamodule.datamodule.drop_last_indices=false",
        "datamodule.datamodule.drop_incomplete_batch=false",
        f"datamodule.datamodule.persistent_workers={'true' if int(args.num_workers) > 0 else 'false'}",
        "model.module._target_=genprot_scldm.models.VAE",
        "model.module.vae_model._target_=genprot_scldm.vae.TransformerVAE",
        "model.decoder_head.gaussian._target_=genprot_scldm.stochastic_layers.GaussianTransformerLayer",
        f"model.batch_size={int(args.batch_size)}",
        f"model.test_batch_size={int(args.batch_size)}",
        f"training.num_epochs={int(args.epochs)}",
        "model.decoder_name=gaussian",
        "model.module.vae_model.input_layer.agg_func=proj",
        f"model.module.vae_model.encoder.n_embed={int(args.n_embed)}",
        f"model.module.vae_model.encoder.n_layer={int(args.n_layer)}",
        f"model.module.vae_model.encoder.n_head={int(args.n_head)}",
        f"model.module.vae_model.encoder.n_head_cross={int(args.n_head_cross)}",
        f"model.module.vae_model.encoder.n_inducing_points={int(args.n_inducing_points)}",
        f"model.module.vae_model.encoder.n_embed_latent={int(args.n_embed_latent)}",
        "datamodule.vocabulary_encoder.class_vocab_sizes={}",
        "datamodule.vocabulary_encoder.guidance_weight={}",
        f"datamodule.vocabulary_encoder.n_genes={n_features}",
        "datamodule.vocabulary_encoder.mu_size_factor=null",
        "datamodule.vocabulary_encoder.sd_size_factor=null",
        "datamodule.vocabulary_encoder.condition_strategy=mutually_exclusive",
        "datamodule.vocabulary_encoder.metadata_genes=null",
        f"datamodule.vocabulary_encoder.metadata_json={metadata_json.as_posix()}",
    ]
    if not args.enable_wandb:
        train_args.append("~training.logger.wandb")

    hydra_available = importlib.util.find_spec("hydra") is not None
    if hydra_available:
        cmd = [sys.executable, *train_args]
    elif shutil.which("uv") is not None:
        cmd = ["uv", "run", "python", *train_args]
    else:
        raise RuntimeError(
            "Could not find hydra in the current environment and `uv` is not available.\n"
            "Install the scLDM dependencies or use `uv` inside `external/scldm`."
        )

    print("[INFO] Prepared train/test AnnData:")
    print(f"       train: {train_path} shape={train_adata.shape}")
    print(f"       test : {test_path} shape={test_adata.shape}")
    print(f"       metadata_json: {metadata_json}")
    print("[INFO] Launching VAE training command:")
    print(" ".join(cmd))

    env = dict(os.environ)
    env["PYTHONPATH"] = build_pythonpath_env(env.get("PYTHONPATH"))
    subprocess.run(cmd, cwd=scldm_root.as_posix(), check=True, env=env)


if __name__ == "__main__":
    main()
