#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA


def _load_tsv(path: Path, transpose: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    if transpose:
        df = df.T
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()[:10]
        raise ValueError(f"Non-numeric values found in {path}. Example problematic columns: {bad_cols}")
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _align_features(obs: pd.DataFrame, gen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = [c for c in obs.columns if c in set(gen.columns)]
    if not shared:
        raise ValueError("No shared proteins between observed and generated data.")
    return obs[shared].copy(), gen[shared].copy()


def _plot_distribution_overlays(
    obs: pd.DataFrame,
    gen: pd.DataFrame,
    out_path: Path,
    n_proteins: int,
    protein_seed: int,
    proteins: list[str] | None,
    bins: int,
) -> list[str]:
    if proteins:
        missing = [p for p in proteins if p not in obs.columns]
        if missing:
            raise ValueError(f"Requested proteins not found after alignment: {missing[:10]}")
        selected = proteins
    else:
        rng = np.random.default_rng(protein_seed)
        n_pick = min(int(n_proteins), obs.shape[1])
        selected = sorted(rng.choice(obs.columns.to_numpy(), size=n_pick, replace=False).tolist())

    n = len(selected)
    n_cols = min(4, max(1, n))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), squeeze=False)

    for i, protein in enumerate(selected):
        ax = axes[i // n_cols][i % n_cols]
        ax.hist(obs[protein].to_numpy(), bins=bins, density=True, alpha=0.45, label="Observed")
        ax.hist(gen[protein].to_numpy(), bins=bins, density=True, alpha=0.45, label="Generated")
        ax.set_title(protein, fontsize=9)
        ax.set_xlabel("Abundance")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, frameon=False)

    # Hide unused panels.
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle("Protein abundance distributions: observed vs generated", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return selected


def _plot_pca_overlay(
    obs: pd.DataFrame,
    gen: pd.DataFrame,
    out_path: Path,
    max_points_per_set: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    obs_n = obs.shape[0]
    gen_n = gen.shape[0]
    obs_idx = rng.choice(obs_n, size=min(obs_n, max_points_per_set), replace=False)
    gen_idx = rng.choice(gen_n, size=min(gen_n, max_points_per_set), replace=False)

    obs_sub = obs.iloc[obs_idx].to_numpy(dtype=np.float64, copy=False)
    gen_sub = gen.iloc[gen_idx].to_numpy(dtype=np.float64, copy=False)

    # Fit standardization and PCA on observed subset to anchor representation.
    mean = obs_sub.mean(axis=0, keepdims=True)
    std = obs_sub.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    obs_std = (obs_sub - mean) / std
    gen_std = (gen_sub - mean) / std

    pca = PCA(n_components=2, random_state=seed)
    obs_p = pca.fit_transform(obs_std)
    gen_p = pca.transform(gen_std)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.scatter(obs_p[:, 0], obs_p[:, 1], s=10, alpha=0.35, label="Observed")
    ax.scatter(gen_p[:, 0], gen_p[:, 1], s=10, alpha=0.35, label="Generated")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title("PCA overlay: observed vs generated")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _compute_covariance_order(cov_obs: np.ndarray, method: str = "average") -> np.ndarray:
    # HAC on protein covariance profiles (rows of covariance matrix).
    dist = pdist(cov_obs, metric="euclidean")
    tree = linkage(dist, method=method)
    return leaves_list(tree)


def _plot_covariance_heatmaps(
    obs: pd.DataFrame,
    gen: pd.DataFrame,
    out_obs: Path,
    out_gen: Path,
    out_order: Path,
) -> None:
    cov_obs = np.cov(obs.to_numpy(dtype=np.float64, copy=False), rowvar=False)
    cov_gen = np.cov(gen.to_numpy(dtype=np.float64, copy=False), rowvar=False)

    order = _compute_covariance_order(cov_obs, method="average")
    proteins_ordered = obs.columns.to_numpy()[order]
    out_order.write_text("\n".join(proteins_ordered.tolist()) + "\n", encoding="utf-8")

    cov_obs_ord = cov_obs[np.ix_(order, order)]
    cov_gen_ord = cov_gen[np.ix_(order, order)]

    vmax = float(np.max(np.abs(np.concatenate([cov_obs_ord.ravel(), cov_gen_ord.ravel()]))))
    vmax = max(vmax, 1e-8)
    vmin = -vmax

    for cov_mat, title, out_path in [
        (cov_obs_ord, "Observed protein covariance (HAC order)", out_obs),
        (cov_gen_ord, "Generated protein covariance (observed HAC order)", out_gen),
    ]:
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(cov_mat, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Proteins (ordered)")
        ax.set_ylabel("Proteins (ordered)")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Covariance")
        fig.tight_layout()
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize observed vs generated proteomics: "
            "(1) sampled protein distribution overlays, (2) PCA overlay, "
            "(3) covariance heatmaps using observed HAC ordering."
        )
    )
    parser.add_argument("--observed-tsv", type=Path, required=True, help="Observed proteomics TSV (samples x proteins).")
    parser.add_argument("--generated-tsv", type=Path, required=True, help="Generated proteomics TSV (samples x proteins).")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--transpose-observed", action="store_true")
    parser.add_argument("--transpose-generated", action="store_true")

    parser.add_argument("--n-proteins", type=int, default=12, help="Number of random proteins for distribution overlays.")
    parser.add_argument("--protein-seed", type=int, default=42, help="Seed for random protein selection.")
    parser.add_argument(
        "--proteins",
        type=str,
        default=None,
        help="Comma-separated protein names to plot instead of random selection.",
    )
    parser.add_argument("--hist-bins", type=int, default=50)
    parser.add_argument("--max-pc-points", type=int, default=10000, help="Max points per set for PCA scatter.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    obs = _load_tsv(args.observed_tsv.resolve(), transpose=args.transpose_observed)
    gen = _load_tsv(args.generated_tsv.resolve(), transpose=args.transpose_generated)
    obs, gen = _align_features(obs, gen)

    proteins = None
    if args.proteins:
        proteins = [p.strip() for p in args.proteins.split(",") if p.strip()]
        if not proteins:
            proteins = None

    selected = _plot_distribution_overlays(
        obs=obs,
        gen=gen,
        out_path=args.output_dir / "01_distribution_overlays.png",
        n_proteins=int(args.n_proteins),
        protein_seed=int(args.protein_seed),
        proteins=proteins,
        bins=int(args.hist_bins),
    )

    _plot_pca_overlay(
        obs=obs,
        gen=gen,
        out_path=args.output_dir / "02_pca_overlay.png",
        max_points_per_set=int(args.max_pc_points),
        seed=int(args.protein_seed),
    )

    _plot_covariance_heatmaps(
        obs=obs,
        gen=gen,
        out_obs=args.output_dir / "03_cov_heatmap_observed_hac.png",
        out_gen=args.output_dir / "04_cov_heatmap_generated_in_observed_order.png",
        out_order=args.output_dir / "03b_observed_hac_protein_order.txt",
    )

    print(f"[OK] Shared proteins: {obs.shape[1]}")
    print(f"[OK] Observed samples: {obs.shape[0]}")
    print(f"[OK] Generated samples: {gen.shape[0]}")
    print(f"[OK] Distribution proteins plotted: {selected}")
    print(f"[OK] Saved outputs in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

