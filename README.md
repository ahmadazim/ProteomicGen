# ProteomicGen: Generating High-Fidelity Proteomic Data

Reproducible training and generation pipeline for the proteomics workflow built on top of upstream `scLDM` plus a local USiT latent diffusion model.

## What This Repo Does

The full pipeline is:

1. Train a proteomics VAE with upstream `scLDM`.
2. Encode the observed proteomics matrix into VAE latents.
3. Train a USiT latent diffusion model on those latents.
4. Sample synthetic latents with USiT and decode them back to proteomics profiles.
5. Produce comparison plots between observed and generated data.

The main orchestrator is `scripts/submit.sh`.

## Repository Layout

- `data/`: input data expected by the pipeline.
- `scripts/`: runnable entrypoints for VAE training, encoding, decoding, USiT training, generation, and visualization.
- `external/scldm/`: clean upstream `scLDM` clone.
- `external/SiT.py`: local copy of the SiT/USiT implementation used by `trainLDM.py` and generation.
- `src/genprot_scldm/`: local compatibility overrides needed for the proteomics VAE workflow. These are injected automatically by the wrapper scripts, so users do not need to edit `external/scldm`.
- `models/`: exported VAE and USiT checkpoints.
- `generated/`: decoded synthetic outputs and QC figures.
- `scldm_proteomics_run/`: scLDM working directory containing intermediate training artifacts.

## Expected Inputs

This repo assumes the data are already prepared in `data/`.

By default, `scripts/submit.sh` expects:

- `data/Plasma_proteomic_data_imputed_scaled.tsv`

The default latent and output names used by the pipeline are created automatically.

## Environment Setup

Clone upstream `scLDM` into `external/scldm`, then install its dependencies in your environment.

Example:

```bash
conda create -n scldm python=3.11 -y
conda activate scldm
pip install -e external/scldm
pip install "cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git"
```

## End-To-End Run

Run on SLURM:

```bash
sbatch scripts/submit.sh
```


## Step-By-Step Commands 

See `scripts/submit.sh` for more details. An example run is as follows.

Train the VAE:

```bash
python scripts/train_proteomics_vae.py \
  --tsv-path data/Plasma_proteomic_data_imputed_scaled.tsv \
  --work-dir scldm_proteomics_run \
  --experiment-name vae_proteomicsUKB_XL
```

Encode observed data:

```bash
python scripts/encode_proteomics_vae.py \
  --ckpt-path models/vae_proteomicsUKB_XL_last.ckpt \
  --config-path models/vae_proteomicsUKB_XL_config.yaml \
  --input-tsv data/Plasma_proteomic_data_imputed_scaled.tsv \
  --output-pt data/latents_Plasma_proteomic.pt \
  --metadata-json models/vae_proteomicsUKB_XL_metadata.json
```

Train USiT:

```bash
python scripts/trainLDM.py \
  --train-latents-path data/latents_Plasma_proteomic.pt \
  --model-output-path models/USiT_M_Plasma_proteomic.pth
```

Generate and decode synthetic data:

```bash
python scripts/generate_proteomics_from_usit.py \
  --usit-ckpt models/USiT_M_Plasma_proteomic.pth \
  --n-syn 50000 \
  --output-dir generated/generated_Plasma_proteomic_USiT_M \
  --vae-ckpt models/vae_proteomicsUKB_XL_last.ckpt \
  --vae-config models/vae_proteomicsUKB_XL_config.yaml \
  --metadata-json models/vae_proteomicsUKB_XL_metadata.json
```

Visualize observed vs generated outputs:

```bash
python scripts/visualize_proteomics_comparison.py \
  --observed-tsv data/Plasma_proteomic_data_imputed_scaled.tsv \
  --generated-tsv generated/generated_Plasma_proteomic_USiT_M/synthetic_proteomics_n50000.tsv \
  --output-dir generated/generated_Plasma_proteomic_USiT_M
```

## Outputs

After a successful run, the main outputs are:

- `models/vae_proteomicsUKB_XL_last.ckpt`
- `models/vae_proteomicsUKB_XL_config.yaml`
- `models/vae_proteomicsUKB_XL_metadata.json`
- `data/latents_Plasma_proteomic.pt`
- `models/USiT_M_Plasma_proteomic.pth`
- `generated/generated_Plasma_proteomic_USiT_M/synthetic_proteomics_n50000.tsv`
- `generated/generated_Plasma_proteomic_USiT_M/01_distribution_overlays.png`
- `generated/generated_Plasma_proteomic_USiT_M/02_pca_overlay.png`
- `generated/generated_Plasma_proteomic_USiT_M/03_cov_heatmap_observed_hac.png`
- `generated/generated_Plasma_proteomic_USiT_M/04_cov_heatmap_generated_in_observed_order.png`
