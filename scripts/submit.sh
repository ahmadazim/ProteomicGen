#!/usr/bin/env bash
#SBATCH -p hsph_gpu,gpu_h200,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_INTERFACE_LAYER=GNU,LP64
export MKL_THREADING_LAYER=GNU
export MKL_DYNAMIC=TRUE
export MKL_NUM_THREADS=1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

if command -v module >/dev/null 2>&1 && [[ -n "${MODULES_TO_LOAD:-}" ]]; then
  for module_name in ${MODULES_TO_LOAD}; do
    module load "${module_name}"
  done
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models}"
GENERATED_DIR="${GENERATED_DIR:-${REPO_ROOT}/generated}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/scldm_proteomics_run}"
mkdir -p "${MODEL_DIR}" "${GENERATED_DIR}" "${WORK_DIR}"

DATA_FILENAME=Plasma_proteomic_data_imputed_scaled.tsv
LATENTS_FILENAME=latents_Plasma_proteomic.pt
GENERATED_SUBDIR=generated_Plasma_proteomic_USiT_M/

VAE_NAME="${VAE_NAME:-vae_proteomicsUKB_XL}"
LDM_NAME="${LDM_NAME:-USiT_M_Plasma_proteomic}"
NUM_GEN="${NUM_GEN:-50000}"
VAE_CKPT="${MODEL_DIR}/${VAE_NAME}_last.ckpt"
VAE_CONFIG="${MODEL_DIR}/${VAE_NAME}_config.yaml"
VAE_METADATA="${MODEL_DIR}/${VAE_NAME}_metadata.json"
LDM_CKPT="${MODEL_DIR}/${LDM_NAME}.pth"
LATENTS_PATH="${DATA_DIR}/${LATENTS_FILENAME}"
GENERATED_OUTPUT_DIR="${GENERATED_DIR}/${GENERATED_SUBDIR}"

# Train VAE
${PYTHON_BIN} "${SCRIPT_DIR}/train_proteomics_vae.py" \
  --tsv-path "${DATA_DIR}/${DATA_FILENAME}" \
  --work-dir "${WORK_DIR}" \
  --experiment-name "${VAE_NAME}" \
  --epochs 40 \
  --batch-size 32 \
  --test-frac 0.1 \
  --n-embed 192 \
  --n-layer 8 \
  --n-head 8 \
  --n-head-cross 4 \
  --n-inducing-points 256 \
  --n-embed-latent 32

cp "${WORK_DIR}/experiments/checkpoints/${VAE_NAME}/last.ckpt" "${VAE_CKPT}"
cp "${WORK_DIR}/experiments/checkpoints/${VAE_NAME}/config.yaml" "${VAE_CONFIG}"
cp "${WORK_DIR}/data/proteomics_metadata.json" "${VAE_METADATA}"

# Encode data
${PYTHON_BIN} -u "${SCRIPT_DIR}/encode_proteomics_vae.py" \
  --ckpt-path "${VAE_CKPT}" \
  --config-path "${VAE_CONFIG}" \
  --input-tsv "${DATA_DIR}/${DATA_FILENAME}" \
  --output-pt "${LATENTS_PATH}" \
  --batch-size 32 \
  --device cuda \
  --metadata-json "${WORK_DIR}/data/proteomics_metadata.json"

# Train LDM
${PYTHON_BIN} -u "${SCRIPT_DIR}/trainLDM.py" \
  --train-latents-path "${LATENTS_PATH}" \
  --model-output-path "${LDM_CKPT}" \
  --latent-length 256 \
  --token-dim 32 \
  --batch-size 64 \
  --num-epochs 100 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --warmup-steps 100 \
  --ema-decay 0.999 \
  --seed 42 \
  --num-workers 0 \
  --save-every 1 \
  --num-layers 17 \
  --num-heads 12 \
  --mlp-ratio 4 \
  --hidden-dim 768 \
  --qkv-bias \
  --device cuda \
  --use-bf16

# Generate
${PYTHON_BIN} -u "${SCRIPT_DIR}/generate_proteomics_from_usit.py" \
  --usit-ckpt "${LDM_CKPT}" \
  --n-syn "${NUM_GEN}" \
  --output-dir "${GENERATED_OUTPUT_DIR}" \
  --vae-ckpt "${VAE_CKPT}" \
  --vae-config "${VAE_CONFIG}" \
  --metadata-json "${VAE_METADATA}" \
  --decode-script "${SCRIPT_DIR}/decode_proteomics_vae.py" \
  --steps 50 \
  --solver heun \
  --batch-size 256 \
  --decode-batch-size 128 \
  --seed 42 \
  --device cuda \
  --library-size-value 1.0

# Visualize
${PYTHON_BIN} -u "${SCRIPT_DIR}/visualize_proteomics_comparison.py" \
  --observed-tsv "${DATA_DIR}/${DATA_FILENAME}" \
  --generated-tsv "${GENERATED_OUTPUT_DIR}/synthetic_proteomics_n${NUM_GEN}.tsv" \
  --output-dir "${GENERATED_OUTPUT_DIR}" \
  --n-proteins 12 \
  --protein-seed 42 \
  --hist-bins 50 \
  --max-pc-points 1000
