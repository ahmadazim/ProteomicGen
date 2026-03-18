from typing import Any

import torch
from torch.utils._pytree import tree_map

from scldm._utils import create_anndata_from_inference_output
from scldm.constants import LossEnum, ModelEnum
from scldm.distributions import log_gaussian, log_nb_positive
from scldm.models import VAE as UpstreamVAE

try:
    from scvi.distributions import NegativeBinomial as NegativeBinomialSCVI
except Exception:
    NegativeBinomialSCVI = None  # type: ignore[assignment]


class VAE(UpstreamVAE):
    def loss(
        self,
        counts: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        head_name = self.vae_model.decoder_head.__class__.__name__
        if head_name == "GaussianTransformerLayer":
            recon_loss = log_gaussian(counts, params["mu"])
        else:
            recon_loss = -log_nb_positive(counts, params["mu"], params["theta"])
        return {
            LossEnum.LLH_LOSS.value: recon_loss.sum(dim=1).mean(),
        }

    @torch.no_grad()
    def shared_step(self, batch, batch_idx, stage: str, ema: bool = False) -> dict[str, Any]:
        counts, genes = batch[ModelEnum.COUNTS.value], batch[ModelEnum.GENES.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]

        params, _ = self.vae_model(
            counts,
            genes,
            library_size,
            counts_subset,
            genes_subset,
        )

        loss_output = self.loss(
            counts=counts,
            params=params,
        )

        loss = sum(loss_output.values())
        metrics = {f"{stage}_loss": loss}
        for key, value in loss_output.items():
            metrics[f"{stage}_{key}"] = value

        head_name = self.vae_model.decoder_head.__class__.__name__
        if head_name == "GaussianTransformerLayer":
            counts_pred = params["mu"]
            counts_pred_scaled = counts_pred
            counts_true_scaled = counts
        else:
            if NegativeBinomialSCVI is None:
                raise ImportError(
                    "NegativeBinomialSCVI requires scvi-tools. Install scvi-tools to use NB decoder paths."
                )
            counts_pred = NegativeBinomialSCVI(mu=params["mu"], theta=params["theta"]).sample()
            counts_pred_scaled = torch.log1p((counts_pred / counts_pred.sum(dim=1, keepdim=True)) * 10_000)
            counts_true_scaled = torch.log1p((counts / counts.sum(dim=1, keepdim=True)) * 10_000)

        if head_name != "GaussianTransformerLayer":
            counts_pred_zeros = (counts_pred == 0).float()
            counts_true_zeros = (counts == 0).float()
            metrics[f"{stage}_zeros_accuracy"] = (counts_pred_zeros == counts_true_zeros).float().mean()

        for key, fn in self.metric_fns.items():
            metrics[f"{stage}_{key}"] = torch.nanmean(fn(counts_pred_scaled, counts_true_scaled))
        return metrics

    @torch.no_grad()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        n_samples: int | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        params, z = self.forward(
            counts=batch[ModelEnum.COUNTS.value],
            genes=batch[ModelEnum.GENES.value],
            library_size=batch[ModelEnum.LIBRARY_SIZE.value],
            counts_subset=batch.get(ModelEnum.COUNTS_SUBSET.value),
            genes_subset=batch.get(ModelEnum.GENES_SUBSET.value),
        )
        head_name = self.vae_model.decoder_head.__class__.__name__
        if head_name == "GaussianTransformerLayer":
            counts_pred = params["mu"]
        else:
            if NegativeBinomialSCVI is None:
                raise ImportError(
                    "NegativeBinomialSCVI requires scvi-tools. Install scvi-tools to use NB decoder paths."
                )
            counts_pred = NegativeBinomialSCVI(mu=params["mu"], theta=params["theta"]).sample()
        inference_outputs: dict[str, torch.Tensor] = {
            "reconstructed_counts": counts_pred.cpu(),
            "z": z.cpu(),
        }
        inference_outputs.update({k: batch[k].cpu().numpy() for k in tree_map(lambda x: x.cpu(), batch)})
        return create_anndata_from_inference_output(inference_outputs, self.trainer.datamodule)
