from typing import Any

from torch.utils.data import DataLoader

from scldm.datamodule import DataModule as UpstreamDataModule
from scldm.datamodule import collate_fn
from scldm.logger import logger


class DataModule(UpstreamDataModule):
    def _effective_num_workers(self) -> int:
        workers = max(0, int(self.num_workers))
        if workers > 1:
            logger.warning(
                "Capping DataLoader num_workers from %s to 1 for iterable AnnData dataset stability.",
                workers,
            )
            return 1
        return workers

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage=stage)
        train_dataset = getattr(self, "train_dataset", None)
        if train_dataset is not None:
            for attr, value in {
                "shuffle": True,
                "drop_last_indices": self.drop_last_indices,
                "drop_incomplete_batch": self.drop_incomplete_batch,
            }.items():
                if hasattr(train_dataset, attr):
                    setattr(train_dataset, attr, value)

    def _loader_kwargs(self, *, drop_last: bool) -> dict[str, Any]:
        workers = self._effective_num_workers()
        loader_kwargs: dict[str, Any] = {
            "collate_fn": collate_fn,
            "num_workers": workers,
            "drop_last": drop_last,
            "pin_memory": True,
            "persistent_workers": self.persistent_workers and workers > 0,
        }
        if workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return loader_kwargs

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._loader_kwargs(drop_last=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._loader_kwargs(drop_last=True))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._loader_kwargs(drop_last=False))

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, **self._loader_kwargs(drop_last=False))
