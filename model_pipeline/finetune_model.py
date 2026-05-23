from __future__ import annotations

import os

import torch

from model_pipeline.model_registry import instantiate_from_checkpoint, load_checkpoint
from model_pipeline.train_model import Trainer, build_windowed_loaders, set_random_seed


class Finetuner:
    def __init__(self, model_state_dir, finetune_csv_dir, dataset, model_save_dir, seed=42):
        set_random_seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = load_checkpoint(model_state_dir, map_location=self.device)
        self.model = instantiate_from_checkpoint(checkpoint, map_location=self.device)
        self.model_name = getattr(self.model, "display_name", self.model.__class__.__name__)
        self.window_length = self.model.get_window_size()
        self.appliance_name_formatted = checkpoint.get("appliance", "appliance")
        self.dataset = dataset
        self.model_save_dir = model_save_dir

        if not getattr(self.model, "supports_gradient", False):
            raise TypeError("Fine-tuning is only supported for gradient-based PyTorch models.")

        if hasattr(self.model, "freeze_for_finetuning"):
            self.model.freeze_for_finetuning()

        train_loader, _ = build_windowed_loaders(
            [finetune_csv_dir],
            window_size=self.model.get_window_size(),
            target_mode=self.model.get_target_type(),
            output_size=self.model.get_output_size(),
            output_offset=self.model.get_output_offset(),
            batch_size=1000,
            val_ratio=0.0,
            seed=seed,
        )

        self.trainer = Trainer(
            model=self.model,
            train_loader=train_loader,
            validation_loader=None,
            appliance=self.appliance_name_formatted,
            dataset=self.dataset,
            model_save_dir=self.model_save_dir,
            result_dir=os.path.join("result", self.dataset.lower(), self.appliance_name_formatted),
            seed=seed,
            batch_size=1000,
            device=self.device,
        )
        self.finetuning_losses = self.trainer.train_losses

    def fineTune(self, max_epochs=50):
        self.trainer.trainModel(num_epochs=max_epochs)

    def plotLosses(self, save_location=None):
        self.trainer.plotLosses()
        if save_location:
            pass
