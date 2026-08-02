from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model_pipeline.classical_data import load_grouped_classical_data
from model_pipeline.contracts import SupervisedData, TrainingContext
from model_pipeline.data_protocol import build_train_validation_splits
from model_pipeline.data_feeder import SlidingWindowDataset
from model_pipeline.model_registry import create_model


STATUS_RULES_FILE = os.path.join(os.path.dirname(__file__), "appliance_status_rules.json")


def resolve_appliance_filter_settings(appliance, appliance_on_threshold, min_on_rate):
    if not appliance:
        return appliance_on_threshold, 0.0 if min_on_rate is None else min_on_rate

    try:
        with open(STATUS_RULES_FILE, "r", encoding="utf-8") as handle:
            rules = json.load(handle)
    except FileNotFoundError:
        return appliance_on_threshold, 0.0 if min_on_rate is None else min_on_rate

    rule = rules.get(str(appliance).replace(" ", "_").lower())
    if not rule:
        return appliance_on_threshold, 0.0 if min_on_rate is None else min_on_rate

    if appliance_on_threshold is None:
        appliance_on_threshold = rule.get("segment_filter_threshold", rule.get("min_threshold"))
    if min_on_rate is None:
        min_on_rate = rule.get("min_on_rate", 0.0)
    return appliance_on_threshold, min_on_rate


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")


def get_worker_init_fn(seed=42):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn


def mean_absolute_error(predictions, ground_truth):
    predictions = np.asarray(predictions, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    return float(np.mean(np.abs(predictions - ground_truth)))


def signal_aggregate_error(predictions, ground_truth, eps=1e-8):
    predictions = np.asarray(predictions, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    denominator = max(float(np.sum(np.abs(ground_truth))), eps)
    return float(np.abs(np.sum(predictions) - np.sum(ground_truth)) / denominator)


def compute_metrics(predictions, ground_truth):
    return {
        "MAE": mean_absolute_error(predictions, ground_truth),
        "SAE": signal_aggregate_error(predictions, ground_truth),
    }


def build_windowed_loaders(
    train_csv_paths,
    validation_csv_paths=None,
    *,
    window_size,
    target_mode,
    output_size,
    output_offset,
    crop=None,
    batch_size=256,
    seed=42,
    appliance_on_threshold=None,
    min_on_points=1,
    min_on_rate=0.0,
    status_column=None,
    include_temporal_features=False,
):
    validation_csv_paths = list(validation_csv_paths or [])
    if validation_csv_paths:
        build_train_validation_splits(train_csv_paths, validation_csv_paths)

    train_dataset = SlidingWindowDataset(
        train_csv_paths,
        window_size,
        crop=crop,
        target_mode=target_mode,
        output_size=output_size,
        output_offset=output_offset,
        appliance_on_threshold=appliance_on_threshold,
        min_on_points=min_on_points,
        min_on_rate=min_on_rate,
        status_column=status_column,
        include_temporal_features=include_temporal_features,
    )
    normalisation_stats = train_dataset.get_normalisation_stats()
    validation_dataset = None
    if validation_csv_paths:
        validation_dataset = SlidingWindowDataset(
            validation_csv_paths,
            window_size,
            crop=crop,
            target_mode=target_mode,
            output_size=output_size,
            output_offset=output_offset,
            normalisation_stats=normalisation_stats,
            appliance_on_threshold=appliance_on_threshold,
            min_on_points=min_on_points,
            min_on_rate=min_on_rate,
            status_column=status_column,
            include_temporal_features=include_temporal_features,
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=get_worker_init_fn(seed),
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=get_worker_init_fn(seed),
        )
    return train_loader, validation_loader, normalisation_stats


class Trainer:
    def __init__(
        self,
        model=None,
        train_loader=None,
        validation_loader=None,
        optimizer=None,
        criterion=None,
        *,
        model_name=None,
        train_csv_dirs=None,
        validation_csv_dirs=None,
        appliance=None,
        dataset=None,
        model_save_dir=None,
        window_length=599,
        result_dir=None,
        seed=42,
        batch_size=256,
        crop=None,
        device=None,
        model_init_kwargs=None,
        normalisation_stats=None,
        appliance_on_threshold=None,
        min_on_points=1,
        min_on_rate=None,
        patience=8,
        min_delta=1e-4,
        minimum_epochs=1,
        select_epoch_zero=False,
        checkpoint_callback=None,
        gradient_clip_norm=None,
    ):
        set_random_seed(seed)
        model_init_kwargs = dict(model_init_kwargs or {})

        if model is None:
            if model_name is None:
                raise ValueError("Provide either a model instance or model_name.")
            model_init_kwargs.setdefault("window_size", window_length)
            model = create_model(model_name, **model_init_kwargs)

        self.model = model
        self.model_name = getattr(model, "display_name", model.__class__.__name__)
        self.appliance = appliance or "appliance"
        self.appliance_name_formatted = self.appliance.replace(" ", "_")
        self.dataset = dataset or "unknown"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.seed = seed
        self.crop = crop
        self.train_csv_dirs = train_csv_dirs
        self.validation_csv_dirs = validation_csv_dirs
        self.train_household_split = None
        self.validation_household_split = None
        if train_csv_dirs and validation_csv_dirs:
            self.train_household_split, self.validation_household_split = (
                build_train_validation_splits(train_csv_dirs, validation_csv_dirs)
            )
        self.joint_classical_data = None
        self.normalisation_stats = None if normalisation_stats is None else dict(normalisation_stats)
        appliance_on_threshold, min_on_rate = resolve_appliance_filter_settings(
            self.appliance_name_formatted,
            appliance_on_threshold,
            min_on_rate,
        )

        if getattr(self.model, "supports_gradient", False):
            self.model.to(self.device)

        if (
            train_loader is None
            and train_csv_dirs is not None
            and not (
                not getattr(self.model, "supports_gradient", False)
                and getattr(self.model, "is_joint_model", lambda: False)()
            )
        ):
            train_loader, validation_loader, normalisation_stats = build_windowed_loaders(
                train_csv_dirs,
                validation_csv_dirs,
                window_size=self.model.get_window_size(),
                target_mode=self.model.get_target_type(),
                output_size=self.model.get_output_size(),
                output_offset=self.model.get_output_offset(),
                crop=crop,
                batch_size=batch_size,
                seed=seed,
                appliance_on_threshold=appliance_on_threshold,
                min_on_points=min_on_points,
                min_on_rate=min_on_rate,
                status_column=(
                    "status" if getattr(self.model, "requires_status_targets", False) else None
                ),
                include_temporal_features=bool(
                    getattr(self.model, "requires_temporal_features", False)
                ),
            )
            self.normalisation_stats = normalisation_stats

        configure_normalisation = getattr(self.model, "set_normalisation_stats", None)
        if callable(configure_normalisation) and self.normalisation_stats is not None:
            configure_normalisation(self.normalisation_stats)

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.model_save_dir = model_save_dir or os.path.join(os.getcwd(), "saved_models")
        self.result_dir = result_dir or os.path.join("result", self.dataset.lower(), self.appliance_name_formatted)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.criterion = criterion or nn.MSELoss()
        if getattr(self.model, "supports_gradient", False):
            self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4)
        else:
            self.optimizer = optimizer
            self.scheduler = None

        self.patience = int(patience)
        self.minimum_epochs = int(minimum_epochs)
        if self.minimum_epochs < 1:
            raise ValueError("minimum_epochs must be at least 1.")
        self.select_epoch_zero = bool(select_epoch_zero)
        self.checkpoint_callback = checkpoint_callback
        self.gradient_clip_norm = (
            None if gradient_clip_norm is None else float(gradient_clip_norm)
        )
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided.")
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.current_epoch = None
        self.current_selection_loss = None
        self.min_delta = float(min_delta)
        self.counter = 0
        self.train_losses = []
        self.val_losses = []
        self.selection_losses = []
        self.epoch_history = []

    def _prepare_torch_batch(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = self.model.prepare_targets(targets.to(self.device))
        return inputs, targets

    def _prepare_loader_batch(self, batch):
        hook = getattr(self.model, "prepare_batch", None)
        if callable(hook):
            return hook(batch, device=self.device)
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError(
                "Trainer batches must be (inputs, targets), unless model.prepare_batch is provided."
            )
        return self._prepare_torch_batch(batch[0], batch[1])

    def _should_early_stop(self, completed_epochs):
        return (
            int(completed_epochs) >= self.minimum_epochs
            and self.counter >= self.patience
        )

    def _model_loss_hook(self):
        hook = getattr(self.model, "compute_loss", None)
        return hook if callable(hook) else None

    def _torch_loss(self, inputs, targets):
        hook = self._model_loss_hook()
        if hook is not None:
            loss_output = hook(inputs, targets, criterion=self.criterion)
            if isinstance(loss_output, tuple):
                loss, outputs = loss_output
                return loss, self.model.prepare_outputs(outputs)
            return loss_output, None

        outputs = self.model(inputs)
        outputs = self.model.prepare_outputs(outputs)
        return self.criterion(outputs, targets), outputs

    def _torch_selection_loss(self, inputs, targets):
        hook = getattr(self.model, "compute_selection_loss", None)
        if callable(hook):
            return hook(inputs, targets, criterion=self.criterion)

        loss, _ = self._torch_loss(inputs, targets)
        return loss

    def _collect_numpy_loader(self, loader):
        all_inputs = []
        all_targets = []
        for inputs, targets in loader:
            all_inputs.append(inputs.detach().cpu().numpy())
            all_targets.append(self.model.prepare_targets(targets.detach().cpu().numpy()))
        return np.concatenate(all_inputs, axis=0), np.concatenate(all_targets, axis=0)

    def _save_checkpoint(self):
        if self.checkpoint_callback is not None:
            return self.checkpoint_callback(self)
        appliance_name = self.appliance_name_formatted
        if getattr(self.model, "is_joint_model", lambda: False)():
            appliance_name = "multi_appliance"
        if not getattr(self.model, "is_joint_model", lambda: False)() and self.normalisation_stats is None:
            raise ValueError("Checkpointing requires training normalisation_stats for non-joint models.")
        checkpoint = {
            "model_name": self.model_name,
            "model_key": getattr(self.model, "_registry_key", self.model_name),
            "window_length": self.model.get_window_size(),
            "output_size": self.model.get_output_size(),
            "output_offset": self.model.get_output_offset(),
            "target_type": self.model.get_target_type(),
            "appliance": appliance_name,
            "dataset": self.dataset,
            "init_kwargs": self.model.get_init_kwargs(),
            "model_state": self.model.export_state(),
            "normalisation_stats": self.normalisation_stats,
        }
        if getattr(self.model, "is_joint_model", lambda: False)():
            checkpoint["joint_mode"] = True
            checkpoint["grouping_key"] = self.model.get_grouping_key()
            checkpoint["appliances"] = list(getattr(self.model, "appliance_order", []))
        if getattr(self.model, "supports_gradient", False):
            checkpoint["model_state_dict"] = checkpoint["model_state"]

        path = os.path.join(self.model_save_dir, f"{self.appliance}_{self.dataset}_{self.model_name}.pth")
        torch.save(checkpoint, path)
        return path

    def _plugin_checkpoint_callback(
        self,
        *,
        model,
        epoch,
        train_loss,
        validation_loss,
        selection_loss,
        record,
    ):
        if model is not self.model:
            raise ValueError("Checkpoint callback received a different model instance.")

        self.current_epoch = int(epoch)
        self.current_selection_loss = float(selection_loss)
        if train_loss is not None:
            self.train_losses.append(float(train_loss))
        self.val_losses.append(float(validation_loss))
        self.selection_losses.append(float(selection_loss))
        self.epoch_history.append(dict(record))

        saved = False
        checkpoint_path = None
        if selection_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = float(selection_loss)
            self.best_epoch = int(epoch)
            checkpoint_path = self._save_checkpoint()
            self.counter = 0
            saved = True
            print(f"Validation improved. Checkpoint saved to {checkpoint_path}")
        else:
            self.counter += 1

        should_stop = self._should_early_stop(epoch)
        if should_stop:
            print(f"Early stopping triggered after {epoch} epochs.")
        return {
            "saved": saved,
            "checkpoint_path": checkpoint_path,
            "should_stop": should_stop,
        }

    def _validate_torch(self):
        if self.validation_loader is None:
            return None, None
        self.model.eval()
        val_loss = 0.0
        selection_loss = 0.0
        with torch.no_grad():
            for batch in self.validation_loader:
                inputs, targets = self._prepare_loader_batch(batch)
                loss, _ = self._torch_loss(inputs, targets)
                val_loss += loss.item()
                selection_hook = getattr(self.model, "compute_selection_loss", None)
                if callable(selection_hook):
                    selected = selection_hook(inputs, targets, criterion=self.criterion)
                else:
                    selected = loss
                selection_loss += selected.item()
        val_loss /= max(1, len(self.validation_loader))
        selection_loss /= max(1, len(self.validation_loader))
        return val_loss, selection_loss

    def trainModel(self, num_epochs=10):
        if self.train_loader is None:
            if not (
                not getattr(self.model, "supports_gradient", False)
                and getattr(self.model, "is_joint_model", lambda: False)()
                and self.train_csv_dirs is not None
            ):
                raise ValueError("Trainer requires a train_loader or train_csv_dirs.")

        if not getattr(self.model, "supports_gradient", False):
            if getattr(self.model, "is_joint_model", lambda: False)():
                if self.train_csv_dirs is None:
                    raise ValueError("Joint classical models require train_csv_dirs.")
                self.joint_classical_data = load_grouped_classical_data(self.train_csv_dirs)
                self.model.fit_joint(self.joint_classical_data)
                self.train_losses.append(0.0)
                self.val_losses.append(0.0)
                checkpoint_path = self._save_checkpoint()
                print(f"Joint classical model fitted. Checkpoint saved to {checkpoint_path}")
                return

            train_inputs, train_targets = self._collect_numpy_loader(self.train_loader)
            self.model.fit(train_inputs, train_targets)
            train_predictions = self.model.prepare_outputs(self.model.disaggregate(train_inputs))
            train_loss = float(self.criterion(
                torch.as_tensor(train_predictions),
                torch.as_tensor(train_targets),
            ).item())
            self.train_losses.append(train_loss)

            val_loss = train_loss
            if self.validation_loader is not None and len(self.validation_loader) > 0:
                val_inputs, val_targets = self._collect_numpy_loader(self.validation_loader)
                val_predictions = self.model.prepare_outputs(self.model.disaggregate(val_inputs))
                val_loss = float(self.criterion(
                    torch.as_tensor(val_predictions),
                    torch.as_tensor(val_targets),
                ).item())
            self.val_losses.append(val_loss)
            self.best_val_loss = val_loss
            checkpoint_path = self._save_checkpoint()
            print(f"Classical model fitted. Checkpoint saved to {checkpoint_path}")
            return

        if num_epochs < self.minimum_epochs:
            raise ValueError(
                f"num_epochs ({num_epochs}) must be >= minimum_epochs ({self.minimum_epochs})."
            )

        train_data = SupervisedData(
            household_split=self.train_household_split,
            loader=self.train_loader,
            normalisation_stats=self.normalisation_stats,
        )
        validation_data = SupervisedData(
            household_split=self.validation_household_split,
            loader=self.validation_loader,
            normalisation_stats=self.normalisation_stats,
        )
        checkpoint_path = Path(self.model_save_dir) / (
            f"{self.appliance}_{self.dataset}_{self.model_name}.pth"
        )
        context = TrainingContext(
            device=self.device,
            seed=self.seed,
            num_epochs=int(num_epochs),
            checkpoint_path=checkpoint_path,
            checkpoint_callback=self._plugin_checkpoint_callback,
            config={
                "criterion": self.criterion,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "gradient_clip_norm": self.gradient_clip_norm,
                "select_epoch_zero": self.select_epoch_zero,
            },
        )
        return self.model.fit(train_data, validation_data, context)

    def plotLosses(self):
        if not self.train_losses:
            return
        plt.plot(self.train_losses, label="Train Loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Loss for {self.appliance_name_formatted} on {self.dataset} using {self.model_name}")
        plot_filename = f"{self.appliance_name_formatted}_{self.dataset}_{self.model_name}_loss.png"
        plot_path = os.path.join(self.result_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss plot saved to {plot_path}")
