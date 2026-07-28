#!/usr/bin/env python3
"""Run the fixed REFIT washing-machine context-routed Adapter Bank experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nilm-matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_seq2point_context_film import (  # noqa: E402
    _git_state,
    _json_dump,
    _reject_existing,
    _save_best,
    _torch_load,
    load_config,
    set_seed,
)
from model_pipeline.context_data_feeder import (  # noqa: E402
    HouseholdUniformBatchSampler,
    LeakageGuard,
    MultiHouseQueryDataset,
    NormalisationStats,
    QueryWindowDataset,
    assert_common_evaluation_timestamps,
    read_refit_block,
    resolve_block_specs,
    select_uniform_context_windows,
    timestamp_fingerprint,
)
from model_pipeline.models.context_adapter.seq2point_adapter_bank import (  # noqa: E402
    ContextRoutedAdapterBankSeq2Point,
    GlobalSharedAdapterSeq2Point,
)
from model_pipeline.models.context_adapter.seq2point_film import module_sha256  # noqa: E402
from model_pipeline.models.seq2point.seq2point import Seq2Point  # noqa: E402
from model_pipeline.test_model import Evaluator  # noqa: E402
from model_pipeline.train_model import Trainer  # noqa: E402


class HouseholdRoutedTrainerModel(ContextRoutedAdapterBankSeq2Point):
    """Pair a one-house query batch with that household's cached A windows."""

    def configure_training_contexts(
        self,
        contexts: dict[str, tuple[torch.Tensor, torch.Tensor]],
        train_houses: list[str],
        validation_house: str,
        device: torch.device,
    ) -> None:
        self._training_contexts = [
            tuple(value.to(device) for value in contexts[house])
            for house in train_houses
        ]
        self._validation_context = tuple(
            value.to(device) for value in contexts[validation_house]
        )

    def prepare_batch(self, batch, *, device):
        if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
            raise ValueError(
                "Routed Adapter batches require query, target, and optional house id."
            )
        query = batch[0].to(device)
        target = self.prepare_targets(batch[1].to(device))
        house_indices = None if len(batch) == 2 else batch[2].to(device)
        return {"query": query, "house_indices": house_indices}, target

    def compute_loss(self, inputs, targets, *, criterion):
        query = inputs["query"]
        house_indices = inputs["house_indices"]
        if house_indices is None:
            windows, mask = self._validation_context
        else:
            unique_houses = torch.unique(house_indices)
            if len(unique_houses) != 1:
                raise RuntimeError(
                    "Each routed Adapter optimizer step must contain exactly one household."
                )
            windows, mask = self._training_contexts[int(unique_houses.item())]
        _, route = self.generate_route(
            windows.unsqueeze(0), mask.unsqueeze(0)
        )
        prediction = self.forward_with_route(query, route).reshape(-1)
        return criterion(prediction, targets)


class Experiment:
    def __init__(self, config_path: Path, device_name: str | None):
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        self._validate_fixed_protocol()
        self.seed = int(self.config["seed"])
        set_seed(self.seed)
        self.device = torch.device(
            device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.run_dir = REPO_ROOT / self.config["artifacts_root"]
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.prediction_dir = self.run_dir / "predictions"
        self.manifest_path = self.run_dir / "manifest.json"
        self.run_manifest_path = self.run_dir / "run_manifest.json"
        self.specs = resolve_block_specs(self.config, REPO_ROOT)
        self.guard = LeakageGuard(test_house="H6")
        self.chunksize = int(self.config["data"].get("chunksize", 250_000))
        self.reference_manifest_path = (
            REPO_ROOT / self.config["film_reference"]["manifest"]
        )
        self.reference_predictions_path = (
            REPO_ROOT / self.config["film_reference"]["predictions"]
        )
        self.reference_manifest = json.loads(
            self.reference_manifest_path.read_text(encoding="utf-8")
        )
        self._validate_reference()

    @property
    def baseline_path(self) -> Path:
        return REPO_ROOT / self.reference_manifest["checkpoints"]["M0"]

    @property
    def global_path(self) -> Path:
        return self.checkpoint_dir / "global_shared_adapter_best.pt"

    @property
    def context_path(self) -> Path:
        return self.checkpoint_dir / "context_routed_adapter_best.pt"

    def _validate_fixed_protocol(self) -> None:
        expected = {
            "seed": 42,
            "window_size": 599,
            "hidden_dim": 1024,
            "context_k": 16,
            "context_selector": "uniform",
            "context_code_dim": 128,
            "num_adapters": 2,
            "adapter_bottleneck_dim": 64,
            "residual_scale": 0.1,
            "router_temperature": 1.0,
            "routing": "softmax",
            "hard_routing": False,
        }
        mismatches = {
            key: (self.config.get(key), value)
            for key, value in expected.items()
            if self.config.get(key) != value
        }
        if self.config["house_split"] != {
            "train": ["H1", "H2", "H3", "H7"],
            "validation": ["H5"],
            "test": ["H6"],
        }:
            mismatches["house_split"] = self.config["house_split"]
        training = self.config["training"]
        for key, value in {
            "optimizer": "AdamW",
            "adapter_learning_rate": 0.001,
            "weight_decay": 0.0001,
            "max_epochs": 20,
            "minimum_epochs": 1,
            "patience": 8,
            "gradient_clip_norm": 5.0,
            "batch_size": 256,
        }.items():
            if training.get(key) != value:
                mismatches[f"training.{key}"] = (training.get(key), value)
        if mismatches:
            raise ValueError(f"Fixed protocol mismatch: {mismatches}")

    def _validate_reference(self) -> None:
        reference = self.reference_manifest
        if reference["house_split"] != self.config["house_split"]:
            raise RuntimeError("FiLM reference household split differs from Adapter config.")
        if reference["fixed_blocks"] != self._fixed_blocks():
            raise RuntimeError("FiLM reference A/B blocks differ from Adapter config.")
        if int(reference.get("evaluation_timestamp_count", -1)) != 95060:
            raise RuntimeError("FiLM reference does not contain the required 95,060 timestamps.")
        if not self.baseline_path.exists():
            raise FileNotFoundError(self.baseline_path)
        if not self.reference_predictions_path.exists():
            raise FileNotFoundError(self.reference_predictions_path)

    def _fixed_blocks(self) -> dict:
        return {
            house: {
                block: {
                    "start": spec.start,
                    "end": spec.end,
                    "csv_path": str(Path(spec.csv_path).relative_to(REPO_ROOT)),
                }
                for block, spec in blocks.items()
            }
            for house, blocks in self.specs.items()
        }

    def _prepare_artifacts(self) -> dict:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        output_config = self.run_dir / "config.yaml"
        config_sha = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        if output_config.exists():
            if hashlib.sha256(output_config.read_bytes()).hexdigest() != config_sha:
                raise RuntimeError("Existing artifact config.yaml differs from requested config.")
        else:
            shutil.copyfile(self.config_path, output_config)
        _json_dump(
            self.run_dir / "normalization.json",
            {
                "stats": self.reference_manifest["normalisation_stats"],
                "provenance": self.reference_manifest["normalisation_provenance"],
                "source_manifest": str(
                    self.reference_manifest_path.relative_to(REPO_ROOT)
                ),
            },
        )
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest["config_sha256"] != config_sha:
                raise RuntimeError("Experiment config changed after artifact creation.")
            return manifest
        manifest = {
            "schema_version": 1,
            "experiment": self.config["experiment_name"],
            "config_sha256": config_sha,
            "config": deepcopy(self.config),
            "dataset_path": self.config["data"]["csv_dir"],
            "house_split": deepcopy(self.config["house_split"]),
            "fixed_blocks": self._fixed_blocks(),
            "normalisation_stats": deepcopy(
                self.reference_manifest["normalisation_stats"]
            ),
            "normalisation_provenance": deepcopy(
                self.reference_manifest["normalisation_provenance"]
            ),
            "film_reference_manifest": str(
                self.reference_manifest_path.relative_to(REPO_ROOT)
            ),
            "baseline_checkpoint": str(self.baseline_path.relative_to(REPO_ROOT)),
            "git": _git_state(),
            "seed": self.seed,
            "checkpoints": {},
            "best_epochs": {},
            "training_history": {},
            "access_audit": [],
        }
        _json_dump(self.manifest_path, manifest)
        return manifest

    def _update_manifest(self, manifest: dict) -> None:
        manifest["access_audit"].extend(self.guard.audit_dicts())
        self.guard.records.clear()
        _json_dump(self.manifest_path, manifest)

    def _stats(self) -> NormalisationStats:
        return NormalisationStats(**self.reference_manifest["normalisation_stats"])

    def _read(
        self, house: str, block: str, columns: list[str], purpose: str
    ) -> pd.DataFrame:
        return read_refit_block(
            self.specs[house][block],
            columns=columns,
            purpose=purpose,
            guard=self.guard,
            chunksize=self.chunksize,
        )

    def _query_dataset(
        self,
        frame: pd.DataFrame,
        house: str,
        *,
        require_target: bool = True,
    ) -> QueryWindowDataset:
        return QueryWindowDataset(
            frame,
            house=house,
            stats=self._stats(),
            window_size=599,
            appliance="washing_machine",
            require_target=require_target,
        )

    def _loader(
        self,
        dataset,
        *,
        shuffle: bool,
        batch_size: int | None = None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size or 256,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=int(self.config["training"]["num_workers"]),
        )

    def _load_baseline(self) -> tuple[Seq2Point, dict]:
        checkpoint = _torch_load(self.baseline_path, self.device)
        baseline = Seq2Point(**checkpoint["init_kwargs"])
        baseline.load_state_dict(checkpoint["model_state"])
        baseline.to(self.device)
        actual_hash = module_sha256(baseline)
        if actual_hash != checkpoint["baseline_sha256"]:
            raise RuntimeError("Referenced Seq2Point checkpoint hash mismatch.")
        if any(parameter.requires_grad is False for parameter in baseline.parameters()):
            raise RuntimeError("Baseline checkpoint unexpectedly loaded already frozen.")
        baseline.eval()
        sample = torch.randn(2, 599, device=self.device)
        with torch.no_grad():
            direct = baseline.network(sample.unsqueeze(1))
            split = baseline.decode(baseline.encode(sample))
        torch.testing.assert_close(direct, split, rtol=1e-6, atol=1e-6)
        return baseline, checkpoint

    def _adapter_frames(self, include_context: bool):
        houses = self.config["house_split"]["train"]
        train_b = {
            house: self._read(
                house,
                "B",
                ["aggregate", "washing_machine"],
                "adapter_training_query",
            )
            for house in houses
        }
        val_house = "H5"
        val_b = self._read(
            val_house,
            "B",
            ["aggregate", "washing_machine"],
            "checkpoint_selection",
        )
        contexts = None
        if include_context:
            contexts = {
                house: self._read(house, "A", ["aggregate"], "context")
                for house in [*houses, val_house]
            }
        return houses, train_b, val_house, val_b, contexts

    def _context_tensors(self, frames: dict[str, pd.DataFrame]):
        tensors = {}
        selections = {}
        for house, frame in frames.items():
            windows, mask, timestamps = select_uniform_context_windows(
                frame,
                stats=self._stats(),
                window_size=599,
                context_k=16,
            )
            tensors[house] = (windows, mask)
            selections[house] = [str(value) for value in timestamps]
        return tensors, selections

    def _new_context_adapter(
        self, baseline: Seq2Point, *, for_training: bool = False
    ) -> ContextRoutedAdapterBankSeq2Point:
        cls = (
            HouseholdRoutedTrainerModel
            if for_training
            else ContextRoutedAdapterBankSeq2Point
        )
        return cls(
            baseline,
            window_size=599,
            code_dim=128,
            num_adapters=2,
            bottleneck_dim=64,
            residual_scale=0.1,
            router_temperature=1.0,
        )

    def _identity_check(
        self,
        baseline: Seq2Point,
        adapter: nn.Module,
        query: torch.Tensor,
        context=None,
    ) -> float:
        baseline.eval()
        adapter.eval()
        with torch.no_grad():
            expected = baseline(query.to(self.device))
            actual = (
                adapter(query.to(self.device))
                if context is None
                else adapter(query.to(self.device), *context)
            )
        error = float(torch.max(torch.abs(expected - actual)).item())
        if error > 1e-6:
            raise RuntimeError(f"Adapter zero-init identity failed: {error:.9g}")
        return error

    def _trainer(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        callback,
    ) -> Trainer:
        training = self.config["training"]
        return Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            appliance="washing_machine",
            dataset="REFIT",
            model_save_dir=str(self.checkpoint_dir),
            result_dir=str(self.run_dir),
            seed=self.seed,
            device=str(self.device),
            normalisation_stats=self._stats().to_dict(),
            patience=int(training["patience"]),
            min_delta=float(training["min_delta"]),
            minimum_epochs=int(training["minimum_epochs"]),
            select_epoch_zero=False,
            checkpoint_callback=callback,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )

    def train_global_adapter(self) -> None:
        manifest = self._prepare_artifacts()
        _reject_existing(self.global_path)
        houses, train_b, val_house, val_b, _ = self._adapter_frames(False)
        baseline, baseline_checkpoint = self._load_baseline()
        adapter = GlobalSharedAdapterSeq2Point(
            baseline, bottleneck_dim=64, residual_scale=0.1
        ).to(self.device)
        adapter.assert_baseline_frozen()
        backbone_hash = module_sha256(adapter.baseline)
        train_dataset = ConcatDataset(
            [self._query_dataset(train_b[house], house) for house in houses]
        )
        val_dataset = self._query_dataset(val_b, val_house)
        identity_error = self._identity_check(
            adapter.baseline, adapter, val_dataset[0][0].unsqueeze(0)
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in adapter.parameters() if parameter.requires_grad],
            lr=0.001,
            weight_decay=0.0001,
        )

        def save_global(trainer):
            checkpoint = {
                "method": "global_shared_adapter",
                "epoch": trainer.current_epoch,
                "validation_mse": trainer.current_selection_loss,
                "identity_max_abs_error": identity_error,
                "baseline_checkpoint": str(self.baseline_path.relative_to(REPO_ROOT)),
                "baseline_sha256": baseline_checkpoint["baseline_sha256"],
                "adapter_state": trainer.model.adapter.state_dict(),
                "backbone_sha256_before": backbone_hash,
                "seed": self.seed,
            }
            _save_best(self.global_path, checkpoint)
            return self.global_path

        trainer = self._trainer(
            adapter,
            self._loader(train_dataset, shuffle=True),
            self._loader(val_dataset, shuffle=False),
            optimizer,
            save_global,
        )
        trainer.trainModel(num_epochs=20)
        after_hash = module_sha256(adapter.baseline)
        if after_hash != backbone_hash:
            raise RuntimeError("Seq2Point changed during global Adapter training.")
        manifest["checkpoints"]["A1"] = str(self.global_path.relative_to(REPO_ROOT))
        manifest["best_epochs"]["A1"] = trainer.best_epoch
        manifest["training_history"]["A1"] = trainer.epoch_history
        manifest["backbone_hash_checks"] = {
            "A1": {
                "before": backbone_hash,
                "after": after_hash,
                "passed": backbone_hash == after_hash,
            }
        }
        manifest["zero_init_checks"] = {
            "A1_max_abs_error": identity_error,
        }
        self._update_manifest(manifest)
        print(f"A1 best epoch: {trainer.best_epoch}; checkpoint: {self.global_path}")

    def train_context_adapter(self) -> None:
        manifest = self._prepare_artifacts()
        if not self.global_path.exists():
            raise FileNotFoundError("Train A1 before A2.")
        _reject_existing(self.context_path)
        houses, train_b, val_house, val_b, context_frames = self._adapter_frames(True)
        baseline, baseline_checkpoint = self._load_baseline()
        adapter = self._new_context_adapter(baseline, for_training=True).to(self.device)
        adapter.assert_baseline_frozen()
        backbone_hash = module_sha256(adapter.baseline)
        contexts, selections = self._context_tensors(context_frames)
        adapter.configure_training_contexts(
            contexts, houses, val_house, self.device
        )
        train_dataset = MultiHouseQueryDataset(
            [self._query_dataset(train_b[house], house) for house in houses]
        )
        sampler = HouseholdUniformBatchSampler(
            train_dataset, batch_size=256, seed=self.seed
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=int(self.config["training"]["num_workers"]),
        )
        val_dataset = self._query_dataset(val_b, val_house)
        windows, mask = contexts[val_house]
        identity_error = self._identity_check(
            adapter.baseline,
            adapter,
            val_dataset[0][0].unsqueeze(0),
            (
                windows.unsqueeze(0).to(self.device),
                mask.unsqueeze(0).to(self.device),
            ),
        )
        with torch.no_grad():
            _, initial_route = adapter.generate_route(
                windows.unsqueeze(0).to(self.device),
                mask.unsqueeze(0).to(self.device),
            )
        torch.testing.assert_close(
            initial_route,
            torch.tensor([[0.5, 0.5]], device=self.device),
            rtol=0,
            atol=0,
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in adapter.parameters() if parameter.requires_grad],
            lr=0.001,
            weight_decay=0.0001,
        )

        def save_context(trainer):
            checkpoint = {
                "method": "correct_context_routed_adapter",
                "shared_methods": ["A2", "A3", "A4"],
                "epoch": trainer.current_epoch,
                "validation_mse": trainer.current_selection_loss,
                "identity_max_abs_error": identity_error,
                "baseline_checkpoint": str(self.baseline_path.relative_to(REPO_ROOT)),
                "baseline_sha256": baseline_checkpoint["baseline_sha256"],
                "context_encoder_state": trainer.model.context_encoder.state_dict(),
                "router_state": trainer.model.router.state_dict(),
                "adapters_state": trainer.model.adapters.state_dict(),
                "backbone_sha256_before": backbone_hash,
                "seed": self.seed,
            }
            _save_best(self.context_path, checkpoint)
            return self.context_path

        trainer = self._trainer(
            adapter,
            train_loader,
            self._loader(val_dataset, shuffle=False),
            optimizer,
            save_context,
        )
        trainer.trainModel(num_epochs=20)
        after_hash = module_sha256(adapter.baseline)
        if after_hash != backbone_hash:
            raise RuntimeError("Seq2Point changed during routed Adapter training.")
        shared = str(self.context_path.relative_to(REPO_ROOT))
        manifest["checkpoints"].update({"A2": shared, "A3": shared, "A4": shared})
        manifest["best_epochs"].update(
            {"A2": trainer.best_epoch, "A3": trainer.best_epoch, "A4": trainer.best_epoch}
        )
        manifest["training_history"]["A2"] = trainer.epoch_history
        manifest["context_window_timestamps"] = selections
        manifest["backbone_hash_checks"]["A2"] = {
            "before": backbone_hash,
            "after": after_hash,
            "passed": backbone_hash == after_hash,
        }
        manifest["zero_init_checks"]["A2_max_abs_error"] = identity_error
        self._update_manifest(manifest)
        print(f"A2 best epoch: {trainer.best_epoch}; checkpoint: {self.context_path}")

    def _load_global(self, baseline: Seq2Point) -> GlobalSharedAdapterSeq2Point:
        checkpoint = _torch_load(self.global_path, self.device)
        if checkpoint["baseline_sha256"] != module_sha256(baseline):
            raise RuntimeError("A1 references a different Seq2Point checkpoint.")
        model = GlobalSharedAdapterSeq2Point(
            baseline, bottleneck_dim=64, residual_scale=0.1
        ).to(self.device)
        model.adapter.load_state_dict(checkpoint["adapter_state"])
        model.eval()
        return model

    def _load_context(
        self, baseline: Seq2Point
    ) -> ContextRoutedAdapterBankSeq2Point:
        checkpoint = _torch_load(self.context_path, self.device)
        if checkpoint["baseline_sha256"] != module_sha256(baseline):
            raise RuntimeError("A2/A3/A4 reference a different Seq2Point checkpoint.")
        model = self._new_context_adapter(baseline).to(self.device)
        model.context_encoder.load_state_dict(checkpoint["context_encoder_state"])
        model.router.load_state_dict(checkpoint["router_state"])
        model.adapters.load_state_dict(checkpoint["adapters_state"])
        model.eval()
        return model

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _predict(
        self,
        dataset: QueryWindowDataset,
        function: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[np.ndarray, float]:
        loader = self._loader(
            dataset,
            shuffle=False,
            batch_size=int(self.config["evaluation"]["batch_size"]),
        )
        self._sync(self.device)
        start = time.perf_counter()
        prediction = Evaluator.predict_aggregate_loader(
            loader, function, device=self.device
        )
        self._sync(self.device)
        elapsed = time.perf_counter() - start
        return prediction, elapsed / max(1, len(loader))

    @staticmethod
    def _parameter_counts(model: nn.Module) -> tuple[int, int]:
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        return total, trainable

    @staticmethod
    def _entropy(route: torch.Tensor) -> float:
        values = route.detach().cpu().reshape(-1).double()
        return float(-(values * torch.log(values.clamp_min(1e-12))).sum().item())

    @staticmethod
    def _distance(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
        left = left.detach().cpu().reshape(-1).float()
        right = right.detach().cpu().reshape(-1).float()
        cosine = nn.functional.cosine_similarity(
            left.unsqueeze(0), right.unsqueeze(0), dim=1
        )
        return {
            "l2": float(torch.linalg.vector_norm(left - right).item()),
            "mean_absolute": float(torch.mean(torch.abs(left - right)).item()),
            "cosine_distance": float((1.0 - cosine).item()),
        }

    def _adapter_diagnostics(
        self,
        model: ContextRoutedAdapterBankSeq2Point,
        query_dataset: QueryWindowDataset,
        generated: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict:
        query = torch.stack([query_dataset[index] for index in range(256)]).to(self.device)
        with torch.no_grad():
            hidden = model.baseline.encode(query)
            outputs = [adapter(hidden) for adapter in model.adapters]
        cosine = nn.functional.cosine_similarity(
            outputs[0].reshape(1, -1), outputs[1].reshape(1, -1), dim=1
        )
        adapters = []
        for index, (adapter, output) in enumerate(zip(model.adapters, outputs), start=1):
            adapters.append(
                {
                    "adapter": index,
                    "down_weight_norm": float(adapter.down.weight.norm().item()),
                    "up_weight_norm": float(adapter.up.weight.norm().item()),
                    "adapter_output_norm": float(output.norm().item()),
                }
            )
        residuals = {}
        with torch.no_grad():
            for house, (_, route) in generated.items():
                adapted, _ = model.adapt_hidden(hidden, route)
                residual_norm = torch.linalg.vector_norm(adapted - hidden)
                hidden_norm = torch.linalg.vector_norm(hidden)
                residuals[house] = {
                    "adapter_residual_norm": float(residual_norm.item()),
                    "relative_residual_norm": float(
                        (residual_norm / hidden_norm.clamp_min(1e-12)).item()
                    ),
                }
        return {
            "adapters": adapters,
            "adapter_output_cosine_similarity": float(cosine.item()),
            "residuals_by_context_house": residuals,
        }

    def _metric_row(
        self,
        method: str,
        query_house: str,
        context_house: str,
        prediction: np.ndarray,
        truth: np.ndarray,
        aggregate: np.ndarray,
        status: np.ndarray,
        timestamps: pd.Series,
        *,
        total_parameters: int,
        trainable_parameters: int,
        context_time: float,
        inference_time: float,
        peak_memory: int,
    ) -> dict:
        metrics = Evaluator.score_aligned_predictions(
            prediction,
            truth,
            aggregate,
            status,
            timestamps,
            appliance_name="washing_machine",
        )
        metrics.pop("gamma_norm", None)
        metrics.pop("beta_norm", None)
        return {
            "method": method,
            "query_house": query_house,
            "context_house": context_house,
            **metrics,
            "trainable_parameter_count": trainable_parameters,
            "total_parameter_count": total_parameters,
            "context_generation_time_s": context_time,
            "average_query_inference_time_s": inference_time,
            "peak_gpu_memory_bytes": peak_memory,
        }

    def evaluate_all(self) -> None:
        manifest = self._prepare_artifacts()
        _reject_existing(self.run_dir / "metrics.csv")
        for path in (self.global_path, self.context_path):
            if not path.exists():
                raise FileNotFoundError(path)
        if len({manifest["checkpoints"].get(key) for key in ("A2", "A3", "A4")}) != 1:
            raise RuntimeError("A2, A3, and A4 must share exactly one checkpoint.")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        query_frame = self._read("H6", "B", ["aggregate"], "final_query_prediction")
        query_dataset = self._query_dataset(
            query_frame, "H6", require_target=False
        )
        query_identity = id(query_dataset)
        baseline, _ = self._load_baseline()
        baseline.eval()
        global_model = self._load_global(deepcopy(baseline))
        context_model = self._load_context(deepcopy(baseline))

        predictions: dict[str, np.ndarray] = {}
        inference_times: dict[str, float] = {}
        predictions["A0"], inference_times["A0"] = self._predict(
            query_dataset, baseline
        )
        predictions["A1"], inference_times["A1"] = self._predict(
            query_dataset, global_model
        )

        context_houses = ["H1", "H2", "H3", "H7", "H5", "H6"]
        context_frames = {
            house: self._read(house, "A", ["aggregate"], "context")
            for house in context_houses
        }
        contexts, selections = self._context_tensors(context_frames)
        generated: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        context_times = {}
        with torch.no_grad():
            for house in context_houses:
                windows, mask = contexts[house]
                self._sync(self.device)
                start = time.perf_counter()
                generated[house] = context_model.generate_route(
                    windows.unsqueeze(0).to(self.device),
                    mask.unsqueeze(0).to(self.device),
                )
                self._sync(self.device)
                context_times[house] = time.perf_counter() - start

        _, correct_route = generated["H6"]
        predictions["A2"], inference_times["A2"] = self._predict(
            query_dataset,
            lambda query: context_model.forward_with_route(query, correct_route),
        )
        for house in ["H1", "H2", "H3", "H7"]:
            _, wrong_route = generated[house]
            key = f"A3_wrong_{house}"
            predictions[key], inference_times[key] = self._predict(
                query_dataset,
                lambda query, route=wrong_route: context_model.forward_with_route(
                    query, route
                ),
            )
            if id(query_dataset) != query_identity:
                raise RuntimeError("Wrong-context evaluation replaced the H6-B query.")
        mean_route = torch.stack(
            [generated[house][1] for house in ["H1", "H2", "H3", "H7"]]
        ).mean(dim=0)
        predictions["A4"], inference_times["A4"] = self._predict(
            query_dataset,
            lambda query: context_model.forward_with_route(query, mean_route),
        )

        expected_prediction_count = 2 + 1 + 4 + 1
        if len(predictions) != expected_prediction_count:
            raise RuntimeError("Not all predictions exist; H6 labels remain sealed.")
        self.guard.mark_predictions_complete()
        label_frame = self._read(
            "H6",
            "B",
            ["aggregate", "washing_machine", "status"],
            "final_metrics_labels",
        )
        if not np.array_equal(
            query_frame["time"].to_numpy(dtype="datetime64[ns]"),
            label_frame["time"].to_numpy(dtype="datetime64[ns]"),
        ):
            raise RuntimeError("Aggregate-only and delayed-label H6-B frames differ.")
        centers = query_dataset.center_indices
        timestamps = label_frame["time"].iloc[centers].reset_index(drop=True)
        expected_count = int(self.config["evaluation"]["expected_timestamp_count"])
        if len(timestamps) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} H6-B timestamps, found {len(timestamps)}."
            )
        fingerprint = assert_common_evaluation_timestamps(
            {method: timestamps for method in predictions}
        )
        if fingerprint != self.reference_manifest["evaluation_timestamp_sha256"]:
            raise RuntimeError("Adapter timestamps differ from the FiLM reference.")

        film_frame = pd.read_csv(
            self.reference_predictions_path,
            usecols=["time", self.config["film_reference"]["method_column"]],
        )
        film_timestamps = pd.to_datetime(film_frame["time"])
        if timestamp_fingerprint(film_timestamps) != fingerprint:
            raise RuntimeError("Existing Context-FiLM prediction timestamps differ.")

        scale = self._stats().appliance_std
        offset = self._stats().appliance_mean
        predictions_watts = {
            key: values * scale + offset for key, values in predictions.items()
        }
        predictions_watts["ContextFiLM"] = film_frame[
            self.config["film_reference"]["method_column"]
        ].to_numpy(dtype=np.float32)
        truth = label_frame["washing_machine"].to_numpy(dtype=np.float32)[centers]
        aggregate = label_frame["aggregate"].to_numpy(dtype=np.float32)[centers]
        status = label_frame["status"].to_numpy(dtype=np.int8)[centers]
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        baseline_counts = self._parameter_counts(baseline)
        baseline_counts = (baseline_counts[0], 0)
        global_counts = self._parameter_counts(global_model)
        context_counts = self._parameter_counts(context_model)
        train_context_mean = float(
            np.mean([context_times[house] for house in ["H1", "H2", "H3", "H7"]])
        )
        specifications = [
            ("A0_seq2point", "A0", "none", baseline_counts, 0.0),
            ("A1_global_shared_adapter", "A1", "global", global_counts, 0.0),
            (
                "Existing_Context_FiLM",
                "ContextFiLM",
                "H6",
                (0, 0),
                context_times["H6"],
            ),
            (
                "A2_correct_context",
                "A2",
                "H6",
                context_counts,
                context_times["H6"],
            ),
            (
                "A3_wrong_H1",
                "A3_wrong_H1",
                "H1",
                context_counts,
                context_times["H1"],
            ),
            (
                "A3_wrong_H2",
                "A3_wrong_H2",
                "H2",
                context_counts,
                context_times["H2"],
            ),
            (
                "A3_wrong_H3",
                "A3_wrong_H3",
                "H3",
                context_counts,
                context_times["H3"],
            ),
            (
                "A3_wrong_H7",
                "A3_wrong_H7",
                "H7",
                context_counts,
                context_times["H7"],
            ),
            (
                "A4_mean_route",
                "A4",
                "mean_train_route",
                context_counts,
                train_context_mean,
            ),
        ]
        rows = []
        for method, key, context_house, counts, context_time in specifications:
            rows.append(
                self._metric_row(
                    method,
                    "H6",
                    context_house,
                    predictions_watts[key],
                    truth,
                    aggregate,
                    status,
                    timestamps,
                    total_parameters=counts[0],
                    trainable_parameters=counts[1],
                    context_time=context_time,
                    inference_time=inference_times.get(key, math.nan),
                    peak_memory=peak_memory,
                )
            )
        metrics = pd.DataFrame(rows)
        metrics.to_csv(self.run_dir / "metrics.csv", index=False)

        wrong = metrics[metrics["method"].str.startswith("A3_wrong_")]
        numeric_columns = wrong.select_dtypes(include=[np.number]).columns
        wrong_summary = pd.DataFrame(
            [
                {"statistic": "mean", **wrong[numeric_columns].mean().to_dict()},
                {"statistic": "std", **wrong[numeric_columns].std(ddof=0).to_dict()},
            ]
        )
        wrong_summary.to_csv(self.run_dir / "wrong_context_summary.csv", index=False)

        routing_rows = []
        for house, (code, route) in generated.items():
            values = route.detach().cpu().reshape(-1)
            routing_rows.append(
                {
                    "context_house": house,
                    "alpha_1": float(values[0].item()),
                    "alpha_2": float(values[1].item()),
                    "routing_entropy": self._entropy(route),
                    "max_route_weight": float(values.max().item()),
                    "context_code_norm": float(code.norm().item()),
                    "context_generation_time_s": context_times[house],
                }
            )
        mean_values = mean_route.detach().cpu().reshape(-1)
        routing_rows.append(
            {
                "context_house": "mean_train_route",
                "alpha_1": float(mean_values[0].item()),
                "alpha_2": float(mean_values[1].item()),
                "routing_entropy": self._entropy(mean_route),
                "max_route_weight": float(mean_values.max().item()),
                "context_code_norm": math.nan,
                "context_generation_time_s": train_context_mean,
            }
        )
        routing = pd.DataFrame(routing_rows)
        if not np.allclose(
            routing[["alpha_1", "alpha_2"]].sum(axis=1), 1.0, atol=1e-7
        ):
            raise RuntimeError("At least one route does not sum to one.")
        routing.to_csv(self.run_dir / "routing_weights.csv", index=False)

        distance_rows = []
        for left_house in context_houses:
            for right_house in context_houses:
                code_distance = self._distance(
                    generated[left_house][0], generated[right_house][0]
                )
                route_distance = self._distance(
                    generated[left_house][1], generated[right_house][1]
                )
                distance_rows.append(
                    {
                        "left_house": left_house,
                        "right_house": right_house,
                        **{
                            f"context_code_{key}": value
                            for key, value in code_distance.items()
                        },
                        **{
                            f"route_{key}": value
                            for key, value in route_distance.items()
                        },
                    }
                )
        pd.DataFrame(distance_rows).to_csv(
            self.run_dir / "context_distance_matrix.csv", index=False
        )
        diagnostics = self._adapter_diagnostics(
            context_model, query_dataset, generated
        )
        _json_dump(self.run_dir / "adapter_diagnostics.json", diagnostics)

        file_names = {
            "A0": "A0_seq2point.csv",
            "A1": "A1_global_shared_adapter.csv",
            "A2": "A2_correct_context.csv",
            "A3_wrong_H1": "A3_wrong_H1.csv",
            "A3_wrong_H2": "A3_wrong_H2.csv",
            "A3_wrong_H3": "A3_wrong_H3.csv",
            "A3_wrong_H7": "A3_wrong_H7.csv",
            "A4": "A4_mean_route.csv",
        }
        clipped_aggregate = np.clip(aggregate, 0.0, None)
        for key, file_name in file_names.items():
            prediction = np.minimum(
                np.clip(predictions_watts[key], 0.0, None), clipped_aggregate
            )
            pd.DataFrame(
                {
                    "time": timestamps,
                    "aggregate": aggregate,
                    "ground_truth": truth,
                    "status": status,
                    "prediction": prediction,
                }
            ).to_csv(self.prediction_dir / file_name, index=False)

        leakage = {
            "passed": True,
            "context_input_columns": ["time", "aggregate", "segment_id"],
            "context_forbidden_columns_absent": True,
            "test_labels_read_after_all_predictions": self.guard.predictions_complete,
            "H6_in_optimizer_batches": False,
            "H6_in_normalisation": False,
            "H6_in_checkpoint_selection": False,
            "H6_in_threshold_search": False,
            "threshold_source": self.config["evaluation"]["status_rule_source"],
            "A2_A3_query_object_reused": id(query_dataset) == query_identity,
            "A2_A3_A4_shared_checkpoint": manifest["checkpoints"]["A2"],
            "evaluation_timestamp_count": len(timestamps),
            "evaluation_timestamp_sha256": fingerprint,
            "film_timestamp_sha256": timestamp_fingerprint(film_timestamps),
            "context_window_timestamps": selections,
            "backbone_hash_checks": manifest["backbone_hash_checks"],
            "zero_init_checks": manifest["zero_init_checks"],
            "access_audit": self.guard.audit_dicts(),
        }
        _json_dump(self.run_dir / "leakage_audit.json", leakage)

        method_lookup = metrics.set_index("method")
        wrong_mean = wrong_summary.set_index("statistic").loc["mean"]
        deltas = {
            "delta_mae_vs_baseline": float(
                method_lookup.loc["A2_correct_context", "MAE"]
                - method_lookup.loc["A0_seq2point", "MAE"]
            ),
            "delta_mae_vs_global_adapter": float(
                method_lookup.loc["A2_correct_context", "MAE"]
                - method_lookup.loc["A1_global_shared_adapter", "MAE"]
            ),
            "delta_mae_vs_wrong_mean": float(
                method_lookup.loc["A2_correct_context", "MAE"] - wrong_mean["MAE"]
            ),
            "delta_mae_vs_mean_route": float(
                method_lookup.loc["A2_correct_context", "MAE"]
                - method_lookup.loc["A4_mean_route", "MAE"]
            ),
            "delta_mae_vs_film": float(
                method_lookup.loc["A2_correct_context", "MAE"]
                - method_lookup.loc["Existing_Context_FiLM", "MAE"]
            ),
            "delta_f1_vs_baseline": float(
                method_lookup.loc["A2_correct_context", "F1"]
                - method_lookup.loc["A0_seq2point", "F1"]
            ),
            "delta_f1_vs_wrong_mean": float(
                method_lookup.loc["A2_correct_context", "F1"] - wrong_mean["F1"]
            ),
            "delta_f1_vs_mean_route": float(
                method_lookup.loc["A2_correct_context", "F1"]
                - method_lookup.loc["A4_mean_route", "F1"]
            ),
        }
        _json_dump(self.run_dir / "key_deltas.json", deltas)

        manifest["evaluation_timestamp_count"] = len(timestamps)
        manifest["evaluation_timestamp_sha256"] = fingerprint
        manifest["context_window_timestamps"].update(selections)
        manifest["evaluation_outputs"] = {
            "metrics": "metrics.csv",
            "wrong_context_summary": "wrong_context_summary.csv",
            "routing_weights": "routing_weights.csv",
            "context_distance_matrix": "context_distance_matrix.csv",
            "adapter_diagnostics": "adapter_diagnostics.json",
            "leakage_audit": "leakage_audit.json",
            "predictions": {
                key: f"predictions/{value}" for key, value in file_names.items()
            },
        }
        self._update_manifest(manifest)
        run_manifest = {
            "dataset_path": str(
                (REPO_ROOT / self.config["data"]["csv_dir"]).resolve()
            ),
            "git_commit": _git_state()["commit"],
            "git_dirty": _git_state()["dirty"],
            "seed": self.seed,
            "house_split": self.config["house_split"],
            "fixed_14_day_ranges": self._fixed_blocks(),
            "normalization": self._stats().to_dict(),
            "normalization_source": str(
                self.reference_manifest_path.relative_to(REPO_ROOT)
            ),
            "checkpoint_paths": manifest["checkpoints"],
            "baseline_checkpoint": str(self.baseline_path.relative_to(REPO_ROOT)),
            "evaluation_timestamp_count": len(timestamps),
            "evaluation_timestamp_sha256": fingerprint,
            "config_sha256": manifest["config_sha256"],
            "device": str(self.device),
        }
        _json_dump(self.run_manifest_path, run_manifest)
        print(metrics.to_string(index=False))
        print(json.dumps(deltas, indent=2))

    def run_all(self) -> None:
        self.train_global_adapter()
        self.train_context_adapter()
        self.evaluate_all()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "train_global_adapter",
            "train_context_adapter",
            "evaluate_all",
            "run_all",
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments/configs/refit_washing_machine_seq2point_adapter_bank_dev.yaml"
        ),
    )
    parser.add_argument("--device", help="For example cpu, cuda, or cuda:0.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = Experiment(args.config, args.device)
    getattr(experiment, args.command)()


if __name__ == "__main__":
    main()
