#!/usr/bin/env python3
"""Run the leakage-controlled REFIT washing-machine Context-FiLM experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
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

from model_pipeline.context_data_feeder import (  # noqa: E402
    HouseholdBalancedBatchSampler,
    LeakageGuard,
    MultiHouseQueryDataset,
    NormalisationStats,
    QueryWindowDataset,
    assert_common_evaluation_timestamps,
    compute_normalisation_stats,
    read_refit_block,
    resolve_block_specs,
    select_uniform_context_windows,
)
from model_pipeline.models.context_adapter.seq2point_film import (  # noqa: E402
    ContextFiLMSeq2Point,
    GlobalFiLMSeq2Point,
    module_sha256,
)
from model_pipeline.models.seq2point.seq2point import Seq2Point  # noqa: E402
from model_pipeline.test_model import Evaluator  # noqa: E402
from model_pipeline.train_model import Trainer  # noqa: E402


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        yaml = None
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) if yaml is not None else json.load(handle)
    required = {
        "appliance",
        "seed",
        "window_size",
        "context_k",
        "context_selector",
        "house_split",
        "fixed_14_day_starts",
    }
    missing = required.difference(config)
    if missing:
        raise KeyError(f"Config is missing: {sorted(missing)}")
    if config["context_selector"] != "uniform":
        raise ValueError("This controlled experiment requires context_selector=uniform.")
    return config


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
    temporary.replace(path)


def _torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _save_best(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _reject_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint {path}. Use a new --run-id."
        )


class HouseholdContextTrainerModel(ContextFiLMSeq2Point):
    """Trainer hook that pairs balanced household queries with cached A contexts."""

    def configure_training_contexts(
        self,
        contexts: dict[str, tuple[torch.Tensor, torch.Tensor]],
        train_houses: list[str],
        validation_house: str,
        device: torch.device,
    ) -> None:
        self._training_contexts = [
            tuple(value.to(device) for value in contexts[house]) for house in train_houses
        ]
        self._validation_context = tuple(
            value.to(device) for value in contexts[validation_house]
        )

    def prepare_batch(self, batch, *, device):
        if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
            raise ValueError("Context-FiLM batches must contain query, target, and optional house id.")
        query = batch[0].to(device)
        target = self.prepare_targets(batch[1].to(device))
        house_indices = None if len(batch) == 2 else batch[2].to(device)
        return {"query": query, "house_indices": house_indices}, target

    def compute_loss(self, inputs, targets, *, criterion):
        query = inputs["query"]
        house_indices = inputs["house_indices"]
        if house_indices is None:
            windows, mask = self._validation_context
            _, gamma, beta = self.generate_film(
                windows.unsqueeze(0), mask.unsqueeze(0)
            )
            prediction = self.forward_with_film(query, gamma, beta).reshape(-1)
            return criterion(prediction, targets)

        loss_sum = torch.zeros((), device=query.device)
        for house_index, (windows, mask) in enumerate(self._training_contexts):
            selected = house_indices == house_index
            if not bool(selected.any()):
                continue
            _, gamma, beta = self.generate_film(
                windows.unsqueeze(0), mask.unsqueeze(0)
            )
            prediction = self.forward_with_film(query[selected], gamma, beta).reshape(-1)
            loss_sum = loss_sum + nn.functional.mse_loss(
                prediction, targets[selected], reduction="sum"
            )
        return loss_sum / len(targets)


class Experiment:
    def __init__(self, config_path: Path, run_id: str | None, device_name: str | None):
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        self.seed = int(self.config["seed"])
        set_seed(self.seed)
        self.device = torch.device(
            device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.run_id = run_id or self.config["default_run_id"]
        self.run_dir = REPO_ROOT / self.config["artifacts_root"] / self.run_id
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.manifest_path = self.run_dir / "manifest.json"
        self.specs = resolve_block_specs(self.config, REPO_ROOT)
        self.guard = LeakageGuard(test_house=self.config["house_split"]["test"][0])
        self.chunksize = int(self.config["data"].get("chunksize", 250_000))
        minimum_epochs = int(self.config["training"]["minimum_epochs"])
        for key in ("baseline_epochs", "global_film_epochs", "context_film_epochs"):
            if int(self.config["training"][key]) < minimum_epochs:
                raise ValueError(f"{key} must be >= minimum_epochs ({minimum_epochs}).")

    @property
    def baseline_path(self) -> Path:
        return self.checkpoint_dir / "m0_seq2point_best.pt"

    @property
    def global_path(self) -> Path:
        return self.checkpoint_dir / "m1_global_film_best.pt"

    @property
    def context_path(self) -> Path:
        return self.checkpoint_dir / "m2_context_film_best.pt"

    def _read(self, house: str, block: str, columns: list[str], purpose: str) -> pd.DataFrame:
        return read_refit_block(
            self.specs[house][block],
            columns=columns,
            purpose=purpose,
            guard=self.guard,
            chunksize=self.chunksize,
        )

    def _base_manifest(self, stats: NormalisationStats) -> dict:
        config_bytes = self.config_path.read_bytes()
        try:
            recorded_config_path = str(self.config_path.relative_to(REPO_ROOT))
        except ValueError:
            recorded_config_path = str(self.config_path)
        return {
            "schema_version": 1,
            "experiment": self.config["experiment_name"],
            "run_id": self.run_id,
            "config_path": recorded_config_path,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "config": deepcopy(self.config),
            "house_split": deepcopy(self.config["house_split"]),
            "fixed_blocks": {
                house: {
                    block: {
                        "start": spec.start,
                        "end": spec.end,
                        "csv_path": str(Path(spec.csv_path).relative_to(REPO_ROOT)),
                    }
                    for block, spec in blocks.items()
                }
                for house, blocks in self.specs.items()
            },
            "block_definition": {
                "interval": "half-open fixed 14 calendar days",
                "A": "days [0, 7), aggregate-only context",
                "B": "days [7, 14), aggregate query and controlled label access",
                "window_policy": "599-point windows remain within one segment_id",
            },
            "normalisation_stats": stats.to_dict(),
            "normalisation_provenance": {
                "aggregate": {"houses": self.config["house_split"]["train"], "blocks": ["A", "B"]},
                "appliance": {"houses": self.config["house_split"]["train"], "blocks": ["B"]},
            },
            "seed": self.seed,
            "git": _git_state(),
            "engines": {
                "training": "model_pipeline.train_model.Trainer",
                "evaluation": "model_pipeline.test_model.Evaluator",
                "context_batch_hook": (
                    "experiments.run_seq2point_context_film."
                    "HouseholdContextTrainerModel"
                ),
            },
            "access_audit": [],
            "checkpoints": {},
            "best_epochs": {},
        }

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {self.manifest_path}; run train_baseline first with the same --run-id."
            )
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest["config_sha256"] != hashlib.sha256(self.config_path.read_bytes()).hexdigest():
            raise RuntimeError("Config changed after the run manifest was created; use a new --run-id.")
        return manifest

    def _update_manifest(self, manifest: dict) -> None:
        manifest["access_audit"].extend(self.guard.audit_dicts())
        self.guard.records.clear()
        _json_dump(self.manifest_path, manifest)

    def _stats(self, manifest: dict) -> NormalisationStats:
        return NormalisationStats(**manifest["normalisation_stats"])

    def _query_dataset(
        self,
        frame: pd.DataFrame,
        house: str,
        stats: NormalisationStats,
        require_target: bool = True,
    ) -> QueryWindowDataset:
        return QueryWindowDataset(
            frame,
            house=house,
            stats=stats,
            window_size=int(self.config["window_size"]),
            appliance=self.config["appliance"],
            require_target=require_target,
        )

    def _new_baseline(self) -> Seq2Point:
        return Seq2Point(
            window_size=int(self.config["window_size"]),
            hidden_dim=int(self.config["hidden_dim"]),
        )

    def _load_baseline(self) -> tuple[Seq2Point, dict]:
        checkpoint = _torch_load(self.baseline_path, self.device)
        baseline = Seq2Point(**checkpoint["init_kwargs"])
        baseline.load_state_dict(checkpoint["model_state"])
        baseline.to(self.device)
        if module_sha256(baseline) != checkpoint["baseline_sha256"]:
            raise RuntimeError("M0 baseline checkpoint hash mismatch.")
        return baseline, checkpoint

    def _loader(self, dataset, *, shuffle: bool, batch_size: int | None = None) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=batch_size or int(self.config["training"]["batch_size"]),
            shuffle=shuffle,
            generator=generator,
            num_workers=int(self.config["training"].get("num_workers", 0)),
        )

    def train_baseline(self) -> None:
        _reject_existing(self.baseline_path)
        if self.manifest_path.exists():
            raise FileExistsError(f"Run manifest already exists: {self.manifest_path}; use a new --run-id.")
        train_houses = self.config["house_split"]["train"]
        aggregate_blocks: list[pd.DataFrame] = []
        train_b: dict[str, pd.DataFrame] = {}
        for house in train_houses:
            aggregate_blocks.append(
                self._read(house, "A", ["aggregate"], "normalization_aggregate")
            )
            block_b = self._read(
                house, "B", ["aggregate", self.config["appliance"]], "baseline_training_query"
            )
            train_b[house] = block_b
            aggregate_blocks.append(block_b[["time", "aggregate", "segment_id"]])
        stats = compute_normalisation_stats(
            aggregate_blocks, list(train_b.values()), appliance=self.config["appliance"]
        )
        val_house = self.config["house_split"]["validation"][0]
        val_b = self._read(
            val_house,
            "B",
            ["aggregate", self.config["appliance"]],
            "checkpoint_selection",
        )
        train_dataset = ConcatDataset(
            [self._query_dataset(train_b[house], house, stats) for house in train_houses]
        )
        val_dataset = self._query_dataset(val_b, val_house, stats)
        train_loader = self._loader(train_dataset, shuffle=True)
        val_loader = self._loader(val_dataset, shuffle=False)

        model = self._new_baseline().to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(self.config["training"]["learning_rate"])
        )
        def save_baseline(trainer):
            checkpoint = {
                "mode": "M0",
                "model_key": "seq2point",
                "init_kwargs": trainer.model.get_init_kwargs(),
                "normalisation_stats": stats.to_dict(),
                "checkpoint_selection_start_epoch": 1,
                "epoch": trainer.current_epoch,
                "validation_mse": trainer.current_selection_loss,
                "model_state": trainer.model.state_dict(),
                "seed": self.seed,
            }
            checkpoint["baseline_sha256"] = module_sha256(trainer.model)
            _save_best(self.baseline_path, checkpoint)
            return self.baseline_path

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            appliance=self.config["appliance"],
            dataset=self.config["dataset"],
            model_save_dir=str(self.checkpoint_dir),
            result_dir=str(self.run_dir),
            seed=self.seed,
            device=str(self.device),
            normalisation_stats=stats.to_dict(),
            patience=int(self.config["training"]["patience"]),
            min_delta=float(self.config["training"]["min_delta"]),
            minimum_epochs=int(self.config["training"]["minimum_epochs"]),
            select_epoch_zero=False,
            checkpoint_callback=save_baseline,
        )
        trainer.trainModel(num_epochs=int(self.config["training"]["baseline_epochs"]))
        manifest = self._base_manifest(stats)
        manifest["checkpoints"]["M0"] = str(self.baseline_path.relative_to(REPO_ROOT))
        manifest["best_epochs"]["M0"] = trainer.best_epoch
        manifest["training_history"] = {"M0": trainer.epoch_history}
        manifest["test_evaluation_timestamp_policy"] = (
            f"{self.config['house_split']['test'][0]}-B valid Seq2Point center timestamps"
        )
        self._update_manifest(manifest)
        print(f"M0 best epoch: {trainer.best_epoch}; checkpoint: {self.baseline_path}")

    def _adapter_frames(self, include_context: bool):
        manifest = self._load_manifest()
        stats = self._stats(manifest)
        train_houses = self.config["house_split"]["train"]
        train_b = {
            house: self._read(
                house, "B", ["aggregate", self.config["appliance"]], "adapter_training_query"
            )
            for house in train_houses
        }
        val_house = self.config["house_split"]["validation"][0]
        val_b = self._read(
            val_house,
            "B",
            ["aggregate", self.config["appliance"]],
            "checkpoint_selection",
        )
        context = None
        if include_context:
            context = {
                house: self._read(house, "A", ["aggregate"], "context")
                for house in [*train_houses, val_house]
            }
        return manifest, stats, train_houses, train_b, val_house, val_b, context

    def _identity_check(self, baseline: Seq2Point, adapter: nn.Module, query: torch.Tensor, context=None) -> float:
        baseline.eval()
        adapter.eval()
        with torch.no_grad():
            reference = baseline(query.to(self.device))
            candidate = adapter(query.to(self.device)) if context is None else adapter(query.to(self.device), *context)
        maximum = float(torch.max(torch.abs(reference - candidate)).item())
        if maximum > 1e-6:
            raise RuntimeError(f"Epoch-0 FiLM identity check failed: max_abs_error={maximum:.9g}")
        return maximum

    def train_global_film(self) -> None:
        _reject_existing(self.global_path)
        manifest, stats, houses, train_b, val_house, val_b, _ = self._adapter_frames(False)
        baseline, baseline_checkpoint = self._load_baseline()
        adapter = GlobalFiLMSeq2Point(baseline).to(self.device)
        adapter.assert_baseline_frozen()
        baseline_hash = module_sha256(adapter.baseline)
        train_dataset = ConcatDataset(
            [self._query_dataset(train_b[house], house, stats) for house in houses]
        )
        val_dataset = self._query_dataset(val_b, val_house, stats)
        train_loader = self._loader(train_dataset, shuffle=True)
        val_loader = self._loader(val_dataset, shuffle=False)
        sample_query = val_dataset[0][0].unsqueeze(0)
        identity_error = self._identity_check(adapter.baseline, adapter, sample_query)
        optimizer = torch.optim.Adam(
            [parameter for parameter in adapter.parameters() if parameter.requires_grad],
            lr=float(self.config["training"]["adapter_learning_rate"]),
        )
        def save_global(trainer):
            checkpoint = self._global_checkpoint(
                trainer.model,
                baseline_checkpoint,
                trainer.current_epoch,
                trainer.current_selection_loss,
                identity_error,
            )
            _save_best(self.global_path, checkpoint)
            return self.global_path

        trainer = Trainer(
            model=adapter,
            train_loader=train_loader,
            validation_loader=val_loader,
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            appliance=self.config["appliance"],
            dataset=self.config["dataset"],
            model_save_dir=str(self.checkpoint_dir),
            result_dir=str(self.run_dir),
            seed=self.seed,
            device=str(self.device),
            normalisation_stats=stats.to_dict(),
            patience=int(self.config["training"]["patience"]),
            min_delta=float(self.config["training"]["min_delta"]),
            minimum_epochs=int(self.config["training"]["minimum_epochs"]),
            select_epoch_zero=True,
            checkpoint_callback=save_global,
        )
        trainer.trainModel(num_epochs=int(self.config["training"]["global_film_epochs"]))
        if module_sha256(adapter.baseline) != baseline_hash:
            raise RuntimeError("Frozen M0 weights changed during Global-FiLM training.")
        manifest["checkpoints"]["M1"] = str(self.global_path.relative_to(REPO_ROOT))
        manifest["best_epochs"]["M1"] = trainer.best_epoch
        manifest.setdefault("training_history", {})["M1"] = trainer.epoch_history
        self._update_manifest(manifest)
        print(f"M1 best epoch: {trainer.best_epoch}; checkpoint: {self.global_path}")

    def _global_checkpoint(self, adapter, baseline_checkpoint, epoch, val_loss, identity_error):
        return {
            "mode": "M1",
            "epoch": epoch,
            "validation_mse": val_loss,
            "identity_max_abs_error": identity_error,
            "baseline_checkpoint": str(self.baseline_path.relative_to(REPO_ROOT)),
            "baseline_sha256": baseline_checkpoint["baseline_sha256"],
            "adapter_state": {
                "gamma_global_raw": adapter.gamma_global_raw.detach().cpu(),
                "beta_global_raw": adapter.beta_global_raw.detach().cpu(),
            },
            "seed": self.seed,
        }

    def _context_tensors(self, frames: dict[str, pd.DataFrame], stats: NormalisationStats):
        tensors = {}
        selections = {}
        for house, frame in frames.items():
            windows, mask, timestamps = select_uniform_context_windows(
                frame,
                stats=stats,
                window_size=int(self.config["window_size"]),
                context_k=int(self.config["context_k"]),
            )
            tensors[house] = (windows, mask)
            selections[house] = [str(value) for value in timestamps]
        return tensors, selections

    def _new_context_adapter(
        self, baseline: Seq2Point, *, for_training: bool = False
    ) -> ContextFiLMSeq2Point:
        adapter_class = (
            HouseholdContextTrainerModel if for_training else ContextFiLMSeq2Point
        )
        return adapter_class(
            baseline,
            window_size=int(self.config["window_size"]),
            code_dim=int(self.config["context_code_dim"]),
            generator_hidden_dim=int(self.config["generator_hidden_dim"]),
        )

    def train_context_film(self) -> None:
        _reject_existing(self.context_path)
        manifest, stats, houses, train_b, val_house, val_b, context_frames = self._adapter_frames(True)
        baseline, baseline_checkpoint = self._load_baseline()
        adapter = self._new_context_adapter(baseline, for_training=True).to(self.device)
        adapter.assert_baseline_frozen()
        baseline_hash = module_sha256(adapter.baseline)
        contexts, selections = self._context_tensors(context_frames, stats)
        adapter.configure_training_contexts(
            contexts, houses, val_house, self.device
        )
        train_dataset = MultiHouseQueryDataset(
            [self._query_dataset(train_b[house], house, stats) for house in houses]
        )
        sampler = HouseholdBalancedBatchSampler(
            train_dataset,
            batch_size=int(self.config["training"]["batch_size"]),
            seed=self.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=int(self.config["training"].get("num_workers", 0)),
        )
        val_dataset = self._query_dataset(val_b, val_house, stats)
        val_loader = self._loader(val_dataset, shuffle=False)
        sample_query = val_dataset[0][0].unsqueeze(0)
        val_windows, val_mask = contexts[val_house]
        identity_error = self._identity_check(
            adapter.baseline,
            adapter,
            sample_query,
            (val_windows.unsqueeze(0).to(self.device), val_mask.unsqueeze(0).to(self.device)),
        )
        optimizer = torch.optim.Adam(
            [parameter for parameter in adapter.parameters() if parameter.requires_grad],
            lr=float(self.config["training"]["adapter_learning_rate"]),
        )
        def save_context(trainer):
            checkpoint = self._context_checkpoint(
                trainer.model,
                baseline_checkpoint,
                trainer.current_epoch,
                trainer.current_selection_loss,
                identity_error,
            )
            _save_best(self.context_path, checkpoint)
            return self.context_path

        trainer = Trainer(
            model=adapter,
            train_loader=train_loader,
            validation_loader=val_loader,
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            appliance=self.config["appliance"],
            dataset=self.config["dataset"],
            model_save_dir=str(self.checkpoint_dir),
            result_dir=str(self.run_dir),
            seed=self.seed,
            device=str(self.device),
            normalisation_stats=stats.to_dict(),
            patience=int(self.config["training"]["patience"]),
            min_delta=float(self.config["training"]["min_delta"]),
            minimum_epochs=int(self.config["training"]["minimum_epochs"]),
            select_epoch_zero=True,
            checkpoint_callback=save_context,
        )
        trainer.trainModel(num_epochs=int(self.config["training"]["context_film_epochs"]))
        if module_sha256(adapter.baseline) != baseline_hash:
            raise RuntimeError("Frozen M0 weights changed during Context-FiLM training.")
        manifest["checkpoints"]["M2"] = str(self.context_path.relative_to(REPO_ROOT))
        manifest["checkpoints"]["M3"] = str(self.context_path.relative_to(REPO_ROOT))
        manifest["best_epochs"]["M2"] = trainer.best_epoch
        manifest["best_epochs"]["M3"] = trainer.best_epoch
        manifest["context_window_timestamps"] = selections
        manifest.setdefault("training_history", {})["M2"] = trainer.epoch_history
        self._update_manifest(manifest)
        print(
            f"M2 best epoch: {trainer.best_epoch}; "
            f"shared M2/M3 checkpoint: {self.context_path}"
        )

    def _context_checkpoint(self, adapter, baseline_checkpoint, epoch, val_loss, identity_error):
        return {
            "mode": "M2_M3_shared",
            "epoch": epoch,
            "validation_mse": val_loss,
            "identity_max_abs_error": identity_error,
            "baseline_checkpoint": str(self.baseline_path.relative_to(REPO_ROOT)),
            "baseline_sha256": baseline_checkpoint["baseline_sha256"],
            "context_encoder_state": adapter.context_encoder.state_dict(),
            "film_generator_state": adapter.generator.state_dict(),
            "seed": self.seed,
        }

    def _load_global(self, baseline: Seq2Point) -> GlobalFiLMSeq2Point:
        checkpoint = _torch_load(self.global_path, self.device)
        if checkpoint["baseline_sha256"] != module_sha256(baseline):
            raise RuntimeError("M1 does not reference the loaded M0 baseline weights.")
        adapter = GlobalFiLMSeq2Point(baseline).to(self.device)
        adapter.gamma_global_raw.data.copy_(checkpoint["adapter_state"]["gamma_global_raw"].to(self.device))
        adapter.beta_global_raw.data.copy_(checkpoint["adapter_state"]["beta_global_raw"].to(self.device))
        adapter.eval()
        return adapter

    def _load_context(self, baseline: Seq2Point) -> ContextFiLMSeq2Point:
        checkpoint = _torch_load(self.context_path, self.device)
        if checkpoint["baseline_sha256"] != module_sha256(baseline):
            raise RuntimeError("M2/M3 do not reference the loaded M0 baseline weights.")
        adapter = self._new_context_adapter(baseline).to(self.device)
        adapter.context_encoder.load_state_dict(checkpoint["context_encoder_state"])
        adapter.generator.load_state_dict(checkpoint["film_generator_state"])
        adapter.eval()
        return adapter

    def _predict(self, dataset: QueryWindowDataset, function: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        loader = self._loader(
            dataset,
            shuffle=False,
            batch_size=int(self.config["evaluation"]["batch_size"]),
        )
        return Evaluator.predict_aggregate_loader(
            loader, function, device=self.device
        )

    @staticmethod
    def _distance(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
        left = left.detach().cpu().reshape(-1).float()
        right = right.detach().cpu().reshape(-1).float()
        cosine = nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=1)
        return {
            "l2": float(torch.linalg.vector_norm(left - right).item()),
            "mean_absolute": float(torch.mean(torch.abs(left - right)).item()),
            "cosine_distance": float((1.0 - cosine).item()),
        }

    def _metric_row(
        self,
        name: str,
        prediction: np.ndarray,
        truth: np.ndarray,
        aggregate: np.ndarray,
        status: np.ndarray,
        timestamps: pd.Series,
        gamma: torch.Tensor | None,
        beta: torch.Tensor | None,
    ) -> dict[str, object]:
        metrics = Evaluator.score_aligned_predictions(
            prediction,
            truth,
            aggregate,
            status,
            timestamps,
            appliance_name=self.config["appliance"],
            gamma=gamma,
            beta=beta,
        )
        return {"method": name, **metrics}

    def evaluate_all(self) -> None:
        _reject_existing(self.run_dir / "results.csv")
        manifest = self._load_manifest()
        for path in (self.baseline_path, self.global_path, self.context_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing required checkpoint: {path}")
        if manifest["checkpoints"].get("M2") != manifest["checkpoints"].get("M3"):
            raise RuntimeError("Correct and Wrong modes must point to the same Context-FiLM checkpoint.")
        stats = self._stats(manifest)
        test_house = self.config["house_split"]["test"][0]
        query_frame = self._read(test_house, "B", ["aggregate"], "final_query_prediction")
        query_dataset = self._query_dataset(query_frame, test_house, stats, require_target=False)
        shared_query_identity = id(query_dataset)

        baseline, _ = self._load_baseline()
        baseline.eval()
        global_adapter = self._load_global(deepcopy(baseline))
        context_adapter = self._load_context(deepcopy(baseline))
        normalized_predictions: dict[str, np.ndarray] = {
            "M0 Seq2Point": self._predict(query_dataset, baseline),
            "M1 Global FiLM": self._predict(query_dataset, global_adapter),
        }
        context_houses = [test_house, *self.config["house_split"]["train"]]
        context_frames = {
            house: self._read(house, "A", ["aggregate"], "context") for house in context_houses
        }
        contexts, selections = self._context_tensors(context_frames, stats)
        generated = {}
        with torch.no_grad():
            for house in context_houses:
                windows, mask = (value.to(self.device) for value in contexts[house])
                generated[house] = context_adapter.generate_film(
                    windows.unsqueeze(0), mask.unsqueeze(0)
                )
        correct_code, correct_gamma, correct_beta = generated[test_house]
        normalized_predictions["M2 Correct Context"] = self._predict(
            query_dataset,
            lambda query: context_adapter.forward_with_film(query, correct_gamma, correct_beta),
        )
        wrong_names = []
        for house in self.config["house_split"]["train"]:
            wrong_names.append(f"M3 Wrong {house}-A")
            _, gamma, beta = generated[house]
            normalized_predictions[wrong_names[-1]] = self._predict(
                query_dataset,
                lambda query, gamma=gamma, beta=beta: context_adapter.forward_with_film(query, gamma, beta),
            )
            if id(query_dataset) != shared_query_identity:
                raise RuntimeError("Wrong mode replaced the fixed H6-B query object.")

        # The only operation that unlocks H6 labels occurs after all seven predictions exist.
        expected_count = 3 + len(self.config["house_split"]["train"])
        if len(normalized_predictions) != expected_count:
            raise RuntimeError("Not all predictions completed; H6 labels remain sealed.")
        self.guard.mark_predictions_complete()
        label_frame = self._read(
            test_house,
            "B",
            ["aggregate", self.config["appliance"], "status"],
            "final_metrics_labels",
        )
        if not np.array_equal(
            query_frame["time"].to_numpy(dtype="datetime64[ns]"),
            label_frame["time"].to_numpy(dtype="datetime64[ns]"),
        ):
            raise RuntimeError("Aggregate-only prediction frame and delayed-label frame do not align.")
        centers = query_dataset.center_indices
        timestamps = label_frame["time"].iloc[centers].reset_index(drop=True)
        truth = label_frame[self.config["appliance"]].to_numpy(dtype=np.float32)[centers]
        status = label_frame["status"].to_numpy(dtype=np.int8)[centers]
        aggregate = label_frame["aggregate"].to_numpy(dtype=np.float32)[centers]
        fingerprint = assert_common_evaluation_timestamps(
            {name: timestamps for name in normalized_predictions}
        )

        scale = stats.appliance_std
        offset = stats.appliance_mean
        predictions = {
            name: values * scale + offset for name, values in normalized_predictions.items()
        }
        rows = []
        rows.append(self._metric_row("M0 Seq2Point", predictions["M0 Seq2Point"], truth, aggregate, status, timestamps, None, None))
        global_gamma, global_beta = global_adapter.film_parameters()
        rows.append(self._metric_row("M1 Global FiLM", predictions["M1 Global FiLM"], truth, aggregate, status, timestamps, global_gamma, global_beta))
        rows.append(self._metric_row("M2 Correct Context", predictions["M2 Correct Context"], truth, aggregate, status, timestamps, correct_gamma, correct_beta))
        for name, house in zip(wrong_names, self.config["house_split"]["train"]):
            _, gamma, beta = generated[house]
            rows.append(self._metric_row(name, predictions[name], truth, aggregate, status, timestamps, gamma, beta))
        result = pd.DataFrame(rows)
        numeric = [column for column in result.columns if column != "method"]
        wrong_values = result[result["method"].isin(wrong_names)][numeric]
        mean_row = {"method": "M3 Wrong mean", **wrong_values.mean().to_dict()}
        std_row = {"method": "M3 Wrong std", **wrong_values.std(ddof=0).to_dict()}
        result = pd.concat((result, pd.DataFrame([mean_row, std_row])), ignore_index=True)

        distances = {"correct_context_house": test_house, "wrong_contexts": {}}
        for house in self.config["house_split"]["train"]:
            code, gamma, beta = generated[house]
            distances["wrong_contexts"][house] = {
                "context_code": self._distance(correct_code, code),
                "gamma": self._distance(correct_gamma, gamma),
                "beta": self._distance(correct_beta, beta),
            }
        distance_summary = {}
        for key in ("context_code", "gamma", "beta"):
            for metric in ("l2", "mean_absolute", "cosine_distance"):
                values = [distances["wrong_contexts"][house][key][metric] for house in self.config["house_split"]["train"]]
                distance_summary[f"{key}_{metric}"] = {
                    "mean": float(np.mean(values)), "std": float(np.std(values))
                }
        distances["mean_std"] = distance_summary

        self.run_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.run_dir / "results.csv", index=False)
        _json_dump(self.run_dir / "results.json", result.to_dict(orient="records"))
        _json_dump(self.run_dir / "context_film_distances.json", distances)
        prediction_frame = pd.DataFrame(
            {"time": timestamps, "aggregate": aggregate, "ground_truth": truth, "status": status}
        )
        for name, values in predictions.items():
            prediction_frame[name] = np.minimum(np.clip(values, 0.0, None), np.clip(aggregate, 0.0, None))
        prediction_frame.to_csv(
            self.run_dir / f"{test_house.lower()}_b_predictions.csv", index=False
        )
        leakage_report = {
            "passed": True,
            "test_house": test_house,
            "test_labels_read_after_all_predictions": self.guard.predictions_complete,
            "test_not_in_normalisation": test_house not in manifest["normalisation_provenance"]["aggregate"]["houses"]
            and test_house not in manifest["normalisation_provenance"]["appliance"]["houses"],
            "test_not_in_checkpoint_selection": test_house not in self.config["house_split"]["validation"],
            "threshold_source": self.config["evaluation"]["status_rule_source"],
            "wrong_query_house_block": f"{test_house}-B",
            "wrong_query_object_reused": id(query_dataset) == shared_query_identity,
            "shared_m2_m3_checkpoint": manifest["checkpoints"]["M2"],
            "evaluation_timestamp_sha256": fingerprint,
            "evaluation_timestamp_count": len(timestamps),
            "all_methods_timestamp_sha256": {name: fingerprint for name in normalized_predictions},
            "context_window_timestamps": selections,
            "access_audit": self.guard.audit_dicts(),
        }
        _json_dump(self.run_dir / "leakage_report.json", leakage_report)
        summary = {
            "best_epochs": manifest["best_epochs"],
            "checkpoints": manifest["checkpoints"],
            "results": str((self.run_dir / "results.csv").relative_to(REPO_ROOT)),
            "leakage_report": str((self.run_dir / "leakage_report.json").relative_to(REPO_ROOT)),
            "distance_statistics": str((self.run_dir / "context_film_distances.json").relative_to(REPO_ROOT)),
        }
        _json_dump(self.run_dir / "summary.json", summary)
        manifest["evaluation_timestamp_sha256"] = fingerprint
        manifest["evaluation_timestamp_count"] = len(timestamps)
        manifest["evaluation_outputs"] = summary
        self._update_manifest(manifest)
        print(result.to_string(index=False))
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    default_config = REPO_ROOT / "experiments/configs/refit_washing_machine_seq2point_film_dev.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("train_baseline", "train_global_film", "train_context_film", "evaluate_all"),
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--run-id", help="New artifact subdirectory; existing checkpoints are never overwritten.")
    parser.add_argument("--device", help="For example cpu, cuda, or cuda:0 (default: auto).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = Experiment(args.config, args.run_id, args.device)
    getattr(experiment, args.command)()


if __name__ == "__main__":
    main()
