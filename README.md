# NILM Baseline Toolkit

A modular toolkit for **Non-Intrusive Load Monitoring (NILM)** benchmarking, providing a standardized pipeline from raw data processing to model training, evaluation, and fine-tuning.

This repository organizes NILM baselines under a shared training and evaluation
pipeline. Two primary reference repositories are used in a complementary way:

- `https://github.com/MingjunZhong/seq2point-nilm` for the Seq2Point paper and
  reference implementation lineage
- `https://github.com/nilmtk/nilmtk-contrib` as a related NILM baseline
  collection and comparison reference

The Seq2Point family in this repository is based on the Seq2Point approach introduced in:

> Mingjun Zhong, Nigel Goddard, Stephen Sutton, and Charles Gillan (2018). "Sequence-to-Point Learning with Neural Networks for Non-Intrusive Load Monitoring"

The original Seq2Point implementation: https://github.com/MingjunZhong/seq2point-nilm

Additional NILM baselines in this repository are implemented locally while
being compared against, or organized with reference to, the related baseline
collection in nilmtk-contrib. Provenance is documented in the relevant code
file and in the model provenance table below.

---

## What This Repository Implements

- **Data Processing Pipeline** — Raw NILM dataset separation, resampling, and CSV generation for 4 major datasets (UKDALE, REDD, REFIT, ECO)
- **Model Registry** — Decorator-based model registration with auto-discovery, aliases, and metadata
- **11 Baseline Models** — Spanning CNN, RNN, LSTM, GRU, DAE, HMM, and Sparse Coding architectures
- **Training Pipeline** — With early stopping, learning rate scheduling, checkpoint saving, and reproducibility
- **Evaluation Pipeline** — MAE/SAE metrics, full + zoomed prediction plots, result CSV export
- **Fine-tuning Pipeline** — Load pre-trained checkpoints, freeze layers, adapt to new data
- **Interactive CLI** — Menu-driven interface supporting both local and Google Colab environments

---

## Project Structure

```
nilm-baseline/
├── main.py                              # Main CLI entry point
├── data_wizard.py                        # Data processing CLI entry point
├── requirements.txt                      # Python dependencies
│
├── dataset_management/                   # Data processing module
│   ├── data_separation/
│   │   ├── data_separator.py             #   Raw data → per-appliance HDF5
│   │   ├── standard_h5_adapter.py        #   STANDARD_H5 adapter
│   │   ├── eco_data_ranges.json          #   ECO dataset date ranges
│   │   ├── ukdale_appliance_mappings.json
│   │   ├── redd_appliance_mappings.json
│   │   ├── refit_appliance_mappings.json
│   │   └── eco_appliance_mappings.json
│   ├── validation/
│   │   ├── file_validator.py             #   Single-file validation
│   │   ├── report_writer.py              #   Validation report export
│   │   └── rules.py                      #   Validation rules
│   ├── repair/
│   │   ├── file_repairer.py              #   Resample + bounded gap repair
│   │   ├── repair_config.py              #   Dataset-specific repair config
│   │   └── repair_utils.py               #   Repair helpers
│   ├── export/
│   │   ├── csv_exporter.py               #   Export CSV with segment_id
│   │   └── segment_builder.py            #   Continuous segment construction
│   └── pipeline/
│       ├── pipeline_runner.py            #   Full dataset pipeline entry
│       └── schemas.py                    #   Pipeline summary schema
│
└── model_pipeline/                       # Model pipeline module
    ├── api.py                             #   Stable plugin input/output contracts
    ├── data_protocol.py                   #   Explicit household splits and target hiding
    ├── model_registry.py                  #   Model discovery and checkpoint utilities
    ├── train_model.py                     #   Public validation-MSE runner and Finetuner adapter
    ├── test_model.py                      #   Common timeline evaluator
    └── models/                            #   Self-contained model plugins
        ├── seq2point/                     #     Point-output CNN/RNN and local variants
        ├── seq2seq/                       #     Sequence-output CNN/Transformer models
        └── classical/                     #     AFHMM, AFHMM-SAC and DSC
```

---

## Supported Datasets

### Raw Datasets

| Dataset | Format | Source |
|---------|--------|--------|
| **UKDALE** | HDF5 (`.h5`) | [CEDA / UK-DALE data browser](https://data.ceda.ac.uk/edc) |
| **REDD** | HDF5 (`.h5`) | Original REDD download links are often unavailable; use a local `redd.h5` converted with NILMTK or another accessible REDD mirror |
| **REFIT** | CSV (CLEAN_House*.csv) | [Download](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned) |
| **ECO** | CSV (unzipped smart meter + plug data) | [Download](https://vs.inf.ethz.ch/res/show.html?what=eco-data), then unzip smart meter and plug-level data into a single folder |

Use `data_wizard.py` to process raw data into training-ready CSV files:

```bash
python data_wizard.py
# Option 1: Run Data Separator  (raw → per-appliance HDF5)
# Option 2: Run Full Data Pipeline (raw/standard_h5 → repaired CSV with segment_id)
```

### Direct CSV Files

If you already have CSV data (or want to prepare your own), files must follow this format:

#### Format Requirements

| Requirement | Specification |
|-------------|---------------|
| **Sampling frequency** | Uniform interval after repair/export (`6s` for UKDALE/REDD/REFIT/ECO, `1ms` for `STANDARD_H5`) |
| **Column 1** | `time` — Timestamp (e.g., `2013-05-17 09:35:12`) |
| **Column 2** | `aggregate` — Aggregate power consumption (Watts, numeric) |
| **Column 3** | `<appliance_name>` – Target appliance power consumption (Watts, numeric) |
| **Column 4** | `segment_id` – Continuous segment identifier |
| **No header required** | Code uses `iloc` positional indexing |
| **No missing values** | NaN rows should be dropped before training |

#### Example CSV

```csv
time,aggregate,dishwasher,segment_id
2013-05-17 09:35:12,223.45,0.00,0
2013-05-17 09:35:18,225.12,0.00,0
2013-05-17 09:35:24,1203.87,981.40,0
2013-05-17 09:35:30,1198.55,976.20,0
```

> **Note:** Exported CSVs are raw power values plus `segment_id`. The `SlidingWindowDataset` in `data_feeder.py` applies z-score normalization automatically during training and prevents windows from crossing segment boundaries.

#### CSV Naming Convention

When generated by the full pipeline: `<appliance>_H<house_number>.csv` (e.g., `dishwasher_H1.csv`)

---

## Supported Models

### Deep Learning Models (PyTorch, gradient-based)

| Model | Registry Key | Family | Target Type | Default Window | Description |
|-------|-------------|--------|-------------|---------------|-------------|
| Seq2Point | `seq2point` | seq2point | point | 599 | Baseline 5-layer CNN |
| Seq2Point_Reduced | `reduced_seq2point` | seq2point | point | 599 | Variant with dropout-heavy reduced CNN |
| Seq2Point_Balanced | `balanced_seq2point` | seq2point | point | 599 | Variant with BatchNorm + MaxPool |
| Seq2Point_LSTM | `legacy_lstm_seq2point` | seq2point | point | 180 | Variant with Conv1D + BiLSTM |
| Seq2Seq | `seq2seq` | seq2seq | sequence | 599 | Fully convolutional seq2seq |
| RNN | `rnn` | rnn | point | 599 | Recurrent baseline |
| WindowGRU | `window_gru` | rnn | point | 599 | Recurrent GRU baseline |
| BiLSTM | `bilstm` | rnn | point | 599 | Recurrent bidirectional LSTM baseline |
| DAE | `dae` | dae | sequence | 599 | Baseline denoising auto-encoder |

### Classical Models (non-gradient)

| Model | Registry Key | Family | Target Type | Default Window | Description |
|-------|-------------|--------|-------------|---------------|-------------|
| AFHMM | `afhmm` | probabilistic | sequence | 599 | Baseline additive factorial hidden Markov model |
| AFHMM_SAC | `afhmm_sac` | probabilistic | sequence | 599 | AFHMM variant with signal aggregate constraints |
| DSC | `dsc` | probabilistic | sequence | 599 | Baseline discriminative sparse coding |

### Adding a Custom Model

1. Create a new file in the most appropriate subdirectory under `model_pipeline/models/`
2. Inherit from `TorchNILMModel` (deep learning) or `ClassicalNILMModel` (classical)
3. Register with the `@register_model` decorator:

```python
from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel

@register_model("my_model", aliases=("My Model",), display_name="My Model")
class MyModel(TorchNILMModel):
    display_name = "My Model"
    model_family = "custom"
    target_type = "point"

    def __init__(self, *, window_size: int = 599, **kwargs):
        super().__init__(window_size=window_size, **kwargs)
        # define layers...

    def forward(self, x):
        # define forward pass...
        return x
```

4. Import it in the relevant subpackage `__init__.py`, then ensure that subpackage is imported by `model_pipeline/models/__init__.py`.

---

## Usage

### Installation

```bash
conda env create -f environment.yml
conda activate nilm
```

### Quick Start

```bash
python main.py
```

This launches the interactive CLI. Before training, prepare your data as CSV files:

- If you already have training-ready CSV files, select them directly in the CLI.
- If you only have raw datasets, run `python data_wizard.py` first to generate CSV files, then select the generated CSVs in the CLI.

```
Welcome to the NILM Training and Evaluation CLI.
1. Train a baseline
2. Evaluate a baseline
3. Fine-tune a baseline
4. Exit
```

### 1. Train a Model

Select option `1`, then follow the prompts:
- Select training CSV file(s)
- Enter appliance name (e.g., `dishwasher`)
- Enter dataset name (e.g., `UKDALE`)
- Select a model from the list
- Configure window size and hyperparameters
- Set model hyperparameter overrides and number of epochs
- Training runs the fixed seeds `42`, `3407`, and `2026`

**Programmatic usage:**

```python
from model_pipeline import train_model

trainer = train_model.Trainer(
    model_name="seq2point",
    train_csv_dirs=["path/to/train.csv"],
    validation_csv_dirs=["path/to/validation.csv"],
    appliance="dishwasher",
    dataset="UKDALE",
    window_length=599,
)
trainer.trainModel(num_epochs=10)
trainer.plotLosses()
```

### 2. Evaluate a Model

Select option `2`, then provide a test CSV and a saved `.pth` checkpoint.

**Programmatic usage:**

```python
from model_pipeline import test_model

evaluator = test_model.Evaluator(
    model_state_dir="saved_models/dishwasher_UKDALE_Seq2Point.pth",
    test_csv_dir="path/to/test.csv",
    dataset="UKDALE",
)
evaluator.evaluate()
evaluator.plotResults()
```

**Output:**
- `prediction_plot_<appliance>_<model>.png` — Full timeline plot
- `zoomed_plot_<appliance>_<model>.png` — Zoomed active region
- `<appliance>_results.csv` — Predictions vs ground truth
- `<appliance>_<model>_metrics.csv` — common test metrics and inference time

### 3. Fine-tune a Model

Select option `3`, then provide a fine-tuning CSV and a pre-trained `.pth` checkpoint.

**Programmatic usage:**

```python
from model_pipeline import train_model

finetuner = train_model.Finetuner(
    model_state_dir="saved_models/dishwasher_UKDALE_Seq2Point.pth",
    finetune_csv_dir="path/to/new_data.csv",
    validation_csv_dirs=["path/to/validation.csv"],
    dataset="UKDALE",
    model_save_dir="saved_models",
)
finetuner.fineTune(max_epochs=50)
finetuner.plotLosses()
```

---

## Data Pipeline Details

### Step 1: Data Separator (`data_separator.py`)

Separates raw dataset files into per-appliance HDF5 files:

```python
from dataset_management.data_separation.data_separator import DataSeparator

separator = DataSeparator(
    file_path="path/to/raw/data",
    save_path="path/to/output",
    dataset_type="UKDALE",       # UKDALE | REDD | REFIT | ECO
    appliance_name="dishwasher",  # optional filter
    num_houses=3,                 # optional limit
)
separator.process_data()
```

Output: `<save_path>/<DATASET>_data_separated/House_<N>/<appliance>_H<N>.h5`

### Step 2: Full Pipeline (`pipeline_runner.py`)

Runs validation, repair, and CSV export after separation:

```python
from dataset_management.pipeline.pipeline_runner import PipelineRunner

summary = PipelineRunner().run(
    dataset_type="UKDALE",  # UKDALE | REDD | REFIT | ECO | STANDARD_H5
    input_path="path/to/raw/data/or/standard_h5",
    workspace_dir="path/to/workspace",
)
print(summary)
```

Output:
- `<workspace>/separated/...`
- `<workspace>/repaired/...`
- `<workspace>/exported_csv/<DATASET>_dataset/<appliance>_H<N>.csv`
- `<workspace>/reports/*.json`

---

## Model Pipeline Details

### Model Registry (`model_registry.py`)

Central registration system using `@register_model` decorator:
- `register_model(name, aliases, display_name)` — Register a model class
- `create_model(name, **kwargs)` — Instantiate a model by name
- `list_models()` — List all registered model display names
- `get_model_entry(name)` — Get model metadata (family, target_type, etc.)
- `load_checkpoint(path)` — Load a `.pth` checkpoint
- `instantiate_from_checkpoint(checkpoint)` — Reconstruct model from checkpoint

### Data Protocol (`data_protocol.py`)

- Loads explicit training, validation and test households into canonical raw-watt partitions.
- If training and validation reference the same CSV set, every continuous series is split chronologically 80:20.
- Validation/test labels are hidden before calling a model plugin.

### Trainer (`train_model.py`)

- Runs fixed household splits and seeds; it does not own model optimizers or losses.
- Selects every model's checkpoint using the same raw-watt validation MSE.
- Keeps checkpoints under `saved_models/` and training plots/history under `result/`.
- Calls each plugin's private `fit()` and `predict()` implementation.

### Evaluator (`test_model.py`)

- Computes MAE, SAE, Precision, Recall, F1, MAE-on and MAE-off.
- Requires every plugin to return raw watts on the complete canonical time axis.
- Clips predictions to `[0, aggregate]`
- Auto-detects active region for zoomed plot
- Exports results CSV and metrics CSV

### Finetuner (`train_model.py`)

- Loads a checkpoint and reuses that model plugin's own training logic.
- Requires separately selected validation households and the same public validation-MSE rule.

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAE** | `mean(|prediction - ground_truth|)` | Mean Absolute Error in Watts |
| **SAE** | `|sum(prediction) - sum(ground_truth)| / sum(|ground_truth|)` | Signal Aggregate Error (energy proportion) |

---

## Environment

- Python 3.10+
- PyTorch >= 2.5.0
- CUDA supported (auto-detected)
- Tested on NVIDIA RTX A6000, CUDA 12.9

