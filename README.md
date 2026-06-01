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
│   ├── dataset_registry.py               #   Dataset & appliance registry
│   ├── dataset_manager.py                #   Data loading, resampling, chunk selection
│   └── data_separation/
│       ├── data_separator.py             #   Raw data → per-appliance HDF5
│       ├── eco_data_ranges.json          #   ECO dataset date ranges
│       ├── ukdale_appliance_mappings.json
│       ├── redd_appliance_mappings.json
│       ├── refit_appliance_mappings.json
│       └── eco_appliance_mappings.json
│
└── model_pipeline/                       # Model pipeline module
    ├── __init__.py                        #   Package entry, exports public API
    ├── model_registry.py                 #   Model registration & checkpoint utilities
    ├── data_feeder.py                    #   SlidingWindowDataset & series reconstruction
    ├── train_model.py                    #   Trainer class
    ├── test_model.py                     #   Evaluator class
    ├── finetune_model.py                 #   Finetuner class
    └── models/                           #   Model implementations
        ├── __init__.py                    #     Auto-imports all model modules
        ├── base_model.py                 #     BaseNILMModel / TorchNILMModel / ClassicalNILMModel
        ├── seq2point/
        │   ├── __init__.py               #     Seq2Point family imports
        │   ├── cnn.py                    #     Seq2Point (baseline) + Reduced + Balanced
        │   └── rnn.py                    #     Seq2Point_LSTM + RNN / WindowGRU / BiLSTM
        ├── seq2seq/
        │   ├── __init__.py               #     Seq2Seq family imports
        │   ├── cnn.py                    #     Seq2Seq (baseline)
        │   └── autoencoder.py            #     DAE (baseline)
        └── classical/
            ├── __init__.py               #     Classical family imports
            ├── afhmm.py                  #     AFHMM (baseline) + AFHMM_SAC (variant)
            └── dsc.py                    #     DSC (baseline)
```

---

## Supported Datasets

### Raw Datasets

| Dataset | Format | Source |
|---------|--------|--------|
| **UKDALE** | HDF5 (`.h5`) | [Download](https://data.ukedc.rl.ac.uk/cgi-bin/data_browser/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.h5.zip) |
| **REDD** | HDF5 (`.h5`) | [Download](https://tokhub.github.io/dbecd/links/redd.html) |
| **REFIT** | CSV (CLEAN_House*.csv) | [Download](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned) |
| **ECO** | CSV (unzipped smart meter + plug data) | Requires pre-processing: unzip smart meter and plug-level data into a single folder |

Use `data_wizard.py` to process raw data into training-ready CSV files:

```bash
python data_wizard.py
# Option 1: Run Data Separator  (raw → per-appliance HDF5)
# Option 2: Run Dataset Manager (HDF5 → training CSV)
```

### Direct CSV Files

If you already have CSV data (or want to prepare your own), files must follow this format:

#### Format Requirements

| Requirement | Specification |
|-------------|---------------|
| **Sampling frequency** | Uniform interval (default: 6 seconds after `dataset_manager` resampling) |
| **Column 1** | `time` — Timestamp (e.g., `2013-05-17 09:35:12`) |
| **Column 2** | `aggregate` — Aggregate power consumption (Watts, numeric) |
| **Column 3** | `<appliance_name>` — Target appliance power consumption (Watts, numeric) |
| **No header required** | Code uses `iloc` positional indexing |
| **No missing values** | NaN rows should be dropped before training |

#### Example CSV

```csv
time,aggregate,dishwasher
2013-05-17 09:35:12,223.45,0.00
2013-05-17 09:35:18,225.12,0.00
2013-05-17 09:35:24,1203.87,981.40
2013-05-17 09:35:30,1198.55,976.20
```

> **Note:** The `dataset_manager` outputs **normalized** values (z-score). The `SlidingWindowDataset` in `data_feeder.py` applies z-score normalization automatically during training. For direct CSV use, you may provide either raw or normalized values — if raw, the `SlidingWindowDataset` will normalize them internally.

#### CSV Naming Convention

When generated by `dataset_manager`: `<appliance>_H<house_number>.csv` (e.g., `dishwasher_H1.csv`)

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

### Model Provenance and Citations

| Model / Family | Current File | Registry Key(s) | Reference Repository | Paper Citation |
|-------|-------------|-------------|----------------------|----------------|
| Seq2Point / Seq2Point_Reduced / Seq2Point_Balanced | `model_pipeline/models/seq2point/seq2point.py`, `model_pipeline/models/seq2point/reduced_seq2point.py`, `model_pipeline/models/seq2point/balanced_seq2point.py` | `seq2point`, `reduced_seq2point`, `balanced_seq2point` | `https://github.com/MingjunZhong/seq2point-nilm` | Zhong, M., Goddard, N., Sutton, C., and Gillian, C. (2018). *Sequence-to-point learning with neural networks for non-intrusive load monitoring*. AAAI. |
| Seq2Point_LSTM | `model_pipeline/models/seq2point/seq2point_lstm.py` | `legacy_lstm_seq2point` | Implemented for this project | Zhong, M., Goddard, N., Sutton, C., and Gillian, C. (2018). *Sequence-to-point learning with neural networks for non-intrusive load monitoring*. AAAI. |
| Seq2Seq | `model_pipeline/models/seq2seq/cnn.py` | `seq2seq` | Related baseline collection: `https://github.com/nilmtk/nilmtk-contrib` | Kelly, J., and Knottenbelt, W. (2015). *Neural NILM: Deep Neural Networks Applied to Energy Disaggregation*. BuildSys. Batra, N. et al. (2019). *Towards Reproducible State-of-the-Art Energy Disaggregation*. BuildSys. |
| RNN / WindowGRU / BiLSTM | `model_pipeline/models/seq2point/rnn.py`, `model_pipeline/models/seq2point/window_gru.py`, `model_pipeline/models/seq2point/bilstm.py` | `rnn`, `window_gru`, `bilstm` | Related baseline collection: `https://github.com/nilmtk/nilmtk-contrib` | Kelly, J., and Knottenbelt, W. (2015). *Neural NILM: Deep Neural Networks Applied to Energy Disaggregation*. BuildSys. Batra, N. et al. (2019). *Towards Reproducible State-of-the-Art Energy Disaggregation*. BuildSys. |
| DAE | `model_pipeline/models/seq2seq/dae.py` | `dae` | Related baseline collection: `https://github.com/nilmtk/nilmtk-contrib` | Kelly, J., and Knottenbelt, W. (2015). *Neural NILM: Deep Neural Networks Applied to Energy Disaggregation*. BuildSys. Batra, N. et al. (2019). *Towards Reproducible State-of-the-Art Energy Disaggregation*. BuildSys. |
| AFHMM / AFHMM_SAC | `model_pipeline/models/classical/afhmm.py` | `afhmm`, `afhmm_sac` | Related baseline collection: `https://github.com/nilmtk/nilmtk-contrib` | Zhong, M., Goddard, N., and Sutton, C. (2014). *Signal Aggregate Constraints in Additive Factorial HMMs, with Application to Energy Disaggregation*. NeurIPS. Kolter, J. Z., and Jaakkola, T. (2012). *Approximate Inference in Additive Factorial HMMs with Application to Energy Disaggregation*. AISTATS. |
| DSC | `model_pipeline/models/classical/dsc.py` | `dsc` | Related baseline collection: `https://github.com/nilmtk/nilmtk-contrib` | Kolter, J. Z., Batra, S., and Ng, A. Y. (2010). *Energy Disaggregation via Discriminative Sparse Coding*. NeurIPS. Batra, N. et al. (2019). *Towards Reproducible State-of-the-Art Energy Disaggregation*. BuildSys. |

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
pip install -r requirements.txt
```

### Quick Start

```bash
python main.py
```

This launches the interactive CLI:

```
Welcome to the NILM Training and Evaluation CLI.
1. Train a baseline
2. Evaluate a baseline
3. Fine-tune a baseline
4. Exit
```

### 1. Train a Model

Select option `1`, then follow the prompts:
- Provide training CSV file(s)
- Enter appliance name (e.g., `dishwasher`)
- Enter dataset name (e.g., `UKDALE`)
- Select a model from the list
- Configure window size and hyperparameters
- Set number of epochs, validation ratio, and random seed

**Programmatic usage:**

```python
from model_pipeline import train_model

trainer = train_model.Trainer(
    model_name="seq2point",
    train_csv_dirs=["path/to/train.csv"],
    appliance="dishwasher",
    dataset="UKDALE",
    window_length=599,
    num_epochs=10,
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
- `<appliance>_<model>_metrics.csv` — MAE, SAE, inference time

### 3. Fine-tune a Model

Select option `3`, then provide a fine-tuning CSV and a pre-trained `.pth` checkpoint.

**Programmatic usage:**

```python
from model_pipeline import finetune_model

finetuner = finetune_model.Finetuner(
    model_state_dir="saved_models/dishwasher_UKDALE_Seq2Point.pth",
    finetune_csv_dir="path/to/new_data.csv",
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

### Step 2: Dataset Manager (`dataset_manager.py`)

Loads separated HDF5 files, resamples to 6-second intervals, selects the best continuous chunk, and exports CSV:

```python
from dataset_management.dataset_manager import DatasetManager

manager = DatasetManager(
    data_directory="path/to/separated/data",
    save_path="path/to/csv/output",
    dataset="ukdale",
    appliance_name="dishwasher",
    max_num_houses=3,
    max_num_rows=1_000_000,
)
manager.createData()
```

Output: `<save_path>/<appliance>_H<N>.csv`

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

### Data Feeder (`data_feeder.py`)

- `SlidingWindowDataset` — PyTorch Dataset that builds sliding windows from CSV files with z-score normalization and train/val splitting
- `reconstruct_series_from_windows()` — Reconstructs a full time series from overlapping window predictions via averaging

### Trainer (`train_model.py`)

- Adam optimizer with ReduceLROnPlateau scheduler
- Early stopping (patience=8)
- Saves best checkpoint (model state + metadata)
- Supports both gradient-based (PyTorch) and classical (sklearn-style `fit`) models
- Loss plotting

### Evaluator (`test_model.py`)

- Computes MAE and SAE metrics
- Denormalizes predictions back to Watts
- Reconstructs full series from windowed output
- Clips predictions to `[0, aggregate]`
- Auto-detects active region for zoomed plot
- Exports results CSV and metrics CSV

### Finetuner (`finetune_model.py`)

- Loads checkpoint and freezes all layers except `nn.Linear`
- Trains with no validation split (full data for adaptation)
- Supports gradient-based models only

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

