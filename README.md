# Sequence-to-Point (Seq2Point) Toolkit for Non-Intrusive Load Monitoring (NILM)

This repository contains an implementation of Seq2Point models for Non-Intrusive Load Monitoring (NILM), based on the approach introduced in the paper:

Mingjun Zhong, Nigel Goddard, Stephen Sutton, and Charles Gillan (2018).
"Sequence-to-Point Learning with Neural Networks for Non-Intrusive Load Monitoring"

The code has been adapted from the original implementation by Mingjun Zhong: https://github.com/MingjunZhong/seq2point-nilm.

This toolkit standardizes benchmarking for Seq2Point architectures across various energy datasets, providing performance metrics (Mean Absolute Error, Signal Aggregate Error, and Inference Time) in an easy-to-compare format across multiple datasets such as REDD, REFIT, ECO, and UKDALE.

## Project Structure

```
nilm-baseline/
├── main.py                          # Main entry point: Train/Evaluate/Fine-tune CLI
├── data_wizard.py                   # Data engineering CLI tool
│
├── dataset_management/              # Data Engineering Module
│   ├── dataset_registry.py          # Dataset registry center
│   ├── dataset_manager.py           # Dataset manager
│   └── data_separation/
│       ├── data_separator.py        # Data separator
│       └── *_appliance_mappings.json # Appliance mapping configurations for each dataset
│
├── model_pipeline/                  # Model Pipeline Module
│   ├── seq2Point_model.py           # Model definitions
│   ├── seq2Point_factory.py         # Model factory
│   ├── data_feeder.py               # Data loading and normalization
│   ├── train_model.py               # Trainer
│   ├── test_model.py                # Tester
│   └── finetune_model.py            # Fine-tuner
│
├── dataset/                         # Processed datasets
│   └── UKDALE_dataset/              # Example: UKDALE processed data
│       ├── dishwasher_H1.csv
│       ├── fridge_H1.csv
│       └── ...
│
├── result/                          # Evaluation results
│   └── ukdale/
│       ├── dishwasher/
│       └── fridge/
│
└── saved_models/                    # Saved model weights
```

## Data Paths

### Supported Datasets

The toolkit supports the following NILM datasets:

- **UKDALE**: HDF5 format - [Download](https://data.ukedc.rl.ac.uk/cgi-bin/data_browser/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.h5.zip)
- **REFIT**: Cleaned CSV format - [Download](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned)
- **REDD**: HDF5 format - [Download](https://tokhub.github.io/dbecd/links/redd.html)
- **ECO**: Requires pre-processing (unzip smart meter and plug-level data to a single folder)

### Data Processing Pipeline

1. **Raw Data** → `DataSeparator` → **Separated HDF5 files**
2. **Separated Data** → `DatasetManager` → **Training-ready CSV files**

### Processed Data Format

After processing, each appliance dataset is saved as a CSV file with the following format:

```csv
time, aggregate, <appliance_name>
2013-05-17 09:35:12, 0.13711391113313803, -0.07084559614012528
2013-05-17 09:35:18, 0.134153645793339, -0.07084559614012528
2013-05-17 09:35:24, 0.1430344418127361, -0.07084559614012528
```

File naming convention: `<appliance>_H<house_number>.csv` (e.g., `dishwasher_H1.csv`)

## Usage

### 1. Data Engineering (data_wizard.py)

Run the data engineering CLI tool:

```bash
python data_wizard.py
```

**Options:**

1. **Run Data Separator** - Separate raw data into appliance-specific HDF5 files
2. **Run Dataset Manager** - Convert separated data to training-ready CSV format

**Data Separator Parameters:**

- File path to raw data
- Save path for separated data
- Dataset type (UKDALE, REDD, REFIT, ECO)
- Appliance name (optional, processes all if not specified)
- Maximum number of houses to process

**Dataset Manager Parameters:**

- Path to separated data
- Save path for processed CSVs
- Dataset type
- Appliance name
- Debug mode (y/n)
- Maximum number of houses
- Maximum number of rows

### 2. Model Training/Evaluation/Fine-tuning (main.py)

Run the main CLI:

```bash
python main.py
```

**Menu Options:**

1. **Train a model** - Train a new Seq2Point model
2. **Evaluate a model** - Test a trained model
3. **Fine-tune a model** - Fine-tune an existing model
4. **Exit**

#### Training a Model

When selecting "Train a model", you will be prompted to:

1. Select training CSV files (via file dialog or manual input)
2. Enter appliance name (e.g., dishwasher, fridge, kettle)
3. Enter dataset name (e.g., UKDALE, REDD)
4. Select model architecture:
   - Original Seq2Point
   - Balanced Seq2Point
   - Dropout-Reduced Seq2Point
   - LSTM-Based Seq2Point
5. Set hyperparameters:
   - Input window length (default: 599)
   - Number of epochs (default: 10)
   - Validation ratio (default: 0.2)
   - Random seed (default: 42)

**Output:**

- Trained model saved to `saved_models/` directory
- Loss plots generated

#### Evaluating a Model

When selecting "Evaluate a model", you will be prompted to:

1. Select test CSV file
2. Select trained model file (.pth)
3. Enter dataset name

**Output:**

- Performance metrics:
  - MAE (Mean Absolute Error) in Watts
  - SAE (Signal Aggregate Error)
  - Inference time in seconds
- Prediction plots
- Results saved to `result/<dataset>/<appliance>/`

#### Fine-tuning a Model

When selecting "Fine-tune a model", you will be prompted to:

1. Select fine-tuning CSV file
2. Select pre-trained model file (.pth)
3. Enter dataset name
4. Set maximum epochs (default: 50)
5. Set random seed (default: 42)

**Output:**

- Fine-tuned model saved to `saved_models/`
- Loss plots generated

## Module Documentation

### dataset_management/

#### dataset_registry.py

Central registry for supported datasets and their configurations.

#### dataset_manager.py

Loads, processes, and prepares appliance-specific data for training:

- Data selection and resampling
- Normalization
- Gap detection and data quality filtering
- CSV file generation

#### data_separator.py

Disaggregates appliance-specific power consumption from raw NILM datasets:

- Supports UKDALE, REFIT, REDD, ECO formats
- Outputs HDF5 files

### model_pipeline/

#### seq2Point_model.py

Neural network model definitions for Seq2Point architectures.

#### seq2Point_factory.py

Factory class for creating different Seq2Point model variants.

#### data_feeder.py

Handles data loading, batching, and normalization for training/testing.

#### train_model.py

Training pipeline with:

- Configurable hyperparameters
- Validation split
- Loss tracking and plotting

#### test_model.py

Evaluation pipeline with:

- Model inference
- Metric calculation (MAE, SAE, inference time)
- Result visualization

#### finetune_model.py

Fine-tuning pipeline for adapting pre-trained models to new data.

## Performance Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual power consumption (Watts)
- **SAE (Signal Aggregate Error)**: Relative error in total energy consumption
- **Inference Time**: Time taken to generate predictions (seconds)

