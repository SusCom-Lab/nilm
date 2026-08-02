# main.py
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from model_pipeline import get_model_entry, list_models, train_model, test_model
from model_pipeline.api import OFFICIAL_EXPERIMENT_SEEDS
from model_pipeline.model_registry import load_checkpoint
from IPython import get_ipython


def runningColab():
    """
    Return True if running in a Colab kernel.
    """
    try:
        import google.colab  # noqa: F401
        return get_ipython() is not None
    except ImportError:
        return False


def get_path_input(prompt: str, multiple: bool = False):
    """
    Ask the user to paste paths when GUI/file‑upload options are unavailable.
    """
    if multiple:
        paths = input(prompt).strip().split()
        return [p for p in paths if p]
    else:
        path = input(prompt).strip()
        return [path] if path else []


def selectCSVFiles(type, using_colab, *, single_file: bool = False):
    """
    Select CSV files for training / validation / testing.
    """
    print(f"Please provide the {type} CSV file{'s' if not single_file else ''}.")
    # ------------------------------------
    if using_colab:
        # ➜ Colab widget route
        from google.colab import files

        uploaded = files.upload()
        csv_files = [f"/content/{name}" for name in uploaded if name.endswith(".csv")]

        if not csv_files:
            print("No CSV files were uploaded.")
            return []

        if single_file and len(csv_files) > 1:
            print("Multiple files uploaded; using the first one.")
            csv_files = [csv_files[0]]

        return csv_files
    # ------------------------------------
    # Not running in a Colab kernel
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if single_file:
            path = filedialog.askopenfilename(
                title=f"Select {type} CSV file",
                filetypes=[("CSV files", "*.csv")],
            )
            if not path:
                # Fallback to text prompt
                return get_path_input(
                    f"Path to the {type} CSV file: ", multiple=False
                )
            return [path]
        else:
            paths = filedialog.askopenfilenames(
                title=f"Select {type} CSV files",
                filetypes=[("CSV files", "*.csv")],
            )
            if not paths:
                return get_path_input(
                    f"Paths to the {type} CSV files (space‑separated): ",
                    multiple=True,
                )
            return list(paths)

    except Exception:
        if single_file:
            return get_path_input(
                f"Path to the {type} CSV file: ", multiple=False
            )
        else:
            return get_path_input(
                f"Paths to the {type} CSV files (space‑separated): ",
                multiple=True,
            )


def selectModelFile(using_colab: bool):
    """
    Choose the .pth model file.
    """
    print("Please provide the model file (*.pth).")
    if using_colab:
        from google.colab import files

        uploaded = files.upload()
        if not uploaded:
            print("No model file uploaded.")
            return None
        model_path = list(uploaded.keys())[0]
        return f"/content/{model_path}"

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        path = filedialog.askopenfilename(
            title="Select model file", filetypes=[("Model files", "*.pth")]
        )
        if not path:
            path = input("Path to the model .pth file: ").strip()

        return path or None

    except Exception:
        path = input("Path to the model .pth file: ").strip()
        return path or None


def _safe_int_input(prompt: str, default: int) -> int:
    value = input(f"{prompt} (default {default}): ").strip()
    return int(value) if value else default


def _safe_float_input(prompt: str, default: float) -> float:
    value = input(f"{prompt} (default {default}): ").strip()
    return float(value) if value else default


def promptModelSelection():
    available_models = list_models()
    numbered_models = {str(i + 1): model_name for i, model_name in enumerate(available_models)}

    print("Available NILM models:")
    for num, model_name in numbered_models.items():
        entry = get_model_entry(model_name)
        default_window = getattr(entry.cls, "default_window_size", "n/a")
        print(
            f"  {num}: {entry.display_name} "
            f"[family={entry.family}, target={entry.target_type}, default_window={default_window}]"
        )

    model_num = input("Select a model by number: ").strip()
    if model_num not in numbered_models:
        raise ValueError("Invalid model selection.")
    return numbered_models[model_num]


def promptModelConfig(model_name: str):
    """Return official defaults unless the user explicitly overrides them."""

    entry = get_model_entry(model_name)
    default_window = getattr(entry.cls, "default_window_size", None)
    model_init_kwargs = {}
    if default_window is not None:
        model_init_kwargs["window_size"] = _safe_int_input(
            "Input window size", int(default_window)
        )
    raw_kwargs = input(
        "Additional model init kwargs as JSON, or press Enter for defaults: "
    ).strip()
    if raw_kwargs:
        parsed = json.loads(raw_kwargs)
        if not isinstance(parsed, dict):
            raise ValueError("Model init kwargs must be a JSON object.")
        model_init_kwargs.update(parsed)
    return model_init_kwargs


# ---------------------------------------------------------------------
# CLI wrappers
# ---------------------------------------------------------------------
def trainModelCLI():
    print("Training a NILM baseline ...")
    using_colab = runningColab()

    model_name = promptModelSelection()
    model_init_kwargs = promptModelConfig(model_name)
    entry = get_model_entry(model_name)
    is_joint_classical = bool(getattr(entry.cls, "is_joint_classical", False))

    if is_joint_classical:
        print("Joint classical models expect multiple appliance CSVs and group them automatically by H1/H2.")
    train_csv_dirs = selectCSVFiles("train", using_colab)

    if not train_csv_dirs:
        print("File selection aborted. Exiting.")
        return

    validation_csv_dirs = selectCSVFiles("validation", using_colab)
    if not validation_csv_dirs:
        print("Validation file selection aborted. Exiting.")
        return

    appliance = input("Enter the appliance name: ")
    dataset = input("Enter the dataset name: ")

    model_save_dir = "/content" if using_colab else os.path.join(os.getcwd(), "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)

    crop_value = input("Crop rows per CSV for quick runs (press enter for full data): ").strip()
    crop = int(crop_value) if crop_value else None
    default_epochs = int(getattr(entry.cls, "default_num_epochs", 10))
    num_epochs = _safe_int_input("Number of epochs", default_epochs)
    print(f"Using fixed seeds: {OFFICIAL_EXPERIMENT_SEEDS}")
    for seed in OFFICIAL_EXPERIMENT_SEEDS:
        trainer = train_model.Trainer(
            model_name=model_name,
            train_csv_dirs=train_csv_dirs,
            validation_csv_dirs=validation_csv_dirs,
            appliance=appliance,
            dataset=dataset,
            model_save_dir=model_save_dir,
            seed=seed,
            crop=crop,
            model_init_kwargs=model_init_kwargs,
        )
        trainer.trainModel(num_epochs)
        trainer.plotLosses()
    print("Model training completed.")


def evaluateModelCLI():
    print("Evaluating a NILM baseline ...")
    using_colab = runningColab()

    model_file_path = selectModelFile(using_colab)
    if not model_file_path:
        print("No model file selected. Exiting.")
        return

    checkpoint = load_checkpoint(model_file_path)
    joint_mode = bool(
        getattr(get_model_entry(checkpoint["model_key"]).cls, "is_joint_classical", False)
    )

    if joint_mode:
        print("Joint classical evaluation expects multiple appliance CSVs from the same house.")
        test_csv_dirs = selectCSVFiles("test", using_colab, single_file=False)
    else:
        test_csv_dirs = selectCSVFiles("test", using_colab, single_file=True)
    if not test_csv_dirs:
        print("No test file selected. Exiting.")
        return
    test_csv_dir = test_csv_dirs if joint_mode else test_csv_dirs[0]

    dataset = input("Enter the dataset name (e.g., REDD, UKDALE): ")

    tester = test_model.Tester(
        model_state_dir=model_file_path, 
        test_csv_dir=test_csv_dir,
        dataset=dataset
    )
    tester.testModel()
    tester.plotResults()

    metrics_result = tester.saveMetrics()
    print(metrics_result.to_string(index=False))
    print("Model testing completed.")


def fineTuneModelCLI():
    print("Fine‑tuning a NILM baseline ...")
    using_colab = runningColab()

    finetune_csv_dirs = selectCSVFiles("finetune", using_colab, single_file=True)
    if not finetune_csv_dirs:
        print("No finetune file selected. Exiting.")
        return
    finetune_csv_dir = finetune_csv_dirs[0]

    validation_csv_dirs = selectCSVFiles("fine-tune validation", using_colab)
    if not validation_csv_dirs:
        print("Validation file selection aborted. Exiting.")
        return

    model_file_path = selectModelFile(using_colab)
    if not model_file_path:
        print("No model file selected. Exiting.")
        return

    model_save_dir = "/content" if using_colab else os.path.join(os.getcwd(), "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)

    dataset = input("Enter the dataset name: ")
    seed = int(input(f"Random seed {OFFICIAL_EXPERIMENT_SEEDS} (default 42): ") or 42)

    finetuner = train_model.Finetuner(
        model_state_dir=model_file_path,
        finetune_csv_dir=finetune_csv_dir,
        validation_csv_dirs=validation_csv_dirs,
        dataset=dataset,
        model_save_dir=model_save_dir,
        seed=seed,
    )
    finetuner.fineTune()
    finetuner.plotLosses()
    print("Model fine‑tuning completed.")


# ---------------------------------------------------------------------
# Menu loop
# ---------------------------------------------------------------------
def main():
    MENU = (
        "Welcome to the NILM Training and Evaluation CLI.\n"
        "1. Train a model\n"
        "2. Evaluate a model\n"
        "3. Fine-tune a model\n"
        "4. Exit"
    )
    while True:
        print(MENU)
        choice = input("Please enter your choice (1‑4): ").strip()
        if choice == "1":
            trainModelCLI()
        elif choice == "2":
            evaluateModelCLI()
        elif choice == "3":
            fineTuneModelCLI()
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
