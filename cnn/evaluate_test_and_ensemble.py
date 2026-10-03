import os

from pathlib import Path

from typing import Dict, List, Tuple, Optional

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from PIL import Image

from sklearn.metrics import (

    accuracy_score,

    confusion_matrix,

    precision_recall_curve,

    precision_recall_fscore_support,

    roc_auc_score,

)

from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================

# 1. MASTER CONFIGURATIONS

# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 2

EXPECTED_FOLDS = 5

TASKS = [

    {

        "name": "RGB - IDENTIFY",

        "modality": "rgb",

        "data_root": "/content/drive/MyDrive/Thesis/data/splits_identify_cv_with_test",

        "out_base_dir": "/content/drive/MyDrive/Thesis/results/models_resnet_pro/rgb_cv_results",

        "img_h": 512,

        "img_w": 768,

        "target_col": "Crack",

        "labels": ["Not Crack", "Crack"],

        # Identify: only Season 1 test

        "test_sets": [

            ("Test Set", "test"),

        ],

    },

    {

        "name": "THERMAL - IDENTIFY",

        "modality": "thermal",

        "data_root": "/content/drive/MyDrive/Thesis/data/splits_identify_cv_with_test",

        "out_base_dir": "/content/drive/MyDrive/Thesis/results/models_resnet_pro/thermal_cv_results",

        "img_h": 768,

        "img_w": 1024,

        "target_col": "Crack",

        "labels": ["Not Crack", "Crack"],

        # Identify: only Season 1 test

        "test_sets": [

            ("Test Set", "test"),

        ],

    },

    {

        "name": "RGB - PREDICT",

        "modality": "rgb",

        "data_root": "/content/drive/MyDrive/Thesis/data/splits_predict_cv",

        "out_base_dir": "/content/drive/MyDrive/Thesis/results/models_resnet_pro_predict/rgb_cv_results",

        "img_h": 512,

        "img_w": 768,

        "target_col": "cracked_next_session",

        "labels": ["Not Crack Next", "Crack Next"],

        # Predict: Season 1 test + external Season 2 test

        "test_sets": [

            ("Test Set (Season 1)", "test"),

            ("External Test Set (Season 2)", "external_test_season2"),

        ],

    },

    {

        "name": "THERMAL - PREDICT",

        "modality": "thermal",

        "data_root": "/content/drive/MyDrive/Thesis/data/splits_predict_cv",

        "out_base_dir": "/content/drive/MyDrive/Thesis/results/models_resnet_pro_predict/thermal_cv_results",

        "img_h": 768,

        "img_w": 1024,

        "target_col": "cracked_next_session",

        "labels": ["Not Crack Next", "Crack Next"],

        # Predict: Season 1 test + external Season 2 test

        "test_sets": [

            ("Test Set (Season 1)", "test"),

            ("External Test Set (Season 2)", "external_test_season2"),

        ],

    },

]

# ============================================================

# 2. CUSTOM DATASET

# ============================================================

def label_to_float(value) -> float:

    """Convert common numeric/string binary labels to 0.0 or 1.0."""

    if pd.isna(value):

        raise ValueError("Encountered a missing target label.")

    if isinstance(value, str):

        normalized = value.strip().lower()

        positive_values = {"1", "true", "yes", "y", "crack", "cracked", "positive"}

        negative_values = {"0", "false", "no", "n", "not crack", "not cracked", "negative"}

        if normalized in positive_values:

            return 1.0

        if normalized in negative_values:

            return 0.0

    numeric_value = float(value)

    if numeric_value not in (0.0, 1.0):

        raise ValueError(f"Target label must be binary 0/1, received: {value}")

    return numeric_value

class PomegranateCSVDataset(Dataset):

    def __init__(self, csv_path: str, cfg: dict, transform=None):

        self.csv_path = Path(csv_path)

        self.csv_dir = self.csv_path.parent

        self.df = pd.read_csv(self.csv_path)

        self.transform = transform

        self.cfg = cfg

        if cfg["target_col"] not in self.df.columns:

            raise KeyError(

                f"Target column '{cfg['target_col']}' was not found in {self.csv_path}. "

                f"Available columns: {list(self.df.columns)}"

            )

        if cfg["modality"] == "rgb":

            path_candidates = [

                "seg_rgb_path",

                "rgb_path",

                "rgb_image_path",

                "rgb_file_path",

            ]

        else:

            path_candidates = [

                "seg_thermal_path",

                "thermal_path",

                "thermal_image_path",

                "thermal_file_path",

            ]

        self.path_col = next((c for c in path_candidates if c in self.df.columns), None)

        if self.path_col is None:

            raise KeyError(

                f"No valid {cfg['modality']} path column was found in {self.csv_path}. "

                f"Tried: {path_candidates}"

            )

        valid_path_mask = (

            self.df[self.path_col].notna()

            & (self.df[self.path_col].astype(str).str.strip() != "")

            & (self.df[self.path_col].astype(str).str.lower() != "missing")

        )

        valid_target_mask = self.df[cfg["target_col"]].notna()

        self.df = self.df[valid_path_mask & valid_target_mask].reset_index(drop=True)

        if self.df.empty:

            raise ValueError(f"No valid rows remained after filtering {self.csv_path}.")

    def __len__(self):

        return len(self.df)

    def resolve_image_path(self, raw_path) -> Path:

        image_path = Path(str(raw_path))

        if not image_path.is_absolute():

            image_path = self.csv_dir / image_path

        return image_path

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = self.resolve_image_path(row[self.path_col])

        try:

            with Image.open(image_path) as img:

                image = img.convert("RGB")

        except Exception as exc:

            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        label = torch.tensor(

            [label_to_float(row[self.cfg["target_col"]])],

            dtype=torch.float32,

        )

        if self.transform:

            image = self.transform(image)

        return image, label

# ============================================================

# 3. MODEL AND METRIC HELPERS

# ============================================================

def build_model() -> nn.Module:

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 1)

    return model

def load_model_from_checkpoint(model_path: Path, device: torch.device) -> nn.Module:

    model = build_model().to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, nn.Module):

        model = checkpoint.to(device)

    else:

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

        # Support checkpoints saved from DataParallel.

        if isinstance(state_dict, dict):

            state_dict = {

                key.replace("module.", "", 1) if key.startswith("module.") else key: value

                for key, value in state_dict.items()

            }

        model.load_state_dict(state_dict)

    model.eval()

    return model

def safe_filename(text: str) -> str:

    result = text.strip()

    for old, new in [

        (" ", "_"),

        ("(", ""),

        (")", ""),

        ("/", "_"),

        ("\\", "_"),

        (":", "_"),

    ]:

        result = result.replace(old, new)

    while "__" in result:

        result = result.replace("__", "_")

    return result

def find_best_threshold_f1(y_true, y_prob) -> Tuple[float, float]:

    y_true = np.asarray(y_true, dtype=int)

    y_prob = np.asarray(y_prob, dtype=float)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:

        default_threshold = 0.5

        default_preds = (y_prob >= default_threshold).astype(int)

        _, _, f1, _ = precision_recall_fscore_support(

            y_true,

            default_preds,

            average="binary",

            zero_division=0,

        )

        return default_threshold, float(f1)

    f1_values = (2 * precisions[:-1] * recalls[:-1]) / (

        precisions[:-1] + recalls[:-1] + 1e-12

    )

    best_idx = int(np.nanargmax(f1_values))

    return float(thresholds[best_idx]), float(f1_values[best_idx])

def calculate_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:

    y_true = np.asarray(y_true, dtype=int)

    y_pred = np.asarray(y_pred, dtype=int)

    y_prob = np.asarray(y_prob, dtype=float)

    precision, recall, f1, _ = precision_recall_fscore_support(

        y_true,

        y_pred,

        average="binary",

        zero_division=0,

    )

    if len(np.unique(y_true)) == 2:

        auc = float(roc_auc_score(y_true, y_prob))

    else:

        auc = np.nan

    return {

        "Accuracy": float(accuracy_score(y_true, y_pred)),

        "Precision": float(precision),

        "Recall": float(recall),

        "F1-Score": float(f1),

        "ROC-AUC": auc,

        "N": int(len(y_true)),

    }

def plot_confusion_matrix(cm, out_path: Path, title: str, labels: List[str]):

    fig, ax = plt.subplots(figsize=(4.8, 4.4))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels(labels, rotation=15, ha="right")

    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")

    ax.set_ylabel("Actual label")

    ax.set_title(title, fontsize=10, pad=10)

    max_value = cm.max() if cm.size else 0

    threshold = max_value / 2 if max_value > 0 else 0

    for i in range(2):

        for j in range(2):

            ax.text(

                j,

                i,

                int(cm[i, j]),

                ha="center",

                va="center",

                color="white" if cm[i, j] > threshold else "black",

                fontsize=12,

            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    fig.savefig(out_path, dpi=180, bbox_inches="tight")

    plt.close(fig)

def classify_outcome(actual: int, predicted: int) -> str:

    if actual == 1 and predicted == 1:

        return "TP"

    if actual == 0 and predicted == 0:

        return "TN"

    if actual == 0 and predicted == 1:

        return "FP"

    return "FN"

def build_predictions_report(

    dataset: PomegranateCSVDataset,

    y_true,

    y_pred,

    y_prob,

    threshold: float,

    model_source: str,

) -> pd.DataFrame:

    y_true = np.asarray(y_true, dtype=int)

    y_pred = np.asarray(y_pred, dtype=int)

    y_prob = np.asarray(y_prob, dtype=float)

    report = pd.DataFrame()

    # Include useful identifying columns when they exist in the metadata.

    possible_metadata_columns = [

        "ID",

        "id",

        "Fruit_ID",

        "fruit_id",

        "Pomegranate_ID",

        "pomegranate_id",

        "Session",

        "session",

        "Tree",

        "tree",

        "Plot",

        "plot",

        "Treatment",

        "treatment",

    ]

    for column in possible_metadata_columns:

        if column in dataset.df.columns and column not in report.columns:

            report[column] = dataset.df[column].tolist()

    report["Image_Path"] = [

        str(dataset.resolve_image_path(path_value))

        for path_value in dataset.df[dataset.path_col].tolist()

    ]

    report["Actual_Numeric"] = y_true

    report["Predicted_Numeric"] = y_pred

    report["Actual"] = [dataset.cfg["labels"][value] for value in y_true]

    report["Predicted"] = [dataset.cfg["labels"][value] for value in y_pred]

    report["Probability"] = y_prob

    report["Threshold"] = float(threshold)

    report["Outcome"] = [

        classify_outcome(actual, predicted)

        for actual, predicted in zip(y_true, y_pred)

    ]

    report["Model_Source"] = model_source

    return report

def run_inference(

    model: nn.Module,

    data_loader: DataLoader,

    device: torch.device,

    description: str,

) -> Tuple[np.ndarray, np.ndarray]:

    all_probs: List[float] = []

    all_labels: List[int] = []

    model.eval()

    with torch.no_grad():

        for images, labels in tqdm(data_loader, desc=description, leave=False):

            images = images.to(device, non_blocking=True)

            logits = model(images)

            probabilities = torch.sigmoid(logits).view(-1)

            all_probs.extend(probabilities.cpu().numpy().tolist())

            all_labels.extend(labels.view(-1).cpu().numpy().astype(int).tolist())

    return np.asarray(all_probs, dtype=float), np.asarray(all_labels, dtype=int)

def create_loader(

    csv_path: Path,

    cfg: dict,

    eval_transforms,

    device: torch.device,

) -> Tuple[PomegranateCSVDataset, DataLoader]:

    dataset = PomegranateCSVDataset(str(csv_path), cfg, transform=eval_transforms)

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(device.type == "cuda"),

    )

    return dataset, loader

def locate_test_csv(root: Path, test_folder: str) -> Optional[Path]:

    primary_path = root / test_folder / "metadata.csv"

    fallback_path = root / f"{test_folder}_metadata.csv"

    if primary_path.exists():

        return primary_path

    if fallback_path.exists():

        return fallback_path

    return None

# ============================================================

# 4. TEST-EVALUATION HELPERS

# ============================================================

def evaluate_single_model_on_test(

    test_csv_path: Path,

    test_name: str,

    model: nn.Module,

    model_name: str,

    model_threshold: float,

    cfg: dict,

    eval_transforms,

    out_dir: Path,

    device: torch.device,

) -> Dict:

    print(f"\n🧪 Running {model_name} on: {test_name}")

    test_dataset, test_loader = create_loader(

        test_csv_path,

        cfg,

        eval_transforms,

        device,

    )

    probabilities, labels = run_inference(

        model,

        test_loader,

        device,

        description=f"{test_name} - {model_name}",

    )

    predictions = (probabilities >= model_threshold).astype(int)

    metrics = calculate_metrics(labels, predictions, probabilities)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    test_safe_name = safe_filename(test_name)

    model_safe_name = safe_filename(model_name.lower())

    plot_confusion_matrix(

        cm,

        out_dir / f"{test_safe_name}_{model_safe_name}_confusion_matrix.png",

        f"{test_name} - {model_name}",

        cfg["labels"],

    )

    report = build_predictions_report(

        test_dataset,

        labels,

        predictions,

        probabilities,

        model_threshold,

        model_source=model_name,

    )

    report.to_csv(

        out_dir / f"{test_safe_name}_{model_safe_name}_predictions_report.csv",

        index=False,

    )

    return {

        "Source": f"{test_name} ({model_name})",

        "Dataset": test_name,

        "Model": model_name,

        **metrics,

        "Threshold": float(model_threshold),

    }

def evaluate_ensemble_on_test(

    test_csv_path: Path,

    test_name: str,

    loaded_models: List[nn.Module],

    ensemble_threshold: float,

    cfg: dict,

    eval_transforms,

    out_dir: Path,

    device: torch.device,

) -> Dict:

    print(f"\n🧪 Running ensemble on: {test_name}")

    test_dataset, test_loader = create_loader(

        test_csv_path,

        cfg,

        eval_transforms,

        device,

    )

    model_probability_arrays: List[np.ndarray] = []

    labels_reference: Optional[np.ndarray] = None

    for model_index, model in enumerate(loaded_models, start=1):

        probabilities, labels = run_inference(

            model,

            test_loader,

            device,

            description=f"{test_name} - Ensemble model {model_index}/{len(loaded_models)}",

        )

        model_probability_arrays.append(probabilities)

        if labels_reference is None:

            labels_reference = labels

        elif not np.array_equal(labels_reference, labels):

            raise RuntimeError("Label order changed between ensemble model evaluations.")

    ensemble_probabilities = np.mean(np.vstack(model_probability_arrays), axis=0)

    ensemble_predictions = (ensemble_probabilities >= ensemble_threshold).astype(int)

    metrics = calculate_metrics(

        labels_reference,

        ensemble_predictions,

        ensemble_probabilities,

    )

    cm = confusion_matrix(labels_reference, ensemble_predictions, labels=[0, 1])

    test_safe_name = safe_filename(test_name)

    plot_confusion_matrix(

        cm,

        out_dir / f"{test_safe_name}_ensemble_confusion_matrix.png",

        f"{test_name} - Ensemble",

        cfg["labels"],

    )

    report = build_predictions_report(

        test_dataset,

        labels_reference,

        ensemble_predictions,

        ensemble_probabilities,

        ensemble_threshold,

        model_source="Ensemble",

    )

    report.to_csv(

        out_dir / f"{test_safe_name}_ensemble_predictions_report.csv",

        index=False,

    )

    return {

        "Source": f"{test_name} (Ensemble)",

        "Dataset": test_name,

        "Model": "Ensemble",

        **metrics,

        "Threshold": float(ensemble_threshold),

    }

# ============================================================

# 5. RUN A SINGLE TASK PIPELINE

# ============================================================

def run_evaluation(cfg: dict, device: torch.device):

    print("\n" + "═" * 80)

    print(f"🚀 Starting task: {cfg['name']}")

    print(f"   Resolution: {cfg['img_h']}x{cfg['img_w']} | Target: {cfg['target_col']}")

    print("═" * 80)

    root = Path(cfg["data_root"])

    out_dir = Path(cfg["out_base_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():

        raise FileNotFoundError(f"Data root does not exist: {root}")

    normalize = transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225],

    )

    eval_transforms = transforms.Compose(

        [

            transforms.Resize((cfg["img_h"], cfg["img_w"])),

            transforms.ToTensor(),

            normalize,

        ]

    )

    loaded_models: List[nn.Module] = []

    loaded_fold_names: List[str] = []

    fold_thresholds: Dict[str, float] = {}

    cv_records: List[Dict] = []

    best_overall_f1 = -1.0

    best_overall_fold_name: Optional[str] = None

    best_overall_cm = None

    best_overall_df = None

    best_fold_summary_row = None

    best_fold_model: Optional[nn.Module] = None

    missing_items: List[str] = []

    # --------------------------------------------------------

    # Validation evaluation for every fold

    # --------------------------------------------------------

    for fold_idx in range(1, EXPECTED_FOLDS + 1):

        fold_name = f"fold_{fold_idx:02d}"

        fold_out_dir = out_dir / fold_name

        model_path = fold_out_dir / "best_model.pt"

        val_csv_path = root / fold_name / "val" / "metadata.csv"

        if not model_path.exists():

            missing_items.append(str(model_path))

            continue

        if not val_csv_path.exists():

            missing_items.append(str(val_csv_path))

            continue

        print(f"🔄 Processing {fold_name.upper()}...")

        model = load_model_from_checkpoint(model_path, device)

        loaded_models.append(model)

        loaded_fold_names.append(fold_name)

        val_dataset, val_loader = create_loader(

            val_csv_path,

            cfg,

            eval_transforms,

            device,

        )

        probabilities, labels = run_inference(

            model,

            val_loader,

            device,

            description=f"Validation inference {fold_name}",

        )

        best_threshold, _ = find_best_threshold_f1(labels, probabilities)

        fold_thresholds[fold_name] = best_threshold

        predictions = (probabilities >= best_threshold).astype(int)

        metrics = calculate_metrics(labels, predictions, probabilities)

        cm = confusion_matrix(labels, predictions, labels=[0, 1])

        fold_report = build_predictions_report(

            val_dataset,

            labels,

            predictions,

            probabilities,

            best_threshold,

            model_source=fold_name.upper(),

        )

        plot_confusion_matrix(

            cm,

            fold_out_dir / f"{fold_name}_confusion_matrix.png",

            f"Validation CM - {fold_name.upper()}",

            cfg["labels"],

        )

        fold_report.to_csv(

            fold_out_dir / f"{fold_name}_predictions_report.csv",

            index=False,

        )

        fold_metrics = {

            "Source": fold_name.upper(),

            "Dataset": "Validation",

            "Model": fold_name.upper(),

            **metrics,

            "Threshold": float(best_threshold),

        }

        cv_records.append(fold_metrics)

        if metrics["F1-Score"] > best_overall_f1:

            best_overall_f1 = metrics["F1-Score"]

            best_overall_fold_name = fold_name

            best_overall_cm = cm

            best_overall_df = fold_report

            best_fold_model = model

            best_fold_summary_row = fold_metrics.copy()

    if missing_items:

        missing_text = "\n".join(f"  - {item}" for item in missing_items)

        raise FileNotFoundError(

            "The following required fold files are missing:\n" + missing_text

        )

    if len(loaded_models) != EXPECTED_FOLDS:

        raise RuntimeError(

            f"Expected {EXPECTED_FOLDS} loaded fold models, but found {len(loaded_models)}."

        )

    if (

        best_overall_fold_name is None

        or best_fold_model is None

        or best_fold_summary_row is None

    ):

        raise RuntimeError("Could not determine the best validation fold.")

    # --------------------------------------------------------

    # Save the best fold's VALIDATION outputs in the task root

    # --------------------------------------------------------

    print(

        f"\n⭐ Best validation fold: {best_overall_fold_name.upper()} "

        f"(F1: {best_overall_f1:.3f})"

    )

    plot_confusion_matrix(

        best_overall_cm,

        out_dir / "best_fold_validation_confusion_matrix.png",

        f"Best Validation Fold CM ({best_overall_fold_name.upper()})",

        cfg["labels"],

    )

    best_overall_df.to_csv(

        out_dir / "best_fold_validation_predictions_report.csv",

        index=False,

    )

    best_fold_summary_row["Source"] = (

        f"BEST FOLD VALIDATION ({best_overall_fold_name.upper()})"

    )

    best_fold_summary_row["Model"] = best_overall_fold_name.upper()

    # Ensemble threshold: mean of the fold-specific validation thresholds.

    ensemble_threshold = float(np.mean(list(fold_thresholds.values())))

    best_fold_threshold = float(fold_thresholds[best_overall_fold_name])

    # --------------------------------------------------------

    # Evaluate BOTH the best fold and ensemble on every test set

    # --------------------------------------------------------

    test_summary_rows: List[Dict] = []

    for test_label, test_folder in cfg["test_sets"]:

        test_csv_path = locate_test_csv(root, test_folder)

        if test_csv_path is None:

            raise FileNotFoundError(

                f"Could not locate metadata.csv for '{test_label}'. Tried:\n"

                f"  - {root / test_folder / 'metadata.csv'}\n"

                f"  - {root / f'{test_folder}_metadata.csv'}"

            )

        best_fold_test_row = evaluate_single_model_on_test(

            test_csv_path=test_csv_path,

            test_name=test_label,

            model=best_fold_model,

            model_name=f"Best Fold {best_overall_fold_name.upper()}",

            model_threshold=best_fold_threshold,

            cfg=cfg,

            eval_transforms=eval_transforms,

            out_dir=out_dir,

            device=device,

        )

        test_summary_rows.append(best_fold_test_row)

        ensemble_test_row = evaluate_ensemble_on_test(

            test_csv_path=test_csv_path,

            test_name=test_label,

            loaded_models=loaded_models,

            ensemble_threshold=ensemble_threshold,

            cfg=cfg,

            eval_transforms=eval_transforms,

            out_dir=out_dir,

            device=device,

        )

        test_summary_rows.append(ensemble_test_row)

    # --------------------------------------------------------

    # Final summary report

    # --------------------------------------------------------

    validation_df = pd.DataFrame(cv_records)

    numeric_columns = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1-Score",

        "ROC-AUC",

        "Threshold",

    ]

    validation_mean = validation_df[numeric_columns].mean().to_dict()

    validation_mean.update(

        {

            "Source": "VAL MEAN",

            "Dataset": "Validation",

            "Model": "Cross-Validation Mean",

            "N": int(validation_df["N"].sum()),

        }

    )

    validation_std = validation_df[numeric_columns].std(ddof=1).to_dict()

    validation_std.update(

        {

            "Source": "VAL STD",

            "Dataset": "Validation",

            "Model": "Cross-Validation Standard Deviation",

            "N": np.nan,

        }

    )

    final_report_df = pd.concat(

        [

            validation_df,

            pd.DataFrame([validation_mean]),

            pd.DataFrame([validation_std]),

            pd.DataFrame([best_fold_summary_row]),

            pd.DataFrame(test_summary_rows),

        ],

        ignore_index=True,

    )

    preferred_column_order = [

        "Source",

        "Dataset",

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1-Score",

        "ROC-AUC",

        "Threshold",

        "N",

    ]

    final_report_df = final_report_df[preferred_column_order]

    report_name = f"final_report_{cfg['name'].replace(' - ', '_').lower()}.csv"

    final_report_path = out_dir / report_name

    final_report_df.to_csv(final_report_path, index=False)

    print("\n📊 FINAL REPORT")

    print(final_report_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n✅ Finished task. Final report saved to:\n{final_report_path}")

# ============================================================

# 6. EXECUTE ALL TASKS

# ============================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🖥️ Using device: {device.type.upper()}")

    successful_tasks: List[str] = []

    failed_tasks: List[Tuple[str, str]] = []

    for cfg in TASKS:

        try:

            run_evaluation(cfg, device)

            successful_tasks.append(cfg["name"])

        except Exception as exc:

            failed_tasks.append((cfg["name"], str(exc)))

            print(f"\n❌ Task failed: {cfg['name']}")

            print(f"   Error: {exc}")

            print("   Continuing to the next task...")

        if device.type == "cuda":

            torch.cuda.empty_cache()

    print("\n" + "═" * 80)

    print("RUN SUMMARY")

    print("═" * 80)

    if successful_tasks:

        print("✅ Successful tasks:")

        for task_name in successful_tasks:

            print(f"   - {task_name}")

    if failed_tasks:

        print("\n❌ Failed tasks:")

        for task_name, error_message in failed_tasks:

            print(f"   - {task_name}: {error_message}")

        raise RuntimeError(

            f"{len(failed_tasks)} task(s) failed. Review the errors printed above."

        )

    print("\n🎉 All four tasks completed successfully.")

if __name__ == "__main__":

    main()
