"""
Revised multimodal intermediate-fusion pipeline for pomegranate cracking prediction.

Main improvements over the original script
------------------------------------------
1. Strict image loading: missing/corrupted images raise an explicit error.
2. Reproducible global random seeds and saved environment information.
3. Duplicate, merge-cardinality, missing-value, finite-value, and group-overlap checks.
4. Modality-specific PCA is fitted only on training embeddings. RGB and thermal
   streams are standardized and compressed independently, with the number of
   components selected separately for each modality to retain 95% explained variance.
5. Meta-only configurations are not redundantly executed with PCA ON/OFF.
6. XGBoost receives a fold-specific scale_pos_weight when the data are imbalanced.
7. PR-AUC, precision, recall, balanced accuracy, prevalence, and confusion counts
   are included in the summaries.
8. SMOTE sample counts and the effective class ratio are recorded.
9. Prediction files preserve available fruit/session/agronomic identifiers.
10. Optional confusion matrices, PR curves, feature importances, and serialized
    inference pipelines.
11. Optional probability-level ensemble across the five fold-specific pipelines.
12. Optional constrained hyperparameter tuning based primarily on validation F1.

The validation-derived threshold protocol is retained:
- fit on train;
- select threshold on the corresponding validation fold;
- freeze and apply the threshold to the test sets.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import random
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from xgboost import XGBClassifier


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

SEED = 42
FORCE_REPROCESS_EMBEDDINGS = False
#RUN_ABLATION = True
RUN_ABLATION = False
#BUILD_GLOBAL_PROBABILITY_ENSEMBLES = True
BUILD_GLOBAL_PROBABILITY_ENSEMBLES = False
RUN_HYPERPARAMETER_TUNING = False
#RUN_HYPERPARAMETER_TUNING = True
EVALUATE_TUNED_MODELS_ON_TEST = True
SKIP_EXISTING_RUNS = False
BUILD_TUNED_FINAL_EVALUATION = True

BATCH_SIZE = 16
NUM_WORKERS = 2
PCA_EXPLAINED_VARIANCE_THRESHOLD = 0.95  # Retain the minimum PCs explaining at least 95% variance per modality.

SAVE_CONFUSION_MATRICES = True
SAVE_PR_CURVES = False  # Turn on only when the number of output figures is manageable.
SAVE_FEATURE_IMPORTANCE = True
SAVE_ABLATION_PIPELINES = False  # May consume substantial storage for SVM/full embeddings.
SAVE_TUNED_PIPELINES = True

# Paths
BASE_SPLIT_CV = Path("/content/drive/MyDrive/Thesis/data/splits_predict_cv")
PREDICT_MODELS_ROOT = Path("/content/drive/MyDrive/Thesis/results/models_resnet_pro_predict")
DETECT_MODELS_ROOT = Path("/content/drive/MyDrive/Thesis/results/models_resnet_pro")
RES_ROOT = Path("/content/drive/MyDrive/Thesis/results/intermediate_fusion_predict_cv_revised_separate_pca_ev95")
EMBEDDINGS_DIR = RES_ROOT / "embedding_features"
TUNING_ROOT = RES_ROOT / "hyperparameter_tuning"

TARGET_COL = "cracked_next_session"
SAMPLE_KEY_COL = "seg_rgb_path"
RGB_PATH_COL = "seg_rgb_path"
THERMAL_PATH_COL = "seg_thermal_path"

# Set this explicitly if possible. When None, the script searches candidates below.
GROUP_ID_COL: str | None = None
GROUP_ID_CANDIDATES = [
    "Pomegranate ID",
    "Pomegranate_ID",
    "PomegranateID",
    "Fruit ID",
    "Fruit_ID",
    "fruit_id",
    "pomegranate_id",
    "ID",
]

# Optional season column for constructing a season-specific fruit identifier.
SEASON_COL: str | None = None
SEASON_COL_CANDIDATES = ["Season", "season", "Year", "year"]
CHECK_EXTERNAL_GROUP_OVERLAP = False

META_FEATURES = [
    "Treatment_Blue",
    "Treatment_Yellow",
    "Session_2",
    "Session_3",
    "Session_4",
    "Loaded branch_1",
    "Old branch_1",
    "temps_mean_new",
    "temps_std_new",
    "temp",
]

# These fields are copied to prediction files when present.
PREDICTION_METADATA_CANDIDATES = [
    SAMPLE_KEY_COL,
    THERMAL_PATH_COL,
    "Pomegranate ID",
    "Pomegranate_ID",
    "Fruit ID",
    "Fruit_ID",
    "fruit_id",
    "ID",
    "Session",
    "session",
    "Treatment",
    "treatment",
    "Tree",
    "tree",
    "Plot",
    "plot",
    "Season",
    "season",
    "Year",
    "year",
]

MODELS_TO_RUN = ["logistic_regression", "xgboost", "random_forest", "svm"]
SMOTE_OPTIONS = [False, True]
PCA_OPTIONS = [False, True]

# The complete ablation is run first. Tuning is restricted to top base runs.
TUNING_TOP_BASE_RUNS = 4
TUNING_MAX_PARAM_COMBINATIONS = 15

# Leave empty for automatic selection by mean validation F1.
# Example manual entry:
# {"config": "4_RgbPred_ThrPred_Meta", "model": "xgboost",
#  "pca": "pca_on", "smote": "smote_off"}
TUNING_BASE_RUNS: list[dict[str, str]] = []

TUNING_PARAM_GRIDS: dict[str, dict[str, Sequence[Any]]] = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
    },
    "svm": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.001, 0.01],
    },
    "random_forest": {
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [2, 5, 10],
        "max_features": ["sqrt", 0.5],
    },
    "xgboost": {
        "max_depth": [2, 3, 4],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    },
}


# =============================================================================
# 1. LOGGING, REPRODUCIBILITY, AND ENVIRONMENT
# =============================================================================


def configure_logging() -> None:
    RES_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = RES_ROOT / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Improves reproducibility. It may reduce GPU throughput slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def save_environment_info() -> None:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit_learn": package_version("scikit-learn"),
        "imbalanced_learn": package_version("imbalanced-learn"),
        "xgboost": package_version("xgboost"),
        "torchvision": package_version("torchvision"),
        "pillow": package_version("Pillow"),
        "seed": SEED,
    }
    with (RES_ROOT / "environment_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, default=str)


# =============================================================================
# 2. GENERAL VALIDATION HELPERS
# =============================================================================


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{context} is missing required columns: {missing}")


def load_metadata_csv(path: Path, required_columns: Sequence[str] | None = None) -> pd.DataFrame:
    require_file(path, "metadata CSV")
    df = pd.read_csv(path).reset_index(drop=True)

    if required_columns:
        require_columns(df, required_columns, str(path))

    return df


def resolve_column(
    df: pd.DataFrame,
    explicit_name: str | None,
    candidates: Sequence[str],
) -> str | None:
    if explicit_name is not None:
        if explicit_name not in df.columns:
            raise KeyError(
                f"Configured column '{explicit_name}' does not exist. "
                f"Available columns include: {list(df.columns)[:30]}"
            )
        return explicit_name

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


def validate_unique_key(df: pd.DataFrame, key_col: str, context: str) -> None:
    require_columns(df, [key_col], context)

    if df[key_col].isna().any():
        raise ValueError(f"{context}: key column '{key_col}' contains missing values.")

    duplicated_mask = df[key_col].duplicated(keep=False)
    if duplicated_mask.any():
        examples = df.loc[duplicated_mask, key_col].astype(str).head(10).tolist()
        raise ValueError(
            f"{context}: key column '{key_col}' is not unique. "
            f"Duplicate examples: {examples}"
        )


def merge_one_to_one(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_col: str,
    context: str,
) -> pd.DataFrame:
    validate_unique_key(left, key_col, f"{context} left table")
    validate_unique_key(right, key_col, f"{context} right table")

    before = len(left)
    merged = pd.merge(
        left,
        right,
        on=key_col,
        how="inner",
        validate="one_to_one",
    )
    after = len(merged)

    if after != before:
        logging.warning(
            "%s: inner merge retained %d/%d rows; %d rows were removed.",
            context,
            after,
            before,
            before - after,
        )

    return merged.reset_index(drop=True)


def make_group_keys(
    df: pd.DataFrame,
    group_col: str,
    season_col: str | None,
) -> set[str]:
    if season_col is not None and season_col in df.columns:
        values = (
            df[season_col].astype(str).str.strip()
            + "::"
            + df[group_col].astype(str).str.strip()
        )
    else:
        values = df[group_col].astype(str).str.strip()

    return set(values)


def validate_group_isolation(
    data_dict: Mapping[str, pd.DataFrame],
    fold_name: str,
) -> None:
    reference_df = data_dict["train"]
    group_col = resolve_column(reference_df, GROUP_ID_COL, GROUP_ID_CANDIDATES)

    if group_col is None:
        logging.warning(
            "%s: no fruit/group ID column was detected; group-overlap assertions were skipped. "
            "Set GROUP_ID_COL explicitly for a strict check.",
            fold_name,
        )
        return

    season_col = resolve_column(reference_df, SEASON_COL, SEASON_COL_CANDIDATES)
    logging.info(
        "%s: validating group isolation using group='%s'%s.",
        fold_name,
        group_col,
        f", season='{season_col}'" if season_col else "",
    )

    train_keys = make_group_keys(data_dict["train"], group_col, season_col)
    val_keys = make_group_keys(data_dict["val"], group_col, season_col)
    internal_keys = make_group_keys(data_dict["test_internal"], group_col, season_col)

    checks = [
        ("train", train_keys, "val", val_keys),
        ("train", train_keys, "test_internal", internal_keys),
        ("val", val_keys, "test_internal", internal_keys),
    ]

    if CHECK_EXTERNAL_GROUP_OVERLAP:
        external_keys = make_group_keys(data_dict["test_external"], group_col, season_col)
        checks.extend(
            [
                ("train", train_keys, "test_external", external_keys),
                ("val", val_keys, "test_external", external_keys),
                ("test_internal", internal_keys, "test_external", external_keys),
            ]
        )

    for left_name, left_keys, right_name, right_keys in checks:
        overlap = left_keys.intersection(right_keys)
        if overlap:
            raise ValueError(
                f"{fold_name}: fruit/group leakage between {left_name} and {right_name}. "
                f"Examples: {sorted(overlap)[:10]}"
            )


def validate_split_dataframe(
    df: pd.DataFrame,
    split_name: str,
    required_features: Sequence[str],
) -> None:
    require_columns(df, [SAMPLE_KEY_COL, TARGET_COL, *required_features], split_name)
    validate_unique_key(df, SAMPLE_KEY_COL, split_name)

    if df.empty:
        raise ValueError(f"{split_name} is empty after merging and filtering.")

    if df[TARGET_COL].isna().any():
        raise ValueError(f"{split_name}: target column contains missing values.")

    unique_targets = sorted(pd.Series(df[TARGET_COL]).dropna().unique().tolist())
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(
            f"{split_name}: target must be binary 0/1, found values {unique_targets}."
        )

    missing_features = df[list(required_features)].isna().sum()
    missing_features = missing_features[missing_features > 0]
    if not missing_features.empty:
        raise ValueError(
            f"{split_name}: missing feature values detected: "
            f"{missing_features.to_dict()}"
        )

    positive_count = int((df[TARGET_COL].astype(int) == 1).sum())
    negative_count = int((df[TARGET_COL].astype(int) == 0).sum())
    logging.info(
        "%s | rows=%d | negatives=%d | positives=%d | positive ratio=%.4f",
        split_name,
        len(df),
        negative_count,
        positive_count,
        positive_count / len(df),
    )


# =============================================================================
# 3. IMAGE DATASET AND TRANSFORMS
# =============================================================================


def get_transform(is_rgb: bool) -> transforms.Compose:
    target_size = (512, 768) if is_rgb else (768, 1024)

    return transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class StrictImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        transform: transforms.Compose,
    ) -> None:
        require_columns(df, [path_col], f"Dataset for {path_col}")
        self.df = df.reset_index(drop=True).copy()
        self.path_col = path_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path_str = str(self.df.loc[idx, self.path_col])

        try:
            with Image.open(path_str) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to load image at dataset index {idx}: {path_str}"
            ) from exc

        return tensor, path_str


def make_dataloader(dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


# =============================================================================
# 4. RESNET FEATURE EXTRACTION
# =============================================================================


class ResNetFeatureExtractor:
    def __init__(self, device: str = "cuda") -> None:
        selected_device = device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(selected_device)
        logging.info("Feature extraction device: %s", self.device)

    def load_model(self, model_path: Path) -> nn.Module:
        require_file(model_path, "ResNet checkpoint")
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise TypeError(f"Unexpected checkpoint format: {model_path}")

        clean_state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(2048, 1)
        model.load_state_dict(clean_state_dict, strict=True)

        extractor = nn.Sequential(*list(model.children())[:-1])
        extractor = extractor.to(self.device)
        extractor.eval()
        return extractor

    def extract_features(
        self,
        extractor: nn.Module,
        dataloader: DataLoader,
        prefix: str,
    ) -> np.ndarray:
        feature_batches: list[np.ndarray] = []

        with torch.inference_mode():
            for inputs, _ in tqdm(dataloader, leave=False, desc=f"Extract {prefix}"):
                inputs = inputs.to(self.device, non_blocking=True)
                features = extractor(inputs)
                features = features.flatten(start_dim=1).cpu().numpy()
                feature_batches.append(features)

        if not feature_batches:
            raise ValueError(f"No features were extracted for prefix '{prefix}'.")

        feature_matrix = np.vstack(feature_batches)
        if feature_matrix.shape[1] != 2048:
            raise ValueError(
                f"Expected 2048-dimensional embeddings for '{prefix}', "
                f"received {feature_matrix.shape[1]}."
            )

        if not np.isfinite(feature_matrix).all():
            raise ValueError(f"Non-finite embeddings detected for '{prefix}'.")

        return feature_matrix


def embeddings_to_dataframe(
    feature_matrix: np.ndarray,
    sample_keys: Sequence[Any],
    prefix: str,
) -> pd.DataFrame:
    if len(feature_matrix) != len(sample_keys):
        raise ValueError(
            f"Embedding/sample-key length mismatch for {prefix}: "
            f"{len(feature_matrix)} vs {len(sample_keys)}"
        )

    columns = [f"{prefix}_f{i}" for i in range(feature_matrix.shape[1])]
    frame = pd.DataFrame(feature_matrix, columns=columns)
    frame.insert(0, SAMPLE_KEY_COL, np.asarray(sample_keys))
    validate_unique_key(frame, SAMPLE_KEY_COL, f"{prefix} embeddings")
    return frame


def release_extractor(extractor: nn.Module) -> None:
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_oof_embeddings() -> pd.DataFrame:
    """Create one OOF embedding row per train/validation sample."""
    engine = ResNetFeatureExtractor()
    oof_results: dict[str, list[pd.DataFrame]] = {
        "rgb_pred": [],
        "thr_pred": [],
        "rgb_det": [],
    }

    extraction_configs = [
        (
            "rgb_pred",
            PREDICT_MODELS_ROOT / "rgb_cv_results",
            RGB_PATH_COL,
            True,
        ),
        (
            "thr_pred",
            PREDICT_MODELS_ROOT / "thermal_cv_results",
            THERMAL_PATH_COL,
            False,
        ),
        (
            "rgb_det",
            DETECT_MODELS_ROOT / "rgb_cv_results",
            RGB_PATH_COL,
            True,
        ),
    ]

    for fold_idx in range(1, 6):
        fold_name = f"fold_{fold_idx:02d}"
        logging.info("Extracting OOF validation embeddings for %s", fold_name)

        val_csv = BASE_SPLIT_CV / fold_name / "val" / "metadata.csv"
        df_val = load_metadata_csv(
            val_csv,
            required_columns=[SAMPLE_KEY_COL, RGB_PATH_COL, THERMAL_PATH_COL],
        )
        df_val = df_val.dropna(subset=[RGB_PATH_COL, THERMAL_PATH_COL]).reset_index(drop=True)
        validate_unique_key(df_val, SAMPLE_KEY_COL, f"{fold_name} validation metadata")

        for prefix, model_dir, path_col, is_rgb in extraction_configs:
            model_path = model_dir / fold_name / "best_model.pt"
            extractor = engine.load_model(model_path)

            dataset = StrictImageDataset(
                df_val,
                path_col=path_col,
                transform=get_transform(is_rgb),
            )
            dataloader = make_dataloader(dataset)
            feature_matrix = engine.extract_features(extractor, dataloader, prefix)

            feature_frame = embeddings_to_dataframe(
                feature_matrix,
                sample_keys=df_val[SAMPLE_KEY_COL].values,
                prefix=prefix,
            )
            oof_results[prefix].append(feature_frame)
            release_extractor(extractor)

    final_frames = {
        prefix: pd.concat(frames, ignore_index=True)
        for prefix, frames in oof_results.items()
    }

    for prefix, frame in final_frames.items():
        validate_unique_key(frame, SAMPLE_KEY_COL, f"complete {prefix} OOF frame")

    merged_oof = merge_one_to_one(
        final_frames["rgb_pred"],
        final_frames["thr_pred"],
        SAMPLE_KEY_COL,
        "Merge RGB-prediction and thermal-prediction OOF embeddings",
    )
    merged_oof = merge_one_to_one(
        merged_oof,
        final_frames["rgb_det"],
        SAMPLE_KEY_COL,
        "Merge detection OOF embeddings",
    )

    out_path = EMBEDDINGS_DIR / "oof_train_val_embeddings.csv"
    merged_oof.to_csv(out_path, index=False)
    logging.info("Saved OOF embeddings: %s | shape=%s", out_path, merged_oof.shape)
    return merged_oof


def extract_test_ensemble(csv_path: Path, out_name: str) -> pd.DataFrame:
    """Average each embedding dimension across the five CNN fold models."""
    engine = ResNetFeatureExtractor()
    df_test = load_metadata_csv(
        csv_path,
        required_columns=[SAMPLE_KEY_COL, RGB_PATH_COL, THERMAL_PATH_COL],
    )
    df_test = df_test.dropna(subset=[RGB_PATH_COL, THERMAL_PATH_COL]).reset_index(drop=True)
    validate_unique_key(df_test, SAMPLE_KEY_COL, f"Test metadata: {csv_path}")

    extraction_configs = [
        (
            "rgb_pred",
            PREDICT_MODELS_ROOT / "rgb_cv_results",
            RGB_PATH_COL,
            True,
        ),
        (
            "thr_pred",
            PREDICT_MODELS_ROOT / "thermal_cv_results",
            THERMAL_PATH_COL,
            False,
        ),
        (
            "rgb_det",
            DETECT_MODELS_ROOT / "rgb_cv_results",
            RGB_PATH_COL,
            True,
        ),
    ]

    final_merged = pd.DataFrame({SAMPLE_KEY_COL: df_test[SAMPLE_KEY_COL].values})

    for prefix, model_dir, path_col, is_rgb in extraction_configs:
        logging.info("Extracting test embeddings for modality '%s'", prefix)

        dataset = StrictImageDataset(
            df_test,
            path_col=path_col,
            transform=get_transform(is_rgb),
        )
        dataloader = make_dataloader(dataset)
        fold_features: list[np.ndarray] = []

        for fold_idx in range(1, 6):
            fold_name = f"fold_{fold_idx:02d}"
            model_path = model_dir / fold_name / "best_model.pt"
            extractor = engine.load_model(model_path)
            feature_matrix = engine.extract_features(extractor, dataloader, prefix)
            fold_features.append(feature_matrix)
            release_extractor(extractor)

        mean_features = np.mean(np.stack(fold_features, axis=0), axis=0)
        mean_frame = embeddings_to_dataframe(
            mean_features,
            sample_keys=df_test[SAMPLE_KEY_COL].values,
            prefix=prefix,
        )
        final_merged = merge_one_to_one(
            final_merged,
            mean_frame,
            SAMPLE_KEY_COL,
            f"Append {prefix} test embeddings",
        )

    out_path = EMBEDDINGS_DIR / out_name
    final_merged.to_csv(out_path, index=False)
    logging.info("Saved test embeddings: %s | shape=%s", out_path, final_merged.shape)
    return final_merged


# =============================================================================
# 5. DATA PREPARATION FOR EACH FUSION FOLD
# =============================================================================


def build_data_dict_for_fold(
    fold_idx: int,
    df_oof: pd.DataFrame,
    df_test_embeddings: pd.DataFrame,
    df_external_embeddings: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    fold_name = f"fold_{fold_idx:02d}"

    df_train_meta = load_metadata_csv(
        BASE_SPLIT_CV / fold_name / "train" / "metadata.csv",
        required_columns=[SAMPLE_KEY_COL, TARGET_COL, *META_FEATURES],
    )
    df_val_meta = load_metadata_csv(
        BASE_SPLIT_CV / fold_name / "val" / "metadata.csv",
        required_columns=[SAMPLE_KEY_COL, TARGET_COL, *META_FEATURES],
    )
    df_test_meta = load_metadata_csv(
        BASE_SPLIT_CV / "test" / "metadata.csv",
        required_columns=[SAMPLE_KEY_COL, TARGET_COL, *META_FEATURES],
    )
    df_external_meta = load_metadata_csv(
        BASE_SPLIT_CV / "external_test_season2" / "metadata.csv",
        required_columns=[SAMPLE_KEY_COL, TARGET_COL, *META_FEATURES],
    )

    data_dict = {
        "train": merge_one_to_one(
            df_train_meta,
            df_oof,
            SAMPLE_KEY_COL,
            f"{fold_name} train metadata + OOF embeddings",
        ),
        "val": merge_one_to_one(
            df_val_meta,
            df_oof,
            SAMPLE_KEY_COL,
            f"{fold_name} validation metadata + OOF embeddings",
        ),
        "test_internal": merge_one_to_one(
            df_test_meta,
            df_test_embeddings,
            SAMPLE_KEY_COL,
            f"{fold_name} internal test metadata + embeddings",
        ),
        "test_external": merge_one_to_one(
            df_external_meta,
            df_external_embeddings,
            SAMPLE_KEY_COL,
            f"{fold_name} external test metadata + embeddings",
        ),
    }

    # Explicitly reject missing metadata rather than silently dropping rows.
    for split_name, frame in data_dict.items():
        missing = frame[META_FEATURES].isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            raise ValueError(
                f"{fold_name}/{split_name}: missing metadata values detected: "
                f"{missing.to_dict()}"
            )

        data_dict[split_name] = frame.reset_index(drop=True)

    validate_group_isolation(data_dict, fold_name)
    return data_dict


# =============================================================================
# 6. BENCHMARKER
# =============================================================================


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    validation_f1: float


class AblationBenchmarker:
    """Benchmark tabular classifiers with modality-specific embedding preprocessing.

    PCA behavior
    ------------
    * Metadata is standardized independently.
    * Every active embedding modality is standardized independently.
    * When PCA is enabled, every active embedding modality receives its own PCA.
    * The number of PCs is selected independently for each modality as the minimum
      number required to retain PCA_EXPLAINED_VARIANCE_THRESHOLD (95%) of the
      training-set variance.
    * Every scaler/PCA object is fitted on the training partition only and then
      applied unchanged to validation and test partitions.
    """

    EMBEDDING_MODALITY_PREFIXES: dict[str, str] = {
        "rgb_pred": "rgb_pred_f",
        "thr_pred": "thr_pred_f",
        "rgb_det": "rgb_det_f",
    }

    def __init__(
        self,
        data_dict: Mapping[str, pd.DataFrame],
        active_features: Sequence[str],
        target: str,
        res_dir: Path,
        config_name: str,
        use_pca: bool,
        fold_name: str,
    ) -> None:
        self.data_dict = {key: value.copy() for key, value in data_dict.items()}
        self.features = list(active_features)
        self.target = target
        self.res_dir = Path(res_dir)
        self.config_name = config_name
        self.requested_pca = bool(use_pca)
        self.fold_name = fold_name

        self.res_dir.mkdir(parents=True, exist_ok=True)

        for split_name, frame in self.data_dict.items():
            validate_split_dataframe(frame, split_name, self.features)

        self.meta_cols = [feature for feature in self.features if feature in META_FEATURES]
        self.embedding_cols = [feature for feature in self.features if feature not in META_FEATURES]

        # Preserve the modality order as it appears in active_features.
        self.embedding_cols_by_modality: dict[str, list[str]] = {}
        unrecognized_embedding_cols: list[str] = []

        for feature in self.embedding_cols:
            matched_modality = None
            for modality, prefix in self.EMBEDDING_MODALITY_PREFIXES.items():
                if feature.startswith(prefix):
                    matched_modality = modality
                    break

            if matched_modality is None:
                unrecognized_embedding_cols.append(feature)
                continue

            self.embedding_cols_by_modality.setdefault(matched_modality, []).append(feature)

        if unrecognized_embedding_cols:
            raise ValueError(
                f"{self.fold_name}/{self.config_name}: embedding features with unknown "
                f"modality prefix: {unrecognized_embedding_cols[:20]}"
            )

        self.active_modalities = list(self.embedding_cols_by_modality.keys())
        self.actual_use_pca = self.requested_pca and bool(self.active_modalities)
        self.pca_suffix = "pca_on" if self.actual_use_pca else "pca_off"
        self._validate_pca_threshold()

        self.scaler_meta: StandardScaler | None = None
        self.scalers_embeddings: dict[str, StandardScaler] = {}
        self.pcas: dict[str, PCA] = {}

        self.pca_variance_threshold = (
            PCA_EXPLAINED_VARIANCE_THRESHOLD if self.actual_use_pca else None
        )
        self.pca_components_by_modality: dict[str, int] = {}
        self.pca_explained_variance_by_modality: dict[str, float] = {}

        self.output_feature_names: list[str] = []
        self.raw_embedding_dim = len(self.embedding_cols)
        self.raw_embedding_dim_by_modality = {
            modality: len(columns)
            for modality, columns in self.embedding_cols_by_modality.items()
        }

        self.X, self.y = self._prepare_data()

    def _validate_pca_threshold(self) -> None:
        """Validate the explained-variance target used for modality-specific PCA."""
        if not self.actual_use_pca:
            return

        if not 0.0 < PCA_EXPLAINED_VARIANCE_THRESHOLD < 1.0:
            raise ValueError(
                "PCA_EXPLAINED_VARIANCE_THRESHOLD must be strictly between 0 and 1."
            )

    def _numeric_matrix(
        self,
        frame: pd.DataFrame,
        columns: Sequence[str],
        context: str,
    ) -> np.ndarray:
        if not columns:
            return np.empty((len(frame), 0), dtype=np.float64)

        numeric_frame = frame[list(columns)].apply(pd.to_numeric, errors="raise")
        values = numeric_frame.to_numpy(dtype=np.float64, copy=True)

        if not np.isfinite(values).all():
            bad_count = int(values.size - np.isfinite(values).sum())
            raise ValueError(f"{context}: {bad_count} non-finite values detected.")

        return values

    def _prepare_data(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        y = {
            split: frame[self.target].astype(int).to_numpy()
            for split, frame in self.data_dict.items()
        }

        # ---------------------------------------------------------------------
        # 1. Metadata: independent scaling, fitted on train only.
        # ---------------------------------------------------------------------
        x_meta = {
            split: self._numeric_matrix(frame, self.meta_cols, f"{split} metadata")
            for split, frame in self.data_dict.items()
        }

        if self.meta_cols:
            self.scaler_meta = StandardScaler()
            x_meta["train"] = self.scaler_meta.fit_transform(x_meta["train"])
            for split in x_meta:
                if split != "train":
                    x_meta[split] = self.scaler_meta.transform(x_meta[split])

        # ---------------------------------------------------------------------
        # 2. Embeddings: scale + optional PCA independently per modality.
        # ---------------------------------------------------------------------
        x_embeddings_by_modality: dict[str, dict[str, np.ndarray]] = {}
        embedding_output_names: list[str] = []

        for modality in self.active_modalities:
            columns = self.embedding_cols_by_modality[modality]
            matrices = {
                split: self._numeric_matrix(
                    frame,
                    columns,
                    f"{split} {modality} embeddings",
                )
                for split, frame in self.data_dict.items()
            }

            # A separate StandardScaler is fitted for this modality.
            scaler = StandardScaler()
            matrices["train"] = scaler.fit_transform(matrices["train"])
            for split in matrices:
                if split != "train":
                    matrices[split] = scaler.transform(matrices[split])
            self.scalers_embeddings[modality] = scaler

            if self.actual_use_pca:
                if min(matrices["train"].shape) < 2:
                    raise ValueError(
                        f"{self.fold_name}/{self.config_name}/{modality}: "
                        "PCA requires at least two training observations/features."
                    )

                # For 0 < n_components < 1, scikit-learn selects the smallest
                # number of principal components whose cumulative explained
                # variance reaches the requested threshold. The 'full' solver is
                # used explicitly because variance-ratio selection is defined on
                # the complete training-set spectrum.
                pca = PCA(
                    n_components=PCA_EXPLAINED_VARIANCE_THRESHOLD,
                    svd_solver="full",
                )
                matrices["train"] = pca.fit_transform(matrices["train"])
                for split in matrices:
                    if split != "train":
                        matrices[split] = pca.transform(matrices[split])

                self.pcas[modality] = pca
                self.pca_components_by_modality[modality] = int(pca.n_components_)
                self.pca_explained_variance_by_modality[modality] = float(
                    pca.explained_variance_ratio_.sum()
                )

                embedding_output_names.extend(
                    f"{modality}_PC_{index + 1}"
                    for index in range(pca.n_components_)
                )

                logging.info(
                    "%s/%s | modality=%s | raw_dim=%d | PCA=%d | target_variance=%.3f | explained_variance=%.4f",
                    self.fold_name,
                    self.config_name,
                    modality,
                    len(columns),
                    pca.n_components_,
                    PCA_EXPLAINED_VARIANCE_THRESHOLD,
                    self.pca_explained_variance_by_modality[modality],
                )
            else:
                embedding_output_names.extend(columns)

            x_embeddings_by_modality[modality] = matrices

        # ---------------------------------------------------------------------
        # 3. Intermediate fusion: metadata + processed modality blocks.
        # ---------------------------------------------------------------------
        self.output_feature_names = [*self.meta_cols, *embedding_output_names]

        x_final: dict[str, np.ndarray] = {}
        for split in self.data_dict:
            blocks: list[np.ndarray] = []

            if x_meta[split].shape[1] > 0:
                blocks.append(x_meta[split])

            for modality in self.active_modalities:
                blocks.append(x_embeddings_by_modality[modality][split])

            if not blocks:
                raise ValueError(
                    f"{self.fold_name}/{self.config_name}: no features available after preprocessing."
                )

            x_final[split] = blocks[0] if len(blocks) == 1 else np.hstack(blocks)

            if not np.isfinite(x_final[split]).all():
                bad_count = int(
                    x_final[split].size - np.isfinite(x_final[split]).sum()
                )
                raise ValueError(
                    f"{self.fold_name}/{self.config_name}/{split}: "
                    f"{bad_count} non-finite processed features."
                )

        if x_final["train"].shape[1] != len(self.output_feature_names):
            raise RuntimeError(
                "Processed feature count does not match output feature names: "
                f"{x_final['train'].shape[1]} vs {len(self.output_feature_names)}"
            )

        return x_final, y

    @staticmethod
    def _class_counts(y_train: np.ndarray) -> tuple[int, int]:
        negative_count = int(np.sum(y_train == 0))
        positive_count = int(np.sum(y_train == 1))

        if positive_count == 0 or negative_count == 0:
            raise ValueError(
                "Training data must contain both classes. "
                f"Found negatives={negative_count}, positives={positive_count}."
            )

        return negative_count, positive_count

    def _get_model(
        self,
        model_type: str,
        y_model_train: np.ndarray,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> tuple[Any, float]:
        negative_count, positive_count = self._class_counts(y_model_train)
        scale_pos_weight = negative_count / positive_count
        overrides = dict(model_overrides or {})

        if model_type == "xgboost":
            params: dict[str, Any] = {
                "objective": "binary:logistic",
                "n_estimators": 300,
                "learning_rate": 0.03,
                "max_depth": 3,
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "logloss",
                "random_state": SEED,
                "n_jobs": -1,
            }
            params.update(overrides)
            return XGBClassifier(**params), float(params["scale_pos_weight"])

        if model_type == "random_forest":
            params = {
                "n_estimators": 300,
                "class_weight": "balanced",
                "random_state": SEED,
                "n_jobs": -1,
            }
            params.update(overrides)
            return RandomForestClassifier(**params), float(scale_pos_weight)

        if model_type == "logistic_regression":
            params = {
                "penalty": "l2",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 2000,
                "solver": "lbfgs",
                "random_state": SEED,
            }
            params.update(overrides)
            return LogisticRegression(**params), float(scale_pos_weight)

        if model_type == "svm":
            params = {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True,
                "class_weight": "balanced",
                "random_state": SEED,
            }
            params.update(overrides)
            return SVC(**params), float(scale_pos_weight)

        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def _select_validation_threshold(
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> ThresholdResult:
        if len(np.unique(y_true)) < 2:
            logging.warning(
                "Validation split contains one class; threshold defaults to 0.5."
            )
            predictions = (probabilities >= 0.5).astype(int)
            return ThresholdResult(
                threshold=0.5,
                validation_f1=float(f1_score(y_true, predictions, zero_division=0)),
            )

        precisions, recalls, thresholds = precision_recall_curve(
            y_true,
            probabilities,
        )

        if thresholds.size == 0:
            predictions = (probabilities >= 0.5).astype(int)
            return ThresholdResult(
                threshold=0.5,
                validation_f1=float(f1_score(y_true, predictions, zero_division=0)),
            )

        f1_values = np.divide(
            2.0 * precisions[:-1] * recalls[:-1],
            precisions[:-1] + recalls[:-1],
            out=np.zeros_like(thresholds, dtype=float),
            where=(precisions[:-1] + recalls[:-1]) != 0,
        )
        best_index = int(np.argmax(f1_values))
        return ThresholdResult(
            threshold=float(thresholds[best_index]),
            validation_f1=float(f1_values[best_index]),
        )

    @staticmethod
    def _safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, probabilities))

    @staticmethod
    def _safe_pr_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, probabilities))

    def _prediction_metadata(self, split: str) -> pd.DataFrame:
        source = self.data_dict[split]
        columns = [
            column
            for column in PREDICTION_METADATA_CANDIDATES
            if column in source.columns
        ]

        if SAMPLE_KEY_COL not in columns:
            columns.insert(0, SAMPLE_KEY_COL)

        # Preserve order while removing duplicates.
        columns = list(dict.fromkeys(columns))
        return source[columns].reset_index(drop=True).copy()

    def _save_confusion_matrix(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        title: str,
        path: Path,
    ) -> None:
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(5, 4))
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Healthy", "Cracked"],
        )
        display.plot(ax=axis, values_format="d")
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)

    def _save_pr_curve(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        title: str,
        path: Path,
    ) -> None:
        figure, axis = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(
            y_true,
            probabilities,
            ax=axis,
            name=title,
        )
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)

    def _save_feature_importance(self, model: Any, run_id: str) -> None:
        importances: np.ndarray | None = None

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coefficients = np.asarray(model.coef_)
            if coefficients.ndim == 2 and coefficients.shape[0] == 1:
                importances = np.abs(coefficients[0].astype(float))

        if importances is None:
            return

        if len(importances) != len(self.output_feature_names):
            logging.warning(
                "%s: feature importance length mismatch (%d vs %d); skipped.",
                run_id,
                len(importances),
                len(self.output_feature_names),
            )
            return

        importance_frame = pd.DataFrame(
            {
                "feature": self.output_feature_names,
                "importance": importances,
                "feature_space": (
                    "metadata_plus_modality_specific_pca_components"
                    if self.actual_use_pca
                    else "metadata_plus_raw_embeddings"
                ),
            }
        ).sort_values("importance", ascending=False)

        importance_frame.head(100).to_csv(
            self.res_dir / f"feature_importance_{run_id}.csv",
            index=False,
        )

    def _save_pipeline(
        self,
        model: Any,
        threshold_result: ThresholdResult,
        run_id: str,
        model_type: str,
        use_smote: bool,
        model_overrides: Mapping[str, Any] | None,
    ) -> None:
        artifact = {
            "model": model,
            "scaler_meta": self.scaler_meta,
            "scalers_embeddings_by_modality": self.scalers_embeddings,
            "pcas_by_modality": self.pcas,
            "pca_strategy": "modality_specific_explained_variance",
            "pca_explained_variance_threshold": self.pca_variance_threshold,
            "pca_components_by_modality": self.pca_components_by_modality,
            "pca_explained_variance_by_modality": self.pca_explained_variance_by_modality,
            "threshold": threshold_result.threshold,
            "config_name": self.config_name,
            "fold_name": self.fold_name,
            "model_type": model_type,
            "smote_used_for_training": bool(use_smote),
            "requested_pca": self.requested_pca,
            "actual_use_pca": self.actual_use_pca,
            "meta_columns": self.meta_cols,
            "embedding_columns": self.embedding_cols,
            "embedding_columns_by_modality": self.embedding_cols_by_modality,
            "active_modalities": self.active_modalities,
            "output_feature_names": self.output_feature_names,
            "target_column": self.target,
            "model_overrides": dict(model_overrides or {}),
            "seed": SEED,
        }
        joblib.dump(artifact, self.res_dir / f"pipeline_{run_id}.joblib")

    def run(
        self,
        model_type: str,
        use_smote: bool,
        model_overrides: Mapping[str, Any] | None = None,
        evaluate_splits: Sequence[str] = (
            "train",
            "val",
            "test_internal",
            "test_external",
        ),
        save_outputs: bool = True,
        save_pipeline: bool = False,
        run_tag: str | None = None,
    ) -> pd.DataFrame:
        for split in evaluate_splits:
            if split not in self.X:
                raise KeyError(f"Unknown evaluation split: {split}")

        x_train_resampled = self.X["train"]
        y_train_resampled = self.y["train"]

        negative_before, positive_before = self._class_counts(self.y["train"])
        smote_applied = False
        smote_suffix = "smote_on" if use_smote else "smote_off"

        if use_smote:
            minority_count = min(negative_before, positive_before)
            if minority_count < 2:
                raise ValueError(
                    f"{self.fold_name}/{self.config_name}: SMOTE requires at least "
                    f"two samples in the minority class, found {minority_count}."
                )

            smote = SMOTE(
                random_state=SEED,
                k_neighbors=min(5, minority_count - 1),
            )
            x_train_resampled, y_train_resampled = smote.fit_resample(
                self.X["train"],
                self.y["train"],
            )
            smote_applied = True

        negative_after, positive_after = self._class_counts(y_train_resampled)
        model, effective_scale_pos_weight = self._get_model(
            model_type,
            y_train_resampled,
            model_overrides=model_overrides,
        )

        parameter_hash = ""
        if model_overrides:
            serialized = json.dumps(model_overrides, sort_keys=True, default=str)
            parameter_hash = "_hp" + hashlib.md5(serialized.encode("utf-8")).hexdigest()[:8]

        run_id = (
            f"{self.config_name}_{model_type}_{self.pca_suffix}_{smote_suffix}"
            f"{parameter_hash}"
        )
        if run_tag:
            run_id = f"{run_tag}_{run_id}"

        summary_path = self.res_dir / f"summary_{run_id}.csv"
        if SKIP_EXISTING_RUNS and save_outputs and summary_path.exists():
            logging.info("Skipping existing run: %s", run_id)
            return pd.read_csv(summary_path)

        logging.info(
            "Training %s | fold=%s | config=%s | model=%s | PCA=%s | SMOTE=%s",
            run_id,
            self.fold_name,
            self.config_name,
            model_type,
            self.pca_suffix,
            smote_suffix,
        )
        model.fit(x_train_resampled, y_train_resampled)

        validation_probabilities = model.predict_proba(self.X["val"])[:, 1]
        threshold_result = self._select_validation_threshold(
            self.y["val"],
            validation_probabilities,
        )

        if save_outputs and SAVE_FEATURE_IMPORTANCE:
            self._save_feature_importance(model, run_id)

        model_params_json = json.dumps(model.get_params(), sort_keys=True, default=str)
        summary_rows: list[dict[str, Any]] = []

        for split in evaluate_splits:
            probabilities = model.predict_proba(self.X[split])[:, 1]
            predictions = (probabilities >= threshold_result.threshold).astype(int)
            y_true = self.y[split]

            matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
            tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
            positive_support = int(np.sum(y_true == 1))
            negative_support = int(np.sum(y_true == 0))

            if save_outputs:
                prediction_frame = self._prediction_metadata(split)
                prediction_frame["actual"] = y_true
                prediction_frame["pred"] = predictions
                prediction_frame["prob"] = probabilities
                prediction_frame["threshold"] = threshold_result.threshold
                prediction_frame["run_id"] = run_id
                prediction_frame["fold"] = self.fold_name
                prediction_frame.to_csv(
                    self.res_dir / f"preds_{run_id}_{split}.csv",
                    index=False,
                )

                if SAVE_CONFUSION_MATRICES:
                    self._save_confusion_matrix(
                        y_true,
                        predictions,
                        title=f"{run_id}\n{split}",
                        path=self.res_dir / f"cm_{run_id}_{split}.png",
                    )

                if SAVE_PR_CURVES and len(np.unique(y_true)) == 2:
                    self._save_pr_curve(
                        y_true,
                        probabilities,
                        title=f"{run_id} – {split}",
                        path=self.res_dir / f"pr_curve_{run_id}_{split}.png",
                    )

            summary_rows.append(
                {
                    "run_id": run_id,
                    "fold": self.fold_name,
                    "config": self.config_name,
                    "model": model_type,
                    "pca": self.pca_suffix,
                    "smote": smote_suffix,
                    "smote_applied": smote_applied,
                    "split": split,
                    "threshold": threshold_result.threshold,
                    "validation_f1_at_optimized_threshold": threshold_result.validation_f1,
                    "accuracy": float(accuracy_score(y_true, predictions)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_true, predictions)
                    ),
                    "f1_class_1": float(
                        f1_score(y_true, predictions, zero_division=0)
                    ),
                    "precision_class_1": float(
                        precision_score(y_true, predictions, zero_division=0)
                    ),
                    "recall_class_1": float(
                        recall_score(y_true, predictions, zero_division=0)
                    ),
                    "roc_auc": self._safe_roc_auc(y_true, probabilities),
                    "pr_auc": self._safe_pr_auc(y_true, probabilities),
                    "positive_prevalence": positive_support / len(y_true),
                    "negative_support": negative_support,
                    "positive_support": positive_support,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "raw_train_size": len(self.y["train"]),
                    "raw_train_negative": negative_before,
                    "raw_train_positive": positive_before,
                    "resampled_train_size": len(y_train_resampled),
                    "resampled_train_negative": negative_after,
                    "resampled_train_positive": positive_after,
                    "effective_scale_pos_weight": effective_scale_pos_weight,
                    "raw_embedding_dimensions": self.raw_embedding_dim,
                    "raw_embedding_dimensions_by_modality": json.dumps(
                        self.raw_embedding_dim_by_modality, sort_keys=True
                    ),
                    "processed_feature_dimensions": self.X["train"].shape[1],
                    "pca_strategy": (
                        "modality_specific_explained_variance"
                        if self.actual_use_pca
                        else "off"
                    ),
                    "pca_explained_variance_threshold": (
                        PCA_EXPLAINED_VARIANCE_THRESHOLD
                        if self.actual_use_pca
                        else np.nan
                    ),
                    "pca_components": (
                        int(sum(self.pca_components_by_modality.values()))
                        if self.actual_use_pca
                        else np.nan
                    ),
                    "pca_components_by_modality": json.dumps(
                        self.pca_components_by_modality, sort_keys=True
                    ),
                    "pca_explained_variance_by_modality": json.dumps(
                        self.pca_explained_variance_by_modality, sort_keys=True
                    ),
                    "pca_components_rgb_pred": self.pca_components_by_modality.get(
                        "rgb_pred", np.nan
                    ),
                    "pca_components_thr_pred": self.pca_components_by_modality.get(
                        "thr_pred", np.nan
                    ),
                    "pca_components_rgb_det": self.pca_components_by_modality.get(
                        "rgb_det", np.nan
                    ),
                    "pca_explained_variance_rgb_pred": (
                        self.pca_explained_variance_by_modality.get("rgb_pred", np.nan)
                    ),
                    "pca_explained_variance_thr_pred": (
                        self.pca_explained_variance_by_modality.get("thr_pred", np.nan)
                    ),
                    "pca_explained_variance_rgb_det": (
                        self.pca_explained_variance_by_modality.get("rgb_det", np.nan)
                    ),
                    "model_overrides": json.dumps(
                        dict(model_overrides or {}), sort_keys=True, default=str
                    ),
                    "model_parameters": model_params_json,
                }
            )

        summary_frame = pd.DataFrame(summary_rows)
        if save_outputs:
            summary_frame.to_csv(summary_path, index=False)

        if save_pipeline:
            self._save_pipeline(
                model,
                threshold_result,
                run_id,
                model_type,
                use_smote,
                model_overrides,
            )

        return summary_frame


# =============================================================================
# 7. CONFIGURATIONS AND ABLATION EXECUTION
# =============================================================================


def build_feature_configurations(df_oof: pd.DataFrame) -> dict[str, list[str]]:
    rgb_pred_cols = [column for column in df_oof.columns if column.startswith("rgb_pred_f")]
    thr_pred_cols = [column for column in df_oof.columns if column.startswith("thr_pred_f")]
    rgb_det_cols = [column for column in df_oof.columns if column.startswith("rgb_det_f")]

    expected_dimensions = {
        "rgb_pred": len(rgb_pred_cols),
        "thr_pred": len(thr_pred_cols),
        "rgb_det": len(rgb_det_cols),
    }
    for prefix, dimension in expected_dimensions.items():
        if dimension != 2048:
            raise ValueError(
                f"Expected 2048 {prefix} features, found {dimension}."
            )

    return {
        "1_MetaOnly": list(META_FEATURES),
        "2_ThrPred_Meta": [*META_FEATURES, *thr_pred_cols],
        "3_RgbPred_Meta": [*META_FEATURES, *rgb_pred_cols],
        "4_RgbPred_ThrPred_Meta": [
            *META_FEATURES,
            *rgb_pred_cols,
            *thr_pred_cols,
        ],
        "5_RgbDet_ThrPred_Meta": [
            *META_FEATURES,
            *rgb_det_cols,
            *thr_pred_cols,
        ],
    }


def run_ablation_experiments(
    df_oof: pd.DataFrame,
    df_test_embeddings: pd.DataFrame,
    df_external_embeddings: pd.DataFrame,
    feature_configs: Mapping[str, Sequence[str]],
) -> None:
    for fold_idx in range(1, 6):
        fold_name = f"fold_{fold_idx:02d}"
        logging.info("=" * 80)
        logging.info("RUNNING FUSION MODELS FOR %s", fold_name)
        logging.info("=" * 80)

        fold_res_dir = RES_ROOT / fold_name
        fold_res_dir.mkdir(parents=True, exist_ok=True)

        data_dict = build_data_dict_for_fold(
            fold_idx,
            df_oof,
            df_test_embeddings,
            df_external_embeddings,
        )

        for config_name, active_features in feature_configs.items():
            # PCA is not applicable to the metadata-only configuration.
            pca_options_for_config = (
                [False] if config_name == "1_MetaOnly" else PCA_OPTIONS
            )

            for use_pca in pca_options_for_config:
                benchmarker = AblationBenchmarker(
                    data_dict=data_dict,
                    active_features=active_features,
                    target=TARGET_COL,
                    res_dir=fold_res_dir,
                    config_name=config_name,
                    use_pca=use_pca,
                    fold_name=fold_name,
                )

                for model_type in MODELS_TO_RUN:
                    for use_smote in SMOTE_OPTIONS:
                        benchmarker.run(
                            model_type=model_type,
                            use_smote=use_smote,
                            save_pipeline=SAVE_ABLATION_PIPELINES,
                        )


# =============================================================================
# 8. RESULT AGGREGATION
# =============================================================================


AGGREGATION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_class_1",
    "precision_class_1",
    "recall_class_1",
    "roc_auc",
    "pr_auc",
    "threshold",
    "validation_f1_at_optimized_threshold",
    "tn",
    "fp",
    "fn",
    "tp",
    "pca_explained_variance_rgb_pred",
    "pca_explained_variance_thr_pred",
    "pca_explained_variance_rgb_det",
]


def flatten_multiindex_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def aggregate_ablation_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Aggregating fold-specific ablation results.")
    all_summaries: list[pd.DataFrame] = []

    for fold_idx in range(1, 6):
        fold_dir = RES_ROOT / f"fold_{fold_idx:02d}"
        for csv_file in sorted(fold_dir.glob("summary_*.csv")):
            frame = pd.read_csv(csv_file)
            all_summaries.append(frame)

    if not all_summaries:
        raise FileNotFoundError(
            "No fold summary files were found. Run the ablation stage first."
        )

    all_results = pd.concat(all_summaries, ignore_index=True)
    all_results.to_csv(RES_ROOT / "MASTER_all_folds_results.csv", index=False)

    group_columns = ["config", "model", "pca", "smote", "split"]
    available_metrics = [
        metric for metric in AGGREGATION_METRICS if metric in all_results.columns
    ]

    aggregation_spec = {
        metric: ["mean", "std"]
        for metric in available_metrics
    }
    average_results = (
        all_results.groupby(group_columns, as_index=False)
        .agg(aggregation_spec)
        .pipe(flatten_multiindex_columns)
    )

    sort_columns = [
    column
    for column in ["split", "f1_class_1_mean", "pr_auc_mean"]
    if column in average_results.columns
    ]
    # sort_columns = [
    #     column
    #     for column in ["split", "pr_auc_mean", "f1_class_1_mean"]
    #     if column in average_results.columns
    # ]
    ascending = [True, False, False][: len(sort_columns)]
    average_results = average_results.sort_values(
        sort_columns,
        ascending=ascending,
    )
    average_results.to_csv(
        RES_ROOT / "MASTER_average_and_std_results.csv",
        index=False,
    )

    for split_name in sorted(all_results["split"].unique()):
        split_table = average_results[average_results["split"] == split_name].copy()
        split_table.to_csv(
            RES_ROOT / f"RANKED_{split_name}_results.csv",
            index=False,
        )

    return all_results, average_results


# =============================================================================
# 9. OPTIONAL PROBABILITY-LEVEL ENSEMBLE ACROSS FOLD PIPELINES
# =============================================================================


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "f1_class_1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision_class_1": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall_class_1": float(recall_score(y_true, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "pr_auc": (
            float(average_precision_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "negative_support": int(np.sum(y_true == 0)),
        "positive_support": int(np.sum(y_true == 1)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def average_test_prediction_files(
    frames: Sequence[pd.DataFrame],
    split_name: str,
) -> pd.DataFrame:
    if len(frames) != 5:
        raise ValueError(
            f"{split_name}: expected five fold prediction files, found {len(frames)}."
        )

    for index, frame in enumerate(frames, start=1):
        require_columns(
            frame,
            [SAMPLE_KEY_COL, "actual", "prob"],
            f"{split_name} fold {index} predictions",
        )
        validate_unique_key(
            frame,
            SAMPLE_KEY_COL,
            f"{split_name} fold {index} predictions",
        )

    base = frames[0].drop(columns=["prob", "pred", "threshold", "fold"], errors="ignore")
    base = base.copy()

    probability_columns: list[str] = []
    merged = base
    for fold_index, frame in enumerate(frames, start=1):
        probability_column = f"prob_fold_{fold_index:02d}"
        probability_columns.append(probability_column)
        current = frame[[SAMPLE_KEY_COL, "actual", "prob"]].rename(
            columns={"actual": f"actual_fold_{fold_index:02d}", "prob": probability_column}
        )
        merged = pd.merge(
            merged,
            current,
            on=SAMPLE_KEY_COL,
            how="inner",
            validate="one_to_one",
        )

    actual_columns = [column for column in merged.columns if column.startswith("actual_fold_")]
    actual_values = merged[actual_columns].to_numpy()
    if not np.all(actual_values == actual_values[:, [0]]):
        raise ValueError(f"{split_name}: actual labels differ across fold files.")

    merged["actual"] = actual_values[:, 0].astype(int)
    merged["prob"] = merged[probability_columns].mean(axis=1)
    merged = merged.drop(columns=actual_columns)
    return merged


def build_global_probability_ensembles(all_results: pd.DataFrame) -> pd.DataFrame:
    ensemble_root = RES_ROOT / "global_probability_ensembles"
    ensemble_root.mkdir(parents=True, exist_ok=True)

    base_runs = (
        all_results[["config", "model", "pca", "smote"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    ensemble_summaries: list[dict[str, Any]] = []

    for _, base_run in tqdm(
        base_runs.iterrows(),
        total=len(base_runs),
        desc="Build probability ensembles",
    ):
        base_run_id = (
            f"{base_run['config']}_{base_run['model']}_"
            f"{base_run['pca']}_{base_run['smote']}"
        )

        val_frames: list[pd.DataFrame] = []
        internal_frames: list[pd.DataFrame] = []
        external_frames: list[pd.DataFrame] = []

        for fold_idx in range(1, 6):
            fold_dir = RES_ROOT / f"fold_{fold_idx:02d}"
            val_path = fold_dir / f"preds_{base_run_id}_val.csv"
            internal_path = fold_dir / f"preds_{base_run_id}_test_internal.csv"
            external_path = fold_dir / f"preds_{base_run_id}_test_external.csv"

            require_file(val_path, "validation prediction file")
            require_file(internal_path, "internal test prediction file")
            require_file(external_path, "external test prediction file")

            val_frames.append(pd.read_csv(val_path))
            internal_frames.append(pd.read_csv(internal_path))
            external_frames.append(pd.read_csv(external_path))

        pooled_validation = pd.concat(val_frames, ignore_index=True)
        validate_unique_key(
            pooled_validation,
            SAMPLE_KEY_COL,
            f"{base_run_id} pooled validation predictions",
        )
        pooled_threshold = AblationBenchmarker._select_validation_threshold(
            pooled_validation["actual"].astype(int).to_numpy(),
            pooled_validation["prob"].to_numpy(dtype=float),
        )
        pooled_validation["pred"] = (
            pooled_validation["prob"] >= pooled_threshold.threshold
        ).astype(int)
        pooled_validation["global_threshold"] = pooled_threshold.threshold
        pooled_validation.to_csv(
            ensemble_root / f"pooled_validation_{base_run_id}.csv",
            index=False,
        )

        validation_metrics = evaluate_probabilities(
            pooled_validation["actual"].astype(int).to_numpy(),
            pooled_validation["prob"].to_numpy(dtype=float),
            pooled_threshold.threshold,
        )
        ensemble_summaries.append(
            {
                **base_run.to_dict(),
                "run_id": base_run_id,
                "split": "pooled_validation",
                **validation_metrics,
            }
        )

        for split_name, fold_frames in [
            ("test_internal_ensemble", internal_frames),
            ("test_external_ensemble", external_frames),
        ]:
            ensemble_frame = average_test_prediction_files(fold_frames, split_name)
            ensemble_frame["pred"] = (
                ensemble_frame["prob"] >= pooled_threshold.threshold
            ).astype(int)
            ensemble_frame["global_threshold"] = pooled_threshold.threshold
            ensemble_frame.to_csv(
                ensemble_root / f"preds_{base_run_id}_{split_name}.csv",
                index=False,
            )

            metrics = evaluate_probabilities(
                ensemble_frame["actual"].astype(int).to_numpy(),
                ensemble_frame["prob"].to_numpy(dtype=float),
                pooled_threshold.threshold,
            )
            ensemble_summaries.append(
                {
                    **base_run.to_dict(),
                    "run_id": base_run_id,
                    "split": split_name,
                    **metrics,
                }
            )

    ensemble_summary_frame = pd.DataFrame(ensemble_summaries)
    ensemble_summary_frame.to_csv(
        ensemble_root / "MASTER_global_probability_ensemble_results.csv",
        index=False,
    )
    return ensemble_summary_frame


# =============================================================================
# 10. OPTIONAL CONSTRAINED HYPERPARAMETER TUNING
# =============================================================================


def parameter_candidates(model_type: str) -> list[dict[str, Any]]:
    grid = TUNING_PARAM_GRIDS[model_type]
    all_candidates = list(ParameterGrid(grid))

    if len(all_candidates) <= TUNING_MAX_PARAM_COMBINATIONS:
        return all_candidates

    sampled = ParameterSampler(
        grid,
        n_iter=TUNING_MAX_PARAM_COMBINATIONS,
        random_state=SEED,
    )
    return [dict(candidate) for candidate in sampled]


def select_tuning_base_runs(
    average_results: pd.DataFrame
) -> list[dict[str, str]]:

    # If configurations were manually specified, use them.
    if TUNING_BASE_RUNS:
        return [dict(item) for item in TUNING_BASE_RUNS]

    # Use validation results only for model/configuration selection.
    validation_rows = average_results[
        average_results["split"] == "val"
    ].copy()

    if "pr_auc_mean" not in validation_rows.columns:
        raise KeyError(
            "The average results do not contain 'pr_auc_mean'."
        )

    if "f1_class_1_mean" not in validation_rows.columns:
        raise KeyError(
            "The average results do not contain 'f1_class_1_mean'."
        )

    # ---------------------------------------------------------
    # Select the best BASE configuration separately
    # for each classifier family.
    #
    # Primary criterion: mean validation F1
    # Secondary criterion: mean validation PR-AUC
    # ---------------------------------------------------------
    selected = (
        validation_rows
        .sort_values(
            ["model", "f1_class_1_mean", "pr_auc_mean"],
            ascending=[True, False, False],
        )
        .groupby("model", as_index=False)
        .head(1)
        [["config", "model", "pca", "smote"]]
        .reset_index(drop=True)
    )

    # Verify that every classifier receives a tuning configuration.
    expected_models = set(MODELS_TO_RUN)
    selected_models = set(selected["model"])

    missing_models = expected_models - selected_models

    if missing_models:
        raise ValueError(
            "No validation configuration was found for: "
            f"{sorted(missing_models)}"
        )

    print("\nSelected base configuration for tuning:")
    print(selected.to_string(index=False))

    return [
        {key: str(value) for key, value in row.items()}
        for row in selected.to_dict("records")
    ]

def run_constrained_tuning(
    df_oof: pd.DataFrame,
    df_test_embeddings: pd.DataFrame,
    df_external_embeddings: pd.DataFrame,
    feature_configs: Mapping[str, Sequence[str]],
    average_results: pd.DataFrame,
) -> pd.DataFrame:
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    base_runs = select_tuning_base_runs(average_results)
    logging.info("Tuning base runs: %s", base_runs)

    tuning_rows: list[dict[str, Any]] = []

    # Cache merged data once per outer fold.
    fold_data_cache = {
        fold_idx: build_data_dict_for_fold(
            fold_idx,
            df_oof,
            df_test_embeddings,
            df_external_embeddings,
        )
        for fold_idx in range(1, 6)
    }

    for base_run in base_runs:
        config_name = base_run["config"]
        model_type = base_run["model"]
        use_pca = base_run["pca"] == "pca_on"
        use_smote = base_run["smote"] == "smote_on"

        if config_name not in feature_configs:
            raise KeyError(f"Unknown tuning configuration: {config_name}")

        candidates = parameter_candidates(model_type)
        logging.info(
            "Tuning %s/%s over %d candidate parameter sets.",
            config_name,
            model_type,
            len(candidates),
        )

        for candidate_index, candidate in enumerate(candidates, start=1):
            fold_pr_auc_values: list[float] = []
            fold_f1_values: list[float] = []

            for fold_idx in range(1, 6):
                fold_name = f"fold_{fold_idx:02d}"
                benchmarker = AblationBenchmarker(
                    data_dict=fold_data_cache[fold_idx],
                    active_features=feature_configs[config_name],
                    target=TARGET_COL,
                    res_dir=TUNING_ROOT / "temporary_validation_runs" / fold_name,
                    config_name=config_name,
                    use_pca=use_pca,
                    fold_name=fold_name,
                )
                summary = benchmarker.run(
                    model_type=model_type,
                    use_smote=use_smote,
                    model_overrides=candidate,
                    evaluate_splits=("val",),
                    save_outputs=False,
                    save_pipeline=False,
                    run_tag="tuning",
                )
                fold_pr_auc_values.append(float(summary.iloc[0]["pr_auc"]))
                fold_f1_values.append(float(summary.iloc[0]["f1_class_1"]))

            tuning_rows.append(
                {
                    **base_run,
                    "candidate_index": candidate_index,
                    "parameters": json.dumps(candidate, sort_keys=True, default=str),
                    "validation_pr_auc_mean": float(np.nanmean(fold_pr_auc_values)),
                    "validation_pr_auc_std": float(np.nanstd(fold_pr_auc_values, ddof=1)),
                    "validation_f1_mean": float(np.nanmean(fold_f1_values)),
                    "validation_f1_std": float(np.nanstd(fold_f1_values, ddof=1)),
                }
            )

    tuning_frame = pd.DataFrame(tuning_rows).sort_values(
    ["validation_f1_mean", "validation_pr_auc_mean"],
    ascending=[False, False],
    )
    tuning_frame.to_csv(TUNING_ROOT / "TUNING_all_candidates.csv", index=False)

    best_candidates = (
    tuning_frame.sort_values(
        ["validation_f1_mean", "validation_pr_auc_mean"],
        ascending=[False, False],
    )
    .groupby(["config", "model", "pca", "smote"], as_index=False)
    .head(1)
    .reset_index(drop=True)
    )
    best_candidates.to_csv(TUNING_ROOT / "TUNING_best_candidates.csv", index=False)

    if EVALUATE_TUNED_MODELS_ON_TEST:
        evaluate_best_tuned_candidates(
            best_candidates,
            fold_data_cache,
            feature_configs,
        )

    return tuning_frame


def evaluate_best_tuned_candidates(
    best_candidates: pd.DataFrame,
    fold_data_cache: Mapping[int, Mapping[str, pd.DataFrame]],
    feature_configs: Mapping[str, Sequence[str]],
) -> None:
    output_root = TUNING_ROOT / "best_tuned_evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []

    for _, row in best_candidates.iterrows():
        config_name = str(row["config"])
        model_type = str(row["model"])
        use_pca = str(row["pca"]) == "pca_on"
        use_smote = str(row["smote"]) == "smote_on"
        model_overrides = json.loads(str(row["parameters"]))

        for fold_idx in range(1, 6):
            fold_name = f"fold_{fold_idx:02d}"
            fold_output_dir = output_root / fold_name
            benchmarker = AblationBenchmarker(
                data_dict=fold_data_cache[fold_idx],
                active_features=feature_configs[config_name],
                target=TARGET_COL,
                res_dir=fold_output_dir,
                config_name=config_name,
                use_pca=use_pca,
                fold_name=fold_name,
            )
            summary = benchmarker.run(
                model_type=model_type,
                use_smote=use_smote,
                model_overrides=model_overrides,
                save_outputs=True,
                save_pipeline=SAVE_TUNED_PIPELINES,
                run_tag="tuned",
            )
            summaries.append(summary)

    tuned_results = pd.concat(summaries, ignore_index=True)
    tuned_results.to_csv(
        output_root / "TUNED_all_fold_results.csv",
        index=False,
    )

    group_columns = ["config", "model", "pca", "smote", "split", "model_overrides"]
    available_metrics = [
        metric for metric in AGGREGATION_METRICS if metric in tuned_results.columns
    ]
    tuned_average = (
        tuned_results.groupby(group_columns, as_index=False)
        .agg({metric: ["mean", "std"] for metric in available_metrics})
        .pipe(flatten_multiindex_columns)
    )
    tuned_average.to_csv(
        output_root / "TUNED_average_and_std_results.csv",
        index=False,
    )

def build_tuned_final_evaluation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build final evaluation outputs for the tuned classifiers.

    For each tuned classifier:
    1. Select the best fold using validation F1
       (PR-AUC is used only as a secondary criterion).
    2. Report the corresponding internal/external test performance.
    3. Build a five-fold probability ensemble.
    4. Select one global threshold using pooled OOF validation predictions only.
    5. Apply the fixed global threshold to the averaged internal/external
       test probabilities.
    """

    tuned_root = TUNING_ROOT / "best_tuned_evaluation"
    tuned_results_path = tuned_root / "TUNED_all_fold_results.csv"

    require_file(
        tuned_results_path,
        "tuned all-fold results"
    )

    tuned_results = pd.read_csv(tuned_results_path)

    require_columns(
        tuned_results,
        [
            "run_id",
            "fold",
            "config",
            "model",
            "pca",
            "smote",
            "split",
            "f1_class_1",
            "pr_auc",
        ],
        "Tuned all-fold results",
    )

    ensemble_root = tuned_root / "tuned_probability_ensembles"
    ensemble_root.mkdir(parents=True, exist_ok=True)

    run_columns = [
        "run_id",
        "config",
        "model",
        "pca",
        "smote",
        "model_overrides",
    ]
    run_columns = [
        column for column in run_columns
        if column in tuned_results.columns
    ]

    tuned_runs = (
        tuned_results[run_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    best_fold_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    metric_columns = [
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "f1_class_1",
        "precision_class_1",
        "recall_class_1",
        "roc_auc",
        "pr_auc",
        "negative_support",
        "positive_support",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    for _, run in tuned_runs.iterrows():

        run_id = str(run["run_id"])

        run_results = tuned_results[
            tuned_results["run_id"].astype(str) == run_id
        ].copy()

        # ============================================================
        # 1. SELECT BEST FOLD USING VALIDATION ONLY
        # ============================================================

        validation_rows = run_results[
            run_results["split"] == "val"
        ].copy()

        if len(validation_rows) != 5:
            raise ValueError(
                f"{run_id}: expected 5 validation fold rows, "
                f"found {len(validation_rows)}."
            )

        # Primary criterion = validation F1
        # Secondary criterion = validation PR-AUC
        best_validation_row = (
            validation_rows
            .sort_values(
                ["f1_class_1", "pr_auc"],
                ascending=[False, False],
            )
            .iloc[0]
        )

        best_fold = str(best_validation_row["fold"])

        logging.info(
            "Best tuned fold | model=%s | fold=%s | val_F1=%.4f | val_PR_AUC=%.4f",
            run["model"],
            best_fold,
            float(best_validation_row["f1_class_1"]),
            float(best_validation_row["pr_auc"]),
        )

        # Save validation/internal/external results from the selected fold
        for split_name in [
            "val",
            "test_internal",
            "test_external",
        ]:

            selected_row = run_results[
                (run_results["fold"].astype(str) == best_fold)
                & (run_results["split"] == split_name)
            ]

            if len(selected_row) != 1:
                raise ValueError(
                    f"{run_id}/{best_fold}/{split_name}: "
                    f"expected one result row, found {len(selected_row)}."
                )

            result_dict = selected_row.iloc[0].to_dict()
            result_dict["evaluation_method"] = "best_fold"
            result_dict["selected_best_fold"] = best_fold
            result_dict["selection_validation_f1"] = float(
                best_validation_row["f1_class_1"]
            )
            result_dict["selection_validation_pr_auc"] = float(
                best_validation_row["pr_auc"]
            )

            best_fold_rows.append(result_dict)

            if split_name in ["test_internal", "test_external"]:
                comparison_row = {
                    "config": run["config"],
                    "model": run["model"],
                    "pca": run["pca"],
                    "smote": run["smote"],
                    "evaluation_method": "best_fold",
                    "split": split_name,
                    "fold": best_fold,
                }

                for metric in metric_columns:
                    if metric in selected_row.columns:
                        comparison_row[metric] = selected_row.iloc[0][metric]

                comparison_rows.append(comparison_row)

        # ============================================================
        # 2. LOAD FIVE TUNED FOLD PREDICTION FILES
        # ============================================================

        val_frames: list[pd.DataFrame] = []
        internal_frames: list[pd.DataFrame] = []
        external_frames: list[pd.DataFrame] = []

        for fold_idx in range(1, 6):

            fold_name = f"fold_{fold_idx:02d}"
            fold_dir = tuned_root / fold_name

            val_path = fold_dir / f"preds_{run_id}_val.csv"
            internal_path = fold_dir / f"preds_{run_id}_test_internal.csv"
            external_path = fold_dir / f"preds_{run_id}_test_external.csv"

            require_file(
                val_path,
                f"{run_id} validation prediction file"
            )
            require_file(
                internal_path,
                f"{run_id} internal-test prediction file"
            )
            require_file(
                external_path,
                f"{run_id} external-test prediction file"
            )

            val_frames.append(pd.read_csv(val_path))
            internal_frames.append(pd.read_csv(internal_path))
            external_frames.append(pd.read_csv(external_path))

        # ============================================================
        # 3. GLOBAL THRESHOLD FROM POOLED OOF VALIDATION
        # ============================================================

        pooled_validation = pd.concat(
            val_frames,
            ignore_index=True,
        )

        validate_unique_key(
            pooled_validation,
            SAMPLE_KEY_COL,
            f"{run_id} pooled tuned validation predictions",
        )

        pooled_threshold = AblationBenchmarker._select_validation_threshold(
            pooled_validation["actual"].astype(int).to_numpy(),
            pooled_validation["prob"].to_numpy(dtype=float),
        )

        pooled_validation["pred"] = (
            pooled_validation["prob"]
            >= pooled_threshold.threshold
        ).astype(int)

        pooled_validation["global_threshold"] = (
            pooled_threshold.threshold
        )

        pooled_validation.to_csv(
            ensemble_root / f"pooled_validation_{run_id}.csv",
            index=False,
        )

        validation_metrics = evaluate_probabilities(
            pooled_validation["actual"].astype(int).to_numpy(),
            pooled_validation["prob"].to_numpy(dtype=float),
            pooled_threshold.threshold,
        )

        ensemble_rows.append(
            {
                **run.to_dict(),
                "evaluation_method": "pooled_validation",
                "split": "pooled_validation",
                **validation_metrics,
            }
        )

        # ============================================================
        # 4. FIVE-FOLD PROBABILITY ENSEMBLE
        # ============================================================

        for split_name, fold_frames in [
            ("test_internal", internal_frames),
            ("test_external", external_frames),
        ]:

            ensemble_frame = average_test_prediction_files(
                fold_frames,
                split_name,
            )

            ensemble_frame["pred"] = (
                ensemble_frame["prob"]
                >= pooled_threshold.threshold
            ).astype(int)

            ensemble_frame["global_threshold"] = (
                pooled_threshold.threshold
            )

            ensemble_frame.to_csv(
                ensemble_root
                / f"preds_{run_id}_{split_name}_ensemble.csv",
                index=False,
            )

            metrics = evaluate_probabilities(
                ensemble_frame["actual"].astype(int).to_numpy(),
                ensemble_frame["prob"].to_numpy(dtype=float),
                pooled_threshold.threshold,
            )

            ensemble_result = {
                **run.to_dict(),
                "evaluation_method": "five_fold_probability_ensemble",
                "split": split_name,
                **metrics,
            }

            ensemble_rows.append(ensemble_result)

            comparison_row = {
                "config": run["config"],
                "model": run["model"],
                "pca": run["pca"],
                "smote": run["smote"],
                "evaluation_method": "ensemble",
                "split": split_name,
                "fold": "all_5_folds",
            }

            for metric in metric_columns:
                if metric in metrics:
                    comparison_row[metric] = metrics[metric]

            comparison_rows.append(comparison_row)

    # ================================================================
    # 5. SAVE FINAL TABLES
    # ================================================================

    best_fold_frame = pd.DataFrame(best_fold_rows)
    ensemble_frame = pd.DataFrame(ensemble_rows)
    comparison_frame = pd.DataFrame(comparison_rows)

    best_fold_frame.to_csv(
        tuned_root / "TUNED_best_fold_results.csv",
        index=False,
    )

    ensemble_frame.to_csv(
        ensemble_root / "TUNED_probability_ensemble_results.csv",
        index=False,
    )

    comparison_frame.to_csv(
        tuned_root / "TUNED_best_fold_vs_ensemble.csv",
        index=False,
    )

    logging.info(
        "Saved tuned best-fold and ensemble evaluation results."
    )

    return best_fold_frame, ensemble_frame, comparison_frame


# =============================================================================
# 11. MAIN
# =============================================================================


def main() -> None:
    configure_logging()
    set_global_seed(SEED)

    RES_ROOT.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    save_environment_info()

    oof_path = EMBEDDINGS_DIR / "oof_train_val_embeddings.csv"
    test_path = EMBEDDINGS_DIR / "test_embeddings.csv"
    external_path = EMBEDDINGS_DIR / "external_test_season2_embeddings.csv"

    if FORCE_REPROCESS_EMBEDDINGS or not oof_path.exists():
        df_oof = extract_oof_embeddings()
    else:
        df_oof = pd.read_csv(oof_path)

    if FORCE_REPROCESS_EMBEDDINGS or not test_path.exists():
        df_test_embeddings = extract_test_ensemble(
            BASE_SPLIT_CV / "test" / "metadata.csv",
            "test_embeddings.csv",
        )
    else:
        df_test_embeddings = pd.read_csv(test_path)

    if FORCE_REPROCESS_EMBEDDINGS or not external_path.exists():
        df_external_embeddings = extract_test_ensemble(
            BASE_SPLIT_CV / "external_test_season2" / "metadata.csv",
            "external_test_season2_embeddings.csv",
        )
    else:
        df_external_embeddings = pd.read_csv(external_path)

    validate_unique_key(df_oof, SAMPLE_KEY_COL, "OOF embeddings")
    validate_unique_key(df_test_embeddings, SAMPLE_KEY_COL, "Internal test embeddings")
    validate_unique_key(
        df_external_embeddings,
        SAMPLE_KEY_COL,
        "External test embeddings",
    )

    feature_configs = build_feature_configurations(df_oof)

    if RUN_ABLATION:
        run_ablation_experiments(
            df_oof,
            df_test_embeddings,
            df_external_embeddings,
            feature_configs,
        )

    all_results, average_results = aggregate_ablation_results()

    if BUILD_GLOBAL_PROBABILITY_ENSEMBLES:
        build_global_probability_ensembles(all_results)

    if RUN_HYPERPARAMETER_TUNING:
        run_constrained_tuning(
            df_oof,
            df_test_embeddings,
            df_external_embeddings,
            feature_configs,
            average_results,
        )
    if BUILD_TUNED_FINAL_EVALUATION:
      build_tuned_final_evaluation()

    logging.info("Pipeline completed successfully. Results: %s", RES_ROOT)


if __name__ == "__main__":
    main()