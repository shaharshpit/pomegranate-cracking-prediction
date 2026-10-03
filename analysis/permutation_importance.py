# ============================================================
# PERMUTATION IMPORTANCE ANALYSIS
# Final tuned fusion models - Internal Test
# NO RETRAINING
# ============================================================

from pathlib import Path
import copy
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, roc_auc_score


# ============================================================
# 1. PATHS
# ============================================================

BASE_SPLIT_CV = Path(
    "/content/drive/MyDrive/Thesis/data/splits_predict_cv"
)

RES_ROOT = Path(
    "/content/drive/MyDrive/Thesis/results/"
    "intermediate_fusion_predict_cv_revised_separate_pca_ev95"
)

EMBEDDINGS_DIR = RES_ROOT / "embedding_features"

TUNED_ROOT = (
    RES_ROOT
    / "hyperparameter_tuning"
    / "best_tuned_evaluation"
)

OUTPUT_DIR = RES_ROOT / "permutation_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

N_REPEATS = 50
SEED = 42

SAMPLE_KEY_COL = "seg_rgb_path"
TARGET_COL = "cracked_next_session"

INTERNAL_METADATA_PATH = (
    BASE_SPLIT_CV / "test" / "metadata.csv"
)

INTERNAL_EMBEDDINGS_PATH = (
    EMBEDDINGS_DIR / "test_embeddings.csv"
)


# ============================================================
# 3. HUMAN-READABLE NAMES
# ============================================================

MODEL_NAMES = {
    "lr": "Logistic Regression",
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "rf": "Random Forest",
    "random_forest": "Random Forest",
    "xgb": "XGBoost",
    "xgboost": "XGBoost",
}

MODALITY_NAMES = {
    "rgb_pred": "RGB-Pred",
    "rgb_det": "RGB-Det",
    "thr_pred": "Thermal-Pred",
}


# ============================================================
# 4. LOAD INTERNAL TEST
# ============================================================

metadata = pd.read_csv(INTERNAL_METADATA_PATH)
embeddings = pd.read_csv(INTERNAL_EMBEDDINGS_PATH)

if SAMPLE_KEY_COL not in metadata.columns:
    raise KeyError(
        f"{SAMPLE_KEY_COL} missing from metadata."
    )

if SAMPLE_KEY_COL not in embeddings.columns:
    raise KeyError(
        f"{SAMPLE_KEY_COL} missing from embeddings."
    )

if TARGET_COL not in metadata.columns:
    raise KeyError(
        f"{TARGET_COL} missing from metadata."
    )

internal_df = metadata.merge(
    embeddings,
    on=SAMPLE_KEY_COL,
    how="inner",
    validate="one_to_one",
)

print("Internal test shape:", internal_df.shape)
print(
    "Positive cases:",
    int(internal_df[TARGET_COL].sum())
)


# ============================================================
# 5. FIND ALL SAVED FINAL TUNED PIPELINES
# ============================================================

pipeline_files = sorted(
    TUNED_ROOT.glob(
        "fold_*/pipeline_tuned_*.joblib"
    )
)

if len(pipeline_files) == 0:
    raise FileNotFoundError(
        f"No tuned pipelines found under:\n{TUNED_ROOT}"
    )

print(
    f"\nFound {len(pipeline_files)} tuned pipeline files."
)

for path in pipeline_files:
    print(path)


# ============================================================
# 6. HELPER: MODEL NAME
# ============================================================

def pretty_model_name(model_type):
    key = str(model_type).lower()
    return MODEL_NAMES.get(key, str(model_type))


# ============================================================
# 7. HELPER: PREPROCESS TEST DATA EXACTLY LIKE TRAINING
# ============================================================

def build_processed_test(artifact, df):

    meta_cols = list(
        artifact.get("meta_columns", [])
    )

    active_modalities = list(
        artifact.get("active_modalities", [])
    )

    embedding_cols_by_modality = artifact.get(
        "embedding_columns_by_modality", {}
    )

    scaler_meta = artifact.get(
        "scaler_meta", None
    )

    embedding_scalers = artifact.get(
        "scalers_embeddings_by_modality", {}
    )

    pcas = artifact.get(
        "pcas_by_modality", {}
    )

    blocks = {}
    ordered_blocks = []


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if meta_cols:

        missing = [
            c for c in meta_cols
            if c not in df.columns
        ]

        if missing:
            raise KeyError(
                f"Missing metadata columns: {missing}"
            )

        x_meta = df[meta_cols].to_numpy(
            dtype=np.float64
        )

        if scaler_meta is not None:
            x_meta = scaler_meta.transform(x_meta)

        blocks["metadata"] = x_meta
        ordered_blocks.append(
            ("metadata", x_meta)
        )


    # --------------------------------------------------------
    # Embedding modalities
    # --------------------------------------------------------

    for modality in active_modalities:

        cols = list(
            embedding_cols_by_modality[modality]
        )

        missing = [
            c for c in cols
            if c not in df.columns
        ]

        if missing:
            raise KeyError(
                f"{modality}: missing embedding columns: "
                f"{missing[:10]}"
            )

        x_mod = df[cols].to_numpy(
            dtype=np.float64
        )

        scaler = embedding_scalers.get(
            modality, None
        )

        if scaler is not None:
            x_mod = scaler.transform(x_mod)

        pca = pcas.get(modality, None)

        if pca is not None:
            x_mod = pca.transform(x_mod)

        blocks[modality] = x_mod

        ordered_blocks.append(
            (modality, x_mod)
        )


    # --------------------------------------------------------
    # Concatenate in the same order used during training
    # --------------------------------------------------------

    X = np.hstack(
        [block for _, block in ordered_blocks]
    )

    expected_dim = len(
        artifact.get(
            "output_feature_names", []
        )
    )

    if expected_dim and X.shape[1] != expected_dim:
        raise ValueError(
            f"Processed feature dimension mismatch: "
            f"{X.shape[1]} vs expected {expected_dim}"
        )

    return X, blocks, ordered_blocks


# ============================================================
# 8. HELPER: BLOCK COLUMN POSITIONS IN FINAL MATRIX
# ============================================================

def get_block_indices(ordered_blocks):

    indices = {}

    start = 0

    for name, block in ordered_blocks:

        end = start + block.shape[1]

        indices[name] = np.arange(
            start, end
        )

        start = end

    return indices


# ============================================================
# 9. HELPER: PERMUTE A GROUP OF COLUMNS TOGETHER
# ============================================================

def permute_columns_together(
    X,
    column_indices,
    rng,
):

    X_perm = X.copy()

    permutation = rng.permutation(
        X.shape[0]
    )

    # Important:
    # all columns in the group receive the SAME row permutation.
    # This preserves relationships within the group.
    X_perm[:, column_indices] = (
        X[permutation][:, column_indices]
    )

    return X_perm


# ============================================================
# 10. METADATA GROUP DEFINITIONS
# ============================================================

def get_metadata_groups(meta_cols):

    proposed_groups = {
        "Treatment": [
            "Treatment_Blue",
            "Treatment_Yellow",
        ],

        "Session": [
            "Session_2",
            "Session_3",
            "Session_4",
        ],

        "Loaded Branch": [
            "Loaded branch_1",
        ],

        "Old Branch": [
            "Old branch_1",
        ],

        "Thermal Mean": [
            "temps_mean_new",
        ],

        "Thermal SD": [
            "temps_std_new",
        ],

        "Ambient Temperature": [
            "temp",
        ],
    }

    groups = {}

    for group_name, cols in proposed_groups.items():

        existing = [
            c for c in cols
            if c in meta_cols
        ]

        if existing:
            groups[group_name] = existing

    return groups


# ============================================================
# 11. RESULTS CONTAINERS
# ============================================================

baseline_rows = []

grouped_repeat_rows = []
metadata_repeat_rows = []


# ============================================================
# 12. RUN ANALYSIS
# ============================================================

for pipeline_path in pipeline_files:

    print("\n" + "=" * 70)
    print("Loading:")
    print(pipeline_path)

    artifact = joblib.load(
        pipeline_path
    )

    model = artifact["model"]

    model_type = artifact.get(
        "model_type",
        model.__class__.__name__,
    )

    model_name = pretty_model_name(
        model_type
    )

    fold_name = artifact.get(
        "fold_name",
        pipeline_path.parent.name,
    )

    config_name = artifact.get(
        "config_name",
        "unknown",
    )

    meta_cols = list(
        artifact.get(
            "meta_columns", []
        )
    )

    active_modalities = list(
        artifact.get(
            "active_modalities", []
        )
    )


    # --------------------------------------------------------
    # Process Internal Test
    # --------------------------------------------------------

    X, blocks, ordered_blocks = (
        build_processed_test(
            artifact,
            internal_df,
        )
    )

    block_indices = get_block_indices(
        ordered_blocks
    )

    y = internal_df[
        TARGET_COL
    ].to_numpy(dtype=int)


    # --------------------------------------------------------
    # Baseline prediction
    # --------------------------------------------------------

    baseline_prob = model.predict_proba(
        X
    )[:, 1]

    baseline_pr_auc = (
        average_precision_score(
            y,
            baseline_prob,
        )
    )

    baseline_roc_auc = (
        roc_auc_score(
            y,
            baseline_prob,
        )
    )

    baseline_rows.append({
        "model": model_name,
        "model_type": model_type,
        "fold": fold_name,
        "config": config_name,
        "active_modalities": "|".join(
            active_modalities
        ),
        "baseline_pr_auc": baseline_pr_auc,
        "baseline_roc_auc": baseline_roc_auc,
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "pipeline_file": str(
            pipeline_path
        ),
    })

    print(
        f"{model_name} | {fold_name}"
    )

    print(
        f"Baseline PR-AUC  = "
        f"{baseline_pr_auc:.6f}"
    )

    print(
        f"Baseline ROC-AUC = "
        f"{baseline_roc_auc:.6f}"
    )


    # ========================================================
    # 12A. GROUPED SOURCE PERMUTATION
    # ========================================================

    source_groups = {
        "Metadata":
            block_indices["metadata"]
    }

    for modality in active_modalities:

        source_name = MODALITY_NAMES.get(
            modality,
            modality,
        )

        source_groups[source_name] = (
            block_indices[modality]
        )


    for source_name, indices in source_groups.items():

        for repeat in range(
            N_REPEATS
        ):

            rng = np.random.default_rng(
                SEED
                + repeat
                + 1000 * (
                    len(grouped_repeat_rows) + 1
                )
            )

            X_perm = (
                permute_columns_together(
                    X,
                    indices,
                    rng,
                )
            )

            prob_perm = (
                model.predict_proba(
                    X_perm
                )[:, 1]
            )

            perm_pr_auc = (
                average_precision_score(
                    y,
                    prob_perm,
                )
            )

            perm_roc_auc = (
                roc_auc_score(
                    y,
                    prob_perm,
                )
            )

            grouped_repeat_rows.append({
                "model": model_name,
                "model_type": model_type,
                "fold": fold_name,
                "config": config_name,
                "feature_source": source_name,
                "repeat": repeat + 1,

                "baseline_pr_auc":
                    baseline_pr_auc,

                "permuted_pr_auc":
                    perm_pr_auc,

                "delta_pr_auc":
                    baseline_pr_auc
                    - perm_pr_auc,

                "baseline_roc_auc":
                    baseline_roc_auc,

                "permuted_roc_auc":
                    perm_roc_auc,

                "delta_roc_auc":
                    baseline_roc_auc
                    - perm_roc_auc,
            })


    # ========================================================
    # 12B. INDIVIDUAL METADATA PERMUTATION
    # ========================================================

    metadata_groups = (
        get_metadata_groups(
            meta_cols
        )
    )

    # Metadata is the first block.
    metadata_start = (
        block_indices["metadata"][0]
    )

    meta_position = {
        col: metadata_start + i
        for i, col in enumerate(
            meta_cols
        )
    }


    for feature_name, feature_cols in metadata_groups.items():

        feature_indices = np.array([
            meta_position[col]
            for col in feature_cols
        ])

        for repeat in range(
            N_REPEATS
        ):

            rng = np.random.default_rng(
                SEED
                + repeat
                + 5000 * (
                    len(metadata_repeat_rows)
                    + 1
                )
            )

            X_perm = (
                permute_columns_together(
                    X,
                    feature_indices,
                    rng,
                )
            )

            prob_perm = (
                model.predict_proba(
                    X_perm
                )[:, 1]
            )

            perm_pr_auc = (
                average_precision_score(
                    y,
                    prob_perm,
                )
            )

            perm_roc_auc = (
                roc_auc_score(
                    y,
                    prob_perm,
                )
            )

            metadata_repeat_rows.append({
                "model": model_name,
                "model_type": model_type,
                "fold": fold_name,
                "config": config_name,

                "metadata_feature":
                    feature_name,

                "columns":
                    "|".join(
                        feature_cols
                    ),

                "repeat":
                    repeat + 1,

                "baseline_pr_auc":
                    baseline_pr_auc,

                "permuted_pr_auc":
                    perm_pr_auc,

                "delta_pr_auc":
                    baseline_pr_auc
                    - perm_pr_auc,

                "baseline_roc_auc":
                    baseline_roc_auc,

                "permuted_roc_auc":
                    perm_roc_auc,

                "delta_roc_auc":
                    baseline_roc_auc
                    - perm_roc_auc,
            })


# ============================================================
# 13. SAVE BASELINE METRICS
# ============================================================

baseline_df = pd.DataFrame(
    baseline_rows
)

baseline_df.to_csv(
    OUTPUT_DIR
    / "baseline_internal_metrics_by_fold.csv",
    index=False,
)


# ============================================================
# 14. GROUPED IMPORTANCE - RAW REPEATS
# ============================================================

grouped_repeat_df = pd.DataFrame(
    grouped_repeat_rows
)

grouped_repeat_df.to_csv(
    OUTPUT_DIR
    / "grouped_permutation_importance_all_repeats.csv",
    index=False,
)


# ============================================================
# 15. GROUPED IMPORTANCE - FIRST AVERAGE WITHIN EACH FOLD
# ============================================================

grouped_by_fold = (
    grouped_repeat_df
    .groupby(
        [
            "model",
            "model_type",
            "fold",
            "config",
            "feature_source",
        ],
        as_index=False,
    )
    .agg(
        delta_pr_auc_mean=(
            "delta_pr_auc",
            "mean",
        ),
        delta_pr_auc_repeat_std=(
            "delta_pr_auc",
            "std",
        ),
        delta_roc_auc_mean=(
            "delta_roc_auc",
            "mean",
        ),
        delta_roc_auc_repeat_std=(
            "delta_roc_auc",
            "std",
        ),
    )
)

grouped_by_fold.to_csv(
    OUTPUT_DIR
    / "grouped_permutation_importance_by_fold.csv",
    index=False,
)


# ============================================================
# 16. GROUPED IMPORTANCE - MEAN ± SD ACROSS 5 FOLDS
# ============================================================

grouped_summary = (
    grouped_by_fold
    .groupby(
        [
            "model",
            "feature_source",
        ],
        as_index=False,
    )
    .agg(
        mean_delta_pr_auc=(
            "delta_pr_auc_mean",
            "mean",
        ),
        std_delta_pr_auc=(
            "delta_pr_auc_mean",
            "std",
        ),
        mean_delta_roc_auc=(
            "delta_roc_auc_mean",
            "mean",
        ),
        std_delta_roc_auc=(
            "delta_roc_auc_mean",
            "std",
        ),
        n_folds=(
            "fold",
            "nunique",
        ),
    )
    .sort_values(
        [
            "model",
            "mean_delta_pr_auc",
        ],
        ascending=[
            True,
            False,
        ],
    )
)

grouped_summary.to_csv(
    OUTPUT_DIR
    / "grouped_permutation_importance_summary.csv",
    index=False,
)


# ============================================================
# 17. METADATA IMPORTANCE - RAW REPEATS
# ============================================================

metadata_repeat_df = pd.DataFrame(
    metadata_repeat_rows
)

metadata_repeat_df.to_csv(
    OUTPUT_DIR
    / "metadata_permutation_importance_all_repeats.csv",
    index=False,
)


# ============================================================
# 18. METADATA IMPORTANCE - WITHIN-FOLD MEAN
# ============================================================

metadata_by_fold = (
    metadata_repeat_df
    .groupby(
        [
            "model",
            "model_type",
            "fold",
            "config",
            "metadata_feature",
        ],
        as_index=False,
    )
    .agg(
        delta_pr_auc_mean=(
            "delta_pr_auc",
            "mean",
        ),
        delta_pr_auc_repeat_std=(
            "delta_pr_auc",
            "std",
        ),
        delta_roc_auc_mean=(
            "delta_roc_auc",
            "mean",
        ),
        delta_roc_auc_repeat_std=(
            "delta_roc_auc",
            "std",
        ),
    )
)

metadata_by_fold.to_csv(
    OUTPUT_DIR
    / "metadata_permutation_importance_by_fold.csv",
    index=False,
)


# ============================================================
# 19. METADATA IMPORTANCE - MEAN ± SD ACROSS 5 FOLDS
# ============================================================

metadata_summary = (
    metadata_by_fold
    .groupby(
        [
            "model",
            "metadata_feature",
        ],
        as_index=False,
    )
    .agg(
        mean_delta_pr_auc=(
            "delta_pr_auc_mean",
            "mean",
        ),
        std_delta_pr_auc=(
            "delta_pr_auc_mean",
            "std",
        ),
        mean_delta_roc_auc=(
            "delta_roc_auc_mean",
            "mean",
        ),
        std_delta_roc_auc=(
            "delta_roc_auc_mean",
            "std",
        ),
        n_folds=(
            "fold",
            "nunique",
        ),
    )
    .sort_values(
        [
            "model",
            "mean_delta_pr_auc",
        ],
        ascending=[
            True,
            False,
        ],
    )
)

metadata_summary.to_csv(
    OUTPUT_DIR
    / "metadata_permutation_importance_summary.csv",
    index=False,
)


# ============================================================
# 20. SIMPLE FIGURE - FEATURE SOURCES
# ============================================================

models = (
    grouped_summary["model"]
    .drop_duplicates()
    .tolist()
)

fig, axes = plt.subplots(
    len(models),
    1,
    figsize=(8, 4 * len(models)),
)

if len(models) == 1:
    axes = [axes]

for ax, model_name in zip(
    axes,
    models,
):

    temp = (
        grouped_summary[
            grouped_summary["model"]
            == model_name
        ]
        .sort_values(
            "mean_delta_pr_auc",
            ascending=True,
        )
    )

    ax.barh(
        temp["feature_source"],
        temp["mean_delta_pr_auc"],
        xerr=temp["std_delta_pr_auc"],
        capsize=4,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_title(
        model_name
    )

    ax.set_xlabel(
        "Mean decrease in PR-AUC after permutation"
    )

    ax.set_ylabel(
        "Feature source"
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "grouped_permutation_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 21. SIMPLE FIGURE - METADATA FEATURES
# ============================================================

fig, axes = plt.subplots(
    len(models),
    1,
    figsize=(9, 5 * len(models)),
)

if len(models) == 1:
    axes = [axes]

for ax, model_name in zip(
    axes,
    models,
):

    temp = (
        metadata_summary[
            metadata_summary["model"]
            == model_name
        ]
        .sort_values(
            "mean_delta_pr_auc",
            ascending=True,
        )
    )

    ax.barh(
        temp["metadata_feature"],
        temp["mean_delta_pr_auc"],
        xerr=temp["std_delta_pr_auc"],
        capsize=4,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_title(
        model_name
    )

    ax.set_xlabel(
        "Mean decrease in PR-AUC after permutation"
    )

    ax.set_ylabel(
        "Metadata feature"
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "metadata_permutation_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 22. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GROUPED FEATURE SOURCE IMPORTANCE")
print("=" * 70)

print(
    grouped_summary.to_string(
        index=False
    )
)

print("\n" + "=" * 70)
print("METADATA FEATURE IMPORTANCE")
print("=" * 70)

print(
    metadata_summary.to_string(
        index=False
    )
)

print("\nSaved all outputs to:")
print(OUTPUT_DIR)