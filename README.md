# Early Prediction of Pomegranate Fruit Cracking

This repository contains the code and experimental results accompanying the M.Sc. thesis **Early Prediction of Pomegranate Fruit Cracking Using Multimodal RGB and Thermal Imaging**. The research investigates whether RGB images, thermal images, and agronomic and environmental metadata can predict, at the individual-fruit level, whether a currently intact pomegranate will crack by the following monitoring session, approximately two weeks later.

The study compares image-only ResNet-50 models with a feature-level fusion framework that combines 2,048-dimensional CNN embeddings with tabular metadata, including fruit-level thermal summaries. The fused representations are evaluated using Logistic Regression, SVM, Random Forest, and XGBoost. Current-crack identification is included as a comparison task to distinguish the detection of visible damage from the prediction of future cracking. Prediction performance is evaluated through five-fold cross-validation grouped by fruit, a held-out internal test set from the 2024 season, and an independent external test set from the 2025 season.

## Repository contents

| Folder or file | Contents |
| --- | --- |
| `preprocessing/` | Photo organization for both seasons, SAM segmentation, season 2 segmentation validation, RGB enhancement, and thermal statistics |
| `data_preparation/` | Metadata and target preparation for detection, prediction, and the external season |
| `splits/` | Fruit-level internal-test selection and five-fold cross-validation splits |
| `cnn/` | ResNet-50 training, validation, independent-test evaluation, and probability ensembles |
| `fusion/` | Main fusion workflow in `run_fusion_separate_pca.py`: embedding extraction, feature combinations, PCA, SMOTE, classifier selection, tuning, and test evaluation |
| `analysis/` | Group-level and individual metadata permutation importance |
| `results/` | Ten original CSV result summaries; see the [results README](results/README.md) |
| `requirements.txt` | Python dependencies used by the supplied code |

This repository presents the main analysis workflow described in the thesis.

The Python files retain the supplied Colab code, Google Drive paths, and saved experiment settings. Filenames and folders were organized for this repository. The CNN test/ensemble evaluation file was recovered from pasted Colab code by removing Markdown formatting escapes, without changing its evaluation logic.

## Main workflow

1. Organize the season 1 and season 2 images using `preprocessing/organize_photos_season1.py` and `preprocessing/organize_photos_season2.py`.
2. Segment the fruit with `preprocessing/sam_segmentation.py`, validate segmentation on the external season with `preprocessing/validate_segmentation_season2.py`, and apply RGB enhancement using `preprocessing/rgb_preprocessing.py`.
3. Prepare the detection, prediction, and season 2 metadata using `data_preparation/build_detection_metadata.py`, `data_preparation/build_prediction_metadata.py`, and `data_preparation/build_season2_metadata.py`.
4. Generate the fruit-level splits using `splits/make_detection_splits.py` and `splits/make_prediction_splits.py`. Then compute the fruit-level thermal summaries using `preprocessing/thermal_statistics.py`.
5. Train the image models using `cnn/train_detection.py` and `cnn/train_prediction.py`. Evaluate the saved fold models and ensembles using `cnn/evaluate_test_and_ensemble.py`.
6. Run the main fusion experiments using `fusion/run_fusion_separate_pca.py`.
7. Analyze feature importance using `analysis/permutation_importance.py`.

The main fusion code extracts 2,048-dimensional ResNet-50 embeddings. When PCA is enabled, it is fitted separately to each active embedding source to retain at least 95% of the training variance. Classifier selection and hyperparameter tuning use mean validation F1-score across the five folds.


## Results

The `results/` folder contains four CNN reports, four fusion summaries, and two permutation-importance summaries. The [results README](results/README.md) describes their contents and identifies the corresponding thesis tables and appendices. Numerical values are preserved from the supplied CSV exports.
