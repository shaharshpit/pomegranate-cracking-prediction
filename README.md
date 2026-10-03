# Early Prediction of Pomegranate Fruit Cracking

Google Colab research code and result tables accompanying the M.Sc. thesis **Early Prediction of Pomegranate Fruit Cracking Using Multimodal RGB and Thermal Imaging**, by Shahar Shpitaler, Ben-Gurion University of the Negev, September 2026.

The study evaluates current-crack identification and next-session crack prediction using RGB images, thermal images, and environmental and agronomic metadata. Image-based ResNet-50 models are evaluated separately before their learned representations are combined with tabular features in the fusion stage.

## Repository contents

| Folder or file | Contents |
| --- | --- |
| `preprocessing/` | Photo organization for both seasons, SAM segmentation, season 2 segmentation validation, RGB enhancement, and thermal statistics |
| `data_preparation/` | Metadata and target preparation for detection, prediction, and the external season |
| `splits/` | Fruit-level internal-test selection and five-fold cross-validation splits |
| `cnn/` | ResNet-50 training, validation, independent-test evaluation, and probability ensembles |
| `fusion/` | Embedding extraction, feature combinations, PCA, SMOTE, classifier selection, tuning, and test evaluation |
| `analysis/` | Group-level and individual metadata permutation importance |
| `results/` | Ten original CSV result summaries; see the [results README](results/README.md) |
| `requirements.txt` | Python dependencies used by the supplied code |
| `SOURCE_MANIFEST.csv` | Original filenames, organized filenames, and source hashes |

The Python files retain the supplied Colab code, Google Drive paths, and saved experiment settings. Filenames and folders were organized for this repository. The CNN test/ensemble evaluation file was recovered from pasted Colab code by removing Markdown formatting escapes, without changing its evaluation logic.

## Main workflow

1. Organize the season 1 and season 2 images using `preprocessing/organize_photos_season1.py` and `preprocessing/organize_photos_season2.py`.
2. Segment the fruit with `preprocessing/sam_segmentation.py`, validate segmentation on the external season with `preprocessing/validate_segmentation_season2.py`, and apply RGB enhancement using `preprocessing/rgb_preprocessing.py`.
3. Prepare the detection, prediction, and season 2 metadata using `data_preparation/build_detection_metadata.py`, `data_preparation/build_prediction_metadata.py`, and `data_preparation/build_season2_metadata.py`.
4. Generate the fruit-level splits using `splits/make_detection_splits.py` and `splits/make_prediction_splits.py`. Place the external prediction metadata at `splits_predict_cv/external_test_season2/metadata.csv`, then compute the fruit-level thermal summaries using `preprocessing/thermal_statistics.py`.
5. Train the image models using `cnn/train_detection.py` and `cnn/train_prediction.py`. Evaluate the saved fold models and ensembles using `cnn/evaluate_test_and_ensemble.py`.
6. Run the main fusion experiments using `fusion/run_fusion_separate_pca.py`.
7. Analyze feature importance using `analysis/permutation_importance.py`.

The main fusion code extracts 2,048-dimensional ResNet-50 embeddings. When PCA is enabled, it is fitted separately to each active embedding source to retain at least 95% of the training variance. Classifier selection and hyperparameter tuning use mean validation F1-score across the five folds.

## Colab execution and saved settings

The files were exported from Google Colab and use paths under `/content/drive/MyDrive/Thesis/`. Mount Google Drive and configure the corresponding data and output paths when running the code. Some files define classes that are instantiated and called from notebook cells.

The saved CNN training loops begin at fold 2 because fold 1 had already completed before a Colab interruption. The test/ensemble evaluation code loads all five saved checkpoints for each task and imaging modality. The training exports retain the modality settings enabled when they were saved.

The main fusion export resumes from existing outputs: its baseline-comparison and hyperparameter-tuning flags are disabled, while final tuned evaluation is enabled. The code for those earlier stages remains in the file; configure the flags for the intended run.

For CNN ensembles, the code averages the five model probabilities and uses the mean of the five validation-selected thresholds. For fusion ensembles, it averages the test probabilities and selects an ensemble threshold by maximizing F1 on pooled validation predictions.

`requirements.txt` lists dependencies inferred from the imports; it is not a historical version lock. FLIR extraction also requires ExifTool. Raw images, workbooks, segmentation annotations, model checkpoints, and saved split CSVs are separate inputs and are not bundled with the result summaries.

## Additional supplied experiments

Files ending in `_alternative.py` preserve additional supplied Colab versions. The main workflow above uses `build_prediction_metadata.py` and `run_fusion_separate_pca.py`.

| Additional file | Difference from the main workflow |
| --- | --- |
| `data_preparation/build_prediction_metadata_alternative.py` | Export of `metadata_preperation(2).py`; writes `metadata_seq.csv` and removes `pom_id`, which the main prediction split code requires |
| `fusion/run_fusion_joint_pca_alternative.py` | Additional fusion version using joint PCA |
| `fusion/run_fusion_june_alternative.py` | Additional supplied fusion experiment |

## Results

The `results/` folder contains four CNN reports, four fusion summaries, and two permutation-importance summaries. The [results README](results/README.md) describes their contents and identifies the corresponding thesis tables and appendices. Numerical values are preserved from the supplied CSV exports.
