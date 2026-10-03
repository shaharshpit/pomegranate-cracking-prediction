# Experiment results

This folder contains ten original CSV summaries from the experiments reported in the thesis **Early Prediction of Pomegranate Fruit Cracking Using Multimodal RGB and Thermal Imaging**. The filenames below match the supplied exports, including their numbered suffixes.

## CNN results

| File | Contents | Related thesis tables and appendices |
| --- | --- | --- |
| `final_report_rgb_identify (1).csv` | RGB current-crack identification: five validation folds, validation mean and standard deviation, selected fold, and internal-test best-fold and ensemble results | Table 10; Appendix A |
| `final_report_thermal_identify (1).csv` | Thermal current-crack identification: five validation folds, validation mean and standard deviation, selected fold, and internal-test best-fold and ensemble results | Table 10; Appendix B |
| `final_report_rgb_predict (2).csv` | RGB next-session prediction: five validation folds, validation summaries, and best-fold/ensemble evaluation on the internal and external test sets | Table 11; Appendix C |
| `final_report_thermal_predict (2).csv` | Thermal next-session prediction: five validation folds, validation summaries, and best-fold/ensemble evaluation on the internal and external test sets | Table 11; Appendix D |

The CNN reports include Accuracy, Precision, Recall, F1-score, ROC-AUC, the decision threshold, and sample count `N`. `VAL MEAN` and `VAL STD` summarize the five validation folds. The best fold is selected using validation F1-score. The sample count on a mean row is the total number of validation observations across the five folds.

## Fusion results

| File | Contents | Related thesis tables and appendices |
| --- | --- | --- |
| `MASTER_average_and_std_results (3).csv` | Baseline comparison across the five feature configurations, four classifiers, PCA settings, and SMOTE settings; fold means and standard deviations for train, validation, internal test, and external test | Tables 15–16; Appendix E |
| `TUNING_best_candidates (2).csv` | Selected hyperparameter candidate for each classifier, with its feature configuration and validation F1-score/PR-AUC summaries | Table 17 |
| `TUNED_average_and_std_results (3).csv` | Five-fold means and standard deviations for the selected tuned classifiers across train, validation, internal test, and external test | Table 17; aggregate rows in Appendices F–G |
| `TUNED_best_fold_vs_ensemble (2).csv` | Best-fold and five-fold ensemble results on the internal and external test sets, including thresholds, class supports, and confusion-matrix counts | Tables 18–19; best-fold/ensemble results in Appendices F–G |

The fusion classifiers are Logistic Regression, RBF SVM, Random Forest, and XGBoost. The five feature configurations are:

| Configuration identifier | Features |
| --- | --- |
| `1_MetaOnly` | Tabular metadata |
| `2_ThrPred_Meta` | Thermal-Pred embeddings and metadata |
| `3_RgbPred_Meta` | RGB-Pred embeddings and metadata |
| `4_RgbPred_ThrPred_Meta` | RGB-Pred embeddings, Thermal-Pred embeddings, and metadata |
| `5_RgbDet_ThrPred_Meta` | RGB-Det embeddings, Thermal-Pred embeddings, and metadata |

`RGB-Pred` and `Thermal-Pred` refer to embeddings from next-session prediction networks. `RGB-Det` refers to embeddings from the current-crack identification network. PCA is applied to the active embedding sources separately; the metadata-only configuration uses PCA OFF.

For aggregated files, the suffixes `_mean` and `_std` summarize fold-specific metrics. These averages are different quantities from the metrics of a probability ensemble, which are reported explicitly in `TUNED_best_fold_vs_ensemble (2).csv`. The tuned aggregate file does not list every fold result separately.

## Permutation feature importance

| File | Contents | Related thesis figures and appendix |
| --- | --- | --- |
| `grouped_permutation_importance_summary (1).csv` | Importance of each active image-embedding group and the metadata group for the tuned classifiers | Figure 18; Table H.1 |
| `metadata_permutation_importance_summary (1).csv` | Importance of seven metadata variables/groups for each tuned classifier | Figure 19; Table H.2 |

Permutation importance was evaluated on the internal test set, with 50 permutations per feature or group for each fold-specific model. The files report mean and standard deviation across the five models. `mean_delta_pr_auc` is the decrease in PR-AUC after permutation: positive values indicate a performance decrease; negative values indicate that permutation improved the score. ROC-AUC changes are provided as additional outputs.

## Dataset and reporting conventions

- Current-crack identification uses 1,716 development observations and a 306-observation internal test set.
- Next-session prediction uses 1,378 development observations, a 244-observation internal test set, and a 175-observation external test set.
- The internal test set is from the 2024 season; the external test set is from the 2025 season. The external test is used for next-session prediction.
- Precision, Recall, and F1 refer to the positive cracking class. ROC-AUC and PR-AUC summarize the ranking of predicted scores.
- Original numerical precision is retained. Thesis table references identify the corresponding analyses; this folder stores the supplied CSV values.

The folder documents CNN, fusion, and feature-importance outputs. Segmentation IoU outputs, raw per-image predictions, model weights, and individual tuned fusion fold-result files are separate experiment artifacts.
