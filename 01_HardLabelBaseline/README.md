# Human Hard-Label Emotion Baselines

Goal: Train same-split ALBERT baselines on human 3-class hard labels for Regular and Augmented inputs.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `01_HardLabelBaseline.ipynb`: notebook for this model set

## Expected Inputs

- `Mental_Health_Dataset.csv`
- `cancersupport_with_ai_labels_mini_combined.csv for augmented ROLE_/CANCER_ tokens`
- `split_assignments.csv`

## Expected Outputs

- `hard_label_regular_metrics.csv`
- `hard_label_augmented_metrics.csv`
- `hard_label_regular_predictions_test.csv`
- `hard_label_augmented_predictions_test.csv`
- `hard_label_archived_regular_logistic_regression_metrics.csv`
- `hard_label_archived_regular_random_forest_metrics.csv`
- `hard_label_archived_regular_lightgbm_metrics.csv`
- `hard_label_archived_regular_gru_metrics.csv`
- `hard_label_archived_paper_baselines_summary.csv`

## Notes

- This directly addresses the reviewer request for a hard-label baseline on the exact same data partition.
- Uses the R0 cleaned-text and ROLE_/CANCER_ augmented-text convention, with the R1 canonical split.
- The main reviewer-response baseline is ALBERT only, matching R0. The earlier archived hard-label model family from `99_Archive/AILabel/CancerSupport_4Class.ipynb` is kept as an optional section and skipped by default (`RUN_ARCHIVED_PAPER_BASELINES=False`).
