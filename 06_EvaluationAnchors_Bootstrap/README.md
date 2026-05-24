# Evaluation Anchors and Bootstrap CIs

Goal: Compute reviewer-requested non-training baselines and uncertainty estimates.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `06_EvaluationAnchors_Bootstrap.ipynb`: notebook for this model set

## Expected Inputs

- `Prediction CSVs from folders 00-04`
- `cancersupport_with_ai_labels_mini_combined.csv`
- `split_assignments.csv`

## Expected Outputs

- `study2_brier_anchors.csv`
- `study2_per_class_metrics.csv`
- `study1_delta_bootstrap_ci.csv`
- `study2_delta_bootstrap_ci.csv`
- `sequence_truncation_analysis.csv`

## Notes

- Compute majority-class and uniform-probability Brier anchors.
- Add paired bootstrap CIs for deltas.
- Add per-class precision, recall, and F1 for Study 2.
- Add QWK and MAE summaries for ordinal HEOR subscales.
