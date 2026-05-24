# LLM Soft-Label Emotion Models

Goal: Train ALBERT using LLM 3-class soft probability targets for Regular and Augmented inputs.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `03_SoftLabelSupervision.ipynb`: notebook for this model set

## Expected Inputs

- `Mental_Health_Dataset.csv`
- `cancersupport_with_ai_labels_mini_combined.csv`
- `split_assignments.csv`

## Expected Outputs

- `soft_label_regular_metrics.csv`
- `soft_label_augmented_metrics.csv`
- `soft_label_regular_predictions_test.csv`
- `soft_label_augmented_predictions_test.csv`

## Notes

- This reruns the original soft-label analysis on the revised annotation file and exact split.
- Report hard metrics against human labels and soft metrics against LLM distributions.
- Uses the R0 cleaned-text and ROLE_/CANCER_ augmented-text convention, with the R1 canonical split.
- Includes a RoBERTa stronger-encoder sensitivity section enabled by default with a small 5-iteration search; ALBERT remains the primary reported model.
