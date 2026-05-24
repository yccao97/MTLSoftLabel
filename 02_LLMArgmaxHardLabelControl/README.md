# LLM-Argmax Hard-Label Control

Goal: Train ALBERT on the argmax of the LLM emotion distribution for Regular and Augmented inputs.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `02_LLMArgmaxHardLabelControl.ipynb`: notebook for this model set

## Expected Inputs

- `Mental_Health_Dataset.csv`
- `cancersupport_with_ai_labels_mini_combined.csv`
- `split_assignments.csv`

## Expected Outputs

- `llm_argmax_regular_metrics.csv`
- `llm_argmax_augmented_metrics.csv`
- `llm_argmax_regular_predictions_test.csv`
- `llm_argmax_augmented_predictions_test.csv`

## Notes

- This separates label source from label form: LLM hard argmax vs LLM soft distribution.
- Evaluate against human labels and against the LLM probability distribution.
- Uses the R0 cleaned-text and ROLE_/CANCER_ augmented-text convention, with the R1 canonical split.
