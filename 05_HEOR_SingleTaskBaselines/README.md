# HEOR Single-Task Baselines

Goal: Train independent ALBERT classifiers for HEOR subscales to benchmark the MTL setup.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `05_HEOR_SingleTaskBaselines.ipynb`: notebook for this model set

## Expected Inputs

- `Mental_Health_Dataset.csv`
- `cancersupport_with_ai_labels_mini_combined.csv`
- `split_assignments.csv`

## Expected Outputs

- `heor_single_task_metrics.csv`
- `heor_single_task_predictions_test.csv`

## Notes

- At minimum run cost burden and perceived harm, as requested by the reviewer.
- If computationally feasible, run all seven HEOR subscales.
- Uses the same cleaned ALBERT text stream as the R0 HEOR notebook, with the R1 canonical split.
