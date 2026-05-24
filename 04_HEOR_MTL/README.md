# HEOR Multi-Task Learning

Goal: Train the R0 HEOR ALBERT MTL condition family: Composite, Composite+RC, Subscales, and Subscales+RC.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `04_HEOR_MTL.ipynb`: notebook for this model set

## Expected Inputs

- `Mental_Health_Dataset.csv`
- `cancersupport_with_ai_labels_mini_combined.csv`
- `split_assignments.csv`

## Expected Outputs

- `heor_mtl_composite_metrics.csv`
- `heor_mtl_composite_rc_metrics.csv`
- `heor_mtl_subscales_metrics.csv`
- `heor_mtl_subscales_rc_metrics.csv`
- `heor_mtl_subscales_rc_role_cap_metrics.csv`
- `kendall_task_weights.csv`

## Notes

- This keeps the original R0 four-condition HEOR model family.
- For Subscales+RC, select checkpoints using only the seven primary HEOR subscale metrics.
- Mask UNKNOWN cancer type and UNCLEAR speaker role in auxiliary losses.
- Report weighted F1, macro F1 or balanced accuracy, QWK, and MAE.
- Includes a RoBERTa Subscales+RC robustness sensitivity on a 40% stratified subsample with a 5-iteration search; ALBERT remains the primary reported model family.
