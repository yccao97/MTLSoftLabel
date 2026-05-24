# Human Annotation Audit

Goal: Compare a human-coded validation subset against LLM annotations using the same integrated rubric.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `07_HumanAnnotationAudit.ipynb`: notebook for this model set

## Expected Inputs

- `Human-coded validation subset`
- `cancersupport_with_ai_labels_mini_combined.csv`
- `split_assignments.csv`

## Expected Outputs

- `human_llm_agreement_metrics.csv`
- `human_interrater_reliability.csv`
- `human_validation_sample_blinded.csv`
- `human_annotation_coder_1.xlsx`
- `human_annotation_coder_2.xlsx`
- `human_annotation_ai_visible_review.xlsx`
- `human_annotation_rubric.docx`

## Notes

- This is not a downstream training model, but it supports the reviewer-requested human validation.
- Use human-coded reference standard phrasing rather than human ground truth.
- Emotion is treated as ordinal for quadratic-weighted kappa using 0=VERY_NEGATIVE, 1=NEGATIVE, 2=NEUTRAL, 3=POSITIVE.
- Report both coder-entered `high_need_flag` agreement and formula-derived `derived_high_need_flag` agreement.
