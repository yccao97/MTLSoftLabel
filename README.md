# MTLSoftLabel

Experiment code for **Multi-Task Learning and Soft-Label Supervision for
Psychosocial Burden Assessment in Cancer Peer-Support Text**.

**Preprint:** https://www.medrxiv.org/content/10.64898/2026.04.03.26350034v1

The notebooks cover two complementary studies on the *Mental Health
Insights: Vulnerable Cancer Survivors & Caregivers* corpus (CC BY 4.0):

- **Study 1 — HEOR Multi-Task Learning.** Joint prediction of 7
  psychosocial burden subscales plus composite/high-need targets using a
  shared ALBERT encoder with Kendall uncertainty weighting.
- **Study 2 — Soft-Label Supervision.** Training on LLM-derived
  probability distributions vs. hard labels under a 2 × 2 design (regular
  vs. augmented × soft-on-train vs. soft-on-all).

This repository contains code only. Data, model checkpoints, and per-run
outputs are not included; provide them yourself via the env vars described
below.

## Notebooks

| Folder | Notebook | Purpose |
|---|---|---|
| `01_HardLabelBaseline/` | `01_HardLabelBaseline.ipynb` | Within-partition hard-label baseline (regular + augmented) |
| `02_LLMArgmaxHardLabelControl/` | `02_LLMArgmaxHardLabelControl.ipynb` | LLM-argmax hard-label control |
| `03_SoftLabelSupervision/` | `03_SoftLabelSupervision.ipynb` | Soft-label supervision (2×2: regular/augmented × train/all) |
| `04_HEOR_MTL/` | `04_HEOR_MTL.ipynb` | HEOR multi-task learning (composite, composite+RC, subscales, subscales+RC) |
| `05_HEOR_SingleTaskBaselines/` | `05_HEOR_SingleTaskBaselines.ipynb` | Per-subscale single-task baselines |
| `06_EvaluationAnchors_Bootstrap/` | `06_EvaluationAnchors_Bootstrap.ipynb` | Bootstrap CIs, paired deltas, Brier anchors, JSD, truncation diagnostic |
| `07_HumanAnnotationAudit/` | `07_HumanAnnotationAudit.ipynb` | Human annotation audit, inter-rater reliability, human-vs-LLM agreement |

Each `NN_*/` folder contains the notebook plus a `README.md` describing
its inputs, configuration, and outputs in more detail.

## Path resolution

Every notebook uses the same portable path-finder block. It resolves
`PROJECT_ROOT` in this order:

1. The `PROJECT_ROOT` environment variable, if set.
2. The default Colab Drive location (`/content/drive/MyDrive/NLP_Projects/VulnerableCancerPatients`), if running on Colab and the path exists.
3. The notebook's own parent directory, as a local fallback.

`SCRIPTS_DIR` and `DATA_DIR` are independently overridable:

```sh
export PROJECT_ROOT=/path/to/project_root
export SCRIPTS_DIR=/path/to/scripts            # optional; defaults to $PROJECT_ROOT/scripts
export DATA_DIR=/path/to/data                  # optional; defaults to $PROJECT_ROOT/data
jupyter lab
```

Per-notebook `outputs/` directories are created automatically under each
`NN_*/` folder.

## Shared helper modules

The notebooks import from a sibling `scripts/` package
(`data_utils.py`, `metrics.py`, `train_eval.py`, `hp_search.py`). These
helpers are not included in this repository; point `SCRIPTS_DIR` at the
directory that holds them before running.

## Dependencies

Standard scientific Python stack plus Hugging Face Transformers:

```
numpy pandas joblib scikit-learn lightgbm torch transformers tqdm
```

`06_EvaluationAnchors_Bootstrap` requires `transformers` for the
sequence-truncation diagnostic (uses the ALBERT tokenizer).
`07_HumanAnnotationAudit` additionally needs `openpyxl` for the xlsm
parser and consensus reconciliation steps.

## Data

The corpus is *Mental Health Insights: Vulnerable Cancer Survivors &
Caregivers* (N = 10,392), released under CC BY 4.0 and available on
Kaggle, GitHub, and Mendeley Data. The notebooks expect the dataset to
sit at `$PROJECT_ROOT/data/`.

## Code contributors

- Zhongyan Wang
- Zhanyi Ding
- Yeyubei Zhang
- Xiaorui Shen
- Yunchong Liu

## License

Code is released under the MIT License (see `LICENSE`).
