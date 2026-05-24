"""Reviewer-facing metric helpers for the R1 rerun."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


DEFAULT_BOOTSTRAP_N = 1000


def _as_numpy(values):
    return np.asarray(values)


def soft_cross_entropy(y_soft, y_prob, eps: float = 1e-12) -> float:
    y_soft = _as_numpy(y_soft).astype(float)
    y_prob = np.clip(_as_numpy(y_prob).astype(float), eps, 1.0)
    return float(-np.mean(np.sum(y_soft * np.log(y_prob), axis=1)))


def multiclass_brier_score(y_target, y_prob) -> float:
    y_target = _as_numpy(y_target).astype(float)
    y_prob = _as_numpy(y_prob).astype(float)
    return float(np.mean(np.sum((y_prob - y_target) ** 2, axis=1)))


def one_hot(labels, num_classes: int) -> np.ndarray:
    labels = _as_numpy(labels).astype(int)
    out = np.zeros((len(labels), num_classes), dtype=float)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def classification_metrics(
    y_true,
    y_pred,
    y_prob=None,
    class_names: Iterable[str] | None = None,
    soft_target=None,
) -> dict:
    y_true = _as_numpy(y_true).astype(int)
    y_pred = _as_numpy(y_pred).astype(int)
    max_true = int(np.max(y_true)) if len(y_true) else 0
    max_pred = int(np.max(y_pred)) if len(y_pred) else 0
    labels = np.arange(max(max_true, max_pred) + 1)
    if class_names is None:
        class_names = [str(i) for i in labels]
    class_names = list(class_names)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    for i, name in enumerate(class_names):
        out[f"precision_{name}"] = float(precision[i])
        out[f"recall_{name}"] = float(recall[i])
        out[f"f1_{name}"] = float(f1[i])
        out[f"support_{name}"] = int(support[i])

    if y_prob is not None:
        y_prob = _as_numpy(y_prob).astype(float)
        if y_prob.shape[1] == len(class_names):
            try:
                out["macro_auc_ovr"] = float(
                    roc_auc_score(y_true, y_prob, labels=np.arange(len(class_names)), multi_class="ovr", average="macro")
                )
            except ValueError:
                out["macro_auc_ovr"] = np.nan
        hard_target = one_hot(y_true, len(class_names))
        out["hard_brier"] = multiclass_brier_score(hard_target, y_prob)

    if soft_target is not None and y_prob is not None:
        out["soft_cross_entropy"] = soft_cross_entropy(soft_target, y_prob)
        out["soft_brier"] = multiclass_brier_score(soft_target, y_prob)

    return out


def ordinal_metrics(y_true, y_pred) -> dict:
    y_true = _as_numpy(y_true).astype(int)
    y_pred = _as_numpy(y_pred).astype(int)
    return {
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def regression_metrics(y_true, y_pred) -> dict:
    y_true = _as_numpy(y_true).astype(float)
    y_pred = _as_numpy(y_pred).astype(float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_regression_metric_rows(
    y_true,
    y_pred,
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    seed: int = 42,
) -> pd.DataFrame:
    y_true = _as_numpy(y_true).astype(float)
    y_pred = _as_numpy(y_pred).astype(float)
    observed = regression_metrics(y_true, y_pred)
    metric_names = _numeric_metric_names(observed)
    boot_values = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            sample = regression_metrics(y_true[idx], y_pred[idx])
        except Exception:
            sample = {name: np.nan for name in metric_names}
        for name in metric_names:
            boot_values[name].append(sample.get(name, np.nan))

    rows = []
    for name in metric_names:
        lo, hi = _ci_bounds(boot_values[name])
        rows.append(
            {
                "metric": name,
                "value": float(observed[name]),
                "ci_low": lo,
                "ci_high": hi,
                "n_boot": int(n_boot),
                "ci_method": "nonparametric_percentile",
            }
        )
    return pd.DataFrame(rows)


def _numeric_metric_names(metrics: dict) -> list[str]:
    names = []
    for key, value in metrics.items():
        if key.startswith("support_"):
            continue
        try:
            float(value)
        except Exception:
            continue
        names.append(key)
    return names


def _ci_bounds(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_classification_metric_rows(
    y_true,
    y_pred,
    y_prob=None,
    class_names: Iterable[str] | None = None,
    soft_target=None,
    include_ordinal: bool = False,
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    seed: int = 42,
) -> pd.DataFrame:
    """Return one row per scalar metric with percentile bootstrap CIs.

    Support counts are intentionally excluded because they are descriptive counts,
    not performance metrics. All scalar metrics emitted by classification_metrics,
    plus optional ordinal metrics, receive CIs.
    """
    y_true = _as_numpy(y_true).astype(int)
    y_pred = _as_numpy(y_pred).astype(int)
    y_prob = None if y_prob is None else _as_numpy(y_prob).astype(float)
    soft_target = None if soft_target is None else _as_numpy(soft_target).astype(float)

    observed = classification_metrics(y_true, y_pred, y_prob, class_names, soft_target)
    if include_ordinal:
        observed.update(ordinal_metrics(y_true, y_pred))

    metric_names = _numeric_metric_names(observed)
    boot_values = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            sample = classification_metrics(
                y_true[idx],
                y_pred[idx],
                None if y_prob is None else y_prob[idx],
                class_names,
                None if soft_target is None else soft_target[idx],
            )
            if include_ordinal:
                sample.update(ordinal_metrics(y_true[idx], y_pred[idx]))
        except Exception:
            sample = {name: np.nan for name in metric_names}
        for name in metric_names:
            boot_values[name].append(sample.get(name, np.nan))

    rows = []
    for name in metric_names:
        lo, hi = _ci_bounds(boot_values[name])
        rows.append(
            {
                "metric": name,
                "value": float(observed[name]),
                "ci_low": lo,
                "ci_high": hi,
                "n_boot": int(n_boot),
                "ci_method": "nonparametric_percentile",
            }
        )
    return pd.DataFrame(rows)


def metric_ci(
    y_true,
    y_pred,
    metric_fn: Callable,
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    seed: int = 42,
    **kwargs,
) -> tuple[float, float, float]:
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    rng = np.random.default_rng(seed)
    observed = float(metric_fn(y_true, y_pred, **kwargs))
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        samples.append(float(metric_fn(y_true[idx], y_pred[idx], **kwargs)))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return observed, float(lo), float(hi)


def paired_delta_ci(
    y_true,
    pred_a,
    pred_b,
    metric_fn: Callable,
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    seed: int = 42,
    **kwargs,
) -> tuple[float, float, float]:
    y_true = _as_numpy(y_true)
    pred_a = _as_numpy(pred_a)
    pred_b = _as_numpy(pred_b)
    rng = np.random.default_rng(seed)
    observed = float(metric_fn(y_true, pred_a, **kwargs) - metric_fn(y_true, pred_b, **kwargs))
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        samples.append(float(metric_fn(y_true[idx], pred_a[idx], **kwargs) - metric_fn(y_true[idx], pred_b[idx], **kwargs)))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return observed, float(lo), float(hi)


def metrics_to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)
