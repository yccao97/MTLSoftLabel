"""Archived-paper hard-label baselines adapted to the R1 canonical split."""

from __future__ import annotations

import json
import math
import random
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from metrics import bootstrap_classification_metric_rows, classification_metrics


def _safe_name(name: str) -> str:
    return str(name).lower().replace(" ", "_").replace("/", "_")


def _split_arrays(df: pd.DataFrame, text_col: str, label_col: str):
    parts = {}
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split].copy()
        if part.empty:
            raise ValueError(f"Expected non-empty {split} split.")
        parts[split] = {
            "df": part,
            "text": part[text_col].fillna("").astype(str).tolist(),
            "y": part[label_col].astype(int).to_numpy(),
            "source_row": part["source_row"].astype(int).to_numpy(),
        }
    return parts


def _sklearn_param_grid(model_kind: str):
    if model_kind == "logistic_regression":
        return [
            {
                "C": C,
                "solver": solver,
                "penalty": "l2",
                "class_weight": class_weight,
            }
            for C, solver, class_weight in product([0.1, 1, 10], ["lbfgs", "saga"], ["balanced", None])
        ]
    if model_kind == "random_forest":
        return [
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "class_weight": class_weight,
            }
            for n_estimators, max_depth, min_samples_split, min_samples_leaf, class_weight in product(
                [50, 100, 200],
                [None, 10, 20],
                [2, 5, 10],
                [1, 2, 4],
                ["balanced", None],
            )
        ]
    if model_kind == "lightgbm":
        return [
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "class_weight": class_weight,
            }
            for n_estimators, learning_rate, num_leaves, max_depth, class_weight in product(
                [100, 200],
                [0.01, 0.1, 0.2],
                [31, 63],
                [-1, 10],
                ["balanced", None],
            )
        ]
    raise ValueError(f"Unknown model_kind: {model_kind}")


def _make_sklearn_model(model_kind: str, params: dict, seed: int):
    if model_kind == "logistic_regression":
        return LogisticRegression(
            **params,
            max_iter=1000,
            random_state=seed,
            multi_class="multinomial",
        )
    if model_kind == "random_forest":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)
    if model_kind == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("Install lightgbm to run the archived LightGBM baseline.") from exc
        return LGBMClassifier(
            objective="multiclass",
            **params,
            random_state=seed,
            importance_type="gain",
            verbose=-1,
        )
    raise ValueError(f"Unknown model_kind: {model_kind}")


def _aligned_predict_proba(model, X, num_classes: int):
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{type(model).__name__} does not expose predict_proba.")
    probs = model.predict_proba(X)
    out = np.zeros((X.shape[0], num_classes), dtype=float)
    for src_idx, cls in enumerate(model.classes_):
        cls_idx = int(cls)
        if 0 <= cls_idx < num_classes:
            out[:, cls_idx] = probs[:, src_idx]
    row_sums = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sums, out=np.full_like(out, 1.0 / num_classes), where=row_sums != 0)


def run_tfidf_hard_label_search(
    df: pd.DataFrame,
    output_dir: Path,
    condition_name: str,
    text_col: str,
    label_col: str,
    target_source: str,
    class_names: list[str],
    model_kind: str,
    bootstrap_n: int = 1000,
    seed: int = 42,
):
    """Run the archived TF-IDF hard-label grid-search baseline on the canonical split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)
    parts = _split_arrays(df, text_col, label_col)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(parts["train"]["text"])
    X_val = vectorizer.transform(parts["val"]["text"])
    X_test = vectorizer.transform(parts["test"]["text"])

    best = None
    rows = []
    for run_idx, params in enumerate(_sklearn_param_grid(model_kind), start=1):
        try:
            model = _make_sklearn_model(model_kind, params, seed)
            model.fit(X_train, parts["train"]["y"])
            val_pred = model.predict(X_val)
            val_f1 = f1_score(parts["val"]["y"], val_pred, average="weighted", zero_division=0)
            rows.append({"condition": condition_name, "run": run_idx, "val_weighted_f1": val_f1, **params})
            if best is None or val_f1 > best["val_weighted_f1"]:
                best = {"model": model, "params": params, "val_weighted_f1": val_f1, "run": run_idx}
        except Exception as exc:
            rows.append({"condition": condition_name, "run": run_idx, "val_weighted_f1": np.nan, "error": str(exc), **params})

    if best is None:
        raise RuntimeError(f"No valid model found for {condition_name}.")

    y_true = parts["test"]["y"]
    y_pred = best["model"].predict(X_test).astype(int)
    y_prob = _aligned_predict_proba(best["model"], X_test, num_classes)
    metrics = classification_metrics(y_true, y_pred, y_prob, class_names)
    metrics.update(
        {
            "condition": condition_name,
            "target_source": target_source,
            "model_family": model_kind,
            "text_col": text_col,
            "bootstrap_n": bootstrap_n,
            "best_run": best["run"],
            "val_weighted_f1": best["val_weighted_f1"],
            **best["params"],
        }
    )
    ci_df = bootstrap_classification_metric_rows(
        y_true,
        y_pred,
        y_prob,
        class_names=class_names,
        n_boot=bootstrap_n,
        seed=seed,
    )
    ci_df.insert(0, "condition", condition_name)
    ci_df.insert(1, "target_source", target_source)

    pred_df = pd.DataFrame(
        {
            "source_row": parts["test"]["source_row"],
            "condition": condition_name,
            "target_source": target_source,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    if "human_emotion_3class_id" in parts["test"]["df"].columns:
        pred_df["y_true_human_emotion"] = parts["test"]["df"]["human_emotion_3class_id"].astype(int).to_numpy()
    for i, name in enumerate(class_names):
        pred_df[f"prob_{_safe_name(name)}"] = y_prob[:, i]

    pd.DataFrame(rows).to_csv(output_dir / f"{condition_name}_hp_search.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / f"{condition_name}_metrics.csv", index=False)
    ci_df.to_csv(output_dir / f"{condition_name}_metrics_ci.csv", index=False)
    pred_df.to_csv(output_dir / f"{condition_name}_predictions_test.csv", index=False)
    joblib.dump(best["model"], output_dir / f"{condition_name}_model.joblib")
    joblib.dump(vectorizer, output_dir / f"{condition_name}_tfidf.joblib")
    with (output_dir / f"{condition_name}_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "condition": condition_name,
                "model_family": model_kind,
                "target_source": target_source,
                "text_col": text_col,
                "label_col": label_col,
                "class_names": class_names,
                "bootstrap_n": bootstrap_n,
                "seed": seed,
                "best_params": best["params"],
            },
            f,
            indent=2,
        )
    return metrics, pred_df


def _texts_to_tensor(texts: list[str], word_to_index: dict[str, int], max_len: int) -> torch.Tensor:
    arr = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = str(text).split()[:max_len]
        ids = [word_to_index.get(token, 0) for token in tokens]
        arr[i, : len(ids)] = ids
    return torch.tensor(arr, dtype=torch.long)


def _build_vocab(texts: list[str], vocab_size: int) -> dict[str, int]:
    counts = {}
    for text in texts:
        for token in str(text).split():
            counts[token] = counts.get(token, 0) + 1
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max(0, vocab_size - 1)]
    return {word: idx + 1 for idx, (word, _) in enumerate(top)}


class GRUDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, labels: np.ndarray):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"input_ids": self.sequences[idx], "label": self.labels[idx]}


class GRUSentimentModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, hidden = self.gru(embedded)
        return self.fc(self.dropout(hidden[-1]))


def _class_weights(labels: np.ndarray, num_classes: int, device: torch.device):
    counts = np.bincount(np.asarray(labels).astype(int), minlength=num_classes).astype(float)
    if counts.sum() == 0:
        return torch.ones(num_classes, dtype=torch.float32, device=device)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _predict_gru(model, loader, device):
    model.eval()
    all_prob, all_pred, all_true = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids)
            probs = torch.softmax(logits, dim=1)
            all_prob.append(probs.detach().cpu().numpy())
            all_pred.append(probs.argmax(dim=1).detach().cpu().numpy())
            all_true.append(labels.detach().cpu().numpy())
    return np.vstack(all_prob), np.concatenate(all_pred), np.concatenate(all_true)


def run_gru_hard_label_search(
    df: pd.DataFrame,
    output_dir: Path,
    condition_name: str,
    text_col: str,
    label_col: str,
    target_source: str,
    class_names: list[str],
    bootstrap_n: int = 1000,
    seed: int = 42,
    n_iter: int = 25,
    max_len: int = 384,
    vocab_size: int = 10000,
    batch_size: int = 16,
):
    """Run the archived GRU hard-label random-search baseline on the canonical split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_names)
    parts = _split_arrays(df, text_col, label_col)
    word_to_index = _build_vocab(parts["train"]["text"], vocab_size)
    actual_vocab_size = len(word_to_index) + 1

    train_ds = GRUDataset(_texts_to_tensor(parts["train"]["text"], word_to_index, max_len), parts["train"]["y"])
    val_ds = GRUDataset(_texts_to_tensor(parts["val"]["text"], word_to_index, max_len), parts["val"]["y"])
    test_ds = GRUDataset(_texts_to_tensor(parts["test"]["text"], word_to_index, max_len), parts["test"]["y"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    rng = np.random.default_rng(seed)
    best = None
    rows = []
    for run_idx in range(1, int(n_iter) + 1):
        hp = {
            "embedding_dim": int(rng.integers(150, 251)),
            "hidden_dim": int(rng.integers(256, 769)),
            "learning_rate": float(10 ** rng.uniform(math.log10(1e-4), math.log10(1e-3))),
            "epochs": int(rng.integers(5, 10)),
            "dropout": 0.3,
        }
        model = GRUSentimentModel(actual_vocab_size, hp["embedding_dim"], hp["hidden_dim"], num_classes, hp["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
        criterion = nn.CrossEntropyLoss(weight=_class_weights(parts["train"]["y"], num_classes, device))

        for epoch in range(1, hp["epochs"] + 1):
            model.train()
            for batch in tqdm(train_loader, desc=f"{condition_name} run {run_idx} epoch {epoch}", leave=False):
                ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(ids), labels)
                loss.backward()
                optimizer.step()

        _, val_pred, val_true = _predict_gru(model, val_loader, device)
        val_f1 = f1_score(val_true, val_pred, average="weighted", zero_division=0)
        rows.append({"condition": condition_name, "run": run_idx, "val_weighted_f1": val_f1, **hp})
        if best is None or val_f1 > best["val_weighted_f1"]:
            best = {
                "model": model,
                "params": hp,
                "val_weighted_f1": val_f1,
                "run": run_idx,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    if best is None:
        raise RuntimeError(f"No valid GRU model found for {condition_name}.")

    best_model = GRUSentimentModel(
        actual_vocab_size,
        best["params"]["embedding_dim"],
        best["params"]["hidden_dim"],
        num_classes,
        best["params"]["dropout"],
    ).to(device)
    best_model.load_state_dict(best["state_dict"])
    y_prob, y_pred, y_true = _predict_gru(best_model, test_loader, device)
    metrics = classification_metrics(y_true, y_pred, y_prob, class_names)
    metrics.update(
        {
            "condition": condition_name,
            "target_source": target_source,
            "model_family": "gru",
            "text_col": text_col,
            "bootstrap_n": bootstrap_n,
            "best_run": best["run"],
            "val_weighted_f1": best["val_weighted_f1"],
            **best["params"],
        }
    )
    ci_df = bootstrap_classification_metric_rows(
        y_true,
        y_pred,
        y_prob,
        class_names=class_names,
        n_boot=bootstrap_n,
        seed=seed,
    )
    ci_df.insert(0, "condition", condition_name)
    ci_df.insert(1, "target_source", target_source)

    pred_df = pd.DataFrame(
        {
            "source_row": parts["test"]["source_row"],
            "condition": condition_name,
            "target_source": target_source,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    if "human_emotion_3class_id" in parts["test"]["df"].columns:
        pred_df["y_true_human_emotion"] = parts["test"]["df"]["human_emotion_3class_id"].astype(int).to_numpy()
    for i, name in enumerate(class_names):
        pred_df[f"prob_{_safe_name(name)}"] = y_prob[:, i]

    pd.DataFrame(rows).to_csv(output_dir / f"{condition_name}_hp_search.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / f"{condition_name}_metrics.csv", index=False)
    ci_df.to_csv(output_dir / f"{condition_name}_metrics_ci.csv", index=False)
    pred_df.to_csv(output_dir / f"{condition_name}_predictions_test.csv", index=False)
    torch.save(best["state_dict"], output_dir / f"{condition_name}_best_model.pt")
    with (output_dir / f"{condition_name}_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(word_to_index, f)
    with (output_dir / f"{condition_name}_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "condition": condition_name,
                "model_family": "gru",
                "target_source": target_source,
                "text_col": text_col,
                "label_col": label_col,
                "class_names": class_names,
                "bootstrap_n": bootstrap_n,
                "seed": seed,
                "max_len": max_len,
                "vocab_size": actual_vocab_size,
                "batch_size": batch_size,
                "best_params": best["params"],
            },
            f,
            indent=2,
        )
    return metrics, pred_df


def run_archived_hard_label_suite(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    text_col: str,
    label_col: str,
    target_source: str,
    class_names: list[str],
    bootstrap_n: int = 1000,
    seed: int = 42,
    run_lightgbm: bool = True,
    run_gru: bool = True,
    gru_n_iter: int = 25,
    max_len: int = 384,
):
    """Run the earlier-paper hard-label model family and save one summary CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    model_kinds = ["logistic_regression", "random_forest"]
    if run_lightgbm:
        model_kinds.append("lightgbm")

    for model_kind in model_kinds:
        condition_name = f"{prefix}_{model_kind}"
        m, _ = run_tfidf_hard_label_search(
            df=df,
            output_dir=output_dir,
            condition_name=condition_name,
            text_col=text_col,
            label_col=label_col,
            target_source=target_source,
            class_names=class_names,
            model_kind=model_kind,
            bootstrap_n=bootstrap_n,
            seed=seed,
        )
        metrics.append(m)

    if run_gru:
        condition_name = f"{prefix}_gru"
        m, _ = run_gru_hard_label_search(
            df=df,
            output_dir=output_dir,
            condition_name=condition_name,
            text_col=text_col,
            label_col=label_col,
            target_source=target_source,
            class_names=class_names,
            bootstrap_n=bootstrap_n,
            seed=seed,
            n_iter=gru_n_iter,
            max_len=max_len,
        )
        metrics.append(m)

    summary = pd.DataFrame(metrics)
    summary.to_csv(output_dir / f"{prefix}_archived_paper_baselines_summary.csv", index=False)
    return summary
