"""Shared model training and evaluation helpers for the R1 rerun notebooks."""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from data_utils import CANCER_LABELS, EMOTION_LABELS_3, EMOTION_PROB_COLS_3, HEOR_SUBSCALES, ROLE_LABELS
from hp_search import HParams, sample_hparams
from metrics import bootstrap_classification_metric_rows, bootstrap_regression_metric_rows, classification_metrics, ordinal_metrics, regression_metrics


SEED = 42


@dataclass
class TrainSettings:
    model_name: str = "albert-base-v2"
    max_length: int = 384
    batch_size: int = 16
    patience: int = 3
    seed: int = SEED
    n_iter: int = 25
    num_workers: int = 0
    grad_clip: float = 1.0
    warmup_ratio: float = 0.06
    bootstrap_n: int = 1000
    bootstrap_seed: int = 42


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextDataset(Dataset):
    def __init__(
        self,
        texts: Iterable[str],
        tokenizer,
        max_length: int,
        hard_labels=None,
        soft_labels=None,
        row_ids=None,
    ):
        texts = [str(text) for text in texts]
        self.enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        self.hard_labels = None if hard_labels is None else torch.tensor(np.asarray(hard_labels), dtype=torch.long)
        self.soft_labels = None if soft_labels is None else torch.tensor(np.asarray(soft_labels), dtype=torch.float32)
        self.row_ids = list(range(len(texts))) if row_ids is None else list(row_ids)

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.enc.items()}
        if self.hard_labels is not None:
            item["hard_labels"] = self.hard_labels[idx]
        if self.soft_labels is not None:
            item["soft_labels"] = self.soft_labels[idx]
        item["source_row"] = int(self.row_ids[idx])
        return item


class EncoderClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = getattr(self.encoder.config, "hidden_size")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if key == "source_row":
            out[key] = value.detach().cpu().tolist() if torch.is_tensor(value) else value
        elif isinstance(value, dict):
            out[key] = {
                nested_key: nested_value.to(device) if torch.is_tensor(nested_value) else nested_value
                for nested_key, nested_value in value.items()
            }
        else:
            out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def make_text_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    text_col: str,
    settings: TrainSettings,
    hard_label_col: str | None = None,
    soft_label_cols: list[str] | None = None,
):
    def arrays(df):
        hard = df[hard_label_col].astype(int).to_numpy() if hard_label_col else None
        soft = df[soft_label_cols].astype(float).to_numpy() if soft_label_cols else None
        return hard, soft

    train_hard, train_soft = arrays(train_df)
    val_hard, val_soft = arrays(val_df)
    test_hard, test_soft = arrays(test_df)

    train_ds = TextDataset(train_df[text_col], tokenizer, settings.max_length, train_hard, train_soft, train_df["source_row"])
    val_ds = TextDataset(val_df[text_col], tokenizer, settings.max_length, val_hard, val_soft, val_df["source_row"])
    test_ds = TextDataset(test_df[text_col], tokenizer, settings.max_length, test_hard, test_soft, test_df["source_row"])

    return (
        DataLoader(train_ds, batch_size=settings.batch_size, shuffle=True, num_workers=settings.num_workers),
        DataLoader(val_ds, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers),
        DataLoader(test_ds, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers),
    )


def class_weights(labels, num_classes: int, device: torch.device):
    counts = np.bincount(np.asarray(labels).astype(int), minlength=num_classes).astype(float)
    if counts.sum() == 0:
        return torch.ones(num_classes, dtype=torch.float32, device=device)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def soft_ce_loss(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def predict_classifier(model, loader, device: torch.device):
    model.eval()
    logits_all, hard_all, soft_all, rows_all = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                token_type_ids=batch.get("token_type_ids"),
            )
            logits_all.append(logits.detach().cpu())
            if "hard_labels" in batch:
                hard_all.append(batch["hard_labels"].detach().cpu())
            if "soft_labels" in batch:
                soft_all.append(batch["soft_labels"].detach().cpu())
            rows_all.extend(batch["source_row"])
    logits = torch.cat(logits_all).numpy()
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred = probs.argmax(axis=1)
    hard = torch.cat(hard_all).numpy() if hard_all else None
    soft = torch.cat(soft_all).numpy() if soft_all else None
    return {"source_row": rows_all, "logits": logits, "probs": probs, "pred": pred, "hard": hard, "soft": soft}


def train_classifier_once(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    hard_label_col: str | None,
    soft_label_cols: list[str] | None,
    loss_type: str,
    num_labels: int,
    settings: TrainSettings,
    hparams: HParams,
    eval_label_col: str | None = None,
    class_names: list[str] | None = None,
    selection_metric: str = "weighted_f1",
):
    set_seed(settings.seed)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
    train_loader, val_loader, test_loader = make_text_loaders(
        train_df, val_df, test_df, tokenizer, text_col, settings, hard_label_col, soft_label_cols
    )
    model = EncoderClassifier(settings.model_name, num_labels=num_labels, dropout=hparams.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    total_steps = max(1, len(train_loader) * hparams.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * settings.warmup_ratio),
        num_training_steps=total_steps,
    )
    ce_weight = None
    if loss_type == "hard" and hard_label_col:
        ce_weight = class_weights(train_df[hard_label_col], num_labels, device)
    hard_loss_fn = nn.CrossEntropyLoss(weight=ce_weight)

    best_state, best_score, best_epoch = None, -math.inf, 0
    bad_epochs = 0
    history = []
    for epoch in range(1, hparams.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                token_type_ids=batch.get("token_type_ids"),
            )
            if loss_type == "soft":
                loss = soft_ce_loss(logits, batch["soft_labels"])
            else:
                loss = hard_loss_fn(logits, batch["hard_labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        val_pred = predict_classifier(model, val_loader, device)
        if eval_label_col:
            val_y = val_df[eval_label_col].astype(int).to_numpy()
        else:
            val_y = val_pred["hard"]
        val_metrics = classification_metrics(val_y, val_pred["pred"], val_pred["probs"], class_names or EMOTION_LABELS_3)
        score = val_metrics.get(selection_metric, val_metrics["weighted_f1"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metrics})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= settings.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred = predict_classifier(model, test_loader, device)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": pd.DataFrame(history),
        "best_score": best_score,
        "best_epoch": best_epoch,
        "test_pred": test_pred,
    }


def run_classifier_search(
    df: pd.DataFrame,
    output_dir: Path,
    condition_name: str,
    text_col: str,
    hard_label_col: str | None,
    soft_label_cols: list[str] | None,
    loss_type: str,
    target_source: str,
    settings: TrainSettings,
    class_names: list[str] | None = None,
    num_labels: int = 3,
    eval_label_col: str | None = None,
    selection_metric: str = "weighted_f1",
    include_ordinal: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = EMOTION_LABELS_3 if class_names is None else class_names
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    runs = []
    best = None
    for i, hp in enumerate(sample_hparams(settings.n_iter, seed=settings.seed), start=1):
        print(f"\n[{condition_name}] search {i}/{settings.n_iter}: {hp}")
        result = train_classifier_once(
            train_df,
            val_df,
            test_df,
            text_col,
            hard_label_col,
            soft_label_cols,
            loss_type,
            num_labels,
            settings,
            hp,
            eval_label_col=eval_label_col,
            class_names=class_names,
            selection_metric=selection_metric,
        )
        runs.append({"condition": condition_name, "run": i, "best_epoch": result["best_epoch"], "best_score": result["best_score"], **hp.to_dict()})
        if best is None or result["best_score"] > best["best_score"]:
            best = {"run": i, "hparams": hp, **result}

    test_pred = best["test_pred"]
    label_col_for_eval = eval_label_col or hard_label_col
    y_eval = test_df[label_col_for_eval].astype(int).to_numpy() if label_col_for_eval else test_pred["hard"]
    soft_target = None
    if soft_label_cols and len(soft_label_cols) == num_labels and set(soft_label_cols).issubset(test_df.columns):
        soft_target = test_df[soft_label_cols].astype(float).to_numpy()
    metrics = classification_metrics(y_eval, test_pred["pred"], test_pred["probs"], class_names, soft_target=soft_target)
    if include_ordinal:
        metrics.update(ordinal_metrics(y_eval, test_pred["pred"]))
    ci_df = bootstrap_classification_metric_rows(
        y_eval,
        test_pred["pred"],
        test_pred["probs"],
        class_names=class_names,
        soft_target=soft_target,
        include_ordinal=include_ordinal,
        n_boot=settings.bootstrap_n,
        seed=settings.bootstrap_seed,
    )
    metrics.update(
        {
            "condition": condition_name,
            "target_source": target_source,
            "text_col": text_col,
            "loss_type": loss_type,
            "model_name": settings.model_name,
            "max_length": settings.max_length,
            "bootstrap_n": settings.bootstrap_n,
            "best_run": best["run"],
            "best_epoch": best["best_epoch"],
            **best["hparams"].to_dict(),
        }
    )

    pred_df = pd.DataFrame(
        {
            "source_row": test_pred["source_row"],
            "condition": condition_name,
            "target_source": target_source,
            "y_true": y_eval,
            "y_pred": test_pred["pred"],
        }
    )
    if "human_emotion_3class_id" in test_df.columns:
        pred_df["y_true_human_emotion"] = test_df["human_emotion_3class_id"].astype(int).to_numpy()
    for i, name in enumerate(class_names):
        pred_df[f"prob_{name.lower()}"] = test_pred["probs"][:, i]
    if soft_target is not None and soft_target.shape[1] == len(class_names):
        for i, name in enumerate(class_names):
            pred_df[f"llm_target_{name.lower()}"] = soft_target[:, i]

    pd.DataFrame(runs).to_csv(output_dir / f"{condition_name}_hp_search.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / f"{condition_name}_metrics.csv", index=False)
    ci_df.insert(0, "condition", condition_name)
    ci_df.insert(1, "target_source", target_source)
    ci_df.to_csv(output_dir / f"{condition_name}_metrics_ci.csv", index=False)
    pred_df.to_csv(output_dir / f"{condition_name}_predictions_test.csv", index=False)
    best["history"].to_csv(output_dir / f"{condition_name}_history.csv", index=False)
    torch.save(best["model"].state_dict(), output_dir / f"{condition_name}_best_model.pt")
    with (output_dir / f"{condition_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / f"{condition_name}_run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"condition": condition_name, "settings": asdict(settings), "hparams": best["hparams"].to_dict()}, f, indent=2)
    return metrics, pred_df


class MultiTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, task_cols: dict[str, str], mask_cols: dict[str, str], max_length: int):
        self.df = df.reset_index(drop=True)
        self.source_rows = self.df["source_row"].astype(int).tolist()
        self.enc = tokenizer(self.df[text_col].astype(str).tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        self.labels = {task: torch.tensor(self.df[col].astype(int).to_numpy(), dtype=torch.long) for task, col in task_cols.items()}
        self.masks = {
            task: torch.tensor(self.df[mask_cols[task]].astype(float).to_numpy(), dtype=torch.float32)
            for task in mask_cols
        }

    def __len__(self):
        return len(self.source_rows)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.enc.items()}
        item["labels"] = {task: values[idx] for task, values in self.labels.items()}
        item["masks"] = {task: values[idx] for task, values in self.masks.items()}
        item["source_row"] = self.source_rows[idx]
        return item


def multitask_collate(batch):
    keys = [key for key in batch[0].keys() if key not in {"labels", "masks", "source_row"}]
    out = {key: torch.stack([item[key] for item in batch]) for key in keys}
    tasks = batch[0]["labels"].keys()
    out["labels"] = {task: torch.stack([item["labels"][task] for item in batch]) for task in tasks}
    out["masks"] = {task: torch.stack([item["masks"][task] for item in batch]) for task in batch[0]["masks"].keys()}
    out["source_row"] = [item["source_row"] for item in batch]
    return out


class EncoderMTL(nn.Module):
    def __init__(self, model_name: str, task_num_classes: dict[str, int], dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = getattr(self.encoder.config, "hidden_size")
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, n) for task, n in task_num_classes.items()})
        self.log_vars = nn.ParameterDict({task: nn.Parameter(torch.zeros(())) for task in task_num_classes})

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {task: head(pooled) for task, head in self.heads.items()}


def make_mtl_loaders(df: pd.DataFrame, tokenizer, text_col: str, task_cols: dict[str, str], mask_cols: dict[str, str], settings: TrainSettings):
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    datasets = [
        MultiTaskDataset(part, tokenizer, text_col, task_cols, mask_cols, settings.max_length)
        for part in (train_df, val_df, test_df)
    ]
    loaders = [
        DataLoader(datasets[0], batch_size=settings.batch_size, shuffle=True, collate_fn=multitask_collate, num_workers=settings.num_workers),
        DataLoader(datasets[1], batch_size=settings.batch_size, shuffle=False, collate_fn=multitask_collate, num_workers=settings.num_workers),
        DataLoader(datasets[2], batch_size=settings.batch_size, shuffle=False, collate_fn=multitask_collate, num_workers=settings.num_workers),
    ]
    return loaders, (train_df, val_df, test_df)


def masked_ce_loss(logits, labels, mask, weight=None):
    losses = nn.CrossEntropyLoss(weight=weight, reduction="none")(logits, labels)
    denom = mask.sum().clamp_min(1.0)
    return (losses * mask).sum() / denom


def mtl_predict(model, loader, device):
    model.eval()
    out = {"source_row": []}
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            logits = model(batch["input_ids"], batch.get("attention_mask"), batch.get("token_type_ids"))
            out["source_row"].extend(batch["source_row"])
            for task, task_logits in logits.items():
                probs = torch.softmax(task_logits, dim=-1).detach().cpu().numpy()
                pred = probs.argmax(axis=1)
                labels = batch["labels"][task].detach().cpu().numpy()
                out.setdefault(task, {"prob": [], "pred": [], "true": []})
                out[task]["prob"].append(probs)
                out[task]["pred"].append(pred)
                out[task]["true"].append(labels)
    for task in [key for key in out.keys() if key != "source_row"]:
        out[task]["prob"] = np.vstack(out[task]["prob"])
        out[task]["pred"] = np.concatenate(out[task]["pred"])
        out[task]["true"] = np.concatenate(out[task]["true"])
    return out


def evaluate_mtl_predictions(pred, primary_tasks: list[str], task_class_names: dict[str, list[str]]) -> dict:
    records = {}
    for task in [key for key in pred.keys() if key != "source_row"]:
        names = task_class_names[task]
        base = classification_metrics(pred[task]["true"], pred[task]["pred"], pred[task]["prob"], names)
        if task in primary_tasks:
            base.update(ordinal_metrics(pred[task]["true"], pred[task]["pred"]))
        records[task] = base
    return records


def train_mtl_once(
    df: pd.DataFrame,
    condition_name: str,
    task_cols: dict[str, str],
    mask_cols: dict[str, str],
    task_num_classes: dict[str, int],
    primary_tasks: list[str],
    settings: TrainSettings,
    hparams: HParams,
    role_precision_cap: float | None = None,
):
    set_seed(settings.seed)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
    (train_loader, val_loader, test_loader), (train_df, val_df, test_df) = make_mtl_loaders(
        df, tokenizer, "text_regular", task_cols, mask_cols, settings
    )
    model = EncoderMTL(settings.model_name, task_num_classes, hparams.dropout).to(device)
    weights = {}
    for task, col in task_cols.items():
        if task in mask_cols:
            labels = train_df.loc[train_df[mask_cols[task]].astype(bool), col].to_numpy()
        else:
            labels = train_df[col].to_numpy()
        weights[task] = class_weights(labels, task_num_classes[task], device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    total_steps = max(1, len(train_loader) * hparams.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * settings.warmup_ratio), total_steps)

    task_class_names = {}
    for task in task_cols:
        if task == "speaker_role":
            task_class_names[task] = ROLE_LABELS
        elif task == "cancer_type":
            task_class_names[task] = CANCER_LABELS
        else:
            task_class_names[task] = ["0", "1", "2", "3"]

    best_state, best_score, best_epoch = None, -math.inf, 0
    bad_epochs, history = 0, []
    for epoch in range(1, hparams.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{condition_name} epoch {epoch}", leave=False):
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch.get("attention_mask"), batch.get("token_type_ids"))
            total_loss = 0.0
            task_losses = {}
            for task, task_logits in logits.items():
                labels = batch["labels"][task]
                mask = batch["masks"].get(task, torch.ones_like(labels, dtype=torch.float32))
                raw_loss = masked_ce_loss(task_logits, labels, mask, weight=weights[task])
                precision = torch.exp(-model.log_vars[task])
                if role_precision_cap is not None and task == "speaker_role":
                    precision = torch.clamp(precision, max=float(role_precision_cap))
                weighted = 0.5 * precision * raw_loss + 0.5 * model.log_vars[task]
                total_loss = total_loss + weighted
                task_losses[task] = float(raw_loss.detach().cpu())
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(float(total_loss.detach().cpu()))

        val_pred = mtl_predict(model, val_loader, device)
        val_metrics = evaluate_mtl_predictions(val_pred, primary_tasks, task_class_names)
        primary_score = float(np.mean([val_metrics[task]["weighted_f1"] for task in primary_tasks]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "primary_mean_weighted_f1": primary_score})
        if primary_score > best_score:
            best_score = primary_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= settings.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred = mtl_predict(model, test_loader, device)
    test_metrics = evaluate_mtl_predictions(test_pred, primary_tasks, task_class_names)
    return {
        "model": model,
        "history": pd.DataFrame(history),
        "best_score": best_score,
        "best_epoch": best_epoch,
        "test_pred": test_pred,
        "test_metrics": test_metrics,
        "task_class_names": task_class_names,
    }


def save_mtl_outputs(output_dir: Path, condition_name: str, result: dict, primary_tasks: list[str], hparams: HParams, settings: TrainSettings):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task, metrics in result["test_metrics"].items():
        records.append(
            {
                "condition": condition_name,
                "task": task,
                "is_primary": task in primary_tasks,
                "bootstrap_n": settings.bootstrap_n,
                **metrics,
            }
        )
    pd.DataFrame(records).to_csv(output_dir / f"{condition_name}_metrics.csv", index=False)
    result["history"].to_csv(output_dir / f"{condition_name}_history.csv", index=False)

    pred_rows = pd.DataFrame({"source_row": result["test_pred"]["source_row"]})
    ci_rows = []
    for task, payload in result["test_pred"].items():
        if task == "source_row":
            continue
        task_ci = bootstrap_classification_metric_rows(
            payload["true"],
            payload["pred"],
            payload["prob"],
            class_names=result["task_class_names"][task],
            include_ordinal=task in primary_tasks,
            n_boot=settings.bootstrap_n,
            seed=settings.bootstrap_seed,
        )
        task_ci.insert(0, "condition", condition_name)
        task_ci.insert(1, "task", task)
        task_ci.insert(2, "is_primary", task in primary_tasks)
        ci_rows.append(task_ci)
        pred_rows[f"{task}_true"] = payload["true"]
        pred_rows[f"{task}_pred"] = payload["pred"]
        for k in range(payload["prob"].shape[1]):
            pred_rows[f"{task}_prob_{k}"] = payload["prob"][:, k]
    pred_rows.to_csv(output_dir / f"{condition_name}_predictions_test.csv", index=False)
    pd.concat(ci_rows, ignore_index=True).to_csv(output_dir / f"{condition_name}_metrics_ci.csv", index=False)

    weight_rows = []
    for task, param in result["model"].log_vars.items():
        log_sigma2 = float(param.detach().cpu())
        precision = math.exp(-log_sigma2)
        weight_rows.append(
            {
                "condition": condition_name,
                "task": task,
                "log_sigma2": log_sigma2,
                "precision_1_over_sigma2": precision,
                "loss_weight_1_over_2sigma2": 0.5 * precision,
            }
        )
    pd.DataFrame(weight_rows).to_csv(output_dir / f"{condition_name}_kendall_task_weights.csv", index=False)
    torch.save(result["model"].state_dict(), output_dir / f"{condition_name}_best_model.pt")
    with (output_dir / f"{condition_name}_run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"condition": condition_name, "settings": asdict(settings), "hparams": hparams.to_dict()}, f, indent=2)


def run_mtl_search(
    df: pd.DataFrame,
    output_dir: Path,
    condition_name: str,
    include_aux: bool,
    settings: TrainSettings,
    role_precision_cap: float | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_tasks = HEOR_SUBSCALES
    task_cols = {task: task for task in HEOR_SUBSCALES}
    mask_cols = {}
    task_num_classes = {task: 4 for task in HEOR_SUBSCALES}
    if include_aux:
        task_cols.update({"speaker_role": "speaker_role_id", "cancer_type": "cancer_type_id"})
        mask_cols.update({"speaker_role": "speaker_role_loss_mask", "cancer_type": "cancer_type_loss_mask"})
        task_num_classes.update({"speaker_role": len(ROLE_LABELS), "cancer_type": len(CANCER_LABELS)})

    best = None
    rows = []
    for i, hp in enumerate(sample_hparams(settings.n_iter, settings.seed), start=1):
        print(f"\n[{condition_name}] search {i}/{settings.n_iter}: {hp}")
        result = train_mtl_once(df, condition_name, task_cols, mask_cols, task_num_classes, primary_tasks, settings, hp, role_precision_cap)
        rows.append({"condition": condition_name, "run": i, "best_epoch": result["best_epoch"], "best_score": result["best_score"], **hp.to_dict()})
        if best is None or result["best_score"] > best["best_score"]:
            best = {"run": i, "hparams": hp, **result}
    pd.DataFrame(rows).to_csv(Path(output_dir) / f"{condition_name}_hp_search.csv", index=False)
    save_mtl_outputs(output_dir, condition_name, best, primary_tasks, best["hparams"], settings)
    return best


class HEORR0Dataset(Dataset):
    """Dataset matching the R0 HEOR notebook task layout."""

    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, heor_mode: str, include_aux: bool, settings: TrainSettings):
        self.df = df.reset_index(drop=True)
        self.source_rows = self.df["source_row"].astype(int).tolist()
        self.heor_mode = heor_mode
        self.include_aux = include_aux
        self.enc = tokenizer(
            self.df[text_col].astype(str).tolist(),
            padding=True,
            truncation=True,
            max_length=settings.max_length,
            return_tensors="pt",
        )
        if heor_mode == "composite":
            self.total_score = torch.tensor(self.df["ai_total_score_normed"].astype(float).to_numpy(), dtype=torch.float32)
            self.high_need = torch.tensor(self.df["ai_high_need_flag"].astype(float).to_numpy(), dtype=torch.float32)
        elif heor_mode == "subscales":
            self.subscales = {
                task: torch.tensor(self.df[task].astype(int).to_numpy(), dtype=torch.long)
                for task in HEOR_SUBSCALES
            }
        else:
            raise ValueError(f"Unknown heor_mode: {heor_mode}")

        if include_aux:
            self.role = torch.tensor(self.df["speaker_role_id"].astype(int).to_numpy(), dtype=torch.long)
            self.cancer = torch.tensor(self.df["cancer_type_id"].astype(int).to_numpy(), dtype=torch.long)
            self.role_mask = torch.tensor(self.df["speaker_role_loss_mask"].astype(float).to_numpy(), dtype=torch.float32)
            self.cancer_mask = torch.tensor(self.df["cancer_type_loss_mask"].astype(float).to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.source_rows)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.enc.items()}
        item["source_row"] = self.source_rows[idx]
        if self.heor_mode == "composite":
            item["total_score"] = self.total_score[idx]
            item["high_need"] = self.high_need[idx]
        else:
            for task, values in self.subscales.items():
                item[task] = values[idx]
        if self.include_aux:
            item["speaker_role"] = self.role[idx]
            item["cancer_type"] = self.cancer[idx]
            item["speaker_role_mask"] = self.role_mask[idx]
            item["cancer_type_mask"] = self.cancer_mask[idx]
        return item


class EncoderHEORR0(nn.Module):
    """R0-compatible HEOR MTL head layout with Kendall task weights."""

    def __init__(self, model_name: str, heor_mode: str, include_aux: bool, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = getattr(self.encoder.config, "hidden_size")
        self.dropout = nn.Dropout(dropout)
        self.heor_mode = heor_mode
        self.include_aux = include_aux
        self.task_names = []

        if heor_mode == "composite":
            self.total_head = nn.Linear(hidden, 1)
            self.high_need_head = nn.Linear(hidden, 1)
            self.task_names.extend(["total_score", "high_need"])
        elif heor_mode == "subscales":
            self.subscale_heads = nn.ModuleDict({task: nn.Linear(hidden, 4) for task in HEOR_SUBSCALES})
            self.task_names.extend(HEOR_SUBSCALES)
        else:
            raise ValueError(f"Unknown heor_mode: {heor_mode}")

        if include_aux:
            self.role_head = nn.Linear(hidden, len(ROLE_LABELS))
            self.cancer_head = nn.Linear(hidden, len(CANCER_LABELS))
            self.task_names.extend(["speaker_role", "cancer_type"])

        self.log_vars = nn.ParameterDict({task: nn.Parameter(torch.zeros(())) for task in self.task_names})

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        out = {}
        if self.heor_mode == "composite":
            out["total_score"] = self.total_head(pooled).squeeze(-1)
            out["high_need"] = self.high_need_head(pooled).squeeze(-1)
        else:
            for task, head in self.subscale_heads.items():
                out[task] = head(pooled)

        if self.include_aux:
            out["speaker_role"] = self.role_head(pooled)
            out["cancer_type"] = self.cancer_head(pooled)
        return out


def make_heor_r0_loaders(df: pd.DataFrame, tokenizer, heor_mode: str, include_aux: bool, settings: TrainSettings):
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    datasets = [
        HEORR0Dataset(part, tokenizer, "text_regular", heor_mode, include_aux, settings)
        for part in (train_df, val_df, test_df)
    ]
    loaders = [
        DataLoader(datasets[0], batch_size=settings.batch_size, shuffle=True, num_workers=settings.num_workers),
        DataLoader(datasets[1], batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers),
        DataLoader(datasets[2], batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers),
    ]
    return loaders, (train_df, val_df, test_df)


def _heor_r0_class_weights(train_df: pd.DataFrame, heor_mode: str, include_aux: bool, use_aux_loss_masks: bool, device: torch.device):
    weights = {}
    if heor_mode == "subscales":
        for task in HEOR_SUBSCALES:
            weights[task] = class_weights(train_df[task].astype(int), 4, device)
    if include_aux:
        role_df = train_df
        cancer_df = train_df
        if use_aux_loss_masks:
            role_df = train_df[train_df["speaker_role_loss_mask"].astype(bool)]
            cancer_df = train_df[train_df["cancer_type_loss_mask"].astype(bool)]
        weights["speaker_role"] = class_weights(role_df["speaker_role_id"].astype(int), len(ROLE_LABELS), device)
        weights["cancer_type"] = class_weights(cancer_df["cancer_type_id"].astype(int), len(CANCER_LABELS), device)
    return weights


def _kendall_weighted_loss(model: EncoderHEORR0, task_losses: dict[str, torch.Tensor], role_precision_cap: float | None = None):
    total_loss = 0.0
    for task, raw_loss in task_losses.items():
        precision = torch.exp(-model.log_vars[task])
        if role_precision_cap is not None and task == "speaker_role":
            precision = torch.clamp(precision, max=float(role_precision_cap))
        total_loss = total_loss + 0.5 * precision * raw_loss + 0.5 * model.log_vars[task]
    return total_loss


def predict_heor_r0(model: EncoderHEORR0, loader, device: torch.device, heor_mode: str, include_aux: bool):
    model.eval()
    out = {"source_row": []}
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            logits = model(batch["input_ids"], batch.get("attention_mask"), batch.get("token_type_ids"))
            out["source_row"].extend(batch["source_row"])

            if heor_mode == "composite":
                total_pred = logits["total_score"].detach().cpu().numpy()
                total_true = batch["total_score"].detach().cpu().numpy()
                high_logits = logits["high_need"].detach().cpu().numpy()
                high_prob = 1.0 / (1.0 + np.exp(-high_logits))
                high_pred = (high_logits > 0).astype(int)
                high_true = batch["high_need"].detach().cpu().numpy().astype(int)
                out.setdefault("total_score", {"pred": [], "true": []})
                out["total_score"]["pred"].append(total_pred)
                out["total_score"]["true"].append(total_true)
                out.setdefault("high_need", {"prob": [], "pred": [], "true": []})
                out["high_need"]["prob"].append(np.column_stack([1.0 - high_prob, high_prob]))
                out["high_need"]["pred"].append(high_pred)
                out["high_need"]["true"].append(high_true)
            else:
                for task in HEOR_SUBSCALES:
                    probs = torch.softmax(logits[task], dim=-1).detach().cpu().numpy()
                    out.setdefault(task, {"prob": [], "pred": [], "true": []})
                    out[task]["prob"].append(probs)
                    out[task]["pred"].append(probs.argmax(axis=1))
                    out[task]["true"].append(batch[task].detach().cpu().numpy())

            if include_aux:
                for task, label_key in [("speaker_role", "speaker_role"), ("cancer_type", "cancer_type")]:
                    probs = torch.softmax(logits[task], dim=-1).detach().cpu().numpy()
                    out.setdefault(task, {"prob": [], "pred": [], "true": []})
                    out[task]["prob"].append(probs)
                    out[task]["pred"].append(probs.argmax(axis=1))
                    out[task]["true"].append(batch[label_key].detach().cpu().numpy())

    for task in [key for key in out if key != "source_row"]:
        if "prob" in out[task]:
            out[task]["prob"] = np.vstack(out[task]["prob"])
            out[task]["pred"] = np.concatenate(out[task]["pred"])
            out[task]["true"] = np.concatenate(out[task]["true"])
        else:
            out[task]["pred"] = np.concatenate(out[task]["pred"])
            out[task]["true"] = np.concatenate(out[task]["true"])
    return out


def evaluate_heor_r0_predictions(pred: dict, heor_mode: str, include_aux: bool) -> dict:
    records = {}
    if heor_mode == "composite":
        true_score = pred["total_score"]["true"] * 100.0
        pred_score = pred["total_score"]["pred"] * 100.0
        records["total_score"] = regression_metrics(true_score, pred_score)
        records["high_need"] = classification_metrics(
            pred["high_need"]["true"],
            pred["high_need"]["pred"],
            pred["high_need"]["prob"],
            ["Low Need", "High Need"],
        )
    else:
        for task in HEOR_SUBSCALES:
            base = classification_metrics(pred[task]["true"], pred[task]["pred"], pred[task]["prob"], ["0", "1", "2", "3"])
            base.update(ordinal_metrics(pred[task]["true"], pred[task]["pred"]))
            records[task] = base

    if include_aux:
        records["speaker_role"] = classification_metrics(
            pred["speaker_role"]["true"],
            pred["speaker_role"]["pred"],
            pred["speaker_role"]["prob"],
            ROLE_LABELS,
        )
        records["cancer_type"] = classification_metrics(
            pred["cancer_type"]["true"],
            pred["cancer_type"]["pred"],
            pred["cancer_type"]["prob"],
            CANCER_LABELS,
        )
    return records


def _heor_r0_selection_score(metrics: dict, heor_mode: str, include_aux: bool, primary_only_selection: bool) -> float:
    scores = []
    if heor_mode == "composite":
        scores.append(metrics["total_score"].get("r2", np.nan))
        scores.append(metrics["high_need"].get("weighted_f1", np.nan))
    else:
        scores.extend(metrics[task].get("weighted_f1", np.nan) for task in HEOR_SUBSCALES)
    if include_aux and not primary_only_selection:
        scores.append(metrics["speaker_role"].get("weighted_f1", np.nan))
        scores.append(metrics["cancer_type"].get("weighted_f1", np.nan))
    scores = [float(score) for score in scores if np.isfinite(score)]
    return float(np.mean(scores)) if scores else -math.inf


def train_heor_r0_once(
    df: pd.DataFrame,
    condition_name: str,
    heor_mode: str,
    include_aux: bool,
    settings: TrainSettings,
    hparams: HParams,
    use_aux_loss_masks: bool = True,
    primary_only_selection: bool = True,
    role_precision_cap: float | None = None,
):
    set_seed(settings.seed)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
    (train_loader, val_loader, test_loader), (train_df, val_df, test_df) = make_heor_r0_loaders(
        df, tokenizer, heor_mode, include_aux, settings
    )
    model = EncoderHEORR0(settings.model_name, heor_mode, include_aux, hparams.dropout).to(device)
    weights = _heor_r0_class_weights(train_df, heor_mode, include_aux, use_aux_loss_masks, device)

    base_params = [p for name, p in model.named_parameters() if "log_vars" not in name]
    logvar_params = [p for name, p in model.named_parameters() if "log_vars" in name]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": hparams.learning_rate, "weight_decay": hparams.weight_decay},
            {"params": logvar_params, "lr": 0.01, "weight_decay": 0.0},
        ]
    )
    total_steps = max(1, len(train_loader) * hparams.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * settings.warmup_ratio), total_steps)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    best_state, best_score, best_epoch = None, -math.inf, 0
    bad_epochs, history = 0, []

    for epoch in range(1, hparams.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{condition_name} epoch {epoch}", leave=False):
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["input_ids"], batch.get("attention_mask"), batch.get("token_type_ids"))
            task_losses = {}

            if heor_mode == "composite":
                task_losses["total_score"] = mse_loss(outputs["total_score"], batch["total_score"])
                task_losses["high_need"] = bce_loss(outputs["high_need"], batch["high_need"])
            else:
                for task in HEOR_SUBSCALES:
                    mask = torch.ones_like(batch[task], dtype=torch.float32)
                    task_losses[task] = masked_ce_loss(outputs[task], batch[task], mask, weight=weights[task])

            if include_aux:
                role_mask = batch["speaker_role_mask"] if use_aux_loss_masks else torch.ones_like(batch["speaker_role"], dtype=torch.float32)
                cancer_mask = batch["cancer_type_mask"] if use_aux_loss_masks else torch.ones_like(batch["cancer_type"], dtype=torch.float32)
                task_losses["speaker_role"] = masked_ce_loss(
                    outputs["speaker_role"], batch["speaker_role"], role_mask, weight=weights["speaker_role"]
                )
                task_losses["cancer_type"] = masked_ce_loss(
                    outputs["cancer_type"], batch["cancer_type"], cancer_mask, weight=weights["cancer_type"]
                )

            loss = _kendall_weighted_loss(model, task_losses, role_precision_cap)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        val_pred = predict_heor_r0(model, val_loader, device, heor_mode, include_aux)
        val_metrics = evaluate_heor_r0_predictions(val_pred, heor_mode, include_aux)
        score = _heor_r0_selection_score(val_metrics, heor_mode, include_aux, primary_only_selection)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_score": score})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= settings.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred = predict_heor_r0(model, test_loader, device, heor_mode, include_aux)
    test_metrics = evaluate_heor_r0_predictions(test_pred, heor_mode, include_aux)
    return {
        "model": model,
        "history": pd.DataFrame(history),
        "best_score": best_score,
        "best_epoch": best_epoch,
        "test_pred": test_pred,
        "test_metrics": test_metrics,
    }


def save_heor_r0_outputs(
    output_dir: Path,
    condition_name: str,
    result: dict,
    heor_mode: str,
    include_aux: bool,
    hparams: HParams,
    settings: TrainSettings,
    use_aux_loss_masks: bool,
    primary_only_selection: bool,
    role_precision_cap: float | None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = result["test_pred"]
    metrics_rows = []
    ci_frames = []
    pred_rows = pd.DataFrame({"source_row": pred["source_row"], "condition": condition_name})

    if heor_mode == "composite":
        true_score = pred["total_score"]["true"] * 100.0
        pred_score = pred["total_score"]["pred"] * 100.0
        total_metrics = regression_metrics(true_score, pred_score)
        metrics_rows.append({"condition": condition_name, "task": "total_score", "task_group": "primary_regression", "bootstrap_n": settings.bootstrap_n, **total_metrics})
        total_ci = bootstrap_regression_metric_rows(true_score, pred_score, n_boot=settings.bootstrap_n, seed=settings.bootstrap_seed)
        total_ci.insert(0, "condition", condition_name)
        total_ci.insert(1, "task", "total_score")
        ci_frames.append(total_ci)
        pred_rows["total_score_true_0_100"] = true_score
        pred_rows["total_score_pred_0_100"] = pred_score

        high_metrics = classification_metrics(
            pred["high_need"]["true"],
            pred["high_need"]["pred"],
            pred["high_need"]["prob"],
            ["Low Need", "High Need"],
        )
        metrics_rows.append({"condition": condition_name, "task": "high_need", "task_group": "primary_classification", "bootstrap_n": settings.bootstrap_n, **high_metrics})
        high_ci = bootstrap_classification_metric_rows(
            pred["high_need"]["true"],
            pred["high_need"]["pred"],
            pred["high_need"]["prob"],
            class_names=["Low Need", "High Need"],
            n_boot=settings.bootstrap_n,
            seed=settings.bootstrap_seed,
        )
        high_ci.insert(0, "condition", condition_name)
        high_ci.insert(1, "task", "high_need")
        ci_frames.append(high_ci)
        pred_rows["high_need_true"] = pred["high_need"]["true"]
        pred_rows["high_need_pred"] = pred["high_need"]["pred"]
        pred_rows["high_need_prob_0"] = pred["high_need"]["prob"][:, 0]
        pred_rows["high_need_prob_1"] = pred["high_need"]["prob"][:, 1]
    else:
        for task in HEOR_SUBSCALES:
            task_metrics = classification_metrics(pred[task]["true"], pred[task]["pred"], pred[task]["prob"], ["0", "1", "2", "3"])
            task_metrics.update(ordinal_metrics(pred[task]["true"], pred[task]["pred"]))
            metrics_rows.append({"condition": condition_name, "task": task, "task_group": "primary_subscale", "bootstrap_n": settings.bootstrap_n, **task_metrics})
            task_ci = bootstrap_classification_metric_rows(
                pred[task]["true"],
                pred[task]["pred"],
                pred[task]["prob"],
                class_names=["0", "1", "2", "3"],
                include_ordinal=True,
                n_boot=settings.bootstrap_n,
                seed=settings.bootstrap_seed,
            )
            task_ci.insert(0, "condition", condition_name)
            task_ci.insert(1, "task", task)
            ci_frames.append(task_ci)
            pred_rows[f"{task}_true"] = pred[task]["true"]
            pred_rows[f"{task}_pred"] = pred[task]["pred"]
            for k in range(pred[task]["prob"].shape[1]):
                pred_rows[f"{task}_prob_{k}"] = pred[task]["prob"][:, k]

    if include_aux:
        for task, names, group in [
            ("speaker_role", ROLE_LABELS, "aux_role"),
            ("cancer_type", CANCER_LABELS, "aux_cancer"),
        ]:
            task_metrics = classification_metrics(pred[task]["true"], pred[task]["pred"], pred[task]["prob"], names)
            metrics_rows.append({"condition": condition_name, "task": task, "task_group": group, "bootstrap_n": settings.bootstrap_n, **task_metrics})
            task_ci = bootstrap_classification_metric_rows(
                pred[task]["true"],
                pred[task]["pred"],
                pred[task]["prob"],
                class_names=names,
                n_boot=settings.bootstrap_n,
                seed=settings.bootstrap_seed,
            )
            task_ci.insert(0, "condition", condition_name)
            task_ci.insert(1, "task", task)
            ci_frames.append(task_ci)
            pred_rows[f"{task}_true"] = pred[task]["true"]
            pred_rows[f"{task}_pred"] = pred[task]["pred"]
            for k in range(pred[task]["prob"].shape[1]):
                pred_rows[f"{task}_prob_{k}"] = pred[task]["prob"][:, k]

    pd.DataFrame(metrics_rows).to_csv(output_dir / f"{condition_name}_metrics.csv", index=False)
    pd.concat(ci_frames, ignore_index=True).to_csv(output_dir / f"{condition_name}_metrics_ci.csv", index=False)
    pred_rows.to_csv(output_dir / f"{condition_name}_predictions_test.csv", index=False)
    result["history"].to_csv(output_dir / f"{condition_name}_history.csv", index=False)
    torch.save(result["model"].state_dict(), output_dir / f"{condition_name}_best_model.pt")

    weight_rows = []
    for task, param in result["model"].log_vars.items():
        log_sigma2 = float(param.detach().cpu())
        precision = math.exp(-log_sigma2)
        weight_rows.append(
            {
                "condition": condition_name,
                "task": task,
                "log_sigma2": log_sigma2,
                "precision_1_over_sigma2": precision,
                "loss_weight_1_over_2sigma2": 0.5 * precision,
            }
        )
    pd.DataFrame(weight_rows).to_csv(output_dir / f"{condition_name}_kendall_task_weights.csv", index=False)
    with (output_dir / f"{condition_name}_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "condition": condition_name,
                "heor_mode": heor_mode,
                "include_aux": include_aux,
                "settings": asdict(settings),
                "hparams": hparams.to_dict(),
                "use_aux_loss_masks": use_aux_loss_masks,
                "primary_only_selection": primary_only_selection,
                "role_precision_cap": role_precision_cap,
            },
            f,
            indent=2,
        )


def run_heor_r0_mtl_search(
    df: pd.DataFrame,
    output_dir: Path,
    condition_name: str,
    heor_mode: str,
    include_aux: bool,
    settings: TrainSettings,
    use_aux_loss_masks: bool = True,
    primary_only_selection: bool = True,
    role_precision_cap: float | None = None,
):
    """Run the R0 HEOR condition family with R1 tuning/evaluation defaults."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = None
    rows = []
    for i, hp in enumerate(sample_hparams(settings.n_iter, settings.seed), start=1):
        print(f"\n[{condition_name}] search {i}/{settings.n_iter}: {hp}")
        result = train_heor_r0_once(
            df,
            condition_name,
            heor_mode,
            include_aux,
            settings,
            hp,
            use_aux_loss_masks=use_aux_loss_masks,
            primary_only_selection=primary_only_selection,
            role_precision_cap=role_precision_cap,
        )
        rows.append({"condition": condition_name, "run": i, "best_epoch": result["best_epoch"], "best_score": result["best_score"], **hp.to_dict()})
        if best is None or result["best_score"] > best["best_score"]:
            best = {"run": i, "hparams": hp, **result}
    pd.DataFrame(rows).to_csv(output_dir / f"{condition_name}_hp_search.csv", index=False)
    save_heor_r0_outputs(
        output_dir,
        condition_name,
        best,
        heor_mode,
        include_aux,
        best["hparams"],
        settings,
        use_aux_loss_masks,
        primary_only_selection,
        role_precision_cap,
    )
    return best
