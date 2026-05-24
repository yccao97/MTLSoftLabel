"""Shared data utilities for the R1 manuscript revision notebooks."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


EMOTION_LABELS_3 = ["Negative", "Neutral", "Positive"]
EMOTION_PROB_COLS_3 = ["llm_prob_negative", "llm_prob_neutral", "llm_prob_positive"]
RAW_LLM_PROB_COLS_4 = [
    "ai_prob_very_negative",
    "ai_prob_negative",
    "ai_prob_neutral",
    "ai_prob_positive",
]

HEOR_SUBSCALES = [
    "ai_benefit_score",
    "ai_harm_score",
    "ai_cost_burden",
    "ai_treatment_burden",
    "ai_life_disruption",
    "ai_uncertainty_conflict",
    "ai_support_coping",
]

HUMAN_HEOR_SUBSCALES = [
    "benefit_score",
    "harm_score",
    "cost_burden",
    "treatment_burden",
    "life_disruption",
    "uncertainty_conflict",
    "support_coping",
]

ROLE_LABELS = ["PATIENT", "CAREGIVER", "UNCLEAR"]
CANCER_LABELS = ["BRAIN", "COLON", "LIVER", "LEUKEMIA", "LUNG", "OTHER", "UNKNOWN"]


def derive_high_need_flag(
    df: pd.DataFrame,
    emotion_col: str,
    benefit_col: str,
    harm_col: str,
    cost_col: str,
    treatment_col: str,
    life_col: str,
    uncertainty_col: str,
    support_col: str,
) -> pd.Series:
    """Apply the prompt-specified deterministic high-need rule.

    The rule mirrors Multimedia Appendix 1: compute the composite score as
    50 + 10*benefit + 6*support - 10*harm - 8*cost - 8*treatment
    - 8*life - 8*uncertainty, then flag high need when the total is <=25,
    emotion is VERY_NEGATIVE, harm is 3, or uncertainty is 3.
    """
    benefit = pd.to_numeric(df[benefit_col], errors="coerce").fillna(0)
    harm = pd.to_numeric(df[harm_col], errors="coerce").fillna(0)
    cost = pd.to_numeric(df[cost_col], errors="coerce").fillna(0)
    treatment = pd.to_numeric(df[treatment_col], errors="coerce").fillna(0)
    life = pd.to_numeric(df[life_col], errors="coerce").fillna(0)
    uncertainty = pd.to_numeric(df[uncertainty_col], errors="coerce").fillna(0)
    support = pd.to_numeric(df[support_col], errors="coerce").fillna(0)
    total = 50 + 10 * benefit + 6 * support - 10 * harm - 8 * cost - 8 * treatment - 8 * life - 8 * uncertainty
    total = total.clip(0, 100).round()
    emotion = df[emotion_col].fillna("").astype(str).str.upper().str.replace(" ", "_", regex=False)
    return ((total <= 25) | emotion.eq("VERY_NEGATIVE") | harm.eq(3) | uncertainty.eq(3)).astype(int)


def clean_text(text: object) -> str:
    """Match the R0 notebook text cleaning before tokenization."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", " urltoken ", text)
    text = re.sub(r"@\w+", " usertoken ", text)
    text = re.sub(r"#(\w+)", r" hashtag_\1 ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def collapse_human_emotion_value(predicted: object = None, intensity: object = None) -> int:
    """Collapse original 4-class human emotion labels to Negative/Neutral/Positive ids."""
    label = str(predicted or "").strip().lower().replace("_", " ")
    value = str(intensity or "").strip()
    if label in {"very negative", "negative"} or value in {"-2", "-1", "-2.0", "-1.0"}:
        return 0
    if label == "neutral" or value in {"0", "0.0"}:
        return 1
    if label == "positive" or value in {"1", "1.0"}:
        return 2
    raise ValueError(f"Unknown human emotion label: predicted={predicted!r}, intensity={intensity!r}")


def collapse_human_emotion(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row: collapse_human_emotion_value(row.get("predicted"), row.get("intensity")), axis=1)


def collapse_llm_probs(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in RAW_LLM_PROB_COLS_4 if col not in df.columns]
    if missing:
        raise KeyError(f"Missing LLM probability columns: {missing}")
    out = pd.DataFrame(
        {
            "llm_prob_negative": df["ai_prob_very_negative"].astype(float)
            + df["ai_prob_negative"].astype(float),
            "llm_prob_neutral": df["ai_prob_neutral"].astype(float),
            "llm_prob_positive": df["ai_prob_positive"].astype(float),
        },
        index=df.index,
    )
    sums = out.sum(axis=1).replace(0, np.nan)
    out = out.div(sums, axis=0)
    out = out.fillna(1.0 / 3.0)
    return out


def add_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    role = df.get("ai_speaker_role", pd.Series(["UNCLEAR"] * len(df), index=df.index))
    cancer = df.get("ai_cancer_type", pd.Series(["UNKNOWN"] * len(df), index=df.index))
    role = role.fillna("UNCLEAR").astype(str).str.upper()
    cancer = cancer.fillna("UNKNOWN").astype(str).str.upper()
    df["clean_text"] = df["posts"].apply(clean_text)
    df["text_regular"] = df["clean_text"]
    df["text_augmented"] = "ROLE_" + role + " CANCER_" + cancer + " " + df["clean_text"]
    return df


def encode_categorical(series: pd.Series, labels: Iterable[str]) -> pd.Series:
    label_list = list(labels)
    mapping = {name: idx for idx, name in enumerate(label_list)}
    values = series.fillna(label_list[-1]).astype(str).str.upper()
    return values.map(lambda value: mapping.get(value, mapping[label_list[-1]])).astype(int)


def prepare_annotation_frame(annotation_path: Path, split_path: Path) -> pd.DataFrame:
    """Load annotation CSV and merge the canonical shared split assignments."""
    annotation_path = Path(annotation_path)
    split_path = Path(split_path)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    needed_cols = [
        "posts",
        "predicted",
        "intensity",
        "ai_speaker_role",
        "ai_cancer_type",
        "ai_total_score_0_100",
        "ai_high_need_flag",
        *RAW_LLM_PROB_COLS_4,
        *HEOR_SUBSCALES,
    ]
    df = pd.read_csv(annotation_path, usecols=lambda col: col in needed_cols)
    missing_annotation_cols = [col for col in needed_cols if col not in df.columns]
    if missing_annotation_cols:
        raise KeyError(f"Annotation file missing required columns: {missing_annotation_cols}")
    df = df.dropna(subset=["posts"]).reset_index(drop=False).rename(columns={"index": "source_row"})
    split_df = pd.read_csv(split_path)
    split_cols = [col for col in ["source_row", "split", "stratify_key"] if col in split_df.columns]
    split_df = split_df[split_cols].copy()
    merged = df.merge(split_df, on="source_row", how="inner", validate="one_to_one")
    if len(merged) != len(split_df):
        raise ValueError(f"Split merge mismatch: merged={len(merged)} split={len(split_df)}")

    merged = add_text_columns(merged)
    merged["human_emotion_3class_id"] = collapse_human_emotion(merged).astype(int)
    merged["human_emotion_3class"] = merged["human_emotion_3class_id"].map(
        dict(enumerate(EMOTION_LABELS_3))
    )
    llm_probs = collapse_llm_probs(merged)
    for col in EMOTION_PROB_COLS_3:
        merged[col] = llm_probs[col].astype(float)
    merged["llm_argmax_3class_id"] = llm_probs.to_numpy().argmax(axis=1).astype(int)
    merged["llm_argmax_3class"] = merged["llm_argmax_3class_id"].map(dict(enumerate(EMOTION_LABELS_3)))

    for col in HEOR_SUBSCALES:
        if col not in merged.columns:
            raise KeyError(f"Missing HEOR subscale column: {col}")
        merged[col] = pd.to_numeric(merged[col], errors="coerce").round().astype(int).clip(0, 3)

    merged["ai_total_score_0_100"] = pd.to_numeric(merged["ai_total_score_0_100"], errors="coerce").clip(0, 100)
    merged = merged.dropna(subset=["ai_total_score_0_100"]).copy()
    merged["ai_total_score_normed"] = merged["ai_total_score_0_100"].astype(float) / 100.0
    merged["ai_high_need_flag"] = (
        merged["ai_high_need_flag"]
        .map({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0, 1: 1, 0: 0})
        .fillna(0)
        .astype(int)
    )

    merged["speaker_role_id"] = encode_categorical(merged["ai_speaker_role"], ROLE_LABELS)
    merged["cancer_type_id"] = encode_categorical(merged["ai_cancer_type"], CANCER_LABELS)
    merged["speaker_role_loss_mask"] = (merged["ai_speaker_role"].astype(str).str.upper() != "UNCLEAR").astype(int)
    merged["cancer_type_loss_mask"] = (merged["ai_cancer_type"].astype(str).str.upper() != "UNKNOWN").astype(int)
    return merged


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if min(len(train_df), len(val_df), len(test_df)) <= 0:
        raise ValueError("Expected non-empty train/val/test splits.")
    return train_df, val_df, test_df


def write_run_manifest(output_dir: Path, payload: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(payload, dtype="object").to_json(output_dir / "run_manifest.json", indent=2)
