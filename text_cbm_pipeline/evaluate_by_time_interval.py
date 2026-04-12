#!/usr/bin/env python3
"""Evaluate the multimodal PCBM by diagnosis-relative time intervals.

This script is designed to mirror the interval-based ADCeleb evaluation in the
Interspeech 2025 paper:
    - Interval -2: 6 to 10 years before diagnosis
    - Interval -1: 1 to 5 years before diagnosis

The implementation assumes that:
    1. `adceleb_multimodal_features.pkl` stores snippet-level multimodal rows.
    2. `speakers_info.csv` provides the target-speaker identity per video and the
       relative distance to diagnosis (`years_from_diagnosis` +
       `before_after_diagnosis`).
    3. `AD_demo.xlsx`, `CN_demo.xlsx`, and `speakers_pairs.xlsx` under
       `data_link/` provide the speaker-level diagnosis-year metadata.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

DEFAULT_INPUT_PKL = Path("~/AD_predict/text_cbm_pipeline/adceleb_multimodal_features.pkl").expanduser()
DEFAULT_DATA_ROOT = Path("~/AD_predict/ADCELEB-main/data_link").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/AD_predict/text_cbm_pipeline/time_interval_eval").expanduser()

INTERVAL_SPECS: Dict[str, Tuple[float, float]] = {
    "Interval -2": (-10.0, -6.0),
    "Interval -1": (-5.0, -1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ADCeleb multimodal PCBM by diagnosis-relative time intervals."
    )
    parser.add_argument("--input-pkl", type=Path, default=DEFAULT_INPUT_PKL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bottleneck-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--target-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only snippets whose Slice_Speaker_ID matches the target speaker in speakers_info.csv.",
    )
    parser.add_argument(
        "--eval-level",
        choices=["speaker", "row"],
        default="speaker",
        help="Aggregate predictions at the speaker level (recommended) or evaluate at the row level.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect metadata extraction and interval coverage; do not train models.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def resolve_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    column_map = {normalize_name(column): column for column in df.columns}
    for candidate in candidates:
        normalized = normalize_name(candidate)
        if normalized in column_map:
            return column_map[normalized]
    if required:
        raise KeyError(
            f"Could not resolve column from candidates={list(candidates)}. "
            f"Available columns: {df.columns.tolist()}"
        )
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_feature_dataframe(input_pkl: Path) -> tuple[pd.DataFrame, Dict[str, object]]:
    if not input_pkl.exists():
        raise FileNotFoundError(f"Input pickle does not exist: {input_pkl}")

    with input_pkl.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        df = payload["dataframe"]
        meta = {key: value for key, value in payload.items() if key != "dataframe"}
    else:
        df = payload
        meta = {}

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected the pickle payload to contain a pandas DataFrame.")

    required_columns = {
        "Speaker_ID",
        "Label",
        "Video_ID",
        "Slice_Speaker_ID",
        "Acoustic_Vector",
        "Text_Concepts_Vector",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Speaker_ID"] = df["Speaker_ID"].astype(str)
    df["Video_ID"] = df["Video_ID"].astype(str)
    df["Slice_Speaker_ID"] = df["Slice_Speaker_ID"].astype(str)
    df["Label"] = df["Label"].astype(int)
    return df, meta


def load_speaker_yod_metadata(data_root: Path) -> pd.DataFrame:
    ad_demo_path = data_root / "AD_demo.xlsx"
    cn_demo_path = data_root / "CN_demo.xlsx"
    pairs_path = data_root / "speakers_pairs.xlsx"

    for path in (ad_demo_path, cn_demo_path, pairs_path):
        if not path.exists():
            raise FileNotFoundError(f"Required speaker metadata file does not exist: {path}")

    ad_demo = pd.read_excel(ad_demo_path)
    cn_demo = pd.read_excel(cn_demo_path)
    pairs = pd.read_excel(pairs_path)

    ad_speaker_col = resolve_column(ad_demo, ["Speaker_ID", "speaker_id"])
    ad_yod_col = resolve_column(ad_demo, ["Year of Diagnosis", "year_of_diagnosis", "YOD"])

    cn_speaker_col = resolve_column(cn_demo, ["Speaker_ID", "speaker_id"])

    pairs_cn_col = resolve_column(pairs, ["Name", "speaker_id", "cn_speaker_id"])
    pairs_ad_col = resolve_column(
        pairs,
        ["Use as a control for", "matched_ad_speaker", "ad_speaker_id"],
    )

    ad_df = ad_demo[[ad_speaker_col, ad_yod_col]].copy()
    ad_df.columns = ["Speaker_ID", "YOD"]
    ad_df["Speaker_ID"] = ad_df["Speaker_ID"].astype(str)
    ad_df["YOD"] = pd.to_numeric(ad_df["YOD"], errors="coerce")
    ad_df["YOD_Source"] = "AD_demo.xlsx"
    ad_df["Matched_AD_Speaker"] = ad_df["Speaker_ID"]

    ad_yod_map = ad_df.set_index("Speaker_ID")["YOD"].to_dict()

    pairs_df = pairs[[pairs_cn_col, pairs_ad_col]].copy()
    pairs_df.columns = ["Speaker_ID", "Matched_AD_Speaker"]
    pairs_df["Speaker_ID"] = pairs_df["Speaker_ID"].astype(str)
    pairs_df["Matched_AD_Speaker"] = pairs_df["Matched_AD_Speaker"].astype(str)

    cn_df = cn_demo[[cn_speaker_col]].copy()
    cn_df.columns = ["Speaker_ID"]
    cn_df["Speaker_ID"] = cn_df["Speaker_ID"].astype(str)
    cn_df = cn_df.merge(pairs_df, on="Speaker_ID", how="left")
    cn_df["YOD"] = cn_df["Matched_AD_Speaker"].map(ad_yod_map)
    cn_df["YOD_Source"] = "speakers_pairs.xlsx -> AD_demo.xlsx"

    speaker_yod_df = pd.concat(
        [ad_df[["Speaker_ID", "YOD", "YOD_Source", "Matched_AD_Speaker"]], cn_df],
        ignore_index=True,
    )
    speaker_yod_df.drop_duplicates(subset=["Speaker_ID"], inplace=True)
    return speaker_yod_df


def compute_years_to_diagnosis(years_from_diagnosis: object, before_after: object) -> float:
    years = pd.to_numeric(years_from_diagnosis, errors="coerce")
    if pd.isna(years):
        return float("nan")

    direction = ""
    if not pd.isna(before_after):
        direction = str(before_after).strip().lower()

    if direction == "before":
        return -abs(float(years))
    if direction == "after":
        return abs(float(years))
    if float(years) == 0:
        return 0.0
    if float(years) < 0:
        return float(years)
    return float("nan")


def select_target_row(df: pd.DataFrame, speaker_col: str, years_col: str) -> pd.Series | None:
    if df.empty:
        return None
    ranked = df.copy()
    ranked["_score"] = ranked[speaker_col].notna().astype(int) + ranked[years_col].notna().astype(int)
    ranked.sort_values(by="_score", ascending=False, inplace=True, kind="stable")
    return ranked.iloc[0]


def load_video_temporal_metadata(data_root: Path, speaker_yod_df: pd.DataFrame) -> pd.DataFrame:
    yod_map = speaker_yod_df.set_index("Speaker_ID")["YOD"].to_dict()
    yod_source_map = speaker_yod_df.set_index("Speaker_ID")["YOD_Source"].to_dict()
    matched_ad_map = speaker_yod_df.set_index("Speaker_ID")["Matched_AD_Speaker"].to_dict()

    records: List[Dict[str, object]] = []
    for speakers_info_path in tqdm(
        sorted(data_root.rglob("speakers_info.csv")),
        desc="Loading speakers_info metadata",
    ):
        video_dir = speakers_info_path.parent
        speaker_dir = video_dir.parent
        speaker_id = speaker_dir.name
        group_name = speaker_dir.parent.name
        video_id = video_dir.name

        try:
            info_df = pd.read_csv(speakers_info_path)
        except Exception as exc:
            warnings.warn(f"Failed to read {speakers_info_path}: {exc}")
            continue

        try:
            speaker_col = resolve_column(info_df, ["speakers", "speaker"])
            status_col = resolve_column(info_df, ["status"])
            years_col = resolve_column(info_df, ["years_from_diagnosis", "year_from_diagnosis"])
            before_after_col = resolve_column(
                info_df, ["before_after_diagnosis", "before_or_after_diagnosis"]
            )
            context_col = resolve_column(info_df, ["video_context"], required=False)
        except KeyError as exc:
            warnings.warn(f"Skipping {speakers_info_path}: {exc}")
            continue

        target_rows = info_df[
            info_df[status_col].astype(str).str.strip().str.lower() == "target"
        ].copy()
        target_row = select_target_row(target_rows, speaker_col=speaker_col, years_col=years_col)

        target_slice_speaker = None
        years_from_diagnosis = float("nan")
        years_to_diagnosis = float("nan")
        before_after = None
        video_context = None

        if target_row is not None:
            target_slice_speaker = str(target_row[speaker_col])
            years_from_diagnosis = pd.to_numeric(target_row[years_col], errors="coerce")
            before_after = target_row[before_after_col]
            years_to_diagnosis = compute_years_to_diagnosis(years_from_diagnosis, before_after)
            if context_col is not None:
                video_context = target_row[context_col]

        yod = yod_map.get(speaker_id, float("nan"))
        snippet_year = (
            float(yod) + float(years_to_diagnosis)
            if not pd.isna(yod) and not pd.isna(years_to_diagnosis)
            else float("nan")
        )

        records.append(
            {
                "Group": group_name,
                "Speaker_ID": speaker_id,
                "Video_ID": video_id,
                "Target_Slice_Speaker": target_slice_speaker,
                "Years_From_Diagnosis": years_from_diagnosis,
                "Before_After_Diagnosis": before_after,
                "Years_To_Diagnosis": years_to_diagnosis,
                "Video_Context": video_context,
                "YOD": yod,
                "YOD_Source": yod_source_map.get(speaker_id),
                "Matched_AD_Speaker": matched_ad_map.get(speaker_id),
                "Snippet_Year": snippet_year,
                "Speakers_Info_Path": str(speakers_info_path),
            }
        )

    if not records:
        raise RuntimeError("No speakers_info.csv files were successfully parsed.")

    video_meta_df = pd.DataFrame(records)
    video_meta_df.drop_duplicates(subset=["Speaker_ID", "Video_ID"], inplace=True)
    return video_meta_df


def enrich_multimodal_rows(
    feature_df: pd.DataFrame,
    video_meta_df: pd.DataFrame,
    target_only: bool,
    max_speakers: int | None,
) -> pd.DataFrame:
    df = feature_df.merge(
        video_meta_df,
        on=["Speaker_ID", "Video_ID"],
        how="left",
        validate="many_to_one",
    )

    df["Is_Target_Speaker"] = (
        df["Slice_Speaker_ID"].astype(str) == df["Target_Slice_Speaker"].astype(str)
    )

    if target_only:
        df = df[df["Is_Target_Speaker"]].copy()

    if max_speakers is not None:
        speaker_table = df[["Speaker_ID", "Label"]].drop_duplicates().sort_values(["Label", "Speaker_ID"])
        buckets = {
            label: group["Speaker_ID"].tolist()
            for label, group in speaker_table.groupby("Label", sort=True)
        }
        keep_speakers: List[str] = []
        while len(keep_speakers) < max_speakers and any(buckets.values()):
            for label in sorted(buckets):
                if buckets[label] and len(keep_speakers) < max_speakers:
                    keep_speakers.append(buckets[label].pop(0))
        df = df[df["Speaker_ID"].isin(keep_speakers)].copy()

    df["Interval"] = None
    for interval_name, (lower, upper) in INTERVAL_SPECS.items():
        mask = df["Years_To_Diagnosis"].between(lower, upper, inclusive="both")
        df.loc[mask, "Interval"] = interval_name

    return df


def print_enrichment_summary(df: pd.DataFrame, target_only: bool) -> None:
    print("=" * 88)
    print("Metadata Enrichment Summary")
    print("=" * 88)
    print(f"Rows after enrichment: {len(df)}")
    print(f"Unique speakers: {df['Speaker_ID'].nunique()}")
    print(f"Target-only filter enabled: {target_only}")
    print(f"Rows with non-null YOD: {int(df['YOD'].notna().sum())}")
    print(f"Rows with non-null Years_To_Diagnosis: {int(df['Years_To_Diagnosis'].notna().sum())}")
    print(f"Rows marked as target speaker: {int(df['Is_Target_Speaker'].sum())}")
    print()

    summary_rows = []
    for interval_name in INTERVAL_SPECS:
        subset = df[df["Interval"] == interval_name]
        speaker_counts = subset[["Speaker_ID", "Label"]].drop_duplicates()["Label"].value_counts().to_dict()
        summary_rows.append(
            {
                "Interval": interval_name,
                "Rows": len(subset),
                "Speakers": subset["Speaker_ID"].nunique(),
                "AD_Speakers": speaker_counts.get(1, 0),
                "CN_Speakers": speaker_counts.get(0, 0),
                "Years_Min": subset["Years_To_Diagnosis"].min() if not subset.empty else np.nan,
                "Years_Max": subset["Years_To_Diagnosis"].max() if not subset.empty else np.nan,
            }
        )

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print()

    preview_cols = [
        "Speaker_ID",
        "Label",
        "Video_ID",
        "Slice_Speaker_ID",
        "Target_Slice_Speaker",
        "Years_To_Diagnosis",
        "YOD",
        "Snippet_Year",
        "Interval",
    ]
    preview_df = df[preview_cols].dropna(subset=["Years_To_Diagnosis"]).head(10)
    if not preview_df.empty:
        print("Example rows:")
        print(preview_df.to_string(index=False))
        print()


def vectorize_multimodal_features(df: pd.DataFrame) -> np.ndarray:
    acoustic = np.stack(
        [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in df["Acoustic_Vector"]]
    )
    text_concepts = np.stack(
        [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in df["Text_Concepts_Vector"]]
    )
    return np.concatenate([acoustic, text_concepts], axis=1).astype(np.float32)


def standardize_train_test(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


class BottleneckClassifier(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.bottleneck = nn.Linear(input_dim, bottleneck_dim)
        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck = torch.sigmoid(self.bottleneck(x))
        logits = self.classifier(bottleneck).squeeze(-1)
        return logits, bottleneck


def make_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y.astype(np.float32)).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> BottleneckClassifier:
    model = BottleneckClassifier(
        input_dim=train_x.shape[1],
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    positive_count = max(1, int(train_y.sum()))
    negative_count = max(1, int(len(train_y) - positive_count))
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_dataloader(train_x, train_y, batch_size=args.batch_size, shuffle=True)

    best_state = None
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch_x, batch_y in progress:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item()) * batch_x.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / max(1, len(train_loader.dataset))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_probabilities(model: BottleneckClassifier, x: np.ndarray, device: torch.device) -> np.ndarray:
    loader = make_dataloader(x, np.zeros(len(x), dtype=np.float32), batch_size=2048, shuffle=False)
    probs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            logits, _ = model(batch_x)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "SENS": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "SPEC": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "TP": float(tp),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
    }


def build_speaker_folds(
    interval_df: pd.DataFrame,
    requested_splits: int,
    seed: int,
) -> tuple[list[tuple[int, set[str], set[str]]], pd.DataFrame]:
    speaker_df = interval_df[["Speaker_ID", "Label"]].drop_duplicates().copy()
    duplicate_labels = speaker_df.groupby("Speaker_ID")["Label"].nunique()
    if (duplicate_labels > 1).any():
        bad = duplicate_labels[duplicate_labels > 1].index.tolist()[:5]
        raise ValueError(f"Found speakers with inconsistent labels: {bad}")

    label_counts = speaker_df["Label"].value_counts()
    effective_splits = min(requested_splits, len(speaker_df), int(label_counts.min()))
    if effective_splits < 2:
        raise RuntimeError(
            "Not enough speakers per class to build at least 2 stratified folds "
            f"(speaker counts by label: {label_counts.to_dict()})."
        )
    if effective_splits < requested_splits:
        warnings.warn(
            f"Reducing n_splits from {requested_splits} to {effective_splits} "
            f"because the interval subset does not have enough speakers per class."
        )

    splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    folds: list[tuple[int, set[str], set[str]]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(speaker_df["Speaker_ID"], speaker_df["Label"]),
        start=1,
    ):
        train_speakers = set(speaker_df.iloc[train_idx]["Speaker_ID"].astype(str).tolist())
        test_speakers = set(speaker_df.iloc[test_idx]["Speaker_ID"].astype(str).tolist())
        folds.append((fold_idx, train_speakers, test_speakers))
    return folds, speaker_df


def aggregate_predictions(
    interval_df: pd.DataFrame,
    test_mask: np.ndarray,
    probs: np.ndarray,
    eval_level: str,
    interval_name: str,
    fold_idx: int,
) -> pd.DataFrame:
    results = interval_df.loc[test_mask, ["Speaker_ID", "Label", "Video_ID", "Years_To_Diagnosis"]].copy()
    results["Pred_Prob"] = probs
    results["Pred_Label"] = (results["Pred_Prob"] >= 0.5).astype(int)
    results["Interval"] = interval_name
    results["Fold"] = fold_idx

    if eval_level == "speaker":
        aggregated = (
            results.groupby("Speaker_ID", as_index=False)
            .agg(
                True_Label=("Label", "first"),
                Pred_Prob=("Pred_Prob", "mean"),
                Num_Snippets=("Pred_Prob", "size"),
                Num_Videos=("Video_ID", "nunique"),
                Mean_Years_To_Diagnosis=("Years_To_Diagnosis", "mean"),
            )
            .copy()
        )
        aggregated["Pred_Label"] = (aggregated["Pred_Prob"] >= 0.5).astype(int)
        aggregated["Interval"] = interval_name
        aggregated["Fold"] = fold_idx
        return aggregated

    results.rename(columns={"Label": "True_Label"}, inplace=True)
    results["Num_Snippets"] = 1
    results["Num_Videos"] = 1
    results["Mean_Years_To_Diagnosis"] = results["Years_To_Diagnosis"]
    return results


def evaluate_interval(
    interval_name: str,
    interval_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 88)
    print(f"Evaluating {interval_name}")
    print("=" * 88)
    print(f"Rows: {len(interval_df)}")
    print(f"Speakers: {interval_df['Speaker_ID'].nunique()}")
    print(f"Label distribution: {interval_df[['Speaker_ID', 'Label']].drop_duplicates()['Label'].value_counts().to_dict()}")

    interval_x = vectorize_multimodal_features(interval_df)
    interval_y = interval_df["Label"].to_numpy(dtype=np.int64)

    folds, _ = build_speaker_folds(interval_df, requested_splits=args.n_splits, seed=args.seed)
    fold_metrics: List[Dict[str, float]] = []
    fold_predictions: List[pd.DataFrame] = []

    for fold_idx, train_speakers, test_speakers in folds:
        train_mask = interval_df["Speaker_ID"].isin(train_speakers).to_numpy()
        test_mask = interval_df["Speaker_ID"].isin(test_speakers).to_numpy()

        train_x = interval_x[train_mask]
        test_x = interval_x[test_mask]
        train_y = interval_y[train_mask]

        train_x, test_x = standardize_train_test(train_x, test_x)

        try:
            model = train_model(train_x=train_x, train_y=train_y, args=args, device=device)
            test_probs = predict_probabilities(model=model, x=test_x, device=device)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.type == "cuda":
                warnings.warn(
                    f"CUDA OOM in {interval_name} fold {fold_idx}; retrying this fold on CPU."
                )
                torch.cuda.empty_cache()
                cpu_device = torch.device("cpu")
                model = train_model(train_x=train_x, train_y=train_y, args=args, device=cpu_device)
                test_probs = predict_probabilities(model=model, x=test_x, device=cpu_device)
            else:
                raise

        prediction_df = aggregate_predictions(
            interval_df=interval_df,
            test_mask=test_mask,
            probs=test_probs,
            eval_level=args.eval_level,
            interval_name=interval_name,
            fold_idx=fold_idx,
        )
        metrics = compute_binary_metrics(
            y_true=prediction_df["True_Label"].to_numpy(dtype=np.int64),
            y_prob=prediction_df["Pred_Prob"].to_numpy(dtype=float),
        )
        metrics.update(
            {
                "Interval": interval_name,
                "Fold": fold_idx,
                "Eval_Level": args.eval_level,
                "Train_Rows": int(train_mask.sum()),
                "Test_Rows": int(test_mask.sum()),
                "Train_Speakers": len(train_speakers),
                "Test_Speakers": len(test_speakers),
            }
        )
        fold_metrics.append(metrics)
        fold_predictions.append(prediction_df)

        print(
            f"Fold {fold_idx}: "
            f"ACC={metrics['ACC']:.4f} "
            f"F1={metrics['F1']:.4f} "
            f"AUC={metrics['AUC']:.4f} "
            f"SENS={metrics['SENS']:.4f} "
            f"SPEC={metrics['SPEC']:.4f}"
        )

    return pd.DataFrame(fold_metrics), pd.concat(fold_predictions, ignore_index=True)


def summarize_metrics(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []
    metric_names = ["ACC", "F1", "AUC", "SENS", "SPEC"]

    for interval_name, group_df in fold_metrics_df.groupby("Interval", sort=False):
        row: Dict[str, object] = {"Interval": interval_name}
        for metric in metric_names:
            row[f"{metric}_Mean"] = float(group_df[metric].mean())
            row[f"{metric}_Std"] = float(group_df[metric].std(ddof=0))
            row[metric] = f"{row[f'{metric}_Mean']:.3f} ± {row[f'{metric}_Std']:.3f}"
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def print_summary_table(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("No interval summary available.")
        return
    display_cols = ["Interval", "ACC", "F1", "AUC", "SENS", "SPEC"]
    print("=" * 88)
    print("Paper-Style Interval Summary")
    print("=" * 88)
    print(summary_df[display_cols].to_string(index=False))
    print()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    feature_df, payload_meta = load_feature_dataframe(args.input_pkl)
    speaker_yod_df = load_speaker_yod_metadata(args.data_root)
    video_meta_df = load_video_temporal_metadata(args.data_root, speaker_yod_df)

    enriched_df = enrich_multimodal_rows(
        feature_df=feature_df,
        video_meta_df=video_meta_df,
        target_only=args.target_only,
        max_speakers=args.max_speakers,
    )

    print(f"Loaded multimodal rows from: {args.input_pkl}")
    if payload_meta:
        print(f"Payload metadata keys: {list(payload_meta.keys())}")
    print_enrichment_summary(enriched_df, target_only=args.target_only)

    if args.inspect_only:
        return

    interval_dfs = {
        interval_name: enriched_df[enriched_df["Interval"] == interval_name].copy()
        for interval_name in INTERVAL_SPECS
    }

    all_fold_metrics: List[pd.DataFrame] = []
    all_predictions: List[pd.DataFrame] = []

    for interval_name, interval_df in interval_dfs.items():
        if interval_df.empty:
            warnings.warn(f"{interval_name} is empty after filtering. Skipping.")
            continue
        fold_metrics_df, predictions_df = evaluate_interval(
            interval_name=interval_name,
            interval_df=interval_df,
            args=args,
            device=device,
        )
        all_fold_metrics.append(fold_metrics_df)
        all_predictions.append(predictions_df)

    if not all_fold_metrics:
        raise RuntimeError("No interval produced evaluation results.")

    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = summarize_metrics(fold_metrics_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_path = args.output_dir / "time_interval_fold_metrics.csv"
    predictions_path = args.output_dir / "time_interval_predictions.csv"
    summary_path = args.output_dir / "time_interval_summary.csv"

    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print_summary_table(summary_df)
    print(f"Saved fold metrics to: {fold_metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
