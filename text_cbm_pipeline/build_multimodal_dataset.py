#!/usr/bin/env python3
"""Build an ADCeleb multimodal sequence dataset from partial transcripts."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

DEFAULT_DATA_ROOT = Path("~/AD_predict/ADCELEB-main/data_link").expanduser()
DEFAULT_INVENTORY_CSV = Path("~/AD_predict/text_cbm_pipeline/transcribed_inventory.csv").expanduser()
DEFAULT_OUTPUT_PKL = Path("~/AD_predict/text_cbm_pipeline/adceleb_multimodal_features.pkl").expanduser()

CONCEPT_LABELS: List[str] = [
    "emotional frustration",
    "simple vocabulary",
    "grammatical errors",
    "short fragmented sentences",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align partial transcripts with acoustic features and text pseudo-concepts."
    )
    parser.add_argument("--inventory-csv", type=Path, default=DEFAULT_INVENTORY_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-pkl", type=Path, default=DEFAULT_OUTPUT_PKL)
    parser.add_argument("--model-name", default="facebook/bart-large-mnli")
    parser.add_argument("--device", type=int, default=0, help="Use -1 to force CPU.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--hypothesis-template",
        default="The text snippet shows {}.",
    )
    parser.add_argument("--default-acoustic-dim", type=int, default=512)
    return parser.parse_args()


def truncate_text(text: str, max_chars: int) -> str:
    clean = " ".join(str(text).split())
    if max_chars > 0 and len(clean) > max_chars:
        return clean[:max_chars].strip()
    return clean


def extract_video_id_from_link(link: str) -> str | None:
    if not isinstance(link, str) or not link.strip():
        return None
    parsed = urlparse(link)
    query = parse_qs(parsed.query)
    if "v" in query and query["v"]:
        return query["v"][0]
    stripped = link.rstrip("/")
    if "watch?v=" in stripped:
        return stripped.split("watch?v=")[-1]
    return stripped.rsplit("/", 1)[-1]


def build_metadata_order_maps(data_root: Path) -> Dict[str, Dict[str, int]]:
    order_maps: Dict[str, Dict[str, int]] = {}
    for group_name in ("AD", "CN"):
        group_dir = data_root / group_name
        if not group_dir.exists():
            continue
        for speaker_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            metadata_path = speaker_dir / "metadata.xlsx"
            if not metadata_path.exists():
                continue
            try:
                metadata_df = pd.read_excel(metadata_path)
            except Exception as exc:
                print(f"Warning: failed to load {metadata_path}: {exc}")
                continue

            video_order: Dict[str, int] = {}
            if "link" in metadata_df.columns:
                for idx, link in enumerate(metadata_df["link"].tolist()):
                    video_id = extract_video_id_from_link(link)
                    if video_id and video_id not in video_order:
                        video_order[video_id] = idx
            order_maps[speaker_dir.name] = video_order
    return order_maps


def candidate_feature_paths(txt_path: Path) -> List[Path]:
    # Slice file names are floating timestamps (for example, "8.05602716468591.txt"),
    # so using Path.with_suffix twice would incorrectly collapse the basename to "8".
    base_name = txt_path.stem
    return [txt_path.parent / f"{base_name}.npy", txt_path.parent / f"{base_name}.csv"]


def load_numeric_csv(path: Path) -> np.ndarray | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        coerced = df.apply(pd.to_numeric, errors="coerce")
        numeric_df = coerced.dropna(axis=1, how="all")
    if numeric_df.empty:
        return None
    return numeric_df.to_numpy(dtype=np.float32).reshape(-1)


def load_acoustic_feature(txt_path: Path) -> tuple[np.ndarray | None, str | None]:
    for feature_path in candidate_feature_paths(txt_path):
        if not feature_path.exists():
            continue
        try:
            if feature_path.suffix == ".npy":
                vector = np.load(feature_path, allow_pickle=False)
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
            else:
                vector = load_numeric_csv(feature_path)
            if vector is not None and vector.size > 0:
                return vector, str(feature_path)
        except Exception as exc:
            print(f"Warning: failed to load acoustic feature {feature_path}: {exc}")
    return None, None


def infer_acoustic_dim(inventory_df: pd.DataFrame, default_dim: int) -> int:
    for txt_path_str in inventory_df["TXT_Path"].tolist():
        vector, _ = load_acoustic_feature(Path(txt_path_str))
        if vector is not None and vector.size > 0:
            return int(vector.size)
    return int(default_dim)


class ZeroShotConceptScorer:
    def __init__(self, model_name: str, device: int):
        self.model_name = model_name
        self.requested_device = device
        self.pipeline = None
        self.resolved_device = None

    def _build_pipeline(self, device: int):
        from transformers import pipeline

        if device >= 0 and not torch.cuda.is_available():
            print("Warning: requested CUDA device but no GPU is available. Falling back to CPU.")
            device = -1

        self.pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device,
        )
        self.resolved_device = device

    def _ensure_pipeline(self, device: int):
        if self.pipeline is None or self.resolved_device != device:
            self._build_pipeline(device)

    def score_batch(
        self,
        texts: Sequence[str],
        candidate_labels: Sequence[str],
        hypothesis_template: str,
    ) -> List[Dict[str, float]]:
        attempted_devices = [self.requested_device]
        if self.requested_device >= 0:
            attempted_devices.append(-1)

        last_error: Exception | None = None
        for device in attempted_devices:
            try:
                self._ensure_pipeline(device)
                outputs = self.pipeline(
                    list(texts),
                    candidate_labels=list(candidate_labels),
                    multi_label=True,
                    truncation=True,
                    batch_size=max(1, len(texts)),
                    hypothesis_template=hypothesis_template,
                )
                if isinstance(outputs, dict):
                    outputs = [outputs]
                batch_scores: List[Dict[str, float]] = []
                for output in outputs:
                    score_lookup = {
                        label: float(score)
                        for label, score in zip(output["labels"], output["scores"])
                    }
                    batch_scores.append(
                        {label: score_lookup.get(label, 0.0) for label in candidate_labels}
                    )
                return batch_scores
            except RuntimeError as exc:
                last_error = exc
                if "out of memory" in str(exc).lower() and device >= 0:
                    print("Warning: CUDA OOM during zero-shot inference. Falling back to CPU.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.pipeline = None
                    self.resolved_device = None
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Zero-shot concept scorer failed without a specific exception.")


def attach_temporal_order(df: pd.DataFrame, metadata_maps: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    df = df.copy()
    df["Metadata_Order"] = df.apply(
        lambda row: metadata_maps.get(row["Speaker_ID"], {}).get(row["Video_ID"], 10**9),
        axis=1,
    )
    df["Segment_Order"] = pd.to_numeric(df["Segment_Float"], errors="coerce").fillna(10**9)
    df.sort_values(
        by=["Speaker_ID", "Metadata_Order", "Video_ID", "Slice_Speaker_ID", "Segment_Order", "Segment_ID"],
        inplace=True,
        kind="stable",
    )
    df["Temporal_Order"] = df.groupby("Speaker_ID").cumcount()
    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    args = parse_args()

    inventory_df = pd.read_csv(args.inventory_csv)
    if args.max_rows is not None:
        inventory_df = inventory_df.head(args.max_rows).copy()

    metadata_maps = build_metadata_order_maps(args.data_root)
    inventory_df = attach_temporal_order(inventory_df, metadata_maps)

    acoustic_dim = infer_acoustic_dim(inventory_df, args.default_acoustic_dim)
    print(f"Inferred acoustic feature dimension: {acoustic_dim}")

    scorer = ZeroShotConceptScorer(model_name=args.model_name, device=args.device)

    records: List[Dict[str, object]] = []
    for start_idx in tqdm(range(0, len(inventory_df), args.batch_size), desc="Building multimodal rows"):
        batch_df = inventory_df.iloc[start_idx : start_idx + args.batch_size].copy()
        batch_texts = [truncate_text(text, args.max_chars) for text in batch_df["Text"].tolist()]
        batch_scores = scorer.score_batch(
            texts=batch_texts,
            candidate_labels=CONCEPT_LABELS,
            hypothesis_template=args.hypothesis_template,
        )

        for (_, row), concept_score_map in zip(batch_df.iterrows(), batch_scores):
            txt_path = Path(row["TXT_Path"])
            acoustic_vector, acoustic_path = load_acoustic_feature(txt_path)
            if acoustic_vector is None:
                acoustic_vector = np.zeros(acoustic_dim, dtype=np.float32)
                acoustic_missing = True
            else:
                acoustic_missing = False
                if acoustic_vector.size != acoustic_dim:
                    adjusted = np.zeros(acoustic_dim, dtype=np.float32)
                    width = min(acoustic_dim, acoustic_vector.size)
                    adjusted[:width] = acoustic_vector[:width]
                    acoustic_vector = adjusted

            concept_vector = np.asarray(
                [concept_score_map[label] for label in CONCEPT_LABELS],
                dtype=np.float32,
            )

            records.append(
                {
                    "Speaker_ID": row["Speaker_ID"],
                    "Label": int(row["Label"]),
                    "Video_ID": row["Video_ID"],
                    "Slice_Speaker_ID": row["Slice_Speaker_ID"],
                    "Segment_ID": row["Segment_ID"],
                    "Temporal_Order": int(row["Temporal_Order"]),
                    "Metadata_Order": int(row["Metadata_Order"]),
                    "Segment_Order": float(row["Segment_Order"]),
                    "Acoustic_Vector": acoustic_vector,
                    "Text_Concepts_Vector": concept_vector,
                    "Transcript_Text": row["Text"],
                    "TXT_Path": row["TXT_Path"],
                    "Acoustic_Path": acoustic_path,
                    "Acoustic_Missing": acoustic_missing,
                }
            )

    feature_df = pd.DataFrame(records)
    payload = {
        "dataframe": feature_df,
        "concept_names": CONCEPT_LABELS,
        "acoustic_dim": acoustic_dim,
        "inventory_csv": str(args.inventory_csv),
        "data_root": str(args.data_root),
    }

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as handle:
        pickle.dump(payload, handle)

    print(f"Saved multimodal dataset to: {args.output_pkl}")
    print(f"Rows saved: {len(feature_df)}")
    print(f"Unique speakers: {feature_df['Speaker_ID'].nunique()}")


if __name__ == "__main__":
    main()
