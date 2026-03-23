#!/usr/bin/env python3
"""Check the live transcription log and build an inventory of transcribed slices.

This script is designed for the ADCeleb partial-transcription workflow:
    1. Inspect the tail of the running transcribe log.
    2. Scan data_link/AD and data_link/CN for already-finished slice .txt files.
    3. Save an immediate, training-ready CSV inventory for downstream scripts.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List

import pandas as pd
from tqdm.auto import tqdm

DEFAULT_LOG_PATH = Path("~/AD_predict/ADCELEB-main/transcribe.log").expanduser()
DEFAULT_DATA_ROOT = Path("~/AD_predict/ADCELEB-main/data_link").expanduser()
DEFAULT_OUTPUT_CSV = Path("~/AD_predict/text_cbm_pipeline/transcribed_inventory.csv").expanduser()

LOG_ERROR_PATTERN = re.compile(r"Traceback|\] error ", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check transcription log health and aggregate finished slice transcripts."
    )
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--tail-lines", type=int, default=50)
    return parser.parse_args()


def tail_lines(path: Path, n_lines: int) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Log file does not exist: {path}")
    buffer: Deque[str] = deque(maxlen=n_lines)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            buffer.append(line.rstrip("\n"))
    return list(buffer)


def summarize_log(log_lines: List[str]) -> None:
    print("=" * 80)
    print(f"Transcription Log Health Check (last {len(log_lines)} lines)")
    print("=" * 80)

    if not log_lines:
        print("Log tail is empty.")
        return

    error_lines = [line for line in log_lines if LOG_ERROR_PATTERN.search(line)]
    start_lines = [line for line in log_lines if "] start " in line]
    file_start_lines = [line for line in log_lines if "] file_start " in line]
    progress_lines = [line for line in log_lines if "] progress " in line]

    print(f"Detected {len(error_lines)} error-like lines in the tail.")
    if error_lines:
        print("Most recent error-like lines:")
        for line in error_lines[-5:]:
            print(f"  {line}")
    else:
        print("No Traceback / explicit error lines found in the recent tail.")

    if start_lines:
        print(f"Most recent start line: {start_lines[-1]}")
    if file_start_lines:
        print(f"Most recent file_start line: {file_start_lines[-1]}")
    if progress_lines:
        # ETA values in the log may be inaccurate, so we display the line but do not interpret ETA.
        print(f"Most recent progress line: {progress_lines[-1]}")


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_transcribed_slice_paths(data_root: Path) -> Iterable[Path]:
    for group_name in ("AD", "CN"):
        group_dir = data_root / group_name
        if not group_dir.exists():
            continue
        for root, _, files in os.walk(group_dir, followlinks=True):
            parent_name = Path(root).name
            if not parent_name.startswith("SPEAKER_"):
                continue
            for file_name in sorted(files):
                if not file_name.endswith(".txt"):
                    continue
                yield Path(root) / file_name


def parse_inventory_record(txt_path: Path, data_root: Path) -> Dict[str, object]:
    rel_parts = txt_path.relative_to(data_root).parts
    if len(rel_parts) < 5:
        raise ValueError(f"Unexpected transcript path layout: {txt_path}")

    group_name, speaker_id, video_id, slice_speaker_id, file_name = rel_parts[:5]
    segment_id = txt_path.stem
    transcript = clean_text(txt_path.read_text(encoding="utf-8", errors="ignore"))
    label = 1 if group_name == "AD" else 0

    return {
        "Group": group_name,
        "Label": label,
        "Speaker_ID": speaker_id,
        "Video_ID": video_id,
        "Slice_Speaker_ID": slice_speaker_id,
        "Segment_ID": segment_id,
        "Segment_Float": safe_float(segment_id),
        "Text": transcript,
        "Text_Length": len(transcript),
        "TXT_Path": str(txt_path),
        "Audio_Path": str(txt_path.with_suffix(".wav")),
    }


def build_inventory(data_root: Path) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    txt_paths = list(iter_transcribed_slice_paths(data_root))
    for txt_path in tqdm(txt_paths, desc="Scanning transcribed slices", unit="file"):
        try:
            record = parse_inventory_record(txt_path, data_root)
            if record["Text"]:
                records.append(record)
        except Exception as exc:
            print(f"Warning: failed to parse {txt_path}: {exc}")

    if not records:
        raise RuntimeError(
            "No transcribed slice .txt files were found under data_link/AD or data_link/CN."
        )

    df = pd.DataFrame(records)
    df.sort_values(
        by=["Group", "Speaker_ID", "Video_ID", "Slice_Speaker_ID", "Segment_Float", "Segment_ID"],
        inplace=True,
        kind="stable",
    )
    df.reset_index(drop=True, inplace=True)
    return df


def print_inventory_summary(df: pd.DataFrame) -> None:
    speaker_counts = df.groupby("Speaker_ID").size().sort_values(ascending=False)
    group_counts = Counter(df["Group"].tolist())

    print("=" * 80)
    print("Inventory Summary")
    print("=" * 80)
    print(f"Rows: {len(df)}")
    print(f"Unique speakers: {df['Speaker_ID'].nunique()}")
    print(f"Unique videos: {df['Video_ID'].nunique()}")
    print(f"Group counts: {dict(group_counts)}")
    print("Top speakers by transcribed slice count:")
    for speaker_id, count in speaker_counts.head(10).items():
        print(f"  {speaker_id}: {count}")


def main() -> None:
    args = parse_args()

    summarize_log(tail_lines(args.log_path, args.tail_lines))

    df = build_inventory(args.data_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print_inventory_summary(df)
    print(f"Saved inventory CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
