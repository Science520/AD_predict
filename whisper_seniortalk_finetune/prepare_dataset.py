#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
import traceback
import unicodedata
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import librosa
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import DatasetDict, Features, Sequence, Value, load_dataset
from huggingface_hub import snapshot_download
from transformers import WhisperProcessor

try:
    import torchaudio
except Exception:
    torchaudio = None


DEFAULT_WORKSPACE_ROOT = Path.home() / "AD_predict" / "whisper_seniortalk_finetune"
DEFAULT_PHYSICAL_DATA_ROOT = Path("/data/saisai/BAAI_SeniorTalk")
DEFAULT_MODEL_NAME = "openai/whisper-medium"
DEFAULT_REPO_ID = "BAAI/SeniorTalk"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
WHITESPACE_RE = re.compile(r"\s+")
GLOBAL_PROCESSOR: WhisperProcessor | None = None
GLOBAL_TEXT_COLUMN = "text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BAAI/SeniorTalk locally and prepare a Whisper-ready dataset."
    )
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--physical-data-root", type=Path, default=None)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-label-length", type=int, default=448)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--processor-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("prepare_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def configure_hf_environment(data_root: Path) -> dict[str, Path]:
    cache_root = data_root / "hf_cache"
    paths = {
        "hf_home": cache_root,
        "datasets_cache": cache_root / "datasets",
        "hub_cache": cache_root / "hub",
        "transformers_cache": cache_root / "transformers",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(paths["hf_home"])
    os.environ["HF_DATASETS_CACHE"] = str(paths["datasets_cache"])
    os.environ["HF_HUB_CACHE"] = str(paths["hub_cache"])
    os.environ["TRANSFORMERS_CACHE"] = str(paths["transformers_cache"])
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    return paths


def retry_call(
    fn: Callable[[], Any],
    description: str,
    logger: logging.Logger,
    max_attempts: int,
    initial_sleep_seconds: float,
) -> Any:
    sleep_seconds = max(initial_sleep_seconds, 1.0)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("%s: attempt %s/%s", description, attempt, max_attempts)
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = exc
            logger.warning("%s failed on attempt %s/%s: %s", description, attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(sleep_seconds)
                sleep_seconds *= 2
    assert last_error is not None
    raise last_error


def normalize_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", "" if text is None else str(text))
    normalized = normalized.replace("\u3000", " ")
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def detect_text_column(column_names: list[str]) -> str:
    preferred = ["text", "sentence", "transcript", "transcription", "label"]
    for candidate in preferred:
        if candidate in column_names:
            return candidate
    raise ValueError(f"Unable to detect transcript column from: {column_names}")


def detect_audio_column(column_names: list[str]) -> str:
    preferred = ["audio", "speech", "wav", "recording"]
    for candidate in preferred:
        if candidate in column_names:
            return candidate
    return "path"


def ensure_raw_snapshot(
    repo_id: str,
    raw_repo_dir: Path,
    logger: logging.Logger,
    max_attempts: int,
    sleep_seconds: float,
    force_download: bool,
) -> Path:
    sentence_dir = raw_repo_dir / "sentence_data"
    existing = sorted(sentence_dir.glob("*.parquet"))
    if existing and not force_download:
        logger.info("Found %s local parquet shards under %s", len(existing), sentence_dir)
        return raw_repo_dir

    raw_repo_dir.mkdir(parents=True, exist_ok=True)

    def do_download() -> str:
        return snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(raw_repo_dir),
            allow_patterns=[
                "README.md",
                "sentence_data/*.parquet",
                "SPKINFO.txt",
                "UTTERANCEINFO.txt",
            ],
            force_download=False,
            max_workers=4,
        )

    retry_call(
        fn=do_download,
        description=f"Downloading {repo_id} snapshot",
        logger=logger,
        max_attempts=max_attempts,
        initial_sleep_seconds=sleep_seconds,
    )
    downloaded = sorted(sentence_dir.glob("*.parquet"))
    if not downloaded:
        raise FileNotFoundError(f"No parquet shards were downloaded under {sentence_dir}")
    logger.info("Downloaded %s parquet shards into %s", len(downloaded), sentence_dir)
    return raw_repo_dir


def build_data_files(raw_repo_dir: Path) -> dict[str, list[str]]:
    sentence_dir = raw_repo_dir / "sentence_data"
    train_files = sorted(sentence_dir.glob("train*.parquet"))
    validation_files = sorted(sentence_dir.glob("dev*.parquet")) or sorted(
        sentence_dir.glob("validation*.parquet")
    )
    test_files = sorted(sentence_dir.glob("test*.parquet"))

    if not train_files or not validation_files or not test_files:
        raise FileNotFoundError(
            "Expected train/dev(or validation)/test parquet shards under "
            f"{sentence_dir}, found train={len(train_files)} validation={len(validation_files)} test={len(test_files)}"
        )

    return {
        "train": [str(path) for path in train_files],
        "validation": [str(path) for path in validation_files],
        "test": [str(path) for path in test_files],
    }


def arrow_type_to_feature(arrow_type: pa.DataType) -> Any:
    if pa.types.is_struct(arrow_type):
        return {field.name: arrow_type_to_feature(field.type) for field in arrow_type}
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return Sequence(feature=arrow_type_to_feature(arrow_type.value_type))
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return Value("string")
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return Value("binary")
    if pa.types.is_bool(arrow_type):
        return Value("bool")
    if pa.types.is_int8(arrow_type):
        return Value("int8")
    if pa.types.is_int16(arrow_type):
        return Value("int16")
    if pa.types.is_int32(arrow_type):
        return Value("int32")
    if pa.types.is_int64(arrow_type):
        return Value("int64")
    if pa.types.is_uint8(arrow_type):
        return Value("uint8")
    if pa.types.is_uint16(arrow_type):
        return Value("uint16")
    if pa.types.is_uint32(arrow_type):
        return Value("uint32")
    if pa.types.is_uint64(arrow_type):
        return Value("uint64")
    if pa.types.is_float16(arrow_type):
        return Value("float16")
    if pa.types.is_float32(arrow_type):
        return Value("float32")
    if pa.types.is_float64(arrow_type):
        return Value("float64")
    raise TypeError(f"Unsupported parquet field type: {arrow_type}")


def build_manual_features(data_files: dict[str, list[str]]) -> Features:
    sample_file = next(iter(data_files.values()))[0]
    schema = pq.read_schema(sample_file)
    feature_mapping = {field.name: arrow_type_to_feature(field.type) for field in schema}
    return Features(feature_mapping)


def load_raw_dataset(data_files: dict[str, list[str]], cache_dir: Path) -> DatasetDict:
    features = build_manual_features(data_files)
    return load_dataset("parquet", data_files=data_files, cache_dir=str(cache_dir), features=features)


def select_subset(dataset: DatasetDict, max_samples_per_split: int | None) -> DatasetDict:
    if max_samples_per_split is None:
        return dataset
    limited = DatasetDict()
    for split_name, split_dataset in dataset.items():
        limited[split_name] = split_dataset.select(range(min(len(split_dataset), max_samples_per_split)))
    return limited


def decode_audio_payload(audio_payload: Any) -> tuple[np.ndarray, int]:
    if isinstance(audio_payload, dict):
        audio_bytes = audio_payload.get("bytes")
        audio_path = audio_payload.get("path")
    else:
        audio_bytes = None
        audio_path = audio_payload

    decode_errors: list[str] = []

    if audio_bytes:
        try:
            array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            return np.asarray(array, dtype=np.float32), int(sample_rate)
        except Exception as exc:
            decode_errors.append(f"soundfile(bytes): {exc}")

        if torchaudio is not None:
            try:
                tensor, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
                array = tensor.numpy()
                if array.ndim > 1 and array.shape[0] == 1:
                    array = array[0]
                return np.asarray(array, dtype=np.float32), int(sample_rate)
            except Exception as exc:
                decode_errors.append(f"torchaudio(bytes): {exc}")

    if audio_path:
        try:
            array, sample_rate = sf.read(audio_path, dtype="float32")
            return np.asarray(array, dtype=np.float32), int(sample_rate)
        except Exception as exc:
            decode_errors.append(f"soundfile(path): {exc}")

        if torchaudio is not None:
            try:
                tensor, sample_rate = torchaudio.load(audio_path)
                array = tensor.numpy()
                if array.ndim > 1 and array.shape[0] == 1:
                    array = array[0]
                return np.asarray(array, dtype=np.float32), int(sample_rate)
            except Exception as exc:
                decode_errors.append(f"torchaudio(path): {exc}")

    raise RuntimeError("Unable to decode audio payload. " + " | ".join(decode_errors))


def preprocess_batch(example: dict[str, Any], max_label_length: int) -> dict[str, Any]:
    assert GLOBAL_PROCESSOR is not None

    array, sample_rate = decode_audio_payload(example["audio"])
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sample_rate != 16000:
        array = librosa.resample(array, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    input_features = GLOBAL_PROCESSOR.feature_extractor(
        array, sampling_rate=sample_rate
    ).input_features[0]

    normalized_text = normalize_text(example[GLOBAL_TEXT_COLUMN])
    labels = GLOBAL_PROCESSOR.tokenizer(
        normalized_text,
        truncation=True,
        max_length=max_label_length,
    ).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
        "text": normalized_text,
        "input_length_seconds": round(len(array) / float(sample_rate), 4),
    }


def save_manifest(
    manifest_path: Path,
    payload: dict[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    workspace_root = args.workspace_root.expanduser().resolve()
    physical_data_root = (
        args.physical_data_root.expanduser().resolve()
        if args.physical_data_root
        else (workspace_root / "data_link").expanduser().resolve()
    )

    logger = setup_logging(workspace_root / "logs" / "prepare_dataset.log")
    logger.info("Workspace root: %s", workspace_root)
    logger.info("Physical data root: %s", physical_data_root)

    paths = configure_hf_environment(physical_data_root)
    logger.info("HF cache root: %s", paths["hf_home"])

    raw_repo_dir = physical_data_root / "raw_repo" / "BAAI_SeniorTalk"
    processed_dir = physical_data_root / "processed" / "seniortalk_whisper_medium"
    manifest_path = physical_data_root / "manifests" / "prepare_dataset_manifest.json"

    if processed_dir.exists() and args.overwrite:
        logger.info("Removing existing processed dataset at %s", processed_dir)
        import shutil

        shutil.rmtree(processed_dir)

    if processed_dir.exists() and not args.overwrite:
        logger.info("Processed dataset already exists at %s", processed_dir)
        logger.info("Use --overwrite to rebuild it.")
        return 0

    try:
        ensure_raw_snapshot(
            repo_id=args.repo_id,
            raw_repo_dir=raw_repo_dir,
            logger=logger,
            max_attempts=args.download_retries,
            sleep_seconds=args.retry_sleep_seconds,
            force_download=args.force_download,
        )

        data_files = build_data_files(raw_repo_dir)
        logger.info(
            "Discovered parquet shards: train=%s validation=%s test=%s",
            len(data_files["train"]),
            len(data_files["validation"]),
            len(data_files["test"]),
        )

        raw_dataset = load_raw_dataset(data_files, cache_dir=physical_data_root / "hf_cache" / "datasets")
        raw_dataset = select_subset(raw_dataset, args.max_samples_per_split)

        sample_columns = raw_dataset["train"].column_names
        text_column = detect_text_column(sample_columns)
        audio_column = detect_audio_column(sample_columns)
        logger.info("Detected columns: audio=%s text=%s", audio_column, text_column)

        def load_processor() -> WhisperProcessor:
            return WhisperProcessor.from_pretrained(
                args.model_name,
                language=args.language,
                task=args.task,
            )

        global GLOBAL_PROCESSOR
        global GLOBAL_TEXT_COLUMN
        GLOBAL_TEXT_COLUMN = text_column
        GLOBAL_PROCESSOR = retry_call(
            fn=load_processor,
            description=f"Loading processor for {args.model_name}",
            logger=logger,
            max_attempts=args.processor_retries,
            initial_sleep_seconds=args.retry_sleep_seconds,
        )

        logger.info("Preparing Whisper features and labels")
        processed_dataset = raw_dataset
        if audio_column != "audio":
            processed_dataset = processed_dataset.rename_column(audio_column, "audio")
        if text_column != "text":
            processed_dataset = processed_dataset.rename_column(text_column, "source_text")
            GLOBAL_TEXT_COLUMN = "source_text"

        remove_columns = list(processed_dataset["train"].column_names)
        preprocess_fn = partial(preprocess_batch, max_label_length=args.max_label_length)
        processed_dataset = processed_dataset.map(
            preprocess_fn,
            remove_columns=remove_columns,
            num_proc=args.num_proc if args.num_proc > 1 else None,
            desc="Extracting input_features and labels",
        )

        processed_dir.parent.mkdir(parents=True, exist_ok=True)
        processed_dataset.save_to_disk(str(processed_dir))

        split_counts = {split_name: len(split_dataset) for split_name, split_dataset in processed_dataset.items()}
        manifest_payload = {
            "repo_id": args.repo_id,
            "model_name": args.model_name,
            "language": args.language,
            "task": args.task,
            "raw_repo_dir": str(raw_repo_dir),
            "processed_dir": str(processed_dir),
            "split_counts": split_counts,
            "data_files": data_files,
            "text_column": text_column,
            "audio_column": audio_column,
            "max_label_length": args.max_label_length,
        }
        save_manifest(manifest_path, manifest_payload)
        logger.info("Saved manifest to %s", manifest_path)
        logger.info("Prepared dataset saved to %s", processed_dir)
        logger.info("Split sizes: %s", split_counts)
        return 0
    except Exception as exc:
        logger.error("Dataset preparation failed: %s", exc)
        logger.error(traceback.format_exc())
        if "403" in str(exc) or "401" in str(exc):
            logger.error(
                "The dataset may require approved Hugging Face access. Run `huggingface-cli login` and verify access to %s.",
                args.repo_id,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
