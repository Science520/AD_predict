#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import re
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor, set_seed


DEFAULT_WORKSPACE_ROOT = Path.home() / "AD_predict" / "whisper_seniortalk_finetune"
DEFAULT_MODEL_NAME = "openai/whisper-medium"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the zero-shot openai/whisper-medium baseline on processed SeniorTalk features."
    )
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--physical-data-root", type=Path, default=None)
    parser.add_argument("--processed-dataset-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--generation-max-length", type=int, default=448)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--auto-reduce-batch-size", action="store_true", default=True)
    parser.add_argument("--no-auto-reduce-batch-size", dest="auto_reduce_batch_size", action="store_false")
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_baseline")
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


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    fn,
    description: str,
    logger: logging.Logger,
    max_attempts: int,
    initial_sleep_seconds: float,
):
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


def to_char_tokens(text: str) -> str:
    return " ".join(list(normalize_text(text).replace(" ", "")))


def load_processor(
    model_name: str,
    language: str,
    task: str,
    logger: logging.Logger,
    max_attempts: int,
    sleep_seconds: float,
) -> WhisperProcessor:
    processor = retry_call(
        fn=lambda: WhisperProcessor.from_pretrained(model_name, language=language, task=task),
        description=f"Loading processor for {model_name}",
        logger=logger,
        max_attempts=max_attempts,
        initial_sleep_seconds=sleep_seconds,
    )
    if hasattr(processor.tokenizer, "set_prefix_tokens"):
        processor.tokenizer.set_prefix_tokens(language=language, task=task)
    return processor


def load_generation_config(
    model_name: str,
    language: str,
    task: str,
    logger: logging.Logger,
    max_attempts: int,
    sleep_seconds: float,
) -> GenerationConfig:
    config = retry_call(
        fn=lambda: GenerationConfig.from_pretrained(model_name),
        description=f"Loading generation config for {model_name}",
        logger=logger,
        max_attempts=max_attempts,
        initial_sleep_seconds=sleep_seconds,
    )
    config.language = language
    config.task = task
    config.forced_decoder_ids = None
    config.suppress_tokens = []
    return config


def load_model(
    model_name: str,
    logger: logging.Logger,
    max_attempts: int,
    sleep_seconds: float,
) -> WhisperForConditionalGeneration:
    return retry_call(
        fn=lambda: WhisperForConditionalGeneration.from_pretrained(model_name),
        description=f"Loading baseline model {model_name}",
        logger=logger,
        max_attempts=max_attempts,
        initial_sleep_seconds=sleep_seconds,
    )


def select_split(dataset: DatasetDict, split: str, max_samples: int | None):
    if split not in dataset:
        raise KeyError(f"Split {split!r} was not found. Available splits: {list(dataset.keys())}")
    selected = dataset[split]
    required_columns = {"input_features", "labels"}
    missing = required_columns - set(selected.column_names)
    if missing:
        raise ValueError(f"Processed dataset split {split!r} is missing required columns: {sorted(missing)}")
    if max_samples is not None:
        selected = selected.select(range(min(max_samples, len(selected))))
    return selected


def collate_batch(features: list[dict[str, Any]]) -> dict[str, Any]:
    input_features = torch.as_tensor(
        np.asarray([feature["input_features"] for feature in features], dtype=np.float32),
        dtype=torch.float32,
    )
    return {
        "input_features": input_features,
        "labels": [feature["labels"] for feature in features],
        "text": [feature.get("text", "") for feature in features],
    }


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message and "memory" in message


def evaluate_loop(
    *,
    dataset,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    generation_config: GenerationConfig,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    generation_max_length: int,
    use_fp16: bool,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    predictions: list[str] = []
    references: list[str] = []
    prediction_rows: list[dict[str, str]] = []
    start_time = time.time()
    num_batches = len(dataloader)

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            input_features = batch["input_features"].to(device=device, non_blocking=True)
            if use_fp16 and device.type == "cuda":
                input_features = input_features.to(dtype=torch.float16)

            generated_ids = model.generate(
                input_features=input_features,
                generation_config=generation_config,
                max_length=generation_max_length,
            )

            pred_texts = [
                normalize_text(text)
                for text in processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            ]
            label_texts = [
                normalize_text(text)
                for text in processor.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            ]

            predictions.extend(pred_texts)
            references.extend(label_texts)
            for pred_text, label_text, original_text in zip(pred_texts, label_texts, batch["text"]):
                prediction_rows.append(
                    {
                        "reference": label_text,
                        "prediction": pred_text,
                        "source_text": normalize_text(original_text),
                    }
                )

            if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == num_batches:
                elapsed = time.time() - start_time
                examples_done = min(batch_idx * batch_size, len(dataset))
                rate = examples_done / max(elapsed, 1e-6)
                eta_seconds = (len(dataset) - examples_done) / max(rate, 1e-6)
                logger.info(
                    "Evaluated %s/%s examples (%s/%s batches), %.2f examples/s, ETA %.1f min",
                    examples_done,
                    len(dataset),
                    batch_idx,
                    num_batches,
                    rate,
                    eta_seconds / 60.0,
                )

    wer = jiwer.wer(
        [to_char_tokens(text) for text in references],
        [to_char_tokens(text) for text in predictions],
    )
    cer = jiwer.cer(references, predictions)
    runtime_seconds = time.time() - start_time
    metrics = {
        "test_wer": float(wer),
        "test_cer": float(cer),
        "baseline_test_wer": float(wer),
        "baseline_test_cer": float(cer),
        "num_examples": len(dataset),
        "runtime_seconds": runtime_seconds,
        "samples_per_second": len(dataset) / max(runtime_seconds, 1e-6),
        "batch_size": batch_size,
    }
    return metrics, prediction_rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reference", "prediction", "source_text"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    workspace_root = args.workspace_root.expanduser().resolve()
    physical_data_root = (
        args.physical_data_root.expanduser().resolve()
        if args.physical_data_root
        else (workspace_root / "data_link").expanduser().resolve()
    )
    processed_dataset_dir = (
        args.processed_dataset_dir.expanduser().resolve()
        if args.processed_dataset_dir
        else physical_data_root / "processed" / "seniortalk_whisper_medium"
    )
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else workspace_root / "reports" / "baseline_result.json"
    )
    predictions_csv = (
        args.predictions_csv.expanduser().resolve()
        if args.predictions_csv
        else workspace_root / "reports" / "baseline_predictions.csv"
    )

    logger = setup_logging(workspace_root / "logs" / "eval_baseline.log")
    logger.info("Workspace root: %s", workspace_root)
    logger.info("Physical data root: %s", physical_data_root)
    logger.info("Processed dataset dir: %s", processed_dataset_dir)
    logger.info("Output JSON: %s", output_json)

    if output_json.exists() and not args.overwrite:
        logger.info("Baseline result already exists at %s", output_json)
        logger.info("Use --overwrite to recompute it.")
        print(output_json.read_text(encoding="utf-8"))
        return 0

    if not processed_dataset_dir.exists():
        logger.error("Processed dataset not found at %s. Run prepare_dataset.py first.", processed_dataset_dir)
        return 1

    try:
        configure_hf_environment(physical_data_root)
        set_seed(args.seed)

        dataset_dict = load_from_disk(str(processed_dataset_dir))
        if not isinstance(dataset_dict, DatasetDict):
            raise TypeError(f"Expected DatasetDict at {processed_dataset_dir}, got {type(dataset_dict)}")
        dataset = select_split(dataset_dict, args.split, args.max_samples)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)
        if device.type == "cuda":
            logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))

        processor = load_processor(
            args.model_name,
            args.language,
            args.task,
            logger,
            max_attempts=args.download_retries,
            sleep_seconds=args.retry_sleep_seconds,
        )
        generation_config = load_generation_config(
            args.model_name,
            args.language,
            args.task,
            logger,
            max_attempts=args.download_retries,
            sleep_seconds=args.retry_sleep_seconds,
        )
        model = load_model(
            args.model_name,
            logger,
            max_attempts=args.download_retries,
            sleep_seconds=args.retry_sleep_seconds,
        )
        model.generation_config = generation_config
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.config.use_cache = True
        if args.fp16 and device.type == "cuda":
            model = model.to(device=device, dtype=torch.float16)
        else:
            model = model.to(device=device)

        batch_size = max(1, args.batch_size)
        last_error: Exception | None = None
        while batch_size >= 1:
            try:
                logger.info("Starting baseline evaluation on split=%s with batch_size=%s", args.split, batch_size)
                metrics, prediction_rows = evaluate_loop(
                    dataset=dataset,
                    processor=processor,
                    model=model,
                    generation_config=generation_config,
                    device=device,
                    batch_size=batch_size,
                    num_workers=args.num_workers,
                    generation_max_length=args.generation_max_length,
                    use_fp16=args.fp16,
                    logger=logger,
                )
                break
            except RuntimeError as exc:
                last_error = exc
                if args.auto_reduce_batch_size and batch_size > 1 and is_cuda_oom(exc):
                    logger.warning("CUDA OOM at batch_size=%s; retrying with batch_size=%s", batch_size, batch_size // 2)
                    batch_size //= 2
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise
        else:
            assert last_error is not None
            raise last_error

        payload = {
            "model_name": args.model_name,
            "dataset": "BAAI/SeniorTalk",
            "split": args.split,
            "language": args.language,
            "task": args.task,
            "processed_dataset_dir": str(processed_dataset_dir),
            "created_at_utc": utc_now(),
            "fp16": bool(args.fp16 and device.type == "cuda"),
            "device": str(device),
            "generation_max_length": args.generation_max_length,
            **metrics,
        }
        write_json(output_json, payload)
        if args.save_predictions:
            write_predictions(predictions_csv, prediction_rows)

        logger.info("Baseline Test WER: %.6f", metrics["test_wer"])
        logger.info("Baseline Test CER: %.6f", metrics["test_cer"])
        logger.info("Saved baseline metrics to %s", output_json)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        logger.error("Baseline evaluation failed: %s", exc)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
