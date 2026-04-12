#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


DEFAULT_WORKSPACE_ROOT = Path.home() / "AD_predict" / "whisper_seniortalk_finetune"
DEFAULT_MODEL_NAME = "openai/whisper-medium"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small hyperparameter sweep for Whisper-medium on SeniorTalk."
    )
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--physical-data-root", type=Path, default=None)
    parser.add_argument("--processed-dataset-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-5, 5e-6])
    parser.add_argument("--effective-batch-sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--generation-max-length", type=int, default=448)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--auto-find-batch-size", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_and_search")
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


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", "" if text is None else str(text))
    normalized = normalized.replace("\u3000", " ")
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def to_char_tokens(text: str) -> str:
    normalized = normalize_text(text)
    return " ".join(list(normalized.replace(" ", "")))


def maybe_limit(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    return dataset.select(range(min(len(dataset), max_samples)))


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return {
            "input_features": batch["input_features"],
            "labels": batch["labels"],
        }


def make_compute_metrics(processor: WhisperProcessor):
    def compute_metrics(pred):
        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        label_ids = np.array(pred.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_texts = [normalize_text(text) for text in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
        label_texts = [normalize_text(text) for text in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

        wer = jiwer.wer(
            [to_char_tokens(text) for text in label_texts],
            [to_char_tokens(text) for text in pred_texts],
        )
        cer = jiwer.cer(label_texts, pred_texts)
        return {
            "wer": float(wer),
            "cer": float(cer),
        }

    return compute_metrics


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


def load_processor(
    model_name: str,
    language: str,
    task: str,
    logger: logging.Logger,
    max_attempts: int,
    sleep_seconds: float,
) -> WhisperProcessor:
    return retry_call(
        fn=lambda: WhisperProcessor.from_pretrained(model_name, language=language, task=task),
        description=f"Loading processor for {model_name}",
        logger=logger,
        max_attempts=max_attempts,
        initial_sleep_seconds=sleep_seconds,
    )


def load_model(
    model_name: str,
    logger: logging.Logger,
    max_attempts: int,
    sleep_seconds: float,
) -> WhisperForConditionalGeneration:
    return retry_call(
        fn=lambda: WhisperForConditionalGeneration.from_pretrained(model_name),
        description=f"Loading model {model_name}",
        logger=logger,
        max_attempts=max_attempts,
        initial_sleep_seconds=sleep_seconds,
    )


def build_experiment_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    grid = []
    per_device_batch = max(1, args.per_device_train_batch_size)
    for lr in args.learning_rates:
        for requested_effective_batch_size in args.effective_batch_sizes:
            gradient_accumulation_steps = max(
                1, (requested_effective_batch_size + per_device_batch - 1) // per_device_batch
            )
            effective_batch_size = per_device_batch * gradient_accumulation_steps
            profile_name = f"effective_batch_{effective_batch_size}"
            name = f"lr_{lr:.0e}_{profile_name}"
            grid.append(
                {
                    "name": name,
                    "learning_rate": lr,
                    "per_device_train_batch_size": per_device_batch,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "profile_name": profile_name,
                    "effective_batch_size": effective_batch_size,
                }
            )
    return grid


def create_training_args(
    run_dir: Path,
    generation_config: GenerationConfig,
    experiment: dict[str, Any],
    args: argparse.Namespace,
) -> Seq2SeqTrainingArguments:
    bf16 = args.bf16 or (not args.fp16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    fp16 = args.fp16 or (not bf16 and torch.cuda.is_available())
    return Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        per_device_train_batch_size=experiment["per_device_train_batch_size"],
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=experiment["gradient_accumulation_steps"],
        learning_rate=experiment["learning_rate"],
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_config=generation_config,
        generation_max_length=args.generation_max_length,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        tf32=True,
        gradient_checkpointing=True,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        save_only_model=False,
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
        auto_find_batch_size=args.auto_find_batch_size,
        use_cache=False,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    serialized = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return serialized


def export_best_model(
    best_result: dict[str, Any],
    processor: WhisperProcessor,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    workspace_root = args.workspace_root.expanduser().resolve()
    export_dir_link = workspace_root / "best_model_export"
    export_dir = export_dir_link.resolve() if export_dir_link.is_symlink() else export_dir_link
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "selected_experiment": best_result["experiment_name"],
        "best_checkpoint": best_result["best_checkpoint"],
        "validation_wer": best_result["validation_wer"],
        "test_wer": best_result["test_wer"],
        "validation_cer": best_result.get("validation_cer"),
        "test_cer": best_result.get("test_cer"),
        "is_lora": best_result["is_lora"],
    }
    write_json(export_dir / "selection_summary.json", summary)

    if best_result["is_lora"]:
        adapter_export_dir = export_dir / "adapter_checkpoint"
        shutil.copytree(best_result["best_checkpoint"], adapter_export_dir, dirs_exist_ok=True)

        logger.info("Merging best LoRA adapter into a standalone Whisper-medium export")
        base_model = load_model(
            args.model_name,
            logger=logger,
            max_attempts=args.download_retries,
            sleep_seconds=args.retry_sleep_seconds,
        )
        base_model.config.use_cache = False
        base_model.generation_config = load_generation_config(
            model_name=args.model_name,
            language=args.language,
            task=args.task,
            logger=logger,
            max_attempts=args.download_retries,
            sleep_seconds=args.retry_sleep_seconds,
        )
        merged_model = PeftModel.from_pretrained(base_model, best_result["best_checkpoint"])
        merged_model = merged_model.merge_and_unload()
        merged_dir = export_dir / "merged_model"
        merged_model.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
    else:
        model_export_dir = export_dir / "full_model"
        best_model = WhisperForConditionalGeneration.from_pretrained(best_result["best_checkpoint"])
        best_model.save_pretrained(str(model_export_dir))
        processor.save_pretrained(str(model_export_dir))


def save_reports(results: list[dict[str, Any]], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "grid_search_summary.csv"
    md_path = reports_dir / "grid_search_summary.md"

    fieldnames = [
        "experiment_name",
        "status",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "validation_wer",
        "validation_cer",
        "test_wer",
        "test_cer",
        "best_checkpoint",
        "run_dir",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key, "") for key in fieldnames}
            writer.writerow(row)

    lines = [
        "# Whisper-medium SeniorTalk Grid Search",
        "",
        "| Experiment | Status | LR | Batch | Grad Accum | Eff Batch | Val WER | Test WER | Best Checkpoint |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| {experiment_name} | {status} | {learning_rate} | {per_device_train_batch_size} | "
            "{gradient_accumulation_steps} | {effective_batch_size} | {validation_wer} | {test_wer} | {best_checkpoint} |".format(
                experiment_name=result.get("experiment_name", ""),
                status=result.get("status", ""),
                learning_rate=result.get("learning_rate", ""),
                per_device_train_batch_size=result.get("per_device_train_batch_size", ""),
                gradient_accumulation_steps=result.get("gradient_accumulation_steps", ""),
                effective_batch_size=result.get("effective_batch_size", ""),
                validation_wer=result.get("validation_wer", ""),
                test_wer=result.get("test_wer", ""),
                best_checkpoint=result.get("best_checkpoint", ""),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.rerun_completed:
        args.skip_completed = False

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

    logger = setup_logging(workspace_root / "logs" / "train_and_search.log")
    logger.info("Workspace root: %s", workspace_root)
    logger.info("Physical data root: %s", physical_data_root)
    logger.info("Processed dataset dir: %s", processed_dataset_dir)
    configure_hf_environment(physical_data_root)

    if not processed_dataset_dir.exists():
        logger.error("Processed dataset not found at %s", processed_dataset_dir)
        logger.error("Run prepare_dataset.py first.")
        return 1

    set_seed(args.seed)
    use_lora = args.use_lora and not args.no_lora

    try:
        dataset = load_from_disk(str(processed_dataset_dir))
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected DatasetDict, got {type(dataset)}")
    except Exception as exc:
        logger.error("Failed to load processed dataset: %s", exc)
        return 1

    train_dataset = dataset["train"]
    validation_dataset = maybe_limit(dataset["validation"], args.max_eval_samples)
    test_dataset = maybe_limit(dataset["test"], args.max_test_samples)

    processor = load_processor(
        args.model_name,
        args.language,
        args.task,
        logger,
        max_attempts=args.download_retries,
        sleep_seconds=args.retry_sleep_seconds,
    )
    if hasattr(processor.tokenizer, "set_prefix_tokens"):
        processor.tokenizer.set_prefix_tokens(language=args.language, task=args.task)
    generation_config = load_generation_config(
        args.model_name,
        args.language,
        args.task,
        logger,
        max_attempts=args.download_retries,
        sleep_seconds=args.retry_sleep_seconds,
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
    )

    results: list[dict[str, Any]] = []
    grid = build_experiment_grid(args)
    reports_dir = workspace_root / "reports"
    runs_root = workspace_root / "outputs" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    for experiment in grid:
        experiment_name = experiment["name"]
        run_dir = runs_root / experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "final_metrics.json"

        if args.skip_completed and metrics_path.exists():
            previous_result = json.loads(metrics_path.read_text(encoding="utf-8"))
            if previous_result.get("status") == "success":
                logger.info("Skipping completed experiment %s", experiment_name)
                results.append(previous_result)
                continue
            logger.info("Previous run for %s was not successful, rerunning it", experiment_name)
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting experiment: %s", experiment_name)
        logger.info("Experiment config: %s", experiment)

        run_summary = {
            "experiment_name": experiment_name,
            "status": "failed",
            "learning_rate": experiment["learning_rate"],
            "per_device_train_batch_size": experiment["per_device_train_batch_size"],
            "gradient_accumulation_steps": experiment["gradient_accumulation_steps"],
            "effective_batch_size": experiment["effective_batch_size"],
            "validation_wer": None,
            "validation_cer": None,
            "test_wer": None,
            "test_cer": None,
            "best_checkpoint": "",
            "run_dir": str(run_dir),
            "error": "",
            "is_lora": use_lora,
        }

        trainer = None
        model = None
        try:
            model = load_model(
                args.model_name,
                logger=logger,
                max_attempts=args.download_retries,
                sleep_seconds=args.retry_sleep_seconds,
            )
            model.generation_config = generation_config
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []
            model.config.use_cache = False

            if use_lora:
                target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=target_modules,
                )
                model = get_peft_model(model, lora_config)
                if hasattr(model, "print_trainable_parameters"):
                    model.print_trainable_parameters()

            training_args = create_training_args(run_dir, generation_config, experiment, args)

            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=data_collator,
                processing_class=processor,
                compute_metrics=make_compute_metrics(processor),
            )

            last_checkpoint = get_last_checkpoint(str(run_dir))
            if last_checkpoint:
                logger.info("Resuming %s from checkpoint %s", experiment_name, last_checkpoint)
            trainer.train(resume_from_checkpoint=last_checkpoint)

            best_checkpoint = trainer.state.best_model_checkpoint or get_last_checkpoint(str(run_dir))
            if not best_checkpoint:
                raise RuntimeError(f"No checkpoint found for experiment {experiment_name}")

            validation_metrics = serialize_metrics(
                trainer.evaluate(eval_dataset=validation_dataset, metric_key_prefix="validation")
            )
            test_metrics = serialize_metrics(trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test"))

            run_summary.update(
                {
                    "status": "success",
                    "validation_wer": validation_metrics.get("validation_wer"),
                    "validation_cer": validation_metrics.get("validation_cer"),
                    "test_wer": test_metrics.get("test_wer"),
                    "test_cer": test_metrics.get("test_cer"),
                    "best_checkpoint": best_checkpoint,
                    "train_runtime_seconds": validation_metrics.get("validation_runtime"),
                }
            )

            if trainer.state.log_history:
                write_json(run_dir / "trainer_log_history.json", {"log_history": trainer.state.log_history})

        except Exception as exc:
            logger.error("Experiment %s failed: %s", experiment_name, exc)
            logger.error(traceback.format_exc())
            run_summary["error"] = str(exc)
        finally:
            write_json(metrics_path, run_summary)
            results.append(run_summary)
            save_reports(results, reports_dir)

            if trainer is not None:
                del trainer
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    successful_results = [result for result in results if result["status"] == "success"]
    if not successful_results:
        logger.error("No experiment finished successfully.")
        return 1

    best_result = min(successful_results, key=lambda item: item["validation_wer"])
    logger.info(
        "Best experiment: %s (validation WER=%s, test WER=%s)",
        best_result["experiment_name"],
        best_result["validation_wer"],
        best_result["test_wer"],
    )
    export_best_model(best_result, processor, args, logger)
    save_reports(results, reports_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
