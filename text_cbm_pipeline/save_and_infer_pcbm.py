#!/usr/bin/env python3
# Required packages:
#   pip install scikit-learn pandas numpy joblib transformers torch

"""Train, save, and run inference with the final production PCBM."""

from __future__ import annotations

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger("save_and_infer_pcbm")

DEFAULT_CBM_CSV = Path("~/AD_predict/text_cbm_pipeline/cbm_features.csv").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/AD_predict/text_cbm_pipeline").expanduser()
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "final_pcbm_model.joblib"
DEFAULT_FEATURE_NAMES_PATH = DEFAULT_OUTPUT_DIR / "pcbm_feature_names.json"

BLOG_LABELS: Dict[str, int] = {
    "creatingmemories": 1,
    "journeywithdementia": 0,
    "earlyonset": 0,
    "helpparentsagewell": 0,
}

CONCEPT_SPECS: List[Tuple[str, str]] = [
    ("concept_emotional_frustration_anxiety", "emotional frustration or anxiety"),
    ("concept_health_issues_doctors", "talking about health issues or doctors"),
    ("concept_nostalgia_distant_past", "nostalgia or talking about the distant past"),
    ("concept_simple_vocabulary", "very simple and basic vocabulary"),
    ("concept_repetitive_language", "repetitive words or sentences"),
    ("concept_grammatical_errors_confused_syntax", "grammatical errors or confused syntax"),
    ("concept_logical_discontinuity", "lack of logical flow or sudden topic changes"),
    ("concept_short_fragmented_sentences", "short and fragmented sentences"),
]

FEATURE_TO_PROMPT = {feature_name: prompt for feature_name, prompt in CONCEPT_SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save the final PCBM, then run a demo inference."
    )
    parser.add_argument("--cbm-csv", type=Path, default=DEFAULT_CBM_CSV)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-names-path", type=Path, default=DEFAULT_FEATURE_NAMES_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--device", type=int, default=0, help="Use -1 to force CPU inference.")
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument(
        "--demo-text",
        default=(
            "I went to the doctor today and I feel worried. "
            "I write the same thing again and again. "
            "The words are simple. I was thinking about long ago, the old house, my mother, long ago. "
            "Then I start another thing and the sentence is not right and not finished. "
            "I am tired. I am upset. I am tired."
        ),
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Train and save the model without running the demo inference.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def canonicalize_blog_name(blog_name: str) -> str:
    cleaned = str(blog_name).strip().lower().rstrip("/")
    cleaned = cleaned.replace(".blogspot.ca", ".blogspot.com")
    cleaned = cleaned.replace("https://", "").replace("http://", "")
    for blog_slug in BLOG_LABELS:
        if blog_slug in cleaned:
            return blog_slug
    raise ValueError(f"Could not map blog_name to an available training blog: {blog_name!r}")


def load_training_data(cbm_csv: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not cbm_csv.exists():
        raise FileNotFoundError(f"Concept feature CSV does not exist: {cbm_csv}")

    df = pd.read_csv(cbm_csv).copy()
    concept_columns = [column for column in df.columns if column.startswith("concept_")]
    if not concept_columns:
        raise ValueError("No concept columns found. Expected columns beginning with 'concept_'.")

    expected_features = [feature_name for feature_name, _ in CONCEPT_SPECS]
    missing_features = [feature for feature in expected_features if feature not in concept_columns]
    if missing_features:
        raise ValueError(
            f"cbm_features.csv is missing expected concept columns: {missing_features}"
        )

    df["blog_slug"] = df["blog_name"].apply(canonicalize_blog_name)
    df["Diagnosis"] = df["blog_slug"].map(BLOG_LABELS)

    for column in expected_features:
        df[column] = pd.to_numeric(df[column], errors="raise")

    LOGGER.info("Loaded %d rows for final PCBM training.", len(df))
    LOGGER.info(
        "Class counts:\n%s",
        df["Diagnosis"].value_counts().rename(index={0: "Control", 1: "Dementia"}).to_string(),
    )
    return df, expected_features


def train_and_save_final_pcbm(
    cbm_csv: Path,
    model_path: Path,
    feature_names_path: Path,
    random_state: int,
    max_iter: int,
    solver: str,
) -> LogisticRegression:
    df, feature_names = load_training_data(cbm_csv)
    X = df.loc[:, feature_names].to_numpy(dtype=float)
    y = df["Diagnosis"].to_numpy(dtype=int)

    model = LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
    )
    model.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    feature_names_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    with feature_names_path.open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

    coefficient_table = [
        (feature_name, float(weight))
        for feature_name, weight in zip(feature_names, model.coef_[0])
    ]
    coefficient_table.sort(key=lambda item: abs(item[1]), reverse=True)

    print("\nFinal production PCBM trained on 100% of available data.")
    print(f"Saved model to: {model_path}")
    print(f"Saved feature order to: {feature_names_path}")
    print("\nFinal model coefficients (sorted by absolute weight)")
    for feature_name, weight in coefficient_table:
        print(f"  {feature_name}: {weight:+.4f}")

    return model


def truncate_for_inference(text: str, max_chars: int) -> str:
    clean_text = " ".join(str(text).split())
    if max_chars <= 0 or len(clean_text) <= max_chars:
        return clean_text
    return clean_text[:max_chars].strip()


@lru_cache(maxsize=2)
def load_feature_names(feature_names_path_str: str) -> Tuple[str, ...]:
    feature_names_path = Path(feature_names_path_str).expanduser().resolve()
    with feature_names_path.open("r", encoding="utf-8") as handle:
        feature_names = json.load(handle)
    return tuple(feature_names)


@lru_cache(maxsize=2)
def load_saved_model(model_path_str: str):
    model_path = Path(model_path_str).expanduser().resolve()
    return joblib.load(model_path)


@lru_cache(maxsize=4)
def load_zero_shot_classifier(model_name: str, device: int):
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for zero-shot inference. Install torch and transformers."
        ) from exc

    if device >= 0 and not torch.cuda.is_available():
        LOGGER.warning("CUDA device=%d requested but no GPU is available. Falling back to CPU.", device)
        device = -1

    # If you hit CUDA OOM with device=0, change it to device=-1 to run on CPU.
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )
    return classifier, device


def score_text_with_concepts(
    text: str,
    feature_names: Sequence[str],
    device: int,
    model_name: str = "facebook/bart-large-mnli",
    hypothesis_template: str = "The content or writing style of this blog post shows {}.",
    max_chars: int = 1000,
) -> Dict[str, float]:
    truncated_text = truncate_for_inference(text, max_chars=max_chars)
    classifier, resolved_device = load_zero_shot_classifier(model_name=model_name, device=device)
    LOGGER.info("Running concept inference with model=%s device=%d.", model_name, resolved_device)

    candidate_labels = [FEATURE_TO_PROMPT[feature_name] for feature_name in feature_names]
    output = classifier(
        truncated_text,
        candidate_labels=candidate_labels,
        hypothesis_template=hypothesis_template,
        multi_label=True,
        truncation=True,
    )

    score_lookup = {
        label: float(score)
        for label, score in zip(output["labels"], output["scores"])
    }
    return {
        feature_name: score_lookup[FEATURE_TO_PROMPT[feature_name]]
        for feature_name in feature_names
    }


def print_diagnostic_report(
    text: str,
    feature_names: Sequence[str],
    concept_scores: Dict[str, float],
    model,
    ad_probability: float,
) -> None:
    control_probability = 1.0 - ad_probability
    predicted_label = "Dementia-like" if ad_probability >= 0.5 else "Control-like"
    intercept = float(model.intercept_[0])

    print("\n" + "=" * 78)
    print("Final Production PCBM Diagnostic Report")
    print("=" * 78)
    print(f"Input text: {text}")
    print(f"\nPredicted label: {predicted_label}")
    print(f"AD probability: {ad_probability:.4f}")
    print(f"Control probability: {control_probability:.4f}")
    print(f"Model intercept: {intercept:+.4f}")

    print("\nConcept scores in exact model feature order")
    for index, feature_name in enumerate(feature_names, start=1):
        prompt = FEATURE_TO_PROMPT[feature_name]
        print(f"  {index:>2}. {feature_name}: {concept_scores[feature_name]:.4f} | {prompt}")

    contributions = []
    for feature_name, weight in zip(feature_names, model.coef_[0]):
        score = concept_scores[feature_name]
        contribution = float(weight) * float(score)
        contributions.append((feature_name, float(weight), float(score), contribution))
    contributions.sort(key=lambda item: abs(item[3]), reverse=True)

    print("\nTop concept contributions to the logit")
    for feature_name, weight, score, contribution in contributions:
        direction = "pushes toward AD" if contribution >= 0 else "pushes toward Control"
        print(
            f"  {feature_name}: score={score:.4f}, weight={weight:+.4f}, "
            f"contribution={contribution:+.4f} -> {direction}"
        )
    print("=" * 78)


def predict_new_text(
    text: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    feature_names_path: str | Path = DEFAULT_FEATURE_NAMES_PATH,
    device: int = 0,
    max_chars: int = 1000,
) -> Dict[str, object]:
    model_path = Path(model_path).expanduser().resolve()
    feature_names_path = Path(feature_names_path).expanduser().resolve()

    model = load_saved_model(str(model_path))
    feature_names = list(load_feature_names(str(feature_names_path)))
    concept_scores = score_text_with_concepts(
        text=text,
        feature_names=feature_names,
        device=device,
        max_chars=max_chars,
    )

    feature_vector = np.array(
        [[concept_scores[feature_name] for feature_name in feature_names]],
        dtype=float,
    )
    probability_vector = model.predict_proba(feature_vector)[0]
    ad_probability = float(probability_vector[1])

    print_diagnostic_report(
        text=text,
        feature_names=feature_names,
        concept_scores=concept_scores,
        model=model,
        ad_probability=ad_probability,
    )

    return {
        "ad_probability": ad_probability,
        "control_probability": float(probability_vector[0]),
        "concept_scores": concept_scores,
        "feature_order": feature_names,
    }


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    cbm_csv = args.cbm_csv.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    feature_names_path = args.feature_names_path.expanduser().resolve()

    train_and_save_final_pcbm(
        cbm_csv=cbm_csv,
        model_path=model_path,
        feature_names_path=feature_names_path,
        random_state=args.random_state,
        max_iter=args.max_iter,
        solver=args.solver,
    )

    if not args.skip_demo:
        predict_new_text(
            text=args.demo_text,
            model_path=model_path,
            feature_names_path=feature_names_path,
            device=args.device,
            max_chars=args.max_chars,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
