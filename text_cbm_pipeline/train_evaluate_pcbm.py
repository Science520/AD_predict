#!/usr/bin/env python3
# Required packages:
#   pip install lxml scikit-learn matplotlib seaborn pandas numpy scipy

"""Train and evaluate the final logistic-regression layer of a text PCBM.

This version reflects the currently available corpus:
    - Dementia: creatingmemories
    - Control: journeywithdementia, earlyonset, helpparentsagewell

Because only one dementia blog is available, the original leave-blog-out benchmark
from the 2017 paper cannot be reproduced without eliminating the positive class from
training. The script therefore uses stratified 5-fold cross-validation at the post level.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

LOGGER = logging.getLogger("train_evaluate_pcbm")

DEFAULT_INPUT_CSV = Path("~/AD_predict/text_cbm_pipeline/cbm_features.csv").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/AD_predict/text_cbm_pipeline").expanduser()
DEFAULT_FIGURE_1 = DEFAULT_OUTPUT_DIR / "figure_1_auc_comparison.png"
DEFAULT_FIGURE_2 = DEFAULT_OUTPUT_DIR / "figure_2_concept_trajectory.png"

BLOG_LABELS: Dict[str, int] = {
    "creatingmemories": 1,
    "journeywithdementia": 0,
    "earlyonset": 0,
    "helpparentsagewell": 0,
}

BLOG_DISPLAY_NAMES: Dict[str, str] = {
    "creatingmemories": "creatingmemories (Dementia)",
    "journeywithdementia": "journeywithdementia (Control)",
    "earlyonset": "earlyonset (Control)",
    "helpparentsagewell": "helpparentsagewell (Control)",
}

RANDOM_BASELINE_AUC = 0.5


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    auc: float
    train_rows: int
    test_rows: int
    train_positive: int
    test_positive: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a logistic-regression PCBM predictor on concept features."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="liblinear")
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
    raise ValueError(f"Could not map blog_name to an available benchmark blog: {blog_name!r}")


def parse_dates(date_series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(date_series, format="%A, %B %d, %Y", errors="coerce")
    if parsed.isna().any():
        fallback_mask = parsed.isna()
        parsed.loc[fallback_mask] = pd.to_datetime(
            date_series.loc[fallback_mask], errors="coerce"
        )
    return parsed


def load_and_prepare_data(input_csv: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    df = pd.read_csv(input_csv)
    required_columns = {"blog_name", "date"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing_columns)}")

    concept_columns = [column for column in df.columns if column.startswith("concept_")]
    if not concept_columns:
        raise ValueError("No concept columns found. Expected columns beginning with 'concept_'.")

    df = df.copy()
    df["blog_slug"] = df["blog_name"].apply(canonicalize_blog_name)
    df["Diagnosis"] = df["blog_slug"].map(BLOG_LABELS)
    df["parsed_date"] = parse_dates(df["date"])

    for column in concept_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    LOGGER.info("Loaded %d rows with %d concept features.", len(df), len(concept_columns))
    LOGGER.info("Blog counts:\n%s", df["blog_slug"].value_counts().sort_index().to_string())
    LOGGER.info(
        "Class counts:\n%s",
        df["Diagnosis"].value_counts().rename(index={0: "Control", 1: "Dementia"}).to_string(),
    )
    return df, concept_columns


def validate_dataset(df: pd.DataFrame, n_splits: int) -> None:
    available_blogs = set(df["blog_slug"].unique())
    expected_blogs = set(BLOG_LABELS.keys())
    missing_blogs = sorted(expected_blogs - available_blogs)
    if missing_blogs:
        raise ValueError(
            f"Missing expected blogs for the adapted 4-blog experiment: {missing_blogs}"
        )

    class_counts = df["Diagnosis"].value_counts()
    if set(class_counts.index) != {0, 1}:
        raise ValueError("Both classes must be present in the dataset.")
    if int(class_counts.min()) < n_splits:
        raise ValueError(
            f"Stratified {n_splits}-fold CV requires at least {n_splits} samples in each class. "
            f"Observed counts: {class_counts.to_dict()}"
        )


def confidence_interval(values: Sequence[float], confidence: float = 0.90) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, 0.0

    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    try:
        from scipy.stats import t

        critical_value = float(t.ppf((1.0 + confidence) / 2.0, df=arr.size - 1))
    except Exception:
        critical_value = 2.131846786326649
    return mean, critical_value * sem


def run_stratified_cross_validation(
    df: pd.DataFrame,
    concept_columns: Sequence[str],
    n_splits: int,
    random_state: int,
    max_iter: int,
    solver: str,
) -> Tuple[List[FoldResult], pd.DataFrame]:
    X = df.loc[:, concept_columns].to_numpy(dtype=float)
    y = df["Diagnosis"].to_numpy(dtype=int)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_results: List[FoldResult] = []
    coefficient_rows: List[Dict[str, float]] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if set(np.unique(y_train)) != {0, 1}:
            raise RuntimeError(f"Fold {fold_id} training data does not contain both classes.")
        if set(np.unique(y_test)) != {0, 1}:
            raise RuntimeError(f"Fold {fold_id} test data does not contain both classes.")

        model = LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_score))

        fold_results.append(
            FoldResult(
                fold_id=fold_id,
                auc=auc,
                train_rows=len(train_idx),
                test_rows=len(test_idx),
                train_positive=int(y_train.sum()),
                test_positive=int(y_test.sum()),
            )
        )

        coefficient_row = {"fold_id": fold_id}
        coefficient_row.update(
            {feature_name: float(weight) for feature_name, weight in zip(concept_columns, model.coef_[0])}
        )
        coefficient_rows.append(coefficient_row)

        LOGGER.info(
            "Fold %d | train_rows=%d | test_rows=%d | train_pos=%d | test_pos=%d | AUC=%.4f",
            fold_id,
            len(train_idx),
            len(test_idx),
            int(y_train.sum()),
            int(y_test.sum()),
            auc,
        )

    return fold_results, pd.DataFrame(coefficient_rows)


def summarize_coefficients(coefficients_df: pd.DataFrame, concept_columns: Sequence[str]) -> pd.DataFrame:
    summary_rows = []
    for column in concept_columns:
        values = coefficients_df[column].to_numpy(dtype=float)
        summary_rows.append(
            {
                "concept": column,
                "mean_weight": float(np.mean(values)),
                "std_weight": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "mean_abs_weight": float(np.mean(np.abs(values))),
            }
        )
    return (
        pd.DataFrame(summary_rows)
        .sort_values(by=["mean_abs_weight", "mean_weight"], ascending=[False, False])
        .reset_index(drop=True)
    )


def prettify_concept_name(concept_column: str) -> str:
    return concept_column.replace("concept_", "").replace("_", " ")


def save_auc_figure(auc_values: Sequence[float], output_path: Path) -> None:
    mean_auc, ci_half_width = confidence_interval(auc_values, confidence=0.90)
    sns.set_theme(style="whitegrid", context="talk")

    labels = [f"Fold {index}" for index in range(1, len(auc_values) + 1)] + ["Mean"]
    values = list(auc_values) + [mean_auc]
    colors = [sns.color_palette("deep")[0]] * len(auc_values) + [sns.color_palette("deep")[3]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    bars[-1].set_linewidth(1.5)

    mean_x = len(labels) - 1
    ax.errorbar(
        mean_x,
        mean_auc,
        yerr=ci_half_width,
        fmt="none",
        ecolor="black",
        elinewidth=2,
        capsize=8,
        zorder=4,
    )

    ax.axhline(RANDOM_BASELINE_AUC, color="#7f7f7f", linestyle="--", linewidth=2, label="Random baseline = 0.5")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("PCBM AUC with Stratified 5-Fold Cross-Validation")
    ax.text(
        mean_x,
        min(mean_auc + ci_half_width + 0.04, 0.98),
        f"{mean_auc:.3f} ± {ci_half_width:.3f} (90% CI)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_concept_trajectory_figure(df: pd.DataFrame, top_concept: str, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    ordered_blogs = [
        "creatingmemories",
        "journeywithdementia",
        "earlyonset",
        "helpparentsagewell",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    scatter_color = sns.color_palette("deep")[0]
    line_color = sns.color_palette("deep")[3]
    concept_label = prettify_concept_name(top_concept).title()

    for ax, blog_slug in zip(axes.flat, ordered_blogs):
        blog_df = (
            df.loc[df["blog_slug"] == blog_slug, ["parsed_date", top_concept]]
            .dropna(subset=["parsed_date", top_concept])
            .sort_values("parsed_date")
        )
        if blog_df.empty:
            raise ValueError(f"Cannot plot trajectory for {blog_slug}: no dated rows are available.")

        x_numeric = mdates.date2num(blog_df["parsed_date"])
        sns.regplot(
            x=x_numeric,
            y=blog_df[top_concept],
            ax=ax,
            ci=None,
            scatter_kws={"s": 24, "alpha": 0.55, "color": scatter_color},
            line_kws={"linewidth": 2.2, "color": line_color},
        )

        ax.set_title(BLOG_DISPLAY_NAMES[blog_slug], fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel(concept_label)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"Longitudinal Trajectory of Top Concept: {concept_label}",
        fontsize=16,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    fold_results: Sequence[FoldResult],
    coefficient_summary: pd.DataFrame,
    top_concept: str,
) -> None:
    auc_values = [result.auc for result in fold_results]
    mean_auc, ci_half_width = confidence_interval(auc_values, confidence=0.90)

    print("\nFold AUCs")
    for result in fold_results:
        print(
            f"  Fold {result.fold_id}: AUC={result.auc:.4f}, "
            f"train_rows={result.train_rows}, test_rows={result.test_rows}, "
            f"train_positive={result.train_positive}, test_positive={result.test_positive}"
        )

    print(f"\nMean AUC across {len(fold_results)} folds: {mean_auc:.4f} +/- {ci_half_width:.4f} (90% CI)")
    print("\nInterpretable concept weights (mean across folds, sorted by mean absolute weight)")
    for _, row in coefficient_summary.iterrows():
        print(
            f"  {row['concept']}: mean_weight={row['mean_weight']:.4f}, "
            f"std_weight={row['std_weight']:.4f}, mean_abs_weight={row['mean_abs_weight']:.4f}"
        )
    print(f"\nTop concept for longitudinal plotting: {top_concept}")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    input_csv = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df, concept_columns = load_and_prepare_data(input_csv)
    validate_dataset(df, n_splits=args.n_splits)

    fold_results, coefficients_df = run_stratified_cross_validation(
        df=df,
        concept_columns=concept_columns,
        n_splits=args.n_splits,
        random_state=args.random_state,
        max_iter=args.max_iter,
        solver=args.solver,
    )

    fold_results_df = pd.DataFrame([result.__dict__ for result in fold_results])
    coefficient_summary = summarize_coefficients(coefficients_df, concept_columns)
    top_concept = str(coefficient_summary.iloc[0]["concept"])

    fold_results_path = output_dir / "pcbm_fold_results.csv"
    coefficient_summary_path = output_dir / "pcbm_feature_weights.csv"
    figure_1_path = output_dir / DEFAULT_FIGURE_1.name
    figure_2_path = output_dir / DEFAULT_FIGURE_2.name

    fold_results_df.to_csv(fold_results_path, index=False)
    coefficient_summary.to_csv(coefficient_summary_path, index=False)
    save_auc_figure(fold_results_df["auc"].to_list(), figure_1_path)
    save_concept_trajectory_figure(df, top_concept, figure_2_path)

    print_summary(fold_results, coefficient_summary, top_concept)
    print(f"\nSaved fold metrics to: {fold_results_path}")
    print(f"Saved coefficient summary to: {coefficient_summary_path}")
    print(f"Saved Figure 1 to: {figure_1_path}")
    print(f"Saved Figure 2 to: {figure_2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
