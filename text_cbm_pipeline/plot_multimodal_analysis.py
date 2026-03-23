#!/usr/bin/env python3
"""Generate publication-quality analysis figures for the multimodal CRF-PCBM model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

DEFAULT_RUN_DIR = Path("~/AD_predict/text_cbm_pipeline/multimodal_crf_runs").expanduser()
DEFAULT_FOLD_METRICS = DEFAULT_RUN_DIR / "multimodal_crf_fold_metrics.csv"
DEFAULT_PREDICTIONS = DEFAULT_RUN_DIR / "multimodal_crf_sequence_predictions.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "figures"

TRANSITION_MATRIX = np.array(
    [
        [0.089, -0.121],
        [-0.121, 0.089],
    ],
    dtype=float,
)

# Replace these constants with the exact fold-averaged absolute weights from your
# final multimodal run if you want the radar chart to mirror a different log dump.
TEXT_CONCEPT_IMPORTANCE = {
    "short fragmented sentences": 0.918,
    "simple vocabulary": 2.619,
    "grammatical errors": 1.232,
    "emotional frustration": 0.575,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CRF transition, concept-importance, and ROC figures for the multimodal CRF-PCBM model."
    )
    parser.add_argument("--fold-metrics", type=Path, default=DEFAULT_FOLD_METRICS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Output formats to save for each figure.",
    )
    return parser.parse_args()


def configure_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        },
    )


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: Iterable[str], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_fold_metrics(path: Path) -> pd.DataFrame:
    ensure_exists(path)
    df = pd.read_csv(path)
    required = {"accuracy", "auc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fold metrics file is missing required columns: {sorted(missing)}")
    return df


def resolve_column(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Could not find a {role} column. Tried: {candidates}. Available columns: {df.columns.tolist()}"
    )


def load_prediction_targets(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ensure_exists(path)
    df = pd.read_csv(path)
    label_col = resolve_column(df, ["True_Label", "Label", "true_label", "label"], "label")
    prob_col = resolve_column(
        df,
        ["Pred_Prob", "Pred_Prob_AD", "pred_prob", "pred_prob_ad"],
        "predicted-probability",
    )
    labels = df[label_col].to_numpy(dtype=int)
    probs = df[prob_col].to_numpy(dtype=float)
    return labels, probs


def plot_transition_heatmap(output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    vmax = float(np.max(np.abs(TRANSITION_MATRIX)))
    sns.heatmap(
        TRANSITION_MATRIX,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "Transition Weight"},
        annot_kws={"fontsize": 13, "fontweight": "semibold"},
    )
    ax.set_xticklabels(["Next: CN", "Next: AD"], rotation=0)
    ax.set_yticklabels(["Current: CN", "Current: AD"], rotation=0)
    ax.set_xlabel("Destination State")
    ax.set_ylabel("Source State")
    ax.set_title("Figure 3. CRF Transition Matrix", pad=12, fontweight="semibold")
    save_figure(fig, output_dir, "figure_3_crf_transition_heatmap", formats, dpi)


def plot_concept_radar(output_dir: Path, formats: list[str], dpi: int) -> None:
    labels = list(TEXT_CONCEPT_IMPORTANCE.keys())
    values = list(TEXT_CONCEPT_IMPORTANCE.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(7.0, 7.0), subplot_kw={"polar": True}, constrained_layout=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, values, color="#B22222", linewidth=2.4)
    ax.fill(angles, values, color="#CD5C5C", alpha=0.22)
    ax.scatter(angles[:-1], values[:-1], color="#7F0000", s=36, zorder=3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [
            "Short fragmented\nsentences",
            "Simple\nvocabulary",
            "Grammatical\nerrors",
            "Emotional\nfrustration",
        ]
    )

    radial_max = max(values) * 1.15
    ax.set_ylim(0, radial_max)
    radial_ticks = np.linspace(0, radial_max, 5)[1:]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.2f}" for tick in radial_ticks])
    ax.grid(color="#B0B0B0", alpha=0.5)
    ax.spines["polar"].set_color("#666666")
    ax.set_title(
        "Figure 4. Text Concept Importance in the Multimodal Bottleneck",
        pad=22,
        fontweight="semibold",
    )

    save_figure(fig, output_dir, "figure_4_concept_importance_radar", formats, dpi)


def plot_roc_curve(
    fold_metrics: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
    ax.plot(
        fpr,
        tpr,
        color="#8B0000",
        linewidth=2.4,
        label=f"Multimodal CRF-PCBM (AUC = {roc_auc:.2f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#4F4F4F", linewidth=1.4, label="Random baseline")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure 5. ROC Curve for Speaker-Level AD Detection", pad=12, fontweight="semibold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.25)

    fold_auc_mean = float(fold_metrics["auc"].mean())
    fold_auc_std = float(fold_metrics["auc"].std(ddof=0))
    fold_acc_mean = float(fold_metrics["accuracy"].mean())
    summary_text = (
        f"5-fold mean AUC = {fold_auc_mean:.3f}\n"
        f"5-fold mean ACC = {fold_acc_mean:.3f}\n"
        f"AUC SD = {fold_auc_std:.3f}"
    )
    ax.text(
        0.97,
        0.14,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.95},
    )

    save_figure(fig, output_dir, "figure_5_roc_curve", formats, dpi)


def main() -> None:
    args = parse_args()
    configure_style()

    fold_metrics = load_fold_metrics(args.fold_metrics)
    labels, probs = load_prediction_targets(args.predictions)

    plot_transition_heatmap(args.output_dir, args.formats, args.dpi)
    plot_concept_radar(args.output_dir, args.formats, args.dpi)
    plot_roc_curve(fold_metrics, labels, probs, args.output_dir, args.formats, args.dpi)

    print(f"Saved figures to: {args.output_dir}")
    print("Generated:")
    for stem in (
        "figure_3_crf_transition_heatmap",
        "figure_4_concept_importance_radar",
        "figure_5_roc_curve",
    ):
        for fmt in args.formats:
            print(f"  {args.output_dir / f'{stem}.{fmt}'}")


if __name__ == "__main__":
    main()
