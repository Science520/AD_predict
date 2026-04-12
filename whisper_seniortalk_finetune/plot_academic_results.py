#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import textwrap
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


EMBEDDED_GRID_SEARCH_CSV = """experiment_name,status,learning_rate,per_device_train_batch_size,gradient_accumulation_steps,effective_batch_size,validation_wer,validation_cer,test_wer,test_cer,best_checkpoint,run_dir,error
lr_1e-05_effective_batch_16,success,1e-05,1,16,16,0.169296822274235,0.1693253119301621,0.20801077078424773,0.20800772675388532,/home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_1e-05_effective_batch_16/checkpoint-2000,/home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_1e-05_effective_batch_16,
lr_1e-05_effective_batch_32,success,1e-05,1,32,32,0.16292126764406026,0.162950306545867,0.20266928131356737,0.20266631545058097,/home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_1e-05_effective_batch_32/checkpoint-1000,/home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_1e-05_effective_batch_32,
lr_5e-06_effective_batch_16,success,5e-06,1,16,16,0.15719188409126808,0.15722141657214236,0.1736935302123425,0.17369098838060115,/home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_5e-06_effective_batch_16/checkpoint-2955,/home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_5e-06_effective_batch_16,
lr_5e-06_effective_batch_32,success,5e-06,1,32,32,0.15871397596243592,0.15874337731704166,0.18499114629827462,0.18498843913717916,/home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_5e-06_effective_batch_32/checkpoint-1478,/home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_5e-06_effective_batch_32,
"""


REQUIRED_COLUMNS = {
    "experiment_name",
    "status",
    "learning_rate",
    "effective_batch_size",
    "validation_wer",
    "test_wer",
    "run_dir",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for the Whisper SeniorTalk grid search."
    )
    parser.add_argument("--workspace-root", type=Path, default=script_dir)
    parser.add_argument("--csv", type=Path, default=None, help="Optional explicit grid-search CSV path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to WORKSPACE_ROOT/paper_figures.")
    parser.add_argument("--include-failed", action="store_true", help="Include non-success rows if present.")
    return parser.parse_args()


def configure_plot_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="colorblind",
        font="DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def format_lr(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(numeric))))
    coefficient = numeric / (10**exponent)
    if abs(coefficient - round(coefficient)) < 1e-8:
        return f"{int(round(coefficient))}e{exponent:+d}"
    return f"{coefficient:.1f}e{exponent:+d}"


def format_experiment_label(row: pd.Series) -> str:
    return f"LR={format_lr(row['learning_rate'])}\nEff. batch={int(row['effective_batch_size'])}"


def normalize_grid_dataframe(df: pd.DataFrame, include_failed: bool) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Grid-search CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    numeric_columns = [
        "learning_rate",
        "effective_batch_size",
        "validation_wer",
        "test_wer",
        "validation_cer",
        "test_cer",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if not include_failed and "status" in df.columns:
        df = df[df["status"].eq("success")].copy()

    df = df.dropna(subset=["learning_rate", "effective_batch_size", "validation_wer", "test_wer"])
    df = df.sort_values(["learning_rate", "effective_batch_size"], ascending=[False, True]).reset_index(drop=True)
    df["plot_label"] = df.apply(format_experiment_label, axis=1)
    return df


def load_grid_dataframe(workspace_root: Path, explicit_csv: Path | None, include_failed: bool) -> tuple[pd.DataFrame, str]:
    candidate_paths = []
    if explicit_csv is not None:
        candidate_paths.append(explicit_csv.expanduser())
    candidate_paths.extend(
        [
            workspace_root / "reports" / "summary.csv",
            workspace_root / "reports" / "grid_search_summary.csv",
        ]
    )

    for path in candidate_paths:
        path = path.expanduser().resolve()
        if path.exists():
            df = pd.read_csv(path)
            return normalize_grid_dataframe(df, include_failed), str(path)

    df = pd.read_csv(StringIO(EMBEDDED_GRID_SEARCH_CSV))
    return normalize_grid_dataframe(df, include_failed), "embedded fallback CSV"


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.35,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            rotation=0,
        )


def plot_grouped_wer_bars(df: pd.DataFrame, output_dir: Path) -> None:
    labels = df["plot_label"].tolist()
    x = np.arange(len(labels))
    width = 0.36
    val_percent = df["validation_wer"].to_numpy(dtype=float) * 100.0
    test_percent = df["test_wer"].to_numpy(dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    palette = sns.color_palette("colorblind", 2)
    val_bars = ax.bar(x - width / 2, val_percent, width, label="Validation WER", color=palette[0], edgecolor="black", linewidth=0.6)
    test_bars = ax.bar(x + width / 2, test_percent, width, label="Test WER", color=palette[1], edgecolor="black", linewidth=0.6)

    add_bar_labels(ax, val_bars)
    add_bar_labels(ax, test_bars)

    upper = max(np.nanmax(val_percent), np.nanmax(test_percent))
    ax.set_ylim(0, upper * 1.18)
    ax.set_ylabel("Word Error Rate (%)")
    ax.set_xlabel("Hyperparameter configuration")
    ax.set_title("Validation and Test WER Across Whisper-medium Fine-tuning Runs")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.grid(False)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure1_grouped_validation_test_wer")


def plot_test_wer_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    pivot = df.pivot_table(
        index="learning_rate",
        columns="effective_batch_size",
        values="test_wer",
        aggfunc="min",
    )
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot_percent = pivot * 100.0

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    sns.heatmap(
        pivot_percent,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu_r",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Test WER (%)"},
        square=True,
    )

    best_lr, best_batch = pivot.stack().idxmin()
    row_idx = list(pivot.index).index(best_lr)
    col_idx = list(pivot.columns).index(best_batch)
    ax.add_patch(Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor="black", linewidth=2.2))
    ax.text(
        col_idx + 0.5,
        row_idx + 0.86,
        "Best",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="black",
    )

    ax.set_title("Test WER Heatmap by Learning Rate and Effective Batch Size")
    ax.set_xlabel("Effective batch size")
    ax.set_ylabel("Learning rate")
    ax.set_yticklabels([format_lr(value) for value in pivot.index], rotation=0)
    ax.set_xticklabels([str(int(value)) for value in pivot.columns], rotation=0)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure2_test_wer_hyperparameter_heatmap")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: could not parse {path}: {exc}")
        return None


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-", 1)[1]), name
        except ValueError:
            pass
    return -1, name


def locate_log_history(run_dir: Path, best_checkpoint: str | float | None = None) -> tuple[list[dict[str, Any]], str | None]:
    candidates: list[Path] = []
    run_dir = run_dir.expanduser()

    if (run_dir / "trainer_log_history.json").exists():
        candidates.append(run_dir / "trainer_log_history.json")

    if isinstance(best_checkpoint, str) and best_checkpoint.strip():
        best_path = Path(best_checkpoint).expanduser()
        if (best_path / "trainer_state.json").exists():
            candidates.append(best_path / "trainer_state.json")

    if run_dir.exists():
        checkpoints = sorted(
            [path for path in run_dir.glob("checkpoint-*") if path.is_dir()],
            key=checkpoint_sort_key,
            reverse=True,
        )
        candidates.extend(path / "trainer_state.json" for path in checkpoints if (path / "trainer_state.json").exists())
        candidates.extend(path for path in run_dir.rglob("trainer_state.json") if path.is_file())

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        payload = read_json(candidate)
        if payload is None:
            continue
        if isinstance(payload, list):
            history = payload
        else:
            history = payload.get("log_history", [])
        if history:
            return history, str(candidate)

    return [], None


def extract_curve_points(
    log_history: list[dict[str, Any]],
    metric_key: str,
) -> pd.DataFrame:
    rows = []
    for entry in log_history:
        if metric_key not in entry:
            continue
        try:
            value = float(entry[metric_key])
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "step": int(entry.get("step", len(rows) + 1)),
                "epoch": float(entry.get("epoch", np.nan)),
                metric_key: value,
            }
        )
    return pd.DataFrame(rows)


def plot_training_curves(df: pd.DataFrame, output_dir: Path) -> tuple[int, list[str]]:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharex=False)
    palette = sns.color_palette("colorblind", len(df))
    warnings: list[str] = []
    plotted_runs = 0

    for color, (_, row) in zip(palette, df.iterrows()):
        label = row["plot_label"].replace("\n", ", ")
        run_dir = Path(str(row["run_dir"]))
        history, source = locate_log_history(run_dir, row.get("best_checkpoint"))
        if not history:
            warnings.append(f"No log history found for {row['experiment_name']} under {run_dir}")
            continue

        eval_df = extract_curve_points(history, "eval_wer")
        loss_df = extract_curve_points(history, "loss")
        if eval_df.empty:
            warnings.append(f"No eval_wer entries found for {row['experiment_name']} in {source}")
        else:
            axes[0].plot(
                eval_df["step"],
                eval_df["eval_wer"] * 100.0,
                marker="o",
                markersize=3.2,
                linewidth=1.7,
                color=color,
                label=label,
            )
            plotted_runs += 1

        if loss_df.empty:
            warnings.append(f"No training loss entries found for {row['experiment_name']} in {source}")
        else:
            axes[1].plot(
                loss_df["step"],
                loss_df["loss"],
                linewidth=1.35,
                alpha=0.9,
                color=color,
                label=label,
            )

    axes[0].set_title("Evaluation WER During Fine-tuning")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Evaluation WER (%)")
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.35)
    axes[0].xaxis.grid(True, linestyle=":", alpha=0.2)

    axes[1].set_title("Training Loss During Fine-tuning")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Training loss")
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.35)
    axes[1].xaxis.grid(True, linestyle=":", alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.08))
        fig.subplots_adjust(bottom=0.24)
    else:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No trainer log history found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
        fig.subplots_adjust(bottom=0.12)

    fig.suptitle("Convergence Trajectories Across Hyperparameter Configurations", y=1.02, fontsize=12.5)
    save_figure(fig, output_dir, "figure3_training_curves_eval_wer_and_loss")
    return plotted_runs, warnings


def write_markdown_summary(df: pd.DataFrame, output_dir: Path, csv_source: str, curve_runs: int, warnings: list[str]) -> None:
    best_by_validation = df.loc[df["validation_wer"].idxmin()]
    best_by_test = df.loc[df["test_wer"].idxmin()]

    table_df = df[
        [
            "experiment_name",
            "learning_rate",
            "effective_batch_size",
            "validation_wer",
            "test_wer",
            "validation_cer",
            "test_cer",
        ]
    ].copy()
    for column in ["validation_wer", "test_wer", "validation_cer", "test_cer"]:
        if column in table_df.columns:
            table_df[column] = (table_df[column] * 100.0).map(lambda value: f"{value:.2f}%" if pd.notna(value) else "")
    table_df["learning_rate"] = table_df["learning_rate"].map(format_lr)
    table_df["effective_batch_size"] = table_df["effective_batch_size"].astype(int)
    markdown_table = dataframe_to_markdown(table_df)

    summary = f"""# Academic Figure Summary

Generated from: `{csv_source}`

The grid search compared four Whisper-medium fine-tuning configurations on BAAI/SeniorTalk by varying learning rate and effective batch size. The selected configuration by validation WER was `{best_by_validation['experiment_name']}`, with validation WER {best_by_validation['validation_wer'] * 100:.2f}% and test WER {best_by_validation['test_wer'] * 100:.2f}%. The best configuration by test WER was `{best_by_test['experiment_name']}`, with test WER {best_by_test['test_wer'] * 100:.2f}%.

## Figure Descriptions

Figure 1 compares validation and test WER for each hyperparameter configuration. The grouped bar chart shows that lower learning rate configurations improved generalization, with the strongest validation and test results obtained for learning rate {format_lr(best_by_validation['learning_rate'])} and effective batch size {int(best_by_validation['effective_batch_size'])}.

Figure 2 visualizes the test WER response surface over learning rate and effective batch size. The heatmap highlights the selected low-WER region, making the interaction between optimization step size and effective batch size visually explicit.

Figure 3 summarizes convergence behavior from Hugging Face trainer logs. The left panel plots evaluation WER over training steps for each run, and the right panel plots training loss over steps. This figure supports the paper's training dynamics discussion by showing whether each configuration converged smoothly and how quickly validation WER improved.

## Grid Search Metrics

{markdown_table}

## Generated Files

- `figure1_grouped_validation_test_wer.png` and `figure1_grouped_validation_test_wer.pdf`
- `figure2_test_wer_hyperparameter_heatmap.png` and `figure2_test_wer_hyperparameter_heatmap.pdf`
- `figure3_training_curves_eval_wer_and_loss.png` and `figure3_training_curves_eval_wer_and_loss.pdf`

Trainer log histories were found for {curve_runs} run(s).
"""
    if warnings:
        summary += "\n## Parser Warnings\n\n"
        summary += "\n".join(f"- {warning}" for warning in warnings)
        summary += "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results_section_plot_summary.md").write_text(
        textwrap.dedent(summary).strip() + "\n",
        encoding="utf-8",
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    workspace_root = args.workspace_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workspace_root / "paper_figures"
    )

    configure_plot_style()
    df, csv_source = load_grid_dataframe(workspace_root, args.csv, args.include_failed)
    if df.empty:
        raise SystemExit("No successful grid-search rows were available for plotting.")

    plot_grouped_wer_bars(df, output_dir)
    plot_test_wer_heatmap(df, output_dir)
    curve_runs, warnings = plot_training_curves(df, output_dir)
    write_markdown_summary(df, output_dir, csv_source, curve_runs, warnings)

    print(f"Loaded grid-search results from: {csv_source}")
    print(f"Generated figures and summary under: {output_dir}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
