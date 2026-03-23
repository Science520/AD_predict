#!/usr/bin/env python3
"""Train a multimodal PCBM + BiLSTM-CRF model for ADCeleb."""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover - depends on sklearn version
    StratifiedGroupKFold = None

try:
    from torchcrf import CRF as TorchCRF
except Exception:  # pragma: no cover - optional dependency
    TorchCRF = None

DEFAULT_INPUT_PKL = Path("~/AD_predict/text_cbm_pipeline/adceleb_multimodal_features.pkl").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/AD_predict/text_cbm_pipeline").expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multimodal PCBM + BiLSTM-CRF sequence classifier on ADCeleb."
    )
    parser.add_argument("--input-pkl", type=Path, default=DEFAULT_INPUT_PKL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cv-mode", choices=["group5", "loso"], default="group5")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--print-top-k", type=int, default=8)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class SequenceExample:
    speaker_id: str
    label: int
    features: np.ndarray
    acoustic_vectors: np.ndarray
    text_concepts: np.ndarray
    temporal_orders: np.ndarray
    video_ids: List[str]


class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[SequenceExample]):
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> SequenceExample:
        return self.sequences[index]


def collate_sequences(batch: Sequence[SequenceExample]) -> Dict[str, object]:
    batch_size = len(batch)
    max_len = max(example.features.shape[0] for example in batch)
    feature_dim = batch[0].features.shape[1]

    features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    labels = torch.tensor([example.label for example in batch], dtype=torch.long)
    speaker_ids = [example.speaker_id for example in batch]

    for idx, example in enumerate(batch):
        seq_len = example.features.shape[0]
        features[idx, :seq_len] = torch.from_numpy(example.features).float()
        mask[idx, :seq_len] = True

    return {
        "features": features,
        "mask": mask,
        "labels": labels,
        "speaker_ids": speaker_ids,
        "examples": batch,
    }


class LinearChainCRF(nn.Module):
    """A small batch-first linear-chain CRF fallback when torchcrf is unavailable."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        for timestep in range(1, seq_len):
            emit_t = emissions[:, timestep].unsqueeze(1)
            next_score = torch.logsumexp(
                score.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_t,
                dim=1,
            )
            score = torch.where(mask[:, timestep].unsqueeze(1), next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        batch_indices = torch.arange(batch_size, device=emissions.device)

        first_tags = tags[:, 0]
        score = self.start_transitions[first_tags] + emissions[:, 0, :][batch_indices, first_tags]

        for timestep in range(1, seq_len):
            current_tags = tags[:, timestep]
            previous_tags = tags[:, timestep - 1]
            emit_score = emissions[:, timestep, :][batch_indices, current_tags]
            transition_score = self.transitions[previous_tags, current_tags]
            score = score + (emit_score + transition_score) * mask[:, timestep]

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags[batch_indices, lengths]
        score = score + self.end_transitions[last_tags]
        return score

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._score_sentence(emissions, tags, mask) - self._compute_log_partition(emissions, mask)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history: List[torch.Tensor] = []

        for timestep in range(1, seq_len):
            broadcast_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_paths = broadcast_score.max(dim=1)
            best_score = best_score + emissions[:, timestep]
            score = torch.where(mask[:, timestep].unsqueeze(1), best_score, score)
            history.append(best_paths)

        score = score + self.end_transitions
        best_last_tags = score.argmax(dim=1)

        decoded: List[List[int]] = []
        lengths = mask.long().sum(dim=1).tolist()
        for batch_idx, seq_len_item in enumerate(lengths):
            best_tag = best_last_tags[batch_idx].item()
            best_path = [best_tag]
            for backpointer_t in reversed(history[: seq_len_item - 1]):
                best_tag = backpointer_t[batch_idx, best_tag].item()
                best_path.append(best_tag)
            decoded.append(list(reversed(best_path)))
        return decoded


def score_tags_with_transitions(
    emissions: torch.Tensor,
    tags: torch.Tensor,
    mask: torch.Tensor,
    start_transitions: torch.Tensor,
    transitions: torch.Tensor,
    end_transitions: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, _ = emissions.shape
    batch_indices = torch.arange(batch_size, device=emissions.device)
    first_tags = tags[:, 0]
    score = start_transitions[first_tags] + emissions[:, 0, :][batch_indices, first_tags]
    for timestep in range(1, seq_len):
        current_tags = tags[:, timestep]
        previous_tags = tags[:, timestep - 1]
        emit_score = emissions[:, timestep, :][batch_indices, current_tags]
        transition_score = transitions[previous_tags, current_tags]
        score = score + (emit_score + transition_score) * mask[:, timestep]
    lengths = mask.long().sum(dim=1) - 1
    last_tags = tags[batch_indices, lengths]
    score = score + end_transitions[last_tags]
    return score


class MultimodalCRFPCBM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        hidden_dim: int,
        dropout: float,
        use_external_crf: bool,
    ):
        super().__init__()
        self.bottleneck = nn.Linear(input_dim, bottleneck_dim)
        self.lstm = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.emission_head = nn.Linear(hidden_dim * 2, 2)
        if use_external_crf and TorchCRF is not None:
            self.crf = TorchCRF(num_tags=2, batch_first=True)
            self.uses_external_crf = True
        else:
            self.crf = LinearChainCRF(num_tags=2)
            self.uses_external_crf = False

    def encode(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = mask.long().sum(dim=1).cpu()
        concept_activations = torch.sigmoid(self.bottleneck(features))
        packed = pack_padded_sequence(
            concept_activations,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=features.size(1),
        )
        emissions = self.emission_head(self.dropout(lstm_output))
        return concept_activations, emissions

    def neg_log_likelihood(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        concepts, emissions = self.encode(features, mask)
        repeated_tags = labels.unsqueeze(1).expand(-1, features.size(1))
        if self.uses_external_crf:
            log_likelihood = self.crf(emissions, repeated_tags, mask=mask, reduction="mean")
        else:
            log_likelihood = self.crf(emissions, repeated_tags, mask).mean()
        return -log_likelihood, concepts, emissions

    def sequence_probabilities(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], torch.Tensor]:
        concepts, emissions = self.encode(features, mask)
        device = emissions.device
        batch_size, seq_len, _ = emissions.shape

        zero_tags = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        one_tags = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

        if self.uses_external_crf:
            start_transitions = self.crf.start_transitions
            transitions = self.crf.transitions
            end_transitions = self.crf.end_transitions
            decoded = self.crf.decode(emissions, mask=mask)
        else:
            start_transitions = self.crf.start_transitions
            transitions = self.crf.transitions
            end_transitions = self.crf.end_transitions
            decoded = self.crf.decode(emissions, mask)

        cn_score = score_tags_with_transitions(
            emissions, zero_tags, mask, start_transitions, transitions, end_transitions
        )
        ad_score = score_tags_with_transitions(
            emissions, one_tags, mask, start_transitions, transitions, end_transitions
        )
        sequence_probs = torch.softmax(torch.stack([cn_score, ad_score], dim=1), dim=1)[:, 1]
        sequence_preds = (sequence_probs >= 0.5).long()
        return sequence_preds, sequence_probs, decoded, concepts


def load_sequences(input_pkl: Path, max_speakers: int | None = None) -> Tuple[List[SequenceExample], int, List[str]]:
    with input_pkl.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        df = payload["dataframe"]
        concept_names = list(payload.get("concept_names", []))
        acoustic_dim = int(payload.get("acoustic_dim", 0))
    else:
        df = payload
        concept_names = []
        acoustic_dim = 0

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected the pickle payload to contain a pandas DataFrame.")

    df = df.copy()
    df.sort_values(by=["Speaker_ID", "Temporal_Order"], inplace=True, kind="stable")

    sequences: List[SequenceExample] = []
    for speaker_id, group_df in df.groupby("Speaker_ID", sort=False):
        acoustic_vectors = np.stack(
            [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in group_df["Acoustic_Vector"]]
        )
        text_concepts = np.stack(
            [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in group_df["Text_Concepts_Vector"]]
        )
        features = np.concatenate([acoustic_vectors, text_concepts], axis=1).astype(np.float32)
        sequences.append(
            SequenceExample(
                speaker_id=str(speaker_id),
                label=int(group_df["Label"].iloc[0]),
                features=features,
                acoustic_vectors=acoustic_vectors,
                text_concepts=text_concepts,
                temporal_orders=group_df["Temporal_Order"].to_numpy(dtype=np.int64),
                video_ids=group_df["Video_ID"].astype(str).tolist(),
            )
        )

    if max_speakers is not None:
        sequences = sequences[:max_speakers]

    if acoustic_dim <= 0 and sequences:
        acoustic_dim = int(sequences[0].acoustic_vectors.shape[1])
    return sequences, acoustic_dim, concept_names


def build_splits(sequences: Sequence[SequenceExample], cv_mode: str, n_splits: int):
    labels = np.asarray([sequence.label for sequence in sequences], dtype=int)
    groups = np.asarray([sequence.speaker_id for sequence in sequences])
    indices = np.arange(len(sequences))

    if cv_mode == "loso":
        splitter = LeaveOneGroupOut()
        return list(splitter.split(indices, labels, groups))

    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(splitter.split(indices, labels, groups))

    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(indices, labels, groups))


def run_model_on_loader(
    model: MultimodalCRFPCBM,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_decoded_paths: List[List[int]] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].cpu().numpy().tolist()

            _, probs, decoded_paths, _ = model.sequence_probabilities(features, mask)
            all_labels.extend(labels)
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_decoded_paths.extend(decoded_paths)

    return np.asarray(all_labels), np.asarray(all_probs), all_decoded_paths


def train_single_fold(
    train_sequences: Sequence[SequenceExample],
    test_sequences: Sequence[SequenceExample],
    acoustic_dim: int,
    concept_names: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, float], pd.DataFrame, MultimodalCRFPCBM]:
    train_loader = DataLoader(
        SequenceDataset(train_sequences),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
    )
    test_loader = DataLoader(
        SequenceDataset(test_sequences),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )

    input_dim = train_sequences[0].features.shape[1]
    model = MultimodalCRFPCBM(
        input_dim=input_dim,
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_external_crf=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in progress_bar:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model.neg_log_likelihood(features, mask, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}: mean_train_loss={mean_loss:.4f}")

    y_true, y_prob, decoded_paths = run_model_on_loader(model, test_loader, device)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "test_size": float(len(y_true)),
    }

    prediction_rows = []
    for example, prob, decoded_path in zip(test_sequences, y_prob.tolist(), decoded_paths):
        decoded_labels = ["AD" if tag == 1 else "CN" for tag in decoded_path]
        prediction_rows.append(
            {
                "Speaker_ID": example.speaker_id,
                "Label": example.label,
                "Pred_Prob_AD": float(prob),
                "Pred_Label": int(prob >= 0.5),
                "Seq_Len": int(example.features.shape[0]),
                "Decoded_Path": " ".join(decoded_labels),
                "Video_ID_Path": " ".join(example.video_ids),
                "Temporal_Order_Path": " ".join(map(str, example.temporal_orders.tolist())),
            }
        )

    prediction_df = pd.DataFrame(prediction_rows)
    return metrics, prediction_df, model


def train_fold_with_fallback(
    train_sequences: Sequence[SequenceExample],
    test_sequences: Sequence[SequenceExample],
    acoustic_dim: int,
    concept_names: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
):
    try:
        return train_single_fold(
            train_sequences=train_sequences,
            test_sequences=test_sequences,
            acoustic_dim=acoustic_dim,
            concept_names=concept_names,
            args=args,
            device=device,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device.type == "cuda":
            print("Warning: CUDA OOM during training. Retrying this fold on CPU.")
            torch.cuda.empty_cache()
            return train_single_fold(
                train_sequences=train_sequences,
                test_sequences=test_sequences,
                acoustic_dim=acoustic_dim,
                concept_names=concept_names,
                args=args,
                device=torch.device("cpu"),
            )
        raise


def print_bottleneck_summary(
    model: MultimodalCRFPCBM,
    acoustic_dim: int,
    concept_names: Sequence[str],
    top_k: int,
) -> None:
    weights = model.bottleneck.weight.detach().cpu().numpy()
    text_weights = weights[:, acoustic_dim:]

    print("\nBottleneck concept units (text side summary)")
    for concept_idx in range(weights.shape[0]):
        row = text_weights[concept_idx]
        ranking = np.argsort(np.abs(row))[::-1][:top_k]
        formatted = ", ".join(
            f"{concept_names[col_idx]}={row[col_idx]:+.3f}"
            for col_idx in ranking
            if col_idx < len(concept_names)
        )
        acoustic_norm = float(np.linalg.norm(weights[concept_idx, :acoustic_dim]))
        print(f"  bottleneck_{concept_idx:02d} | acoustic_l2={acoustic_norm:.3f} | {formatted}")


def print_crf_transitions(model: MultimodalCRFPCBM) -> None:
    if model.uses_external_crf:
        transitions = model.crf.transitions.detach().cpu().numpy()
        start = model.crf.start_transitions.detach().cpu().numpy()
        end = model.crf.end_transitions.detach().cpu().numpy()
    else:
        transitions = model.crf.transitions.detach().cpu().numpy()
        start = model.crf.start_transitions.detach().cpu().numpy()
        end = model.crf.end_transitions.detach().cpu().numpy()

    tag_names = ["CN", "AD"]
    print("\nCRF transition matrix")
    for src_idx, src_name in enumerate(tag_names):
        row = ", ".join(
            f"{src_name}->{dst_name}={transitions[src_idx, dst_idx]:+.3f}"
            for dst_idx, dst_name in enumerate(tag_names)
        )
        print(f"  {row}")
    print(f"  start={dict(zip(tag_names, start.round(3).tolist()))}")
    print(f"  end={dict(zip(tag_names, end.round(3).tolist()))}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    sequences, acoustic_dim, concept_names = load_sequences(args.input_pkl, args.max_speakers)
    if not sequences:
        raise RuntimeError("No speaker sequences found in the input pickle.")

    device = resolve_device(args.device)
    splits = build_splits(sequences, args.cv_mode, args.n_splits)
    print(f"Loaded {len(sequences)} speaker sequences.")
    print(f"Using device: {device}")
    print(f"Using external torchcrf: {TorchCRF is not None}")
    print(f"Number of folds: {len(splits)}")

    fold_rows: List[Dict[str, float]] = []
    prediction_frames: List[pd.DataFrame] = []
    last_model: MultimodalCRFPCBM | None = None

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        train_sequences = [sequences[index] for index in train_idx]
        test_sequences = [sequences[index] for index in test_idx]
        print(
            f"\nFold {fold_idx}/{len(splits)} | "
            f"train_speakers={len(train_sequences)} | test_speakers={len(test_sequences)}"
        )

        metrics, prediction_df, model = train_fold_with_fallback(
            train_sequences=train_sequences,
            test_sequences=test_sequences,
            acoustic_dim=acoustic_dim,
            concept_names=concept_names,
            args=args,
            device=device,
        )
        metrics["fold"] = fold_idx
        fold_rows.append(metrics)
        prediction_df["Fold"] = fold_idx
        prediction_frames.append(prediction_df)
        last_model = model

        print(
            f"Fold {fold_idx}: accuracy={metrics['accuracy']:.4f} "
            f"auc={metrics['auc']:.4f}"
        )
        print_bottleneck_summary(model, acoustic_dim, concept_names, args.print_top_k)
        print_crf_transitions(model)

    fold_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    overall_accuracy = float(
        accuracy_score(predictions_df["Label"], predictions_df["Pred_Label"])
    )
    overall_auc = float(
        roc_auc_score(predictions_df["Label"], predictions_df["Pred_Prob_AD"])
    ) if predictions_df["Label"].nunique() > 1 else float("nan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.output_dir / "multimodal_crf_fold_metrics.csv"
    pred_path = args.output_dir / "multimodal_crf_sequence_predictions.csv"
    fold_df.to_csv(fold_path, index=False)
    predictions_df.to_csv(pred_path, index=False)

    print("\n" + "=" * 80)
    print("Overall Evaluation")
    print("=" * 80)
    print(f"Sequence-level Accuracy: {overall_accuracy:.4f}")
    print(f"Sequence-level AUC:      {overall_auc:.4f}")
    print(f"Saved fold metrics to:   {fold_path}")
    print(f"Saved predictions to:    {pred_path}")

    if last_model is not None:
        print_bottleneck_summary(last_model, acoustic_dim, concept_names, args.print_top_k)
        print_crf_transitions(last_model)


if __name__ == "__main__":
    main()
