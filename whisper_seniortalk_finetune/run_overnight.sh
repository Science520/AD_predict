#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$SCRIPT_DIR}"
PHYSICAL_DATA_ROOT="${PHYSICAL_DATA_ROOT:-/data/saisai/BAAI_SeniorTalk}"
SESSION_NAME="${SESSION_NAME:-asr_tune}"
GPU_ID="${GPU_ID:-0}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ad_env}"
INSTALL_MISSING=0

usage() {
  cat <<'EOF'
Usage: ./run_overnight.sh [options]

Options:
  --session-name NAME       tmux session name. Default: asr_tune
  --physical-data-root PATH HDD-backed dataset location.
  --gpu ID                  CUDA_VISIBLE_DEVICES value. Default: 0
  --install-missing         Run setup_workspace.sh --install-missing.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --physical-data-root)
      PHYSICAL_DATA_ROOT="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --install-missing)
      INSTALL_MISSING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed." >&2
  exit 1
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda activation script not found: $CONDA_SH" >&2
  exit 1
fi

for required in setup_workspace.sh prepare_dataset.py train_and_search.py; do
  if [[ ! -e "$WORKSPACE_ROOT/$required" ]]; then
    echo "Missing required file: $WORKSPACE_ROOT/$required" >&2
    exit 1
  fi
done

mkdir -p "$WORKSPACE_ROOT/logs"
LOG_FILE="$WORKSPACE_ROOT/training_overnight.log"

if [[ -f "$LOG_FILE" ]]; then
  mv "$LOG_FILE" "$WORKSPACE_ROOT/logs/training_overnight_$(date -u +%Y%m%dT%H%M%SZ).log"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  echo "Attach with: tmux attach -t $SESSION_NAME" >&2
  echo "Or stop it with: tmux kill-session -t $SESSION_NAME" >&2
  exit 1
fi

SETUP_INSTALL_ARG=""
if [[ "$INSTALL_MISSING" -eq 1 ]]; then
  SETUP_INSTALL_ARG="--install-missing"
fi

tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "cd \"$WORKSPACE_ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME" "exec > >(tee -a \"$LOG_FILE\") 2>&1" C-m
tmux send-keys -t "$SESSION_NAME" "set -Eeuo pipefail" C-m
tmux send-keys -t "$SESSION_NAME" "echo \"[START] \$(date -u +%Y-%m-%dT%H:%M:%SZ)\"" C-m
tmux send-keys -t "$SESSION_NAME" "echo \"Workspace: $WORKSPACE_ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME" "echo \"Physical data root: $PHYSICAL_DATA_ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME" "echo \"CUDA_VISIBLE_DEVICES: $GPU_ID\"" C-m
tmux send-keys -t "$SESSION_NAME" "source \"$CONDA_SH\"" C-m
tmux send-keys -t "$SESSION_NAME" "conda activate \"$CONDA_ENV_NAME\"" C-m
tmux send-keys -t "$SESSION_NAME" "export WORKSPACE_ROOT=\"$WORKSPACE_ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME" "export PHYSICAL_DATA_ROOT=\"$PHYSICAL_DATA_ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES=\"$GPU_ID\"" C-m
tmux send-keys -t "$SESSION_NAME" "export PYTHONUNBUFFERED=1" C-m
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  tmux send-keys -t "$SESSION_NAME" "export HF_ENDPOINT=\"$HF_ENDPOINT\"" C-m
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  tmux send-keys -t "$SESSION_NAME" "export HF_TOKEN=\"$HF_TOKEN\"" C-m
fi
tmux send-keys -t "$SESSION_NAME" "nvidia-smi || true" C-m
tmux send-keys -t "$SESSION_NAME" "./setup_workspace.sh --physical-data-root \"$PHYSICAL_DATA_ROOT\" $SETUP_INSTALL_ARG" C-m
tmux send-keys -t "$SESSION_NAME" "python prepare_dataset.py --workspace-root \"$WORKSPACE_ROOT\" --physical-data-root \"$PHYSICAL_DATA_ROOT\" --overwrite" C-m
tmux send-keys -t "$SESSION_NAME" "python train_and_search.py --workspace-root \"$WORKSPACE_ROOT\" --physical-data-root \"$PHYSICAL_DATA_ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME" "echo \"[END] \$(date -u +%Y-%m-%dT%H:%M:%SZ)\"" C-m

cat <<EOF
Started tmux session: $SESSION_NAME
Log file: $LOG_FILE

Useful commands:
  tmux attach -t $SESSION_NAME
  tail -f $LOG_FILE
  tmux kill-session -t $SESSION_NAME
EOF
