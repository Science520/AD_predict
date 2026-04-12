#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$SCRIPT_DIR}"
PHYSICAL_DATA_ROOT="${PHYSICAL_DATA_ROOT:-/data/saisai/BAAI_SeniorTalk}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ad_env}"
INSTALL_MISSING=0
CLEAN_GENERATED=0

usage() {
  cat <<'EOF'
Usage: ./setup_workspace.sh [options]

Options:
  --physical-data-root PATH  Store raw and processed dataset artifacts here.
  --install-missing          Install missing Python dependencies into ad_env.
  --clean                    Remove generated outputs inside the workspace first.
  -h, --help                 Show this help.

Environment overrides:
  WORKSPACE_ROOT
  PHYSICAL_DATA_ROOT
  CONDA_SH
  CONDA_ENV_NAME
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --physical-data-root)
      PHYSICAL_DATA_ROOT="$2"
      shift 2
      ;;
    --install-missing)
      INSTALL_MISSING=1
      shift
      ;;
    --clean)
      CLEAN_GENERATED=1
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

LOG_DIR="$WORKSPACE_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup_workspace.log"
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

echo "[$(timestamp)] Initializing Whisper SeniorTalk workspace"
echo "[$(timestamp)] Workspace root: $WORKSPACE_ROOT"
echo "[$(timestamp)] Physical data root: $PHYSICAL_DATA_ROOT"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda activation script not found: $CONDA_SH" >&2
  exit 1
fi

if [[ "$CLEAN_GENERATED" -eq 1 ]]; then
  echo "[$(timestamp)] Removing generated workspace artifacts"
  rm -rf \
    "$WORKSPACE_ROOT/best_model_export" \
    "$WORKSPACE_ROOT/outputs" \
    "$WORKSPACE_ROOT/reports" \
    "$WORKSPACE_ROOT/tmp"
  rm -f "$WORKSPACE_ROOT/training_overnight.log"
fi

mkdir -p \
  "$WORKSPACE_ROOT/best_model_export" \
  "$WORKSPACE_ROOT/logs" \
  "$WORKSPACE_ROOT/outputs/runs" \
  "$WORKSPACE_ROOT/reports" \
  "$WORKSPACE_ROOT/tmp"

mkdir -p \
  "$PHYSICAL_DATA_ROOT/hf_cache/datasets" \
  "$PHYSICAL_DATA_ROOT/hf_cache/hub" \
  "$PHYSICAL_DATA_ROOT/hf_cache/transformers" \
  "$PHYSICAL_DATA_ROOT/raw_repo" \
  "$PHYSICAL_DATA_ROOT/processed" \
  "$PHYSICAL_DATA_ROOT/manifests"

if [[ -L "$WORKSPACE_ROOT/data_link" ]]; then
  CURRENT_TARGET="$(readlink -f "$WORKSPACE_ROOT/data_link")"
  EXPECTED_TARGET="$(readlink -f "$PHYSICAL_DATA_ROOT")"
  if [[ "$CURRENT_TARGET" != "$EXPECTED_TARGET" ]]; then
    echo "Existing data_link points to $CURRENT_TARGET, expected $EXPECTED_TARGET" >&2
    exit 1
  fi
elif [[ -e "$WORKSPACE_ROOT/data_link" ]]; then
  echo "data_link exists but is not a symlink: $WORKSPACE_ROOT/data_link" >&2
  exit 1
else
  ln -s "$PHYSICAL_DATA_ROOT" "$WORKSPACE_ROOT/data_link"
fi

echo "[$(timestamp)] data_link -> $(readlink -f "$WORKSPACE_ROOT/data_link")"

for cmd in tmux python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[$(timestamp)] GPU status"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
  echo "[$(timestamp)] Warning: nvidia-smi not found"
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

echo "[$(timestamp)] Active Python: $(command -v python)"
echo "[$(timestamp)] Active conda env: ${CONDA_DEFAULT_ENV:-unknown}"

if ! python - <<'PY'
import json
import os
from importlib import import_module

required = {
    "transformers": "required",
    "datasets": "required",
    "librosa": "required",
    "soundfile": "required",
    "evaluate": "required",
    "jiwer": "required",
    "accelerate": "required",
    "peft": "required",
    "torch": "required",
    "huggingface_hub": "required",
}

results = {"missing": [], "versions": {}, "hf_token_present": False}

for name in required:
    try:
        module = import_module(name)
        results["versions"][name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        results["missing"].append({"package": name, "error": str(exc)})

try:
    from huggingface_hub import get_token

    token = get_token()
    results["hf_token_present"] = bool(token or os.environ.get("HF_TOKEN"))
except Exception:
    results["hf_token_present"] = bool(os.environ.get("HF_TOKEN"))

print(json.dumps(results, ensure_ascii=True))
raise SystemExit(1 if results["missing"] else 0)
PY
then
  MISSING_PACKAGES="$(python - <<'PY'
import json
import os
from importlib import import_module

packages = [
    "transformers",
    "datasets",
    "librosa",
    "soundfile",
    "evaluate",
    "jiwer",
    "accelerate",
    "peft",
    "torch",
    "huggingface_hub",
]

missing = []
for name in packages:
    try:
        import_module(name)
    except Exception:
        missing.append(name)

print(" ".join(missing))
PY
)"
  echo "[$(timestamp)] Missing Python packages: ${MISSING_PACKAGES:-none}"
  if [[ "$INSTALL_MISSING" -eq 1 && -n "$MISSING_PACKAGES" ]]; then
    echo "[$(timestamp)] Installing missing dependencies into $CONDA_ENV_NAME"
    python -m pip install --upgrade $MISSING_PACKAGES
  else
    echo "Use --install-missing to install the missing packages automatically." >&2
    exit 1
  fi
fi

python - <<'PY'
import json
import os
from importlib import import_module

packages = [
    "transformers",
    "datasets",
    "librosa",
    "soundfile",
    "evaluate",
    "jiwer",
    "accelerate",
    "peft",
    "torch",
    "huggingface_hub",
]

report = {}
for name in packages:
    module = import_module(name)
    report[name] = getattr(module, "__version__", "unknown")

from huggingface_hub import get_token

report["hf_token_present"] = bool(get_token() or os.environ.get("HF_TOKEN"))
print(json.dumps(report, indent=2, ensure_ascii=True))
PY

HF_TOKEN_PRESENT="$(python - <<'PY'
import os
from huggingface_hub import get_token

print("yes" if (get_token() or os.environ.get("HF_TOKEN")) else "no")
PY
)"

if [[ "$HF_TOKEN_PRESENT" != "yes" ]]; then
  echo "[$(timestamp)] Warning: no Hugging Face token detected."
  echo "[$(timestamp)] If BAAI/SeniorTalk is gated for your account, run: huggingface-cli login"
fi

cat > "$WORKSPACE_ROOT/README_SETUP_HINTS.txt" <<EOF
Workspace root: $WORKSPACE_ROOT
Physical data root: $PHYSICAL_DATA_ROOT
HF cache root: $PHYSICAL_DATA_ROOT/hf_cache
Processed dataset path: $PHYSICAL_DATA_ROOT/processed/seniortalk_whisper_medium

If BAAI/SeniorTalk access is gated in your account, run:
  huggingface-cli login

Then start the unattended job with:
  cd $WORKSPACE_ROOT
  ./run_overnight.sh
EOF

echo "[$(timestamp)] Workspace setup completed"
