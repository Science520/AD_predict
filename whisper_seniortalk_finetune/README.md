# Whisper-medium SeniorTalk Overnight Pipeline

This workspace is isolated from the rest of `~/AD_predict/` and keeps heavy data on the HDD-backed path exposed through `data_link/`.

The training script now defaults to full `whisper-medium` fine-tuning on one visible GPU. LoRA remains available as an opt-in flag (`--use-lora`), but it is not the default because the current `ad_env` PEFT/Whisper combination is not stable.

## 1. One-time prep

If `BAAI/SeniorTalk` is gated for your Hugging Face account, log in first:

```bash
huggingface-cli login
```

Then make the shell scripts executable:

```bash
cd ~/AD_predict/whisper_seniortalk_finetune
chmod +x setup_workspace.sh run_overnight.sh
```

## 2. Initialize the workspace

The default HDD path is `/mnt/hdd/datasets/BAAI_SeniorTalk`. If you want a different disk location, export `PHYSICAL_DATA_ROOT` first or pass `--physical-data-root`.

If your server reaches Hugging Face through a mirror, export it before setup and before the tmux run:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

```bash
cd ~/AD_predict/whisper_seniortalk_finetune
./setup_workspace.sh --install-missing
```

What this does:

- Creates `data_link -> /mnt/hdd/datasets/BAAI_SeniorTalk`
- Creates `outputs/`, `reports/`, `logs/`, and `best_model_export/`
- Checks the required Python packages inside `ad_env`
- Warns if your Hugging Face token is missing

## 3. Optional manual dry-run

If you want to verify each stage before leaving it overnight:

```bash
conda activate ad_env
python prepare_dataset.py
python train_and_search.py
```

The prepared dataset is saved under:

```text
~/AD_predict/whisper_seniortalk_finetune/data_link/processed/seniortalk_whisper_medium
```

## 4. Overnight unattended run

Start the full tmux job:

```bash
cd ~/AD_predict/whisper_seniortalk_finetune
./run_overnight.sh --install-missing
```

This creates a tmux session named `asr_tune`, activates `ad_env`, runs:

1. `setup_workspace.sh`
2. `prepare_dataset.py`
3. `train_and_search.py`

and writes the full console stream to:

```text
~/AD_predict/whisper_seniortalk_finetune/training_overnight.log
```

## 5. Monitoring and outputs

Monitor the job:

```bash
tmux attach -t asr_tune
tail -f ~/AD_predict/whisper_seniortalk_finetune/training_overnight.log
```

Key outputs after training:

- `reports/grid_search_summary.csv`
- `reports/grid_search_summary.md`
- `outputs/runs/<experiment_name>/`
- `best_model_export/`

The best model is selected by the lowest validation `wer`, and the export folder will contain the best checkpoint summary plus a merged standalone model when LoRA is enabled.
