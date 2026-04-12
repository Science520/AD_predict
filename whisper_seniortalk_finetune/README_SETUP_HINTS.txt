Workspace root: /home/saisai/AD_predict/whisper_seniortalk_finetune
Physical data root: /data/saisai/BAAI_SeniorTalk
HF cache root: /data/saisai/BAAI_SeniorTalk/hf_cache
Processed dataset path: /data/saisai/BAAI_SeniorTalk/processed/seniortalk_whisper_medium

If BAAI/SeniorTalk access is gated in your account, run:
  huggingface-cli login

Then start the unattended job with:
  cd /home/saisai/AD_predict/whisper_seniortalk_finetune
  ./run_overnight.sh
