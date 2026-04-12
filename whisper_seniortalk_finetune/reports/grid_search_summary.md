# Whisper-medium SeniorTalk Grid Search

| Experiment | Status | LR | Batch | Grad Accum | Eff Batch | Val WER | Test WER | Best Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_1e-05_effective_batch_16 | success | 1e-05 | 1 | 16 | 16 | 0.169296822274235 | 0.20801077078424773 | /home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_1e-05_effective_batch_16/checkpoint-2000 |
| lr_1e-05_effective_batch_32 | success | 1e-05 | 1 | 32 | 32 | 0.16292126764406026 | 0.20266928131356737 | /home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_1e-05_effective_batch_32/checkpoint-1000 |
| lr_5e-06_effective_batch_16 | success | 5e-06 | 1 | 16 | 16 | 0.15719188409126808 | 0.1736935302123425 | /home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_5e-06_effective_batch_16/checkpoint-2955 |
| lr_5e-06_effective_batch_32 | success | 5e-06 | 1 | 32 | 32 | 0.15871397596243592 | 0.18499114629827462 | /home/saisai/AD_predict/whisper_seniortalk_finetune/outputs/runs/lr_5e-06_effective_batch_32/checkpoint-1478 |
