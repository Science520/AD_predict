# Academic Figure Summary

Generated from: `/home/saisai/AD_predict/whisper_seniortalk_finetune/reports/grid_search_summary.csv`

The grid search compared four Whisper-medium fine-tuning configurations on BAAI/SeniorTalk by varying learning rate and effective batch size. The selected configuration by validation WER was `lr_5e-06_effective_batch_16`, with validation WER 15.72% and test WER 17.37%. The best configuration by test WER was `lr_5e-06_effective_batch_16`, with test WER 17.37%. The zero-shot `openai/whisper-medium` baseline test WER is 47.98% from `/home/saisai/AD_predict/whisper_seniortalk_finetune/reports/baseline_result.json`. The best fine-tuned model reduces test WER by 63.8% relative to this baseline.

## Figure Descriptions

Figure 1 compares the zero-shot baseline test WER against validation and test WER for each fine-tuned hyperparameter configuration. The grouped bar chart shows that lower learning rate configurations improved generalization, with the strongest validation and test results obtained for learning rate 5e-6 and effective batch size 16.

Figure 2 visualizes the test WER response surface over learning rate and effective batch size. The heatmap highlights the selected low-WER region, making the interaction between optimization step size and effective batch size visually explicit.

Figure 3 summarizes convergence behavior from Hugging Face trainer logs. The left panel plots evaluation WER over training steps for each run and includes the zero-shot baseline test WER as a dashed horizontal reference line; the right panel plots training loss over steps. This figure supports the paper's training dynamics discussion by showing whether each configuration converged smoothly and how quickly validation WER improved.

## Grid Search Metrics

| experiment_name | learning_rate | effective_batch_size | validation_wer | test_wer | validation_cer | test_cer |
| --- | --- | --- | --- | --- | --- | --- |
| lr_1e-05_effective_batch_16 | 1e-5 | 16 | 16.93% | 20.80% | 16.93% | 20.80% |
| lr_1e-05_effective_batch_32 | 1e-5 | 32 | 16.29% | 20.27% | 16.30% | 20.27% |
| lr_5e-06_effective_batch_16 | 5e-6 | 16 | 15.72% | 17.37% | 15.72% | 17.37% |
| lr_5e-06_effective_batch_32 | 5e-6 | 32 | 15.87% | 18.50% | 15.87% | 18.50% |

## Generated Files

- `figure1_grouped_validation_test_wer.png` and `figure1_grouped_validation_test_wer.pdf`
- `figure2_test_wer_hyperparameter_heatmap.png` and `figure2_test_wer_hyperparameter_heatmap.pdf`
- `figure3_training_curves_eval_wer_and_loss.png` and `figure3_training_curves_eval_wer_and_loss.pdf`

Trainer log histories were found for 4 run(s).
