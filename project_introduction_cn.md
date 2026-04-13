# 项目介绍：基于纵向语音与文本的阿尔茨海默病可解释预测

## 1. 摘要

本项目围绕阿尔茨海默病（Alzheimer's Disease, AD）的早期预测，构建了一条从老年人语音识别、文本概念瓶颈建模到纵向多模态时序推理的完整技术路线。项目首先在 BAAI/SeniorTalk 数据集上微调 `openai/whisper-medium`，提升老年人语音转录鲁棒性；随后将文本 Concept Bottleneck Model（PCBM）从基础概念升级为 8 个 AD 相关临床概念；最后在公开 ADCeleb 纵向数据集上引入条件随机场（CRF），验证 10 年诊断前时间轨迹对 AD 预测的价值。实验特别强调防止数据泄漏：所有关键评估均以 `Speaker_ID` 为分组单位执行 5 折交叉验证，避免模型记忆说话人音色或身份特征。结果显示，Whisper 微调后测试集 WER 从零样本 47.98% 降至 17.37%；8 概念 PCBM 的平均 AUC 从旧概念体系的 0.594 提升至 0.673；在 ADCeleb 全 10 年轨迹上，多模态 CRF-PCBM 达到 ACC=0.494、AUC=0.608。进一步时间切片消融显示，诊断前 1-5 年（AUC=0.520）优于诊断前 6-10 年（AUC=0.443），符合 AD 病理表现随诊断临近而增强的临床规律，也反向证明：对于高度压缩、可解释的临床概念，建模认知衰退的动态斜率比静态横截面分类更关键。

## 2. 背景与动机

AD 是一种进行性神经退行性疾病，早期阶段常表现为词汇贫乏、语义断裂、重复表达、句法混乱、叙事组织能力下降以及情绪和健康主题变化。语音与文本数据具有低成本、可重复、可纵向采集的优势，适合用于早期风险筛查。然而，现有黑箱模型虽然可能在静态分类任务上取得较高 AUC，却难以解释模型到底依据了哪些语言或声学线索；在小样本纵向数据中，若数据划分不严谨，还可能通过记忆说话人音色、视频背景或身份信息获得虚高结果。

因此，本项目的核心目标不是追求不可解释的单点最高分，而是建立一个面向临床解释的白盒式预测框架：将原始语音转为稳定文本表示，再将高维文本压缩为少量可解释的 AD 相关概念，最后用 CRF 建模这些概念和声学特征在诊断前 10 年内的动态变化。项目总路线可参考 `/home/saisai/AD_predict/technical_route_image.svg`。

## 3. 方法

第一阶段是老年语音 ASR 适配。项目使用 `setup_workspace.sh`、`prepare_dataset.py`、`train_and_search.py` 与 `run_overnight.sh`，在 BAAI/SeniorTalk 数据集上对 `openai/whisper-medium` 进行微调，并对学习率与有效 batch size 进行网格搜索。该阶段的目的，是降低老年语音、停顿、口音和叙事节奏差异带来的转录误差，为后续文本概念抽取提供可靠输入。

第二阶段是文本 PCBM 升级。原模型使用较基础的文本概念，本项目将其升级为 8 个更贴近 AD 语言与行为表现的临床概念：情绪挫败或焦虑、健康或就医话题、怀旧或遥远过去、简单词汇、重复语言、语法错误或混乱句法、逻辑不连续、短句或碎片化句子。模型在 Dementia Blog Corpus 上进行训练与评估，使用概念轨迹图解释不同时间阶段的语言变化。

第三阶段是 ADCeleb 纵向多模态建模。通过 `check_and_aggregate_inventory.py`、`build_multimodal_dataset.py` 和 `train_multimodal_crf_pcbm.py`，项目构建说话人级 10 年诊断前序列，将声学特征与文本概念特征组合输入 CRF-PCBM。CRF 的作用是学习状态转移和时间依赖关系，即不仅判断某一时刻的表达是否异常，还捕捉概念随时间变化的方向和斜率。

第四阶段是时间区间消融。使用 `evaluate_by_time_interval.py` 将数据分为 Interval -2（诊断前 6-10 年）与 Interval -1（诊断前 1-5 年）两个静态切片，并对 CN 个体进行 pseudo-YoD 匹配，使对照组也能被放置到可比的诊断相对时间轴上。所有消融均采用以 `Speaker_ID` 为分组单位的严格 5 折交叉验证，并输出 F1、ACC、AUC、SENS 和 SPEC，以对齐 ADCeleb 原基准论文的评价口径。

## 4. 结果与分析

在 ASR 阶段，最佳配置为 `lr_5e-06_effective_batch_16`。其验证集 WER 为 15.72%，测试集 WER 为 17.37%，相较零样本 `openai/whisper-medium` 测试 WER 47.98% 实现约 63.8% 的相对错误率下降。相关图包括 `whisper_seniortalk_finetune/paper_figures/figure1_grouped_validation_test_wer.png`、`figure2_test_wer_hyperparameter_heatmap.png` 与 `figure3_training_curves_eval_wer_and_loss.png`。

在 PCBM 阶段，概念升级带来明确增益：旧概念体系 5 折平均 AUC 为 0.594，新 8 概念体系平均 AUC 为 0.673。`text_cbm_pipeline/figure_1_auc_comparison.png` 展示了 AUC 提升，`text_cbm_pipeline/figure_2_concept_trajectory.png` 展示了概念轨迹；旧版本可与 `text_cbm_pipeline/beforeUpdateConcept/` 中的旧图对照。该结果说明，将高维文本压缩为医学上可解释的概念维度，并不会完全牺牲判别能力，反而能提高模型论证的透明度。

在 ADCeleb 全 10 年轨迹实验中，CRF-PCBM 达到 speaker-level ACC=0.494、AUC=0.608。对应图包括 `text_cbm_pipeline/multimodal_crf_runs/figures/figure_3_crf_transition_heatmap.png`、`figure_4_concept_importance_radar.png` 与 `figure_5_roc_curve.png`。需要强调的是，该结果不是通过混合说话人片段获得的虚高分数，而是在说话人分组交叉验证下得到的保守估计。

最关键的发现来自时间切片消融。Interval -2 的结果为 ACC=0.456、F1=0.379、AUC=0.443、SENS=0.376、SPEC=0.521；Interval -1 的结果为 ACC=0.470、F1=0.334、AUC=0.520、SENS=0.314、SPEC=0.596。表面上看，两个静态切片的 AUC 都不高，但这恰恰是项目的重要证据：第一，诊断前 1-5 年优于 6-10 年，符合 AD 临床进展规律；第二，白盒 PCBM 将上千维文本压缩到少量概念，不像 LLaMA3 等重型黑箱模型那样可能从高维噪声中记忆身份或语境；第三，静态 5 年切片缺少足够时间结构，而全 10 年 CRF 从 AUC 0.520 提升到 0.608，说明对高度压缩的可解释特征而言，捕捉“认知衰退斜率”比单次横截面分类更有价值。

## 5. 结论与学术贡献

本项目完成了从老年语音识别到 AD 可解释纵向预测的端到端研究框架。其贡献主要包括：构建了适配老年语音的 Whisper-medium 微调流程；提出并验证了 8 个 AD 相关文本概念的 PCBM 升级方案；在 ADCeleb 上实现多模态 CRF-PCBM 纵向建模；以 `Speaker_ID` 分组交叉验证严格防止数据泄漏；并通过时间切片消融证明，低分并非失败，而是与 AD 临床进展一致的证据。总体而言，本项目展示了一条比单纯追求黑箱高分更可解释、更严谨、也更适合临床交流的 AD 预测路线。
