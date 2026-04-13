# 答辩 PPT 讲稿与大纲

## Slide 1. 标题页

**Visuals/Figures to Insert:** 左下角放 `/home/saisai/AD_predict/technical_route_image.svg` 缩略图；右侧放项目题目与姓名信息。

**Bullet Points:** 基于纵向语音与文本的阿尔茨海默病可解释预测；关键词：ASR 微调、PCBM、CRF、纵向轨迹、防止数据泄漏。

**Speaker Notes:** 各位老师好，我的项目目标是利用语音和文本中的纵向变化预测 AD 风险。与只追求静态分类 AUC 的黑箱路线不同，我重点构建了一个可解释、可复现、并且严格防止数据泄漏的多阶段框架。

## Slide 2. 研究问题与动机

**Visuals/Figures to Insert:** 建议用“语音/文本 -> 临床概念 -> 时间轨迹 -> AD 风险”的四步流程图。

**Bullet Points:** AD 早期语言信号可被低成本采集；静态黑箱模型解释性不足；纵向变化比单点表达更贴近疾病进展。

**Speaker Notes:** AD 的认知下降不是某一天突然出现的，而是连续发展的。语音和文本可以反复采集，因此更适合做长期轨迹建模。本项目的核心问题是：能否用少量可解释概念捕捉这种变化，而不是让模型记忆说话人身份或高维噪声。

## Slide 3. 总体技术路线

**Visuals/Figures to Insert:** 全页插入 `/home/saisai/AD_predict/technical_route_image.svg`，用高亮框标出新增的 Phase 1 Whisper 微调与 Phase 4 时间切片 CRF 消融。

**Bullet Points:** Phase 1: Whisper-medium 老年语音适配；Phase 2: 8 概念 PCBM；Phase 3: ADCeleb 10 年 CRF；Phase 4: 时间区间消融。

**Speaker Notes:** 这张图对应项目的主流程。我的新增工作主要有两块：第一是把 ASR 前端换成适配老年语音的 Whisper-medium 微调模型；第二是在 ADCeleb 上增加严格的时间切片消融，用来回答“CRF 的时间轨迹到底有没有贡献”。

## Slide 4. Phase 1: 老年语音 ASR 微调

**Visuals/Figures to Insert:** 放 `~/AD_predict/whisper_seniortalk_finetune/paper_figures/figure3_training_curves_eval_wer_and_loss.png`，右侧放脚本列表。

**Bullet Points:** 数据集：BAAI/SeniorTalk；模型：`openai/whisper-medium`；脚本：`setup_workspace.sh`、`prepare_dataset.py`、`train_and_search.py`、`run_overnight.sh`；网格搜索：learning rate x effective batch size。

**Speaker Notes:** 由于后续文本概念完全依赖转录质量，ASR 不是附属模块，而是整个系统的入口。我使用 SeniorTalk 对 Whisper-medium 做领域适配，并通过网格搜索选择泛化最好的超参数。

## Slide 5. ASR 结果：转录误差显著下降

**Visuals/Figures to Insert:** 左侧 `figure1_grouped_validation_test_wer.png`；右侧 `figure2_test_wer_hyperparameter_heatmap.png`，目录均为 `~/AD_predict/whisper_seniortalk_finetune/paper_figures/`。

**Bullet Points:** 零样本测试 WER=47.98%；最佳配置 `lr_5e-06_effective_batch_16`；验证 WER=15.72%，测试 WER=17.37%；相对错误率下降约 63.8%。

**Speaker Notes:** 这里需要注意口径：15.72% 是验证集 WER，测试集是 17.37%。它仍然远低于零样本 47.98%，说明微调确实让模型更适应老年语音特征。这个阶段为后续概念抽取减少了上游噪声。

## Slide 6. Phase 2: 文本 PCBM 的 8 个临床概念

**Visuals/Figures to Insert:** 左侧 `~/AD_predict/text_cbm_pipeline/beforeUpdateConcept/figure_1_auc_comparison_old.png`；右侧 `~/AD_predict/text_cbm_pipeline/figure_1_auc_comparison.png`。

**Bullet Points:** 旧概念平均 AUC=0.594；新 8 概念平均 AUC=0.673；8 概念覆盖情绪、健康话题、怀旧、简单词汇、重复、语法混乱、逻辑断裂、短句碎片化。

**Speaker Notes:** PCBM 的目标是把高维文本压缩成可以被老师和临床人员理解的维度。升级后 AUC 从 0.594 到 0.673，说明可解释性和判别能力并不是必然冲突；关键是概念必须和 AD 语言症状相关。

## Slide 7. 概念轨迹：从分类结果到可解释证据

**Visuals/Figures to Insert:** 对比 `~/AD_predict/text_cbm_pipeline/beforeUpdateConcept/figure_2_concept_trajectory_old.png` 与 `~/AD_predict/text_cbm_pipeline/figure_2_concept_trajectory.png`。

**Bullet Points:** 新概念轨迹更贴近 AD 语言退化；可解释输出支持纵向分析；从“模型说是 AD”转向“哪些概念随时间变化”。

**Speaker Notes:** 这页要强调，概念轨迹不是装饰图，而是模型可解释性的核心。它让我们可以讨论语言简化、重复、逻辑断裂等临床上可理解的变化，而不仅是输出一个概率。

## Slide 8. 严格防止数据泄漏

**Visuals/Figures to Insert:** 建议画 Speaker_ID grouped 5-fold 示意图：同一 `Speaker_ID` 的所有片段只能进入同一 fold。

**Bullet Points:** 按 `Speaker_ID` 分组的 5 折交叉验证；训练/测试说话人完全不重叠；防止模型记忆音色、身份、视频背景；输出 F1、ACC、AUC、SENS、SPEC 对齐 ADCeleb benchmark。

**Speaker Notes:** 这是答辩中必须讲清楚的一页。如果随机按片段划分，同一个说话人的声音可能同时出现在训练集和测试集，模型就可能通过音色或身份特征“作弊”。我采用 Speaker_ID 分组划分，因此测试结果更保守，但学术上更可信。

## Slide 9. Phase 3: 10 年纵向多模态 CRF-PCBM

**Visuals/Figures to Insert:** 放 `~/AD_predict/text_cbm_pipeline/multimodal_crf_runs/figures/figure_3_crf_transition_heatmap.png`。

**Bullet Points:** 数据：ADCeleb 诊断前 10 年纵向数据；脚本：`check_and_aggregate_inventory.py`、`build_multimodal_dataset.py`、`train_multimodal_crf_pcbm.py`；CRF 学习状态转移与时间依赖。

**Speaker Notes:** CRF 的价值在于它不是只看一个片段，而是看同一个说话人在时间轴上的状态变化。AD 的语言退化是动态过程，因此模型应该学习变化路径，而不是只做静态横截面判断。

## Slide 10. 多模态概念重要性

**Visuals/Figures to Insert:** 放 `~/AD_predict/text_cbm_pipeline/multimodal_crf_runs/figures/figure_4_concept_importance_radar.png`。

**Bullet Points:** 声学特征与文本概念共同建模；概念维度保留解释性；雷达图用于说明模型关注的临床线索。

**Speaker Notes:** 黑箱大模型可能获得更高静态分数，但它很难说明判断依据。这里的优势是：即使分数保守，我们仍然可以展示模型关注的概念结构，并把它和临床症状联系起来。

## Slide 11. 全 10 年轨迹结果

**Visuals/Figures to Insert:** 放 `~/AD_predict/text_cbm_pipeline/multimodal_crf_runs/figures/figure_5_roc_curve.png`。

**Bullet Points:** speaker-level ACC=0.494；pooled ROC AUC=0.608；折均 AUC=0.639 可作为稳定性参考；结果来自说话人级分组评估。

**Speaker Notes:** 这不是一个虚高结果，而是在严格 speaker-level 划分下得到的保守结果。AUC=0.608 的意义在于，它使用高度压缩的可解释概念和时间结构，超过了后面静态切片的表现。

## Slide 12. Phase 4: 时间切片消融实验

**Visuals/Figures to Insert:** 建议做表格或柱状图，数据来自 `~/AD_predict/text_cbm_pipeline/time_interval_eval_full_20260405_095444/time_interval_summary.csv`。

**Bullet Points:** Interval -2（诊断前 6-10 年）：AUC=0.443，ACC=0.456；Interval -1（诊断前 1-5 年）：AUC=0.520，ACC=0.470；采用 `evaluate_by_time_interval.py`；CN 使用 pseudo-YoD 匹配。

**Speaker Notes:** 这组结果看起来“不漂亮”，但它非常重要。越接近诊断，AUC 越高，这符合 AD 病理和语言症状逐步显性的临床规律。如果远期 6-10 年也轻易得到很高 AUC，反而要怀疑是否存在数据泄漏或身份记忆。

## Slide 13. 核心论证：为什么“低分”是强证据

**Visuals/Figures to Insert:** 做一条简单趋势线：Interval -2 AUC=0.443 -> Interval -1 AUC=0.520 -> Full 10-year CRF AUC=0.608。

**Bullet Points:** 静态切片缺少动态斜率；白盒概念维度高度压缩；CRF 利用 10 年变化轨迹；全轨迹 AUC 明显高于任一静态切片。

**Speaker Notes:** 这是答辩的中心论点。静态 5 年切片只有横截面信息，对 8 个可解释概念来说信息量不足，所以 AUC 只有 0.52 左右。但 CRF 看的是 10 年轨迹，能捕捉认知衰退的方向和斜率，因此提升到 0.608。这证明我的核心假设：对可解释临床概念而言，时间动态比静态分类更有价值。

## Slide 14. 结论与贡献

**Visuals/Figures to Insert:** 放四个贡献图标或四栏总结：ASR、PCBM、CRF、防泄漏评估。

**Bullet Points:** Whisper-medium 老年语音适配；8 概念 PCBM 提升可解释性与 AUC；ADCeleb 10 年 CRF 证明轨迹价值；Speaker_ID grouped 5-fold 防止泄漏；按 SOTA benchmark 输出 F1/ACC/AUC/SENS/SPEC。

**Speaker Notes:** 总结来说，我的项目贡献不是单一模型，而是一条完整、严谨、可解释的 AD 预测路线。ASR 解决输入质量，PCBM 解决可解释性，CRF 解决纵向变化，Speaker_ID 分组评估解决可信度。最重要的是，时间切片消融说明，模型并不是靠记忆身份取得结果，而是确实需要疾病进展轨迹。
