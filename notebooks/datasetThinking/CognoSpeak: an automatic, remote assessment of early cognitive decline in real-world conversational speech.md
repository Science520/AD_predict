

用于特征提取、分类和评估的 Python 脚本: https://zenodo.org/records/14515541

通过自发语音分别使用声学和语言学特征，在检测阿尔茨海默病（AD）方面取得了74.65%和84.51%的高准确率
结论来自：Y. Pan, B. Mirheidari, J. M. Harris, J. C. Thompson, M. Jones, J. S.
 Snowden, D. Blackburn, and H. Christensen, “Using the Outputs of
 Different Automatic Speech Recognition Paradigms for Acoustic-and
 BERT-based Alzheimer’s Dementia Detection Through Spontaneous
 Speech.” in Interspeech, 2021, pp. 3810–3814.

### 主要支持健康对照组（HC）和阿尔茨海默病（AD）之间的二元区分
1. 一组来自独立音频的紧凑特征集帮助实现了87.6%的准确率[16]。
[16] M. R. Kumar, S. Vekkot, S. Lalitha, D. Gupta, V. J. Govindraj,
 K. Shaukat, Y. A. Alotaibi, and M. Zakariah, “Dementia detection from
 speech using machine learning and deep learning architectures,” Sensors,
 vol. 22, no. 23, p. 9311, 2022.


 2. 像[17]、[18]这样的自动系统通过同时使用音频和文本（手动转录）特征，取得了高达约90%的准确率，表现非常出色。
  [17] Z. S. Syed, M. S. S. Syed, M. Lech, and E. Pirogova, “Tackling the
 ADRESSO challenge 2021: The MUET-RMIT system for Alzheimer’s
 Dementia Recognition from Spontaneous Speech.” in Interspeech, 2021,
 pp. 3815–3819.
 [18] E. Edwards, C. Dognin, B. Bollepalli, M. K. Singh, and V. Analytics,
 “Multiscale System for Alzheimer’s Dementia Recognition Through
 Spontaneous Speech.” in INTERSPEECH, 2020, pp. 2197–2201.

 ### 轻度认知障碍（MCI）的研究
Mirheidari等人的工作[19]，他们使用基于BERT的特征结合声学和文本特征来训练分类器
[19] B. Mirheidari, R. O’Malley, D. Blackburn, and H. Christensen, “Iden
tifying people with mild cognitive impairment at risk of developing  dementia using speech analysis,” in 2023 IEEE Automatic Speech
 Recognition and Understanding Workshop (ASRU). IEEE, 2023, pp.
 1–6.
 
Amini等人[20]在一个包含三组受试者数据集上使用AlBERT和BERT
[20] S. Amini, B. Hao, L. Zhang, M. Song, A. Gupta, C. Karjadi, V. B.
 Kolachalama, R. Au, and I. C. Paschalidis, “Automated detection of mild
 cognitive impairment and dementia from voice recordings: A natural
 language processing approach,” Alzheimer’s & Dementia, 2022.

But 数据集规模较小，缺乏多个诊断类别，以及更广泛的人口统计信息（包括种族信息）和丰富的临床元数据。



MoCA分数较高的成年人在信息丰富度、语言连贯性和找词能力方面表现更好

### CognoSpeak数据集
像CognoSpeak这样的自动工具可能有助于监测纵向跟踪。

处理：
音频/视频->强制对齐->分割片段（30秒）
->
特征提取与分类 (两条并行路径):
- 声学 -> 声学特征: 使用 openSMILE 提取 (eGeMAPS, ComParE 特征集) -> 标准分类器: 逻辑回归 (LR), 支持向量机 (SVM)
- 语言学 -> 语言学特征: 通过自动语音识别（ASR）获得 -> 序列分类器: DistilBERT, BART, RoBERTa

-> 预测


根据双尾t检验（p < 0.001），性别分布没有统计学上的显著差异

### 模型被证明在这有效
近年来，基础模型在情感分析、语言和文本分析方面被证明是有效的 [31]，因此我们在这里探讨了它们在检测认知障碍方面的能力 [32]。
 [31] Y. Zhang, J. Gao, M. Zhou, X. Wang, Y. Qiao, S. Zhang, and D. Wang,
 “Text-guided foundation model adaptation for pathological image clas
sification,” in International Conference on Medical Image Computing
 and Computer-Assisted Intervention. Springer, 2023, pp. 272–282.
 [32] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von
 Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill et al.,
 “On the opportunities and risks of foundation models,” arXiv preprint
 arXiv:2108.07258, 2021.
使用语言学特征的基础模型优于使用声学特征的标准分类器。


 实验采用k折交叉验证（k = 5）进行，并使用相同的数据划分，确保各折之间严格没有重叠，以便对不同方法进行严谨的比较，并充分利用每个类别中相对较少的数据。
 - 通过循环使用数据，让每一份数据都有机会被用作测试集。每一个样本最终都被用于训练了 k-1 次，被用于测试了 1 次。降低偶然性，提供更稳定的评估指标

### 实验结果
#### 声学
 eGeMAPS特征的逻辑回归（LR）分类器
#### 文本
 RoBERTa在使用长期记忆任务时超越了所有其他模型
#### 综合
结果表明，总体而言，LR（逻辑回归）的表现优于SVM（支持向量机），而在使用声学特征进行分类时，eGeMAPS是首选特征。
在使用总体声学特征时，ComParE是首选特征，因为LR和SVM的表现都优于eGeMAPS。然而，语言学特征在单项和总体性能上都大幅超越了声学特征。
这些观察结果也支持了我们之前的发现，即复杂的、预训练和微调的架构比标准分类器表现更好[38]。
 M. Pahar, M. Klopper, R. Warren, and T. Niesler, “COVID-19 detection
 in cough, breath and speech using deep transfer learning and bottleneck
 features,” Computers in Biology and Medicine, vol. 141, p. 105153,
 2022. [Online]. Available: https://doi.org/10.1016/j.compbiomed.2021.
 105153