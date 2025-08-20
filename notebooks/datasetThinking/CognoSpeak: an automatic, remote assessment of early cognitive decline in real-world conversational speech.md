

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