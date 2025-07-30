#### 其他国家老年人有必要考虑进来训练吗。老年人AD在语言上有什么共性，语言间区别是什么。
如果考虑到我们训练到的文本特征，好像只用给一个bias，某语言本来文本深度是怎么样的，或者从某语言本来正常人说话的标准（这感觉有相关工作）看老年人与其偏离。
#### 引入其他语言好处是什么？
1. 发现疾病导致语言能力上的共性 2. 增加我们不足的数据库资源，这样就不考虑自己模型生成数据来训练了 3. 泛化更强 4. 发现语言之间联系

#### 为什么要区分汉藏和印欧语言体系？（这感觉有论文和相关工作）
**A. 结构差异**:
-   声调语言 vs. 非声调语言
-   不同的形态系统
-   不同的句法结构

**B. AD 表现差异**:
-   **汉藏语系特有标记**:
    -   声调错误
    -   量词省略
-   **印欧语系特有标记**:
    -   冠词错误
    -   动词一致性错误

#### 引入其他语言坏处是什么？
1. 是否针对特定语言能力就减弱了 2. 增加工作在语言研究上（对我日后在德国也许有帮助） 3. 如果语言间联系不大，则未解决数据库小的问题

#### 不同语言间什么是模态（数据收集形式）区别
##### AD 检测中的跨语言差异
影响概念（concepts）的表现方式：
**A. 语言无关特征**（通用标记）：
-   停顿模式
-   言语节律
-   声学特征，如抖动（jitter）和微光（shimmer）
-   脑电图（EEG）信号

**B. 语言相关特征**：
-   句法复杂度（因语言结构而异）
-   词汇丰富度度量
-   信息密度

#### 文本和语音区别是什么，模态间关系
我觉得文本可以体现语音特征。我感觉语言就是涉及文本、语音

#### 当今趋势都是多模态，多模态好处。思考见“MultiModalThinking.md”。不同老人方言的问题在transformer的rerank已经完成？为什么不考虑其为模态的可能，那不同语言也能rerank？  
多模态分析与重排序是独立但互补的方法（be treated as separate but complementary approaches）：
Speech Input
   ↓
[Dialect/Language Identification & Reranking] 专注于处理一种语言内部的方言变体。不应与跨语言分析相混淆。
   ↓
[Multimodal Feature Extraction]
   - Language-independent features
   - Language-specific features
   ↓
[Concept Extraction]
   - Universal concepts
   - Language-specific concepts
   ↓
[CRF Classification]

#### 多模态分析:
根据 Yamada 等人（2022）的研究，阿尔茨海默症（AD）和路易体痴呆（DLB）在不同模态上表现出不同的模式：
- AD：表现出更严重的语言损伤。
- DLB：表现出更严重的韵律/声学损伤。

#### 论文论述
##### AD与其他疾病对比证明概念层合理
1.  来自 Yamada 等人（2022）的论文 **"Speech and language characteristics differentiate Alzheimer's disease and dementia with Lewy bodies"（言语和语言特征区分阿尔茨海默症和路易体痴呆）**:
    [链接](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9614050/)

    **主要发现**:
    -   与对照组相比，AD（阿尔茨海默症）患者在**语言特征**上表现出更大的差异。
    -   DLB（路易体痴呆）患者在**韵律和声学特征**上表现出更大的差异。
    -   具体引文:
        ```
        “与对照组（CN）相比，AD 组在语言特征上的差异大于 DLB 组，而 DLB 组在韵律和声学特征上的差异更大。”

        “在图片描述任务中，AD 组在动作类别的信息单元减少（P = 0.047）以及语义流畅性任务的正确答案减少（P = 0.018）等语言特征上表现出显著差异。”
        ```

2.  来自 Yamada 等人（2021）的论文 **"Tablet-Based Automatic Assessment for Early Detection of Alzheimer's Disease Using Speech Responses to Daily Life Questions"（基于平板电脑的自动评估，利用对日常生活问题的言语反应进行阿尔茨海默症的早期检测）**:
    [链接](https://www.frontiersin.org/articles/10.3389/fdgth.2021.653904/full)

    **主要发现**:
    -   日常生活问题可以跨语言检测出 AD 的早期迹象。
    -   在对照组和 MCI（轻度认知障碍）患者之间显示出差异的言语特征中，约有 80% 是跨语言重叠的。
    -   具体引文:
        ```
        “我们对日语神经心理学任务言语反应的分析结果，与先前关于英语言语反应的研究结果是一致的。”
        ```

3.  来自 Yamada 等人（2024）的论文 **"Distinct eye movement patterns to complex scenes in Alzheimer's disease and Lewy body disease"（阿尔茨海默症和路易体病在复杂场景下的不同眼动模式）**:
    [链接](https://www.frontiersin.org/articles/10.3389/fnins.2024.1333894/full)

    **主要发现**:
    -   AD 和 LBD（路易体病）在视觉注意力和眼动方面表现出不同的模式。
    -   这些模式在不同语言中是一致的，反映了潜在的认知缺陷。
    -   具体引文:
        ```
        “我们关于视觉探索减少的发现与先前对 AD 患者的研究一致，并使用包含更广泛对象类别的更大刺激集提供了进一步的支持。”
        ```

**这些论文表明，尽管 AD 和 DLB 有一些特定于语言的表现形式，但许多核心特征，如信息内容减少、韵律改变以及视觉注意力模式的变化，似乎在不同语言中都是一致的。** 这表明，与这些疾病相关的潜在认知和神经变化，无论使用何种具体语言，都以相似的方式影响着语言和交流。

**研究表明：**

1.  **跨语言的通用标记**:
    -   停顿模式和言语节律
    -   声学特征，如抖动（jitter）和微光（shimmer）
    -   信息内容减少
    -   句法简化

2.  **特定于语言的差异**:
    -   语法错误的表现形式因语言结构而异
    -   词汇丰富度需要不同的衡量标准
    -   声调错误是声调语言特有的
    -   冠词/量词的使用因语系而异

这项研究支持在跨语言的 AD 检测中使用**通用标记和特定于语言的标记相结合**的方法。


##### 
1.  来自 Liu 等人（2022）的论文 **"Dyslexia and dysgraphia of primary progressive aphasia in Chinese"（原发性进行性失语症中的汉语阅读障碍和书写障碍）**:
    [链接](https://www.frontiersin.org/articles/10.3389/fneur.2022.1025660/full)

    **主要发现**:
    -   中国 AD 患者表现出印欧语系中没有的**独特的声调错误**。
    -   由于汉语的**语素文字（logographic）**特性，其读写错误有所不同。
    -   具体引文:
        ```
        “声调错误是中国患者独有的，在日语或英语患者中未发现。因此，声调任务可以作为说汉语的 PPA（原发性进行性失语症）患者的潜在诊断工具。”
        ```

2.  来自 Li 等人（2022）的论文 **"The 32-Item Multilingual Naming Test: Cultural and Linguistic Biases"（32项多语言命名测试：文化和语言偏见）**:
    [链接](https://www.cambridge.org/core/journals/journal-of-the-international-neuropsychological-society/article/abs/32item-multilingual-naming-test-cultural-and-linguistic-biases-in-monolingual-chinesespeaking-older-adults/6CA2F433A791C04B4A8694860488D584)

    **主要发现**:
    -   文化和语言差异会影响测试表现。
    -   需要使用文化上适宜的评估工具。
    -   具体引文:
        ```
        “我们的研究强调了可能影响测试表现的文化和语言差异。未来的研究需要使用在不同文化和语言群体中具有相似词频且被普遍认可的项目来修订 MINT（多语言命名测试）。”
        ```

3.  来自 He 等人（2025）的论文 **"Exploring Gender Bias in Alzheimer's Disease Detection"（探索阿尔茨海默症检测中的性别偏见）**:
    [链接](https://arxiv.org/abs/2507.12356)

    **主要发现**:
    -   在跨语言的 AD 语音感知中存在性别偏见。
    -   像微光（shimmer）值这样的声学特征对不同性别的感知影响不同。
    -   具体引文:
        ```
        “尽管语言对 AD 感知没有显著影响，但我们的发现强调了性别偏见在 AD 语音感知中的关键作用。”
        ```

**在 AD 的表现上，汉语和印欧语系之间的主要差异包括：**

1.  **声调特征**:
    -   由于汉语的声调特性，中国 AD 患者表现出独特的声调错误。
    -   这些错误在非声调的印欧语系中是找不到的。
    -   可以作为针对汉语使用者的特定诊断标记。

2.  **书写系统影响**:
    -   由于汉语的语素文字系统与字母系统的差异，导致了不同的错误模式。
    -   汉语在汉字书写中表现出更多的视觉/结构性错误。
    -   印欧语系则表现出更多的音韵错误。

3.  **语法差异**:
    -   由于缺乏屈折变化，中国 AD 患者表现出不同的语法错误模式。
    -   与印欧语系中的动词变位错误相比，更侧重于词序和功能词的错误。

4.  **文化因素**:
    -   需要文化上适宜的评估工具。
    -   测试项目需要在不同文化背景下进行验证。
    -   一些概念可能无法在语言之间很好地转换。

**研究表明，虽然 AD 的核心特征在不同语言中是相似的，但具体的表现形式会因语言和文化因素而异。这凸显了开发针对特定语言和文化的评估工具及诊断标准的必要性。**


##### AD与汉语特征
1.  来自 Liu 等人（2022）的论文 **"Dyslexia and dysgraphia of primary progressive aphasia in Chinese: A systematic review"（原发性进行性失语症中的汉语阅读障碍和书写障碍：系统性综述）**
    [链接](https://www.frontiersin.org/articles/10.3389/fneur.2022.1025660/full)

    **关于语言差异的主要发现**:
    -   声调错误是中国患者独有的，在日语或英语患者中未发现。
    -   汉语没有严格的形态变化（如时态、复数），因此语法错误的表现形式不同。
    -   引文：“在汉语中，没有严格的形态变化（如单复数、时态和主谓一致）。因此，在汉语中，非流利/语法变异型原发性进行性失语症（nfvPPA）的语法错误类型主要表现在词序、功能词和复杂性上。”

2.  来自 Tee 等人（2022）的论文 **"Dysgraphia phenotypes in native Chinese speakers with primary progressive aphasia"（母语为汉语的原发性进行性失语症患者的书写障碍表型）**
    [链接](https://n.neurology.org/content/98/22/e2245)

    **主要发现**:
    -   由于视觉空间的复杂性，汉字比英语单词需要更多的正字法工作记忆。
    -   汉语 PPA 患者的书写准确性与同音词密度相关，而非语音规律性。
    -   引文：“由于其视觉空间的复杂性，汉字比英语单词更依赖于正字法工作记忆。”

3.  来自 Gorno-Tempini & Tee（2019）的论文 **"Linguistic tone as a potential clinical diagnostic feature for Chinese-speaking primary progressive aphasia"（语言声调作为说汉语的原发性进行性失语症的潜在临床诊断特征）**
    [链接](https://alz-journals.onlinelibrary.wiley.com/doi/10.1016/j.jalz.2019.08.109)

    **主要发现**:
    -   声调任务可以专门用作说汉语的 PPA 患者的诊断工具。
    -   不同的 PPA 变体在声调产生和理解上表现出不同的模式。
    -   引文：“nfvPPA 患者倾向于犯声调替换错误，而 svPPA（语义变异型PPA）患者在连续声调朗读任务中容易出现规则化错误。”

4.  来自 Law & Or（2001）的论文 **"A case study of acquired dyslexia and dysgraphia in Cantonese"（一例粤语获得性阅读障碍和书写障碍的个案研究）**
    [链接](https://www.tandfonline.com/doi/abs/10.1080/02643290143000024)

    **主要发现**:
    -   针对汉语的读写提出了一个特有的“三角模型”，而不是用于字母语言的双通路模型。
    -   汉语中的语音表征具有独特的多层形式，音段和声调在不同的层级上。
    -   引文：“音段特征（即辅音和元音）和超音段特征（即声调）处于分离的层级，并各自独立地产生连接。”

**这些论文共同表明：**

1.  汉语和印欧语系之间的结构差异导致了 AD 中语言障碍的不同表现形式。
2.  像声调和汉字这样的汉语特有特征，创造了独特的诊断机会和挑战。
3.  由于正字法和语法的根本差异，不同语系之间，导致读写错误的认知机制是不同的。

##### 汉藏语系和印欧语系， AD与神经系统：

1.  来自 Thompson 等人（2012）的论文 **"Dissociations between fluency and agrammatism in primary progressive aphasia"（原发性进行性失语症中流畅性与语法缺失的分离）**
    [链接](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3244868/)

    **关于语系差异的主要发现**:
    -   印欧语系拥有丰富的形态系统（动词屈折变化、时态、一致性）。
    -   引文：“对印欧语系的研究表明，非流利/语法变异型PPA（nfvPPA）患者的语法缺失特征是动词屈折变化和动词论元结构的产生受损。”

2.  来自 Friederici 等人（2003）的论文 **"The role of left inferior frontal and superior temporal cortex in sentence comprehension"（左侧额下回和颞上回在句子理解中的作用）**
    [链接](https://academic.oup.com/cercor/article/13/2/170/312871)

    **主要发现**:
    -   处理句法与语义信息涉及不同的神经网络。
    -   汉藏语系更多地依赖词序和功能词来表达语法。
    -   引文：“在汉语中，没有严格的形态变化（如单复数、时态和主谓一致）。因此，语法缺失的错误类型主要表现在词序、功能词和复杂性上。”

3.  来自 Tan 等人（2005）的论文 **"Neuroanatomical correlates of phonological processing of Chinese characters and alphabetic words"（汉字和字母词语音加工的神经解剖学关联）**
    [链接](https://onlinelibrary.wiley.com/doi/10.1002/hbm.20134)

    **主要发现**:
    -   大脑在处理汉语与字母语言方面存在根本差异。
    -   由于其语素文字（logographic）的特性，汉语在阅读时依赖于不同的神经网络。
    -   引文：“汉字是映射到语言的语素/音节层面，而不是音位层面。”

**将语言划分为汉藏语系和印欧语系是基于历史语言学和类型学特征的：**

1.  **结构差异**:
    -   **印欧语系**：形态丰富，有屈折变化系统。
    -   **汉藏语系**：孤立语形态，有声调系统。

2.  **语法特征**:
    -   **印欧语系**：动词变位、性/数一致性。
    -   **汉藏语系**：通过词序和功能词表达语法。

3.  **书写系统**:
    -   **印欧语系**：字母/音位文字。
    -   **汉藏语系**：语素文字/词素文字（Logographic/morphemic）。

##### 汉藏-印欧语系
1.  **关于将语言分为汉藏语系和印欧语系：**

    来自 van Driem (2007) 的论文 **"The diversity of the Tibeto-Burman language family and the linguistic ancestry of Chinese"（藏缅语族的多样性与汉语的语言学起源）**：

    -   将语言划分为汉藏语系和印欧语系，在历史上是基于**结构和类型学**的差异。
    -   关键的结构差异包括：
        -   **形态系统**（印欧语系有丰富的屈折形态）
        -   **词序模式**
        -   **声调系统**（存在于汉藏语系中）

2.  **关于特定于语言的特征和差异：**

    来自 Sandman & Simon (2016) 的论文 **"Tibetan as a model language in the Amdo Sprachbund"（作为安多语言联盟典范语言的藏语）**：

    -   藏语显示出独特的特征，例如：
        -   复杂的**声调系统**
        -   **动词在句末**的词序
        -   **示证（evidential marking）系统**
        -   **量词系统**

3.  **关于汉语和藏语之间的关系：**

    来自 LaPolla (2012) 的论文 **"Comments on methodology and evidence in Sino-Tibetan comparative linguistics"（关于汉藏比较语言学中方法论和证据的评论）**：

    -   尽管同属一个语系，汉语和藏语表现出显著差异：
        -   汉语发展出了**孤立语形态**，而藏语保留了更复杂的形态。
        -   不同的**语法化**模式。
        -   不同的**示证系统**。

4.  **关于语言接触和趋同：**

    来自 Zeisler (2009) 的论文 **"Reducing phonetical complexity and grammatical opaqueness"（降低语音复杂性和语法不透明性）**：

    -   处于接触区域的语言会表现出趋同效应：
        -   借用语法特征
        -   简化复杂系统
        -   发展出共享的区域特征

**研究表明，虽然语系之间存在明确的类型学差异，但语言接触和历史发展导致了既有分化又有趋同的复杂模式。** 






## 对老年人语言的某个层面，如语音、韵律、词法、句法、语篇、语用等层面的特征描写
其建库系统，但语料库体较量小，典型性上略显不足。
有：该类语料库以老年人语言参照语料库（a  reference  corpus  for  the  elderly’s  language,  Corpage）、老年人语言多模态语料库（a  multimodal  corpus  for  the  elderly’s  language,  CorpAGEst）、老化互动研究语料库（Videos  to  Study  Interaction  in  AGEing，VIntAGE）（Catherine Bolly团队）和卡罗来纳会话集（Carolinas  Conversations  Collection，CCC）（Boyd Davis团队）为代表

## 多次追踪构建的老化数据库
其记录研究中被试参与研究的过程，并非针对语言老化，因此与语言维度的关联程度受很大限制。
有：痴呆银行（DementiaBank），成年发展的跨学科纵向研究（Interdisciplinary  Longitudinal  Study  of  Adult  Development，ILSE①），波恩老化纵向研究（The Bonn Longitudinal Study on Ageing，BOLSA）。

## 构建本族语言在全生命周期范围内语言全貌
俄语国家语料库（НАЦИОНАЛЬНЫЙ КОРПУС РУССКОГО ЯЗЫКА，RNC）、日语自发话语语料库（日本語話し言葉コーパス，Corpus of Spontaneous Japanese，CSJ）和现代汉语现场即席话语语料库（Spoken  Chinese  Corpora  of  Situated  Discourse,  SCCSD）

## 文本语料库以及包括音频信息的多模态语料库的共享
英国国家语料库（British National Corpus，BNC）、布朗语料库（BROWN Corpus）、汉语口语语篇语料库（Discourse-Chinese Annotated Spontaneous  Speech，Discourse-CASS）等



————————————————
对于文本概率有：语速=从文本的词数和音频总时长计算、停顿比例=从文本中<pause>标记的数量和总次数计算、词汇丰富度=从文 本计算（如Type-Token）、句法复杂度=从文本计算（如依存句法树的深度）

Speech Input
   ↓
[Dialect/Language Identification & Reranking]
   ↓
[Multimodal Feature Extraction]
   - Language-independent features
   - Language-specific features
   ↓
[Concept Extraction]
   - Universal concepts
   - Language-specific concepts
   ↓
[CRF Classification]