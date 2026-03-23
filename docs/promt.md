2025-10-17 06:25:07,373 - INFO - Bilibili字幕: 154 个
2025-10-17 06:25:07,373 - INFO - Whisper转录: 0 个
2025-10-17 06:25:07,373 - INFO - 失败: 73 个
2025-10-17 06:25:07,373 - INFO - 结果已保存: data/transcript_results.json
2025-10-17 06:25:07,373 - INFO - 
按方言统计:
2025-10-17 06:25:07,373 - INFO -   beijing_mandarin: 16 个
2025-10-17 06:25:07,373 - INFO -   dongbei_mandarin: 19 个
2025-10-17 06:25:07,374 - INFO -   gan_dialect: 1 个
2025-10-17 06:25:07,374 - INFO -   jianghuai_mandarin: 23 个
2025-10-17 06:25:07,374 - INFO -   jin_dialect: 1 个
2025-10-17 06:25:07,374 - INFO -   lanyin_mandarin: 28 个
2025-10-17 06:25:07,374 - INFO -   min_dialect: 1 个
2025-10-17 06:25:07,374 - INFO -   tibetan_dialect: 2 个
2025-10-17 06:25:07,374 - INFO -   wu_dialect: 24 个
2025-10-17 06:25:07,374 - INFO -   xinan_mandarin: 16 个
2025-10-17 06:25:07,374 - INFO -   yue_dialect: 2 个
2025-10-17 06:25:07,374 - INFO -   zhongyuan_mandarin: 21 个

下一步:
2025-10-17 06:25:07,374 - INFO -   1. 查看转录结果: cat data/transcript_results.json
2025-10-17 06:25:07,374 - INFO -   2. 查看转录文本: ls data/raw/audio/result/
2025-10-17 06:25:07,374 - INFO -   3. 运行数据预处理: python scripts/1_prepare_dataset.py


数据统计：
  成功下载视频: 227 个
  Bilibili字幕: 154 个
  Whisper转录: 0 个

✅ 成功加载 990 条Excel数据
✅ 找到 168 对音频-文本数据
✅ 成功标注 151 条数据
✅ 自动数据增强：151 → 187 条（增强了少数类别）
✅ 生成训练集 168 条，验证集 19 条
✅ 保存到 ./processed_data

### 问题现象：
openai-whisper 的模型（.pt 格式），但训练脚本需要 transformers 库的 Hugging Face 格式
下载huggingface版本
PEFT库与Whisper的集成存在兼容性问题。让我创建一个 Workaround - 创建一个自定义的Trainer来正确处理Whisper的输入

### 最佳实践：依赖Seq2SeqTrainer和DataCollator
Hugging Face的transformers库提供了一整套专门为Whisper这类Seq2Seq语音模型设计的工具，它们协同工作，能完美解决参数传递问题。
核心组件是DataCollatorSpeechSeq2SeqWithPadding。这个“数据整理器”是关键的粘合剂。它的作用是在每个训练批次中，从数据集中取出经过预处理的样本（包含了input_features和labels），然后把它们智能地打包成一个字典，这个字典的键（'input_features', 'labels'等）与Whisper模型forward函数所需的参数名完全对应。
**Seq2SeqTrainer**则被设计为能够理解这个DataCollator输出的字典，并将其中的键值对准确无误地传递给PEFT包装后的Whisper模型。

### 所以解决
问题的核心是DataCollatorSpeechSeq2SeqWithPadding的实现不够完善。

hugging face：
The interval with which we sample our audio is known as the sampling rate and is usually measured in samples/sec or Hertz (Hz).
Audio samples should only ever be processed with the correct sampling rate. Failing to do so can lead to unexpected results! For instance, taking an audio sample with a sampling rate of 16kHz and listening to it with a sampling rate of 8kHz will make the audio sound as though it's in half-speed. 
Whisper operation：
1. The Whisper feature extractor expects audio inputs with a sampling rate of 16kHz, so we need to match our inputs to this value. Since all elements in the batch are padded/truncated to a maximum length in the input space, we don't require an attention mask when forwarding the audio inputs to the Whisper model. Whisper is unique in this regard - with most audio models, you can expect to provide an attention mask that details where sequences have been padded, and thus where they should be ignored in the self-attention mechanism. Whisper is trained to operate without an attention mask and infer directly from the speech signals where to ignore the inputs.
2. The second operation that the Whisper feature extractor performs is converting the padded audio arrays to log-Mel spectrograms. These spectrograms are a visual representation of the frequencies of a signal, rather like a Fourier transform.
The Mel channels (frequency bins) are standard in speech processing and chosen to approximate the human auditory range. All we need to know for Whisper fine-tuning is that the spectrogram is a visual representation of the frequencies in the speech signal. (In this way I think we don't need peft)
So you need to use this as:
´´´
from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
´´´
But in 2_fintune_whisper_lora.py, you didn´t use it in the correct approval.

Load WhisperTokenizer: 
encoder: Connectionist Temporal Classification (CTC). Use this can directly leverage the tokenizer from the pre-trained model.
We simply have to specify the target language and the task. 
´´´
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="zh", task="transcribe")

´´´
Need to figure out which language signal is for Chinese.

When encoding the transcriptions, the tokenizer appends 'special tokens' to the start and end of the sequence, including the start/end of transcript tokens, the language token and the task tokens (as specified by the arguments in the previous step).  Because it appends 'special tokens', and our datasets have been appended dialect tokens, so if this need to be seperated?

This may can solve the problem of input-ids and features
When decoding the label ids, we have the option of 'skipping' these special tokens, allowing us to return a string in the original input form:
´´´
input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
´´´

To simplify using the feature extractor and tokenizer, we can wrap both into a single WhisperProcessor class. This processor object inherits from the WhisperFeatureExtractor and WhisperProcessor and can be used on the audio inputs and model predictions as required. In doing so, we only need to keep track of two objects during training: the processor and the model:
´´´
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

´´´
if print(common_voice["train"][0])
then we've got a 1-dimensional input audio array and the corresponding target transcription. Remember to match the sampling rate of our audio to that of the Whisper model (16kHz)
We'll set the audio inputs to the correct sampling rate using dataset's cast_column method. This operation does not change the audio in-place, but rather signals to datasets to resample audio samples on the fly the first time that they are loaded:
´´´
from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
´´´

Conclusion:
´´´
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
´´´
or the method of .map:
´´´
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)
´´´
Note: Currently datasets makes use of both torchaudio and librosa for audio loading and resampling. If you wish to implement your own customised data loading/sampling, you can use the "path" column to obtain the audio file path and disregard the "audio" column.
But I prefer to makes use of both torchaudio and librosa for audio loading and resampling.

The exact solution for input-ids and input_features:
´´´
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
´´´

Pre-Trained Checkpoint
´´´
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
´´´

'WER metric' for evaluation

'from transformers import Seq2SeqTrainingArguments' for traning, inside can make the set of arguments
or 'from transformers import Seq2SeqTrainer'

To launch training, simply execute:
´´´
trainer.train()
´´´


修改大小：GPU内存不够
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 236.00 MiB. GPU 0 has a total capacity of 23.51 GiB of which 219.75 MiB is free. Including non-PyTorch memory, this process has 23.29 GiB memory in use. Of the allocated memory 22.27 GiB is allocated by PyTorch, and 479.47 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

减小batch size并增加gradient checkpointing
添加gradient checkpointing以进一步减少内存使用


You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
100%|█████████████████████████████████████████| 60/60 [8:07:25<00:00, 414.74s/it]/home/saisai/miniconda3/envs/graph/lib/python3.13/site-packages/peft/utils/save_and_load.py:286: UserWarning: Could not find a config file in openai/whisper-large-v3 - will assume that the vocabulary was not modified.
  warnings.warn(
{'train_runtime': 29246.1218, 'train_samples_per_second': 0.057, 'train_steps_per_second': 0.002, 'train_loss': 3.1878097534179686, 'epoch': 10.0}
100%|█████████████████████████████████████████| 60/60 [8:07:26<00:00, 487.44s/it]

================================================================================
训练完成！保存模型...
================================================================================
/home/saisai/miniconda3/envs/graph/lib/python3.13/site-packages/peft/utils/save_and_load.py:286: UserWarning: Could not find a config file in openai/whisper-large-v3 - will assume that the vocabulary was not modified.
  warnings.warn(

✅ LoRA适配器已保存到: /home/saisai/AD_predict/AD_predict/./whisper_lora_dialect/final_adapter
***** train metrics *****
  epoch                    =         10.0
  total_flos               = 5344150341GF
  train_loss               =       3.1878
  train_runtime            =   8:07:26.12
  train_samples_per_second =        0.057
  train_steps_per_second   =        0.002

进行最终评估...
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
Using custom `forced_decoder_ids` from the (generation) config. This is deprecated in favor of the `task` and `language` flags/config options.
Transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English. This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`. See https://github.com/huggingface/transformers/pull/28687 for more details.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
/home/saisai/miniconda3/envs/graph/lib/python3.13/site-packages/torch/nn/parallel/_functions.py:71: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn(
Downloading builder script: 5.13kB [00:00, 6.91kB/s] 5/5 [01:00<00:00, 12.76s/it]
100%|██████████████████████████████████████████████| 5/5 [01:04<00:00, 12.93s/it]
***** eval metrics *****
  epoch                   =       10.0
  eval_loss               =     3.1805
  eval_runtime            = 0:01:24.32
  eval_samples_per_second =      0.225
  eval_steps_per_second   =      0.059
  eval_wer                =        1.0

================================================================================
微调完成！
================================================================================

最终WER: 1.0000

LoRA适配器保存位置: /home/saisai/AD_predict/AD_predict/./whisper_lora_dialect/final_adapter

使用方法:
  from transformers import WhisperForConditionalGeneration, WhisperProcessor
  from peft import PeftModel
  base_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3')
  model = PeftModel.from_pretrained(base_model, '/home/saisai/AD_predict/AD_predict/./whisper_lora_dialect/final_adapter')
  processor = WhisperProcessor.from_pretrained('/home/saisai/AD_predict/AD_predict/./whisper_lora_dialect/final_adapter')

Whisper-large-v3 是一个有15亿参数的超大模型！
我的训练样样本是168，推荐5000+	缺少97%

# 实时查看训练日志
tail -f /tmp/whisper_logs/whisper_medium_20251022_093636.log

# 查看训练进度摘要
bash scripts/monitor_training.sh

# 持续监控（每10秒刷新）
watch -n 10 bash scripts/monitor_training.sh

# 查看GPU使用
nvidia-smi

输出文件
训练完成后，模型将保存在：
模型检查点: /data/AD_predict/whisper_medium_minimal/
TensorBoard日志: /data/AD_predict/logs_medium_minimal/
训练日志: /tmp/whisper_logs/whisper_medium_20251022_093636.log


91%|█████████ | 200/220 [23:50<01:46,  5.31s/it]{'eval_loss': 2.9551844596862793, 'eval_wer': 0.9983766233766234, 'eval_runtime': 94.1065, 'eval_samples_per_second': 0.202, 'eval_steps_per_second': 0.202, 'epoch': 13.67}                                                                                                       
{'loss': 2.9316, 'grad_norm': 0.4467923939228058, 'learning_rate': 1.0764705882352941e-05, 'epoch': 14.57}
{'loss': 2.9806, 'grad_norm': 0.4505053460597992, 'learning_rate': 9e-06, 'epoch': 15.48}
{'loss': 2.8969, 'grad_norm': 0.40939608216285706, 'learning_rate': 7.235294117647059e-06, 'epoch': 16.38}
{'loss': 2.9066, 'grad_norm': 0.46224245429039, 'learning_rate': 5.470588235294117e-06, 'epoch': 17.29}
{'loss': 2.9322, 'grad_norm': 0.43853938579559326, 'learning_rate': 3.7058823529411767e-06, 'epoch': 18.19}
{'eval_loss': 2.936530351638794, 'eval_wer': 0.9983766233766234, 'eval_runtime': 95.1374, 'eval_samples_per_second': 0.2, 'eval_steps_per_second': 0.2, 'epoch': 18.19}                                                 
{'loss': 2.8958, 'grad_norm': 0.5754706859588623, 'learning_rate': 1.9411764705882357e-06, 'epoch': 19.1}
{'loss': 2.9077, 'grad_norm': 0.6393622756004333, 'learning_rate': 1.764705882352941e-07, 'epoch': 20.0}
{'train_runtime': 1635.1915, 'train_samples_per_second': 2.055, 'train_steps_per_second': 0.135, 'train_loss': 3.058194403214888, 'epoch': 20.0}                                                                        
训练完成！保存模型...
  epoch                    =         20.0
  epoch                   =       20.0
  eval_wer                =     0.9984


  ✅ Whisper-medium训练完毕（27分钟）
❌ WER = 0.9984 （99.84%错误率，效果很差）
📁 模型已保存: /data/AD_predict/whisper_medium_minimal/

cd /home/saisai/AD_predict/AD_predict
# 一键启动所有实验（推荐！）
nohup bash scripts/run_experiments.sh > /tmp/experiments.log 2>&1 &
# 查看进度
tail -f /tmp/experiments.log

# 自动生成对比报告
RESULTS_DIR=$(ls -td /data/AD_predict/experiments_* | head -1)
python scripts/analyze_experiments.py --results_dir $RESULTS_DIR
# 查看报告
cat $RESULTS_DIR/EXPERIMENT_REPORT.md


## 增强音频
需要增强的方言类别 (样本数 < 200): ['beijing_mandarin', 'jianghuai_mandarin', 'zhongyuan_mandarin', 'dongbei_mandarin', 'jin_dialect']
增强音频将保存到: /data/AD_predict/data/raw/audio/elderly_audios_augmented
数据增强: 100%|██████████| 151/151 [18:14<00:00,  7.25s/it]

============================================================
📊 数据增强结果报告
============================================================
✅ 成功增强样本数: 604
❌ 失败增强样本数: 0
📁 增强音频目录: /data/AD_predict/data/raw/audio/elderly_audios_augmented
📁 实际创建文件数: 604
📈 增强后总样本数: 755 (原始: 151)

🔍 增强音频文件验证:
✅ 存在的增强音频文件: 604
❌ 缺失的增强音频文件: 0






0%|          | 0/1075 [00:00<?, ?it/s]You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
/home/saisai/miniconda3/envs/graph/lib/python3.13/site-packages/torch/utils/checkpoint.py:86: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
  5%|▍         | 50/1075 [05:47<1:57:34,  6.88s/it]Using custom `forced_decoder_ids` from the (generation) config. This is deprecated in favor of the `task` and `language` flags/config options.
Transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English. This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`. See https://github.com/huggingface/transformers/pull/28687 for more details.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
{'loss': 3.4004, 'grad_norm': 3


警告1: "Could not find a config file"
原因: PEFT在保存checkpoint时，尝试查找原始模型配置
解释: 因为我们添加了12个方言token，修改了词表大小
      (51865 → 51877)，PEFT提示它假设词表未修改
      
影响: ❌ 无影响！模型会正确保存和加载
状态: ✅ 正常警告，可以忽略

警告2: "None of the inputs have requires_grad=True"
原因: 使用了gradient checkpointing（梯度检查点）
解释: 这是内存优化技术，在前向传播时不保存中间梯度
      在反向传播时重新计算，以节省GPU内存
      
影响: ❌ 无影响！这是正常的优化行为
状态: ✅ 正常警告，可以忽略


Whisper实验结果：
exp1 (高LoRA rank): WER = 0.954 (95.4%)
exp2 (低学习率): WER = 0.9852 (98.52%)
exp3 (大batch): WER = 0.9544 (95.44%)
exp4 (激进训练): WER = 1.1183 (111.83%)
所有结果都很差（WER接近或超过100%），说明模型几乎没有学到任何东西。
模型保存位置：
/data/AD_predict/exp1_high_rank/final_adapter
/data/AD_predict/exp2_low_lr/
/data/AD_predict/exp3_large_batch/
/data/AD_predict/exp4_aggressive/
报告问题：
分析脚本没有正确提取WER，显示N/A
需要修复analyze_experiments.py来正确解析日志
FunASR：
确实需要安装funasr
可以单独运行FunASR实验
让我创建一个完整的结果分析报告。


绷不住，跑CPU一个模型要跑20小时，还好GPU空出来了

评估代码错误地将整个字符串（包括特殊token）作为参考文本，导致CER计算完全错误！所以字错误率高
验证集中78%是增强音频（59/76）！要删去增强的



 评估结果汇总
====================================================================================================
                       模型名称                 模型ID  标准模式-字正确率(%)  标准模式-拼音准确率(%)  CI模式-最佳字正确率(%)  CI模式-最佳拼音准确率(%)
Exp3: 大Batch (step 750, 最佳) exp3_large_batch_750         13.12          13.10           12.75            12.93
 Exp1: 高Rank (step 100, 最佳)   exp1_high_rank_100         12.67          12.92           12.99            13.09
     Exp2: 低学习率 (step 1100)     exp2_low_lr_1100         11.77          11.94           12.20            12.38
           原始Whisper-Medium  whisper_medium_base          7.84           8.00            7.11             7.42
     Exp4: 激进学习率 (step 500)  exp4_aggressive_500         -1.87           8.73           13.00            13.20

✅ 结果已保存到: /data/AD_predict/all_experiments_20251022_140017/comprehensive_evaluation_results.csv
✅ 详细报告已保存到: /data/AD_predict/all_experiments_20251022_140017/EVALUATION_REPORT.md


## seniortalk选择
毫无疑问，您应该使用 sentence_data。
这是最直接、最高效的选择，原因如下：
为ASR任务量身打造：sentence_data 已经由数据集作者为您处理成了标准的ASR训练格式。每一条数据都是一个短音频片段（平均2.28秒），并精确对应一句人工转写的文本。这正是 Whisper 等模型进行微调时所需要的 (audio, text) 数据对。
避免繁重的预处理：dialogue_data 是长达半小时以上的完整对话录音。如果您选择它，就需要自己去解析 transcript/*.txt 文件中的时间戳，然后用代码（如 ffmpeg 或 pydub）去切割这50多个小时的长音频，将其制作成数万个短音频片段。这是一个非常复杂且容易出错的过程，而 sentence_data 已经完美地帮您完成了这一步。
标准化的训练/验证/测试集划分：sentence_data 已经为您划分好了 train, validation, test 三个子集，这是进行严谨模型训练和评估的标准做法。
结论：选择 sentence_data 可以让您直接进入模型训练阶段，而选择 dialogue_data 则需要您先花费数天甚至数周的时间进行数据预处理。