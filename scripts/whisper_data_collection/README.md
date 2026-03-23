# Whisper方言数据智能收集系统

这是一个智能的数据采集和标注系统，用于Whisper多方言微调项目。

## 🎯 核心特性

1. **智能采样分析** - 自动分析数据分布，计算每个方言需要的样本数
2. **选择性下载** - 只下载需要的视频，避免浪费带宽和存储空间
3. **自动字幕爬取** - 优先使用Bilibili字幕，支持Whisper备用转录
4. **方言标签关联** - 自动将转录文本与方言标签关联

## 📂 文件结构

```
whisper_data_collection/
├── 0_run_data_pipeline.sh          # 完整流程脚本（推荐）
├── 1_analyze_and_sample.py         # 智能采样分析
├── 2_selective_download.py         # 选择性下载
├── 3_scrape_subtitles.py           # 字幕爬取和标注
└── README.md                       # 本文件
```

## 🚀 快速开始

### 方式1: 一键运行（推荐）

```bash
cd /home/saisai/AD_predict/AD_predict

# 赋予执行权限
chmod +x scripts/whisper_data_collection/0_run_data_pipeline.sh

# 运行完整流程
bash scripts/whisper_data_collection/0_run_data_pipeline.sh
```

流程会依次：
1. 分析数据分布并生成采样计划
2. 选择性下载视频
3. 爬取字幕或使用Whisper转录
4. 生成标注好的数据集

### 方式2: 分步执行（更可控）

#### 步骤1: 智能采样分析

```bash
python scripts/whisper_data_collection/1_analyze_and_sample.py
```

**功能**：
- 分析Excel中的方言分布
- 检查已下载的音频文件
- 计算每个方言需要的目标样本数
- 生成下载计划JSON

**输出**：
- `data/sampling_plan.json` - 完整的采样计划
- `data/sampling_plan_indices.txt` - 需要下载的视频索引列表

**示例输出**：
```
当前方言分布（Excel全量）:
  beijing_mandarin         :  314 个视频 (已下载: 3)
  wu_dialect              :  170 个视频 (已下载: 2)
  dongbei_mandarin        :   96 个视频 (已下载: 1)
  ...

目标样本分配:
  方言                      当前    目标    需要    可用
  --------------------------------------------------------------
  beijing_mandarin            3     30     27    311
  wu_dialect                  2     30     28    168
  yue_dialect                 0     30     30      6  ⚠️ 不足
  ...
```

#### 步骤2: 选择性下载

```bash
# 完整下载
python scripts/whisper_data_collection/2_selective_download.py

# 测试模式（只下载5个）
python scripts/whisper_data_collection/2_selective_download.py --max_downloads 5

# 断点续传（从第10个开始）
python scripts/whisper_data_collection/2_selective_download.py --start_from 10

# 使用cookies（部分视频可能需要登录）
python scripts/whisper_data_collection/2_selective_download.py --cookies bilibili_cookies.txt
```

**功能**：
- 根据采样计划选择性下载视频
- 自动提取音频并转换为16kHz WAV格式
- 支持断点续传
- 智能跳过已下载的文件

**输出**：
- `data/raw/audio/elderly_videos/` - 下载的视频文件
- `data/raw/audio/elderly_audios/` - 提取的音频文件
- `data/download_results.json` - 下载结果统计

#### 步骤3: 爬取字幕

```bash
# 使用Bilibili字幕 + Whisper备选
python scripts/whisper_data_collection/3_scrape_subtitles.py

# 只使用Bilibili字幕
python scripts/whisper_data_collection/3_scrape_subtitles.py --no_whisper_fallback

# 使用更大的Whisper模型
python scripts/whisper_data_collection/3_scrape_subtitles.py --whisper_model large-v3

# 使用cookies
python scripts/whisper_data_collection/3_scrape_subtitles.py --cookies bilibili_cookies.txt
```

**功能**：
- 从Bilibili爬取视频字幕（AI生成或人工上传）
- 如果没有字幕，使用Whisper进行转录
- 自动保存为文本文件

**输出**：
- `data/raw/audio/result/test*.txt` - 转录文本文件
- `data/transcript_results.json` - 转录结果统计

## ⚙️ 配置参数

### 采样策略配置

编辑 `1_analyze_and_sample.py` 中的参数：

```python
# 目标总样本数（None表示自动计算）
target_total_samples = None

# 每个方言最少样本数
min_samples_per_dialect = 30

# 平衡策略
balance_strategy = 'weighted'  # 'weighted', 'uniform', 'proportional'
```

**平衡策略说明**：

- **`weighted`** (加权): 保留原始分布，但确保少数类至少有最小样本数
- **`uniform`** (均匀): 每个方言的目标样本数相同
- **`proportional`** (按比例): 严格按照原始分布比例扩展

### Whisper模型选择

| 模型 | 速度 | 质量 | 推荐场景 |
|-----|-----|------|---------|
| tiny | ⚡⚡⚡⚡⚡ | ⭐ | 快速测试 |
| base | ⚡⚡⚡⚡ | ⭐⭐ | 日常使用（推荐） |
| small | ⚡⚡⚡ | ⭐⭐⭐ | 平衡选择 |
| medium | ⚡⚡ | ⭐⭐⭐⭐ | 高质量需求 |
| large/large-v3 | ⚡ | ⭐⭐⭐⭐⭐ | 最高质量 |

## 📊 数据流程图

```
┌─────────────────────────────────────┐
│  Excel (990条视频信息 + 方言标签)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  1. 智能采样分析                     │
│  - 分析方言分布                      │
│  - 计算目标样本数                    │
│  - 生成下载计划                      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. 选择性下载                       │
│  - 下载指定视频                      │
│  - 提取音频(16kHz WAV)               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. 字幕爬取                         │
│  - 尝试Bilibili字幕                  │
│  - 备选Whisper转录                   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  标注好的数据集                      │
│  - 音频文件 + 转录文本 + 方言标签    │
└─────────────────────────────────────┘
```

## 🔧 依赖工具

### 必需工具

```bash
# Python包
pip install pandas openpyxl requests beautifulsoup4 tqdm pyyaml

# 视频下载工具
pip install you-get

# 音频处理工具
apt-get install ffmpeg  # Ubuntu/Debian
# 或
brew install ffmpeg     # macOS
```

### 可选工具

```bash
# Whisper（用于备用转录）
pip install openai-whisper

# GPU加速（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📝 输出文件说明

### 1. 采样计划 (`data/sampling_plan.json`)

```json
{
  "config": {
    "target_total_samples": null,
    "min_samples_per_dialect": 30,
    "balance_strategy": "weighted"
  },
  "current_stats": {
    "total_videos": 990,
    "downloaded_count": 10,
    "excel_distribution": {...},
    "downloaded_distribution": {...}
  },
  "target_info": {
    "target_samples": {
      "beijing_mandarin": 30,
      "wu_dialect": 30,
      ...
    },
    "needed_samples": {
      "beijing_mandarin": 27,
      "wu_dialect": 28,
      ...
    }
  },
  "download_plan": [
    {
      "index": 15,
      "dialect": "beijing_mandarin",
      "uploader": "闲聊北京",
      "title": "...",
      "url": "https://..."
    },
    ...
  ]
}
```

### 2. 下载结果 (`data/download_results.json`)

```json
{
  "success": [
    {
      "index": 15,
      "dialect": "beijing_mandarin",
      "video_path": "data/raw/audio/elderly_videos/elderly_video_0015.mp4",
      "audio_path": "data/raw/audio/elderly_audios/elderly_audio_0015.wav",
      "uploader": "闲聊北京",
      "title": "..."
    }
  ],
  "failed": [],
  "skipped": []
}
```

### 3. 转录结果 (`data/transcript_results.json`)

```json
{
  "bilibili_subtitle": [
    {
      "index": 15,
      "dialect": "beijing_mandarin",
      "transcript_path": "data/raw/audio/result/test15.txt",
      "length": 523
    }
  ],
  "whisper_transcribed": [...],
  "failed": []
}
```

## ⚠️ 常见问题

### Q1: you-get下载失败

**原因**：网络问题、视频权限、cookies过期

**解决**：
```bash
# 1. 更新you-get
pip install --upgrade you-get

# 2. 使用cookies（需要登录Bilibili）
# 浏览器登录后导出cookies到文件
python scripts/whisper_data_collection/2_selective_download.py --cookies cookies.txt

# 3. 手动下载单个视频测试
you-get "https://www.bilibili.com/video/BV..."
```

### Q2: 某些方言可用视频不足

**现象**：
```
⚠️  警告: yue_dialect 只有 6 个可用视频（需要 30 个）
```

**解决**：
1. 调整`min_samples_per_dialect`参数
2. 修改`balance_strategy`为`proportional`
3. 手动添加更多该方言的视频到Excel

### Q3: Whisper转录速度太慢

**解决**：
```bash
# 1. 使用更小的模型
python scripts/whisper_data_collection/3_scrape_subtitles.py --whisper_model base

# 2. 使用GPU加速（需要CUDA）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 只使用Bilibili字幕
python scripts/whisper_data_collection/3_scrape_subtitles.py --no_whisper_fallback
```

### Q4: Bilibili字幕爬取失败

**原因**：
- 视频没有字幕
- API变更
- 请求过快被限制

**解决**：
```bash
# 1. 启用Whisper备选
python scripts/whisper_data_collection/3_scrape_subtitles.py --whisper_model base

# 2. 增加请求延迟（修改脚本中的delay参数）

# 3. 使用cookies
python scripts/whisper_data_collection/3_scrape_subtitles.py --cookies cookies.txt
```

### Q5: 断点续传

如果下载中断，可以从失败的地方继续：

```bash
# 查看已完成数量
ls data/raw/audio/elderly_audios/*.wav | wc -l

# 假设已完成30个，从第31个继续
python scripts/whisper_data_collection/2_selective_download.py --start_from 30
```

## 📈 性能优化建议

### 1. 下载加速

```bash
# 使用多个下载进程（需修改脚本）
# 或分批下载
python scripts/whisper_data_collection/2_selective_download.py --max_downloads 50 --start_from 0
python scripts/whisper_data_collection/2_selective_download.py --max_downloads 50 --start_from 50
```

### 2. Whisper加速

```bash
# 使用GPU
export CUDA_VISIBLE_DEVICES=0

# 使用FP16精度
# (需要修改脚本，添加fp16=True参数)
```

### 3. 并行处理

```bash
# 字幕爬取和视频下载可以并行
# 先爬取已有视频的字幕，同时下载新视频
```

## 🎯 最佳实践

1. **先小规模测试**：
   ```bash
   # 先下载5个测试
   python scripts/whisper_data_collection/2_selective_download.py --max_downloads 5
   ```

2. **检查数据质量**：
   ```bash
   # 检查转录文本
   cat data/raw/audio/result/test*.txt | head -20
   ```

3. **定期备份**：
   ```bash
   # 备份关键数据
   tar -czf whisper_data_backup_$(date +%Y%m%d).tar.gz \
       data/sampling_plan.json \
       data/download_results.json \
       data/transcript_results.json \
       data/raw/audio/elderly_audios/ \
       data/raw/audio/result/
   ```

4. **监控存储空间**：
   ```bash
   # 视频文件较大，及时清理
   du -sh data/raw/audio/elderly_videos/
   
   # 如果空间不足，提取音频后可删除视频
   rm data/raw/audio/elderly_videos/*.mp4
   ```

## 🔗 下一步

数据收集完成后：

1. **验证数据**：
   ```bash
   python scripts/0_validate_data.py
   ```

2. **数据预处理**：
   ```bash
   python scripts/1_prepare_dataset.py
   ```

3. **开始微调**：
   ```bash
   python scripts/2_finetune_whisper_lora.py
   ```

## 📚 参考资源

- [you-get文档](https://github.com/soimort/you-get)
- [Whisper文档](https://github.com/openai/whisper)
- [FFmpeg文档](https://ffmpeg.org/documentation.html)
- [Bilibili API参考](https://github.com/SocialSisterYi/bilibili-API-collect)

---

**祝数据收集顺利！🎉**

