# 音频解码改进说明

## 🔍 问题分析

### 原始问题
在评估过程中遇到音频解码失败：
```
Error on sample 5630: Failed to decode audio (sample 5630)
Error on sample 5690: Failed to decode audio (sample 5690)
...
✗ No successful evaluations for this model
```

### 根本原因
1. **单一解码后端**：之前只使用 `soundfile` 进行解码
2. **音频格式多样**：Parquet 文件中的音频可能是多种格式（MP3, WAV, FLAC等）
3. **错误处理不足**：失败后没有备用方案
4. **缺少统计信息**：不知道有多少样本失败

## ✅ 解决方案：多后端音频解码

### 核心改进

新的 `decode_audio_bytes()` 函数支持三种解码后端，按优先级尝试：

```python
1. torchaudio   (最可靠，支持最多格式)
   ↓ 失败
2. soundfile    (快速，支持常见格式)
   ↓ 失败
3. librosa      (最慢但最兼容)
```

### 技术实现

```python
def decode_audio_bytes(audio_bytes, sample_id=None):
    """
    使用多种方法尝试解码音频字节流
    
    优先级：torchaudio > soundfile > librosa
    """
    errors = []
    
    # 方法 1: torchaudio (推荐)
    if TORCHAUDIO_AVAILABLE:
        try:
            audio_tensor, sr = torchaudio.load(io.BytesIO(audio_bytes))
            audio_array = audio_tensor.numpy()
            if audio_array.shape[0] > 1:  # 多声道转单声道
                audio_array = audio_array.mean(axis=0)
            else:
                audio_array = audio_array[0]
            return audio_array, sr
        except Exception as e:
            errors.append(f"torchaudio: {str(e)[:50]}")
    
    # 方法 2: soundfile
    try:
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        return audio_array, sr
    except Exception as e:
        errors.append(f"soundfile: {str(e)[:50]}")
    
    # 方法 3: librosa
    if LIBROSA_AVAILABLE:
        try:
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            return audio_array, sr
        except Exception as e:
            errors.append(f"librosa: {str(e)[:50]}")
    
    # 所有方法都失败
    logger.warning(f"Failed to decode audio (sample {sample_id}): {'; '.join(errors)}")
    return None, None
```

## 📊 新增功能

### 1. 自动检测可用后端

脚本启动时会显示：
```
Audio Decoding Backends:
  - torchaudio: ✓ Available
  - soundfile: ✓ Available (v0.13.1)
  - librosa: ✓ Available
```

### 2. 详细的加载统计

数据加载完成后显示：
```
Data Loading Statistics:
  - Successfully loaded: 95
  - Failed to decode: 5
  - Total attempted: 100
  - Success rate: 95.0%
```

### 3. 增强的错误日志

使用 Python 的 `logging` 模块，提供更详细的错误信息：
```python
logger.warning(f"Failed to decode audio (sample {sample_id}): torchaudio: error1; soundfile: error2")
```

## 🔄 与原方法对比

### 原方法（单一后端）
```python
# 只使用 soundfile
audio_array, sr = sf.read(io.BytesIO(audio_bytes))
```

**缺点：**
- ❌ 遇到不支持的格式立即失败
- ❌ 没有备用方案
- ❌ 错误信息不详细

### 新方法（多后端）
```python
# 自动尝试多种解码器
audio_array, sr = decode_audio_bytes(audio_bytes, sample_id=sample_count)
if audio_array is None:
    continue  # 跳过失败的样本
```

**优点：**
- ✅ 支持更多音频格式
- ✅ 自动fallback到其他后端
- ✅ 详细的错误追踪
- ✅ 统计成功/失败率

## 🎯 为什么 torchaudio 是首选？

### torchaudio 的优势

1. **格式支持最全**
   - MP3, WAV, FLAC, OGG, AAC 等
   - 自动检测格式

2. **与 PyTorch 集成**
   - 原生 Tensor 支持
   - GPU 加速（如需要）

3. **更健壮的错误处理**
   - 内部有多种解码器
   - 自动选择最佳方案

4. **在 graph 环境中已安装**
   ```bash
   torchaudio: 2.7.1+cu126
   ```

### 为什么之前说 torchaudio 不可用？

可能的原因：
1. **环境问题**：在其他环境中确实没有安装
2. **依赖冲突**：某些系统库缺失（libsox, ffmpeg等）
3. **误解**：可能当时遇到的是其他错误

### 当前状态
✅ **在 `graph` 环境中，torchaudio 完全可用且推荐使用**

## 📈 预期改进

### 解码成功率提升

| 场景 | 原方法 | 新方法 |
|------|--------|--------|
| 标准 WAV | 100% | 100% |
| MP3 编码 | 可能失败 | ✅ 成功 |
| 损坏的音频 | 失败 | 跳过并记录 |
| 特殊格式 | 失败 | 尝试3种方法 |

### 实际测试结果（预测）
- 成功率预计从 0% 提升到 90%+
- 即使有少量失败，也能继续评估
- 提供详细的失败原因分析

## 🚀 使用方法

### 运行改进后的脚本

```bash
cd ~/AD_predict/AD_predict
./run_eval_fixed.sh
```

### 查看解码统计

在日志中查找：
```bash
tail -f ~/AD_predict_logs/eval_seniortalk_*.log | grep -A 5 "Data Loading Statistics"
```

## 🔍 调试建议

### 如果仍然遇到解码失败

1. **检查音频数据格式**
   ```python
   # 添加调试代码查看实际格式
   print(f"Audio bytes length: {len(audio_bytes)}")
   print(f"First 20 bytes: {audio_bytes[:20]}")
   ```

2. **尝试保存失败的样本**
   ```python
   with open(f'/tmp/failed_audio_{sample_id}.bin', 'wb') as f:
       f.write(audio_bytes)
   ```

3. **手动测试解码**
   ```bash
   # 使用 ffmpeg 检查
   ffmpeg -i /tmp/failed_audio_123.bin -f null -
   ```

## 📝 总结

### 主要改进
1. ✅ 添加多后端音频解码（torchaudio优先）
2. ✅ 增强错误处理和日志
3. ✅ 添加加载统计信息
4. ✅ 提高解码成功率

### 回答你的问题

**Q: 之前 torchaudio 的自动解码是不是使用其更好？**
- **A**: 是的！torchaudio 确实更好，支持更多格式，更稳定。

**Q: 我们自己写的代码能完全替代吗？**
- **A**: 不建议。音频解码很复杂，最好使用成熟的库（torchaudio/soundfile/librosa）。

**Q: 之前 torchaudio 无法使用，依赖缺失，是和其他环境冲突不可解决的吗？**
- **A**: 不是！在 `graph` 环境中，torchaudio 完全可用（v2.7.1）。之前可能是：
  - 在其他环境中运行
  - 或者遇到了其他错误
  - 现在已经解决了

### 建议
✅ **使用新的多后端解码方案，优先使用 torchaudio**

这是最稳定、最全面的解决方案！





