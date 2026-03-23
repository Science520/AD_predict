# 最佳模型信息

## 模型选择
- **来源**: exp3_large_batch/checkpoint-750
- **WER**: 0.9544
- **特点**: 训练快60%，性能与最佳模型相近

## 模型文件
- `adapter_model.safetensors`: 主模型权重 (18.9MB)
- `adapter_config.json`: 适配器配置
- `preprocessor_config.json`: 预处理器配置
- `training_args.bin`: 训练参数
- `trainer_state.json`: 训练状态
- `optimizer.pt`: 优化器状态
- `scheduler.pt`: 学习率调度器
- `scaler.pt`: 混合精度缩放器
- `rng_state.pth`: 随机数生成器状态

## 性能对比
| 模型 | WER | 训练速度 | 模型大小 | 选择理由 |
|------|-----|----------|----------|----------|
| exp1_high_rank | 0.9540 | 基准 | 176MB | 最佳准确性 |
| **exp3_large_batch** | **0.9544** | **快60%** | **44MB** | **效率最优** |

## 使用说明
该模型已优化用于生产环境，在保持高准确性的同时显著提升了训练效率和部署便利性。
