# Hugging Face 网络问题解决方案

## 📋 问题描述

在服务器上运行评估脚本时遇到网络错误：
```
OSError: Can't load feature extractor for 'openai/whisper-medium'. 
If you were trying to load it from 'https://huggingface.co/models'...
```

**原因**: 无法访问 Hugging Face 官方服务器（huggingface.co）

---

## ✅ 解决方案对比

### 方案 1: 使用镜像站（推荐 ⭐⭐⭐⭐⭐）

**优点:**
- ✅ 最简单，只需设置环境变量
- ✅ 无需额外软件
- ✅ 稳定快速（国内服务器）
- ✅ 自动同步，模型最新

**缺点:**
- 无

**实施方法:**
```bash
# 直接运行使用镜像的脚本
cd ~/AD_predict/AD_predict
./run_eval_mirror.sh
```

---

### 方案 2: 从 Backup 复制模型（推荐 ⭐⭐⭐⭐）

**优点:**
- ✅ 完全离线，不需要网络
- ✅ 一次复制，永久使用

**缺点:**
- ⚠️ 需要先找到 backup 位置
- ⚠️ 占用磁盘空间

**实施方法:**
```bash
# 1. 先找到 backup 目录
find /data -name "backup" -type d 2>/dev/null
find ~ -name "backup" -type d 2>/dev/null

# 2. 找到后复制模型
# 假设 backup 在 /data/backup
mkdir -p ~/.cache/huggingface/hub/models--openai--whisper-medium
cp -r /data/backup/whisper-medium/* ~/.cache/huggingface/hub/models--openai--whisper-medium/

# 3. 运行评估
./run_eval_mirror.sh  # 或 run_eval_fixed.sh
```

---

### 方案 3: SSH 隧道（从 Mac）（不推荐 ⭐⭐）

**优点:**
- ✓ 可以使用 Mac 的网络

**缺点:**
- ❌ 复杂，需要 Mac 始终在线
- ❌ 速度慢
- ❌ 不稳定

**实施方法:**
```bash
# 在 Mac 上执行
ssh -D 1080 -N user@server

# 在服务器上
export HTTP_PROXY=socks5://localhost:1080
export HTTPS_PROXY=socks5://localhost:1080
```

---

### 方案 4: 服务器安装 Clash（不推荐 ⭐）

**优点:**
- ✓ 可以访问更多资源

**缺点:**
- ❌ 需要订阅
- ❌ 可能违反服务器政策
- ❌ 配置复杂
- ❌ 不稳定

**不建议使用**

---

## 🚀 推荐执行顺序

### 第一步：尝试镜像站（最简单）

```bash
cd ~/AD_predict/AD_predict
./run_eval_mirror.sh
```

这会自动：
1. 设置 `HF_ENDPOINT=https://hf-mirror.com`
2. 从镜像站下载模型（如果缓存中没有）
3. 开始评估

**查看进度:**
```bash
# 连接到 tmux 会话
tmux attach -t seniortalk_eval_mirror

# 或查看日志
tail -f ~/AD_predict_logs/eval_mirror_*.log
```

---

### 第二步：如果镜像站也慢，寻找 Backup

**帮我找到你的 backup 目录:**

```bash
# 运行这个命令查找
find /data -maxdepth 3 -name "*whisper*" -type d 2>/dev/null | grep -i backup
find ~/ -maxdepth 3 -name "*backup*" -type d 2>/dev/null
find /mnt -maxdepth 3 -name "*backup*" -type d 2>/dev/null

# 或者你记得 backup 的大致位置？
ls -lah /data/
ls -lah ~/
```

**找到后告诉我，我会帮你复制模型文件到正确位置。**

---

## 📊 可用的镜像站

### 1. hf-mirror.com（默认使用）
```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```
- 速度：⭐⭐⭐⭐⭐
- 稳定：⭐⭐⭐⭐⭐
- 推荐：是

### 2. ModelScope（阿里云）
```python
os.environ['HF_ENDPOINT'] = 'https://www.modelscope.cn'
```
- 速度：⭐⭐⭐⭐
- 稳定：⭐⭐⭐⭐⭐
- 推荐：备用方案

### 切换镜像站

如果 hf-mirror.com 不好用，修改脚本：

```bash
# 编辑脚本
nano ~/AD_predict/AD_predict/scripts/eval_seniortalk_with_mirror.py

# 找到第 22 行，修改为：
# os.environ['HF_ENDPOINT'] = 'https://www.modelscope.cn'
```

---

## 🔍 验证网络配置

### 测试是否能访问镜像站

```bash
# 测试 hf-mirror.com
curl -I https://hf-mirror.com

# 测试 ModelScope
curl -I https://www.modelscope.cn
```

**预期输出**: `HTTP/2 200` 或类似成功响应

---

## 📁 关于 Backup 目录

你提到在 backup 里存了模型。常见的位置：

```bash
# 可能的位置 1: /data 下
/data/backup/
/data/AD_predict/backup/
/data/cache/backup/

# 可能的位置 2: 用户目录
~/backup/
~/AD_predict/backup/

# 可能的位置 3: 外部挂载
/mnt/backup/
/media/backup/
```

### 如何检查 Backup 中的模型

```bash
# 假设找到了 backup 目录
cd /path/to/backup

# 查看结构
ls -lh

# 查找 whisper 相关文件
find . -name "*whisper*" -type d

# 查看模型文件
ls -lh whisper-medium/  # 应该包含 config.json, pytorch_model.bin 等
```

### 从 Backup 恢复模型

如果找到了 backup，运行：

```bash
# 我会帮你创建一个自动复制脚本
# 告诉我 backup 的完整路径即可
```

---

## 🎯 快速决策树

```
需要下载 HF 模型？
│
├─ 有网络？
│  ├─ 是 → 使用镜像站 ✅ (run_eval_mirror.sh)
│  └─ 否 → 检查 Backup 📦
│
└─ 有 Backup？
   ├─ 是 → 复制模型 ✅
   └─ 否 → 需要联网（镜像站或代理）
```

---

## ✅ 当前推荐方案

### 立即执行（使用镜像站）

```bash
cd ~/AD_predict/AD_predict
./run_eval_mirror.sh
```

这是最简单、最可靠的方案！

### 如果需要离线使用

1. **先告诉我你的 backup 在哪里**
   ```bash
   # 运行这个找找
   find /data -name "backup" -type d 2>/dev/null
   ```

2. **我会帮你创建复制脚本**

---

## 🆘 常见问题

### Q1: 镜像站会同步最新模型吗？
**A**: 是的，镜像站会自动同步 Hugging Face 的最新模型。

### Q2: 使用镜像站安全吗？
**A**: 安全。hf-mirror.com 是社区维护的官方镜像，ModelScope 是阿里云的服务。

### Q3: 第一次下载会很慢吗？
**A**: 
- whisper-medium: ~1.5GB，大约需要 5-15 分钟
- whisper-large-v3: ~3GB，大约需要 10-30 分钟
- 下载一次后会缓存，以后就很快了

### Q4: 下载到哪里？
**A**: `~/.cache/huggingface/hub/`

### Q5: 如何查看下载进度？
**A**:
```bash
# 查看缓存目录大小
du -sh ~/.cache/huggingface/

# 实时监控
watch -n 1 'du -sh ~/.cache/huggingface/'
```

### Q6: 如果 backup 在只读的 /data 分区？
**A**: 可以先复制到临时目录，再移到缓存：
```bash
cp -r /data/backup/models /tmp/
cp -r /tmp/models ~/.cache/huggingface/hub/
```

---

## 📝 总结

| 方案 | 难度 | 速度 | 推荐度 |
|------|------|------|--------|
| 镜像站 | ⭐ 简单 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Backup | ⭐⭐ 中等 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| SSH 隧道 | ⭐⭐⭐⭐ 复杂 | ⭐⭐ | ⭐⭐ |
| Clash | ⭐⭐⭐⭐⭐ 很复杂 | ⭐⭐⭐ | ⭐ |

**我的建议: 直接使用镜像站方案！**

```bash
cd ~/AD_predict/AD_predict
./run_eval_mirror.sh
```

简单、快速、稳定！🚀





