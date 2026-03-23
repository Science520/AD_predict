# 🖥️ Tmux使用指南 - 确保后台运行

## ⚡ 快速开始

### 启动评估任务（在tmux中）
```bash
cd /home/saisai/AD_predict/AD_predict
bash scripts/run_eval_tmux.sh
```

**就这么简单！** 启动后即可关闭电脑，任务继续运行。

---

## 📖 Tmux基础知识

### 什么是tmux？
Tmux是一个终端复用器，可以：
- ✅ SSH断开后程序继续运行
- ✅ 随时重新连接查看进度
- ✅ 管理多个终端窗口

### 基本概念
```
tmux会话 (Session)
├── 窗口1 (Window)  [运行评估]
├── 窗口2 (Window)  [查看日志]
└── 窗口3 (Window)  [其他任务]
```

---

## 🎮 常用命令

### 1. 会话管理

```bash
# 创建新会话
tmux new -s 会话名

# 列出所有会话
tmux list-sessions
tmux ls

# 连接到会话
tmux attach -t 会话名
tmux a -t 会话名

# 杀掉会话
tmux kill-session -t 会话名
```

### 2. 在会话中的操作

**所有tmux命令都以 `Ctrl+B` 开头（先按Ctrl+B，松开，再按其他键）**

```bash
Ctrl+B 然后 D     # 分离会话（程序继续运行）
Ctrl+B 然后 C     # 创建新窗口
Ctrl+B 然后 N     # 切换到下一个窗口
Ctrl+B 然后 P     # 切换到上一个窗口
Ctrl+B 然后 ,     # 重命名当前窗口
Ctrl+B 然后 [     # 进入滚动模式（查看历史输出）
Ctrl+B 然后 ?     # 显示所有快捷键
```

---

## 🚀 评估任务专用命令

### 场景1: 启动评估并外出
```bash
# 1. 启动tmux中的评估
bash scripts/run_eval_tmux.sh

# 2. 等几秒，看到评估开始运行

# 3. 分离会话
按键: Ctrl+B 然后 D

# 4. 现在可以关电脑了！
```

### 场景2: 回来后查看结果
```bash
# SSH重新连接到服务器后

# 1. 列出会话
tmux ls

# 2. 连接到评估会话
tmux attach -t whisper_eval

# 3. 查看进度
# 如果评估还在运行，会看到实时输出
# 如果已完成，会看到最终结果
```

### 场景3: 查看日志（不进入tmux）
```bash
# 实时查看
tail -f /tmp/cpu_eval_tmux.log

# 查看最后100行
tail -100 /tmp/cpu_eval_tmux.log

# 查看全部
cat /tmp/cpu_eval_tmux.log
```

### 场景4: 出问题了，需要重启
```bash
# 1. 杀掉旧会话
tmux kill-session -t whisper_eval

# 2. 重新启动
bash scripts/run_eval_tmux.sh
```

---

## 💡 实用技巧

### 技巧1: 滚动查看历史输出
```bash
# 在tmux会话中
Ctrl+B 然后 [     # 进入滚动模式
↑↓ 或 PgUp/PgDn   # 滚动
q                  # 退出滚动模式
```

### 技巧2: 分屏查看
```bash
Ctrl+B 然后 "     # 上下分屏
Ctrl+B 然后 %     # 左右分屏
Ctrl+B 然后 方向键 # 切换分屏
Ctrl+B 然后 x     # 关闭当前分屏
```

### 技巧3: 多窗口管理
```bash
# 窗口1: 运行评估
# 窗口2: 查看GPU状态 (watch -n 2 nvidia-smi)
# 窗口3: 查看日志 (tail -f /tmp/cpu_eval_tmux.log)

Ctrl+B 然后 C     # 创建新窗口
Ctrl+B 然后 0-9   # 快速切换到指定窗口
Ctrl+B 然后 W     # 显示窗口列表
```

---

## 🔧 常见问题

### Q1: tmux命令不存在？
```bash
# Ubuntu/Debian
sudo apt-get install tmux

# CentOS/RHEL
sudo yum install tmux
```

### Q2: 如何知道会话还在运行？
```bash
# 列出会话
tmux ls

# 如果看到 "whisper_eval: 1 windows" 就说明还在运行
```

### Q3: 忘记会话名称了？
```bash
# 列出所有会话
tmux ls

# 或者直接连接到最近的会话
tmux attach
```

### Q4: 不小心在tmux里按了Ctrl+C怎么办？
```bash
# 如果评估停止了
# 1. 退出tmux: Ctrl+B 然后 D
# 2. 重新启动: bash scripts/run_eval_tmux.sh
```

### Q5: 如何同时查看多个任务？
```bash
# 创建多个窗口
Ctrl+B 然后 C     # 新窗口1: 运行训练
Ctrl+B 然后 C     # 新窗口2: 监控GPU
Ctrl+B 然后 C     # 新窗口3: 查看日志

# 快速切换
Ctrl+B 然后 0     # 切换到窗口0
Ctrl+B 然后 1     # 切换到窗口1
Ctrl+B 然后 2     # 切换到窗口2
```

---

## 📊 当前评估任务的tmux工作流

### 完整流程
```bash
# 1️⃣ 启动（只需一次）
cd /home/saisai/AD_predict/AD_predict
bash scripts/run_eval_tmux.sh
# 看到 "开始CPU评估" 后按 Ctrl+B D 分离

# 2️⃣ 外出期间
# 关电脑，服务器继续跑

# 3️⃣ 回来后
# SSH连接服务器
tmux attach -t whisper_eval
# 查看进度或结果

# 4️⃣ 获取结果
cat /data/AD_predict/all_experiments_20251022_140017/CPU_EVALUATION_REPORT.md
```

---

## 🎯 与nohup的对比

| 特性 | nohup | tmux |
|------|-------|------|
| 后台运行 | ✅ | ✅ |
| SSH断开继续 | ✅ | ✅ |
| 重新连接查看 | ❌ | ✅ ⭐ |
| 交互式操作 | ❌ | ✅ ⭐ |
| 多窗口管理 | ❌ | ✅ ⭐ |
| 学习成本 | 低 | 中 |

**推荐**: 长时间任务用tmux，短任务用nohup

---

## 🚨 重要提醒

### ✅ 做到这些，任务永不丢失
1. 总是在tmux中运行长时间任务
2. 分离会话前确认任务已启动
3. 定期检查会话是否还在运行

### ❌ 避免这些错误
1. 不要在tmux外运行长任务
2. 不要关闭tmux窗口（用Ctrl+B D分离）
3. 不要在tmux里按Ctrl+C停止重要任务

---

## 📚 深入学习

### 官方资源
- [tmux官方文档](https://github.com/tmux/tmux/wiki)
- [tmux速查表](https://tmuxcheatsheet.com/)

### 配置文件（可选）
创建 `~/.tmux.conf` 自定义快捷键：
```bash
# 示例配置
# 更改前缀键为 Ctrl+A（可选）
# set -g prefix C-a
# unbind C-b

# 鼠标支持
set -g mouse on

# 窗口编号从1开始
set -g base-index 1

# 更好的颜色
set -g default-terminal "screen-256color"
```

---

## ✅ 检查清单

在外出前确认：
- [ ] 评估已在tmux中启动
- [ ] 看到 "开始CPU评估" 输出
- [ ] 按 Ctrl+B D 成功分离
- [ ] `tmux ls` 确认会话存在
- [ ] `tail /tmp/cpu_eval_tmux.log` 确认有输出

---

**🎉 现在您可以安心外出了！任务会在服务器上继续运行。**

回来后只需：
```bash
tmux attach -t whisper_eval
```

