"""
查看Whisper训练进度
实时显示训练状态、loss、WER等指标
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
import glob

def format_time(seconds):
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_latest_checkpoint(output_dir):
    """获取最新的checkpoint"""
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    latest = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
    return latest

def read_trainer_state(checkpoint_dir):
    """读取trainer state"""
    state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return None

def display_progress():
    """显示训练进度"""
    output_dir = "./whisper_lora_dialect"
    
    print("=" * 80)
    print("🎯 Whisper LoRA 训练进度监控")
    print("=" * 80)
    
    if not os.path.exists(output_dir):
        print(f"\n❌ 输出目录不存在: {output_dir}")
        print("训练可能尚未开始")
        return
    
    # 检查最新checkpoint
    latest_checkpoint = get_latest_checkpoint(output_dir)
    
    if not latest_checkpoint:
        print("\n⏳ 尚未生成checkpoint，训练可能刚开始...")
        
        # 检查训练日志
        log_file = "./whisper_training.log"
        if os.path.exists(log_file):
            print(f"\n📝 训练日志最后20行:")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(line.rstrip())
        return
    
    print(f"\n📂 最新checkpoint: {os.path.basename(latest_checkpoint)}")
    
    # 读取trainer state
    state = read_trainer_state(latest_checkpoint)
    
    if state:
        print("\n" + "=" * 80)
        print("📊 训练统计")
        print("=" * 80)
        
        # 基本信息
        current_step = state.get('global_step', 0)
        max_steps = state.get('max_steps', 0)
        epoch = state.get('epoch', 0)
        
        print(f"\n🔢 进度:")
        print(f"  当前步数: {current_step} / {max_steps}")
        if max_steps > 0:
            progress_pct = (current_step / max_steps) * 100
            bar_length = 50
            filled = int(bar_length * current_step / max_steps)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"  进度条:   [{bar}] {progress_pct:.1f}%")
        print(f"  当前轮次: {epoch:.2f}")
        
        # Loss信息
        log_history = state.get('log_history', [])
        if log_history:
            print(f"\n📉 最近指标:")
            
            # 获取最近的训练loss
            train_losses = [entry for entry in log_history if 'loss' in entry]
            if train_losses:
                latest_train = train_losses[-1]
                print(f"  训练Loss: {latest_train.get('loss', 'N/A'):.4f} (step {latest_train.get('step', 'N/A')})")
            
            # 获取最近的验证指标
            eval_metrics = [entry for entry in log_history if 'eval_loss' in entry or 'eval_wer' in entry]
            if eval_metrics:
                latest_eval = eval_metrics[-1]
                print(f"  验证Loss: {latest_eval.get('eval_loss', 'N/A'):.4f}")
                print(f"  验证WER:  {latest_eval.get('eval_wer', 'N/A'):.4f}")
                print(f"  验证步数: {latest_eval.get('step', 'N/A')}")
        
        # 时间信息
        best_metric = state.get('best_metric')
        best_model_checkpoint = state.get('best_model_checkpoint')
        
        if best_metric is not None:
            print(f"\n🏆 最佳模型:")
            print(f"  最佳WER: {best_metric:.4f}")
            if best_model_checkpoint:
                print(f"  保存位置: {os.path.basename(best_model_checkpoint)}")
    
    # 显示所有checkpoints
    all_checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    if all_checkpoints:
        print(f"\n💾 已保存的checkpoints ({len(all_checkpoints)}):")
        for cp in all_checkpoints[-5:]:  # 显示最近5个
            cp_name = os.path.basename(cp)
            # 获取文件修改时间
            mtime = os.path.getmtime(cp)
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {cp_name} (更新: {mtime_str})")
    
    # 检查是否完成
    final_adapter = os.path.join(output_dir, "final_adapter")
    if os.path.exists(final_adapter):
        print("\n" + "=" * 80)
        print("✅ 训练已完成!")
        print("=" * 80)
        print(f"最终适配器保存位置: {final_adapter}")
        
        # 读取最终指标
        train_results = os.path.join(output_dir, "train_results.json")
        if os.path.exists(train_results):
            with open(train_results, 'r') as f:
                results = json.load(f)
                print("\n📊 最终训练指标:")
                for key, value in results.items():
                    print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def watch_progress(interval=30):
    """持续监控训练进度"""
    print("🔄 开始实时监控 (按Ctrl+C退出)")
    print(f"刷新间隔: {interval}秒\n")
    
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            display_progress()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 停止监控")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="监控Whisper训练进度")
    parser.add_argument("--watch", action="store_true", help="持续监控模式")
    parser.add_argument("--interval", type=int, default=30, help="刷新间隔(秒)")
    
    args = parser.parse_args()
    
    if args.watch:
        watch_progress(args.interval)
    else:
        display_progress()

