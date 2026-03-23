"""
数据验证脚本
在运行数据预处理前，使用此脚本检查数据完整性和质量
"""

import os
import sys
import pandas as pd
import yaml
from pathlib import Path
from collections import Counter
import librosa


def load_config(config_path="configs/training_args.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def validate_excel_file(excel_path):
    """
    验证Excel文件
    
    Returns:
        DataFrame or None
    """
    print("\n" + "=" * 80)
    print("1. 验证Excel文件")
    print("=" * 80)
    
    if not os.path.exists(excel_path):
        print(f"❌ Excel文件不存在: {excel_path}")
        return None
    
    print(f"✅ Excel文件存在: {excel_path}")
    
    try:
        df = pd.read_excel(excel_path)
        print(f"✅ 成功读取Excel文件")
        print(f"   总行数: {len(df)}")
        print(f"   列名: {df.columns.tolist()}")
        
        # 检查必需列
        required_columns = ['dialect_label']
        for col in required_columns:
            if col in df.columns:
                print(f"   ✅ 包含必需列: {col}")
            else:
                print(f"   ❌ 缺少必需列: {col}")
        
        # 检查方言标签
        if 'dialect_label' in df.columns:
            dialect_counts = df['dialect_label'].value_counts()
            print(f"\n   方言分布:")
            for dialect, count in dialect_counts.items():
                print(f"     {dialect}: {count}")
            
            # 检查空值
            null_count = df['dialect_label'].isnull().sum()
            if null_count > 0:
                print(f"   ⚠️  警告: {null_count} 行方言标签为空")
        
        return df
    
    except Exception as e:
        print(f"❌ 读取Excel文件失败: {e}")
        return None


def validate_audio_files(base_dir):
    """
    验证音频文件
    
    Returns:
        list of audio files
    """
    print("\n" + "=" * 80)
    print("2. 验证音频文件")
    print("=" * 80)
    
    audio_dir = os.path.join(base_dir, "elderly_audios")
    
    if not os.path.exists(audio_dir):
        print(f"❌ 音频目录不存在: {audio_dir}")
        return []
    
    print(f"✅ 音频目录存在: {audio_dir}")
    
    # 查找音频文件
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    print(f"✅ 找到 {len(audio_files)} 个音频文件")
    
    if len(audio_files) > 0:
        # 验证几个音频文件
        sample_size = min(3, len(audio_files))
        print(f"\n   验证前 {sample_size} 个音频文件:")
        
        for i, audio_file in enumerate(audio_files[:sample_size]):
            audio_path = os.path.join(audio_dir, audio_file)
            try:
                audio, sr = librosa.load(audio_path, sr=None)
                duration = len(audio) / sr
                print(f"   ✅ {audio_file}")
                print(f"      采样率: {sr} Hz, 时长: {duration:.2f} 秒")
            except Exception as e:
                print(f"   ❌ {audio_file}: {e}")
    
    return audio_files


def validate_transcripts(base_dir):
    """
    验证转录文本
    
    Returns:
        list of transcript files
    """
    print("\n" + "=" * 80)
    print("3. 验证转录文本")
    print("=" * 80)
    
    result_dir = os.path.join(base_dir, "result")
    
    if not os.path.exists(result_dir):
        print(f"❌ 转录文本目录不存在: {result_dir}")
        return []
    
    print(f"✅ 转录文本目录存在: {result_dir}")
    
    # 查找文本文件
    txt_files = [f for f in os.listdir(result_dir) if f.endswith('.txt')]
    print(f"✅ 找到 {len(txt_files)} 个文本文件")
    
    if len(txt_files) > 0:
        # 验证几个文本文件
        sample_size = min(3, len(txt_files))
        print(f"\n   验证前 {sample_size} 个文本文件:")
        
        for txt_file in txt_files[:sample_size]:
            txt_path = os.path.join(result_dir, txt_file)
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    lines = content.split('\n')
                    print(f"   ✅ {txt_file}")
                    print(f"      行数: {len(lines)}, 字符数: {len(content)}")
                    print(f"      前50字: {content[:50]}...")
            except Exception as e:
                print(f"   ❌ {txt_file}: {e}")
    
    return txt_files


def check_data_balance(df):
    """
    检查数据平衡性
    """
    print("\n" + "=" * 80)
    print("4. 数据平衡性分析")
    print("=" * 80)
    
    if df is None or 'dialect_label' not in df.columns:
        print("❌ 无法分析：Excel数据不可用")
        return
    
    dialect_counts = df['dialect_label'].value_counts()
    total = len(df)
    
    print("\n   详细分布:")
    print(f"   {'方言':<25} {'样本数':<10} {'占比':<10} {'状态'}")
    print("   " + "-" * 70)
    
    for dialect, count in dialect_counts.items():
        percentage = (count / total) * 100
        
        # 判断状态
        if count < 5:
            status = "❌ 严重不足"
        elif count < 10:
            status = "⚠️  需要增强"
        elif count < 50:
            status = "✓ 可用"
        else:
            status = "✅ 充足"
        
        print(f"   {str(dialect):<25} {count:<10} {percentage:>6.2f}%   {status}")
    
    # 建议
    print("\n   建议:")
    minority_dialects = [d for d, c in dialect_counts.items() if c < 10]
    if len(minority_dialects) > 0:
        print(f"   ⚠️  以下方言样本不足，建议启用数据增强:")
        for d in minority_dialects:
            print(f"      - {d}: {dialect_counts[d]} 个样本")
    else:
        print("   ✅ 所有方言类别样本数量充足")


def check_audio_transcript_mapping(audio_files, txt_files, base_dir):
    """
    检查音频和转录文本的映射关系
    """
    print("\n" + "=" * 80)
    print("5. 音频-文本映射检查")
    print("=" * 80)
    
    # 提取音频ID
    audio_ids = set()
    for audio_file in audio_files:
        # elderly_audio_0001.wav -> 0001
        audio_id = audio_file.replace('elderly_audio_', '').replace('.wav', '')
        audio_ids.add(int(audio_id))
    
    # 提取文本ID
    txt_ids = set()
    for txt_file in txt_files:
        # test1.txt -> 1
        txt_id = txt_file.replace('test', '').replace('.txt', '')
        try:
            txt_ids.add(int(txt_id))
        except ValueError:
            continue
    
    print(f"   音频ID范围: {min(audio_ids) if audio_ids else 'N/A'} - {max(audio_ids) if audio_ids else 'N/A'}")
    print(f"   文本ID范围: {min(txt_ids) if txt_ids else 'N/A'} - {max(txt_ids) if txt_ids else 'N/A'}")
    
    # 检查匹配
    matched = audio_ids & txt_ids
    audio_only = audio_ids - txt_ids
    txt_only = txt_ids - audio_ids
    
    print(f"\n   匹配统计:")
    print(f"   ✅ 匹配的对: {len(matched)}")
    print(f"   ⚠️  只有音频: {len(audio_only)}")
    print(f"   ⚠️  只有文本: {len(txt_only)}")
    
    if len(matched) > 0:
        match_rate = len(matched) / max(len(audio_ids), len(txt_ids)) * 100
        print(f"   匹配率: {match_rate:.2f}%")
        
        if match_rate < 50:
            print("   ❌ 警告: 匹配率过低，请检查文件命名规则")
    
    # 显示一些不匹配的ID
    if len(audio_only) > 0:
        print(f"\n   只有音频的ID (前10个): {sorted(list(audio_only))[:10]}")
    if len(txt_only) > 0:
        print(f"   只有文本的ID (前10个): {sorted(list(txt_only))[:10]}")


def generate_data_report(config, df, audio_files, txt_files):
    """
    生成数据报告
    """
    print("\n" + "=" * 80)
    print("6. 生成数据报告")
    print("=" * 80)
    
    report = []
    report.append("# 数据验证报告\n")
    report.append(f"## 概览\n")
    report.append(f"- Excel行数: {len(df) if df is not None else 0}\n")
    report.append(f"- 音频文件数: {len(audio_files)}\n")
    report.append(f"- 文本文件数: {len(txt_files)}\n")
    
    if df is not None and 'dialect_label' in df.columns:
        report.append(f"\n## 方言分布\n")
        dialect_counts = df['dialect_label'].value_counts()
        for dialect, count in dialect_counts.items():
            report.append(f"- {dialect}: {count}\n")
    
    report_path = "data_validation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✅ 报告已保存到: {report_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("数据验证工具")
    print("=" * 80)
    
    # 加载配置
    try:
        config = load_config()
        excel_path = config['data']['excel_path']
        audio_base_dir = config['data']['audio_base_dir'].replace('/elderly_audios', '')
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        print("使用默认路径...")
        excel_path = "/data/AD_predict/data/raw/audio/老人视频信息_final_complete_20251016_214400.xlsx"
        audio_base_dir = "/data/AD_predict/data/raw/audio"
    
    # 1. 验证Excel
    df = validate_excel_file(excel_path)
    
    # 2. 验证音频
    audio_files = validate_audio_files(audio_base_dir)
    
    # 3. 验证转录文本
    txt_files = validate_transcripts(audio_base_dir)
    
    # 4. 检查数据平衡
    check_data_balance(df)
    
    # 5. 检查映射
    if len(audio_files) > 0 and len(txt_files) > 0:
        check_audio_transcript_mapping(audio_files, txt_files, audio_base_dir)
    
    # 6. 生成报告
    generate_data_report(config, df, audio_files, txt_files)
    
    # 总结
    print("\n" + "=" * 80)
    print("验证完成！")
    print("=" * 80)
    
    # 提供建议
    print("\n📋 下一步:")
    if df is not None and len(audio_files) > 0 and len(txt_files) > 0:
        print("✅ 数据看起来正常，可以运行:")
        print("   python scripts/1_prepare_dataset.py")
    else:
        print("⚠️  请先解决上述问题，然后重新运行此验证脚本")
    
    print("\n💡 提示:")
    print("   - 如果有方言样本不足，请在 configs/training_args.yaml 中启用数据增强")
    print("   - 检查 data_validation_report.txt 获取详细信息")


if __name__ == "__main__":
    main()


