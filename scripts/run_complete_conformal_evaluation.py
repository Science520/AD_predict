#!/usr/bin/env python3
"""
完整的Conformal ASR评估流程

集成：
1. 视频下载和音频提取（可选）
2. ASR评估（有/无Conformal Inference）
3. 结果可视化
4. PMM患者分层测试
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str):
    """运行命令并处理输出"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{description}")
    logger.info(f"{'='*60}")
    logger.info(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} 失败: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ {description} 异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="完整的Conformal ASR评估流程")
    
    # 数据选项
    parser.add_argument('--download_videos', action='store_true',
                       help='下载新视频（需要you-get）')
    parser.add_argument('--max_videos', type=int, default=10,
                       help='最大下载视频数')
    parser.add_argument('--start_idx', type=int, default=1,
                       help='视频起始索引')
    
    # 评估选项
    parser.add_argument('--audio_dir', type=str,
                       default='data/raw/audio/elderly_audios',
                       help='音频目录')
    parser.add_argument('--subtitle_dir', type=str,
                       default='data/raw/audio/result',
                       help='字幕目录（真实标注）')
    parser.add_argument('--max_samples', type=int, default=20,
                       help='最大评估样本数')
    parser.add_argument('--calibration_ratio', type=float, default=0.3,
                       help='校准集比例')
    
    # 模型选项
    parser.add_argument('--model_name', type=str, default='large-v3',
                       help='Whisper模型名称')
    parser.add_argument('--coverage', type=float, default=0.95,
                       help='Conformal覆盖率')
    
    # 输出选项
    parser.add_argument('--output_dir', type=str,
                       default='experiments/conformal_evaluation',
                       help='输出目录')
    
    # 流程控制
    parser.add_argument('--skip_download', action='store_true',
                       help='跳过视频下载')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='跳过ASR评估')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='跳过可视化')
    parser.add_argument('--skip_pmm', action='store_true',
                       help='跳过PMM测试')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("Conformal ASR完整评估流程")
    logger.info("="*80)
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    success_steps = []
    failed_steps = []
    
    # 步骤1: 下载视频和提取音频（可选）
    if not args.skip_download and args.download_videos:
        logger.info("\n步骤1: 下载视频并提取音频")
        
        cmd = [
            'python', 'scripts/download_elderly_videos_updated.py',
            '--max_videos', str(args.max_videos),
            '--start', str(args.start_idx),
            '--audio_dir', args.audio_dir
        ]
        
        if run_command(cmd, "视频下载和音频提取"):
            success_steps.append("视频下载和音频提取")
        else:
            failed_steps.append("视频下载和音频提取")
            logger.warning("⚠️ 视频下载失败，将使用现有音频文件继续")
    else:
        logger.info("\n跳过视频下载步骤")
    
    # 检查音频文件是否存在
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists() or len(list(audio_dir.glob("*.wav"))) == 0:
        logger.error(f"音频目录不存在或为空: {audio_dir}")
        logger.info("尝试使用已有的样本音频...")
        
        # 使用已有的样本音频
        sample_audio_dir = Path("data/processed/seniortalk_samples/audio")
        if sample_audio_dir.exists():
            args.audio_dir = str(sample_audio_dir)
            logger.info(f"使用样本音频目录: {sample_audio_dir}")
        else:
            logger.error("没有可用的音频文件，退出")
            return
    
    # 步骤2: ASR评估
    if not args.skip_evaluation:
        logger.info("\n步骤2: 运行Conformal ASR评估")
        
        cmd = [
            'python', 'scripts/evaluate_conformal_asr.py',
            '--audio_dir', args.audio_dir,
            '--subtitle_dir', args.subtitle_dir,
            '--output_dir', args.output_dir,
            '--model_name', args.model_name,
            '--coverage', str(args.coverage),
            '--calibration_ratio', str(args.calibration_ratio),
            '--max_samples', str(args.max_samples)
        ]
        
        if run_command(cmd, "Conformal ASR评估"):
            success_steps.append("ASR评估")
        else:
            failed_steps.append("ASR评估")
            logger.error("ASR评估失败，无法继续后续步骤")
            return
    else:
        logger.info("\n跳过ASR评估步骤")
    
    # 步骤3: 可视化
    if not args.skip_visualization:
        logger.info("\n步骤3: 生成可视化图表")
        
        cmd = [
            'python', 'scripts/visualize_conformal_comparison.py',
            '--results_dir', args.output_dir,
            '--output_dir', f'{args.output_dir}/visualizations'
        ]
        
        if run_command(cmd, "生成可视化图表"):
            success_steps.append("可视化")
        else:
            failed_steps.append("可视化")
    else:
        logger.info("\n跳过可视化步骤")
    
    # 步骤4: PMM患者分层测试
    if not args.skip_pmm:
        logger.info("\n步骤4: 测试PMM患者分层")
        
        cmd = [
            'python', 'scripts/test_pmm_stratification.py'
        ]
        
        if run_command(cmd, "PMM患者分层测试"):
            success_steps.append("PMM分层测试")
        else:
            failed_steps.append("PMM分层测试")
    else:
        logger.info("\n跳过PMM分层测试")
    
    # 总结
    logger.info("\n" + "="*80)
    logger.info("评估流程完成总结")
    logger.info("="*80)
    
    if success_steps:
        logger.info("\n✅ 成功完成的步骤:")
        for step in success_steps:
            logger.info(f"  - {step}")
    
    if failed_steps:
        logger.info("\n❌ 失败的步骤:")
        for step in failed_steps:
            logger.info(f"  - {step}")
    
    logger.info(f"\n📁 结果保存位置:")
    logger.info(f"  - 评估结果: {args.output_dir}")
    logger.info(f"  - 可视化图表: {args.output_dir}/visualizations")
    logger.info(f"  - PMM分层结果: experiments/pmm_evaluation")
    
    logger.info("\n" + "="*80)
    
    if len(failed_steps) == 0:
        logger.info("✅ 所有步骤成功完成！")
    else:
        logger.warning(f"⚠️ {len(failed_steps)} 个步骤失败")


if __name__ == "__main__":
    main()

