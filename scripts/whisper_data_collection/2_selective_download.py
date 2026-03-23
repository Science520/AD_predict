"""
选择性下载脚本
根据采样计划选择性下载视频和提取音频
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import logging
import time
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sampling_plan(plan_path="data/sampling_plan.json"):
    """加载采样计划"""
    
    if not os.path.exists(plan_path):
        logger.error(f"采样计划文件不存在: {plan_path}")
        logger.error("请先运行: python scripts/whisper_data_collection/1_analyze_and_sample.py")
        return None
    
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    logger.info(f"加载采样计划: {plan_path}")
    logger.info(f"需要下载 {len(plan['download_plan'])} 个视频")
    
    return plan


def download_single_video(
    video_info,
    output_dir,
    cookies_file=None,
    timeout=600
):
    """
    下载单个视频
    
    Returns:
        str: 下载的视频文件路径，失败返回None
    """
    
    idx = video_info['index']
    url = video_info['url']
    
    if not url:
        logger.warning(f"视频 {idx} URL为空")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_name = f"elderly_video_{idx:04d}"
    
    # 检查是否已下载
    for ext in ['.mp4', '.flv', '.mkv']:
        existing_file = output_path / f"{file_name}{ext}"
        if existing_file.exists():
            logger.info(f"视频 {idx} 已存在: {existing_file.name}")
            return str(existing_file)
    
    logger.info(f"下载视频 {idx}: {video_info.get('title', '')[:40]}...")
    
    # 构建you-get命令
    cmd = ['you-get', url, '-O', file_name, '-o', str(output_path)]
    
    if cookies_file and Path(cookies_file).exists():
        cmd.extend(['--cookies', cookies_file])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            # 查找下载的文件
            for ext in ['.mp4', '.flv', '.mkv']:
                video_file = output_path / f"{file_name}{ext}"
                if video_file.exists():
                    size_mb = video_file.stat().st_size / (1024 * 1024)
                    logger.info(f"✅ 下载成功: {video_file.name} ({size_mb:.1f} MB)")
                    return str(video_file)
            
            logger.warning(f"⚠️  下载命令成功但未找到文件")
            return None
        else:
            logger.error(f"❌ 下载失败: {result.stderr[:200]}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ 下载超时（{timeout}秒）")
        return None
    except Exception as e:
        logger.error(f"❌ 下载异常: {e}")
        return None


def extract_audio_from_video(
    video_path,
    output_audio_path,
    sample_rate=16000
):
    """从视频提取音频"""
    
    video_path = Path(video_path)
    output_audio_path = Path(output_audio_path)
    
    if not video_path.exists():
        logger.error(f"视频文件不存在: {video_path}")
        return False
    
    # 检查是否已提取
    if output_audio_path.exists():
        logger.info(f"音频已存在: {output_audio_path.name}")
        return True
    
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', '1',
        '-y',
        str(output_audio_path)
    ]
    
    try:
        logger.info(f"提取音频: {video_path.name} -> {output_audio_path.name}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0 and output_audio_path.exists():
            size_mb = output_audio_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ 音频提取成功 ({size_mb:.1f} MB)")
            return True
        else:
            logger.error(f"❌ 音频提取失败")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ 音频提取超时")
        return False
    except Exception as e:
        logger.error(f"❌ 音频提取异常: {e}")
        return False


def batch_download_and_extract(
    sampling_plan,
    video_output_dir="data/raw/audio/elderly_videos",
    audio_output_dir="data/raw/audio/elderly_audios",
    cookies_file=None,
    max_downloads=None,
    start_from=0,
    delay_between_downloads=2
):
    """
    批量下载视频并提取音频
    
    Args:
        sampling_plan: 采样计划
        video_output_dir: 视频输出目录
        audio_output_dir: 音频输出目录
        cookies_file: Cookies文件
        max_downloads: 最大下载数量（None表示全部）
        start_from: 从第几个开始（用于断点续传）
        delay_between_downloads: 下载间隔（秒）
    """
    
    download_plan = sampling_plan['download_plan']
    
    # 限制下载数量
    if max_downloads:
        download_plan = download_plan[start_from:start_from + max_downloads]
    else:
        download_plan = download_plan[start_from:]
    
    logger.info(f"\n开始批量下载 {len(download_plan)} 个视频...")
    logger.info(f"视频保存到: {video_output_dir}")
    logger.info(f"音频保存到: {audio_output_dir}")
    
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    for i, video_info in enumerate(tqdm(download_plan, desc="下载进度")):
        logger.info(f"\n[{i+1}/{len(download_plan)}] 处理视频 {video_info['index']}")
        logger.info(f"  方言: {video_info['dialect']}")
        logger.info(f"  标题: {video_info['title'][:50]}")
        
        # 1. 下载视频
        video_path = download_single_video(
            video_info,
            video_output_dir,
            cookies_file
        )
        
        if not video_path:
            results['failed'].append({
                'index': video_info['index'],
                'reason': 'download_failed'
            })
            continue
        
        # 2. 提取音频
        idx = video_info['index']
        audio_filename = f"elderly_audio_{idx:04d}.wav"
        audio_path = Path(audio_output_dir) / audio_filename
        
        success = extract_audio_from_video(video_path, audio_path)
        
        if success:
            results['success'].append({
                'index': idx,
                'dialect': video_info['dialect'],
                'video_path': video_path,
                'audio_path': str(audio_path),
                'uploader': video_info['uploader'],
                'title': video_info['title']
            })
        else:
            results['failed'].append({
                'index': idx,
                'reason': 'audio_extraction_failed'
            })
        
        # 延迟，避免请求过快
        if i < len(download_plan) - 1:
            time.sleep(delay_between_downloads)
    
    # 保存结果
    results_path = "data/download_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("下载完成！")
    logger.info("="*80)
    logger.info(f"成功: {len(results['success'])} 个")
    logger.info(f"失败: {len(results['failed'])} 个")
    logger.info(f"结果已保存: {results_path}")
    
    # 按方言统计
    by_dialect = {}
    for item in results['success']:
        dialect = item['dialect']
        by_dialect[dialect] = by_dialect.get(dialect, 0) + 1
    
    logger.info(f"\n按方言统计:")
    for dialect, count in sorted(by_dialect.items()):
        logger.info(f"  {dialect}: {count} 个")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="选择性下载视频和提取音频")
    parser.add_argument('--plan', type=str,
                       default='data/sampling_plan.json',
                       help='采样计划JSON文件')
    parser.add_argument('--video_dir', type=str,
                       default='data/raw/audio/elderly_videos',
                       help='视频输出目录')
    parser.add_argument('--audio_dir', type=str,
                       default='data/raw/audio/elderly_audios',
                       help='音频输出目录')
    parser.add_argument('--cookies', type=str,
                       default=None,
                       help='Cookies文件路径')
    parser.add_argument('--max_downloads', type=int,
                       default=None,
                       help='最大下载数量（用于测试）')
    parser.add_argument('--start_from', type=int,
                       default=0,
                       help='从第几个开始（断点续传）')
    parser.add_argument('--delay', type=int,
                       default=2,
                       help='下载间隔（秒）')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("选择性视频下载")
    logger.info("="*80)
    
    # 加载采样计划
    sampling_plan = load_sampling_plan(args.plan)
    
    if not sampling_plan:
        return
    
    # 检查依赖
    try:
        subprocess.run(['you-get', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ 未安装 you-get，请安装: pip install you-get")
        return
    
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ 未安装 ffmpeg，请安装: apt-get install ffmpeg")
        return
    
    # 执行下载
    results = batch_download_and_extract(
        sampling_plan,
        video_output_dir=args.video_dir,
        audio_output_dir=args.audio_dir,
        cookies_file=args.cookies,
        max_downloads=args.max_downloads,
        start_from=args.start_from,
        delay_between_downloads=args.delay
    )
    
    logger.info("\n下一步:")
    logger.info("  1. 查看下载结果: cat data/download_results.json")
    logger.info("  2. 爬取字幕并标注: python scripts/whisper_data_collection/3_scrape_subtitles.py")


if __name__ == "__main__":
    main()

