#!/usr/bin/env python3
"""
下载老年人视频并提取音频

基于原始capture.py，适配当前环境
"""

import os
import sys
import pandas as pd
import argparse
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_videos_with_you_get(
    excel_path: str = "data/raw/audio/老人视频信息.xlsx",
    output_dir: str = "data/raw/audio/elderly_videos",
    start_idx: int = 1,
    max_videos: int = 10,
    cookies_file: str = None
):
    """
    使用you-get下载视频
    
    Args:
        excel_path: Excel文件路径
        output_dir: 输出目录
        start_idx: 起始索引
        max_videos: 最大下载数量
        cookies_file: cookies文件路径（可选）
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 读取Excel文件
    if not Path(excel_path).exists():
        logger.error(f"Excel文件不存在: {excel_path}")
        return []
    
    df = pd.read_excel(excel_path)
    logger.info(f"读取到 {len(df)} 条视频信息")
    
    # 提取视频URL
    video_urls = []
    for i, row in df.iterrows():
        if i >= start_idx - 1 and i < start_idx - 1 + max_videos:
            video_urls.append({
                'index': i + 1,
                'uploader': row.get('up主', 'unknown'),
                'url': row.get('url', '')
            })
    
    logger.info(f"准备下载 {len(video_urls)} 个视频")
    
    # 下载视频
    downloaded_files = []
    
    for video_info in video_urls:
        idx = video_info['index']
        url = video_info['url']
        uploader = video_info['uploader']
        
        if not url:
            logger.warning(f"跳过索引 {idx}：URL为空")
            continue
        
        file_name = f"elderly_video_{idx:04d}"
        
        logger.info(f"\n下载视频 {idx}/{len(video_urls)}: {uploader}")
        logger.info(f"  URL: {url}")
        
        # 构建you-get命令
        cmd = ['you-get', url, '-O', file_name, '-o', str(output_path)]
        
        if cookies_file and Path(cookies_file).exists():
            cmd.extend(['--cookies', cookies_file])
        
        try:
            # 执行下载
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            if result.returncode == 0:
                # 查找下载的文件
                possible_files = [
                    output_path / f"{file_name}.mp4",
                    output_path / f"{file_name}.flv",
                    output_path / f"{file_name}.mkv"
                ]
                
                downloaded_file = None
                for f in possible_files:
                    if f.exists():
                        downloaded_file = f
                        break
                
                if downloaded_file:
                    logger.info(f"✅ 下载成功: {downloaded_file.name}")
                    downloaded_files.append({
                        'index': idx,
                        'video_path': str(downloaded_file),
                        'uploader': uploader
                    })
                else:
                    logger.warning(f"⚠️ 下载命令成功但未找到文件")
            else:
                logger.error(f"❌ 下载失败: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ 下载超时")
        except Exception as e:
            logger.error(f"❌ 下载异常: {e}")
    
    logger.info(f"\n完成！成功下载 {len(downloaded_files)}/{len(video_urls)} 个视频")
    
    return downloaded_files


def extract_audio_from_video(
    video_path: str,
    output_audio_path: str,
    sample_rate: int = 16000
):
    """
    从视频提取音频
    
    使用ffmpeg提取音频并转换为16kHz WAV格式
    """
    
    video_path = Path(video_path)
    output_audio_path = Path(output_audio_path)
    
    if not video_path.exists():
        logger.error(f"视频文件不存在: {video_path}")
        return False
    
    # 创建输出目录
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用ffmpeg提取音频
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',  # 不处理视频
        '-acodec', 'pcm_s16le',  # PCM 16-bit
        '-ar', str(sample_rate),  # 采样率
        '-ac', '1',  # 单声道
        '-y',  # 覆盖输出文件
        str(output_audio_path)
    ]
    
    try:
        logger.info(f"提取音频: {video_path.name} -> {output_audio_path.name}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0 and output_audio_path.exists():
            # 获取文件大小
            size_mb = output_audio_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ 音频提取成功: {size_mb:.2f} MB")
            return True
        else:
            logger.error(f"❌ 音频提取失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ 音频提取超时")
        return False
    except Exception as e:
        logger.error(f"❌ 音频提取异常: {e}")
        return False


def process_videos_to_audio(
    video_infos: list,
    audio_output_dir: str = "data/raw/audio/elderly_audios"
):
    """
    批量处理视频，提取音频
    
    Args:
        video_infos: 视频信息列表
        audio_output_dir: 音频输出目录
    """
    
    audio_output_path = Path(audio_output_dir)
    audio_output_path.mkdir(parents=True, exist_ok=True)
    
    audio_files = []
    
    for video_info in video_infos:
        idx = video_info['index']
        video_path = video_info['video_path']
        
        # 生成音频文件名
        audio_filename = f"elderly_audio_{idx:04d}.wav"
        audio_path = audio_output_path / audio_filename
        
        # 提取音频
        success = extract_audio_from_video(video_path, audio_path)
        
        if success:
            audio_files.append({
                'index': idx,
                'audio_path': str(audio_path),
                'video_path': video_path,
                'uploader': video_info['uploader']
            })
    
    logger.info(f"\n音频提取完成！成功: {len(audio_files)}/{len(video_infos)}")
    
    return audio_files


def main():
    parser = argparse.ArgumentParser(description="下载老年人视频并提取音频")
    parser.add_argument('--excel', type=str,
                       default='data/raw/audio/老人视频信息.xlsx',
                       help='视频信息Excel文件')
    parser.add_argument('--video_dir', type=str,
                       default='data/raw/audio/elderly_videos',
                       help='视频输出目录')
    parser.add_argument('--audio_dir', type=str,
                       default='data/raw/audio/elderly_audios',
                       help='音频输出目录')
    parser.add_argument('--start', type=int, default=1,
                       help='起始索引')
    parser.add_argument('--max_videos', type=int, default=10,
                       help='最大下载数量')
    parser.add_argument('--cookies', type=str, default=None,
                       help='Cookies文件路径')
    parser.add_argument('--skip_download', action='store_true',
                       help='跳过下载，只提取已有视频的音频')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("老年人视频下载和音频提取")
    logger.info("="*60)
    
    if not args.skip_download:
        # 1. 下载视频
        logger.info("\n步骤1: 下载视频")
        video_infos = download_videos_with_you_get(
            excel_path=args.excel,
            output_dir=args.video_dir,
            start_idx=args.start,
            max_videos=args.max_videos,
            cookies_file=args.cookies
        )
    else:
        # 扫描已有视频
        logger.info("\n跳过下载，扫描已有视频...")
        video_dir = Path(args.video_dir)
        video_infos = []
        
        for video_file in sorted(video_dir.glob("*.mp4")):
            # 提取索引
            import re
            match = re.search(r'(\d+)', video_file.stem)
            if match:
                idx = int(match.group(1))
                video_infos.append({
                    'index': idx,
                    'video_path': str(video_file),
                    'uploader': 'unknown'
                })
        
        logger.info(f"找到 {len(video_infos)} 个视频文件")
    
    if len(video_infos) == 0:
        logger.error("没有可用的视频文件")
        return
    
    # 2. 提取音频
    logger.info("\n步骤2: 提取音频")
    audio_files = process_videos_to_audio(
        video_infos,
        audio_output_dir=args.audio_dir
    )
    
    # 3. 保存音频文件信息
    if audio_files:
        import json
        
        manifest_path = Path(args.audio_dir) / 'audio_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(audio_files, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n音频清单已保存: {manifest_path}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ 完成！")
    logger.info("="*60)


if __name__ == "__main__":
    main()

