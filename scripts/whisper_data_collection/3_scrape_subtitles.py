"""
字幕爬取和标注脚本
功能：
1. 从Bilibili视频爬取字幕（ASR生成或人工上传）
2. 如果没有字幕，使用Whisper生成转录文本
3. 将转录文本与方言标签关联并保存
"""

import os
import sys
import json
import re
import requests
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd
import whisper
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_bvid_from_url(url):
    """从Bilibili URL提取BV号"""
    
    # 匹配 BV 号的正则表达式
    patterns = [
        r'BV[\w]+',  # 直接匹配BV号
        r'/video/(BV[\w]+)',  # 从URL路径提取
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            bvid = match.group(0) if 'BV' in match.group(0) else match.group(1)
            return bvid
    
    return None


def get_bilibili_subtitle(bvid, cookies=None):
    """
    从Bilibili获取字幕
    
    Args:
        bvid: 视频BV号
        cookies: 登录cookies（可选，有些视频需要登录）
        
    Returns:
        str: 字幕文本，失败返回None
    """
    
    try:
        # 1. 获取视频信息（包含cid）
        video_info_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bilibili.com'
        }
        
        if cookies:
            headers['Cookie'] = cookies
        
        response = requests.get(video_info_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"获取视频信息失败: HTTP {response.status_code}")
            return None
        
        video_data = response.json()
        
        if video_data['code'] != 0:
            logger.warning(f"视频信息API返回错误: {video_data.get('message', '')}")
            return None
        
        cid = video_data['data']['cid']
        
        # 2. 获取字幕列表
        subtitle_url = f"https://api.bilibili.com/x/player/v2?cid={cid}&bvid={bvid}"
        
        response = requests.get(subtitle_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"获取字幕列表失败: HTTP {response.status_code}")
            return None
        
        subtitle_data = response.json()
        
        if subtitle_data['code'] != 0:
            logger.warning(f"字幕API返回错误: {subtitle_data.get('message', '')}")
            return None
        
        # 检查是否有字幕
        subtitle_info = subtitle_data.get('data', {}).get('subtitle', {})
        subtitles = subtitle_info.get('subtitles', [])
        
        if not subtitles:
            logger.info(f"视频 {bvid} 没有字幕")
            return None
        
        # 3. 下载字幕（选择第一个，通常是中文）
        subtitle_url = "https:" + subtitles[0]['subtitle_url']
        
        response = requests.get(subtitle_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"下载字幕失败: HTTP {response.status_code}")
            return None
        
        subtitle_json = response.json()
        
        # 4. 解析字幕
        subtitle_text = ""
        for item in subtitle_json.get('body', []):
            content = item.get('content', '').strip()
            if content:
                subtitle_text += content + " "
        
        return subtitle_text.strip()
        
    except requests.exceptions.Timeout:
        logger.error(f"请求超时")
        return None
    except Exception as e:
        logger.error(f"爬取字幕异常: {e}")
        return None


def transcribe_audio_with_whisper(audio_path, model_name='large-v3'):
    """
    使用Whisper对音频进行转录
    
    Args:
        audio_path: 音频文件路径
        model_name: Whisper模型名称
        
    Returns:
        str: 转录文本
    """
    
    logger.info(f"使用Whisper模型 {model_name} 转录音频...")
    
    try:
        # 加载模型（首次会下载）
        model = whisper.load_model(model_name)
        
        # 转录
        result = model.transcribe(
            audio_path,
            language='zh',
            verbose=False
        )
        
        return result['text'].strip()
        
    except Exception as e:
        logger.error(f"Whisper转录失败: {e}")
        return None


def process_video_transcripts(
    download_results_path="data/download_results.json",
    excel_path="/data/AD_predict/data/raw/audio/老人视频信息_final_complete_20251016_214400.xlsx",
    output_dir="data/raw/audio/result",
    use_whisper_fallback=True,
    whisper_model='base',  # 使用base模型速度更快
    cookies=None
):
    """
    处理所有下载的视频，提取字幕并标注
    
    Args:
        download_results_path: 下载结果JSON
        excel_path: Excel文件（包含URL）
        output_dir: 输出目录
        use_whisper_fallback: 如果没有字幕，是否使用Whisper
        whisper_model: Whisper模型名称
        cookies: Bilibili cookies
    """
    
    # 加载下载结果
    if not os.path.exists(download_results_path):
        logger.error(f"下载结果文件不存在: {download_results_path}")
        logger.error("请先运行: python scripts/whisper_data_collection/2_selective_download.py")
        return
    
    with open(download_results_path, 'r', encoding='utf-8') as f:
        download_results = json.load(f)
    
    success_items = download_results['success']
    
    if len(success_items) == 0:
        logger.error("没有成功下载的视频")
        return
    
    logger.info(f"处理 {len(success_items)} 个视频的字幕...")
    
    # 加载Excel获取URL
    df = pd.read_excel(excel_path)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'bilibili_subtitle': [],
        'whisper_transcribed': [],
        'failed': []
    }
    
    for item in tqdm(success_items, desc="处理字幕"):
        idx = item['index']
        dialect = item['dialect']
        audio_path = item['audio_path']
        
        logger.info(f"\n处理视频 {idx} ({dialect})")
        
        # 获取URL
        if idx <= len(df):
            url = df.iloc[idx-1]['url']
        else:
            logger.warning(f"索引 {idx} 超出Excel范围")
            results['failed'].append({'index': idx, 'reason': 'no_url'})
            continue
        
        # 输出文件路径
        output_txt = output_path / f"test{idx}.txt"
        
        # 如果已存在，跳过
        if output_txt.exists():
            logger.info(f"转录文本已存在: {output_txt.name}")
            results['bilibili_subtitle'].append({
                'index': idx,
                'dialect': dialect,
                'transcript_path': str(output_txt)
            })
            continue
        
        transcript = None
        source = None
        
        # 1. 尝试从Bilibili获取字幕
        bvid = extract_bvid_from_url(url)
        
        if bvid:
            logger.info(f"  从Bilibili爬取字幕 (BV号: {bvid})...")
            transcript = get_bilibili_subtitle(bvid, cookies)
            
            if transcript:
                source = 'bilibili'
                logger.info(f"  ✅ 成功获取Bilibili字幕")
        
        # 2. 如果没有字幕，使用Whisper
        if not transcript and use_whisper_fallback:
            logger.info(f"  使用Whisper转录...")
            
            if not os.path.exists(audio_path):
                logger.error(f"  音频文件不存在: {audio_path}")
                results['failed'].append({'index': idx, 'reason': 'audio_not_found'})
                continue
            
            transcript = transcribe_audio_with_whisper(audio_path, whisper_model)
            
            if transcript:
                source = 'whisper'
                logger.info(f"  ✅ Whisper转录成功")
        
        # 3. 保存转录文本
        if transcript:
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            logger.info(f"  保存到: {output_txt.name}")
            
            if source == 'bilibili':
                results['bilibili_subtitle'].append({
                    'index': idx,
                    'dialect': dialect,
                    'transcript_path': str(output_txt),
                    'length': len(transcript)
                })
            else:
                results['whisper_transcribed'].append({
                    'index': idx,
                    'dialect': dialect,
                    'transcript_path': str(output_txt),
                    'length': len(transcript)
                })
        else:
            logger.error(f"  ❌ 无法获取转录文本")
            results['failed'].append({'index': idx, 'reason': 'no_transcript'})
    
    # 保存处理结果
    results_path = "data/transcript_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("字幕处理完成！")
    logger.info("="*80)
    logger.info(f"Bilibili字幕: {len(results['bilibili_subtitle'])} 个")
    logger.info(f"Whisper转录: {len(results['whisper_transcribed'])} 个")
    logger.info(f"失败: {len(results['failed'])} 个")
    logger.info(f"结果已保存: {results_path}")
    
    # 按方言统计
    by_dialect = {}
    for item in results['bilibili_subtitle'] + results['whisper_transcribed']:
        dialect = item['dialect']
        by_dialect[dialect] = by_dialect.get(dialect, 0) + 1
    
    logger.info(f"\n按方言统计:")
    for dialect, count in sorted(by_dialect.items()):
        logger.info(f"  {dialect}: {count} 个")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="爬取字幕并标注")
    parser.add_argument('--download_results', type=str,
                       default='data/download_results.json',
                       help='下载结果JSON文件')
    parser.add_argument('--excel', type=str,
                       default='/data/AD_predict/data/raw/audio/老人视频信息_final_complete_20251016_214400.xlsx',
                       help='视频信息Excel文件')
    parser.add_argument('--output_dir', type=str,
                       default='data/raw/audio/result',
                       help='转录文本输出目录')
    parser.add_argument('--whisper_model', type=str,
                       default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper模型大小')
    parser.add_argument('--no_whisper_fallback', action='store_true',
                       help='不使用Whisper作为备选')
    parser.add_argument('--cookies', type=str,
                       default=None,
                       help='Bilibili cookies')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("字幕爬取和标注")
    logger.info("="*80)
    
    # 检查Whisper是否可用
    if not args.no_whisper_fallback:
        try:
            import whisper
            logger.info(f"✅ Whisper可用，将使用 {args.whisper_model} 模型作为备选")
        except ImportError:
            logger.warning("⚠️  未安装Whisper，请安装: pip install openai-whisper")
            logger.warning("将仅使用Bilibili字幕")
            args.no_whisper_fallback = True
    
    # 处理字幕
    results = process_video_transcripts(
        download_results_path=args.download_results,
        excel_path=args.excel,
        output_dir=args.output_dir,
        use_whisper_fallback=not args.no_whisper_fallback,
        whisper_model=args.whisper_model,
        cookies=args.cookies
    )
    
    logger.info("\n下一步:")
    logger.info("  1. 查看转录结果: cat data/transcript_results.json")
    logger.info("  2. 查看转录文本: ls data/raw/audio/result/")
    logger.info("  3. 运行数据预处理: python scripts/1_prepare_dataset.py")


if __name__ == "__main__":
    main()

