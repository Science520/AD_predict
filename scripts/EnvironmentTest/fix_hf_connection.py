#!/usr/bin/env python3
"""
修复HuggingFace连接问题的脚本
"""
import os
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_hf_environment():
    """配置HuggingFace环境"""
    
    print("🔧 配置HuggingFace环境...")
    
    # 方案1: 使用镜像源
    mirror_configs = [
        ("HF_ENDPOINT", "https://hf-mirror.com"),
        ("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface")),
        ("HF_HUB_DISABLE_TELEMETRY", "1"),  # 禁用遥测减少网络请求
        ("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # 禁用进度条
    ]
    
    for key, value in mirror_configs:
        os.environ[key] = value
        print(f"✅ 设置环境变量: {key}={value}")
    
    # 方案2: 配置超时和重试参数
    timeout_configs = [
        ("REQUESTS_TIMEOUT", "30"),
        ("HF_HUB_DOWNLOAD_TIMEOUT", "300"),
        ("CURL_CONNECT_TIMEOUT", "10")
    ]
    
    for key, value in timeout_configs:
        os.environ[key] = value
        print(f"⏱️ 设置超时参数: {key}={value}")

def test_connection():
    """测试连接状况"""
    
    print("\n🔍 测试网络连接...")
    
    # 测试基本连接
    test_urls = [
        "https://www.baidu.com",  # 国内网站
        "https://hf-mirror.com",  # HF镜像
        "https://huggingface.co"  # 原始HF
    ]
    
    for url in test_urls:
        try:
            import requests
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {url} - 连接成功")
            else:
                print(f"⚠️ {url} - 状态码: {response.status_code}")
        except Exception as e:
            print(f"❌ {url} - 连接失败: {e}")

def alternative_download_seniortalk():
    """替代下载方案"""
    
    print("\n📦 替代下载方案...")
    
    # 方案1: 使用git clone
    print("方案1: 使用git克隆仓库")
    git_commands = [
        "git lfs install",
        "git clone https://huggingface.co/datasets/BAAI/SeniorTalk"
    ]
    
    for cmd in git_commands:
        print(f"  $ {cmd}")
    
    # 方案2: 使用wget直接下载
    print("\n方案2: 使用wget下载特定文件")
    wget_examples = [
        "wget https://huggingface.co/datasets/BAAI/SeniorTalk/raw/main/README.md",
        "wget https://huggingface.co/datasets/BAAI/SeniorTalk/resolve/main/data/train-00000-of-00001.parquet"
    ]
    
    for cmd in wget_examples:
        print(f"  $ {cmd}")
    
    # 方案3: 使用Python requests分块下载
    print("\n方案3: 编程方式分块下载")
    print("  详见下面的代码示例")

def download_with_requests():
    """使用requests分块下载"""
    
    download_code = '''
import requests
import os
from pathlib import Path

def download_file_chunked(url, local_filename, chunk_size=8192):
    """分块下载文件"""
    
    print(f"开始下载: {url}")
    
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            
            # 创建目录
            Path(local_filename).parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        
        print(f"下载完成: {local_filename}")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False

# 使用示例
# download_file_chunked(
#     "https://hf-mirror.com/datasets/BAAI/SeniorTalk/raw/main/README.md",
#     "data/seniortalk/README.md"
# )
'''
    
    # 保存代码到文件
    code_file = "download_chunked.py"
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(download_code)
    
    print(f"✅ 分块下载代码已保存到: {code_file}")

def main():
    print("🚀 HuggingFace连接问题修复工具")
    print("="*50)
    
    # 1. 配置环境
    setup_hf_environment()
    
    # 2. 测试连接
    test_connection()
    
    # 3. 提供替代方案
    alternative_download_seniortalk()
    
    # 4. 生成下载工具
    download_with_requests()
    
    print("\n" + "="*50)
    print("🎯 推荐操作步骤:")
    print("1. 首先尝试使用镜像源重新下载")
    print("2. 如果仍然失败，使用git clone方式")
    print("3. 最后考虑使用分块下载代码")
    print("4. 联系管理员检查网络策略")

if __name__ == "__main__":
    main() 