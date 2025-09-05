
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
