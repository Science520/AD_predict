import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

# List of Chinese provinces for extraction
PROVINCES = [
    "河北", "山西", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海", "台湾",
    "内蒙古", "广西", "西藏", "宁夏", "新疆", "北京", "天津", "上海", "重庆", "香港", "澳门"
]

def get_title_from_url(url):
    """
    Scrapes the video title from a given URL.
    """
    if not url or not url.startswith('http'):
        return ""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title')
        return title.string.strip() if title else ""
    except requests.exceptions.RequestException as e:
        print(f"Error fetching title for {url}: {e}")
        return ""

def extract_province_from_title(title):
    """
    Extracts the province from the video title.
    """
    if not isinstance(title, str):
        return ""
    for province in PROVINCES:
        if province in title:
            return province
    return ""

def process_excel_file(input_path, output_path):
    """
    Processes the video information Excel file to fill missing titles and extract provinces.
    """
    try:
        df = pd.read_excel(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # Ensure correct column names, feel free to adjust if they are different
    df.columns = ['up主', '视频名称', 'url']

    if '省份' not in df.columns:
        df['省份'] = ""

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        title = row['视频名称']
        url = row['url']

        # If title is missing, scrape it from the URL
        if pd.isna(title) or title.strip() == "":
            new_title = get_title_from_url(url)
            if new_title:
                df.at[index, '视频名称'] = new_title
                title = new_title

        # Extract province from the title
        province = extract_province_from_title(title)
        if province:
            df.at[index, '省份'] = province

    df.to_excel(output_path, index=False)
    print(f"Processing complete. Updated file saved to {output_path}")

if __name__ == "__main__":
    # Path to the input and output Excel files
    input_file = "/home/saisai/AD_predict/AD_predict/data/raw/audio/老人视频信息.xlsx"
    output_file = "/home/saisai/AD_predict/AD_predict/data/raw/audio/老人视频信息_updated.xlsx"
    process_excel_file(input_file, output_file)
