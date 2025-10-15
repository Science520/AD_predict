import os
import pandas as pd
import argparse
import re
import time

parser = argparse.ArgumentParser()
parser.add_argument('--s', default=1, type=int)
args = parser.parse_args()
name_to_area = {
    '闲聊北京':'844,1080,0,1920',
    '北京大明':'844,1080,0,1920',
    '三哥视线':'844,1080,0,1920',
    '杰森纪实人文':'851,1080,0,1920',
    '西北俐均':'851,1080,0,1920',
    '上海王秋裤裤裤':'925,1080,0,1920',
    '杭州阿杜游记':'810,1080,0,1920',
}
video_urls = []
df = pd.read_excel("./老人视频信息.xlsx")
for i, row in df.iterrows():
    video_urls.append([row['up主'],row['url']])
cookies_file = "/home/kc/cases/cookies.sqlite"
target_dir = "/home/kc/cases/videos"
for i, [name,url] in enumerate(video_urls):
    # if i+1 < args.s or i+1 in downloaded:
    #     continue
    if i+1 < args.s:
        continue
    file_name = "test" + str(i+1)
    os.system(f"you-get {url} --cookies {cookies_file} -O {file_name} -o {target_dir}")
    file_path = os.path.join(target_dir, file_name) + '.mp4'
    print(file_path)
    os.system(f"python /home/kc/video-subtitle-extractor/backend/main_default_conf.py --video {file_path} --area {name_to_area[name]}")