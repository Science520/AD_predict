import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import time
import random

# ==============================================================================
# 核心配置：方言关键词映射表 (中文关键词 -> 标准英文标签)
# 维护指南：
# 1. 英文标签 (Value) 尽量使用城市名拼音，粒度尽可能细。
# 2. 中文关键词 (Key) 包含该方言区下的具体地名（区、县、市）。
# 3. 匹配逻辑是：只要标题包含 Key，就打上 Value 的标签。
# ==============================================================================
DIALECT_MAPPING = {
    # ==============================================================================
    # 方言映射表 - 按方言区分类，以母语为准，使用较大一级的方言分类
    # 优先级：具体城市 > 省份 > 方言区
    # ==============================================================================
    
    # --- 北京官话区 (北京、天津、河北) ---
    "北京": "beijing_mandarin",
    "天津": "beijing_mandarin", 
    "河北": "beijing_mandarin",
    "石家庄": "beijing_mandarin",
    "唐山": "beijing_mandarin",
    "秦皇岛": "beijing_mandarin",
    "邯郸": "beijing_mandarin",
    "邢台": "beijing_mandarin",
    "保定": "beijing_mandarin",
    "张家口": "beijing_mandarin",
    "承德": "beijing_mandarin",
    "沧州": "beijing_mandarin",
    "廊坊": "beijing_mandarin",
    "衡水": "beijing_mandarin",

    # --- 东北官话区 ---
    "东北": "dongbei_mandarin",
    "辽宁": "dongbei_mandarin",
    "吉林": "dongbei_mandarin", 
    "黑龙江": "dongbei_mandarin",
    "沈阳": "dongbei_mandarin",
    "大连": "dongbei_mandarin",
    "哈尔滨": "dongbei_mandarin",
    "长春": "dongbei_mandarin",
    "鞍山": "dongbei_mandarin",
    "抚顺": "dongbei_mandarin",
    "本溪": "dongbei_mandarin",
    "丹东": "dongbei_mandarin",
    "锦州": "dongbei_mandarin",
    "营口": "dongbei_mandarin",
    "阜新": "dongbei_mandarin",
    "辽阳": "dongbei_mandarin",
    "盘锦": "dongbei_mandarin",
    "铁岭": "dongbei_mandarin",
    "朝阳": "dongbei_mandarin",
    "葫芦岛": "dongbei_mandarin",
    "齐齐哈尔": "dongbei_mandarin",
    "鸡西": "dongbei_mandarin",
    "鹤岗": "dongbei_mandarin",
    "双鸭山": "dongbei_mandarin",
    "大庆": "dongbei_mandarin",
    "伊春": "dongbei_mandarin",
    "佳木斯": "dongbei_mandarin",
    "七台河": "dongbei_mandarin",
    "牡丹江": "dongbei_mandarin",
    "黑河": "dongbei_mandarin",
    "绥化": "dongbei_mandarin",
    "四平": "dongbei_mandarin",
    "辽源": "dongbei_mandarin",
    "通化": "dongbei_mandarin",
    "白山": "dongbei_mandarin",
    "松原": "dongbei_mandarin",
    "白城": "dongbei_mandarin",
    "延边": "dongbei_mandarin",

    # --- 中原官话区 (河南、山东、山西、陕西) ---
    "河南": "zhongyuan_mandarin",
    "山东": "zhongyuan_mandarin", 
    "山西": "zhongyuan_mandarin",
    "陕西": "zhongyuan_mandarin",
    "郑州": "zhongyuan_mandarin",
    "开封": "zhongyuan_mandarin",
    "洛阳": "zhongyuan_mandarin",
    "平顶山": "zhongyuan_mandarin",
    "安阳": "zhongyuan_mandarin",
    "鹤壁": "zhongyuan_mandarin",
    "新乡": "zhongyuan_mandarin",
    "焦作": "zhongyuan_mandarin",
    "濮阳": "zhongyuan_mandarin",
    "许昌": "zhongyuan_mandarin",
    "漯河": "zhongyuan_mandarin",
    "三门峡": "zhongyuan_mandarin",
    "南阳": "zhongyuan_mandarin",
    "商丘": "zhongyuan_mandarin",
    "信阳": "zhongyuan_mandarin",
    "周口": "zhongyuan_mandarin",
    "驻马店": "zhongyuan_mandarin",
    "济源": "zhongyuan_mandarin",
    "济南": "zhongyuan_mandarin",
    "青岛": "zhongyuan_mandarin",
    "淄博": "zhongyuan_mandarin",
    "枣庄": "zhongyuan_mandarin",
    "东营": "zhongyuan_mandarin",
    "烟台": "zhongyuan_mandarin",
    "潍坊": "zhongyuan_mandarin",
    "济宁": "zhongyuan_mandarin",
    "泰安": "zhongyuan_mandarin",
    "威海": "zhongyuan_mandarin",
    "日照": "zhongyuan_mandarin",
    "临沂": "zhongyuan_mandarin",
    "德州": "zhongyuan_mandarin",
    "聊城": "zhongyuan_mandarin",
    "滨州": "zhongyuan_mandarin",
    "菏泽": "zhongyuan_mandarin",
    "太原": "zhongyuan_mandarin",
    "大同": "zhongyuan_mandarin",
    "阳泉": "zhongyuan_mandarin",
    "长治": "zhongyuan_mandarin",
    "晋城": "zhongyuan_mandarin",
    "朔州": "zhongyuan_mandarin",
    "晋中": "zhongyuan_mandarin",
    "运城": "zhongyuan_mandarin",
    "忻州": "zhongyuan_mandarin",
    "临汾": "zhongyuan_mandarin",
    "吕梁": "zhongyuan_mandarin",
    "西安": "zhongyuan_mandarin",
    "铜川": "zhongyuan_mandarin",
    "宝鸡": "zhongyuan_mandarin",
    "咸阳": "zhongyuan_mandarin",
    "渭南": "zhongyuan_mandarin",
    "延安": "zhongyuan_mandarin",
    "汉中": "zhongyuan_mandarin",
    "榆林": "zhongyuan_mandarin",
    "安康": "zhongyuan_mandarin",
    "商洛": "zhongyuan_mandarin",

    # --- 江淮官话区 (安徽、江苏北部) ---
    "安徽": "jianghuai_mandarin",
    "江苏": "jianghuai_mandarin",
    "合肥": "jianghuai_mandarin",
    "芜湖": "jianghuai_mandarin",
    "蚌埠": "jianghuai_mandarin",
    "淮南": "jianghuai_mandarin",
    "马鞍山": "jianghuai_mandarin",
    "淮北": "jianghuai_mandarin",
    "铜陵": "jianghuai_mandarin",
    "安庆": "jianghuai_mandarin",
    "黄山": "jianghuai_mandarin",
    "滁州": "jianghuai_mandarin",
    "阜阳": "jianghuai_mandarin",
    "宿州": "jianghuai_mandarin",
    "六安": "jianghuai_mandarin",
    "亳州": "jianghuai_mandarin",
    "池州": "jianghuai_mandarin",
    "宣城": "jianghuai_mandarin",
    "南京": "jianghuai_mandarin",
    "无锡": "jianghuai_mandarin",
    "徐州": "jianghuai_mandarin",
    "常州": "jianghuai_mandarin",
    "苏州": "jianghuai_mandarin",
    "南通": "jianghuai_mandarin",
    "连云港": "jianghuai_mandarin",
    "淮安": "jianghuai_mandarin",
    "盐城": "jianghuai_mandarin",
    "扬州": "jianghuai_mandarin",
    "镇江": "jianghuai_mandarin",
    "泰州": "jianghuai_mandarin",
    "宿迁": "jianghuai_mandarin",

    # --- 西南官话区 (四川、重庆、贵州、云南、湖北、湖南) ---
    "四川": "xinan_mandarin",
    "重庆": "xinan_mandarin",
    "贵州": "xinan_mandarin", 
    "云南": "xinan_mandarin",
    "湖北": "xinan_mandarin",
    "湖南": "xinan_mandarin",
    "成都": "xinan_mandarin",
    "自贡": "xinan_mandarin",
    "攀枝花": "xinan_mandarin",
    "泸州": "xinan_mandarin",
    "德阳": "xinan_mandarin",
    "绵阳": "xinan_mandarin",
    "广元": "xinan_mandarin",
    "遂宁": "xinan_mandarin",
    "内江": "xinan_mandarin",
    "乐山": "xinan_mandarin",
    "南充": "xinan_mandarin",
    "眉山": "xinan_mandarin",
    "宜宾": "xinan_mandarin",
    "广安": "xinan_mandarin",
    "达州": "xinan_mandarin",
    "雅安": "xinan_mandarin",
    "巴中": "xinan_mandarin",
    "资阳": "xinan_mandarin",
    "贵阳": "xinan_mandarin",
    "六盘水": "xinan_mandarin",
    "遵义": "xinan_mandarin",
    "安顺": "xinan_mandarin",
    "毕节": "xinan_mandarin",
    "铜仁": "xinan_mandarin",
    "昆明": "xinan_mandarin",
    "曲靖": "xinan_mandarin",
    "玉溪": "xinan_mandarin",
    "保山": "xinan_mandarin",
    "昭通": "xinan_mandarin",
    "丽江": "xinan_mandarin",
    "普洱": "xinan_mandarin",
    "临沧": "xinan_mandarin",
    "武汉": "xinan_mandarin",
    "黄石": "xinan_mandarin",
    "十堰": "xinan_mandarin",
    "宜昌": "xinan_mandarin",
    "襄阳": "xinan_mandarin",
    "鄂州": "xinan_mandarin",
    "荆门": "xinan_mandarin",
    "孝感": "xinan_mandarin",
    "荆州": "xinan_mandarin",
    "黄冈": "xinan_mandarin",
    "咸宁": "xinan_mandarin",
    "随州": "xinan_mandarin",
    "长沙": "xinan_mandarin",
    "株洲": "xinan_mandarin",
    "湘潭": "xinan_mandarin",
    "衡阳": "xinan_mandarin",
    "邵阳": "xinan_mandarin",
    "岳阳": "xinan_mandarin",
    "常德": "xinan_mandarin",
    "张家界": "xinan_mandarin",
    "益阳": "xinan_mandarin",
    "郴州": "xinan_mandarin",
    "永州": "xinan_mandarin",
    "怀化": "xinan_mandarin",
    "娄底": "xinan_mandarin",
    "湘西": "xinan_mandarin",

    # --- 兰银官话区 (甘肃、宁夏、青海、新疆) ---
    "甘肃": "lanyin_mandarin",
    "宁夏": "lanyin_mandarin",
    "青海": "lanyin_mandarin",
    "新疆": "lanyin_mandarin",
    "兰州": "lanyin_mandarin",
    "天水": "lanyin_mandarin",
    "嘉峪关": "lanyin_mandarin",
    "金昌": "lanyin_mandarin",
    "白银": "lanyin_mandarin",
    "武威": "lanyin_mandarin",
    "张掖": "lanyin_mandarin",
    "平凉": "lanyin_mandarin",
    "酒泉": "lanyin_mandarin",
    "庆阳": "lanyin_mandarin",
    "定西": "lanyin_mandarin",
    "陇南": "lanyin_mandarin",
    "临夏": "lanyin_mandarin",
    "甘南": "lanyin_mandarin",
    "银川": "lanyin_mandarin",
    "石嘴山": "lanyin_mandarin",
    "吴忠": "lanyin_mandarin",
    "固原": "lanyin_mandarin",
    "中卫": "lanyin_mandarin",
    "西宁": "lanyin_mandarin",
    "海东": "lanyin_mandarin",
    "乌鲁木齐": "lanyin_mandarin",
    "克拉玛依": "lanyin_mandarin",

    # --- 吴语区 (上海、浙江、江苏南部) ---
    "上海": "wu_dialect",
    "浙江": "wu_dialect",
    "杭州": "wu_dialect",
    "宁波": "wu_dialect",
    "温州": "wu_dialect",
    "嘉兴": "wu_dialect",
    "湖州": "wu_dialect",
    "绍兴": "wu_dialect",
    "金华": "wu_dialect",
    "衢州": "wu_dialect",
    "舟山": "wu_dialect",
    "台州": "wu_dialect",
    "丽水": "wu_dialect",
    "浦东": "wu_dialect",
    "金泽": "wu_dialect",
    "崇明": "wu_dialect",
    "余杭": "wu_dialect",
    "昆山": "wu_dialect",

    # --- 粤语区 (广东、广西、香港、澳门) ---
    "广东": "yue_dialect",
    "广西": "yue_dialect",
    "香港": "yue_dialect",
    "澳门": "yue_dialect",
    "广州": "yue_dialect",
    "深圳": "yue_dialect",
    "珠海": "yue_dialect",
    "汕头": "yue_dialect",
    "佛山": "yue_dialect",
    "韶关": "yue_dialect",
    "湛江": "yue_dialect",
    "肇庆": "yue_dialect",
    "江门": "yue_dialect",
    "茂名": "yue_dialect",
    "惠州": "yue_dialect",
    "梅州": "yue_dialect",
    "汕尾": "yue_dialect",
    "河源": "yue_dialect",
    "阳江": "yue_dialect",
    "清远": "yue_dialect",
    "东莞": "yue_dialect",
    "中山": "yue_dialect",
    "潮州": "yue_dialect",
    "揭阳": "yue_dialect",
    "云浮": "yue_dialect",
    "南宁": "yue_dialect",
    "柳州": "yue_dialect",
    "桂林": "yue_dialect",
    "梧州": "yue_dialect",
    "北海": "yue_dialect",
    "防城港": "yue_dialect",
    "钦州": "yue_dialect",
    "贵港": "yue_dialect",
    "玉林": "yue_dialect",
    "百色": "yue_dialect",
    "贺州": "yue_dialect",
    "河池": "yue_dialect",
    "来宾": "yue_dialect",
    "崇左": "yue_dialect",
    "粤语": "yue_dialect",

    # --- 闽语区 (福建、台湾) ---
    "福建": "min_dialect",
    "台湾": "min_dialect",
    "厦门": "min_dialect",
    "福州": "min_dialect",
    "莆田": "min_dialect",
    "三明": "min_dialect",
    "泉州": "min_dialect",
    "漳州": "min_dialect",
    "南平": "min_dialect",
    "龙岩": "min_dialect",
    "宁德": "min_dialect",
    "潮汕": "min_dialect",

    # --- 赣语区 (江西) ---
    "江西": "gan_dialect",
    "南昌": "gan_dialect",
    "景德镇": "gan_dialect",
    "萍乡": "gan_dialect",
    "九江": "gan_dialect",
    "新余": "gan_dialect",
    "鹰潭": "gan_dialect",
    "赣州": "gan_dialect",
    "吉安": "gan_dialect",
    "宜春": "gan_dialect",
    "抚州": "gan_dialect",
    "上饶": "gan_dialect",

    # --- 客家话区 ---
    "客家": "hakka_dialect",

    # --- 其他地区 ---
    "内蒙古": "other_dialect",
    "西藏": "other_dialect",
    "海南": "other_dialect",
}

# 为了防止短词错误匹配（例如有地名叫“上海岸”），
# 我们可以按关键词长度降序排列，优先匹配长词。
# 将字典的项转换为列表并排序
SORTED_KEYWORDS = sorted(DIALECT_MAPPING.keys(), key=len, reverse=True)


def get_title_from_url(url):
    """
    从URL抓取网页标题，并进行简单清洗。
    """
    if not url or not isinstance(url, str) or not url.startswith('http'):
        return ""
    try:
        # 增加一些随机User-Agent和延时，防止被反爬
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
        ]
        headers = {'User-Agent': random.choice(user_agents)}
        
        # 简单的请求延时
        time.sleep(random.uniform(0.5, 1.5))
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 处理编码问题，防止中文乱码
        response.encoding = response.apparent_encoding
        
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('title')
        
        if title_tag and title_tag.string:
            raw_title = title_tag.string.strip()
            # 清洗标题：去除常见的网站后缀，如 " - 哔哩哔哩" 或 " - YouTube"
            # 这有助于提高后续关键词匹配的准确性
            clean_title = re.split(r'[-|_]', raw_title)[0].strip()
            return clean_title
        return ""
    except Exception as e:
        print(f"Error fetching title for {url}: {e}")
        return ""

def extract_dialect_label(title):
    """
    根据映射表，从标题中提取英文方言标签。
    优先匹配表示出生地/母语的关键词。
    """
    if not isinstance(title, str) or not title:
        return None # 返回None以便在Excel中留空
    
    # 定义出生地关键词的优先级模式
    birthplace_patterns = [
        r'是(.+?)人',  # "是XX人"
        r'(.+?)县人',  # "XX县人" 
        r'(.+?)市人',  # "XX市人"
        r'(.+?)省人',  # "XX省人"
        r'(.+?)出生',  # "XX出生"
        r'(.+?)籍',    # "XX籍"
    ]
    
    import re
    
    # 首先尝试匹配出生地模式
    for pattern in birthplace_patterns:
        matches = re.findall(pattern, title)
        for match in matches:
            # 检查匹配到的地名是否在我们的映射表中
            for keyword in SORTED_KEYWORDS:
                if keyword in match and keyword in title:
                    return DIALECT_MAPPING[keyword]
    
    # 如果没有找到出生地模式，使用原来的逻辑
    for keyword in SORTED_KEYWORDS:
        if keyword in title:
            # 找到匹配，返回对应的英文标签
            return DIALECT_MAPPING[keyword]
            
    return None # 没有找到匹配，返回None由人工标注

def process_excel_file(input_path, output_path):
    """
    处理Excel文件：补全标题，提取方言标签。
    """
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_excel(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 确保列名正确 (根据您的描述)
    required_columns = ['up主', '视频名称', 'url']
    # 简单的列名映射，防止 Excel 中有细微差别
    column_mapping = {col: col for col in df.columns}
    for col in df.columns:
        if '视频' in col and '名' in col: column_mapping[col] = '视频名称'
        if 'url' in col.lower() or '链接' in col: column_mapping[col] = 'url'
    
    df.rename(columns=column_mapping, inplace=True)

    # 新增标签列名
    label_col_name = 'dialect_label'
    if label_col_name not in df.columns:
        df[label_col_name] = None # 初始化为空

    print("Starting processing. This may take a while depending on network requests...")
    
    # 使用 tqdm 显示进度
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        title = row.get('视频名称')
        url = row.get('url')
        current_label = row.get(label_col_name)

        # 1. 如果没有标题，尝试联网抓取
        title_fetched = False
        if pd.isna(title) or str(title).strip() == "":
            new_title = get_title_from_url(url)
            if new_title:
                df.at[index, '视频名称'] = new_title
                title = new_title
                title_fetched = True
        
        # 确保title是字符串
        title_str = str(title) if pd.notna(title) else ""

        # 2. 如果还没有标签（或者刚抓取了新标题），尝试提取标签
        if pd.isna(current_label) or str(current_label).strip() == "" or title_fetched:
            label = extract_dialect_label(title_str)
            if label:
                df.at[index, label_col_name] = label

    # 保存结果
    print(f"Saving updated data to {output_path}...")
    try:
        df.to_excel(output_path, index=False)
        print("Processing complete successfully.")
        
        # 打印一些统计信息
        labeled_count = df[label_col_name].notna().sum()
        total_count = df.shape[0]
        print(f"Statistics: {labeled_count}/{total_count} ({(labeled_count/total_count):.1%}) videos labeled automatically.")
        print(f"Please manually label the remaining {total_count - labeled_count} entries in '{output_path}'.")
        
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    print("脚本开始执行...")
    # Path to the input and output Excel files
    # 建议：输出文件名加上时间戳，防止覆盖原始数据
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    input_file = "/home/saisai/AD_predict/AD_predict/data/raw/audio/老人视频信息.xlsx"
    output_file = f"/home/saisai/AD_predict/AD_predict/data/raw/audio/老人视频信息_labeled_{timestamp}.xlsx"
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    process_excel_file(input_file, output_file)
    print("脚本执行完成")