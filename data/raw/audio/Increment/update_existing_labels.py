import pandas as pd
from process_video_info_label import extract_dialect_label, SORTED_KEYWORDS, DIALECT_MAPPING
from tqdm import tqdm

def update_existing_labels(input_file, output_file):
    """
    更新已标注的文件，只处理那些还没有标签的记录
    """
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 检查是否有dialect_label列
    if 'dialect_label' not in df.columns:
        print("Error: No 'dialect_label' column found in the file")
        return

    print("Starting label update for unlabeled entries...")
    
    # 统计信息
    total_count = df.shape[0]
    already_labeled = df['dialect_label'].notna().sum()
    unlabeled_count = df['dialect_label'].isna().sum()
    
    print(f"Total entries: {total_count}")
    print(f"Already labeled: {already_labeled}")
    print(f"Unlabeled entries: {unlabeled_count}")
    
    if unlabeled_count == 0:
        print("All entries are already labeled!")
        return
    
    # 只处理未标注的记录
    updated_count = 0
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Updating labels"):
        current_label = row.get('dialect_label')
        title = row.get('视频名称')
        
        # 只处理没有标签的记录
        if pd.isna(current_label) or str(current_label).strip() == "":
            # 确保title是字符串
            title_str = str(title) if pd.notna(title) else ""
            
            # 尝试提取标签
            label = extract_dialect_label(title_str)
            if label:
                df.at[index, 'dialect_label'] = label
                updated_count += 1

    # 保存结果
    print(f"Saving updated data to {output_file}...")
    try:
        df.to_excel(output_file, index=False)
        print("Update complete successfully.")
        
        # 打印统计信息
        final_labeled_count = df['dialect_label'].notna().sum()
        final_unlabeled_count = df['dialect_label'].isna().sum()
        
        print(f"Statistics after update:")
        print(f"  - Newly labeled: {updated_count}")
        print(f"  - Total labeled: {final_labeled_count}/{total_count} ({(final_labeled_count/total_count):.1%})")
        print(f"  - Still unlabeled: {final_unlabeled_count}")
        
        if final_unlabeled_count > 0:
            print(f"Please manually label the remaining {final_unlabeled_count} entries.")
            
            # 显示一些未标注的标题示例
            unlabeled = df[df['dialect_label'].isna()]
            print("\nUnlabeled titles (first 10):")
            for i, title in enumerate(unlabeled['视频名称'].head(10)):
                print(f"  {i+1}. {title}")
        
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    import datetime
    
    # 输入和输出文件路径
    input_file = "/home/saisai/AD_predict/AD_predict/data/raw/audio/老人视频信息_labeled_20251015_080950.xlsx"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/home/saisai/AD_predict/AD_predict/data/raw/audio/老人视频信息_updated_{timestamp}.xlsx"
    
    print("开始更新标签...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    update_existing_labels(input_file, output_file)
    print("更新完成")
