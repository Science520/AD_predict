import pandas as pd
from process_video_info_label import extract_dialect_label
from tqdm import tqdm

def reprocess_all_labels(input_file, output_file):
    """
    重新处理所有标签，使用新的方言分类
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

    print("Starting complete label reprocessing...")
    
    # 统计信息
    total_count = df.shape[0]
    print(f"Total entries: {total_count}")
    
    # 重新处理所有记录
    updated_count = 0
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Reprocessing labels"):
        title = row.get('视频名称')
        
        # 确保title是字符串
        title_str = str(title) if pd.notna(title) else ""
        
        # 重新提取标签
        label = extract_dialect_label(title_str)
        if label:
            df.at[index, 'dialect_label'] = label
            updated_count += 1
        else:
            # 如果没有匹配到标签，设为空
            df.at[index, 'dialect_label'] = None

    # 保存结果
    print(f"Saving updated data to {output_file}...")
    try:
        df.to_excel(output_file, index=False)
        print("Reprocessing complete successfully.")
        
        # 打印统计信息
        final_labeled_count = df['dialect_label'].notna().sum()
        final_unlabeled_count = df['dialect_label'].isna().sum()
        
        print(f"Statistics after reprocessing:")
        print(f"  - Total labeled: {final_labeled_count}/{total_count} ({(final_labeled_count/total_count):.1%})")
        print(f"  - Still unlabeled: {final_unlabeled_count}")
        
        # 显示标签分布
        print(f"\nLabel distribution:")
        label_counts = df['dialect_label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        
        if final_unlabeled_count > 0:
            print(f"\nPlease manually label the remaining {final_unlabeled_count} entries.")
            
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
    input_file = "/home/saisai/AD_predict/AD_predict/data/raw/audio/老人视频信息.xlsx"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/home/saisai/AD_predict/AD_predict/data/raw/audio/老人视频信息_final_{timestamp}.xlsx"
    
    print("开始重新处理所有标签...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    reprocess_all_labels(input_file, output_file)
    print("重新处理完成")
