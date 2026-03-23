#!/usr/bin/env python3
"""
将Markdown文件转换为PDF
"""
import markdown
import weasyprint
from pathlib import Path
import sys

def markdown_to_pdf(md_file, output_file=None):
    """
    将Markdown文件转换为PDF
    
    Args:
        md_file: Markdown文件路径
        output_file: 输出PDF文件路径（可选）
    """
    # 读取Markdown文件
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # 转换为HTML
    html = markdown.markdown(md_content, extensions=['tables', 'toc'])
    
    # 添加CSS样式
    css_style = """
    <style>
    body {
        font-family: "Microsoft YaHei", "SimSun", Arial, sans-serif;
        line-height: 1.6;
        margin: 40px;
        color: #333;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    h1 {
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    h2 {
        border-bottom: 1px solid #bdc3c7;
        padding-bottom: 5px;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    blockquote {
        border-left: 4px solid #3498db;
        margin: 20px 0;
        padding: 10px 20px;
        background-color: #f8f9fa;
    }
    code {
        background-color: #f4f4f4;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: "Courier New", monospace;
    }
    pre {
        background-color: #f4f4f4;
        padding: 15px;
        border-radius: 5px;
        overflow-x: auto;
    }
    strong {
        color: #2c3e50;
    }
    @page {
        margin: 2cm;
        size: A4;
    }
    </style>
    """
    
    # 完整的HTML文档
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>月度工作报告</title>
        {css_style}
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    # 如果没有指定输出文件，使用默认名称
    if output_file is None:
        md_path = Path(md_file)
        output_file = md_path.with_suffix('.pdf')
    
    # 转换为PDF
    try:
        weasyprint.HTML(string=full_html).write_pdf(output_file)
        print(f"✅ PDF转换成功: {output_file}")
        return str(output_file)
    except Exception as e:
        print(f"❌ PDF转换失败: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python convert_to_pdf.py <markdown_file> [output_pdf]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(md_file).exists():
        print(f"❌ 文件不存在: {md_file}")
        sys.exit(1)
    
    result = markdown_to_pdf(md_file, output_file)
    if result:
        print(f"📄 PDF文件已生成: {result}")
    else:
        sys.exit(1)
