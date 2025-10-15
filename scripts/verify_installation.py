#!/usr/bin/env python3
"""
验证安装和模块导入

快速检查所有依赖和模块是否正常工作
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_imports():
    """检查所有必要的导入"""
    
    print("="*60)
    print("验证依赖安装")
    print("="*60)
    
    checks = []
    
    # 1. 基础依赖
    print("\n1. 检查基础依赖...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        checks.append(("PyTorch", True))
    except ImportError as e:
        print(f"  ✗ PyTorch 导入失败: {e}")
        checks.append(("PyTorch", False))
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
        checks.append(("NumPy", True))
    except ImportError as e:
        print(f"  ✗ NumPy 导入失败: {e}")
        checks.append(("NumPy", False))
    
    try:
        import pandas as pd
        print(f"  ✓ Pandas {pd.__version__}")
        checks.append(("Pandas", True))
    except ImportError as e:
        print(f"  ✗ Pandas 导入失败: {e}")
        checks.append(("Pandas", False))
    
    # 2. 音频处理
    print("\n2. 检查音频处理库...")
    
    try:
        import librosa
        print(f"  ✓ Librosa {librosa.__version__}")
        checks.append(("Librosa", True))
    except ImportError as e:
        print(f"  ✗ Librosa 导入失败: {e}")
        checks.append(("Librosa", False))
    
    try:
        import whisper
        print(f"  ✓ Whisper")
        checks.append(("Whisper", True))
    except ImportError as e:
        print(f"  ✗ Whisper 导入失败: {e}")
        checks.append(("Whisper", False))
    
    # 3. 可视化
    print("\n3. 检查可视化库...")
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
        checks.append(("Matplotlib", True))
    except ImportError as e:
        print(f"  ✗ Matplotlib 导入失败: {e}")
        checks.append(("Matplotlib", False))
    
    try:
        import seaborn as sns
        print(f"  ✓ Seaborn {sns.__version__}")
        checks.append(("Seaborn", True))
    except ImportError as e:
        print(f"  ✗ Seaborn 导入失败: {e}")
        checks.append(("Seaborn", False))
    
    # 4. 项目模块
    print("\n4. 检查项目模块...")
    
    try:
        from src.models.pmm.gmlvq_stratifier import GMLVQStratifier
        print("  ✓ PMM GMLVQ分层器")
        checks.append(("PMM GMLVQ", True))
    except ImportError as e:
        print(f"  ✗ PMM GMLVQ导入失败: {e}")
        checks.append(("PMM GMLVQ", False))
    
    try:
        from src.models.pmm.feature_extractor import AcousticFeatureExtractor
        print("  ✓ PMM 特征提取器")
        checks.append(("PMM Feature Extractor", True))
    except ImportError as e:
        print(f"  ✗ PMM特征提取器导入失败: {e}")
        checks.append(("PMM Feature Extractor", False))
    
    try:
        from src.models.conformal.conformal_asr import ConformalASR
        print("  ✓ Conformal ASR")
        checks.append(("Conformal ASR", True))
    except ImportError as e:
        print(f"  ✗ Conformal ASR导入失败: {e}")
        checks.append(("Conformal ASR", False))
    
    try:
        from src.models.conformal.calibrator import ConformalCalibrator
        print("  ✓ Conformal 校准器")
        checks.append(("Conformal Calibrator", True))
    except ImportError as e:
        print(f"  ✗ Conformal校准器导入失败: {e}")
        checks.append(("Conformal Calibrator", False))
    
    try:
        from src.models.asr.conformal_enhanced_asr import ConformalEnhancedASR
        print("  ✓ 增强版 Conformal ASR")
        checks.append(("Enhanced ASR", True))
    except ImportError as e:
        print(f"  ✗ 增强版ASR导入失败: {e}")
        checks.append(("Enhanced ASR", False))
    
    try:
        from src.models.asr.chinese_asr import ChineseASR
        print("  ✓ 中文 ASR")
        checks.append(("Chinese ASR", True))
    except ImportError as e:
        print(f"  ✗ 中文ASR导入失败: {e}")
        checks.append(("Chinese ASR", False))
    
    # 5. 可选依赖
    print("\n5. 检查可选依赖...")
    
    try:
        import openpyxl
        print(f"  ✓ OpenPyXL {openpyxl.__version__}")
        checks.append(("OpenPyXL", True))
    except ImportError:
        print("  ⚠ OpenPyXL 未安装（Excel支持，可选）")
        checks.append(("OpenPyXL", None))
    
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            print("  ✓ FFmpeg")
            checks.append(("FFmpeg", True))
        else:
            print("  ✗ FFmpeg 未正确安装")
            checks.append(("FFmpeg", False))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ⚠ FFmpeg 未安装（视频处理，可选）")
        checks.append(("FFmpeg", None))
    
    # 汇总
    print("\n" + "="*60)
    print("检查汇总")
    print("="*60)
    
    success = sum(1 for _, status in checks if status is True)
    failed = sum(1 for _, status in checks if status is False)
    optional = sum(1 for _, status in checks if status is None)
    
    print(f"\n✓ 成功: {success}")
    print(f"✗ 失败: {failed}")
    print(f"⚠ 可选未安装: {optional}")
    
    if failed == 0:
        print("\n🎉 所有必要依赖已正确安装!")
        return True
    else:
        print(f"\n❌ 有 {failed} 个必要依赖未安装，请检查")
        print("\n安装命令:")
        print("  pip install -r requirements.txt")
        return False


def test_basic_functionality():
    """测试基本功能"""
    
    print("\n" + "="*60)
    print("测试基本功能")
    print("="*60)
    
    # 1. 测试GMLVQ
    print("\n1. 测试GMLVQ分层器...")
    try:
        from src.models.pmm.gmlvq_stratifier import GMLVQStratifier
        import numpy as np
        
        # 创建简单的测试数据
        features = np.random.randn(10, 18)
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        
        stratifier = GMLVQStratifier(n_features=18, n_strata=3)
        stratifier.fit(features, labels, n_epochs=5, verbose=False)
        
        results = stratifier.stratify(features[:3], ["test_1", "test_2", "test_3"])
        
        print(f"  ✓ GMLVQ正常工作，分层了 {len(results)} 个样本")
        
    except Exception as e:
        print(f"  ✗ GMLVQ测试失败: {e}")
        return False
    
    # 2. 测试特征提取器
    print("\n2. 测试声学特征提取器...")
    try:
        from src.models.pmm.feature_extractor import AcousticFeatureExtractor
        import numpy as np
        
        extractor = AcousticFeatureExtractor()
        feature_names = extractor.get_feature_names()
        
        print(f"  ✓ 特征提取器正常工作，支持 {len(feature_names)} 个特征")
        
    except Exception as e:
        print(f"  ✗ 特征提取器测试失败: {e}")
        return False
    
    # 3. 测试Conformal校准器
    print("\n3. 测试Conformal校准器...")
    try:
        from src.models.conformal.calibrator import ConformalCalibrator
        import numpy as np
        
        calibrator = ConformalCalibrator(coverage=0.95)
        scores = np.random.randn(100)
        threshold = calibrator.calibrate_standard(scores)
        
        print(f"  ✓ Conformal校准器正常工作，阈值: {threshold:.4f}")
        
    except Exception as e:
        print(f"  ✗ Conformal校准器测试失败: {e}")
        return False
    
    print("\n✓ 所有基本功能测试通过!")
    return True


def main():
    """主函数"""
    
    print("\n" + "="*80)
    print("AD预测系统 - 安装验证")
    print("PMM患者分层 & Conformal Inference ASR")
    print("="*80)
    
    # 检查导入
    imports_ok = check_imports()
    
    if not imports_ok:
        print("\n请先修复依赖问题")
        sys.exit(1)
    
    # 测试功能
    functionality_ok = test_basic_functionality()
    
    if functionality_ok:
        print("\n" + "="*80)
        print("✅ 验证完成！系统已准备就绪")
        print("="*80)
        print("\n下一步:")
        print("  1. 阅读快速开始指南: cat QUICKSTART.md")
        print("  2. 测试PMM分层: python scripts/test_pmm_stratification.py")
        print("  3. 运行完整评估: python scripts/run_complete_evaluation.py --help")
        print("\n")
    else:
        print("\n❌ 功能测试失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()

