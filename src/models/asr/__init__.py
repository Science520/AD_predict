"""自动语音识别模块"""

from .transformer_asr import TransformerASR
from .pause_detector import PauseDetector

__all__ = [
    "TransformerASR",
    "PauseDetector"
] 