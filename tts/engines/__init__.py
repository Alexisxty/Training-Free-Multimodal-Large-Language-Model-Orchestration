from enum import Enum
from .edge_engine import EdgeTTSEngine
from .cosy_engine import CosyVoiceTTSEngine
from .gpustack_engine import GPUStackTTSEngine


class TTSEngine(Enum):
    """TTS引擎类型枚举"""
    EDGE = "edge"
    COSYVOICE = "cosyvoice"
    GPUSTACK = "gpustack"


__all__ = ['TTSEngine', 'EdgeTTSEngine', 'CosyVoiceTTSEngine', 'GPUStackTTSEngine'] 