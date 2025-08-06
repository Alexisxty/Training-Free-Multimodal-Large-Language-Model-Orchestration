from abc import ABC, abstractmethod
import asyncio
from typing import Optional


class BaseTTSEngine(ABC):
    """TTS引擎的基类，定义所有TTS引擎必须实现的接口"""
    
    @abstractmethod
    async def synthesize(self, text: str, output_file: str, silent: bool = False) -> bool:
        """
        将文本合成为语音并保存到文件
        
        Args:
            text: 要合成的文本
            output_file: 输出文件路径
            silent: 是否静默模式（不输出日志）
            
        Returns:
            bool: 合成是否成功
        """
        pass
    
    @abstractmethod
    def set_voice(self, voice: str) -> None:
        """
        设置语音角色
        
        Args:
            voice: 语音角色标识
        """
        pass
    
    @abstractmethod
    def get_engine_id(self) -> str:
        """
        获取引擎ID（用于缓存等）
        
        Returns:
            str: 引擎唯一标识
        """
        pass
        
    def log(self, message: str, debug: bool = False) -> None:
        """
        记录日志（默认实现）
        
        Args:
            message: 日志消息
            debug: 是否为调试信息
        """
        print(message)