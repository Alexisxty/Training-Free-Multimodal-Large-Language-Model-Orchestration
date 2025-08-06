import edge_tts
import asyncio
import os
from typing import Optional, Callable

from ..core.base_engine import BaseTTSEngine


class EdgeTTSEngine(BaseTTSEngine):
    """Edge TTS引擎实现"""
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%", volume: str = "+0%"):
        """
        初始化Edge TTS引擎
        
        Args:
            voice: 语音角色
            rate: 语速
            volume: 音量
        """
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.log_callback: Optional[Callable] = None
    
    def set_log_callback(self, callback: Callable) -> None:
        """设置日志回调函数"""
        self.log_callback = callback
    
    def log(self, message: str, debug: bool = False) -> None:
        """输出日志"""
        if debug:
            print(message)  # 调试信息只在终端显示
        elif self.log_callback:
            self.log_callback(message)  # 重要信息在界面显示
        else:
            print(message)
    
    def set_voice(self, voice: str) -> None:
        """设置语音角色"""
        self.voice = voice
    
    def set_rate(self, rate: str) -> None:
        """设置语速"""
        self.rate = rate
    
    def set_volume(self, volume: str) -> None:
        """设置音量"""
        self.volume = volume
    
    def get_engine_id(self) -> str:
        """获取引擎ID"""
        return "edge"
    
    async def synthesize(self, text: str, output_file: str, silent: bool = False) -> bool:
        """
        将文本合成为语音并保存到文件
        
        Args:
            text: 要合成的文本
            output_file: 输出文件路径
            silent: 是否静默模式
            
        Returns:
            bool: 合成是否成功
        """
        try:
            if not text.strip():
                if not silent:
                    self.log("[EdgeTTS] 文本为空，跳过合成", debug=True)
                return False
            
            if not silent:
                self.log(f"[EdgeTTS] 开始合成: {text[:20]}...", debug=True)
                
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, volume=self.volume)
            await communicate.save(output_file)
            
            # 检查文件是否生成成功
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                if not silent:
                    self.log(f"[EdgeTTS] 合成成功: {output_file}", debug=True)
                return True
            else:
                if not silent:
                    self.log(f"[EdgeTTS] 合成失败: 文件为空", debug=True)
                return False
                
        except Exception as e:
            if not silent:
                self.log(f"[EdgeTTS] 合成出错: {str(e)}", debug=True)
            return False 