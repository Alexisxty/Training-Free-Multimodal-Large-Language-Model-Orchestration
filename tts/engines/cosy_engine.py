import aiohttp
import asyncio
import os
from typing import Optional, Callable

from ..core.base_engine import BaseTTSEngine


class CosyVoiceTTSEngine(BaseTTSEngine):
    """CosyVoice TTS引擎实现"""
    
    def __init__(self, api_key: str, api_base: str = "https://api.siliconflow.cn/v1/audio/speech", voice: str = "anna"):
        """
        初始化CosyVoice TTS引擎
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            voice: 语音角色
        """
        self.api_key = api_key
        self.api_base = api_base
        self.voice = voice
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
    
    def get_engine_id(self) -> str:
        """获取引擎ID"""
        return "cosy"
    
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
                    self.log("[CosyVoice] 文本为空，跳过合成", debug=True)
                return False
            
            # 构建请求数据
            payload = {
                "model": "FunAudioLLM/CosyVoice2-0.5B",
                "input": text,
                "voice": self.voice,
                "response_format": "mp3",
                "sample_rate": 32000,
                "stream": False,
                "speed": 1,
                "gain": 0
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            if not silent:
                self.log(f"[CosyVoice] 开始合成: {text[:20]}...", debug=True)
                
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_base, json=payload, headers=headers) as response:
                    if response.status == 200:
                        # 保存音频文件
                        with open(output_file, 'wb') as f:
                            f.write(await response.read())
                            
                        # 检查文件是否生成成功
                        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                            if not silent:
                                self.log(f"[CosyVoice] 合成成功: {output_file}", debug=True)
                            return True
                        else:
                            if not silent:
                                self.log(f"[CosyVoice] 合成失败: 文件为空", debug=True)
                            return False
                    else:
                        if not silent:
                            self.log(f"[CosyVoice] 请求失败: {response.status} - {await response.text()}", debug=True)
                        return False
                        
        except Exception as e:
            if not silent:
                self.log(f"[CosyVoice] 合成出错: {str(e)}", debug=True)
            return False 