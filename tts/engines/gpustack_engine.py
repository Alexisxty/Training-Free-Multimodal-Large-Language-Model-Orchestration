import aiohttp
import asyncio
import os
import json
from typing import Optional, Callable

from ..core.base_engine import BaseTTSEngine


class GPUStackTTSEngine(BaseTTSEngine):
    """GPUStack TTS引擎实现"""
    
    def __init__(self, api_key: str, api_base: str = "http://10.255.0.150:82", 
                 model: str = "cosyvoice-300m-instruct", voice: str = "Chinese Female"):
        """
        初始化GPUStack TTS引擎
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
            voice: 语音角色
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
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
    
    def set_model(self, model: str) -> None:
        """设置模型名称"""
        self.model = model
    
    def get_engine_id(self) -> str:
        """获取引擎ID"""
        return "gpus"
    
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
                    self.log("[GPUStack] 文本为空，跳过合成", debug=True)
                return False
            
            url = f"{self.api_base}/v1-openai/audio/speech"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "multipart/form-data"
            }
            
            data = aiohttp.FormData()
            payload = {
                "model": self.model,
                "voice": self.voice,
                "response_format": "mp3",
                "input": text
            }
            data.add_field('json', json.dumps(payload), content_type='application/json')
            
            if not silent:
                self.log(f"[GPUStack] 开始合成: {text[:20]}...", debug=True)
                
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as response:
                    if response.status == 200:
                        # 保存音频文件
                        with open(output_file, 'wb') as f:
                            f.write(await response.read())
                            
                        # 检查文件是否生成成功
                        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                            if not silent:
                                self.log(f"[GPUStack] 合成成功: {output_file}", debug=True)
                            return True
                        else:
                            if not silent:
                                self.log(f"[GPUStack] 合成失败: 文件为空", debug=True)
                            return False
                    else:
                        if not silent:
                            self.log(f"[GPUStack] 请求失败: {response.status} - {await response.text()}", debug=True)
                        return False
                        
        except Exception as e:
            if not silent:
                self.log(f"[GPUStack] 合成出错: {str(e)}", debug=True)
            return False 