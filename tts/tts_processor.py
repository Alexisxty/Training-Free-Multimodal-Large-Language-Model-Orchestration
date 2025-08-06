import edge_tts
import asyncio
import os
import pygame
import io
import time
import aiohttp
import json
from datetime import datetime
from enum import Enum
from utils.config import CONFIG, TEMP_FILE_CONFIG
from collections import deque
import uuid
import re
from typing import List, Dict, Tuple, Optional, Callable
import hashlib
import warnings

from .tts_manager import TTSManager
from .engines import TTSEngine

class TTSProcessor:
    """
    兼容层，将旧的TTSProcessor实现为对TTSManager的封装
    供已有代码使用，新代码请直接使用TTSManager
    """
    
    def __init__(self):
        """初始化TTSProcessor"""
        # 显示弃用警告
        warnings.warn(
            "TTSProcessor类已弃用，请使用TTSManager代替。此兼容层将在未来版本中移除。",
            DeprecationWarning,
            stacklevel=2
        )
        
        # 创建TTSManager实例
        self.manager = TTSManager()
        
        # 保持与旧API兼容的属性
        self.engine = TTSEngine.EDGE
        self.voice = self.manager.engines[TTSEngine.EDGE].voice if TTSEngine.EDGE in self.manager.engines else ""
        self.rate = "+0%"
        self.volume = "+0%"
        self.is_speaking = False
        self.should_stop = False
        self.temp_dir = self.manager.temp_dir
        self.cache_dir = self.manager.cache_dir
        self.use_cache = True
        self.log_callback = None
        self.asr_processor = None
        self.interrupt_callback = None
        
    def set_log_callback(self, callback: Callable) -> None:
        """设置日志回调函数"""
        self.log_callback = callback
        self.manager.set_log_callback(callback)
    
    def log(self, message: str, debug: bool = False) -> None:
        """输出日志"""
        self.manager.log(message, debug)

    def _split_text(self, text: str) -> List[str]:
        """分割文本为句子"""
        return self.manager.text_processor.split_text(text)
    
    def set_engine(self, engine: TTSEngine) -> None:
        """设置TTS引擎"""
        self.engine = engine
        self.manager.set_engine(engine)
        
    def set_cosy_voice(self, voice: str) -> None:
        """设置CosyVoice声音"""
        if TTSEngine.COSYVOICE in self.manager.engines:
            self.manager.engines[TTSEngine.COSYVOICE].set_voice(voice)
        
    def set_asr_processor(self, asr_processor) -> None:
        """设置ASR处理器引用"""
        self.asr_processor = asr_processor
        self.manager.set_asr_processor(asr_processor)
        
    def set_interrupt_callback(self, callback: Callable) -> None:
        """设置打断检测回调"""
        self.manager.set_interrupt_callback(callback)
        
    async def stop_speaking(self) -> None:
        """停止当前播放"""
        self.is_speaking = False
        self.should_stop = True
        await self.manager.audio_player.stop_speaking()
        self.log("[TTS] 停止播放", debug=True)
    
    async def process_text(self, text: str) -> None:
        """处理文本转语音"""
        self.is_speaking = True
        self.should_stop = False
        await self.manager.text_to_speech(text)
        self.is_speaking = self.manager.audio_player.is_speaking
        self.should_stop = self.manager.audio_player.should_stop

    async def text_to_speech(self, text: str) -> bool:
        """文本转语音"""
        print(f"[TTS_DEBUG] 接收到要播放的文本: '{text}'")
        if not text or not text.strip():
            print("[TTS_DEBUG] 文本为空，跳过播放")
            return False
            
        self.is_speaking = True
        self.should_stop = False
        print(f"[TTS_DEBUG] 将文本发送给TTSManager处理: '{text[:30]}...'")
        result = await self.manager.text_to_speech(text)
        print(f"[TTS_DEBUG] TTSManager处理结果: {result}")
        self.is_speaking = self.manager.audio_player.is_speaking
        self.should_stop = self.manager.audio_player.should_stop
        return result
    
    async def process_streaming_text(self, text_chunk: str) -> None:
        """处理流式文本片段"""
        await self.manager.process_streaming_text(text_chunk)
        self.is_speaking = self.manager.audio_player.is_speaking
        self.should_stop = self.manager.audio_player.should_stop

    async def play_audio(self, audio_path: str) -> None:
        """播放音频文件"""
        self.is_speaking = True
        self.should_stop = False
        await self.manager.play_audio(audio_path)
        self.is_speaking = self.manager.audio_player.is_speaking
        self.should_stop = self.manager.audio_player.should_stop
    
    def cleanup_all_temp_files(self) -> None:
        """清理所有临时文件"""
        self.manager.cleanup_all_temp_files()
    
    def __del__(self) -> None:
        """析构函数，清理资源"""
        # 通过manager的析构函数清理资源 