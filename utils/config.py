# config/config.py

import os
import json
import uuid
from dotenv import load_dotenv
from enum import Enum
from typing import Dict, Optional

# 加载.env文件
load_dotenv(override=True)

# 临时文件管理配置
TEMP_FILE_CONFIG = {
    'CLEAN_ON_EXIT': False,  # 退出时是否清理临时文件
    'AUTO_CLEAN_TTS': False,  # 自动清理TTS临时文件
    'TEMP_ROOT_DIR': 'temp',  # 临时文件根目录
    'AUDIO_DIR': 'audio',  # 音频文件目录
    'VIDEO_DIR': 'video',  # 视频文件目录
    'DEBUG_DIR': 'debug',  # 调试文件目录
    'TTS_DIR': 'tts',  # TTS音频文件目录
    'TTS_CACHE_DIR': 'tts/cache',  # TTS缓存目录
    'ASR_DIR': 'asr',  # ASR临时文件目录
    'DIALOGUE_HISTORY_DIR': 'dialogue_history',  # 对话历史目录
    'VIDEO_FRAMES_DIR': 'video_frames',  # 视频帧目录
}

# 确保临时文件根目录存在
os.makedirs(TEMP_FILE_CONFIG['TEMP_ROOT_DIR'], exist_ok=True)

# 基础配置
CONFIG = {
    # OpenAI配置
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
    'OPENAI_API_BASE': os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
    'OPENAI_API_MODEL': os.getenv('OPENAI_API_MODEL', 'gpt-4-turbo-preview'),
    'OPENAI_API_SYSTEM_PROMPT': os.getenv('OPENAI_API_SYSTEM_PROMPT', '你是一个有用的智能助手。'),
    'OPENAI_API_TEMPERATURE': float(os.getenv('OPENAI_API_TEMPERATURE', '0.7')),
    'OPENAI_API_MAX_TOKENS': int(os.getenv('OPENAI_API_MAX_TOKENS', '4000')),
    'OPENAI_API_TOP_P': float(os.getenv('OPENAI_API_TOP_P', '1.0')),
    'OPENAI_API_PRESENCE_PENALTY': float(os.getenv('OPENAI_API_PRESENCE_PENALTY', '0.0')),
    'OPENAI_API_FREQUENCY_PENALTY': float(os.getenv('OPENAI_API_FREQUENCY_PENALTY', '0.0')),
    
    # 模型配置
    'MAIN_DIALOGUE_MODEL': os.getenv('MAIN_DIALOGUE_MODEL', 'gpt-4-turbo-preview'),
    'MAIN_DIALOGUE_MODEL_BACKEND': os.getenv('MAIN_DIALOGUE_MODEL_BACKEND', 'third_party_api'),
    'VISION_MODEL': os.getenv('VISION_MODEL', 'gpt-4-vision-preview'),
    'VISION_MODEL_BACKEND': os.getenv('VISION_MODEL_BACKEND', 'third_party_api'),
    'QVQ_MODEL': os.getenv('QVQ_MODEL', 'qwen-vl-plus'),
    'QVQ_MODEL_BACKEND': os.getenv('QVQ_MODEL_BACKEND', 'third_party_api'),
    
    # LLM配置
    'API_TIMEOUT': float(os.getenv('API_TIMEOUT', '60.0')),
    'RETRY_DELAY': float(os.getenv('RETRY_DELAY', '1.0')),
    'MAX_RETRIES': int(os.getenv('MAX_RETRIES', '3')),
    'USE_STREAMING': os.getenv('USE_STREAMING', 'True').lower() == 'true',
    
    # API配置
    'ASR_API_URL': os.getenv('ASR_API_URL', 'http://127.0.0.1:8001/asr'),
    'ASR_API_KEY': os.getenv('ASR_API_KEY', ''),
    'ASR_MODEL': os.getenv('ASR_MODEL', 'large-v3'),
    'TTS_RETRY_COUNT': 3,
    'TTS_RETRY_DELAY': 1.0,  # 重试延迟（秒）
    
    # VAD配置
    'VAD_SENSITIVITY': int(os.getenv('VAD_SENSITIVITY', 2)),  # VAD灵敏度 (0-3)
    'VAD_ENERGY_THRESHOLD': float(os.getenv('VAD_ENERGY_THRESHOLD', 0.01)),
    'VAD_MIN_SPEECH_FRAMES': int(os.getenv('VAD_MIN_SPEECH_FRAMES', 10)),
    'VAD_MAX_SILENCE_FRAMES': int(os.getenv('VAD_MAX_SILENCE_FRAMES', 30)),
    'VAD_SILENCE_THRESHOLD': float(os.getenv('VAD_SILENCE_THRESHOLD', 0.3)),
    
    # ASR其他配置
    'ASR_GAIN': float(os.getenv('ASR_GAIN', 1.5)),
    'ASR_NORMALIZE_FACTOR': float(os.getenv('ASR_NORMALIZE_FACTOR', 0.9)),
    'ASR_CHANNELS': int(os.getenv('ASR_CHANNELS', 1)),
    'ASR_SAMPLE_RATE': int(os.getenv('ASR_SAMPLE_RATE', 16000)),
    
    # 回音抑制配置
    'ECHO_SUPPRESSION': {
        'ENABLED': os.getenv('ECHO_SUPPRESSION_ENABLED', 'true').lower() == 'true',
        'NORMAL_THRESHOLD': float(os.getenv('ECHO_NORMAL_THRESHOLD', '1.0')),
        'TTS_PLAYBACK_THRESHOLD': float(os.getenv('ECHO_TTS_THRESHOLD', '3.0')),
        'COOLDOWN_TIME': float(os.getenv('ECHO_COOLDOWN_TIME', '0.5')),
    },
    
    # 日志配置
    'LOG_FILE': os.getenv('LOG_FILE', 'dialogue.log'),
    'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
    'DEBUG': os.getenv('DEBUG', 'False').lower() == 'true',
    'DEBUG_AUDIO_DIR': os.path.join(TEMP_FILE_CONFIG['TEMP_ROOT_DIR'], TEMP_FILE_CONFIG['DEBUG_DIR']),
    
    # CosyVoice2配置
    'COSY_API_KEY': os.getenv('COSY_API_KEY', ''),
    'COSY_API_BASE': os.getenv('COSY_API_BASE', 'https://api.siliconflow.cn/v1/audio/speech'),
    'COSY_VOICE': os.getenv('COSY_VOICE', 'FunAudioLLM/CosyVoice2-0.5B:anna'),
    'COSY_SPEED': float(os.getenv('COSY_SPEED', '1.0')),
    'COSY_GAIN': int(os.getenv('COSY_GAIN', '0')),
    
    # TTS临时目录
    'TTS_TEMP_DIR': os.path.join(TEMP_FILE_CONFIG['TEMP_ROOT_DIR'], TEMP_FILE_CONFIG['TTS_DIR']),
    
    # TTS引擎
    'TTS_ENGINE': os.getenv('TTS_ENGINE', 'edge'),
}

# 音频处理配置
ASR_CONFIG = {
    'SAMPLE_RATE': 16000,
    'CHANNELS': 1,
    'CHUNK_SIZE': 480,  # 30ms at 16kHz
    'FORMAT': 'float32',
    'VAD_LEVEL': 2,  # VAD灵敏度 (0-3)
    'TEMP_DIR': os.path.join(TEMP_FILE_CONFIG['TEMP_ROOT_DIR'], TEMP_FILE_CONFIG['ASR_DIR']),
}

# TTS配置
TTS_CONFIG = {
    'ENGINE': os.getenv('TTS_ENGINE', 'edge'),  # 默认使用edge-tts
    # Edge TTS配置
    'TTS_VOICE': os.getenv('TTS_VOICE', 'zh-CN-XiaoxiaoNeural'),
    'TTS_RATE': os.getenv('TTS_RATE', '+0%'),
    'TTS_VOLUME': os.getenv('TTS_VOLUME', '+0%'),
    # CosyVoice2配置
    'COSY_API_KEY': os.getenv('COSY_API_KEY', ''),
    'COSY_API_BASE': os.getenv('COSY_API_BASE', 'https://api.siliconflow.cn/v1/audio/speech'),
    'COSY_VOICE': os.getenv('COSY_VOICE', 'FunAudioLLM/CosyVoice2-0.5B:anna'),
    'COSY_SPEED': float(os.getenv('COSY_SPEED', '1.0')),
    'COSY_GAIN': int(os.getenv('COSY_GAIN', '0')),
    # GPUStack TTS配置
    'GPUSTACK_API_KEY': os.getenv('GPUSTACK_API_KEY', ''),
    'GPUSTACK_API_BASE': os.getenv('GPUSTACK_API_BASE', ''),
    'GPUSTACK_TTS_MODEL': os.getenv('GPUSTACK_TTS_MODEL', 'cosyvoice-300m-instruct'),
    'GPUSTACK_TTS_VOICE': os.getenv('GPUSTACK_TTS_VOICE', 'Chinese Female'),
}

# 确保所有临时目录存在
for dir_name in ['AUDIO_DIR', 'VIDEO_DIR', 'DEBUG_DIR', 'TTS_DIR', 'ASR_DIR', 'DIALOGUE_HISTORY_DIR', 'VIDEO_FRAMES_DIR']:
    dir_path = os.path.join(TEMP_FILE_CONFIG['TEMP_ROOT_DIR'], TEMP_FILE_CONFIG[dir_name])
    os.makedirs(dir_path, exist_ok=True)

# 特殊处理TTS缓存目录
os.makedirs(os.path.join(TEMP_FILE_CONFIG['TEMP_ROOT_DIR'], TEMP_FILE_CONFIG['TTS_CACHE_DIR']), exist_ok=True)

print("配置加载完成")

class LLMBackendType(Enum):
    THIRD_PARTY_API = "third_party_api"
    OLLAMA = "ollama"
    VLLM = "vllm"

class LLMConfig:
    def __init__(
        self,
        backend_type: LLMBackendType,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_p: float = 1.0,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        self.backend_type = backend_type
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

# LLM配置
LLM_CONFIGS = {
    "main_dialogue": LLMConfig(
        backend_type=LLMBackendType(os.getenv('MAIN_LLM_BACKEND', 'third_party_api')),
        model_name=os.getenv('MAIN_LLM_MODEL'),
        api_base=os.getenv('MAIN_LLM_API_BASE'),
        api_key=os.getenv('MAIN_LLM_API_KEY'),
        temperature=float(os.getenv('MAIN_LLM_TEMPERATURE', '0.7')),
        max_tokens=int(os.getenv('MAIN_LLM_MAX_TOKENS', '500')),
        top_p=float(os.getenv('MAIN_LLM_TOP_P', '1.0')),
    ),
    "vision_dialogue": LLMConfig(
        backend_type=LLMBackendType(os.getenv('VISION_LLM_BACKEND', 'third_party_api')),
        model_name=os.getenv('VISION_LLM_MODEL'),
        api_base=os.getenv('VISION_LLM_API_BASE'),
        api_key=os.getenv('VISION_LLM_API_KEY'),
        temperature=float(os.getenv('VISION_LLM_TEMPERATURE', '0.7')),
        max_tokens=int(os.getenv('VISION_LLM_MAX_TOKENS', '500')),
        top_p=float(os.getenv('VISION_LLM_TOP_P', '1.0')),
    ),
    "qvq": LLMConfig(
        backend_type=LLMBackendType(os.getenv('QVQ_LLM_BACKEND', 'third_party_api')),
        model_name=os.getenv('QVQ_LLM_MODEL', 'Qwen/QVQ-72B-Preview'),
        api_base=os.getenv('QVQ_LLM_API_BASE'),
        api_key=os.getenv('QVQ_LLM_API_KEY'),
        temperature=0.8,
        max_tokens=2000,
        top_p=0.9,
    ),
} 