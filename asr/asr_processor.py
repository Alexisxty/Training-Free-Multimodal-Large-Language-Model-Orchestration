import asyncio
import aiohttp
import numpy as np
import wave
import os
import time
import queue
import io
import json
from datetime import datetime
from utils.config import CONFIG, ASR_CONFIG
import webrtcvad
import pyaudio
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor
import tempfile
import uuid

# 音频参数
RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = 'int16'

class ASRProcessor:
    def __init__(self, api_url=CONFIG['ASR_API_URL']):
        self.api_url = api_url
        self.api_key = CONFIG['ASR_API_KEY']
        self.model = CONFIG['ASR_MODEL']
        self.is_listening = False
        self.vad = webrtcvad.Vad(CONFIG['VAD_SENSITIVITY'])
        self.text_callback = None
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.loop = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.log_callback = None
        
        # 回音抑制相关属性
        self.is_tts_playing = False
        self.tts_end_time = 0
        self.echo_config = CONFIG['ECHO_SUPPRESSION']
        
        # 音频处理参数
        self.gain = CONFIG['ASR_GAIN']
        self.normalize_factor = CONFIG['ASR_NORMALIZE_FACTOR']
        
        # VAD参数
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.min_speech_frames = CONFIG['VAD_MIN_SPEECH_FRAMES']
        self.max_silence_frames = CONFIG['VAD_MAX_SILENCE_FRAMES']
        self.audio_buffer = []
        self.max_buffer_size = 48
        self.last_text = ""
        self.last_text_time = 0
        
        # 音频能量参数
        self.base_energy_threshold = CONFIG['VAD_ENERGY_THRESHOLD']
        self.dynamic_threshold = self.base_energy_threshold
        self.alpha = 0.05
        
        # 音频状态跟踪
        self.speech_start_time = None
        self.last_status_print_time = 0
        self.status_update_interval = 1.0
        self.total_audio_duration = 0
        self.accumulated_audio = []
        self.silence_threshold = CONFIG['VAD_SILENCE_THRESHOLD']
        self.text_interval = 0.5
        self.min_text_length = 2
        self.end_punctuation = set(['。', '！', '？', '!', '?', '.'])
        
        # 添加状态跟踪
        self.last_listening_log_time = 0
        self.listening_log_interval = 2.0  # 两秒内不重复输出
        
        self.log("[ASR] ASRProcessor initialized with echo suppression")
        
    def set_tts_state(self, is_playing: bool):
        """设置TTS播放状态"""
        self.is_tts_playing = is_playing
        if not is_playing:
            self.tts_end_time = time.time()
            
    def is_in_cooldown(self) -> bool:
        """检查是否在TTS结束后的冷却期"""
        if not self.echo_config['ENABLED']:
            return False
        return time.time() - self.tts_end_time < self.echo_config['COOLDOWN_TIME']
        
    def get_current_vad_threshold(self) -> float:
        """获取当前VAD阈值"""
        if not self.echo_config['ENABLED']:
            return self.echo_config['NORMAL_THRESHOLD']
            
        if self.is_tts_playing:
            return self.echo_config['TTS_PLAYBACK_THRESHOLD']
        elif self.is_in_cooldown():
            # 在冷却期内使用较高的阈值
            cooldown_progress = (time.time() - self.tts_end_time) / self.echo_config['COOLDOWN_TIME']
            return self.echo_config['TTS_PLAYBACK_THRESHOLD'] * (1 - cooldown_progress) + \
                   self.echo_config['NORMAL_THRESHOLD'] * cooldown_progress
        else:
            return self.echo_config['NORMAL_THRESHOLD']
            
    def is_speech(self, audio_data):
        """检测是否是语音"""
        energy = np.mean(np.abs(audio_data))
        
        # 使用更平滑的动态阈值调整
        if not self.is_speaking:
            self.dynamic_threshold = (1 - self.alpha) * self.dynamic_threshold + self.alpha * energy
            
        # 获取当前VAD阈值
        current_threshold = self.get_current_vad_threshold()
        threshold = max(self.base_energy_threshold, self.dynamic_threshold * current_threshold)
        
        # TTS播放时或冷却期内，提高检测标准
        if self.is_tts_playing or self.is_in_cooldown():
            # 计算信噪比
            signal_mean = np.mean(audio_data)
            signal_std = np.std(audio_data)
            snr = abs(signal_mean) / (signal_std + 1e-10)
            
            # 只有当信噪比足够高且能量超过阈值时才认为是语音
            return energy > threshold and snr > 2.0
            
        return energy > threshold

    def is_complete_sentence(self, text):
        """快速判断句子是否完整"""
        if not text:
            return False
        
        # 1. 长度检查
        if len(text) >= self.min_text_length:
            return True
            
        # 2. 标点符号检查
        if text[-1] in self.end_punctuation:
            return True
            
        # 3. 特殊模式检查（打断命令等）
        interrupt_words = ["停", "别说", "闭嘴", "等一下"]
        if any(word in text for word in interrupt_words):
            return True
            
        # 4. 语气词结尾
        if text[-1] in ['吗', '', '啊', '呀', '吧']:
            return True
            
        return False

    async def process_audio(self, audio_data):
        """处理音频数据"""
        try:
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
            
            float_data = audio_data.astype(np.float32) / 32768.0
            is_current_speech = self.is_speech(float_data)
            
            if is_current_speech:
                self.speech_frames += 1
                self.silence_frames = 0
            else:
                self.silence_frames += 1
                self.speech_frames = max(0, self.speech_frames - 1)
                
            if not self.is_speaking and self.speech_frames >= self.min_speech_frames:
                self.is_speaking = True
                self.audio_buffer = []
                
            if self.is_speaking:
                self.audio_buffer.append(audio_data)
                
                if (self.silence_frames >= self.max_silence_frames and len(self.audio_buffer) > self.min_speech_frames) or \
                   len(self.audio_buffer) >= self.max_buffer_size:
                    
                    if len(self.audio_buffer) > self.min_speech_frames:
                        # 合并音频数据
                        combined_audio = np.concatenate(self.audio_buffer)
                        
                        # 保存原始音频文件
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        raw_filename = os.path.join("debug_audio", f"audio_{timestamp}_001_raw.wav")
                        with wave.open(raw_filename, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(2)
                            wf.setframerate(RATE)
                            wf.writeframes(combined_audio.tobytes())
                        print(f"[DEBUG] 已保存音频: {raw_filename}")
                        
                        # 创建WAV式音频
                        wav_buffer = io.BytesIO()
                        with wave.open(wav_buffer, 'wb') as wav_file:
                            wav_file.setnchannels(CHANNELS)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(RATE)
                            wav_file.writeframes(combined_audio.tobytes())
                        wav_data = wav_buffer.getvalue()

                        # 发送到硅基流动ASR服务
                        async with aiohttp.ClientSession() as session:
                            form = aiohttp.FormData()
                            form.add_field('file', wav_data, filename='audio.wav', content_type='audio/wav')
                            form.add_field('model', self.model)

                            headers = {
                                "Authorization": f"Bearer {self.api_key}"
                            }

                            async with session.post(self.api_url, data=form, headers=headers) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    if result and 'text' in result:
                                        text = result['text']
                                        if text and len(text.strip()) > 0:
                                            current_time = time.time()
                                            
                                            # 更新或接本
                                            if current_time - self.last_text_time > self.text_interval:
                                                self.last_text = text
                                            else:
                                                self.last_text += text
                                            
                                            self.last_text_time = current_time
                                            
                                            # 检查句子完整性
                                            if self.is_complete_sentence(self.last_text):
                                                print(f"\n[ASR] 完整语句: {self.last_text}")
                                                if self.text_callback:
                                                    try:
                                                        await self.text_callback(f"✓ 完整句子: {self.last_text}")
                                                    except Exception as e:
                                                        print(f"[ASR] 回调错误: {str(e)}")
                                                        import traceback
                                                        print(f"[ASR] 错误堆栈: {traceback.format_exc()}")
                                                self.last_text = ""
                                            else:
                                                print(f"\n[ASR] 累积中: {self.last_text}")
                                                if self.text_callback:
                                                    try:
                                                        await self.text_callback(f"⋯ 累积中: {self.last_text}")
                                                    except Exception as e:
                                                        print(f"[ASR] 回调错误: {str(e)}")
                                                        import traceback
                                                        print(f"[ASR] 错误堆栈: {traceback.format_exc()}")
                    
                    self.is_speaking = False
                    self.speech_frames = 0
                    self.silence_frames = 0
                    self.audio_buffer = []

        except Exception as e:
            print(f"[ASR] 处理错误: {str(e)}")
            import traceback
            print(f"[ASR] 错误堆栈: {traceback.format_exc()}")

    async def start_listening(self):
        """开始监听音频输入"""
        self.is_listening = True
        self.loop = asyncio.get_running_loop()
        self.last_listening_log_time = 0  # 重置日志时间
        self.log("[ASR] 等待语音输入...")
        
        # 在执行器中启动音频流
        await self.loop.run_in_executor(self.executor, self._start_audio_stream)
        
    def _start_audio_stream(self):
        """在单独的线程中启动音频流"""
        # 查找最佳输入设备
        default_input_device_info = self.audio.get_default_input_device_info()
        input_device_index = default_input_device_info['index']
        
        # 尝试设置较高的输入音量
        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,  # 使用浮点数格式以便于处理
                channels=CONFIG['ASR_CHANNELS'],
                rate=CONFIG['ASR_SAMPLE_RATE'],
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=480,  # 30ms的音频块
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
        except Exception as e:
            print(f"[ASR] 音频流启动错误: {str(e)}")
            # 如果失败，尝试使用默认参数
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=CONFIG['ASR_CHANNELS'],
                rate=CONFIG['ASR_SAMPLE_RATE'],
                input=True,
                frames_per_buffer=480,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            
    async def stop_listening(self):
        """停止监听"""
        self.is_listening = False
        if self.stream:
            try:
                # 先停止音频流
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                # 关闭PyAudio实例
                self.audio.terminate()
            except Exception as e:
                print(f"[ASR] 停止音频流错误: {str(e)}")
        
        # 最后关闭executor
        if self.executor:
            try:
                self.executor.shutdown(wait=True)
            except Exception as e:
                print(f"[ASR] 关闭executor错误: {str(e)}")
        
    def _stop_audio_stream(self):
        """在单独的线程中停止音频流"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数，在非异步上下文中运行"""
        if self.is_listening and self.loop:
            # 使用 call_soon_threadsafe 在事件循环中安排协程
            self.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self.process_audio_chunk(in_data))
            )
        return (in_data, pyaudio.paContinue)
        
    async def process_audio_chunk(self, audio_data: bytes):
        """处理音频数据块"""
        try:
            # 将字节数据转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # 应用增益
            audio_array = audio_array * self.gain
            
            # 归一化以防止过载
            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                audio_array = audio_array * (self.normalize_factor / max_val)
            
            # 转换回16位整数格式用于VAD
            audio_int16 = (audio_array * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # VAD检测
            is_speech = self.vad.is_speech(audio_bytes, CONFIG['ASR_SAMPLE_RATE'])
            current_time = time.time()
            
            # 更新音频状态和显示
            if is_speech:
                if self.speech_start_time is None:
                    self.speech_start_time = current_time
                    # 检查是否需要输出"正在倾听"
                    if current_time - self.last_listening_log_time >= self.listening_log_interval:
                        self.log("[ASR] 正在倾听...")
                        self.last_listening_log_time = current_time
                
                self.total_audio_duration = current_time - self.speech_start_time
                self.last_speech_time = current_time
                self.accumulated_audio.append(audio_bytes)
            else:
                # 如果之前在说话，现在停止了
                if self.speech_start_time is not None:
                    if current_time - self.last_speech_time > self.silence_threshold:
                        duration = current_time - self.speech_start_time
                        
                        # 重要：在发送音频之前重置状态
                        temp_start_time = self.speech_start_time
                        temp_audio = self.accumulated_audio.copy()
                        
                        self.speech_start_time = None
                        self.total_audio_duration = 0
                        self.accumulated_audio = []
                        
                        # 发送累积的音频到ASR
                        if temp_audio:
                            combined_audio = b''.join(temp_audio)
                            await self.send_audio_to_asr(combined_audio)
                            
        except Exception as e:
            self.log(f"[ASR] 处理音频块错误: {str(e)}")
            
    async def send_audio_to_asr(self, audio_data: bytes):
        """发送音频数据到ASR服务器"""
        if not audio_data:
            return
            
        temp_file = None
        try:
            # 创建临时文件用于发送，使用配置的临时目录
            temp_dir = ASR_CONFIG['TEMP_DIR']
            os.makedirs(temp_dir, exist_ok=True)
            
            # 生成唯一的文件名
            filename = f"asr_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
            temp_file_path = os.path.join(temp_dir, filename)
            
            # 写入音频数据
            with wave.open(temp_file_path, 'wb') as wf:
                wf.setnchannels(CONFIG['ASR_CHANNELS'])
                wf.setsampwidth(2)
                wf.setframerate(CONFIG['ASR_SAMPLE_RATE'])
                wf.writeframes(audio_data)
            
            # 打开临时文件
            temp_file = open(temp_file_path, 'rb')
            
            async with aiohttp.ClientSession() as session:
                # 准备表单数据
                form = aiohttp.FormData()
                form.add_field('file', temp_file, filename='audio.wav', content_type='audio/wav')
                form.add_field('model', self.model)

                headers = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                # 发送请求
                async with session.post(self.api_url, data=form, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result and 'text' in result:
                            text = result['text'].strip()
                            if text and self.text_callback:
                                # 只有在成功识别出文本时才输出日志
                                self.log(f"[ASR] 识别结果: {text}")
                                await self.text_callback(text)
                            # 移除未识别文本的日志输出
                        # 移除识别失败的日志输出
                    # 移除识别失败的日志输出
                    
        except Exception as e:
            # 只在发生异常时输出错误日志
            self.log(f"[ASR] 识别错误: {str(e)}")
            
        finally:
            # 确保临时文件被关闭和删除
            if temp_file:
                try:
                    temp_file.close()
                except Exception:
                    pass
                
            # 删除临时文件
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            except Exception:
                pass

    def set_text_callback(self, callback: Callable):
        """设置文本回调函数"""
        self.text_callback = callback
        self.log("[ASR] 文本回调函数已设置")

    def set_log_callback(self, callback):
        """设置日志回调函数"""
        self.log_callback = callback
        
    def log(self, message):
        """输出日志"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message) 