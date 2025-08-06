import sounddevice as sd
import numpy as np
import time
from typing import Optional, Callable
import queue

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.int16
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.callback: Optional[Callable] = None
        self.device_id = self._find_input_device()
        
    def _find_input_device(self) -> Optional[int]:
        """查找可用的音频输入设备"""
        try:
            devices = sd.query_devices()
            print("\n可用音频设备:")
            for i, device in enumerate(devices):
                print(f"{i}: {device['name']} (输入通道: {device['max_input_channels']})")
                
            # 查找默认输入设备
            default_device = sd.query_devices(kind='input')
            if default_device:
                device_id = devices.index(default_device)
                print(f"\n选择默认输入设备: {default_device['name']}")
                return device_id
                
            # 如果没有默认设备，查找第一个有输入通道的设备
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"\n选择第一个可用输入设备: {device['name']}")
                    return i
                    
            raise RuntimeError("未找到可用的音频输入设备")
            
        except Exception as e:
            print(f"查找音频设备错误: {e}")
            return None
            
    def start_recording(self):
        """开始录音"""
        if self.device_id is None:
            raise RuntimeError("未找到可用的音频输入设备")
            
        try:
            self.is_recording = True
            
            # 使用更简单的配置
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=self.dtype,
                callback=self.audio_callback,
                blocksize=1024
            )
            
            print(f"准备开始录音...")
            print(f"设备ID: {self.device_id}")
            print(f"设备名称: {sd.query_devices(self.device_id)['name']}")
            print(f"采样率: {self.sample_rate}")
            print(f"通道数: {self.channels}")
            
            self.stream.start()
            print(f"录音已启动")
            
        except Exception as e:
            self.is_recording = False
            print(f"录音启动详细错误: {str(e)}")
            raise RuntimeError(f"启动录音失败: {e}")
            
    def audio_callback(self, indata, frames, time_info, status):
        """音频回调函数"""
        if status:
            print(f"音频状态: {status}")
            
        if self.is_recording:
            try:
                # 检查音频数据
                if np.any(indata):
                    # 计算音量
                    volume_norm = np.linalg.norm(indata) * 10
                    
                    # 只处理有声音的数据
                    if volume_norm > 0.1:
                        audio_data = indata.copy()
                        # 归一化
                        audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
                        self.audio_queue.put(audio_data.tobytes())
                
            except Exception as e:
                print(f"音频处理错误: {e}")
                
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
                print("录音已停止")
            except Exception as e:
                print(f"停止录音错误: {e}")
                
    def get_audio_data(self) -> Optional[bytes]:
        """获取音频数据"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None 