import cv2
import numpy as np
import time
import threading
import queue
import os
from datetime import datetime
from utils.config import TEMP_FILE_CONFIG

class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.should_save_frames = False  # 控制是否保存帧
        self.log_callback = None  # 添加日志回调
        self.last_save_time = 0  # 上次保存帧的时间
        self.save_interval = 1.0  # 保存帧的最小时间间隔（秒）
        
        # 帧缓存队列
        self.frame_buffer = queue.Queue(maxsize=30)  # 保存最近30帧
        
        # 创建帧保存目录，使用统一的临时文件配置
        self.frames_dir = os.path.join(TEMP_FILE_CONFIG['TEMP_ROOT_DIR'], TEMP_FILE_CONFIG['VIDEO_DIR'])
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        
        # 使用日期作为会话目录
        self.session_id = datetime.now().strftime("%Y%m%d")
        self.session_dir = os.path.join(self.frames_dir, self.session_id)
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
        
        self.log("VideoProcessor initialized")
        
    def set_log_callback(self, callback):
        """设置日志回调函数"""
        self.log_callback = callback
        
    def log(self, message):
        """输出日志"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
            
    def start_capture(self):
        """开始视频捕获"""
        if self.is_running:
            return
            
        self.cap = cv2.VideoCapture(0)  # 使用默认摄像头
        if not self.cap.isOpened():
            self.log("无法打开摄像头")
            return
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
    def _capture_loop(self):
        """视频捕获循环"""
        try:
            while self.is_running:
                if not self.cap or not self.cap.isOpened():
                    self.log("[VIDEO] 摄像头未打开")
                    break
                    
                ret, frame = self.cap.read()
                if not ret:
                    self.log("[VIDEO] 无法读取视频帧")
                    break
                    
                # 更新最新帧
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    
                # 更新帧缓存
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()  # 移除最旧的帧
                    except queue.Empty:
                        pass
                self.frame_buffer.put(frame.copy())
                
        except Exception as e:
            self.log(f"[VIDEO] 视频捕获错误: {str(e)}")
        finally:
            self.is_running = False
            
    def get_latest_frame(self, save_frame=False):
        """获取最新的视频帧，可选是否保存"""
        if not self.is_running:
            return None
            
        with self.frame_lock:
            if self.latest_frame is None:
                return None
                
            frame = self.latest_frame.copy()
            
            # 只在需要时保存帧，并且检查时间间隔
            current_time = time.time()
            if (save_frame or self.should_save_frames) and \
               (current_time - self.last_save_time >= self.save_interval):
                try:
                    timestamp = datetime.now().strftime("%H%M%S")
                    frame_path = os.path.join(self.session_dir, f"frame_{timestamp}.jpg")
                    cv2.imwrite(frame_path, frame)
                    self.last_save_time = current_time  # 更新保存时间
                    self.log(f"已保存视觉帧")  # 简化日志输出
                except Exception as e:
                    self.log(f"保存视觉帧失败: {str(e)}")
                    
            return frame
            
    def set_save_frames(self, should_save):
        """设置是否保存帧"""
        self.should_save_frames = should_save
            
    def get_frame_sequence(self, num_frames=5, save_frames=False):
        """获取最近的多帧图像"""
        frames = []
        try:
            # 从帧缓存中获取最近的帧
            while len(frames) < num_frames and not self.frame_buffer.empty():
                frame = self.frame_buffer.get()
                frames.append(frame)
                
                # 只在需要时保存帧
                if save_frames or self.should_save_frames:
                    timestamp = datetime.now().strftime("%H%M%S_%f")
                    frame_path = os.path.join(self.session_dir, f"frame_{timestamp}.jpg")
                    cv2.imwrite(frame_path, frame)
                    print(f"[VIDEO] 保存视觉帧: {frame_path}")
        except Exception as e:
            print(f"[VIDEO] 获取帧序列失败: {str(e)}")
            
        return frames
            
    def stop_capture(self):
        """停止视频捕获"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if threading.current_thread() != self.capture_thread:
            if self.capture_thread and self.capture_thread.is_alive():
                try:
                    self.capture_thread.join(timeout=1.0)
                except RuntimeError:
                    print("[VIDEO] 无法等待捕获线程结束")
                    
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                print(f"[VIDEO] 释放摄像头失败: {str(e)}")
            
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        try:
            self.stop_capture()
        except Exception as e:
            print(f"[VIDEO] 清理资源失败: {str(e)}")
            
    def get_latest_frame_path(self) -> str:
        """获取最新保存的帧路径"""
        try:
            frames = os.listdir(self.session_dir)
            if frames:
                frames.sort(reverse=True)
                return os.path.join(self.session_dir, frames[0])
        except Exception as e:
            print(f"[VIDEO] 获取最新帧路径失败: {str(e)}")
        return None