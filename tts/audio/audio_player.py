import pygame
import os
import asyncio
from typing import List, Tuple, Optional, Callable, Deque
from collections import deque
from utils.config import TEMP_FILE_CONFIG
import time
from threading import Thread
import heapq


class AudioPlayer:
    """音频播放器类，负责TTS音频播放管理"""
    
    def __init__(self):
        """初始化音频播放器"""
        # 初始化pygame音频
        try:
            pygame.mixer.init(frequency=24000)  # 提高音频质量
            init_status = pygame.mixer.get_init()
            print(f"[播放器] pygame混音器初始化状态: {init_status}")
            if not init_status:
                print("[播放器] 警告: pygame混音器初始化失败!")
            else:
                print(f"[播放器] pygame混音器成功初始化: 频率={init_status[0]}Hz, 格式={init_status[1]}, 通道={init_status[2]}")
        except Exception as e:
            print(f"[播放器] pygame初始化错误: {str(e)}")
        
        # 日志回调函数
        self.log_callback: Optional[Callable] = None
        
        # 播放状态
        self.is_speaking = False
        self.should_stop = False
        
        # 播放控制标志
        self.is_playing_enabled = True  # 控制是否实际播放音频，用于性能测试
        
        # 使用优先队列存储(索引,文件路径)元组
        self.play_queue = []  # 使用heapq作为优先队列
        self.play_task: Optional[asyncio.Task] = None
        self.next_play_index = 0  # 下一个要播放的序号
        
        # 添加队列锁，防止并发访问
        self.queue_lock = asyncio.Lock()
        
        # 添加工作器标志，确保只有一个工作器运行
        self.worker_running = False
        
        # 初始化worker属性
        self.worker = None
        
        # 打断检测回调
        self.interrupt_callback: Optional[Callable] = None
        
        # ASR处理器引用（用于通知TTS状态）
        self.asr_processor = None
        
        # 当前正在播放的索引
        self.current_play_index = -1
        
        # 期望的下一个索引值，用于检测是否有缺失的句子
        self.expected_next_index = 0
        
        # 等待缺失句子的最大时间（秒）
        self.max_wait_time = 3.0
    
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
    
    def set_interrupt_callback(self, callback: Callable) -> None:
        """设置打断检测回调"""
        self.interrupt_callback = callback
    
    def set_asr_processor(self, asr_processor) -> None:
        """设置ASR处理器引用"""
        self.asr_processor = asr_processor
    
    def ensure_worker_running(self):
        """确保播放队列处理器正在运行"""
        if not self.worker or not self.worker.is_alive():
            print("[播放器] 启动新的播放队列处理器")
            # 重置状态
            self.is_speaking = True
            self.should_stop = False
            
            # 显示当前状态
            print(f"[播放器] 当前状态: is_speaking={self.is_speaking}, should_stop={self.should_stop}, 队列长度={len(self.play_queue)}")
            
            # 启动工作线程
            self.worker = Thread(target=self.play_queue_worker, daemon=True)
            self.worker.start()
            print("[播放器] 播放队列处理器启动")

    def _sync_stop_speaking(self):
        """同步版本：停止播放并清空队列"""
        print(f"[播放器] 同步：收到停止指令: 当前状态 is_speaking={self.is_speaking}, should_stop={self.should_stop}")
        self.should_stop = True
        self.play_queue.clear()
        print("[播放器] 同步：队列已清空，等待播放器停止")
        try:
            pygame.mixer.music.stop()
            print("[播放器] 同步：pygame播放已停止")
        except Exception as e:
            print(f"[播放器] 同步：停止pygame失败: {str(e)}")
        time.sleep(0.2)  # 给播放器一些时间来停止
        self.is_speaking = False
        self.current_play_index = -1
        self.expected_next_index = 0  # 重置期望的下一个索引
        print("[播放器] 同步：停止完成")
        
    async def stop_speaking(self):
        """异步版本：停止播放并清空队列"""
        print(f"[播放器] 异步：收到停止指令: 当前状态 is_speaking={self.is_speaking}, should_stop={self.should_stop}")
        
        # 设置停止标志
        self.should_stop = True
        
        # 清空队列前保存当前状态
        queue_length = len(self.play_queue)
        self.play_queue.clear()
        print(f"[播放器] 异步：队列已清空，原队列长度={queue_length}")
        
        # 停止当前播放
        try:
            pygame.mixer.music.stop()
            print("[播放器] 异步：pygame播放已停止")
        except Exception as e:
            print(f"[播放器] 异步：停止pygame失败: {str(e)}")
            
        # 等待一段时间，确保播放器完全停止
        await asyncio.sleep(0.2)
        
        # 重置状态
        self.is_speaking = False
        self.current_play_index = -1
        self.expected_next_index = 0  # 重置期望的下一个索引
        
        # 如果工作线程正在运行，尝试等待它退出
        if self.worker and self.worker.is_alive():
            print("[播放器] 异步：等待工作线程停止...")
            await asyncio.sleep(0.3)  # 给线程一些时间来退出
            
        print("[播放器] 异步：停止完成")
        
        # 创建一个全新的队列对象，彻底清除引用
        self.play_queue = []

    def play_queue_worker(self):
        """播放队列处理器"""
        print(f"[播放器] 播放队列处理器启动 - 当前状态: is_speaking={self.is_speaking}, should_stop={self.should_stop}, 队列长度={len(self.play_queue)}")
        
        try:
            while self.is_speaking and not self.should_stop:
                if len(self.play_queue) > 0:
                    print(f"[播放器] 队列中有{len(self.play_queue)}个文件待播放")
                    
                    # 检查队列中最小的索引
                    min_index = self.play_queue[0][0] if self.play_queue else -1
                    
                    # 检查是否是期望的下一个索引
                    if min_index > self.expected_next_index:
                        # 有缺失的句子，等待一段时间
                        print(f"[播放器] 警告: 期望播放索引 {self.expected_next_index}，但队列中最小索引为 {min_index}，等待缺失的句子...")
                        
                        # 记录开始等待的时间
                        wait_start = time.time()
                        waited = False
                        
                        # 等待缺失的句子到达，但不超过最大等待时间
                        while (len(self.play_queue) > 0 and 
                               self.play_queue[0][0] > self.expected_next_index and 
                               time.time() - wait_start < self.max_wait_time and
                               not self.should_stop):
                            time.sleep(0.2)  # 短暂休眠，等待缺失句子
                            waited = True
                            
                        if waited:
                            print(f"[播放器] 等待结束，当前队列最小索引: {self.play_queue[0][0] if self.play_queue else 'N/A'}")
                            
                        # 如果仍然缺失，更新期望索引为当前最小索引
                        if (len(self.play_queue) > 0 and 
                            self.play_queue[0][0] > self.expected_next_index):
                            print(f"[播放器] 索引 {self.expected_next_index} 的句子未到达，将继续播放索引 {self.play_queue[0][0]}")
                            self.expected_next_index = self.play_queue[0][0]
                    
                    # 从优先队列中获取最小索引的文件（按索引排序）
                    index, temp_file = heapq.heappop(self.play_queue)
                    
                    # 更新当前播放索引和期望的下一个索引
                    self.current_play_index = index
                    self.expected_next_index = index + 1
                    
                    # 播放音频文件
                    print(f"[播放器] 开始播放音频 [索引={index}]: {temp_file}")
                    
                    # 如果启用了播放功能，才实际播放
                    if self.is_playing_enabled:
                        try:
                            # 处理音频播放
                            pygame.mixer.music.load(temp_file)
                            pygame.mixer.music.play()
                        
                            # 检查是否需要通知ASR处理器TTS开始播放
                            if self.asr_processor:
                                self.asr_processor.on_tts_start()
                        
                            # 等待播放结束，同时检查是否被要求停止
                            while pygame.mixer.music.get_busy() and not self.should_stop:
                                # 如果存在打断检测回调，定期调用它
                                if self.interrupt_callback and self.interrupt_callback():
                                    print("[播放器] 检测到打断，停止播放")
                                    pygame.mixer.music.stop()
                                    self.should_stop = True
                                    break
                                    
                                # 短暂休眠，避免CPU过度占用
                                time.sleep(0.1)
                            
                            # 检查是否需要通知ASR处理器TTS结束播放
                            if self.asr_processor:
                                self.asr_processor.on_tts_end()
                                
                        except Exception as e:
                            print(f"[播放器] 播放音频文件时出错: {str(e)}")
                    else:
                        # 性能测试模式，不实际播放
                        print(f"[播放器] 性能测试模式，跳过实际播放 [索引={index}]: {temp_file}")
                else:
                    # 队列为空，等待
                    time.sleep(0.1)
        except Exception as e:
            print(f"[播放器] 播放队列处理器错误: {str(e)}")
        
        # 工作器结束
        print("[播放器] 播放队列处理器结束")
        self.is_speaking = False
    
    async def add_to_queue(self, index: int, file_path: str) -> None:
        """
        添加文件到播放队列
        
        Args:
            index: 文件索引（用于保证顺序）
            file_path: 音频文件路径
        """
        # 使用锁保护队列操作
        async with self.queue_lock:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                self.log(f"[播放器] 文件不存在，无法添加到队列: {file_path}", debug=True)
                return
            
            # 如果当前正在播放的索引比要添加的索引大，说明顺序有问题
            if self.current_play_index > index and self.current_play_index != -1:
                print(f"[播放器] 警告：尝试添加较早的句子（索引{index}），当前播放索引为{self.current_play_index}，但仍添加到队列")
            
            # 添加到优先队列，按索引排序
            heapq.heappush(self.play_queue, (index, file_path))
            print(f"[播放器] 添加文件到队列: {os.path.basename(file_path)}，索引:{index}，当前队列长度: {len(self.play_queue)}")
            
            # 确保工作器运行
            self._sync_ensure_worker_running()
            
            # 如果队列原本为空，给播放器一些时间启动
            if len(self.play_queue) == 1:
                # 释放锁，让播放线程能够取出并播放文件
                self.queue_lock.release()
                await asyncio.sleep(0.3)  # 给播放器一些启动时间
                # 重新获取锁以便正常退出函数
                await self.queue_lock.acquire()
    
    async def ensure_worker_running(self):
        """异步版本的工作器启动方法"""
        # 避免递归调用，使用不同的方法名
        self._sync_ensure_worker_running()
        
    def _sync_ensure_worker_running(self):
        """同步版本的工作器启动方法"""
        # 如果工作线程已存在但需要停止，先等待它结束
        if self.should_stop and self.worker and self.worker.is_alive():
            print("[播放器] 等待现有工作线程停止...")
            time.sleep(0.2)  # 等待线程退出
        
        # 如果不存在工作线程或已经停止，则创建新线程
        if not self.worker or not self.worker.is_alive():
            print("[播放器] 启动新的播放队列处理器")
            # 重置状态
            self.is_speaking = True
            self.should_stop = False
            
            # 显示当前状态
            print(f"[播放器] 当前状态: is_speaking={self.is_speaking}, should_stop={self.should_stop}, 队列长度={len(self.play_queue)}")
            
            # 启动工作线程
            self.worker = Thread(target=self.play_queue_worker, daemon=True)
            self.worker.start()
            print("[播放器] 播放队列处理器启动")
        else:
            print(f"[播放器] 工作线程已在运行中 (is_alive={self.worker.is_alive()})")
    
    async def play_audio(self, file_path: str) -> bool:
        """
        播放单个音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            bool: 是否播放成功
        """
        try:
            if not os.path.exists(file_path):
                self.log(f"[播放器] 音频文件不存在: {file_path}", debug=True)
                return False
            
            # 清理临时文件
            print("[播放器] 清理临时文件...")
            await self.clean_temp_files()
                
            # 停止当前播放并等待完全停止
            print(f"[播放器] 播放新音频前停止当前播放: {file_path}")
            await self.stop_speaking()
            
            # 等待一小段时间，确保之前的播放器彻底停止
            await asyncio.sleep(0.1)
            
            # 确保我们使用新的队列，彻底避开之前队列中可能的引用问题
            self.play_queue = []
            self.current_play_index = -1
            
            # 添加新文件到队列
            print(f"[播放器] 添加新文件到播放队列: {file_path}")
            async with self.queue_lock:
                self.next_play_index = 0
                # 添加到优先队列
                heapq.heappush(self.play_queue, (0, file_path))
            
            # 设置状态并启动播放
            self.is_speaking = True
            self.should_stop = False
            
            # 确保工作器运行 (重新启动一个工作线程)
            print("[播放器] 启动工作线程处理新文件")
            self._sync_ensure_worker_running()
                
            # 等待播放完成
            while len(self.play_queue) > 0 and not self.should_stop:
                await asyncio.sleep(0.1)
                
            print(f"[播放器] 新文件播放完成: {file_path}")
            return True
                
        except Exception as e:
            self.log(f"[播放器] 播放音频出错: {str(e)}", debug=True)
            import traceback
            print(traceback.format_exc())
            return False
    
    async def preload_audio(self, file_path: str) -> bool:
        """
        预加载音频文件到内存
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            bool: 是否预加载成功
        """
        try:
            # 这里只是读取文件内容，不做其他处理
            with open(file_path, "rb") as f:
                _ = f.read()
            return True
        except Exception:
            return False
    
    def __del__(self):
        """析构函数，清理资源"""
        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
        except:
            pass
            
    async def clean_temp_files(self):
        """清理临时文件，防止旧文件干扰"""
        try:
            # 获取TTS临时目录
            temp_dir = os.path.join('temp', 'tts')
            if os.path.exists(temp_dir):
                print(f"[播放器] 尝试清理临时文件目录: {temp_dir}")
                # 寻找临时TTS文件
                for file_name in os.listdir(temp_dir):
                    if file_name.startswith('tts_') and file_name.endswith('.mp3'):
                        try:
                            file_path = os.path.join(temp_dir, file_name)
                            # 尝试删除文件
                            os.remove(file_path)
                            print(f"[播放器] 已删除临时文件: {file_path}")
                        except Exception as e:
                            # 如果文件正在使用中，可能无法删除
                            print(f"[播放器] 无法删除文件 {file_name}: {str(e)}")
                print("[播放器] 临时文件清理完成")
        except Exception as e:
            print(f"[播放器] 清理临时文件出错: {str(e)}")
            import traceback
            print(traceback.format_exc())