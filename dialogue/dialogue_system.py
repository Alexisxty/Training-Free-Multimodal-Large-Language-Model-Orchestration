import asyncio
import time
import os
from datetime import datetime
from utils.file_manager import FileManager
from video.video_processor import VideoProcessor
from asr.asr_processor import ASRProcessor
from llm.llm_manager import LLMManager
from tts.tts_processor import TTSProcessor, TTSEngine
from memory.memory_manager import MemoryManager
from utils.config import CONFIG, TEMP_FILE_CONFIG

class DialogueSystem:
    def __init__(self):
        """初始化对话系统"""
        # 创建文件管理器
        self.file_manager = FileManager()
        
        # 初始化各个处理器
        self.video_processor = VideoProcessor()
        self.asr_processor = ASRProcessor(api_url=CONFIG['ASR_API_URL'])
        self.llm_manager = LLMManager()
        self.tts_processor = TTSProcessor()
        
        self.log_callback = None
        
        # 根据配置设置TTS引擎
        if CONFIG['TTS_ENGINE'].lower() == 'cosyvoice':
            self.tts_processor.set_engine(TTSEngine.COSYVOICE)
            self.tts_processor.set_cosy_voice(CONFIG['COSY_VOICE'])
            
        self.dialogue_logger = MemoryManager()
        
        # 设置ASR和TTS的相互引用
        self.tts_processor.set_asr_processor(self.asr_processor)
        
        # 设置LLM管理器的对话系统引用
        self.llm_manager.set_dialogue_system(self)
        
        # 设置视频处理器
        self.llm_manager.set_video_processor(self.video_processor)
        
        self.is_running = True
        
        # 防重复处理
        self.last_processed_text = ""
        self.last_process_time = 0
        self.min_process_interval = 1.0
        self.is_processing = False
        
        # 流式响应状态
        self.streaming_response = ""
        self.is_streaming = False
        
        # 设置打断检测
        def should_interrupt():
            latest_response = self.llm_manager.get_latest_response()
            if latest_response and "<S.STOP>" in latest_response:
                self.log("[SYSTEM] LLM检测到打断意图")
                return True
            return False
            
        self.tts_processor.set_interrupt_callback(should_interrupt)
        
        # 设置ASR回调
        self.log("[SYSTEM] 正在设置ASR回调函数...")
        self.asr_processor.set_text_callback(self.handle_asr_result)
        
    def set_log_callback(self, callback):
        """设置日志回调函数"""
        self.log_callback = callback
        # 设置所有组件的日志回调
        self.asr_processor.set_log_callback(callback)
        self.tts_processor.set_log_callback(callback)
        self.video_processor.set_log_callback(callback)
        self.llm_manager.set_log_callback(callback)
        
    def log(self, message):
        """输出日志"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
            
    async def handle_asr_result(self, text: str):
        """处理ASR结果"""
        if not text:  # 如果文本为空，直接返回
            return
            
        current_time = time.time()
        cleaned_text = text.strip()
        
        # 快速检查打断命令
        stop_keywords = ["停", "停下", "别说了", "闭嘴", "等等", "等一下"]
        if any(keyword in cleaned_text for keyword in stop_keywords):
            self.log("[SYSTEM] 检测到打断命令")
            await self.tts_processor.stop_speaking()
            self.is_processing = False
            return
            
        # 简化重复检查逻辑
        if cleaned_text == self.last_processed_text and \
           (current_time - self.last_process_time < self.min_process_interval):
            return
            
        # 记录当前输入
        self.last_processed_text = cleaned_text
        self.last_process_time = current_time
        
        # 异步记录用户输入
        asyncio.create_task(self.dialogue_logger.log_interaction("user", cleaned_text, current_time))
        
        # 获取上下文（可以考虑缓存）
        context = self.dialogue_logger.get_formatted_context()
        
        # 处理对话
        response = await self.llm_manager.process_dialogue(cleaned_text, context)
        
        if response:
            # 处理响应
            await self.process_llm_response(response)
        else:
            self.log("[SYSTEM] LLM没有返回响应")
            
    async def process_streaming_text(self, text_chunk: str):
        """处理LLM的流式文本响应片段"""
        try:
            # 确保输入是有效的字符串
            if not isinstance(text_chunk, str):
                self.log(f"[SYSTEM] 警告: 收到非字符串类型的响应片段: {type(text_chunk)}")
                if isinstance(text_chunk, bytes):
                    text_chunk = text_chunk.decode('utf-8', errors='replace')
                else:
                    text_chunk = str(text_chunk)
            
            # 预处理流式文本：移除控制字符、替换无效字符
            text_chunk = ''.join(char for char in text_chunk if ord(char) > 31 or char in '\n\t')
            text_chunk = text_chunk.replace('\ufffd', '')
            
            # 累积到完整响应
            old_length = len(self.streaming_response)
            self.streaming_response += text_chunk
            
            # 检查是否包含特殊标记
            has_special_marker = False
            cleaned_text = text_chunk
            
            for marker in ["[S.SPEAK]", "[C.LISTEN]", "<S.STOP>"]:
                if marker in text_chunk:
                    has_special_marker = True
                    cleaned_text = text_chunk.replace(marker, "").strip()
                    self.log(f"[SYSTEM] 收到特殊标记: {marker}")
                    break
            
            # 发送到TTS处理
            if cleaned_text.strip():
                # 检查特殊字符组合，这些可能是编码问题的表现
                if '' in cleaned_text:
                    #self.log(f"[SYSTEM] 警告: 检测到可能的编码问题, 将尝试修复")
                    cleaned_text = cleaned_text.replace('', '')
                
                # 每收到一定长度的文本才处理，避免太频繁处理小片段
                # 通常每个句子最少需要10个字符
                if len(cleaned_text) >= 2 or '。' in cleaned_text or '！' in cleaned_text or '？' in cleaned_text:
                    print(f"[SYSTEM_DEBUG] 处理流式文本片段: '{cleaned_text}'")
                    # 启动TTS处理流式文本
                    await self.tts_processor.process_streaming_text(cleaned_text)
                else:
                    print(f"[SYSTEM_DEBUG] 文本片段太短，暂不处理: '{cleaned_text}'")
                
            # 如果收到了停止标记
            if "<S.STOP>" in text_chunk:
                self.log("[SYSTEM] 收到停止指令")
                self.is_running = False
                
        except Exception as e:
            import traceback
            self.log(f"[SYSTEM] 处理流式文本失败: {str(e)}")
            self.log(f"[SYSTEM] 错误详情: {traceback.format_exc()}")
            # 即使出错了，也尝试处理文本
            try:
                if text_chunk and text_chunk.strip():
                    await self.tts_processor.process_streaming_text(text_chunk.strip())
            except:
                pass

    async def process_llm_response(self, response: str):
        """处理LLM的响应，包括中间反馈"""
        try:
            print(f"[SYSTEM_DEBUG] 原始LLM响应: '{response}'")
            # 简化标记处理
            cleaned_response = response
            for marker in ["[S.SPEAK]", "[C.LISTEN]", "<S.STOP>"]:
                if marker in response:
                    cleaned_response = marker + response.replace(marker, "").strip()
                    print(f"[SYSTEM_DEBUG] 发现标记: {marker}, 清理后响应: '{cleaned_response}'")
                    break
            
            self.log(f"[SYSTEM] LLM响应: {cleaned_response}")
            
            # 异步记录助手响应（除了中间反馈）
            if not any(feedback in cleaned_response for feedback in ["请让我看看...", "请让我思考一下..."]):
                asyncio.create_task(self.dialogue_logger.log_interaction("assistant", cleaned_response, time.time()))
            
            if "[S.SPEAK]" in cleaned_response:
                text_to_speak = cleaned_response.replace("[S.SPEAK]", "").strip()
                print(f"[SYSTEM_DEBUG] 处理[S.SPEAK]响应，将播放文本: '{text_to_speak}'")
                
                # 判断如果已经在流式播放同一句话，则不重复处理
                # 通过获取tts_processor的manager中的processed_text
                # 检查新文本是否是已处理文本的扩展
                processed_text = self.tts_processor.manager.processed_text.strip()
                
                # 计算完整响应和已处理文本的相似度
                if processed_text:
                    # 如果已处理文本是完整响应的前缀或两者非常相似
                    if text_to_speak.startswith(processed_text):
                        print(f"[SYSTEM_DEBUG] 检测到已处理流式文本是完整响应的前缀")
                        # 只处理新增部分
                        print(f"[SYSTEM_DEBUG] 继续处理增量内容: '{text_to_speak[len(processed_text):]}'")
                        await self.tts_processor.text_to_speech(text_to_speak)
                    elif processed_text.startswith(text_to_speak):
                        # 如果完整响应是已处理文本的前缀（不太可能发生）
                        print(f"[SYSTEM_DEBUG] 完整响应是已处理文本的前缀，无需处理")
                        return
                    elif len(processed_text) > 0.7 * len(text_to_speak):
                        # 如果已处理文本已经包含了大部分完整响应
                        print(f"[SYSTEM_DEBUG] 已处理文本已包含大部分内容，继续处理完整响应")
                        await self.tts_processor.text_to_speech(text_to_speak)
                    else:
                        # 如果内容差异较大，则重新开始
                        print(f"[SYSTEM_DEBUG] 内容差异较大，完全重新处理")
                        # 重置已处理文本
                        self.tts_processor.manager.processed_text = ""
                        self.tts_processor.manager.processed_sentences.clear()
                        # 停止当前播放并重新开始
                        await self.tts_processor.stop_speaking()
                        await asyncio.sleep(0.2)  # 给停止过程一些时间
                        await self.tts_processor.text_to_speech(text_to_speak)
                else:
                    # 如果还没有处理过任何文本
                    print(f"[SYSTEM_DEBUG] 尚未处理任何流式文本，开始处理完整响应")
                    await self.tts_processor.text_to_speech(text_to_speak)
                    
            elif "[C.LISTEN]" in cleaned_response:
                self.log("[SYSTEM] 继续等待用户输入...")
            elif "<S.STOP>" in cleaned_response:
                self.log("[SYSTEM] 收到停止指令")
                self.is_running = False
            else:
                print(f"[SYSTEM_DEBUG] 未检测到特殊标记，将直接播放文本: '{cleaned_response}'")
                await self.tts_processor.text_to_speech(cleaned_response)
                
        except Exception as e:
            self.log(f"[SYSTEM] 处理LLM响应失败: {str(e)}")
            print(f"[SYSTEM_DEBUG] 处理LLM响应失败详情: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.is_processing = False
            
    async def run(self):
        """运行对话系统"""
        self.log("[SYSTEM] 正在启动对话系统...")
        self.log("[SYSTEM] 准备开始录音...")
        await self.asr_processor.start_listening()
        self.log("[SYSTEM] 开始视频捕获...")
        self.video_processor.start_capture()
        
        try:
            while self.is_running:
                await asyncio.sleep(0.1)
        except Exception as e:
            self.log(f"[SYSTEM] 运行错误: {str(e)}")
        finally:
            await self.stop()
            
    async def stop(self):
        """停止对话系统"""
        self.log("[SYSTEM] 正在停止对话系统...")
        self.is_running = False
        await self.asr_processor.stop_listening()
        self.video_processor.stop_capture()
        await self.llm_manager.close()
        
        # 根据配置决定是否清理临时文件
        if TEMP_FILE_CONFIG['CLEAN_ON_EXIT']:
            self.log("[SYSTEM] 清理临时文件...")
            self.tts_processor.cleanup_all_temp_files()  # 清理TTS临时文件
            self.file_manager.clean_session()  # 清理会话临时文件
        else:
            self.log(f"[SYSTEM] 保留临时文件于: {self.file_manager.session_dir}")
            
        self.log("[SYSTEM] 对话系统已停止")