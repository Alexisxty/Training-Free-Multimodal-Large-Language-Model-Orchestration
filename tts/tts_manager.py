import os
import asyncio
import uuid
from typing import Optional, Callable, List
from utils.config import CONFIG, TEMP_FILE_CONFIG

from .audio import AudioPlayer
from .cache import CacheManager
from .text import TextProcessor
from .engines import TTSEngine, EdgeTTSEngine, CosyVoiceTTSEngine, GPUStackTTSEngine
from .core import BaseTTSEngine


class TTSManager:
    """TTS管理器，整合音频播放、缓存管理、文本处理和TTS引擎"""
    
    def __init__(self):
        """初始化TTS管理器"""
        # 创建临时目录
        self.temp_dir = CONFIG['TTS_TEMP_DIR']
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 创建缓存目录
        self.cache_dir = os.path.join(TEMP_FILE_CONFIG['TEMP_ROOT_DIR'], TEMP_FILE_CONFIG['TTS_CACHE_DIR'])
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 日志回调函数
        self.log_callback: Optional[Callable] = None
        
        # 创建各个组件
        self.cache_manager = CacheManager(self.cache_dir)
        self.audio_player = AudioPlayer()
        self.text_processor = TextProcessor()
        
        # 创建TTS引擎
        self.engines = {}
        self._init_engines()
        
        # 当前使用的引擎
        self.current_engine: Optional[BaseTTSEngine] = None
        self.engine_type = TTSEngine.EDGE  # 默认使用edge-tts
        
        # 设置默认引擎
        self.set_engine(TTSEngine.EDGE)
        
        # 流式处理缓冲区
        self.stream_buffer = ""
        self.stream_tasks = []
        self.sentence_count = 0  # 跟踪句子顺序
        
        # 已处理文本跟踪
        self.processed_text = ""  # 保存已经处理过的全部文本内容
        self.processed_sentences = set()  # 保存已处理句子的哈希值，用于快速去重
        
        # ASR处理器引用
        self.asr_processor = None
    
    def _init_engines(self) -> None:
        """初始化所有TTS引擎"""
        # Edge TTS
        edge_engine = EdgeTTSEngine(
            voice=CONFIG.get('TTS_VOICE', "zh-CN-XiaoxiaoNeural"),
            rate=CONFIG.get('TTS_RATE', "+0%"),
            volume=CONFIG.get('TTS_VOLUME', "+0%")
        )
        
        # CosyVoice
        cosy_engine = CosyVoiceTTSEngine(
            api_key=CONFIG.get('COSY_API_KEY', ''),
            api_base=CONFIG.get('COSY_API_BASE', 'https://api.siliconflow.cn/v1/audio/speech'),
            voice=CONFIG.get('COSY_VOICE', 'anna')
        )
        
        # GPUStack
        gpu_engine = GPUStackTTSEngine(
            api_key=CONFIG.get('GPUSTACK_API_KEY', ''),
            api_base=CONFIG.get('GPUSTACK_API_BASE', 'http://10.255.0.150:82'),
            model=CONFIG.get('GPUSTACK_TTS_MODEL', 'cosyvoice-300m-instruct'),
            voice=CONFIG.get('GPUSTACK_TTS_VOICE', 'Chinese Female')
        )
        
        # 注册引擎
        self.engines = {
            TTSEngine.EDGE: edge_engine,
            TTSEngine.COSYVOICE: cosy_engine,
            TTSEngine.GPUSTACK: gpu_engine
        }
    
    def set_log_callback(self, callback: Callable) -> None:
        """设置日志回调函数"""
        self.log_callback = callback
        # 传递给所有组件
        self.cache_manager.set_log_callback(callback)
        self.audio_player.set_log_callback(callback)
        self.text_processor.set_log_callback(callback)
        # 传递给所有引擎
        for engine in self.engines.values():
            engine.set_log_callback(callback)
    
    def log(self, message: str, debug: bool = False) -> None:
        """输出日志"""
        if debug:
            print(message)  # 调试信息只在终端显示
        elif self.log_callback:
            self.log_callback(message)  # 重要信息在界面显示
        else:
            print(message)
    
    def set_engine(self, engine_type: TTSEngine) -> None:
        """设置TTS引擎"""
        if engine_type in self.engines:
            self.engine_type = engine_type
            self.current_engine = self.engines[engine_type]
            self.log(f"[TTS] 设置引擎: {engine_type.value}")
        else:
            self.log(f"[TTS] 引擎类型不存在: {engine_type}")
    
    def set_asr_processor(self, asr_processor) -> None:
        """设置ASR处理器引用"""
        self.asr_processor = asr_processor
        self.audio_player.set_asr_processor(asr_processor)
        # 预生成常用短语
        self._pregenerate_common_phrases()
    
    def set_interrupt_callback(self, callback: Callable) -> None:
        """设置打断检测回调"""
        self.audio_player.set_interrupt_callback(callback)
    
    async def stop_speaking(self) -> None:
        """停止当前播放"""
        self.should_stop = True
        await self.audio_player.stop_speaking()
        
        # 重置流式处理状态
        self.stream_buffer = ""
        self.sentence_count = 0
        
        # 取消所有流式处理任务
        for task in self.stream_tasks:
            if not task.done():
                task.cancel()
        self.stream_tasks.clear()
        
        # 注意：不要重置processed_text和processed_sentences，因为我们需要跟踪已处理的内容
    
    def _pregenerate_common_phrases(self) -> None:
        """预生成常用短语的音频文件"""
        if not self.current_engine:
            return
            
        try:
            self.log("[TTS] 开始预生成常用短语...", debug=True)
            
            # 异步启动预生成任务
            asyncio.create_task(self._pregenerate_async())
                
        except Exception as e:
            self.log(f"[TTS] 预生成常用短语失败: {str(e)}", debug=True)
    
    async def _pregenerate_async(self) -> None:
        """异步预生成常用短语"""
        for phrase in self.cache_manager.common_phrases.keys():
            engine_id = self.current_engine.get_engine_id()
            cache_path = self.cache_manager.get_cache_path(phrase, engine_id)
            
            if not os.path.exists(cache_path):
                # 生成临时文件路径
                temp_file = os.path.join(self.temp_dir, f"tts_{uuid.uuid4()}.mp3")
                
                # 使用当前引擎合成
                success = await self.current_engine.synthesize(phrase, temp_file, silent=True)
                
                if success:
                    # 复制到缓存
                    import shutil
                    shutil.copy2(temp_file, cache_path)
                    
                    # 更新常用短语缓存
                    self.cache_manager.update_common_phrase_cache(phrase, cache_path)
                    
                    # 删除临时文件
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            else:
                # 更新常用短语缓存
                self.cache_manager.update_common_phrase_cache(phrase, cache_path)
                
        self.log("[TTS] 常用短语预生成完成", debug=True)
    
    async def text_to_speech(self, text: str) -> bool:
        """
        将文本转换为语音并播放
        
        Args:
            text: 要转换为语音的文本
            
        Returns:
            bool: 是否成功处理
        """
        try:
            print(f"[TTS_MGR_DEBUG] 开始处理文本: '{text}'")
            if not text or not text.strip():
                print("[TTS_MGR_DEBUG] 文本为空，不处理")
                return False
                
            if not self.current_engine:
                print("[TTS_MGR_DEBUG] 未设置TTS引擎，无法处理")
                return False
            
            # 检查是否已经处理过的相同文本（完全匹配）
            if text.strip() == self.processed_text.strip():
                print("[TTS_MGR_DEBUG] 收到相同的完整文本，跳过重复处理")
                return True
                
            # 检查新文本是否包含已处理文本，如果是，只处理新增部分
            if self.processed_text and text.startswith(self.processed_text):
                new_text = text[len(self.processed_text):]
                print(f"[TTS_MGR_DEBUG] 检测到新文本是已处理文本的扩展，只处理新增部分: '{new_text}'")
                
                # 如果新增部分不为空，处理新增部分
                if new_text.strip():
                    # 不停止当前播放，直接添加新的句子到队列
                    sentences = self.text_processor.split_text(new_text)
                    print(f"[TTS_MGR_DEBUG] 新增部分分割为 {len(sentences)} 个句子: {sentences}")
                    
                    # 处理每个新句子
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                            
                        # 检查是否已处理过此句子（使用句子哈希值）
                        sentence_hash = hash(sentence)
                        if sentence_hash in self.processed_sentences:
                            print(f"[TTS_MGR_DEBUG] 跳过已处理的句子: '{sentence[:20]}...'")
                            continue
                            
                        print(f"[TTS_MGR_DEBUG] 处理新增句子: '{sentence}'")
                        # 处理新句子并添加到播放队列
                        await self._process_and_queue_sentence(sentence, self.sentence_count)
                        self.sentence_count += 1
                        
                        # 标记为已处理
                        self.processed_sentences.add(sentence_hash)
                    
                    # 更新已处理文本
                    self.processed_text = text
                    return True
                    
            # 检查是否是常用短语
            engine_id = self.current_engine.get_engine_id()
            cache_path = self.cache_manager.get_common_phrase_cache(text, engine_id)
            
            if cache_path:
                print(f"[TTS_MGR_DEBUG] 使用缓存的常用短语: {text[:20]}..., 路径: {cache_path}")
                self.log(f"[TTS] 使用缓存的常用短语: {text[:20]}...", debug=True)
                # 停止当前播放
                await self.stop_speaking()
                # 播放缓存文件
                result = await self.audio_player.play_audio(cache_path)
                print(f"[TTS_MGR_DEBUG] 播放缓存结果: {result}")
                
                # 更新已处理文本
                self.processed_text = text
                self.processed_sentences = {hash(text)}
                return result
            
            # 停止当前播放（只有当不是增量处理时）
            await self.stop_speaking()
            
            # 分割文本
            sentences = self.text_processor.split_text(text)
            print(f"[TTS_MGR_DEBUG] 文本已分割为 {len(sentences)} 个句子: {sentences}")
            self.log(f"[TTS] 文本已分割为 {len(sentences)} 个句子", debug=True)
            
            # 重置状态
            self.sentence_count = 0
            self.processed_sentences.clear()
            
            # 处理每个句子
            for sentence in sentences:
                if not sentence.strip():
                    print(f"[TTS_MGR_DEBUG] 跳过空句子")
                    continue
                
                # 处理句子并添加到播放队列
                await self._process_and_queue_sentence(sentence, self.sentence_count)
                self.sentence_count += 1
                
                # 标记为已处理
                self.processed_sentences.add(hash(sentence))
            
            # 更新已处理文本
            self.processed_text = text
            
            # 不需要手动启动播放队列处理器，add_to_queue方法中会自动处理
            print(f"[TTS_MGR_DEBUG] 文本处理完成，队列中有 {len(self.audio_player.play_queue)} 个项目")
            return True
            
        except Exception as e:
            print(f"[TTS_MGR_DEBUG] 处理文本时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.log(f"[TTS] 处理文本时出错: {str(e)}", debug=True)
            return False
            
    async def _process_and_queue_sentence(self, sentence: str, sentence_index: int) -> None:
        """
        处理单个句子并添加到播放队列
        
        Args:
            sentence: 要处理的句子
            sentence_index: 句子索引
        """
        if not sentence.strip():
            return
            
        print(f"[TTS_MGR_DEBUG] 开始完整处理句子[{sentence_index}]: '{sentence[:30]}...'")
            
        # 检查缓存
        engine_id = self.current_engine.get_engine_id()
        cache_path = self.cache_manager.get_cache_path(sentence, engine_id)
        if self.cache_manager.is_cached(sentence, engine_id):
            print(f"[TTS_MGR_DEBUG] 句子已缓存[{sentence_index}]: {sentence[:20]}..., 路径: {cache_path}")
            self.log(f"[TTS] 使用缓存[{sentence_index}]: {sentence[:20]}...", debug=True)
            # 添加到播放队列，索引保证顺序
            await self.audio_player.add_to_queue(sentence_index, cache_path)
            return
        
        # 生成临时文件路径
        temp_file = os.path.join(self.temp_dir, f"tts_{uuid.uuid4()}.mp3")
        print(f"[TTS_MGR_DEBUG] 将合成音频到临时文件[{sentence_index}]: {temp_file}")
        
        # 使用当前引擎合成
        success = await self.current_engine.synthesize(sentence, temp_file)
        print(f"[TTS_MGR_DEBUG] 合成结果[{sentence_index}]: {success}, 检查文件是否存在: {os.path.exists(temp_file)}")
        
        if success:
            # 添加到缓存
            print(f"[TTS_MGR_DEBUG] 添加到缓存[{sentence_index}]")
            self.cache_manager.add_to_cache(sentence, engine_id, temp_file)
            
            # 添加到播放队列，使用索引保证顺序
            print(f"[TTS_MGR_DEBUG] 添加到播放队列[{sentence_index}]")
            await self.audio_player.add_to_queue(sentence_index, temp_file)
            print(f"[TTS_MGR_DEBUG] 完成处理句子[{sentence_index}]")

    async def process_streaming_text(self, text_chunk: str) -> None:
        """
        处理流式文本片段
        
        Args:
            text_chunk: 从LLM流式接收的文本片段
        """
        try:
            if not self.current_engine:
                return
                
            # 检查是否是常用短语
            engine_id = self.current_engine.get_engine_id()
            cache_path = self.cache_manager.get_common_phrase_cache(text_chunk, engine_id)
            
            if cache_path:
                self.log(f"[TTS] 使用缓存: {text_chunk[:20]}...", debug=True)
                
                # 检查是否已处理过此短语
                if hash(text_chunk) in self.processed_sentences:
                    print(f"[TTS_MGR_DEBUG] 跳过已处理的常用短语: '{text_chunk[:20]}...'")
                    return
                    
                # 添加到播放队列
                sentence_index = self.sentence_count
                self.sentence_count += 1
                await self.audio_player.add_to_queue(sentence_index, cache_path)
                
                # 更新已处理内容
                self.processed_sentences.add(hash(text_chunk))
                self.processed_text += text_chunk
                return
            
            # 处理文本块，更新缓冲区并获取完整句子
            new_buffer, sentences = self.text_processor.process_streaming_chunk(self.stream_buffer, text_chunk)
            
            # 更新流缓冲区
            self.stream_buffer = new_buffer
            
            # 首先收集所有要处理的句子，确保索引连续
            valid_sentences = []
            for sentence in sentences:
                if sentence.strip():
                    # 检查是否已处理过此句子
                    sentence_hash = hash(sentence)
                    if sentence_hash in self.processed_sentences:
                        print(f"[TTS_MGR_DEBUG] 跳过已处理的句子: '{sentence[:20]}...'")
                        continue
                    
                    valid_sentences.append(sentence)
                    # 标记为已处理
                    self.processed_sentences.add(sentence_hash)
                    # 累积到已处理文本
                    self.processed_text += sentence
            
            # 然后按顺序处理这些句子
            pending_tasks = []
            start_index = self.sentence_count  # 记录句子开始的索引
            
            for i, sentence in enumerate(valid_sentences):
                # 为句子分配连续的索引
                sentence_index = self.sentence_count
                self.sentence_count += 1
                
                self.log(f"[TTS] 处理流式句子[{sentence_index}]: {sentence}", debug=True)
                
                # 创建处理任务，但先不执行
                task = self._process_sentence(sentence, sentence_index)
                pending_tasks.append(task)
            
            # 确保我们有待处理的任务
            if not pending_tasks:
                return
                
            # 异步处理所有句子，但收集结果以确保按顺序播放
            print(f"[TTS_MGR_DEBUG] 同时开始{len(pending_tasks)}个合成任务，起始索引:{start_index}")
            
            # 使用gather同时开始所有合成任务
            start_tasks = asyncio.gather(*pending_tasks)
            self.stream_tasks.append(start_tasks)
            
            # 清理已完成的任务
            self.stream_tasks = [t for t in self.stream_tasks if not t.done()]
                
        except Exception as e:
            self.log(f"[TTS] 处理流式文本出错: {str(e)}", debug=True)
            import traceback
            self.log(traceback.format_exc(), debug=True)
    
    async def _process_sentence(self, sentence: str, sentence_index: int) -> None:
        """
        处理流式句子并添加到播放队列
        
        Args:
            sentence: 要处理的句子
            sentence_index: 句子索引
        """
        try:
            if not sentence.strip():
                return
                
            print(f"[TTS_MGR_DEBUG] 开始处理句子[{sentence_index}]: '{sentence[:30]}...'")
                
            # 检查缓存
            engine_id = self.current_engine.get_engine_id()
            cache_path = self.cache_manager.get_cache_path(sentence, engine_id)
            if self.cache_manager.is_cached(sentence, engine_id):
                # 句子已缓存
                self.log(f"[TTS] 使用缓存[{sentence_index}]: {sentence[:20]}...", debug=True)
                # 添加到播放队列，索引确保按顺序播放
                await self.audio_player.add_to_queue(sentence_index, cache_path)
                return
            
            # 生成临时文件路径
            temp_file = os.path.join(self.temp_dir, f"tts_{uuid.uuid4()}.mp3")
            
            # 使用当前引擎合成
            self.log(f"[{self.current_engine.get_engine_id()}] 开始合成[{sentence_index}]: {sentence[:30]}...", debug=True)
            success = await self.current_engine.synthesize(sentence, temp_file)
            
            if success:
                # 添加到缓存
                self.cache_manager.add_to_cache(sentence, engine_id, temp_file)
                self.log(f"[缓存] 添加缓存[{sentence_index}]: {sentence[:20]}...", debug=True)
                
                # 添加到播放队列，使用正确的索引保证顺序
                await self.audio_player.add_to_queue(sentence_index, temp_file)
                print(f"[TTS_MGR_DEBUG] 完成处理句子[{sentence_index}]")
                
        except Exception as e:
            self.log(f"[TTS] 处理流式句子[{sentence_index}]出错: {str(e)}", debug=True)
            import traceback
            print(traceback.format_exc())
    
    async def play_audio(self, audio_path: str) -> bool:
        """
        播放音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            bool: 是否成功开始播放
        """
        return await self.audio_player.play_audio(audio_path)
    
    def cleanup_all_temp_files(self) -> None:
        """清理所有临时文件"""
        # 如果配置不允许自动清理TTS文件，则跳过
        if not TEMP_FILE_CONFIG['AUTO_CLEAN_TTS']:
            self.log("[TTS] 根据配置，跳过清理临时文件")
            return
            
        try:
            for file in os.listdir(self.temp_dir):
                if file.startswith("tts_") and file.endswith(".mp3"):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except Exception as e:
                        self.log(f"[TTS] 清理临时文件失败: {file} - {str(e)}")
        except Exception as e:
            self.log(f"[TTS] 清理临时文件失败: {str(e)}")
    
    def __del__(self) -> None:
        """析构函数，清理资源"""
        if TEMP_FILE_CONFIG['CLEAN_ON_EXIT'] and TEMP_FILE_CONFIG['AUTO_CLEAN_TTS']:
            self.cleanup_all_temp_files() 