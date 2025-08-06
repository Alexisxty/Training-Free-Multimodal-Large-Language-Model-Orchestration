from typing import List, Optional, Callable


class TextProcessor:
    """文本处理类，负责文本分割和预处理"""
    
    def __init__(self):
        # 日志回调函数
        self.log_callback: Optional[Callable] = None
        # 最小句子长度
        self.min_length = 5
        # 最大句子长度（合并短句子时使用）
        self.max_length = 100
    
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
    
    def split_text(self, text: str) -> List[str]:
        """
        分割文本为句子
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的句子列表
        """
        if not text:
            return []
            
        # 移除特殊标记
        text = text.replace("[S.SPEAK]", "").replace("[S.STOP]", "")
        
        # 定义分隔符
        separators = ['。', '！', '？', '；', '!', '?', ';']
        
        # 保存所有句子
        sentences = []
        current_sentence = ""
        
        # 遍历文本
        for char in text:
            current_sentence += char
            
            # 如果当前字符是分隔符，且当前句子不为空
            if char in separators:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
                
        # 处理最后一个句子
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
            
        # 合并过短的句子
        merged_sentences = []
        temp_sentence = ""
        
        for sentence in sentences:
            if len(temp_sentence) + len(sentence) < self.max_length:  # 控制最大长度
                temp_sentence += sentence
            else:
                if temp_sentence:
                    merged_sentences.append(temp_sentence)
                temp_sentence = sentence
                
        # 添加最后一个句子
        if temp_sentence:
            merged_sentences.append(temp_sentence)
            
        # 过滤空句子
        return [s for s in merged_sentences if len(s.strip()) >= self.min_length]
    
    def is_sentence_end(self, text: str) -> bool:
        """
        判断文本是否以句子结束符结尾
        
        Args:
            text: 要检查的文本
            
        Returns:
            bool: 是否以句子结束符结尾
        """
        if not text:
            return False
            
        separators = ['。', '！', '？', '；', '!', '?', ';']
        return text[-1] in separators
    
    def process_streaming_chunk(self, buffer: str, chunk: str) -> tuple[str, List[str]]:
        """
        处理流式文本块，返回更新后的缓冲区和完整句子
        
        Args:
            buffer: 当前缓冲区
            chunk: 新接收的文本块
            
        Returns:
            tuple: (新缓冲区, 完整句子列表)
        """
        try:
            # 处理文本编码问题，替换不可识别的字符
            if isinstance(chunk, bytes):
                chunk = chunk.decode('utf-8', errors='replace')
            elif isinstance(chunk, str):
                # 严格过滤不可打印字符和乱码
                chunk = ''.join(char for char in chunk if ord(char) > 31 and ord(char) != 127)
                # 处理UTF-8特殊情况
                chunk = chunk.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            
            # 特殊情况：过滤掉Unicode替换字符
            chunk = chunk.replace('\ufffd', '')
            
            # 检测并修复中文字符截断问题
            if buffer and chunk and len(buffer) > 0:
                # 检查buffer最后一个字节是否是多字节字符的一部分
                last_buffer_bytes = buffer[-1].encode('utf-8')
                if len(last_buffer_bytes) > 1 and len(last_buffer_bytes) < 4:  # 可能是不完整的UTF-8字符
                    # 尝试合并看是否能形成有效字符
                    combined = (buffer[-1] + chunk[0]).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                    if combined != buffer[-1] + chunk[0]:
                        # 修复截断的字符
                        buffer = buffer[:-1] + combined
                        chunk = chunk[1:]
            
            # 更新缓冲区
            buffer += chunk
            
            # 定义分隔符
            separators = ['。', '！', '？', '；', '!', '?', ';', '，', ',']
            
            # 检查缓冲区是否包含分隔符
            sentences = []
            new_buffer = buffer
            
            for sep in separators:
                if sep in new_buffer:
                    # 分割句子
                    parts = new_buffer.split(sep)
                    # 除了最后一个部分，其他都是完整句子
                    for i in range(len(parts) - 1):
                        # 确保句子中不包含乱码
                        clean_sentence = parts[i] + sep
                        if clean_sentence.strip():
                            sentences.append(clean_sentence)
                    # 最后一个部分作为新的缓冲区
                    new_buffer = parts[-1]
            
            # 最后对所有句子进行二次清理
            clean_sentences = []
            for sentence in sentences:
                # 去除连续重复的标点符号
                for sep in separators:
                    sentence = sentence.replace(sep + sep, sep)
                # 确保句子长度合理
                if len(sentence.strip()) >= 2:
                    clean_sentences.append(sentence)
            
            return new_buffer, clean_sentences
        
        except Exception as e:
            print(f"处理流式文本块时出错: {str(e)}")
            # 出错时返回原缓冲区和空句子列表
            return buffer, [] 