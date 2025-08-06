import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any
from utils.config import CONFIG, TEMP_FILE_CONFIG

class MemoryManager:
    """
    记忆管理器 - 负责对话历史的存储、检索和摘要生成
    """
    def __init__(self, log_dir=None):
        """
        初始化对话历史记录器
        
        Args:
            log_dir: 对话历史记录的根目录
        """
        if log_dir is None:
            # 使用配置中的对话历史目录
            log_dir = os.path.join(TEMP_FILE_CONFIG['TEMP_ROOT_DIR'], TEMP_FILE_CONFIG['DIALOGUE_HISTORY_DIR'])
            
        self.log_dir = log_dir
        self.memory_dir = os.path.join(log_dir, "memories")
        self.raw_dir = os.path.join(log_dir, "raw")
        
        # 创建必要的目录
        for dir_path in [self.log_dir, self.memory_dir, self.raw_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # 创建新的对话ID
        self.dialogue_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.raw_log_file = os.path.join(self.raw_dir, f"dialogue_{self.dialogue_id}.json")
        self.memory_file = os.path.join(self.memory_dir, f"memory_{self.dialogue_id}.json")
        
        # 初始化对话历史和记忆
        self.raw_history: List[Dict] = []
        self.memory_history: List[Dict] = []
        self.conversation_count = 0
        self.memory_summary_interval = 4  # 每4轮对话总结一次
        
        # 日志回调
        self.log_callback = None
        
        # LLM处理回调
        self.llm_processor = None
        
        # 初始化文件
        self._initialize_files()
        
    def set_log_callback(self, callback: Callable) -> None:
        """设置日志回调函数"""
        self.log_callback = callback
        
    def log(self, message: str) -> None:
        """输出日志"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def set_llm_processor(self, processor: Callable) -> None:
        """
        设置LLM处理器回调函数
        
        Args:
            processor: 一个可调用函数，用于处理LLM请求
        """
        self.llm_processor = processor
    
    def _initialize_files(self) -> None:
        """初始化日志文件"""
        # 初始化原始对话记录文件
        raw_data = {
            "dialogue_id": self.dialogue_id,
            "start_time": datetime.now().isoformat(),
            "history": []
        }
        with open(self.raw_log_file, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
            
        # 初始化记忆文件
        memory_data = {
            "dialogue_id": self.dialogue_id,
            "start_time": datetime.now().isoformat(),
            "memories": []
        }
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
    
    async def log_interaction(self, role: str, content: str, timestamp: float = None) -> None:
        """记录一次交互"""
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        # 创建交互记录
        interaction = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "time": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加到原始历史记录
        self.raw_history.append(interaction)
        
        # 更新文件
        await self._update_raw_log()
        
        # 增加对话计数
        if role == "user":
            self.conversation_count += 1
            
        # 每隔指定轮数，生成记忆总结
        if self.conversation_count >= self.memory_summary_interval:
            await self._generate_memory_summary()
            self.conversation_count = 0
    
    async def _update_raw_log(self) -> None:
        """更新原始对话记录文件"""
        try:
            with open(self.raw_log_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["history"] = self.raw_history
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.truncate()
        except Exception as e:
            self.log(f"[MEMORY] 更新原始记录失败: {str(e)}")
    
    async def _generate_memory_summary(self) -> None:
        """生成记忆总结"""
        try:
            # 检查是否设置了LLM处理器
            if not self.llm_processor:
                self.log("[MEMORY] 无法生成记忆总结：未设置LLM处理器")
                return
                
            # 获取最近的对话片段
            recent_dialogues = self.raw_history[-self.memory_summary_interval*2:]
            
            # 构建提示词
            prompt = "请总结以下对话的主要内容，提取关键信息和主题：\n\n"
            for item in recent_dialogues:
                role_name = "用户" if item["role"] == "user" else "助手"
                prompt += f"{role_name}: {item['content']}\n"
                
            # 调用LLM生成总结
            messages = [
                {"role": "system", "content": "你是一个对话总结专家，请提取对话中的关键信息和主题，生成简洁的总结。"},
                {"role": "user", "content": prompt}
            ]
            
            # 使用传入的LLM处理器生成摘要
            summary_response = await self.llm_processor(messages)
            summary = ""
            
            # 处理返回结果，可能是字符串或特定格式的对象
            if isinstance(summary_response, str):
                summary = summary_response
            elif isinstance(summary_response, dict) and "content" in summary_response:
                summary = summary_response["content"]
            elif isinstance(summary_response, dict) and "choices" in summary_response:
                summary = summary_response["choices"][0]["message"]["content"]
            else:
                # 尝试从其他可能的返回格式中提取
                summary = str(summary_response)
            
            # 创建记忆条目
            memory_entry = {
                "timestamp": datetime.now().timestamp(),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary": summary,
                "original_dialogues": recent_dialogues
            }
            
            # 添加到记忆历史
            self.memory_history.append(memory_entry)
            
            # 更新记忆文件
            await self._update_memory_file()
            
            self.log(f"[MEMORY] 成功生成记忆总结: {summary[:50]}...")
            
        except Exception as e:
            self.log(f"[MEMORY] 生成记忆总结失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
    
    async def _update_memory_file(self) -> None:
        """更新记忆文件"""
        try:
            with open(self.memory_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["memories"] = self.memory_history
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.truncate()
        except Exception as e:
            self.log(f"[MEMORY] 更新记忆文件失败: {str(e)}")
    
    def get_recent_memories(self, count: int = 5) -> List[Dict]:
        """获取最近的记忆"""
        return self.memory_history[-count:]
    
    def get_all_memories(self) -> List[Dict]:
        """获取所有记忆"""
        return self.memory_history
    
    def get_raw_history(self, count: Optional[int] = None) -> List[Dict]:
        """获取原始对话历史"""
        if count is None:
            return self.raw_history
        return self.raw_history[-count:]
    
    def get_formatted_context(self) -> str:
        """获取格式化的上下文信息"""
        context = "历史记忆摘要：\n"
        
        # 添加最近的记忆总结
        recent_memories = self.get_recent_memories(3)
        for memory in recent_memories:
            context += f"- {memory['summary']}\n"
            
        # 添加最近的原始对话
        context += "\n最近的对话：\n"
        recent_history = self.get_raw_history(6)  # 最近3轮对话
        for item in recent_history:
            role_name = "用户" if item["role"] == "user" else "助手"
            context += f"{role_name}: {item['content']}\n"
            
        return context
    
    def load_previous_session(self, session_id: str) -> bool:
        """
        加载之前的会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功加载
        """
        raw_log_file = os.path.join(self.raw_dir, f"dialogue_{session_id}.json")
        memory_file = os.path.join(self.memory_dir, f"memory_{session_id}.json")
        
        if not os.path.exists(raw_log_file) or not os.path.exists(memory_file):
            self.log(f"[MEMORY] 加载会话失败: 找不到会话 {session_id} 的文件")
            return False
            
        try:
            # 加载原始历史
            with open(raw_log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.raw_history = data.get("history", [])
                
            # 加载记忆摘要
            with open(memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.memory_history = data.get("memories", [])
                
            # 更新会话ID和文件路径
            self.dialogue_id = session_id
            self.raw_log_file = raw_log_file
            self.memory_file = memory_file
            
            self.log(f"[MEMORY] 成功加载会话 {session_id}")
            return True
            
        except Exception as e:
            self.log(f"[MEMORY] 加载会话失败: {str(e)}")
            return False
            
    def list_available_sessions(self) -> List[Dict]:
        """
        列出所有可用的会话
        
        Returns:
            会话信息列表，包含会话ID和开始时间
        """
        sessions = []
        
        try:
            for file_name in os.listdir(self.raw_dir):
                if file_name.startswith("dialogue_") and file_name.endswith(".json"):
                    session_id = file_name[9:-5]  # 提取会话ID
                    file_path = os.path.join(self.raw_dir, file_name)
                    
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        start_time = data.get("start_time", "")
                        
                        sessions.append({
                            "session_id": session_id,
                            "start_time": start_time,
                            "file_path": file_path
                        })
            
            # 按开始时间降序排序
            sessions.sort(key=lambda x: x["start_time"], reverse=True)
            
        except Exception as e:
            self.log(f"[MEMORY] 列出会话失败: {str(e)}")
            
        return sessions 