import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from utils.config import CONFIG, TEMP_FILE_CONFIG

class DialogueLogger:
    def __init__(self, log_dir=None):
        """初始化对话历史记录器"""
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
        
        # 初始化文件
        self._initialize_files()
        
    def _initialize_files(self):
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
            
    async def log_interaction(self, role: str, content: str, timestamp: float = None):
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
            
    async def _update_raw_log(self):
        """更新原始对话记录文件"""
        try:
            with open(self.raw_log_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["history"] = self.raw_history
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.truncate()
        except Exception as e:
            print(f"[LOGGER] 更新原始记录失败: {str(e)}")
            
    async def _generate_memory_summary(self):
        """生成记忆总结"""
        try:
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
            
            # 这里需要从LLM管理器获取实例
            from llm_manager import LLMManager
            llm = LLMManager()
            summary = await llm._process_with_llm("main_dialogue", messages)
            
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
            
        except Exception as e:
            print(f"[LOGGER] 生成记忆总结失败: {str(e)}")
            
    async def _update_memory_file(self):
        """更新记忆文件"""
        try:
            with open(self.memory_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["memories"] = self.memory_history
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.truncate()
        except Exception as e:
            print(f"[LOGGER] 更新记忆文件失败: {str(e)}")
            
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