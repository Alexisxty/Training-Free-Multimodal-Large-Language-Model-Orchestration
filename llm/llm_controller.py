import time
import asyncio
from openai import OpenAI, APIConnectionError
import json
from utils.config import CONFIG
from llm.prompts import MAIN_DIALOGUE_PROMPT
from tts.tts_processor import TTSProcessor
from memory.memory_manager import MemoryManager
import httpx
import os

class LLMController:
    def __init__(self, video_processor):
        self.video_processor = video_processor
        self.conversation_history = []
        self.last_response_time = 0
        self.min_response_interval = 1.0  # 最小响应间隔（秒）
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 1.0  # 重试延迟（秒）
        
        # 初始化 TTS 处理器
        self.tts_processor = TTSProcessor()
        
        # 初始化记忆管理器
        self.dialogue_logger = MemoryManager()
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=f"{CONFIG['OPENAI_BASE_URL']}/v1",
            api_key=CONFIG['OPENAI_API_KEY'],
            timeout=10.0  # 设置超时时间
        )
        
        print(f"LLMController initialized with API URL: {CONFIG['OPENAI_BASE_URL']}")
        
    def _clean_message(self, message: str) -> str:
        """清理消息中的特殊标记"""
        markers = ["[S.SPEAK]", "[C.LISTEN]", "[S.LISTEN]", "[S.STOP]", "[C.SPEAK]"]
        cleaned = message
        for marker in markers:
            cleaned = cleaned.replace(marker, "")
        return cleaned.strip()

    def _is_similar_response(self, response1: str, response2: str) -> bool:
        """检查两个响应是否相似"""
        from difflib import SequenceMatcher
        # 清理特殊标记后比较
        clean1 = self._clean_message(response1)
        clean2 = self._clean_message(response2)
        
        similarity = SequenceMatcher(None, clean1, clean2).ratio()
        return similarity > 0.8

    async def _call_api_with_retry(self, messages, retry_count=0):
        """带重试机制的API调用"""
        try:
            # 发送请求到DeepSeek API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    'https://api.deepseek.com/v1/chat/completions',
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {os.getenv("DEEPSEEK_API_KEY")}'
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 500,
                        "presence_penalty": 0,
                        "frequency_penalty": 0,
                        "top_p": 1
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
            
        except httpx.HTTPError as e:
            if retry_count < self.max_retries:
                retry_count += 1
                wait_time = self.retry_delay * (2 ** (retry_count - 1))  # 指数退避
                print(f"\n[LLM] API连接错误，第{retry_count}次重试... (等待{wait_time:.1f}秒)")
                await asyncio.sleep(wait_time)
                return await self._call_api_with_retry(messages, retry_count)
            else:
                print("\n[LLM] API连接失败，已达到最大重试次数")
                raise e
                
        except Exception as e:
            print(f"\n[LLM] API调用出现其他错误: {str(e)}")
            raise e

    async def process_message(self, text: str) -> dict:
        try:
            current_time = time.time()
            
            # 检查消息间隔
            if current_time - self.last_response_time < self.min_response_interval:
                print("[LLM] 响应间隔过短，跳过处理")
                return {"state": "IDLE", "response": ""}
            
            # 更新响应时间
            self.last_response_time = current_time
            
            # 保存用户消息
            user_message = {
                "role": "user",
                "content": text,
                "timestamp": current_time
            }
            self.conversation_history.append(user_message)
            
            # 记录用户消息
            await self.dialogue_logger.log_interaction("user", text, current_time)
            
            # 准备发送给GPT的消息
            messages = [
                {"role": "system", "content": MAIN_DIALOGUE_PROMPT},
                *[{
                    "role": msg["role"],
                    "content": msg["content"]
                } for msg in self.conversation_history[-10:]]  # 只保留最近的10条消息
            ]
            
            print("\n发送到GPT的消息列表:")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            
            # 调用GPT API（带重试）
            print("\n[LLM] 发送请求到GPT API...")
            try:
                completion = await self._call_api_with_retry(messages)
                
                response_text = completion.choices[0].message.content
                print("\n[LLM] GPT响应:")
                print(response_text)
                
                # 检查是否与上一条响应相似
                if (len(self.conversation_history) >= 2 and 
                    self.conversation_history[-2]["role"] == "assistant" and
                    self._is_similar_response(self.conversation_history[-2]["content"], response_text)):
                    print("[LLM] 检测到相似响应，跳过")
                    return {"state": "IDLE", "response": ""}
                
                # 保存助手消息
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": time.time()
                }
                self.conversation_history.append(assistant_message)
                
                # 记录助手消息
                await self.dialogue_logger.log_interaction("assistant", response_text, assistant_message["timestamp"])
                
                # 解析响应状态
                state = self._parse_state(response_text)
                print(f"\n[LLM] 解析状态: {state}")
                
                cleaned_response = self._clean_message(response_text)
                print(f"\n[LLM] 清理后的响应: {cleaned_response}")
                
                # 使用TTS播放响应
                if cleaned_response:
                    print("[LLM] 正在使用TTS播放响应...")
                    await self.tts_processor.text_to_speech(cleaned_response)
                
                return {
                    "state": state,
                    "response": cleaned_response
                }
                
            except Exception as e:
                print(f"\n[LLM] API调用错误: {str(e)}")
                print(f"[LLM] 错误类型: {type(e).__name__}")
                return {"state": "ERROR", "response": f"API调用出错: {str(e)}"}
                
        except Exception as e:
            print(f"\n[LLM] 消息处理错误: {str(e)}")
            print(f"[LLM] 错误类型: {type(e).__name__}")
            return {"state": "ERROR", "response": f"处理出错了: {str(e)}"}

    def _parse_state(self, response: str) -> str:
        """解析响应状态"""
        if "[S.SPEAK]" in response:
            return "SPEAKING"
        elif "[C.LISTEN]" in response:
            return "LISTENING"
        elif "[S.STOP]" in response:
            return "STOPPED"
        return "IDLE"

    async def get_response(self, text: str) -> str:
        """
        获取对用户输入的响应
        Args:
            text: 用户输入的文本
        Returns:
            LLM的响应文本
        """
        result = await self.process_message(text)
        return result["response"]

    def clear_history(self):
        """清理对话历史"""
        self.conversation_history = []
        self.last_response_time = 0