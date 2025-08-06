import asyncio
from openai import AsyncOpenAI
import json
import base64
from utils.config import CONFIG, LLMBackendType, LLM_CONFIGS, TEMP_FILE_CONFIG
from llm.prompts import MAIN_DIALOGUE_PROMPT, VISION_DIALOGUE_PROMPT, VQA_DIALOGUE_PROMPT
import cv2
import httpx
import os
from datetime import datetime

class LLMManager:
    def __init__(self):
        # 从配置模块中直接获取LLM配置
        self.llm_configs = LLM_CONFIGS
        self.clients = {}
        self._initialize_clients()
        self.latest_response = None
        self.model_state = "text"  # 'text' or 'vision' or 'vqa'
        self.video_processor = None
        self.is_processing_vision = False
        self.is_processing_vqa = False
        self.vision_process_lock = asyncio.Lock()
        self.vqa_process_lock = asyncio.Lock()
        self.dialogue_system = None  # 添加对话系统引用
        self.log_callback = None
        self.use_streaming = CONFIG.get('USE_STREAMING', True)  # 是否使用流式输出，默认开启
        
        # 打印当前配置用于调试
        for name, config in self.llm_configs.items():
            print(f"[LLM] {name} 配置:")
            print(f"  - 模型: {config.model_name}")

    def _initialize_clients(self):
        """初始化不同类型的LLM客户端"""
        for llm_name, config in self.llm_configs.items():
            if config.backend_type == LLMBackendType.THIRD_PARTY_API:
                # 对于第三方API，我们将使用httpx直接发送请求
                self.clients[llm_name] = None
            elif config.backend_type == LLMBackendType.VLLM:
                self.clients[llm_name] = httpx.AsyncClient(
                    base_url=config.api_base or "http://localhost:8000",
                    timeout=30.0
                )
                
    def set_video_processor(self, video_processor):
        """设置视频处理器"""
        self.video_processor = video_processor
        
    async def _call_third_party_api(self, client, messages, config):
        """调用第三方API"""
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                # 构建请求头
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.api_key}"
                }
                
                # 构建请求数据
                if "qvq" in config.model_name.lower():
                    # QVQ模型的特殊请求格式
                    data = {
                        "model": config.model_name,
                        "messages": messages,
                        "stream": False,  # QVQ 模型暂不支持流式
                        "max_tokens": 512,
                        "stop": ["null"],
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": 50,
                        "frequency_penalty": config.frequency_penalty,
                        "n": 1,
                        "response_format": {"type": "text"}
                    }
                elif "qwen" in config.model_name.lower() and "vl" in config.model_name.lower():
                    # Qwen VL模型的请求格式
                    data = {
                        "model": config.model_name,
                        "messages": messages,
                        "stream": False,  # 视觉模型暂不使用流式
                        "max_tokens": config.max_tokens,
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": 50,
                        "frequency_penalty": config.frequency_penalty,
                        "n": 1,
                        "response_format": {"type": "text"}
                    }
                else:
                    # 其他模型的标准请求格式
                    data = {
                        "model": config.model_name,
                        "messages": messages,
                        "stream": False,  # 非流式模式
                        "max_tokens": config.max_tokens,
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "frequency_penalty": config.frequency_penalty,
                    }
                
                # 使用httpx直接发送请求
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    # 根据不同的模型确定API URL
                    api_url = config.api_base.rstrip('/')
                    if "deepseek" in config.model_name.lower():
                        if not api_url.endswith('/chat/completions'):
                            api_url = f"{api_url}/chat/completions"
                    elif "qwen" in config.model_name.lower():
                        if not api_url.endswith('/chat/completions'):
                            api_url = f"{api_url}/chat/completions"
                    
                    try:
                        response = await http_client.post(
                            api_url,
                            json=data,
                            headers=headers,
                            timeout=30.0
                        )
                        response.raise_for_status()
                        return response.json()
                        
                    except httpx.TimeoutException as e:
                        print(f"[LLM] 请求超时: {str(e)}")
                        if retry < max_retries - 1:
                            await asyncio.sleep(retry_delay * (2 ** retry))
                            continue
                        raise Exception(f"API请求超时: {str(e)}")
                        
                    except httpx.HTTPStatusError as e:
                        print(f"[LLM] HTTP错误 {e.response.status_code}: {e.response.text}")
                        if retry < max_retries - 1 and e.response.status_code in [429, 500, 502, 503, 504]:
                            await asyncio.sleep(retry_delay * (2 ** retry))
                            continue
                        raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
                        
            except Exception as e:
                print(f"[LLM] API调用错误 (尝试 {retry + 1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** retry))
                    continue
                raise

    async def _call_third_party_api_stream(self, client, messages, config):
        """调用第三方API的流式版本"""
        max_retries = 3
        retry_delay = 1.0
        full_response = ""
        buffer = ""
        
        # 定义分隔符
        separators = ['。', '！', '？', '；', '!', '?', ';', '，', ',']
        
        for retry in range(max_retries):
            try:
                # 构建请求头
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.api_key}"
                }
                
                # 构建请求数据 - 启用流式输出
                data = {
                    "model": config.model_name,
                    "messages": messages,
                    "stream": True,  # 启用流式输出
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "frequency_penalty": config.frequency_penalty,
                }
                
                # 使用httpx直接发送流式请求
                async with httpx.AsyncClient(timeout=60.0) as http_client:
                    # 根据不同的模型确定API URL
                    api_url = config.api_base.rstrip('/')
                    if "deepseek" in config.model_name.lower():
                        if not api_url.endswith('/chat/completions'):
                            api_url = f"{api_url}/chat/completions"
                    elif "qwen" in config.model_name.lower():
                        if not api_url.endswith('/chat/completions'):
                            api_url = f"{api_url}/chat/completions"
                    
                    try:
                        self.log("[LLM] 开始流式请求...")
                        
                        # 增加缓冲区大小和超时设置
                        http_client.timeout = httpx.Timeout(60.0)
                        
                        # 发送请求并获取流
                        request = http_client.build_request(
                            method="POST",
                            url=api_url,
                            json=data,
                            headers=headers
                        )
                        response = await http_client.send(request, stream=True)
                        response.raise_for_status()
                        
                        # 临时缓存，用于合并不完整的UTF-8序列
                        partial_utf8 = b''
                        # JSON解析缓冲区
                        accumulated_chunk = ""
                            
                        # 处理返回的事件流
                        async for chunk in response.aiter_bytes(1024):
                            try:
                                # 处理不完整的UTF-8序列
                                if partial_utf8:
                                    chunk = partial_utf8 + chunk
                                    partial_utf8 = b''
                                
                                # 尝试解码为UTF-8
                                try:
                                    chunk_str = chunk.decode('utf-8')
                                except UnicodeDecodeError:
                                    # 如果解码失败，可能是因为最后几个字节是不完整的UTF-8序列
                                    # 尝试找到最后一个完整的UTF-8字符
                                    for i in range(1, min(4, len(chunk)) + 1):
                                        try:
                                            chunk_str = chunk[:-i].decode('utf-8')
                                            # 成功解码，保存剩余的字节到下一次迭代
                                            partial_utf8 = chunk[-i:]
                                            break
                                        except UnicodeDecodeError:
                                            continue
                                    else:
                                        # 如果所有尝试都失败，使用replace模式
                                        chunk_str = chunk.decode('utf-8', errors='replace')
                                        partial_utf8 = b''
                                
                                accumulated_chunk += chunk_str
                                
                                # 按行处理JSON数据
                                lines = accumulated_chunk.split("\n")
                                # 最后一行可能不完整，保留到下一次处理
                                accumulated_chunk = lines.pop()
                                
                                for line in lines:
                                    line = line.strip()
                                    if not line:
                                        continue
                                        
                                    if line.startswith("data: "):
                                        line = line[6:]  # 移除 "data: " 前缀
                                        
                                        if line.strip() == "[DONE]":
                                            # 流结束
                                            continue
                                            
                                        try:
                                            chunk_data = json.loads(line)
                                            if "choices" in chunk_data and chunk_data["choices"] and "delta" in chunk_data["choices"][0]:
                                                delta = chunk_data["choices"][0]["delta"]
                                                if "content" in delta and delta["content"]:
                                                    content = delta["content"]
                                                    
                                                    # 清理内容中的无效字符
                                                    content = ''.join(char for char in content if ord(char) > 31 or char in '\n\t')
                                                    content = content.replace('\ufffd', '')
                                                    
                                                    # 累积到完整响应
                                                    full_response += content
                                                    
                                                    # 累积到缓冲区
                                                    buffer += content
                                                    
                                                    # 分析缓冲区，当积累足够的内容或遇到分隔符时处理
                                                    should_process = False
                                                    # 当有分隔符或积累了足够多的字符时处理
                                                    if len(buffer) >= 5:
                                                        for sep in separators:
                                                            if sep in buffer:
                                                                should_process = True
                                                                break
                                                        
                                                        # 即使没有分隔符，如果缓冲区足够大也处理
                                                        if len(buffer) >= 20:
                                                            should_process = True
                                                    
                                                    # 如果应该处理，发送到对话系统
                                                    if should_process:
                                                        if self.dialogue_system:
                                                            # 使用缓冲区中的所有内容
                                                            await self.dialogue_system.process_streaming_text(buffer)
                                                            buffer = ""  # 重置缓冲区
                                        except json.JSONDecodeError:
                                            self.log(f"[LLM] JSON解析错误: {line}")
                                            continue
                            except Exception as e:
                                self.log(f"[LLM] 流处理错误: {str(e)}")
                                # 清空积累的数据，避免错误累积
                                accumulated_chunk = ""
                                continue
                        
                        # 处理缓冲区中剩余的文本
                        if buffer:
                            if self.dialogue_system:
                                await self.dialogue_system.process_streaming_text(buffer)
                                    
                        return {"choices": [{"message": {"content": full_response}}]}
                        
                    except httpx.TimeoutException as e:
                        self.log(f"[LLM] 流式请求超时: {str(e)}")
                        if retry < max_retries - 1:
                            await asyncio.sleep(retry_delay * (2 ** retry))
                            continue
                        raise Exception(f"流式API请求超时: {str(e)}")
                        
                    except httpx.HTTPStatusError as e:
                        self.log(f"[LLM] 流式HTTP错误 {e.response.status_code}: {e.response.text}")
                        if retry < max_retries - 1 and e.response.status_code in [429, 500, 502, 503, 504]:
                            await asyncio.sleep(retry_delay * (2 ** retry))
                            continue
                        raise Exception(f"流式HTTP {e.response.status_code}: {e.response.text}")
                        
            except Exception as e:
                self.log(f"[LLM] 流式API调用错误 (尝试 {retry + 1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** retry))
                    continue
                raise

    async def _call_vllm(self, client, messages, config):
        """调用vLLM"""
        response = await client.post("/v1/chat/completions", json={
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        })
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def _process_with_llm(self, llm_name: str, messages: list) -> str:
        """使用指定的LLM处理消息"""
        try:
            config = self.llm_configs[llm_name]
            client = self.clients[llm_name]
            
            if config.backend_type == LLMBackendType.THIRD_PARTY_API:
                # 判断是否使用流式输出
                if self.use_streaming and llm_name == "main_dialogue":
                    return await self._call_third_party_api_stream(client, messages, config)
                else:
                    return await self._call_third_party_api(client, messages, config)
            elif config.backend_type == LLMBackendType.VLLM:
                return await self._call_vllm(client, messages, config)
            else:
                raise ValueError(f"不支持的LLM后端类型: {config.backend_type}")
                
        except Exception as e:
            print(f"[LLM] {llm_name} 调用出错: {str(e)}")
            raise

    async def process_dialogue(self, text: str, context: str = "") -> str:
        """处理对话"""
        try:
            self.log("[LLM] 正在处理对话...")
            # 首先使用主对话模型处理
            response = await self._process_main_dialogue(text, context)
            
            # 检查是否需要视觉处理
            if "[NEED_VISION]" in response:
                return await self._process_vision_dialogue(text, context)
            
            # 检查是否需要视觉推理
            if "[NEED_VQA]" in response:
                return await self._process_vqa_dialogue(text, context)
            
            return response
            
        except Exception as e:
            print(f"[LLM] 对话处理错误: {str(e)}")
            return "[S.SPEAK]抱歉，处理您的请求时出现了问题，请重试。"

    async def _process_main_dialogue(self, text: str, context: str) -> str:
        """使用主对话模型处理文本"""
        try:
            self.log("[LLM] 使用主对话模型处理...")
            # 构建系统提示词
            system_prompt = MAIN_DIALOGUE_PROMPT + "\n\n当前对话上下文：\n" + context
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            
            # 如果是视觉相关的问题，直接转到视觉处理
            if any(keyword in text for keyword in ["看到", "画面", "视觉", "图像", "照片", "图片"]):
                print("[LLM] 检测到视觉相关问题，转到视觉处理")
                return await self._process_vision_dialogue(text, context)
            
            result = await self._process_with_llm("main_dialogue", messages)
            
            # 处理返回的结果
            if isinstance(result, dict) and 'choices' in result:
                response = result['choices'][0]['message']['content']
            elif isinstance(result, str):
                response = result
            else:
                raise ValueError("无效的响应格式")
            
            # 如果LLM判断需要视觉理解
            if "[NEED_VISION]" in response:
                print("[LLM] LLM请求视觉理解")
                return await self._process_vision_dialogue(text, context)
                
            # 如果LLM判断需要视觉推理
            if "[NEED_VQA]" in response:
                print("[LLM] LLM请求视觉推理")
                return await self._process_vqa_dialogue(text, context)
                
            return response
            
        except Exception as e:
            self.log(f"[LLM] 主对话处理错误: {str(e)}")
            return "[S.SPEAK]抱歉，处理您的请求时出现了问题，请重试。"
            
    async def _process_vision_dialogue(self, text: str, context: str) -> str:
        """使用视觉对话模型处理文本"""
        try:
            self.log("[LLM] 使用视觉对话模型处理...")
            await asyncio.wait_for(self.vision_process_lock.acquire(), timeout=1.0)
            
            if self.is_processing_vision:
                self.vision_process_lock.release()
                self.log("[LLM] 正在处理其他视觉请求，跳过")
                return "[S.SPEAK]请稍等，我正在处理上一个视觉请求。"
                
            try:
                self.is_processing_vision = True
                self.model_state = "vision"
                
                # 先给出中间反馈
                self.log("[LLM] 发送中间反馈")
                await self._send_intermediate_response("请让我看看...")
                
                if not self.video_processor:
                    self.log("[LLM] 视频处理器未初始化")
                    return "[S.SPEAK]抱歉，我现在看不到画面，请用文字描述您的问题。"
                    
                # 获取最新的视觉帧并压缩以提高性能
                frame = self.video_processor.get_latest_frame()
                if frame is None:
                    self.log("[LLM] 无法获取视觉帧，回退到文本模式")
                    return "[S.SPEAK]抱歉，我现在看不到画面，请用文字描述您的问题。"
                    
                # 压缩图像以减少处理时间和API开销
                frame_height, frame_width = frame.shape[:2]
                if frame_width > 800 or frame_height > 600:
                    # 保持纵横比缩小图像
                    scale = min(800 / frame_width, 600 / frame_height)
                    new_width = int(frame_width * scale)
                    new_height = int(frame_height * scale)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    self.log(f"[LLM] 图像已压缩: {frame_width}x{frame_height} -> {new_width}x{new_height}")
                
                # 转换图像质量降低为90%以节省带宽
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 保存当前帧用于调试
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_dir = os.path.join(TEMP_FILE_CONFIG['TEMP_ROOT_DIR'], TEMP_FILE_CONFIG['VIDEO_FRAMES_DIR'], timestamp)
                os.makedirs(frame_dir, exist_ok=True)
                frame_path = os.path.join(frame_dir, f"frame_{timestamp}_{str(datetime.now().microsecond).zfill(6)}.jpg")
                self.log(f"[VIDEO] 保存视觉帧: {frame_path}")
                cv2.imwrite(frame_path, frame)
                
                # 构建系统提示词
                system_prompt = VISION_DIALOGUE_PROMPT + "\n\n当前对话上下文：\n" + context
                
                # 构建消息
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
                
                # 调用视觉模型
                config = self.llm_configs.get('vision_dialogue')
                if not config:
                    self.log("[LLM] 视觉模型未配置")
                    return "[S.SPEAK]抱歉，视觉模型未配置。"
                    
                self.log(f"[LLM] 开始调用视觉模型: {config.model_name}")
                
                # 设置更长的超时时间，但加上timeout以避免长时间等待
                try:
                    response = await asyncio.wait_for(
                        self._call_third_party_api(None, messages, config),
                        timeout=10.0  # 10秒超时
                    )
                except asyncio.TimeoutError:
                    self.log("[LLM] 视觉模型调用超时")
                    return "[S.SPEAK]抱歉，视觉处理超时，请重试或者描述您看到的内容。"
                except httpx.HTTPStatusError as api_error:
                    self.log(f"[LLM] 视觉API调用HTTP错误: {str(api_error)}")
                    self.log(f"[LLM] HTTP状态码: {api_error.response.status_code}")
                    self.log(f"[LLM] 响应内容: {api_error.response.text}")
                    return "[S.SPEAK]抱歉，视觉模型暂时无法访问，请稍后再试。错误代码：" + str(api_error.response.status_code)
                except Exception as api_error:
                    self.log(f"[LLM] 视觉API调用错误: {str(api_error)}")
                    return "[S.SPEAK]抱歉，视觉模型调用出错，请稍后再试。"
                
                # 处理返回的结果
                if isinstance(response, dict) and 'choices' in response:
                    content = response['choices'][0]['message']['content']
                    self.log(f"[LLM] 视觉模型响应内容: {content}")
                elif isinstance(response, str):
                    content = response
                    self.log(f"[LLM] 视觉模型响应字符串: {content}")
                else:
                    self.log(f"[LLM] 无效的响应格式: {type(response)}")
                    return "[S.SPEAK]抱歉，视觉模型返回了无效的响应格式。"

                if not content.startswith("[S.SPEAK]") and not content.startswith("[S.STOP]"):
                    content = f"[S.SPEAK]{content}"
                    
                self.log(f"[LLM] 最终响应: {content}")
                return content
                
            except Exception as e:
                self.log(f"[LLM] 视觉对话处理错误: {str(e)}")
                if isinstance(e, httpx.HTTPStatusError):
                    self.log(f"[LLM] 响应内容: {e.response.text}")
                return "[S.SPEAK]抱歉，处理视觉请求时出现了问题，请重试。"
            finally:
                self.is_processing_vision = False
                self.model_state = "text"
                self.vision_process_lock.release()
                
        except asyncio.TimeoutError:
            self.log("[LLM] 等待视觉处理锁超时")
            return "[S.SPEAK]系统正忙，请稍后再试。"
            
        except Exception as e:
            self.log(f"[LLM] 视觉处理锁错误: {str(e)}")
            return "[S.SPEAK]抱歉，系统暂时无法处理您的请求。"

    async def _process_vqa_dialogue(self, text: str, context: str = "") -> str:
        """处理视觉推理对话"""
        async with self.vqa_process_lock:
            try:
                self.is_processing_vqa = True
                self.model_state = "vqa"
                
                # 先给出中间反馈
                print("[LLM] 发送中间反馈")
                await self._send_intermediate_response("请让我思考一下...")
                
                # 获取当前图像
                if not self.video_processor:
                    return "[S.SPEAK]抱歉，视频处理器未初始化。"
                    
                frame = self.video_processor.get_latest_frame()
                if frame is None:
                    return "[S.SPEAK]抱歉，我没有获取到图像。"
                    
                # 将图像编码为base64
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 构建系统提示词
                system_prompt = VQA_DIALOGUE_PROMPT + "\n\n当前对话上下文：\n" + context
                
                # 构建消息
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
                
                # 调用QVQ模型
                config = self.llm_configs.get('qvq')
                if not config:
                    return "[S.SPEAK]抱歉，视觉推理模型未配置。"
                    
                response = await self._call_third_party_api(None, messages, config)
                
                # 处理返回的结果
                if isinstance(response, dict) and 'choices' in response:
                    content = response['choices'][0]['message']['content']
                elif isinstance(response, str):
                    content = response
                else:
                    raise ValueError("无效的响应格式")
                
                if not content.startswith("[S.SPEAK]") and not content.startswith("[S.STOP]"):
                    content = f"[S.SPEAK]{content}"
                    
                return content
                
            except Exception as e:
                print(f"[ERROR] 视觉推理处理错误: {str(e)}")
                return f"[S.SPEAK]抱歉，视觉推理处理出现错误：{str(e)}"
            finally:
                self.is_processing_vqa = False
                self.model_state = "text"

    def set_dialogue_system(self, dialogue_system):
        """设置对话系统引用"""
        self.dialogue_system = dialogue_system

    async def _send_intermediate_response(self, message: str):
        """发送中间反馈并触发TTS播放"""
        try:
            print(f"[LLM] 中间反馈: {message}")
            if self.dialogue_system:
                # 构造标准格式的响应
                response = f"[S.SPEAK]{message}"
                # 通过对话系统处理响应
                await self.dialogue_system.process_llm_response(response)
            return f"[S.SPEAK]{message}"
        except Exception as e:
            print(f"[LLM] 发送中间反馈失败: {str(e)}")
            return None

    def get_latest_response(self) -> str:
        return self.latest_response
        
    async def close(self):
        """关闭LLM管理器"""
        # 关闭所有客户端
        for client in self.clients.values():
            if isinstance(client, httpx.AsyncClient):
                await client.aclose()
        self.video_processor = None
        self.video_processor = None

    def set_log_callback(self, callback):
        """设置日志回调函数"""
        self.log_callback = callback
        
    def log(self, message):
        """输出日志"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)