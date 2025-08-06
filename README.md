# training-free-mllm
<div align="center">

![training-free-mllm](https://img.shields.io/badge/training-free-mllm-blueviolet?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

<div align="center">
<p><b>先进的免训练多模态实时交互智能助手系统</b></p>
<p><i>Advanced Real-time Multimodal Interaction Assistant</i></p>
</div>

<p align="center">
  <img src="https://github.com/siliconflow/cosyvoice2/raw/main/images/logo.jpg" width="240" alt="training-free-mllm Logo">
</p>


## ✨ 核心功能

<div align="center">

| 功能模块 | 描述 |
|---------|------|
| 🎤 **语音识别** | 实时语音输入，支持动态降噪和多轮对话 |
| 👁️ **视觉理解** | 摄像头实时图像分析，支持物体识别与场景理解 |
| 🧠 **自然语言处理** | 基于大型语言模型的对话理解与生成 |
| 🔊 **语音合成** | 高质量、高自然度的语音输出，支持情感表达 |
| 🔄 **全双工交互** | 支持实时语音打断与响应，模拟自然人际对话 |
| 📝 **对话记忆** | 长期记忆与上下文理解，支持复杂对话逻辑 |
| 🛠️ **多模式部署** | 支持 GUI 与 CLI 两种运行模式 |

</div>

## 🔌 系统架构

training-free-mllm 采用模块化设计，各组件松耦合高内聚，实现了高效的数据流与控制流。

<div align="center">
<img src="https://mermaid.ink/img/pako:eNqFkl9LwzAUxb_KJc-CMFkH4uwhVAWfhOGb-KDF9npDTa4kKdbRb2-70m27D3tJTu755Sf3ZkI7ZiElUmOD8LrGdA17bwpwVKHYkG_RaMHe0FTgxvMK-iqkFcKCrRroDLaotU0V-Cx_kEVR6c4i9aSK3HhT0-4eMjQFbJlDKnGntd0aDGACT6qCnRVauTYkTdNCSwV0xgbpPCZZVdvXpkdZd-GEwKNSRkGDZG3IwKHCPNp_JtO0SnqBnD9qCnkXfXjEdKpNQPexiT5jlk6_t1k_W7rphL6M_Ui2-1WehyNXh0F_vV9mGR9l2XJ6yMZ4lM1nR5aGnL-n36Zs9XK_mN1eXO_eXhZfAsvdK8GgDo6MppvZyLSU-NrYmKSk2HpS2hZ72g_SnhqpGSohp-j3pJyWBCWbXZDSbY1LYh78YClQ7Vf9JEnptpk5JZdL5iJ1QTr4GCiNp_ZsRpz4AaA2pxg?type=png" alt="Architecture Diagram">
</div>

### 主要组件

* **对话系统 (Dialogue System)**: 整体控制中心，协调各模块的工作流
* **LLM 管理器 (LLM Manager)**: 负责大语言模型的请求和响应处理(包括控制器LLM、视觉、QVQ LLM)
* **ASR 处理器 (ASR Processor)**: 处理语音转文本，支持实时降噪和语音活动检测
* **TTS 处理器 (TTS Processor)**: 将文本转换为自然语音，支持多种引擎和声音定制
* **视频处理器 (Video Processor)**: 处理摄像头输入，支持实时图像分析
* **GUI 界面 (GUI Interface)**: 基于 PyQt5 的现代化用户界面

## 🚀 快速开始

### 系统要求

* Python 3.10+
* Windows/Linux/macOS

### 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/Alexisxty/traninfree-omni.git
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **配置环境**

```bash
# 复制并编辑环境变量文件
cp .env.example .env
# 编辑 .env 文件，填入您的 API 密钥
```

4. **启动系统**

```bash
# 图形界面模式
python main.py --mode=gui

# 命令行模式
python main.py --mode=cli
```

## 🧩 核心功能详解

### 智能对话系统

training-free-mllm 的对话系统基于大型语言模型，具有出色的自然语言理解和生成能力。系统采用多级 LLM 架构：

* **主对话 LLM**: 负责核心对话管理与响应生成
* **视觉 LLM**: 处理与视觉相关的多模态理解任务
* **QVQ LLM**: 处理复杂的视觉问答和推理任务

### 全双工语音交互

系统实现了类似人类对话的全双工交互体验：

* **实时打断**: 当用户开始说话时，系统能够自动停止当前输出并聆听
* **流式响应**: LLM 的响应以流式方式传输，实现边思考边回答的体验
* **语音活动检测**: 智能检测用户何时开始和结束说话

### 多模态视觉理解

系统整合了先进的视觉理解能力：

* **实时视频分析**: 捕获环境中的视觉信息
* **物体识别与跟踪**: 识别环境中的物体并进行跟踪
* **视觉问答**: 回答关于视觉内容的问题
* **场景理解**: 理解复杂的视觉场景和上下文

### 高质量语音合成

支持多种语音合成引擎，实现自然流畅的语音输出：

* **Edge TTS**: Microsoft 的高质量语音合成服务
* **CosyVoice2**: 先进的神经网络语音合成，支持丰富的情感表达
* **GPUStack TTS**: 高性能本地部署语音合成引擎

## 📊 技术规格

<div align="center">

| 模块 | 技术实现 | 特性 |
|------|----------|------|
| **LLM 引擎** | Qwen 2.5-32B, Qwen 2-VL-72B | 支持流式响应、上下文记忆、多轮对话 |
| **语音识别** | FunAudioLLM/SenseVoiceSmall | 实时转录、降噪、VAD 活动检测 |
| **语音合成** | EdgeTTS、CosyVoice2、GPUStack | 多引擎支持、情感表达、自然停顿 |
| **视觉处理** | OpenCV、Qwen-VL、QVQ | 实时视频分析、多模态理解、视觉问答 |
| **用户界面** | PyQt5、qasync | 响应式设计、异步处理、多线程优化 |
| **内存系统** | 结构化存储、上下文管理 | 长期记忆、会话状态保持、知识整合 |

</div>

## 🔧 高级配置

系统支持丰富的配置选项，可通过 `.env` 文件自定义：

* **语言模型参数**: 温度、最大长度、采样策略等
* **语音合成设置**: 声音选择、语速、音量等
* **ASR 参数**: 语言模型、采样率、降噪强度等
* **系统行为**: 调试模式、响应策略、记忆长度等

## 🤝 贡献指南

我们欢迎各种形式的贡献，包括功能改进、文档完善、错误修复等。请遵循以下步骤：

1. Fork 项目并创建您的分支
2. 实现您的修改并添加测试
3. 确保所有测试通过
4. 提交 Pull Request

## 📜 许可证

本项目采用 [MIT 许可证](LICENSE) 进行授权。

## 🔗 相关链接

* [SiliconFlow](https://www.siliconflow.cn/) - 提供 Qwen 模型和 CosyVoice 语音合成
* [Qwen AI](https://github.com/QwenLM/Qwen) - Qwen 大型语言模型
* [CosyVoice2](https://github.com/siliconflow/cosyvoice2) - 高质量语音合成引擎
* [FunASR](https://github.com/FunASR/FunASR) - 先进的语音识别系统

<div align="center">
<p>Powered by Intelligent Technology</p>
<p>Made with ❤️ by AI Enthusiasts</p>
</div>

