import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTextEdit, QLabel, QPushButton, 
                           QFrame, QSplitter, QStatusBar)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QPainter, QLinearGradient, QColor, QPalette
from dialogue.dialogue_system import DialogueSystem
import asyncio
import qasync
from functools import partial
import time
import pygame
import os

# 定义颜色常量
PRIMARY_COLOR = "#3B82F6"        # 主色调 - 天蓝色
SECONDARY_COLOR = "#8B5CF6"      # 次要色调 - 紫色
ACCENT_COLOR = "#10B981"         # 强调色 - 翠绿色
BG_COLOR = "#F9FAFB"             # 背景色 - 浅灰白
DARK_BG_COLOR = "#1E293B"        # 深色背景
TEXT_COLOR = "#1F2937"           # 文本色 - 深灰
LIGHT_TEXT_COLOR = "#6B7280"     # 浅色文本
ERROR_COLOR = "#EF4444"          # 错误色 - 红色
SUCCESS_COLOR = "#10B981"        # 成功色 - 绿色
WARNING_COLOR = "#F59E0B"        # 警告色 - 橙色
BORDER_COLOR = "#E5E7EB"         # 边框色 - 浅灰色

class LogTextEdit(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        
        # 设置字体
        font = QFont("Segoe UI", 10)
        self.setFont(font)
        
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 10px;
                padding: 12px;
            }}
            
            QScrollBar:vertical {{
                border: none;
                background: {BG_COLOR};
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }}
            
            QScrollBar::handle:vertical {{
                background: {PRIMARY_COLOR};
                min-height: 20px;
                border-radius: 5px;
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        
        self.document().setDefaultStyleSheet(f"""
            .system {{ color: {PRIMARY_COLOR}; font-weight: bold; }}
            .asr {{ color: {SECONDARY_COLOR}; }}
            .llm {{ color: {ACCENT_COLOR}; }}
            .tts {{ color: #047857; }}
            .video {{ color: #8B5CF6; }}
            .error {{ color: {ERROR_COLOR}; }}
            .success {{ color: {SUCCESS_COLOR}; }}
            .warning {{ color: {WARNING_COLOR}; }}
            .timestamp {{ color: {LIGHT_TEXT_COLOR}; font-size: 9px; }}
            .message {{ line-height: 1.5; margin-bottom: 4px; }}
        """)

    def append_log(self, text):
        """添加带样式的日志"""
        # 根据不同的日志类型设置不同的样式类
        style_class = "default"
        if "[SYSTEM]" in text:
            style_class = "system"
        elif "[ASR]" in text:
            style_class = "asr"
        elif "[LLM]" in text:
            style_class = "llm"
        elif "[TTS]" in text:
            style_class = "tts"
        elif "[VIDEO]" in text:
            style_class = "video"
        elif "错误" in text or "失败" in text:
            style_class = "error"
        elif "成功" in text:
            style_class = "success"
        elif "警告" in text:
            style_class = "warning"
            
        # 添加时间戳
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        
        # 格式化日志文本
        formatted_text = f'''
        <div class="message">
            <span class="timestamp">[{timestamp}]</span>
            <span class="{style_class}">{text}</span>
        </div>
        '''
        
        # 添加到文本框
        self.append(formatted_text)
        
        # 滚动到底部
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class StatusWidget(QFrame):
    """状态显示组件"""
    def __init__(self, title, initial_status="未启动"):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        
        self.setStyleSheet(f"""
            StatusWidget {{
                background-color: {BG_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 10px;
                padding: 4px;
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        
        # 标题
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_COLOR};
                font-weight: bold;
                font-size: 12px;
            }}
        """)
        
        # 状态指示灯
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(12, 12)
        self.status_indicator.setStyleSheet(f"""
            QLabel {{
                background-color: {LIGHT_TEXT_COLOR};
                border-radius: 6px;
                border: 1px solid {BORDER_COLOR};
            }}
        """)
        
        # 状态文本
        self.status_text = QLabel(initial_status)
        self.status_text.setStyleSheet(f"""
            QLabel {{
                color: {LIGHT_TEXT_COLOR};
                font-size: 12px;
            }}
        """)
        
        layout.addWidget(title_label)
        layout.addStretch()
        layout.addWidget(self.status_indicator)
        layout.addWidget(self.status_text)
        
    def update_status(self, status, color):
        """更新状态"""
        self.status_text.setText(status)
        self.status_text.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 12px;
            }}
        """)
        self.status_indicator.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                border-radius: 6px;
            }}
        """)

class VideoFrame(QFrame):
    """视频显示框架"""
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(480, 360)
        
        self.setStyleSheet(f"""
            VideoFrame {{
                background-color: {BG_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 15px;
                padding: 0px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 视频标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: {DARK_BG_COLOR};
                color: {LIGHT_TEXT_COLOR};
                border-radius: 15px;
                font-size: 14px;
            }}
        """)
        self.video_label.setText("摄像头未启动")
        
        layout.addWidget(self.video_label)

class TitleBar(QFrame):
    """自定义标题栏"""
    def __init__(self, title):
        super().__init__()
        self.setFixedHeight(60)
        self.setStyleSheet(f"""
            TitleBar {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                             stop:0 {PRIMARY_COLOR}, stop:1 {SECONDARY_COLOR});
                border-radius: 15px 15px 0 0;
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 15, 0)
        
        # Logo
        logo_label = QLabel()
        logo_label.setFixedSize(42, 42)
        # 如果有logo图片，可以设置
        # logo_label.setPixmap(QPixmap("path/to/logo.png").scaled(42, 42, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_label.setStyleSheet(f"""
            QLabel {{
                background-color: white;
                border-radius: 21px;
                color: {PRIMARY_COLOR};
                font-weight: bold;
                font-size: 16px;
            }}
        """)
        logo_label.setText("OMNI")
        logo_label.setAlignment(Qt.AlignCenter)
        
        # 标题
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        
        layout.addWidget(logo_label)
        layout.addWidget(title_label)
        layout.addStretch()

class StyleButton(QPushButton):
    """自定义样式按钮"""
    def __init__(self, text, color, hover_color, pressed_color):
        super().__init__(text)
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(120, 40)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
            QPushButton:disabled {{
                background-color: {LIGHT_TEXT_COLOR};
                color: {BG_COLOR};
            }}
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training free mllm")
        self.setMinimumSize(1280, 800)
        
        # 设置窗口图标
        # self.setWindowIcon(QIcon("path/to/icon.png"))
        
        # 初始化pygame mixer
        try:
            pygame.mixer.quit()  # 确保先关闭
            pygame.mixer.init(frequency=16000, size=-16, channels=1)
        except Exception as e:
            print(f"初始化音频系统错误: {str(e)}")
        
        # 设置应用程序级别的样式
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BG_COLOR};
            }}
            QSplitter::handle {{
                background: {BORDER_COLOR};
                width: 1px;
            }}
            QStatusBar {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
                border-top: 1px solid {BORDER_COLOR};
                padding: 5px;
            }}
            QWidget {{
                font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif;
            }}
        """)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 添加标题栏
        title_bar = TitleBar("Training-Free Multimodal Large Language Model Orchestration")
        main_layout.addWidget(title_bar)
        
        # 创建内容区域
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)
        
        # 左侧视频和状态区域
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 视频显示框
        self.video_frame = VideoFrame()
        left_layout.addWidget(self.video_frame)
        
        # 状态指示器区域
        status_container = QWidget()
        status_layout = QHBoxLayout(status_container)
        status_layout.setSpacing(10)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        # 添加不同的状态指示器
        self.video_status = StatusWidget("视频状态")
        self.asr_status = StatusWidget("语音识别")
        self.llm_status = StatusWidget("语言模型")
        
        status_layout.addWidget(self.video_status)
        status_layout.addWidget(self.asr_status)
        status_layout.addWidget(self.llm_status)
        
        left_layout.addWidget(status_container)
        
        # 右侧日志区域
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 添加日志标题
        log_title = QLabel("系统日志")
        log_title.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_COLOR};
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }}
        """)
        
        # 日志显示
        self.log_display = LogTextEdit()
        
        # 控制按钮
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(10)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.start_button = StyleButton("开始对话", PRIMARY_COLOR, "#4338CA", "#3730A3")
        self.stop_button = StyleButton("停止对话", ERROR_COLOR, "#DC2626", "#B91C1C")
        self.stop_button.setEnabled(False)
        
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        
        right_layout.addWidget(log_title)
        right_layout.addWidget(self.log_display)
        right_layout.addWidget(button_container)
        
        # 将左右两边添加到分割器
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        
        # 设置默认大小比例
        splitter.setSizes([500, 700])
        
        # 添加分割器到内容布局
        content_layout.addWidget(splitter)
        
        # 添加内容到主布局
        main_layout.addWidget(content_widget)
        
        # 添加状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('系统就绪，点击"开始对话"按钮启动')
        
        # 初始化对话系统
        self.dialogue_system = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        
        # 获取事件循环
        self.loop = asyncio.get_event_loop()
        
        # 连接信号
        self.start_button.clicked.connect(self._start_dialogue_wrapper)
        self.stop_button.clicked.connect(self._stop_dialogue_wrapper)

    def _start_dialogue_wrapper(self):
        """包装异步start_dialogue方法"""
        asyncio.create_task(self.start_dialogue())

    def _stop_dialogue_wrapper(self):
        """包装异步stop_dialogue方法"""
        asyncio.create_task(self.stop_dialogue())

    async def start_dialogue(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_display.append_log("[系统] 正在启动对话系统...")
        self.statusBar.showMessage('正在启动对话系统...')
        
        # 更新状态指示器
        self.video_status.update_status("正在启动", WARNING_COLOR)
        self.asr_status.update_status("正在启动", WARNING_COLOR)
        self.llm_status.update_status("正在初始化", WARNING_COLOR)
        
        self.dialogue_system = DialogueSystem()
        # 设置日志回调
        self.dialogue_system.set_log_callback(self.log_display.append_log)
        
        # 启动视频更新定时器
        self.video_timer.start(33)  # 约30fps
        
        # 启动对话系统
        await self.dialogue_system.run()

    async def stop_dialogue(self):
        if self.dialogue_system:
            self.video_timer.stop()
            
            # 更新状态指示器
            self.video_status.update_status("正在停止", WARNING_COLOR)
            self.asr_status.update_status("正在停止", WARNING_COLOR)
            self.llm_status.update_status("正在停止", WARNING_COLOR)
            
            self.statusBar.showMessage('正在停止对话系统...')
            
            await self.dialogue_system.stop()
            self.dialogue_system = None
            
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.log_display.append_log("[系统] 对话系统已停止")
            
            # 更新状态指示器
            self.video_status.update_status("已停止", LIGHT_TEXT_COLOR)
            self.asr_status.update_status("已停止", LIGHT_TEXT_COLOR)
            self.llm_status.update_status("已停止", LIGHT_TEXT_COLOR)
            
            self.statusBar.showMessage('对话系统已停止，点击"开始对话"按钮重新启动')
            
            # 清除视频显示
            self.video_frame.video_label.setText("摄像头已关闭")
            self.video_frame.video_label.setPixmap(QPixmap())

    def update_video_frame(self):
        if self.dialogue_system and hasattr(self.dialogue_system, 'video_processor'):
            frame = self.dialogue_system.video_processor.get_latest_frame(save_frame=False)
            if frame is not None:
                # 更新视频状态
                self.video_status.update_status("正在运行", SUCCESS_COLOR)
                self.asr_status.update_status("正在监听", SUCCESS_COLOR)
                self.llm_status.update_status("已就绪", SUCCESS_COLOR)
                
                # 状态栏更新
                self.statusBar.showMessage('系统正常运行中')
                
                # 转换BGR到RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = rgb_frame.shape[:2]
                bytes_per_line = 3 * width
                q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                # 获取视频标签尺寸
                label_size = self.video_frame.video_label.size()
                
                # 保持纵横比缩放
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # 设置视频帧
                self.video_frame.video_label.setPixmap(scaled_pixmap)
            else:
                self.video_status.update_status("无视频信号", ERROR_COLOR)
                self.video_frame.video_label.setText("无法获取视频画面")
                self.video_frame.video_label.setPixmap(QPixmap())

async def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 设置应用程序字体
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 创建事件循环
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    asyncio.run(main()) 