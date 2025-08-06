import os
import shutil
import datetime
from typing import List
from utils.config import TEMP_FILE_CONFIG

class FileManager:
    def __init__(self, session_id=None):
        """初始化文件管理器"""
        if session_id is None:
            session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
        self.session_id = session_id
        self.temp_root = TEMP_FILE_CONFIG['TEMP_ROOT_DIR']
        
        # 创建会话目录
        self.session_dir = os.path.join(self.temp_root, session_id)
        self.audio_dir = os.path.join(self.session_dir, TEMP_FILE_CONFIG['AUDIO_DIR'])
        self.video_dir = os.path.join(self.session_dir, TEMP_FILE_CONFIG['VIDEO_DIR'])
        self.debug_dir = os.path.join(self.session_dir, TEMP_FILE_CONFIG['DEBUG_DIR'])
        
        # 创建必要的目录
        for dir_path in [self.session_dir, self.audio_dir, self.video_dir, self.debug_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
        print(f"[FILE] 创建会话目录: {self.session_dir}")
        
    def get_temp_path(self, file_type: str, filename: str) -> str:
        """获取临时文件路径"""
        if file_type == 'audio':
            return os.path.join(self.audio_dir, filename)
        elif file_type == 'video':
            return os.path.join(self.video_dir, filename)
        elif file_type == 'debug':
            return os.path.join(self.debug_dir, filename)
        else:
            return os.path.join(self.session_dir, filename)
            
    def clean_session(self):
        """清理当前会话的所有临时文件"""
        if os.path.exists(self.session_dir):
            try:
                shutil.rmtree(self.session_dir)
                print(f"[FILE] 已清理会话目录: {self.session_dir}")
            except Exception as e:
                print(f"[FILE] 清理会话目录失败: {str(e)}")
                
    @staticmethod
    def clean_all_sessions():
        """清理所有会话的临时文件"""
        temp_root = TEMP_FILE_CONFIG['TEMP_ROOT_DIR']
        if os.path.exists(temp_root):
            try:
                shutil.rmtree(temp_root)
                print(f"[FILE] 已清理所有临时文件: {temp_root}")
            except Exception as e:
                print(f"[FILE] 清理临时文件失败: {str(e)}")
                
    @staticmethod
    def list_sessions():
        """列出所有会话"""
        temp_root = TEMP_FILE_CONFIG['TEMP_ROOT_DIR']
        if os.path.exists(temp_root):
            sessions = []
            for session_id in os.listdir(temp_root):
                session_path = os.path.join(temp_root, session_id)
                if os.path.isdir(session_path):
                    # 获取会话信息
                    session_info = {
                        'id': session_id,
                        'path': session_path,
                        'created': datetime.datetime.fromtimestamp(os.path.getctime(session_path)).strftime("%Y-%m-%d %H:%M:%S"),
                        'size': FileManager._get_dir_size(session_path)
                    }
                    sessions.append(session_info)
            return sessions
        return []
        
    @staticmethod
    def _get_dir_size(path: str) -> int:
        """获取目录大小（字节）"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        return total_size 