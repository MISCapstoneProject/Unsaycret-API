# utils/log_utils.py

import os
import sys
from datetime import datetime

# 為了還原用，放在模組層級變數
_original_stdout = None

class Tee:
    def __init__(self, log_file, mode="w"):
        global _original_stdout
        _original_stdout = sys.stdout  # 備份原始 stdout
        self.terminal = sys.stdout
        self.log = open(log_file, mode, encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def setup_logging(log_dir="logs", mode="w") -> str:
    """
    初始化 log 記錄，將 sys.stdout 輸出至 log 檔與 terminal 同步。
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{timestamp}.log")

    sys.stdout = Tee(log_path, mode)
    return log_path

def restore_stdout() -> None:
    """恢復原始標準輸出"""
    global _original_stdout
    if '_original_stdout' in globals() and _original_stdout is not None:
        sys.stdout = _original_stdout
