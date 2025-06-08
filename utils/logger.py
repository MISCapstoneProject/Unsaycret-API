"""
===============================================================================
Voice_ID: 集中式日誌系統
===============================================================================

這個模組提供了一個集中式的日誌系統，可被專案中所有模組使用。
主要功能包括:
- 統一的日誌格式和顏色設置
- 模組識別：在日誌中包含模組名稱
- 彈性文件處理：支援輸出到不同的日誌檔案
- 單例模式：確保每個命名的日誌器只創建一次

使用方式:
-----------
1. 在任何模組中導入:
   ```python
   from VID_logger import get_logger
   ```

2. 獲取或創建一個配置好的日誌器:
   ```python
   # 使用默認設置
   logger = get_logger("模組名稱")
   
   # 自訂配置
   logger = get_logger(
       name="模組名稱",
       log_file="custom_log.log",
       console_level=logging.INFO,
       file_level=logging.DEBUG,
       append_mode=True
   )
   ```

3. 使用日誌器:
   ```python
   logger.debug("除錯訊息")
   logger.info("資訊訊息")
   logger.warning("警告訊息")
   logger.error("錯誤訊息")
   logger.critical("嚴重錯誤訊息")
   ```

高級用法:
-----------
- 使用簡化輸出 (無前綴):
  ```python
  logger.info("簡潔訊息", extra={"simple": True})
  ```
  
- 獲取現有的日誌器:
  ```python
  # 若日誌器已存在，直接返回，不會更改配置
  existing_logger = get_logger("已存在的模組名稱")
  ```

作者: CYouuu
最後更新: 2025-05-16
===============================================================================
"""

import logging
import sys
from typing import Dict, Optional, Any

# 保存已配置的日誌器的全域字典
_LOGGERS: Dict[str, logging.Logger] = {}

class CustomFormatter(logging.Formatter):
    """自訂日誌格式，根據日誌級別使用不同顏色"""
    
    # 不同日誌級別的顏色代碼
    COLORS = {
        logging.DEBUG: '\033[94m',     # 藍色
        logging.INFO: '\033[92m',      # 綠色
        logging.WARNING: '\033[93m',   # 黃色
        logging.ERROR: '\033[91m',     # 紅色
        logging.CRITICAL: '\033[41m',  # 紅底
    }
    RESET = '\033[0m'  # 結束顏色
    
    def __init__(self, use_color: bool = True, module_name: str = None) -> None:
        """
        初始化格式器
        
        Args:
            use_color: 是否使用顏色輸出
            module_name: 模組名稱，用於在日誌中標識
        """
        super().__init__(fmt='%(message)s', datefmt='%H:%M:%S')
        self.use_color = use_color
        self.module_name = module_name
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日誌記錄
        
        Args:
            record: 日誌記錄物件
            
        Returns:
            str: 格式化後的日誌字串
        """
        # 根據時間和級別生成前綴
        if hasattr(record, 'simple') and record.simple:
            # 簡單輸出，無前綴
            log_fmt = '%(message)s'
        else:
            # 標準輸出，添加時間、模組和級別
            module_part = f"[{self.module_name}] " if self.module_name else ""
            log_fmt = f'[%(asctime)s] {module_part}%(levelname)s: %(message)s'
        
        # 設置格式字串
        self._style._fmt = log_fmt
        
        # 如果使用顏色，則為不同級別設置顏色
        if self.use_color and not (hasattr(record, 'simple') and record.simple):
            color = self.COLORS.get(record.levelno, '')
            reset = self.RESET
            record.levelname = f"{color}{record.levelname}{reset}"
        
        result = super().format(record)
        return result

def get_logger(
    name: str = "test",
    log_file: Optional[str] = "system_output.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    append_mode: bool = True
) -> logging.Logger:
    """
    獲取或創建一個配置好的日誌器
    
    Args:
        name: 日誌器名稱，用於區分不同模組
        log_file: 日誌文件路徑，若為None則只輸出到控制台
        console_level: 控制台輸出的日誌級別
        file_level: 文件輸出的日誌級別
        append_mode: True使用追加模式，False使用覆寫模式
        
    Returns:
        logging.Logger: 配置好的日誌物件
    """
    # 若日誌器已存在，直接返回
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    # 創建日誌物件
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 避免重複輸出
    
    # 清除現有的處理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(CustomFormatter(use_color=True, module_name=name))
    
    # 添加控制台處理器
    logger.addHandler(console_handler)
    
    # 如果指定了日誌檔案，添加文件處理器
    if log_file:
        file_mode = 'a' if append_mode else 'w'
        file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(
            f'[%(asctime)s] [{name}] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    
    # 保存日誌器引用
    _LOGGERS[name] = logger
    
    return logger

def set_external_logger_level(logger_name: str, level: int = logging.WARNING) -> None:
    """
    設置外部庫的日誌級別
    
    Args:
        logger_name: 外部日誌器名稱
        level: 日誌級別
    """
    logging.getLogger(logger_name).setLevel(level)

# 設置常用外部庫的日誌級別
set_external_logger_level("speechbrain", logging.WARNING)

# 主日誌器 - 可以在導入時直接使用
logger = get_logger("Voice_ID")
