"""
環境變數載入模組
負責從 .env 檔案載入環境變數，並提供優雅的降級處理
"""

import os
import shutil
from typing import Union
from dotenv import load_dotenv
from pathlib import Path

# 設定檔案路徑
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
env_example_path = project_root / '.env.example'

# 檢查 .env 檔案是否存在
env_file_missing = not env_path.exists()

if env_file_missing:
    print("⚠️" * 50)
    print("🚨 警告：找不到 .env 環境配置檔案！")
    print("🚨 WARNING: .env environment configuration file not found!")
    print("⚠️" * 50)
    print()
    print("📁 請執行以下步驟設定環境變數：")
    print("📁 Please follow these steps to set up environment variables:")
    print()
    print("1️⃣  複製範例檔案 | Copy example file:")
    print(f"   cp {env_example_path.name} {env_path.name}")
    print()
    print("2️⃣  編輯配置檔案 | Edit configuration file:")
    print(f"   # 編輯 {env_path.name} 並調整以下設定：")
    print("   # Edit .env and adjust these settings:")
    print("   - WEAVIATE_HOST (資料庫主機)")
    print("   - WEAVIATE_PORT (資料庫端口)")  
    print("   - API_HOST (API 服務主機)")
    print("   - API_PORT (API 服務端口)")
    print("   - MODELS_BASE_DIR (模型存放路徑)")
    print()
    
    # 如果 .env.example 存在，嘗試自動複製
    if env_example_path.exists():
        try:
            shutil.copy2(env_example_path, env_path)
            print("✅ 已自動複製 .env.example 為 .env")
            print("✅ Automatically copied .env.example to .env")
            print("🔧 請編輯 .env 檔案以符合你的環境需求")
            print("🔧 Please edit .env file to match your environment")
            print()
            env_file_missing = False
        except Exception as e:
            print(f"❌ 自動複製失敗：{e}")
            print(f"❌ Auto-copy failed: {e}")
            print("🔧 請手動複製檔案")
            print("🔧 Please copy file manually")
            print()
    else:
        print("❌ .env.example 檔案也不存在！請檢查專案結構")
        print("❌ .env.example file not found! Please check project structure")
        print()

# 載入環境變數
if not env_file_missing:
    load_dotenv(dotenv_path=env_path)
    
    # 記錄配置來源
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info(f"已載入環境配置檔案: {env_path}")
else:
    # 使用 .env.example 作為後備
    if env_example_path.exists():
        load_dotenv(dotenv_path=env_example_path)
        print("⚠️  使用 .env.example 作為後備配置")
        print("⚠️  Using .env.example as fallback configuration")
        print()
        
        # 延遲導入 logger 避免循環導入
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"使用後備配置檔案: {env_example_path}")
            logger.warning("建議建立 .env 檔案以進行個人化配置")
        except ImportError:
            pass

def get_env_bool(key: str, default: bool = False) -> bool:
    """
    取得布林類型的環境變數
    
    Args:
        key: 環境變數名稱
        default: 預設值
        
    Returns:
        bool: 環境變數值
    """
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_env_int(key: str, default: int = 0) -> int:
    """
    取得整數類型的環境變數
    
    Args:
        key: 環境變數名稱
        default: 預設值
        
    Returns:
        int: 環境變數值
    """
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def get_env_float(key: str, default: float = 0.0) -> float:
    """
    取得浮點數類型的環境變數
    
    Args:
        key: 環境變數名稱
        default: 預設值
        
    Returns:
        float: 環境變數值
    """
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

def get_env_str(key: str, default: str = "") -> str:
    """
    取得字串類型的環境變數
    
    Args:
        key: 環境變數名稱
        default: 預設值
        
    Returns:
        str: 環境變數值
    """
    return os.getenv(key, default)

# Hugging Face 配置 (從環境變數讀取)
HF_ACCESS_TOKEN = get_env_str('HF_ACCESS_TOKEN', '')

# API 服務配置 (從環境變數讀取)
API_HOST = get_env_str('API_HOST', '0.0.0.0')
API_PORT = get_env_int('API_PORT', 8000)
API_DEBUG = get_env_bool('API_DEBUG', False)
API_LOG_LEVEL = get_env_str('API_LOG_LEVEL', 'info')

# Weaviate 資料庫配置 (從環境變數讀取)
WEAVIATE_HOST = get_env_str('WEAVIATE_HOST', 'localhost')
WEAVIATE_PORT = get_env_int('WEAVIATE_PORT', 8080)
WEAVIATE_SCHEME = get_env_str('WEAVIATE_SCHEME', 'http')
WEAVIATE_MAX_RETRIES = get_env_int('WEAVIATE_MAX_RETRIES', 3)
WEAVIATE_CONNECTION_TIMEOUT = get_env_int('WEAVIATE_CONNECTION_TIMEOUT', 30)

# 檔案路徑配置 (從環境變數讀取)
MODELS_BASE_DIR = get_env_str('MODELS_BASE_DIR', './models')
EMBEDDING_BASE_DIR = get_env_str('EMBEDDING_BASE_DIR', './embeddingFiles')  
OUTPUT_BASE_DIR = get_env_str('OUTPUT_BASE_DIR', './stream_output')

# 設備配置 (從環境變數讀取)
FORCE_CPU = get_env_bool('FORCE_CPU', False)
CUDA_DEVICE_INDEX = get_env_int('CUDA_DEVICE_INDEX', 0)

# 系統配置 (從環境變數讀取)
LOG_LEVEL = get_env_str('LOG_LEVEL', 'INFO')
LOG_FILE = get_env_str('LOG_FILE', './system_output.log')
DEVELOPMENT_MODE = get_env_bool('DEVELOPMENT_MODE', True)
VERBOSE_OUTPUT = get_env_bool('VERBOSE_OUTPUT', True)
CONTAINER_MODE = get_env_bool('CONTAINER_MODE', False)
MODEL_DOWNLOAD_TIMEOUT = get_env_int('MODEL_DOWNLOAD_TIMEOUT', 300)
AUTO_DOWNLOAD_MODELS = get_env_bool('AUTO_DOWNLOAD_MODELS', True)

# 計算音訊速率 (為了向後相容)
AUDIO_RATE = 44100  # 錄音時的原始取樣率

def get_model_save_dir(model_type: str) -> str:
    """
    取得模型儲存目錄的絕對路徑
    
    Args:
        model_type: 模型類型
        
    Returns:
        str: 模型目錄的絕對路徑
    """
    return os.path.abspath(os.path.join(MODELS_BASE_DIR, model_type))

def get_weaviate_url() -> str:
    """
    取得 Weaviate 連接 URL
    
    Returns:
        str: Weaviate URL
    """
    return f"{WEAVIATE_SCHEME}://{WEAVIATE_HOST}:{WEAVIATE_PORT}"

def is_development() -> bool:
    """
    檢查是否為開發模式
    
    Returns:
        bool: 是否為開發模式
    """
    return DEVELOPMENT_MODE

def is_env_file_missing() -> bool:
    """
    檢查 .env 檔案是否缺失
    
    Returns:
        bool: .env 檔案是否缺失
    """
    return env_file_missing
