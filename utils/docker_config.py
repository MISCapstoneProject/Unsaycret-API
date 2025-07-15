"""
Docker 環境配置模組
處理路徑、環境變數等配置
"""
import os
from pathlib import Path
from typing import Optional

# 基礎路徑配置
APP_ROOT = Path(__file__).parent.parent
DATA_ROOT = APP_ROOT / "data"
MODELS_ROOT = APP_ROOT / "models"
WORK_OUTPUT_ROOT = APP_ROOT / "work_output"
STREAM_OUTPUT_ROOT = APP_ROOT / "stream_output"
EMBEDDING_FILES_ROOT = APP_ROOT / "embeddingFiles"
AUDIO_16K_ROOT = APP_ROOT / "16K-model" / "Audios-16K-IDTF"
LOGS_ROOT = APP_ROOT / "logs"

# 確保目錄存在
def ensure_directories():
    """確保所有必要目錄存在"""
    directories = [
        DATA_ROOT,
        MODELS_ROOT,
        WORK_OUTPUT_ROOT,
        STREAM_OUTPUT_ROOT,
        EMBEDDING_FILES_ROOT,
        AUDIO_16K_ROOT,
        LOGS_ROOT
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# 環境變數配置
def get_env_config() -> dict:
    """取得環境變數配置"""
    return {
        "WEAVIATE_HOST": os.getenv("WEAVIATE_HOST", "localhost"),
        "WEAVIATE_PORT": os.getenv("WEAVIATE_PORT", "8200"),
        "WEAVIATE_SCHEME": os.getenv("WEAVIATE_SCHEME", "http"),
        "FASTAPI_HOST": os.getenv("FASTAPI_HOST", "0.0.0.0"),
        "FASTAPI_PORT": int(os.getenv("FASTAPI_PORT", "18000")),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "MAX_WORKERS": int(os.getenv("MAX_WORKERS", "3")),
        "GPU_ENABLED": os.getenv("GPU_ENABLED", "false").lower() == "true",
    }

# 日誌檔案路徑
def get_log_file_path(filename: str = "system_output.log") -> str:
    """取得日誌檔案路徑"""
    return str(LOGS_ROOT / filename)

# 模型路徑
def get_model_path(model_type: str, model_name: str = "") -> str:
    """取得模型路徑"""
    if model_name:
        return str(MODELS_ROOT / model_type / model_name)
    return str(MODELS_ROOT / model_type)

# 輸出路徑
def get_work_output_path(session_id: str = "") -> str:
    """取得工作輸出路徑"""
    if session_id:
        return str(WORK_OUTPUT_ROOT / session_id)
    return str(WORK_OUTPUT_ROOT)

def get_stream_output_path(session_id: str = "") -> str:
    """取得串流輸出路徑"""
    if session_id:
        return str(STREAM_OUTPUT_ROOT / session_id)
    return str(STREAM_OUTPUT_ROOT)

# 初始化
ensure_directories()
