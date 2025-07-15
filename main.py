"""
語音項目API服務啟動腳本
包含語音轉錄、語者改名和聲紋轉移功能
"""
import uvicorn
import sys
import os
from services.api import app
from utils.init_collections import ensure_weaviate_collections
from utils.logger import get_logger

logger = get_logger(__name__)

def initialize_system() -> bool:
    """
    初始化系統環境
    
    Returns:
        bool: 初始化是否成功
    """
    try:
        # 初始化 Weaviate 集合
        logger.info("正在初始化 Weaviate 資料庫...")
        if not ensure_weaviate_collections():
            logger.error("Weaviate 資料庫初始化失敗")
            return False
        
        logger.info("系統初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"系統初始化時發生錯誤: {str(e)}")
        return False

if __name__ == "__main__":
    print("正在啟動語音項目API服務...")
    
    # 初始化系統
    if not initialize_system():
        print("❌ 系統初始化失敗，無法啟動服務")
        sys.exit(1)
    
    print("✅ 系統初始化成功")
    print("🔌 可用的API端點：")
    print("  📝 POST /transcribe         - 語音轉錄（分離+辨識+ASR）")
    print("  📁 POST /transcribe_dir     - 批次轉錄（目錄/ZIP檔）")
    print("  🔄 POST /speaker/rename     - 語者改名")
    print("  🔀 POST /speaker/transfer   - 聲紋轉移")
    print("  🔍 POST /speaker/verify     - 語音驗證（識別語者身份）")
    print("  👤 GET  /speaker/{id}       - 獲取語者資訊")
    print("  📋 GET  /speakers           - 列出所有語者")
    print("  🗑️  DELETE /speaker/{id}     - 刪除語者")
    print("  🌐 WebSocket /ws/stream     - 即時語音處理")
    print("  📖 GET  /docs              - API互動式文檔")
    print("  📚 GET  /redoc             - API文檔（ReDoc）")
    print("-" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
