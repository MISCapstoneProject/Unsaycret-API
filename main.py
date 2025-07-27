"""
ｖ0.4.1
語音項目API服務啟動腳本
包含語音轉錄、語者改名和聲紋轉移功能
"""
import uvicorn
import sys
import os
from api.api import app
from modules.database.init_v2_collections import ensure_weaviate_collections
from utils.logger import get_logger
from utils.env_config import API_HOST, API_PORT, API_LOG_LEVEL, WEAVIATE_HOST, WEAVIATE_PORT

logger = get_logger(__name__)

def initialize_system() -> bool:
    """
    初始化系統環境
    
    Returns:
        bool: 初始化是否成功
    """
    try:
        # 初始化 Weaviate 集合
        logger.info(f"正在初始化 Weaviate 資料庫 ({WEAVIATE_HOST}:{WEAVIATE_PORT})...")
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
    try:
        logger.info("\n" + "="*60 + "\n🚀 [系統啟動] Unsaycret-API 服務啟動中...\n" + "="*60)
        # 初始化系統
        if not initialize_system():
            print("❌ 系統初始化失敗，無法啟動服務")
            logger.error("系統初始化失敗，無法啟動服務")
            sys.exit(1)
        
        print("✅ 系統初始化成功")
        print("🔌 可用的API端點：")
        print("  📝 POST /transcribe              - 語音轉錄（分離+辨識+ASR）")
        print("  📁 POST /transcribe_dir          - 批次轉錄（目錄/ZIP檔）")
        print("  🌐 WebSocket /ws/stream          - 即時語音處理")
        print("")
        print("  � 語者管理 API:")
        print("     📋 GET    /speakers           - 列出所有語者")
        print("     👤 GET    /speakers/{id}      - 獲取語者資訊")
        print("     ✏️  PATCH  /speakers/{id}      - 更新語者資料")
        print("     🗑️  DELETE /speakers/{id}      - 刪除語者")
        print("     🔍 POST   /speakers/verify    - 語音驗證（識別語者身份）")
        print("     � POST   /speakers/transfer  - 聲紋轉移")
        print("")
        print("  📅 會議管理 API:")
        print("     📋 GET    /sessions           - 列出所有會議")
        print("     ➕ POST   /sessions           - 建立新會議")
        print("     📖 GET    /sessions/{id}      - 獲取會議資訊")
        print("     ✏️  PATCH  /sessions/{id}      - 更新會議資料")
        print("     �️  DELETE /sessions/{id}      - 刪除會議（級聯刪除語音記錄）")
        print("")
        print("  💬 語音記錄 API:")
        print("     �📋 GET    /speechlogs         - 列出所有語音記錄")
        print("     ➕ POST   /speechlogs         - 建立新語音記錄")
        print("     📖 GET    /speechlogs/{id}    - 獲取語音記錄")
        print("     ✏️  PATCH  /speechlogs/{id}    - 更新語音記錄")
        print("     🗑️  DELETE /speechlogs/{id}    - 刪除語音記錄")
        print("")
        print("  🔗 關聯查詢 API:")
        print("     📋 GET    /speakers/{id}/sessions    - 語者參與的會議")
        print("     📋 GET    /speakers/{id}/speechlogs  - 語者的語音記錄")
        print("     📋 GET    /sessions/{id}/speechlogs  - 會議中的語音記錄")
        print("")
        print("  📖 GET  /docs                   - API互動式文檔（Swagger）")
        print("  📚 GET  /redoc                  - API文檔（ReDoc）")
        print("-" * 50)
        
        uvicorn.run(
            app, 
            host=API_HOST, 
            port=API_PORT,
            log_level=API_LOG_LEVEL
        )
        logger.info("\n" + "="*60 + "\n🟢 [系統已正常關閉] (主動結束/服務停止)\n" + "="*60)
    except KeyboardInterrupt:
        logger.info("\n" + "="*60 + "\n🟡 [系統已正常關閉] (使用者 Ctrl+C 中斷)\n" + "="*60)
    except Exception as e:
        logger.error("\n" + "="*60 + f"\n🔴 [系統異常關閉] {e}\n" + "="*60)
        raise
    finally:
        pass
