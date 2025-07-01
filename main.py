"""
語音項目API服務啟動腳本
包含語音轉錄、語者改名和聲紋轉移功能
"""
import uvicorn
from services.api import app

if __name__ == "__main__":
    print("正在啟動語音項目API服務...")
    print("可用的API端點：")
    print("  POST /transcribe        - 語音轉錄")
    print("  POST /speaker/rename    - 語者改名")
    print("  POST /speaker/transfer  - 聲紋轉移")
    print("  GET  /speaker/{id}      - 獲取語者資訊")
    print("  GET  /speakers          - 列出所有語者")
    print("  GET  /docs              - API文檔")
    print("  DELETE /speaker/{id}    - 刪除語者")
    print("-" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
