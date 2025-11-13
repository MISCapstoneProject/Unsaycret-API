"""
MongoDB 初始化模組

功能：
1. 建立 MongoDB 連線
2. 初始化 Beanie ODM（Object Document Mapper）
3. 註冊所有 Document 模型
4. 自動建立索引

使用方式（在 main.py）：
```python
from modules.database.init_mongodb import initialize_mongodb
import asyncio

if not await initialize_mongodb():
    logger.error("MongoDB 初始化失敗")
    sys.exit(1)
```

注意：
- MongoDB 是 schema-less，collections 會在第一次插入資料時自動建立
- 但 Beanie 需要先初始化，才能使用 Document 模型
- 索引會在初始化時自動建立（根據 Document.Settings.indexes）
"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

# 匯入所有 Document 模型
from modules.database.models.session import Session
from modules.database.models.transcript import Transcript
from modules.database.models.ai_summary import AISummary

logger = logging.getLogger(__name__)

# 全域變數（用於單例模式）
_mongodb_client: Optional[AsyncIOMotorClient] = None


def get_mongo_config() -> tuple[str, str]:
    """
    從環境變數取得 MongoDB 配置
    
    Returns:
        tuple[str, str]: (MONGO_URL, MONGO_DB_NAME)
    
    Raises:
        ValueError: 如果環境變數不存在
    """
    try:
        # 方法 1: 從 env_config 匯入（如果存在）
        try:
            from utils.env_config import MONGO_URL, MONGO_DB_NAME
            return MONGO_URL, MONGO_DB_NAME
        except (ImportError, AttributeError):
            pass
        
        # 方法 2: 直接從環境變數讀取
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        mongo_url = os.getenv("MONGO_URL")
        mongo_db_name = os.getenv("MONGO_DB_NAME", "unsaycret")  # 預設資料庫名稱
        
        if not mongo_url:
            raise ValueError(
                "找不到 MONGO_URL 環境變數！\n"
                "請在 .env 檔案中設定：\n"
                "MONGO_URL=mongodb://root:admin123@localhost:27017\n"
                "MONGO_DB_NAME=unsaycret"
            )
        
        return mongo_url, mongo_db_name
        
    except Exception as e:
        logger.error(f"❌ 無法讀取 MongoDB 配置: {e}")
        raise


async def test_mongodb_connection() -> bool:
    """
    測試 MongoDB 連線（不初始化 Beanie）
    
    用途：在不需要完整初始化的情況下，快速檢查 MongoDB 是否可用
    
    Returns:
        bool: 連線是否成功
    """
    try:
        mongo_url, _ = get_mongo_config()
        
        client = AsyncIOMotorClient(mongo_url)
        await client.admin.command('ping')
        logger.info("✅ MongoDB ping 成功")
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB ping 失敗: {e}")
        return False


async def initialize_mongodb() -> bool:
    """
    初始化 MongoDB 連線與 Beanie ODM
    
    執行步驟：
    1. 讀取環境變數（MONGO_URL, MONGO_DB_NAME）
    2. 建立 AsyncIOMotorClient 連線
    3. 初始化 Beanie（註冊 Document 模型）
    4. Beanie 自動建立索引（根據 Document.Settings.indexes）
    5. 測試連線（ping）
    
    Returns:
        bool: 初始化是否成功
    
    異常處理：
    - 連線失敗：返回 False
    - 初始化失敗：返回 False
    - 任何 Exception：記錄錯誤並返回 False
    """
    global _mongodb_client
    
    try:
        # 1. 讀取環境變數
        mongo_url, mongo_db_name = get_mongo_config()
        
        logger.info(f"正在連線到 MongoDB: {mongo_db_name}")
        
        # 2. 建立 MongoDB 客戶端（async）
        _mongodb_client = AsyncIOMotorClient(mongo_url)
        
        # 3. 測試連線（ping）
        try:
            await _mongodb_client.admin.command('ping')
            logger.info("✅ MongoDB 連線成功")
        except Exception as e:
            logger.error(f"❌ MongoDB 連線失敗: {e}")
            return False
        
        # 4. 初始化 Beanie（註冊所有 Document 模型）
        database = _mongodb_client[mongo_db_name]
        
        document_models = [
            Session,     # 會議資料
            Transcript,  # 逐字稿
            AISummary,   # AI 摘要
        ]
        
        await init_beanie(
            database=database,
            document_models=document_models
        )
        logger.info("✅ Beanie ODM 初始化完成（已註冊 Session, Transcript, AISummary）")
        
        # 5. 顯示已註冊的 collections
        collections = await database.list_collection_names()
        if collections:
            logger.info(f"現有 collections: {collections}")
        else:
            logger.info("目前沒有任何 collections（首次啟動）")
        
        # 6. 檢查索引是否建立
        for collection_name in ["sessions", "transcripts", "ai_summaries"]:
            if collection_name in collections:
                indexes = await database[collection_name].index_information()
                logger.info(f"Collection '{collection_name}' 索引: {list(indexes.keys())}")
        
        logger.info("✅ MongoDB 初始化完成")
        return True
        
    except ValueError as e:
        # 環境變數配置錯誤
        logger.error(f"❌ MongoDB 配置錯誤: {e}")
        return False
    except ImportError as e:
        logger.error(f"❌ 無法匯入模型或環境變數: {e}")
        logger.error("   請確認 modules/database/models/ 目錄已建立")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB 初始化失敗: {e}", exc_info=True)
        return False


async def close_mongodb():
    """
    關閉 MongoDB 連線
    
    用途：在應用程式關閉時清理資源
    """
    global _mongodb_client
    
    if _mongodb_client:
        _mongodb_client.close()
        logger.info("✅ MongoDB 連線已關閉")
        _mongodb_client = None


# 如果直接執行此檔案，進行測試
if __name__ == "__main__":
    import asyncio
    
    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("MongoDB 初始化測試")
    print("=" * 60)
    print()
    
    # 測試連線
    print("【步驟 1】測試 MongoDB 連線...")
    if asyncio.run(test_mongodb_connection()):
        print("✅ 連線測試成功\n")
    else:
        print("❌ 連線測試失敗\n")
        print("故障排除：")
        print("1. 確認 Docker Compose 已啟動：docker-compose ps")
        print("2. 確認 MongoDB 健康狀態：docker-compose logs mongodb")
        print("3. 確認 .env 檔案包含 MONGO_URL")
        exit(1)
    
    # 完整初始化
    print("【步驟 2】執行完整初始化...")
    if asyncio.run(initialize_mongodb()):
        print("✅ 初始化成功\n")
        print("下一步：")
        print("1. 使用 Mongo Express 查看資料庫: http://localhost:8082")
        print("2. 建立 Session/Transcript/AISummary 模型")
        print("3. 重新執行此測試以驗證模型註冊")
    else:
        print("❌ 初始化失敗\n")
    
    # 清理資源
    asyncio.run(close_mongodb())
