"""
MongoDB 連線管理模組

功能：
1. 單例模式管理 MongoDB 連線
2. 提供資料庫實例存取
3. 健康檢查與連線測試
4. 優雅的連線關閉

使用方式：
```python
from modules.database.mongodb_connection import get_mongodb_client, get_database

# 取得 MongoDB 客戶端（單例）
client = await get_mongodb_client()

# 取得資料庫實例
db = await get_database()

# 健康檢查
is_healthy = await ping_mongodb()
```

注意：
- 使用單例模式，確保全域只有一個 MongoDB 連線
- 連線會在首次呼叫時建立，之後重複使用
- 應用程式關閉時應呼叫 close_mongodb_client() 清理資源
"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

# 全域變數（單例模式）
_mongodb_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


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


async def get_mongodb_client() -> AsyncIOMotorClient:
    """
    取得 MongoDB 客戶端（單例模式）
    
    Returns:
        AsyncIOMotorClient: MongoDB 客戶端實例
    
    Raises:
        Exception: 如果連線失敗
    
    範例：
    ```python
    client = await get_mongodb_client()
    db = client["unsaycret"]
    collection = db["sessions"]
    ```
    """
    global _mongodb_client
    
    if _mongodb_client is None:
        try:
            mongo_url, _ = get_mongo_config()
            _mongodb_client = AsyncIOMotorClient(mongo_url)
            
            # 測試連線
            await _mongodb_client.admin.command('ping')
            logger.info("✅ MongoDB 客戶端已建立")
            
        except Exception as e:
            logger.error(f"❌ 建立 MongoDB 客戶端失敗: {e}")
            _mongodb_client = None
            raise
    
    return _mongodb_client


async def get_database() -> AsyncIOMotorDatabase:
    """
    取得 MongoDB 資料庫實例（單例模式）
    
    Returns:
        AsyncIOMotorDatabase: MongoDB 資料庫實例
    
    Raises:
        Exception: 如果連線失敗
    
    範例：
    ```python
    db = await get_database()
    collection = db["sessions"]
    result = await collection.find_one({"title": "專案討論"})
    ```
    """
    global _database
    
    if _database is None:
        try:
            _, mongo_db_name = get_mongo_config()
            client = await get_mongodb_client()
            _database = client[mongo_db_name]
            logger.info(f"✅ MongoDB 資料庫 '{mongo_db_name}' 已連線")
            
        except Exception as e:
            logger.error(f"❌ 取得 MongoDB 資料庫失敗: {e}")
            _database = None
            raise
    
    return _database


async def ping_mongodb() -> bool:
    """
    測試 MongoDB 連線是否正常（健康檢查）
    
    Returns:
        bool: 連線是否正常
    
    範例：
    ```python
    if await ping_mongodb():
        print("MongoDB 連線正常")
    else:
        print("MongoDB 連線失敗")
    ```
    """
    try:
        client = await get_mongodb_client()
        await client.admin.command('ping')
        logger.debug("✅ MongoDB ping 成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB ping 失敗: {e}")
        return False


async def close_mongodb_client():
    """
    關閉 MongoDB 連線（清理資源）
    
    用途：在應用程式關閉時呼叫，確保連線正確關閉
    
    範例：
    ```python
    # 在 FastAPI 的 lifespan 中使用
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        yield
        # Shutdown
        await close_mongodb_client()
    ```
    """
    global _mongodb_client, _database
    
    if _mongodb_client:
        _mongodb_client.close()
        logger.info("✅ MongoDB 連線已關閉")
        _mongodb_client = None
        _database = None


async def get_collection_names() -> list[str]:
    """
    取得所有 collection 名稱
    
    Returns:
        list[str]: collection 名稱列表
    
    範例：
    ```python
    collections = await get_collection_names()
    print(f"現有 collections: {collections}")
    # 輸出：現有 collections: ['sessions', 'transcripts', 'ai_summaries']
    ```
    """
    try:
        db = await get_database()
        collections = await db.list_collection_names()
        return collections
        
    except Exception as e:
        logger.error(f"❌ 取得 collection 名稱失敗: {e}")
        return []


# 如果直接執行此檔案，進行測試
if __name__ == "__main__":
    import asyncio
    
    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test():
        print("=" * 60)
        print("MongoDB 連線管理測試")
        print("=" * 60)
        print()
        
        # 測試 1: 取得客戶端
        print("【測試 1】取得 MongoDB 客戶端...")
        try:
            client = await get_mongodb_client()
            print(f"✅ 成功取得客戶端: {client}")
        except Exception as e:
            print(f"❌ 失敗: {e}")
            return
        
        # 測試 2: 取得資料庫
        print("\n【測試 2】取得資料庫實例...")
        try:
            db = await get_database()
            print(f"✅ 成功取得資料庫: {db.name}")
        except Exception as e:
            print(f"❌ 失敗: {e}")
            return
        
        # 測試 3: Ping 測試
        print("\n【測試 3】健康檢查（ping）...")
        if await ping_mongodb():
            print("✅ MongoDB 連線正常")
        else:
            print("❌ MongoDB 連線異常")
        
        # 測試 4: 取得 collection 名稱
        print("\n【測試 4】取得所有 collections...")
        collections = await get_collection_names()
        if collections:
            print(f"✅ 現有 collections: {collections}")
        else:
            print("⚠️ 目前沒有任何 collections（首次啟動）")
        
        # 測試 5: 關閉連線
        print("\n【測試 5】關閉連線...")
        await close_mongodb_client()
        print("✅ 連線已關閉")
        
        print("\n" + "=" * 60)
        print("測試完成！")
        print("=" * 60)
    
    # 執行測試
    asyncio.run(test())
