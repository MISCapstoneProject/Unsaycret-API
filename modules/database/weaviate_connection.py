"""
Weaviate 連線管理模組

功能：
1. 單例模式管理 Weaviate 連線
2. 提供客戶端實例存取
3. 健康檢查與連線測試
4. 優雅的連線關閉

使用方式：
```python
from modules.database.weaviate_connection import get_weaviate_client, ping_weaviate

# 取得 Weaviate 客戶端（單例）
client = get_weaviate_client()

# 健康檢查
is_healthy = ping_weaviate()
```

注意：
- 使用單例模式，確保全域只有一個 Weaviate 連線
- 連線會在首次呼叫時建立，之後重複使用
- 應用程式關閉時應呼叫 close_weaviate_client() 清理資源
- 本模組從 database.py 拆分而來，保留原有邏輯
"""

import logging
from typing import Optional
import weaviate
from weaviate.client import WeaviateClient

logger = logging.getLogger(__name__)

# 全域變數（單例模式）
_weaviate_client: Optional[WeaviateClient] = None


def get_weaviate_config() -> tuple[str, int]:
    """
    從環境變數取得 Weaviate 配置
    
    Returns:
        tuple[str, int]: (WEAVIATE_HOST, WEAVIATE_PORT)
    
    Raises:
        ValueError: 如果環境變數不存在
    """
    try:
        # 方法 1: 從 env_config 匯入（如果存在）
        try:
            from utils.env_config import WEAVIATE_HOST, WEAVIATE_PORT
            return WEAVIATE_HOST, WEAVIATE_PORT
        except (ImportError, AttributeError):
            pass
        
        # 方法 2: 直接從環境變數讀取
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
        weaviate_port = int(os.getenv("WEAVIATE_PORT", "8080"))
        
        return weaviate_host, weaviate_port
        
    except Exception as e:
        logger.error(f"❌ 無法讀取 Weaviate 配置: {e}")
        raise


def get_weaviate_client() -> WeaviateClient:
    """
    取得 Weaviate 客戶端（單例模式）
    
    Returns:
        WeaviateClient: Weaviate 客戶端實例
    
    Raises:
        Exception: 如果連線失敗
    
    範例：
    ```python
    client = get_weaviate_client()
    speaker_collection = client.collections.get("Speaker")
    ```
    """
    global _weaviate_client
    
    if _weaviate_client is None:
        try:
            weaviate_host, weaviate_port = get_weaviate_config()
            
            _weaviate_client = weaviate.connect_to_local(
                host=weaviate_host,
                port=weaviate_port,
            )
            
            # 測試連線（檢查是否可以列出 collections）
            _weaviate_client.collections.list_all()
            logger.info(f"✅ Weaviate 客戶端已建立 ({weaviate_host}:{weaviate_port})")
            
        except Exception as e:
            logger.error(f"❌ 建立 Weaviate 客戶端失敗: {e}")
            logger.error("請確認：")
            logger.error("1. Docker 服務是否正在運行")
            logger.error("2. Weaviate 容器是否已經啟動")
            logger.error("3. docker-compose.yml 中的配置是否正確")
            _weaviate_client = None
            raise
    
    return _weaviate_client


def ensure_connection() -> WeaviateClient:
    """
    確保 Weaviate 連線正常，如果斷線則重新連線
    
    Returns:
        WeaviateClient: Weaviate 客戶端實例
    
    Raises:
        Exception: 如果重新連線失敗
    
    使用時機：
    - 在長時間運行的操作前檢查連線
    - 當懷疑連線可能已斷開時
    
    範例：
    ```python
    client = ensure_connection()
    # 安全地執行操作
    ```
    """
    global _weaviate_client
    
    try:
        # 檢查現有連線是否可用
        if _weaviate_client is not None:
            # 測試連線（嘗試列出 collections）
            _weaviate_client.collections.list_all()
            return _weaviate_client
    except Exception:
        logger.warning("⚠️ 檢測到 Weaviate 連線異常，嘗試重新連線...")
    
    # 連線不存在或已斷開，重新建立
    try:
        if _weaviate_client is not None:
            try:
                _weaviate_client.close()
            except:
                pass
        
        _weaviate_client = None
        return get_weaviate_client()
        
    except Exception as e:
        logger.error(f"❌ 重新連線 Weaviate 失敗: {e}")
        raise


def ping_weaviate() -> bool:
    """
    測試 Weaviate 連線是否正常（健康檢查）
    
    Returns:
        bool: 連線是否正常
    
    範例：
    ```python
    if ping_weaviate():
        print("Weaviate 連線正常")
    else:
        print("Weaviate 連線失敗")
    ```
    """
    try:
        client = get_weaviate_client()
        # 測試操作：列出所有 collections
        client.collections.list_all()
        logger.debug("✅ Weaviate ping 成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ Weaviate ping 失敗: {e}")
        return False


def check_collection_exists(collection_name: str) -> bool:
    """
    檢查 collection 是否存在
    
    Args:
        collection_name: collection 名稱（例如："Speaker", "VoicePrint"）
    
    Returns:
        bool: collection 是否存在
    
    範例：
    ```python
    if check_collection_exists("Speaker"):
        print("Speaker collection 已存在")
    else:
        print("需要建立 Speaker collection")
    ```
    """
    try:
        client = get_weaviate_client()
        exists = client.collections.exists(collection_name)
        
        if exists:
            logger.debug(f"✅ Collection '{collection_name}' 存在")
        else:
            logger.warning(f"⚠️ Collection '{collection_name}' 不存在")
        
        return exists
        
    except Exception as e:
        logger.error(f"❌ 檢查 collection '{collection_name}' 失敗: {e}")
        return False


def close_weaviate_client():
    """
    關閉 Weaviate 連線（清理資源）
    
    用途：在應用程式關閉時呼叫，確保連線正確關閉
    
    範例：
    ```python
    # 在 FastAPI 的 lifespan 中使用
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        yield
        # Shutdown
        close_weaviate_client()
    ```
    """
    global _weaviate_client
    
    if _weaviate_client:
        try:
            _weaviate_client.close()
            logger.info("✅ Weaviate 連線已關閉")
        except Exception as e:
            logger.warning(f"⚠️ 關閉 Weaviate 連線時發生錯誤: {e}")
        finally:
            _weaviate_client = None


def get_collection_names() -> list[str]:
    """
    取得所有 collection 名稱
    
    Returns:
        list[str]: collection 名稱列表
    
    範例：
    ```python
    collections = get_collection_names()
    print(f"現有 collections: {collections}")
    # 輸出：現有 collections: ['Speaker', 'VoicePrint']
    ```
    """
    try:
        client = get_weaviate_client()
        collections = client.collections.list_all()
        collection_names = [c.name for c in collections]
        return collection_names
        
    except Exception as e:
        logger.error(f"❌ 取得 collection 名稱失敗: {e}")
        return []


# 如果直接執行此檔案，進行測試
if __name__ == "__main__":
    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def test():
        print("=" * 60)
        print("Weaviate 連線管理測試")
        print("=" * 60)
        print()
        
        # 測試 1: 取得客戶端
        print("【測試 1】取得 Weaviate 客戶端...")
        try:
            client = get_weaviate_client()
            print(f"✅ 成功取得客戶端: {type(client)}")
        except Exception as e:
            print(f"❌ 失敗: {e}")
            return
        
        # 測試 2: Ping 測試
        print("\n【測試 2】健康檢查（ping）...")
        if ping_weaviate():
            print("✅ Weaviate 連線正常")
        else:
            print("❌ Weaviate 連線異常")
        
        # 測試 3: 取得 collection 名稱
        print("\n【測試 3】取得所有 collections...")
        collections = get_collection_names()
        if collections:
            print(f"✅ 現有 collections: {collections}")
        else:
            print("⚠️ 目前沒有任何 collections（首次啟動）")
        
        # 測試 4: 檢查特定 collection
        print("\n【測試 4】檢查 Speaker collection...")
        if check_collection_exists("Speaker"):
            print("✅ Speaker collection 存在")
        else:
            print("⚠️ Speaker collection 不存在")
        
        # 測試 5: 確保連線
        print("\n【測試 5】確保連線...")
        try:
            client = ensure_connection()
            print("✅ 連線確認成功")
        except Exception as e:
            print(f"❌ 連線確認失敗: {e}")
        
        # 測試 6: 關閉連線
        print("\n【測試 6】關閉連線...")
        close_weaviate_client()
        print("✅ 連線已關閉")
        
        print("\n" + "=" * 60)
        print("測試完成！")
        print("=" * 60)
    
    # 執行測試
    test()
