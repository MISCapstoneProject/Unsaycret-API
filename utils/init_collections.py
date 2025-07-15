"""
Weaviate Collections 初始化模組

此模組負責確保 Weaviate 資料庫中的必要集合存在，如果不存在則自動建立。
包含完整的錯誤處理和資源管理。
"""

import weaviate  # type: ignore
import weaviate.classes.config as wc  # type: ignore
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import sys
from utils.logger import get_logger

# 創建本模組的日誌器
logger = get_logger(__name__)

class WeaviateCollectionManager:
    """Weaviate 集合管理器"""
    
    def __init__(self, host: str = "localhost", port: int = 8200):
        """
        初始化 Weaviate 集合管理器
        
        Args:
            host: Weaviate 服務器主機地址
            port: Weaviate 服務器端口
        """
        # 支援環境變數配置
        from .docker_config import get_env_config
        env_config = get_env_config()
        
        self.host = host if host != "localhost" else env_config["WEAVIATE_HOST"]
        self.port = port if port != 8200 else int(env_config["WEAVIATE_PORT"])
        self.scheme = env_config["WEAVIATE_SCHEME"]
        self.client: Optional[weaviate.WeaviateClient] = None
    
    def __enter__(self):
        """進入 context manager"""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """離開 context manager"""
        self.disconnect()
    
    def connect(self) -> 'WeaviateCollectionManager':
        """
        連接到 Weaviate 資料庫
        
        Returns:
            WeaviateCollectionManager: 自身實例
            
        Raises:
            ConnectionError: 連接失敗時拋出
        """
        try:
            self.client = weaviate.connect_to_local(
                host=self.host,
                port=self.port
            )
            logger.info(f"成功連接到 Weaviate 服務器 ({self.host}:{self.port})")
            return self
        except Exception as e:
            logger.error(f"連接 Weaviate 服務器失敗: {str(e)}")
            raise ConnectionError(f"無法連接到 Weaviate 服務器: {str(e)}")
    
    def disconnect(self) -> None:
        """斷開與 Weaviate 的連接"""
        if self.client:
            try:
                self.client.close()
                logger.info("已斷開 Weaviate 連接")
            except Exception as e:
                logger.warning(f"斷開連接時發生錯誤: {str(e)}")
            finally:
                self.client = None
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        檢查集合是否存在
        
        Args:
            collection_name: 集合名稱
            
        Returns:
            bool: 集合是否存在
            
        Raises:
            RuntimeError: 客戶端未連接時拋出
        """
        if not self.client:
            raise RuntimeError("客戶端未連接，請先調用 connect() 方法")
        
        try:
            exists = self.client.collections.exists(collection_name)
            logger.info(f"集合 '{collection_name}' 存在狀態: {exists}")
            return exists
        except Exception as e:
            logger.error(f"檢查集合 '{collection_name}' 時發生錯誤: {str(e)}")
            raise
    
    def create_speaker_collection(self) -> bool:
        """
        建立 Speaker 集合
        
        Returns:
            bool: 是否成功建立
        """
        if not self.client:
            raise RuntimeError("客戶端未連接，請先調用 connect() 方法")
        
        try:
            # 如果已存在則先刪除
            if self.collection_exists("Speaker"):
                self.client.collections.delete("Speaker")
                logger.info("已刪除現有的 Speaker 集合")
            
            # 建立 Speaker 集合
            speaker_collection = self.client.collections.create(
                name="Speaker",
                properties=[
                    wc.Property(name="name", data_type=wc.DataType.TEXT),
                    wc.Property(name="create_time", data_type=wc.DataType.DATE),
                    wc.Property(name="last_active_time", data_type=wc.DataType.DATE),
                    wc.Property(name="voiceprint_ids", data_type=wc.DataType.UUID_ARRAY),
                    wc.Property(name="first_audio", data_type=wc.DataType.TEXT),
                ],
                vectorizer_config=wc.Configure.Vectorizer.none()
            )
            logger.info("成功建立 Speaker 集合")
            return True
            
        except Exception as e:
            logger.error(f"建立 Speaker 集合時發生錯誤: {str(e)}")
            return False
    
    def create_voiceprint_collection(self) -> bool:
        """
        建立 VoicePrint 集合
        
        Returns:
            bool: 是否成功建立
        """
        if not self.client:
            raise RuntimeError("客戶端未連接，請先調用 connect() 方法")
        
        try:
            # 如果已存在則先刪除
            if self.collection_exists("VoicePrint"):
                self.client.collections.delete("VoicePrint")
                logger.info("已刪除現有的 VoicePrint 集合")
            
            # 建立 VoicePrint 集合
            voice_print_collection = self.client.collections.create(
                name="VoicePrint",
                properties=[
                    wc.Property(name="create_time", data_type=wc.DataType.DATE),
                    wc.Property(name="updated_time", data_type=wc.DataType.DATE),
                    wc.Property(name="update_count", data_type=wc.DataType.INT),
                    wc.Property(name="speaker_name", data_type=wc.DataType.TEXT),
                ],
                references=[
                    wc.ReferenceProperty(
                        name="speaker",
                        target_collection="Speaker"
                    )
                ],
                vectorizer_config=wc.Configure.Vectorizer.none(),
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    distance_metric=wc.VectorDistances.COSINE
                )
            )
            logger.info("成功建立 VoicePrint 集合")
            return True
            
        except Exception as e:
            logger.error(f"建立 VoicePrint 集合時發生錯誤: {str(e)}")
            return False
    
    def create_all_collections(self) -> bool:
        """
        建立所有必要的集合
        
        Returns:
            bool: 是否成功建立所有集合
        """
        try:
            # 先建立 Speaker 集合（因為 VoicePrint 需要引用它）
            if not self.create_speaker_collection():
                return False
            
            # 再建立 VoicePrint 集合
            if not self.create_voiceprint_collection():
                return False
            
            logger.info("成功建立所有 Weaviate 集合")
            return True
            
        except Exception as e:
            logger.error(f"建立集合時發生錯誤: {str(e)}")
            return False
    
    def verify_collections(self) -> Dict[str, bool]:
        """
        驗證所有集合是否存在
        
        Returns:
            Dict[str, bool]: 集合名稱與存在狀態的對應
        """
        collections = ["Speaker", "VoicePrint"]
        results = {}
        
        for collection_name in collections:
            try:
                results[collection_name] = self.collection_exists(collection_name)
            except Exception as e:
                logger.error(f"驗證集合 '{collection_name}' 時發生錯誤: {str(e)}")
                results[collection_name] = False
        
        return results


def ensure_weaviate_collections(host: str = "localhost", port: int = 8200) -> bool:
    """
    確認 Weaviate collections 存在，若不存在則建立
    
    Args:
        host: Weaviate 服務器主機地址
        port: Weaviate 服務器端口
        
    Returns:
        bool: 是否成功確保所有集合存在
    """
    logger.info("開始初始化 Weaviate 集合...")
    
    try:
        with WeaviateCollectionManager(host, port) as manager:
            # 驗證現有集合
            existing_collections = manager.verify_collections()
            logger.info(f"現有集合狀態: {existing_collections}")
            
            # 檢查是否需要建立集合
            need_create = not all(existing_collections.values())
            
            if need_create:
                logger.info("正在建立 Weaviate 集合...")
                if manager.create_all_collections():
                    # 再次驗證
                    final_status = manager.verify_collections()
                    logger.info(f"最終集合狀態: {final_status}")
                    
                    if all(final_status.values()):
                        logger.info("所有 Weaviate 集合已成功建立並驗證")
                        return True
                    else:
                        logger.error("集合建立後驗證失敗")
                        return False
                else:
                    logger.error("建立集合失敗")
                    return False
            else:
                logger.info("所有必要的集合已存在，無需建立")
                return True
                
    except ConnectionError as e:
        logger.error(f"連接錯誤: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"初始化過程中發生未預期錯誤: {str(e)}")
        return False


def main() -> None:
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="初始化 Weaviate 集合")
    parser.add_argument("--host", default="localhost", help="Weaviate 服務器主機地址")
    parser.add_argument("--port", type=int, default=8200, help="Weaviate 服務器端口")
    
    args = parser.parse_args()
    
    success = ensure_weaviate_collections(
        host=args.host,
        port=args.port
    )
    
    if success:
        print("✅ Weaviate 集合初始化成功！")
        sys.exit(0)
    else:
        print("❌ Weaviate 集合初始化失敗！")
        sys.exit(1)


if __name__ == "__main__":
    main()