"""
Weaviate Collections V2 初始化模組

此模組負責建立新版本的 Weaviate 資料庫結構，包含 4 個正規化的集合：
- Speaker: 說話者主檔（包含 speaker_id INT 從 1 開始遞增）
- Session: 對話場景（取代原本的 Meeting）  
- SpeechLog: 一句話記錄（正規化的語音內容）
- VoicePrint: 聲紋特徵庫（改進版）

⚠️ 重要變更警告 ⚠️
本次重構將大幅改變資料庫結構，會導致現有資料不相容！
執行前請務必備份現有 Weaviate 資料庫。

🎯 向量化策略：
- DEFAULT_VECTORIZER_MODULE: 'none' (預設不向量化)
- Session: 啟用 text2vec_transformers
  - title: vectorize_property_name=True (語義搜尋)
  - summary: vectorize_property_name=True (語義搜尋)
  - session_id, session_type: vectorize_property_name=False (不需要向量化)
- SpeechLog: 啟用 text2vec_transformers
  - content: vectorize_property_name=True (語義搜尋)
  - language: vectorize_property_name=False (不需要向量化)
- Speaker, VoicePrint: 使用 none (關聯查詢即可)

📦 使用方法：
    from utils.init_v2_collections import ensure_weaviate_v2_collections, WeaviateV2CollectionManager
    
    # 作為模組使用
    success = ensure_weaviate_v2_collections(host="localhost", port=8080)
    
    # 直接執行
    python -m utils.init_v2_collections --host localhost --port 8080

🧪 測試功能：
    測試功能已移至 utils.test_init_v2_collections 模組
    python -m utils.test_init_v2_collections
"""

import weaviate  # type: ignore
import weaviate.classes.config as wc  # type: ignore
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import sys
import time
import warnings
from tqdm import tqdm
from utils.logger import get_logger

# 禁用 Weaviate 的 deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="weaviate")

# 創建本模組的日誌器
logger = get_logger(__name__)


class WeaviateV2CollectionManager:
    """Weaviate V2 集合管理器"""
    
    def __init__(self, host: str = "localhost", port: int = 8080, max_retries: int = 3):
        """
        初始化 Weaviate V2 集合管理器
        
        Args:
            host: Weaviate 服務器主機地址
            port: Weaviate 服務器端口  
            max_retries: 最大重試次數
        """
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.client: Optional[weaviate.WeaviateClient] = None
    
    def __enter__(self):
        """進入 context manager"""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """離開 context manager"""
        self.disconnect()
    
    def connect(self) -> 'WeaviateV2CollectionManager':
        """
        連接到 Weaviate 資料庫（包含重試機制）
        
        Returns:
            WeaviateV2CollectionManager: 自身實例
            
        Raises:
            ConnectionError: 連接失敗時拋出
        """
        for attempt in range(self.max_retries):
            try:
                self.client = weaviate.connect_to_local(
                    host=self.host,
                    port=self.port
                )
                logger.info(f"成功連接到 Weaviate 服務器 ({self.host}:{self.port})")
                return self
            except Exception as e:
                logger.warning(f"連接嘗試 {attempt + 1}/{self.max_retries} 失敗: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指數退避
                else:
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
            logger.debug(f"集合 '{collection_name}' 存在狀態: {exists}")
            return exists
        except Exception as e:
            logger.error(f"檢查集合 '{collection_name}' 時發生錯誤: {str(e)}")
            raise
    
    def create_speaker_v2(self) -> bool:
        """
        建立 Speaker V2 集合（增加 speaker_id INT、meet_count、meet_days）
        
        Returns:
            bool: 是否成功建立
        """
        if not self.client:
            raise RuntimeError("客戶端未連接，請先調用 connect() 方法")
        
        try:
            collection_name = "Speaker"
            
            # 檢查是否已存在，如果存在且是冪等運行則跳過
            if self.collection_exists(collection_name):
                logger.info(f"集合 '{collection_name}' 已存在，跳過建立")
                return True
            
            # 建立 Speaker V2 集合
            speaker_collection = self.client.collections.create(
                name=collection_name,
                properties=[
                    wc.Property(name="speaker_id", data_type=wc.DataType.INT),
                    wc.Property(name="full_name", data_type=wc.DataType.TEXT),
                    wc.Property(name="nickname", data_type=wc.DataType.TEXT),
                    wc.Property(name="gender", data_type=wc.DataType.TEXT),
                    wc.Property(name="created_at", data_type=wc.DataType.DATE),
                    wc.Property(name="last_active_at", data_type=wc.DataType.DATE),
                    wc.Property(name="meet_count", data_type=wc.DataType.INT),
                    wc.Property(name="meet_days", data_type=wc.DataType.INT),
                    wc.Property(name="voiceprint_ids", data_type=wc.DataType.UUID_ARRAY),
                    wc.Property(name="first_audio", data_type=wc.DataType.TEXT),
                ],
                vectorizer_config=wc.Configure.Vectorizer.none()
            )
            logger.info(f"成功建立 {collection_name} V2 集合")
            return True
            
        except Exception as e:
            logger.error(f"建立 {collection_name} V2 集合時發生錯誤: {str(e)}")
            return False
    
    def create_session(self) -> bool:
        """
        建立 Session 集合（取代 Meeting）
        
        Returns:
            bool: 是否成功建立
        """
        if not self.client:
            raise RuntimeError("客戶端未連接，請先調用 connect() 方法")
        
        try:
            collection_name = "Session"
            
            # 檢查是否已存在，如果存在且是冪等運行則跳過
            if self.collection_exists(collection_name):
                logger.info(f"集合 '{collection_name}' 已存在，跳過建立")
                return True
            
            # 建立 Session 集合
            session_collection = self.client.collections.create(
                name=collection_name,
                properties=[
                    wc.Property(name="session_id", data_type=wc.DataType.TEXT),
                    wc.Property(name="session_type", data_type=wc.DataType.TEXT),
                    wc.Property(name="title", data_type=wc.DataType.TEXT),  # 語意搜尋
                    wc.Property(name="start_time", data_type=wc.DataType.DATE),
                    wc.Property(name="end_time", data_type=wc.DataType.DATE),
                    wc.Property(name="summary", data_type=wc.DataType.TEXT),    # 語意搜尋
                ],
                references=[
                    wc.ReferenceProperty(
                        name="participants",
                        target_collection="Speaker"
                    )
                ],
                vectorizer_config=[
                    wc.Configure.NamedVectors.text2vec_transformers(      
                        name="text_emb",                                   #   任意命名
                        source_properties=["title", "summary"],
                        vectorize_collection_name=False,
                    )
                ]
            )
            logger.info(f"成功建立 {collection_name} 集合")
            return True
            
        except Exception as e:
            logger.error(f"建立 {collection_name} 集合時發生錯誤: {str(e)}")
            return False
    
    def create_speechlog(self) -> bool:
        """
        建立 SpeechLog 集合（一句話記錄）
        
        Returns:
            bool: 是否成功建立
        """
        if not self.client:
            raise RuntimeError("客戶端未連接，請先調用 connect() 方法")
        
        try:
            collection_name = "SpeechLog"
            
            # 檢查是否已存在，如果存在且是冪等運行則跳過
            if self.collection_exists(collection_name):
                logger.info(f"集合 '{collection_name}' 已存在，跳過建立")
                return True
            
            # 建立 SpeechLog 集合
            speechlog_collection = self.client.collections.create(
                name=collection_name,
                properties=[
                    wc.Property(name="content", data_type=wc.DataType.TEXT),    # 語義搜尋
                    wc.Property(name="timestamp", data_type=wc.DataType.DATE),
                    wc.Property(name="confidence", data_type=wc.DataType.NUMBER),
                    wc.Property(name="duration", data_type=wc.DataType.NUMBER),
                    wc.Property(name="language", data_type=wc.DataType.TEXT),
                ],
                references=[
                    wc.ReferenceProperty(
                        name="speaker",
                        target_collection="Speaker"
                    ),
                    wc.ReferenceProperty(
                        name="session",
                        target_collection="Session"
                    )
                ],
                vectorizer_config=[
                    wc.Configure.NamedVectors.text2vec_transformers(     
                        name="text_emb",
                        source_properties=["content"],
                        vectorize_collection_name=False,
                    )
                ]
            )
            logger.info(f"成功建立 {collection_name} 集合")
            return True
            
        except Exception as e:
            logger.error(f"建立 {collection_name} 集合時發生錯誤: {str(e)}")
            return False
    
    def create_voiceprint_v2(self) -> bool:
        """
        建立 VoicePrint V2 集合（增加 update_count、speaker_name）
        
        Returns:
            bool: 是否成功建立
        """
        if not self.client:
            raise RuntimeError("客戶端未連接，請先調用 connect() 方法")
        
        try:
            collection_name = "VoicePrint"
            
            # 檢查是否已存在，如果存在且是冪等運行則跳過
            if self.collection_exists(collection_name):
                logger.info(f"集合 '{collection_name}' 已存在，跳過建立")
                return True
            
            # 建立 VoicePrint V2 集合
            voiceprint_collection = self.client.collections.create(
                name=collection_name,
                properties=[
                    wc.Property(name="created_at", data_type=wc.DataType.DATE),
                    wc.Property(name="updated_at", data_type=wc.DataType.DATE),
                    wc.Property(name="update_count", data_type=wc.DataType.INT),
                    wc.Property(name="sample_count", data_type=wc.DataType.INT),
                    wc.Property(name="quality_score", data_type=wc.DataType.NUMBER),
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
            logger.info(f"成功建立 {collection_name} V2 集合")
            return True
            
        except Exception as e:
            logger.error(f"建立 {collection_name} V2 集合時發生錯誤: {str(e)}")
            return False
    
    def create_all_v2_collections(self) -> bool:
        """
        建立所有 V2 必要的集合（按正確順序）
        
        Returns:
            bool: 是否成功建立所有集合
        """
        try:
            logger.info("開始建立所有 V2 集合...")
            
            # 1. 先建立 Speaker 集合（因為其他集合需要引用它）
            if not self.create_speaker_v2():
                logger.error("建立 Speaker V2 集合失敗")
                return False
            
            # 2. 建立 Session 集合（引用 Speaker）
            if not self.create_session():
                logger.error("建立 Session 集合失敗")
                return False
            
            # 3. 建立 SpeechLog 集合（引用 Speaker 和 Session）
            if not self.create_speechlog():
                logger.error("建立 SpeechLog 集合失敗")
                return False
            
            # 4. 建立 VoicePrint V2 集合（引用 Speaker）
            if not self.create_voiceprint_v2():
                logger.error("建立 VoicePrint V2 集合失敗")
                return False
            
            logger.info("成功建立所有 Weaviate V2 集合")
            return True
            
        except Exception as e:
            logger.error(f"建立 V2 集合時發生錯誤: {str(e)}")
            return False
    
    def verify_v2_collections(self) -> Dict[str, bool]:
        """
        驗證所有 V2 集合是否存在
        
        Returns:
            Dict[str, bool]: 集合名稱與存在狀態的對應
        """
        collections = ["Speaker", "Session", "SpeechLog", "VoicePrint"]
        results = {}
        
        logger.info("開始驗證 V2 集合...")
        for collection_name in collections:
            try:
                results[collection_name] = self.collection_exists(collection_name)
                status = "✅" if results[collection_name] else "❌"
                logger.info(f"{status} {collection_name}: {results[collection_name]}")
            except Exception as e:
                logger.error(f"驗證集合 '{collection_name}' 時發生錯誤: {str(e)}")
                results[collection_name] = False
        
        return results


def ensure_weaviate_v2_collections(host: str = "localhost", port: int = 8080) -> bool:
    """
    確認 Weaviate V2 collections 存在，若不存在則建立（具備冪等性）
    
    Args:
        host: Weaviate 服務器主機地址
        port: Weaviate 服務器端口
        
    Returns:
        bool: 是否成功確保所有集合存在
    """
    logger.info("🚀 開始初始化 Weaviate V2 集合...")
    logger.warning("⚠️  重要變更警告：本次重構將大幅改變資料庫結構！")
    
    try:
        with WeaviateV2CollectionManager(host, port) as manager:
            # 驗證現有集合
            existing_collections = manager.verify_v2_collections()
            all_exist = all(existing_collections.values())
            
            if all_exist:
                logger.info("✅ 所有必要的 V2 集合已存在，無需建立")
            else:
                logger.info("🔧 正在建立缺失的 Weaviate V2 集合...")
                
                # 顯示進度條
                with tqdm(total=4, desc="建立集合", unit="collection") as pbar:
                    if not existing_collections.get("Speaker", False):
                        if manager.create_speaker_v2():
                            pbar.update(1)
                        else:
                            logger.error("❌ 建立 Speaker V2 集合失敗")
                            return False
                    else:
                        pbar.update(1)
                    
                    if not existing_collections.get("Session", False):
                        if manager.create_session():
                            pbar.update(1)
                        else:
                            logger.error("❌ 建立 Session 集合失敗")
                            return False
                    else:
                        pbar.update(1)
                    
                    if not existing_collections.get("SpeechLog", False):
                        if manager.create_speechlog():
                            pbar.update(1)
                        else:
                            logger.error("❌ 建立 SpeechLog 集合失敗")
                            return False
                    else:
                        pbar.update(1)
                    
                    if not existing_collections.get("VoicePrint", False):
                        if manager.create_voiceprint_v2():
                            pbar.update(1)
                        else:
                            logger.error("❌ 建立 VoicePrint V2 集合失敗")
                            return False
                    else:
                        pbar.update(1)
            
            # 最終驗證
            final_status = manager.verify_v2_collections()
            
            if all(final_status.values()):
                logger.info("🎉 所有 Weaviate V2 集合已成功建立並驗證")
                return True
            else:
                logger.error("❌ 集合建立後驗證失敗")
                logger.error(f"失敗的集合: {[k for k, v in final_status.items() if not v]}")
                return False
                
    except ConnectionError as e:
        logger.error(f"❌ 連接錯誤: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ 初始化過程中發生未預期錯誤: {str(e)}")
        return False


def main() -> None:
    """主函數 - 初始化 Weaviate V2 集合"""
    import argparse
    
    parser = argparse.ArgumentParser(description="初始化 Weaviate V2 集合")
    parser.add_argument("--host", default="localhost", help="Weaviate 服務器主機地址")
    parser.add_argument("--port", type=int, default=8080, help="Weaviate 服務器端口")
    
    args = parser.parse_args()
    
    success = ensure_weaviate_v2_collections(
        host=args.host,
        port=args.port
    )
    
    if success:
        print("🎉 Weaviate V2 集合初始化成功！")
        print("📋 建立的集合:")
        print("   - Speaker V2 (包含 speaker_id INT)")
        print("   - Session (取代 Meeting)")  
        print("   - SpeechLog (正規化語音記錄)")
        print("   - VoicePrint V2 (增強版聲紋)")
        print("\n💡 如需測試功能，請執行：")
        print("   python -m utils.test_init_v2_collections")
        sys.exit(0)
    else:
        print("❌ Weaviate V2 集合初始化失敗！")
        print("請檢查錯誤日誌並確保 Weaviate 服務器正在運行")
        sys.exit(1)


if __name__ == "__main__":
    main()
