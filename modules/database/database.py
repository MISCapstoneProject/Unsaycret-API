"""
===============================================================================
語者與聲紋資料庫接口 (Speaker and Voiceprint Database Interface)
===============================================================================

版本：v1.0.1 
作者：CYouuu
最後更新：2025-05-19

主要功能接口：
-----------
【語者管理】
- list_all_speakers()：列出所有語者
- get_speaker(speaker_id)：獲取語者信息
- create_speaker(speaker_name)：創建新語者
- update_speaker_name(speaker_id, new_name)：更新語者名稱
- update_speaker_last_active(speaker_id, timestamp)：更新活動時間
- delete_speaker(speaker_id)：刪除語者

【聲紋管理】
- create_voiceprint(speaker_id, embedding, audio_source, timestamp)：建立聲紋
- get_voiceprint(voiceprint_id, include_vector)：獲取聲紋
- update_voiceprint(voiceprint_id, new_embedding, update_count)：更新聲紋
- delete_voiceprint(voiceprint_id)：刪除聲紋
- get_speaker_voiceprints(speaker_id, include_vectors)：獲取語者的所有聲紋
- get_speaker_id_from_voiceprint(voiceprint_id)：獲取聲紋關聯的語者

【向量搜索】
- find_similar_voiceprints(embedding, limit)：搜索相似聲紋

【語者與聲紋關聯】
- add_voiceprint_to_speaker(speaker_id, voiceprint_id)：添加聲紋到語者
- transfer_voiceprints(source_id, dest_id, voiceprint_ids)：轉移聲紋

【其他工具】
- check_database_connection()：檢查數據庫連接
- check_collection_exists(collection_name)：檢查集合是否存在
- database_cleanup()：數據庫清理

功能摘要：
-----------
本模組提供與 Weaviate 資料庫交互的統一介面，集中所有資料庫操作，
使其他模組（如 VID_identify.py 和 VID_manager.py）可以通過抽象接口
訪問資料庫功能，而不直接操作資料庫底層。主要功能包括：

 1. 連接管理：建立與維護 Weaviate 資料庫連接
 2. 語者操作：建立、讀取、更新、刪除語者資訊
 3. 聲紋操作：建立、讀取、更新、刪除聲紋向量
 4. 向量搜索：基於相似度的語者匹配與搜索
 5. 資料一致性：確保語者和聲紋資料之間的引用完整性

技術架構：
-----------
 - 資料庫：Weaviate 向量資料庫
 - 架構：單例模式，確保全局只有一個資料庫連接實例
 - 接口：提供清晰的函數接口，封裝底層實現細節

使用方式：
-----------
1. 初始化資料庫連接：
   ```python
   from VID_database import DatabaseService
   
   db = DatabaseService()
   ```

2. 語者操作：
   ```python
   # 創建新語者
   speaker_id = db.create_speaker("王小明")
   
   # 列出所有語者
   speakers = db.list_speakers()
   
   # 獲取特定語者
   speaker = db.get_speaker(speaker_id)
   ```

3. 聲紋操作：
   ```python
   # 添加聲紋向量到語者
   voiceprint_id = db.add_voiceprint(speaker_id, embedding_vector)
   
   # 更新聲紋向量
   db.update_voiceprint(voiceprint_id, new_embedding)
   ```

4. 向量搜索：
   ```python
   # 搜索最相似的聲紋
   results = db.find_similar_voiceprints(embedding_vector)
   ```

前置需求：
-----------
 - Python 3.9+
 - weaviate-client 套件 v4+

注意事項：
-----------
 - 使用前請確保 Weaviate 已啟動並初始化必要集合
 - 本模組為單例模式，全局共享同一連接實例
 - 資料庫操作異常由內部處理並適當記錄，呼叫者可檢查返回值判斷操作是否成功

===============================================================================
"""

import os
import re
import sys
import uuid
import logging
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime, timezone, timedelta
import numpy as np
from collections import defaultdict

try:
    import weaviate
    from weaviate.classes.query import Filter, MetadataQuery, QueryReference
except ImportError:
    print("請先安裝 weaviate-client：pip install weaviate-client")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging 設定
# ---------------------------------------------------------------------------
from utils.logger import get_logger

# 創建本模組的日誌器
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 常數與輔助函數
# ---------------------------------------------------------------------------
# RFC3339 格式時間處理
def format_rfc3339(dt: Optional[datetime] = None) -> str:
    """
    將 datetime 轉換為 RFC3339 格式字串，用於 Weaviate 時間戳記。
    若未提供時間，則使用當前時間。
    
    Args:
        dt: 要格式化的 datetime 物件，若 None 則使用當前時間
        
    Returns:
        str: RFC3339 格式的時間字串
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

# UUID 驗證工具
UUID_PATTERN = re.compile(r"^[0-9a-fA-F-]{36}$")

def valid_uuid(value: str) -> bool:
    """檢查字串是否為有效 UUID 格式。"""
    if not value:
        return False
    return bool(UUID_PATTERN.match(value))

# 默認常數
DEFAULT_SPEAKER_NAME = "未命名語者"

# ---------------------------------------------------------------------------
# 資料庫服務類 (單例模式)
# ---------------------------------------------------------------------------
class DatabaseService:
    """
    提供與 Weaviate 資料庫交互的統一接口。
    實現單例模式，確保全局只有一個實例。
    """
    _instance = None
    _initialized = False
    
    # Weaviate 集合（類）名稱
    SPEAKER_CLASS = "Speaker"
    VOICEPRINT_CLASS = "VoicePrint"
    
    def __new__(cls) -> 'DatabaseService':
        """實現單例模式，確保全局只有一個資料庫連接實例。"""
        if cls._instance is None:
            cls._instance = super(DatabaseService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """初始化資料庫連接。如果已初始化，則跳過。"""
        if DatabaseService._initialized:
            return
            
        try:
            # 從環境變數讀取 Weaviate 配置
            from utils.docker_config import get_env_config
            config = get_env_config()
            
            weaviate_host = config["WEAVIATE_HOST"]
            weaviate_port = config["WEAVIATE_PORT"]
            weaviate_scheme = config["WEAVIATE_SCHEME"]
            
            # 根據環境選擇連接方式
            if weaviate_host == "localhost":
                # 本地開發環境
                self.client = weaviate.connect_to_local()
                logger.info("成功連接到本地 Weaviate 資料庫")
            else:
                # Docker 環境或遠端連接
                self.client = weaviate.connect_to_custom(
                    http_host=weaviate_host,
                    http_port=weaviate_port,
                    http_secure=weaviate_scheme == "https"
                )
                logger.info(f"成功連接到 Weaviate 資料庫 ({weaviate_scheme}://{weaviate_host}:{weaviate_port})")
            
            # 檢查必要的集合是否存在
            if not self.client.collections.exists(self.SPEAKER_CLASS) or \
               not self.client.collections.exists(self.VOICEPRINT_CLASS):
                logger.warning(f"Weaviate 中缺少必要的集合 ({self.SPEAKER_CLASS} / {self.VOICEPRINT_CLASS})!")
                logger.warning("請先運行 weaviate_study/create_collections.py 建立所需的集合")
            
        except Exception as e:
            logger.error(f"連接 Weaviate 資料庫失敗: {e}")
            logger.error("請確認：")
            logger.error("1. Docker 服務是否正在運行")
            logger.error("2. Weaviate 容器是否已經啟動")
            logger.error("3. weaviate_study/docker-compose.yml 中的配置是否正確")
            raise
        
        DatabaseService._initialized = True
    
    def close(self) -> None:
        """關閉資料庫連接。"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("已關閉 Weaviate 連接")
    
    # -------------------------------------------------------------------------
    # 語者 (Speaker) 相關操作
    # -------------------------------------------------------------------------
    
    def list_all_speakers(self) -> List[Dict[str, Any]]:
        """
        列出所有語者及其基本資訊。
        
        Returns:
            List[Dict[str, Any]]: 語者列表，每個項目包含 uuid、name、create_time 等資訊
        """
        try:
            results = (
                self.client.collections.get(self.SPEAKER_CLASS)
                .query.fetch_objects()
            )
            speakers = []
            for obj in results.objects:
                voiceprint_ids = obj.properties.get("voiceprint_ids", [])
                speakers.append({
                    "uuid": str(obj.uuid),
                    "name": obj.properties.get("name", "未命名"),
                    "create_time": obj.properties.get("create_time", "未知"),
                    "last_active_time": obj.properties.get("last_active_time", "未知"),
                    "first_audio_id": obj.properties.get("first_audio_id"),
                    "voiceprint_count": len(voiceprint_ids),
                    "voiceprint_ids": voiceprint_ids,
                })
            speakers.sort(key=lambda s: s["last_active_time"], reverse=True)
            return speakers
        except Exception as exc:
            logger.error(f"列出語者時發生錯誤: {exc}")
            return []
    
    def get_speaker(self, speaker_id: str) -> Optional[Any]:
        """
        獲取特定語者的詳細資訊。
        
        Args:
            speaker_id: 語者的 UUID
            
        Returns:
            Optional[Any]: 語者物件，若找不到則返回 None
        """
        try:
            if not valid_uuid(speaker_id):
                logger.error(f"無效的語者 ID 格式: {speaker_id}")
                return None
                
            return (
                self.client.collections.get(self.SPEAKER_CLASS)
                .query.fetch_object_by_id(uuid=speaker_id)
            )
        except Exception as exc:
            logger.error(f"獲取語者詳細資訊時發生錯誤: {exc}")
            return None
    
    def create_speaker(self, speaker_name: str = DEFAULT_SPEAKER_NAME, first_audio_id: Optional[str] = None) -> str:
        """
        創建新的語者。
        
        Args:
            speaker_name: 語者名稱，默認為「未命名語者」
            first_audio_id: 第一次生成該語者時使用的音檔ID（UUID格式）
            
        Returns:
            str: 新建立的語者 ID，若建立失敗則返回空字符串
        """
        try:
            # 創建新的語者
            speaker_collection = self.client.collections.get(self.SPEAKER_CLASS)
            speaker_id = str(uuid.uuid4())
            
            # 如果是默認名稱，生成唯一的名稱 (類似 n1, n2, ...)
            if speaker_name == DEFAULT_SPEAKER_NAME:
                # 獲取所有語者
                results = speaker_collection.query.fetch_objects(
                    limit=100,
                    return_properties=["name"],
                )
                
                # 提取所有以 'n' 開頭的數字
                numbers = []
                pattern = re.compile(r'^n(\d+)')
                for obj in results.objects:
                    name = obj.properties.get("name", "")
                    match = pattern.match(name)
                    if match:
                        numbers.append(int(match.group(1)))
                
                # 生成下一個編號
                next_number = max(numbers) + 1 if numbers else 1
                speaker_name = f"n{next_number}"
            
            # 創建語者
            properties = {
                "name": speaker_name,
                "create_time": format_rfc3339(),
                "last_active_time": format_rfc3339(),
                "voiceprint_ids": []  # 初始時沒有聲紋向量
            }
            
            # 如果提供了 first_audio_id，則加入屬性中
            if first_audio_id:
                properties["first_audio_id"] = first_audio_id
            
            speaker_collection.data.insert(
                properties=properties,
                uuid=speaker_id
            )
            
            logger.info(f"已建立新語者 {speaker_name} (ID: {speaker_id})")
            return speaker_id
            
        except Exception as e:
            logger.error(f"創建新語者時發生錯誤: {e}")
            return ""
    
    def update_speaker_name(self, speaker_id: str, new_name: str) -> bool:
        """
        更改語者名稱，並同步更新所有該語者底下聲紋的 speaker_name。
        
        Args:
            speaker_id: 語者 ID
            new_name: 新的語者名稱
            
        Returns:
            bool: 是否更新成功
        """
        try:
            if not valid_uuid(speaker_id):
                logger.error(f"無效的語者 ID 格式: {speaker_id}")
                return False
                
            # 1.先更新 Speaker 本身
            sp_col = self.client.collections.get(self.SPEAKER_CLASS)
            sp_col.data.update(uuid=speaker_id, properties={"name": new_name})

            # 2.拿回這個 Speaker 物件，讀出 voiceprint_ids
            sp_obj = sp_col.query.fetch_object_by_id(uuid=speaker_id)
            if not sp_obj:
                logger.error(f"找不到語者 (ID: {speaker_id})")
                return False
                
            vp_ids = sp_obj.properties.get("voiceprint_ids", [])

            # 3.逐一更新每支 VoicePrint
            vp_col = self.client.collections.get(self.VOICEPRINT_CLASS)
            for vp_id in vp_ids:
                vp_col.data.update(
                    uuid=vp_id,
                    properties={"speaker_name": new_name}
                )

            logger.info(f"已更新語者 {speaker_id} 的名稱為 {new_name}")
            return True
        except Exception as exc:
            logger.error(f"更改語者名稱時發生錯誤: {exc}")
            return False
    
    def update_speaker_last_active(self, speaker_id: str, timestamp: Optional[datetime] = None) -> bool:
        """
        更新語者的最後活動時間。
        
        Args:
            speaker_id: 語者 ID
            timestamp: 時間戳記，若為 None 則使用當前時間
            
        Returns:
            bool: 是否更新成功
        """
        try:
            if not valid_uuid(speaker_id):
                logger.error(f"無效的語者 ID 格式: {speaker_id}")
                return False
                
            time_str = format_rfc3339(timestamp) if timestamp else format_rfc3339()
            
            sp_col = self.client.collections.get(self.SPEAKER_CLASS)
            sp_col.data.update(
                uuid=speaker_id, 
                properties={"last_active_time": time_str}
            )
            
            return True
        except Exception as exc:
            logger.error(f"更新語者最後活動時間時發生錯誤: {exc}")
            return False
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """
        刪除語者，同時刪除該語者底下的所有聲紋。
        
        Args:
            speaker_id: 語者 ID
            
        Returns:
            bool: 是否刪除成功
        """
        try:
            if not valid_uuid(speaker_id):
                logger.error(f"無效的語者 ID 格式: {speaker_id}")
                return False
            
            # 1. 獲取語者的所有聲紋
            speaker_collection = self.client.collections.get(self.SPEAKER_CLASS)
            speaker_obj = speaker_collection.query.fetch_object_by_id(
                uuid=speaker_id,
                return_properties=["name", "voiceprint_ids"]
            )
            
            if not speaker_obj:
                logger.error(f"找不到語者 (ID: {speaker_id})")
                return False
                
            speaker_name = speaker_obj.properties.get("name", "未命名")
            voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
            
            # 2. 刪除語者的所有聲紋
            deleted_count = 0
            voiceprint_collection = self.client.collections.get(self.VOICEPRINT_CLASS)
            for vp_id in voiceprint_ids:
                try:
                    voiceprint_collection.data.delete_by_id(uuid=vp_id)
                    deleted_count += 1
                except Exception as vp_exc:
                    logger.error(f"刪除聲紋 {vp_id} 時發生錯誤: {vp_exc}")
            
            # 3. 刪除語者本身
            speaker_collection.data.delete_by_id(uuid=speaker_id)
            
            logger.info(f"已刪除語者 {speaker_name} (ID: {speaker_id}) 及其 {deleted_count} 個聲紋")
            return True
        except Exception as exc:
            logger.error(f"刪除語者時發生錯誤: {exc}")
            return False
    
    def add_voiceprint_to_speaker(self, speaker_id: str, voiceprint_id: str) -> bool:
        """
        將聲紋向量添加到語者的聲紋列表中。
        
        Args:
            speaker_id: 語者 ID
            voiceprint_id: 聲紋向量 ID
            
        Returns:
            bool: 是否添加成功
        """
        try:
            if not valid_uuid(speaker_id) or not valid_uuid(voiceprint_id):
                logger.error(f"無效的 ID 格式: speaker_id={speaker_id}, voiceprint_id={voiceprint_id}")
                return False
                
            # 獲取語者的聲紋列表
            speaker_collection = self.client.collections.get(self.SPEAKER_CLASS)
            speaker_obj = speaker_collection.query.fetch_object_by_id(
                uuid=speaker_id,
                return_properties=["voiceprint_ids", "name"]
            )
            
            if not speaker_obj:
                logger.error(f"找不到語者 (ID: {speaker_id})")
                return False
                
            # 更新語者的聲紋列表
            voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
            if voiceprint_id not in voiceprint_ids:
                voiceprint_ids.append(voiceprint_id)
                
                speaker_collection.data.update(
                    uuid=speaker_id,
                    properties={
                        "voiceprint_ids": voiceprint_ids,
                        "last_active_time": format_rfc3339()
                    }
                )
                
                # 更新聲紋的語者名稱
                speaker_name = speaker_obj.properties.get("name", DEFAULT_SPEAKER_NAME)
                voiceprint_collection = self.client.collections.get(self.VOICEPRINT_CLASS)
                voiceprint_collection.data.update(
                    uuid=voiceprint_id,
                    properties={"speaker_name": speaker_name},
                    references={"speaker": [speaker_id]}
                )
                
                logger.info(f"已將聲紋 {voiceprint_id} 添加到語者 {speaker_id} 的聲紋列表")
            
            return True
            
        except Exception as e:
            logger.error(f"添加聲紋到語者時發生錯誤: {e}")
            return False
    
    def transfer_voiceprints(
        self, source_id: str, dest_id: str, voiceprint_ids: Optional[List[str]] = None
    ) -> bool:
        """
        將聲紋從一個語者轉移到另一個語者。
        
        Args:
            source_id: 來源語者 ID
            dest_id: 目標語者 ID
            voiceprint_ids: 要轉移的聲紋 ID 列表，若為 None 則轉移全部
            
        Returns:
            bool: 是否轉移成功
        """
        try:
            if not valid_uuid(source_id) or not valid_uuid(dest_id):
                logger.error(f"無效的語者 ID 格式: source_id={source_id}, dest_id={dest_id}")
                return False
                
            collection = self.client.collections.get(self.SPEAKER_CLASS)
            src_obj = collection.query.fetch_object_by_id(uuid=source_id)
            dest_obj = collection.query.fetch_object_by_id(uuid=dest_id)
            
            if not src_obj or not dest_obj:
                logger.warning("來源或目標語者不存在。")
                return False
            
            src_vps = set(src_obj.properties.get("voiceprint_ids", []))
            dest_vps = set(dest_obj.properties.get("voiceprint_ids", []))
            move_set = set(src_vps) if voiceprint_ids is None else set(voiceprint_ids).intersection(src_vps)
            dest_vps.update(move_set)   #聯集，把 move_set 中的元素加入到 dest_vps 中
            src_vps.difference_update(move_set) #差集，把 move_set 中的元素從 src_vps 中刪除
            
            # 更新來源與目標語者的聲紋
            collection.data.update(uuid=source_id, properties={"voiceprint_ids": list(src_vps)})
            collection.data.update(uuid=dest_id, properties={"voiceprint_ids": list(dest_vps)})
            
            # 取得目標語者名稱
            dest_name = dest_obj.properties.get("name", "未命名")
            
            # 批次更新被轉移聲紋的 speaker_id 與 speaker_name
            vp_collection = self.client.collections.get(self.VOICEPRINT_CLASS)
            for vp_id in move_set:
                try:
                    vp_collection.data.update(uuid=vp_id, properties={
                        "speaker_name": dest_name
                    }, references={"speaker": [dest_id]})
                except Exception as e:
                    logger.error(f"轉移聲紋 {vp_id} 時發生錯誤: {e}")
            
            # 若來源語者已無聲紋，自動刪除
            if not src_vps:
                try:
                    collection.data.delete_by_id(uuid=source_id)
                    logger.info(f"來源語者 {source_id} 已無聲紋，自動刪除。")
                except Exception as del_exc:
                    logger.error(f"自動刪除來源語者時發生錯誤: {del_exc}")
                    
            logger.info(f"已成功將 {len(move_set)} 個聲紋從語者 {source_id} 轉移到語者 {dest_id}")
            return True
        except Exception as exc:
            logger.error(f"轉移聲紋時發生錯誤: {exc}")
            return False
    
    # -------------------------------------------------------------------------
    # 聲紋 (VoicePrint) 相關操作
    # -------------------------------------------------------------------------
    
    def create_voiceprint(
        self, 
        speaker_id: str, 
        embedding: np.ndarray,
        audio_source: str = "",
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        為語者創建新的聲紋向量。
        
        Args:
            speaker_id: 語者 ID
            embedding: 聲紋嵌入向量
            audio_source: 音訊來源描述
            timestamp: 時間戳記，用於設定聲紋的創建時間和更新時間
            
        Returns:
            str: 新建立的聲紋向量 ID，若創建失敗則返回空字符串
        """
        try:
            if not valid_uuid(speaker_id):
                logger.error(f"無效的語者 ID 格式: {speaker_id}")
                return ""
                
            # 獲取語者資訊
            speaker_collection = self.client.collections.get(self.SPEAKER_CLASS)
            speaker_obj = speaker_collection.query.fetch_object_by_id(
                uuid=speaker_id,
                return_properties=["name", "voiceprint_ids"]
            )
            
            if not speaker_obj:
                logger.error(f"找不到語者 (ID: {speaker_id})")
                return ""
                
            speaker_name = speaker_obj.properties.get("name", DEFAULT_SPEAKER_NAME)
            voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
            
            # 格式化時間或使用當前時間
            time_str = format_rfc3339(timestamp) if timestamp else format_rfc3339()
            
            # 創建新的聲紋向量
            voiceprint_collection = self.client.collections.get(self.VOICEPRINT_CLASS)
            voiceprint_id = str(uuid.uuid4())
            
            voiceprint_collection.data.insert(
                properties={
                    "create_time": time_str,
                    "updated_time": time_str,
                    "update_count": 1,
                    "speaker_name": speaker_name,
                    "audio_source": audio_source
                },
                uuid=voiceprint_id,
                vector=embedding.tolist(),
                references={
                    "speaker": [speaker_id]
                }
            )
            
            # 更新語者的聲紋列表
            voiceprint_ids.append(voiceprint_id)
            speaker_collection.data.update(
                uuid=speaker_id,
                properties={
                    "voiceprint_ids": voiceprint_ids,
                    "last_active_time": time_str
                }
            )
            
            logger.info(f"已為語者 {speaker_name} 創建新的聲紋向量 (ID: {voiceprint_id})")
            return voiceprint_id
            
        except Exception as e:
            logger.error(f"創建聲紋向量時發生錯誤: {e}")
            return ""
    
    
    def get_voiceprint(self, voiceprint_id: str,
                include_vector: bool = False,
                include_refs: bool = False)-> Optional[Any]:
        """
            獲取特定聲紋向量的詳細資訊。
            
            Args:
                voiceprint_id: 聲紋向量 ID
                include_vector: 是否包含向量數據
                include_refs: 是否包含引用的語者 ID
                
            Returns:
                Optional[Any]: 聲紋向量物件，若找不到則返回 None
            """
        if not valid_uuid(voiceprint_id):
            logger.error("無效的 VoicePrint UUID: %s", voiceprint_id)
            return None

        coll = self.client.collections.get(self.VOICEPRINT_CLASS)

        try:
            return coll.query.fetch_object_by_id(
                uuid=voiceprint_id,
                include_vector=include_vector,
                return_references=(
                    QueryReference(link_on="speaker") if include_refs else None
                ),
            )
        except Exception as e:
            logger.error("抓取 VoicePrint %s 失敗: %s", voiceprint_id[:8], e)
            return None
    
    def get_voiceprint_properties(self, voiceprint_id: str, properties: List[str]) -> Optional[Dict[str, Any]]:
        """
        獲取聲紋向量的特定屬性。
        
        Args:
            voiceprint_id: 聲紋向量 ID
            properties: 需要獲取的屬性列表
            
        Returns:
            Optional[Dict[str, Any]]: 屬性字典，若找不到則返回 None
        """
        try:
            if not valid_uuid(voiceprint_id):
                logger.error(f"無效的聲紋向量 ID 格式: {voiceprint_id}")
                return None
                
            voiceprint_collection = self.client.collections.get(self.VOICEPRINT_CLASS)
            result = voiceprint_collection.query.fetch_object_by_id(
                uuid=voiceprint_id,
                return_properties=properties
            )
            
            if not result:
                return None
                
            return result.properties
            
        except Exception as e:
            logger.error(f"獲取聲紋向量屬性時發生錯誤: {e}")
            return None
    
    def update_voiceprint(self, voiceprint_id: str, new_embedding: np.ndarray, update_count: Optional[int] = None) -> int:
        """
        使用加權移動平均更新現有的聲紋向量。
        
        Args:
            voiceprint_id: 聲紋向量 ID
            new_embedding: 新的嵌入向量
            update_count: 當前更新次數，若為 None 則從資料庫讀取
            
        Returns:
            int: 更新後的次數，若更新失敗則返回 0
        """
        try:
            if not valid_uuid(voiceprint_id):
                logger.error(f"無效的聲紋向量 ID 格式: {voiceprint_id}")
                return 0
                
            # 獲取現有的嵌入向量
            voiceprint_collection = self.client.collections.get(self.VOICEPRINT_CLASS)
            existing_object = voiceprint_collection.query.fetch_object_by_id(
                uuid=voiceprint_id,
                include_vector=True
            )
            
            if not existing_object:
                logger.error(f"找不到 ID 為 {voiceprint_id} 的聲紋向量")
                return 0
            
            # 如果未提供 update_count，則從資料庫讀取
            if update_count is None:
                update_count = existing_object.properties.get("update_count")
                if update_count is None:
                    logger.error(f"無法獲取聲紋向量 {voiceprint_id} 的更新計數")
                    return 0
            
            # 獲取現有的嵌入向量            
            vec_dict = existing_object.vector
            raw_old = vec_dict["default"] if isinstance(vec_dict, dict) else vec_dict
            old_embedding = np.array(raw_old, dtype=float)
            
            # 使用加權移動平均更新嵌入向量
            updated_embedding = (old_embedding * update_count + new_embedding) / (update_count + 1)
            new_update_count = update_count + 1
            
            # 更新資料庫中的向量
            voiceprint_collection.data.update(
                uuid=voiceprint_id,
                properties={
                    "updated_time": format_rfc3339(),
                    "update_count": new_update_count
                },
                vector=updated_embedding.tolist()
            )
            
            logger.info(f"已更新聲紋向量 {voiceprint_id}，新的更新次數: {new_update_count}")
            return new_update_count
            
        except Exception as e:
            logger.error(f"更新聲紋向量時發生錯誤: {e}")
            return 0
    
    def delete_voiceprint(self, voiceprint_id: str) -> bool:
        """
        刪除聲紋向量，並從相關語者的列表中移除。
        
        Args:
            voiceprint_id: 聲紋向量 ID
            
        Returns:
            bool: 是否刪除成功
        """
        try:
            if not valid_uuid(voiceprint_id):
                logger.error(f"無效的聲紋向量 ID 格式: {voiceprint_id}")
                return False
                
            # 獲取聲紋關聯的語者 ID
            speaker_id = self.get_speaker_id_from_voiceprint(voiceprint_id)
            
            # 如果找到關聯的語者，從語者的聲紋列表中移除
            if speaker_id:
                speaker_collection = self.client.collections.get(self.SPEAKER_CLASS)
                speaker_obj = speaker_collection.query.fetch_object_by_id(
                    uuid=speaker_id,
                    return_properties=["voiceprint_ids"]
                )
                
                if speaker_obj:
                    voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
                    if voiceprint_id in voiceprint_ids:
                        voiceprint_ids.remove(voiceprint_id)
                        
                        speaker_collection.data.update(
                            uuid=speaker_id,
                            properties={"voiceprint_ids": voiceprint_ids}
                        )
            
            # 刪除聲紋向量
            voiceprint_collection = self.client.collections.get(self.VOICEPRINT_CLASS)
            voiceprint_collection.data.delete_by_id(uuid=voiceprint_id)
            
            logger.info(f"已刪除聲紋向量 {voiceprint_id}")
            return True
            
        except Exception as e:
            logger.error(f"刪除聲紋向量時發生錯誤: {e}")
            return False
    
    def get_speaker_id_from_voiceprint(self, voiceprint_id: str) -> str:
        """
        根據聲紋向量 ID 獲取關聯的語者 ID。
        
        Args:
            voiceprint_id: 聲紋向量 ID
            
        Returns:
            str: 語者 ID，若找不到則返回空字符串
        """
        try:
            if not valid_uuid(voiceprint_id):
                logger.error(f"無效的聲紋向量 ID 格式: {voiceprint_id}")
                return ""
                
            voiceprint_collection = self.client.collections.get(self.VOICEPRINT_CLASS)
            qr = QueryReference(
                link_on="speaker",
                return_properties=["uuid"]
            )
            
            voiceprint_obj = voiceprint_collection.query.fetch_object_by_id(
                uuid=voiceprint_id,
                return_references=qr
            )
            
            refs = voiceprint_obj.references.get("speaker", []).objects
            if not refs:
                return ""
                
            return refs[0].uuid
        except Exception as e:
            logger.error(f"獲取聲紋關聯的語者 ID 時發生錯誤: {e}")
            return ""
    
    def get_speaker_voiceprints(self, speaker_id: str, include_vectors: bool = False) -> List[Dict[str, Any]]:
        """
        獲取語者的所有聲紋向量。
        
        Args:
            speaker_id: 語者 ID
            include_vectors: 是否包含向量數據
            
        Returns:
            List[Dict[str, Any]]: 聲紋向量列表
        """
        try:
            if not valid_uuid(speaker_id):
                logger.error(f"無效的語者 ID 格式: {speaker_id}")
                return []
                
            # 獲取語者的聲紋列表
            speaker_obj = self.get_speaker(speaker_id)
            if not speaker_obj:
                logger.error(f"找不到語者 (ID: {speaker_id})")
                return []
                
            voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
            if not voiceprint_ids:
                return []
            
            # 獲取每個聲紋向量的詳細資訊
            voiceprint_collection = self.client.collections.get(self.VOICEPRINT_CLASS)
            voiceprints = []
            
            for vp_id in voiceprint_ids:
                try:
                    vp_obj = voiceprint_collection.query.fetch_object_by_id(
                        uuid=vp_id,
                        include_vector=include_vectors
                    )
                    
                    if vp_obj:
                        vp_data = {
                            "uuid": vp_obj.uuid,
                            "create_time": vp_obj.properties.get("create_time"),
                            "updated_time": vp_obj.properties.get("updated_time"),
                            "update_count": vp_obj.properties.get("update_count"),
                            "speaker_name": vp_obj.properties.get("speaker_name"),
                            "audio_source": vp_obj.properties.get("audio_source", "")
                        }
                        
                        if include_vectors:
                            vec_dict = vp_obj.vector
                            vp_data["vector"] = vec_dict["default"] if isinstance(vec_dict, dict) else vec_dict
                            
                        voiceprints.append(vp_data)
                except Exception as e:
                    logger.error(f"獲取聲紋向量 {vp_id} 時發生錯誤: {e}")
            
            return voiceprints
            
        except Exception as e:
            logger.error(f"獲取語者聲紋向量列表時發生錯誤: {e}")
            return []
    
    # -------------------------------------------------------------------------
    # 向量搜索相關操作
    # -------------------------------------------------------------------------
    
    def find_similar_voiceprints(
        self, 
        embedding: np.ndarray, 
        limit: int = 3
    ) -> Tuple[Optional[str], Optional[str], float, List[Tuple[str, str, float, int]]]:
        """
        比較新的嵌入向量與資料庫中所有現有嵌入向量的相似度。
        
        Args:
            embedding: 待比較的嵌入向量
            limit: 返回結果的數量限制
            
        Returns:
            Tuple: (最佳匹配ID, 最佳匹配語者名稱, 最小距離, 所有距離列表)
        """
        try:
            voiceprint_collection = self.client.collections.get(self.VOICEPRINT_CLASS)
            
            # 計算新向量與資料庫中所有向量的距離
            results = voiceprint_collection.query.near_vector(
                near_vector=embedding.tolist(),
                limit=limit,
                return_properties=["speaker_name", "update_count", "create_time", "updated_time"],
                return_metadata=MetadataQuery(distance=True)
            )
            
            # 如果沒有找到任何結果
            if not results.objects:
                logger.info("資料庫中尚無任何嵌入向量")
                return None, None, float('inf'), []
            
            # 處理結果，計算距離
            distances = []
            for obj in results.objects:
                # 距離信息可能在不同位置，根據 Weaviate 版本進行適配
                distance = None
                if hasattr(obj, 'metadata') and hasattr(obj.metadata, 'distance'):
                    distance = obj.metadata.distance
                
                # 處理 distance 可能是 None 的情況
                if distance is None:
                    distance = -1
                    logger.warning(f"無法從結果中獲取距離信息，使用預設值 {distance}")
                
                object_id = obj.uuid
                speaker_name = obj.properties.get("speaker_name")
                update_count = obj.properties.get("update_count")
                
                logger.debug(f"比對 - 語者: {speaker_name}, "
                          f"更新次數: {update_count}, 餘弦距離: {distance:.4f}")
                
                # 保存距離資訊
                distances.append((object_id, speaker_name, distance, update_count))
            
            # 找出最小距離
            if distances:
                best_match = min(distances, key=lambda x: x[2])
                best_id, best_name, best_distance, _ = best_match
                return best_id, best_name, best_distance, distances
            else:
                # 如果沒有有效的距離信息，返回空結果
                logger.warning("未能獲取有效的距離信息")
                return None, None, float('inf'), []
            
        except Exception as e:
            logger.error(f"比對嵌入向量時發生錯誤: {e}")
            return None, None, float('inf'), []
    
    # -------------------------------------------------------------------------
    # 其他輔助函數
    # -------------------------------------------------------------------------
    def database_cleanup(self) -> bool:
        """
        資料庫清理，包括檢查並修復無效引用等。
        
        Returns:
            bool: 是否清理成功
        """
        try:
            self.check_and_repair_database()
            return True
        except Exception as exc:
            logger.error(f"執行資料庫檢查與修復時發生錯誤: {exc}")
            return False
            
    def check_and_repair_database(self) -> Dict[str, Any]:
        """
        資料庫檢查與修復：
        1. 檢查每個 Speaker 的 voiceprint_ids 是否都存在，並移除不存在的 id。
        2. 檢查每個 VoicePrint 的 speaker_name 與 ReferenceProperty 是否正確，並修正。
        3. 找出沒有被任何 Speaker 參考的 VoicePrint（孤兒）。
        4. 如果能判斷歸屬，將孤兒 VoicePrint 自動掛回對應 Speaker。
        """
        report = {
            "step1_missing_vp_count": 0,
            "step2_error_vp_ids": [],
            "step3_unreferenced_vp_ids": [],
            "step4_relinked_vp": {},      
            "success": False
        }

        voice_print_coll = self.client.collections.get(self.VOICEPRINT_CLASS)

        try:
            # === 步驟 1：清理 Speaker → 移除不存在的 voiceprint_ids ==========
            speakers = self.list_all_speakers()
            if not speakers:
                logger.warning("資料庫沒有任何 Speaker。")
            speaker_id_to_name = {}
            all_speaker_vp_ids: set[str] = set()
            missing_vp_count = 0

            for sp in speakers:
                sp_id = sp["uuid"]
                sp_obj = self.get_speaker(sp_id)
                if not sp_obj:
                    continue

                props = sp_obj.properties
                speaker_id_to_name[sp_id] = props.get("name", "未命名")
                vp_ids = props.get("voiceprint_ids", [])
                if not vp_ids:                         # ❷ 空陣列 -> 不進一步檢查
                    continue

                valid_vp_ids, missing_ids = [], []
                for vp_id in vp_ids:
                    if voice_print_coll.data.exists(uuid=str(vp_id)):
                        valid_vp_ids.append(vp_id)
                        all_speaker_vp_ids.add(str(vp_id))
                    else:
                        missing_ids.append(vp_id)


                if missing_ids:
                    logger.info(f"[步驟1] Speaker {sp_id[:8]} 有 {len(missing_ids)} 個不存在的 voiceprint_ids，已移除。")
                    missing_vp_count += len(missing_ids)
                    # 只有在 valid_vp_ids 非空 or 原本有缺再寫回，避免誤覆蓋
                    self.client.collections.get(self.SPEAKER_CLASS).data.update(
                        uuid=sp_id,
                        properties={
                            "voiceprint_ids": valid_vp_ids,
                            "voiceprint_count": len(valid_vp_ids)
                        }
                    )

            report["step1_missing_vp_count"] = missing_vp_count

            # === 步驟 2：修正 VoicePrint → speaker_name / reference ==========
            error_vp_ids = []

            for sp in speakers:
                sp_id = sp["uuid"]
                sp_name = speaker_id_to_name.get(sp_id, "未命名")
                vp_ids = self.get_speaker(sp_id).properties.get("voiceprint_ids", [])

                for vp_id in vp_ids:
                    vp_obj = self.get_voiceprint(str(vp_id), include_refs=True)
                    all_speaker_vp_ids.add(str(vp_id))
                    if not vp_obj:
                        continue

                    need_update = False
                    # 2-1 speaker_name
                    if vp_obj.properties.get("speaker_name") != sp_name:
                        need_update = True
                    # 2-2 reference 檢查
                    ref_ids = {str(obj.uuid) for obj in
                            (vp_obj.references.get('speaker').objects
                                if vp_obj.references and vp_obj.references.get('speaker') else [])}
                    if ref_ids != {sp_id}:
                        need_update = True

                    if need_update:
                        error_vp_ids.append(vp_id)
                        voice_print_coll.data.update(
                            uuid=vp_id,
                            properties={"speaker_name": sp_name}
                        )
                        voice_print_coll.data.reference_replace(
                            from_uuid=vp_id,
                            from_property="speaker",
                            to=sp_id
                        )

            report["step2_error_vp_ids"] = error_vp_ids

            # === 步驟 3：找孤兒 VoicePrint =====================================
            all_voiceprints = voice_print_coll.query.fetch_objects()
            all_vp_ids = {str(obj.uuid) for obj in all_voiceprints.objects}
            orphan_vp_ids = all_vp_ids.difference(all_speaker_vp_ids)
            report["step3_unreferenced_vp_ids"] = list(orphan_vp_ids)            # === 步驟 4：能判斷歸屬者，自動掛回，無法確定歸屬者，刪除 ==========
            if orphan_vp_ids:
                # 快速對照：speaker_name -> (speaker_uuid, 現有 vp set)
                name_to_sp = {
                    name: (sp_uuid, set(self.get_speaker(sp_uuid)
                                        .properties.get("voiceprint_ids", [])))
                    for sp_uuid, name in speaker_id_to_name.items()
                }
                pending_updates = defaultdict(set)
                deleted_orphans = []
                
                for vp_obj in all_voiceprints.objects:
                    vp_id = str(vp_obj.uuid)
                    if vp_id not in orphan_vp_ids:
                        continue

                    sp_name = vp_obj.properties.get("speaker_name")
                    # 無法確定歸屬的情況，直接刪除
                    if not sp_name or sp_name not in name_to_sp:
                        try:
                            voice_print_coll.data.delete_by_id(uuid=vp_id)
                            deleted_orphans.append(vp_id)
                        except Exception as del_exc:
                            logger.error(f"刪除孤兒聲紋 {vp_id[:8]} 時發生錯誤: {del_exc}")
                        continue

                    sp_uuid, owned_vps = name_to_sp[sp_name]
                    # 交叉驗證 reference（可略過以提升速度）
                    refs = vp_obj.references.get("speaker") if vp_obj.references else None
                    ref_ok = not refs or {str(o.uuid) for o in refs.objects} == {sp_uuid}

                    if ref_ok and vp_id not in owned_vps:
                        pending_updates[sp_uuid].add(vp_id)

                # 批次寫回 Speaker
                speaker_coll = self.client.collections.get(self.SPEAKER_CLASS)
                for sp_uuid, add_set in pending_updates.items():
                    sp_obj = self.get_speaker(sp_uuid)
                    vp_ids = sp_obj.properties.get("voiceprint_ids", []) + list(add_set)
                    speaker_coll.data.update(
                        uuid=sp_uuid,
                        properties={
                            "voiceprint_ids": vp_ids,
                            "voiceprint_count": len(vp_ids)
                        }
                    )

                # 報表
                report["step4_relinked_vp"] = {k: list(v) for k, v in pending_updates.items()}
                report["step4_deleted_orphans"] = deleted_orphans

            report["success"] = True
            return report

        except Exception as exc:
            logger.exception("執行資料庫檢查與修復時發生錯誤")
            report["error"] = str(exc)
            return report
    
    def check_database_connection(self) -> bool:
        """
        檢查資料庫連接是否正常。
        
        Returns:
            bool: 連接是否正常
        """
        try:
            # 測試是否可以讀取集合
            self.client.collections.list_all()
            return True
        except Exception as e:
            logger.error(f"資料庫連接檢查失敗: {e}")
            return False
    
    def check_collection_exists(self, collection_name: str) -> bool:
        """
        檢查集合是否存在。
        
        Args:
            collection_name: 集合名稱
            
        Returns:
            bool: 集合是否存在
        """
        try:
            return self.client.collections.exists(collection_name)
        except Exception as e:
            logger.error(f"檢查集合 {collection_name} 是否存在時發生錯誤: {e}")
            return False


# 單元測試代碼
if __name__ == "__main__":
    # 簡單的功能測試
    try:
        db = DatabaseService()
        if db.check_database_connection():
            print("成功連接到 Weaviate 資料庫，並通過測試")
        else:
            print("資料庫連接測試失敗")
    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")
