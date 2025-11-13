"""
Weaviate CRUD 操作模組

功能：
1. Speaker CRUD（語者管理）
2. VoicePrint CRUD（聲紋管理）
3. 向量搜尋
4. Speaker ↔ VoicePrint 關聯管理

使用方式：
```python
from modules.database.weaviate_crud import (
    create_speaker, get_speaker, list_all_speakers,
    create_voiceprint, find_similar_voiceprints
)

# 建立語者
speaker_uuid = create_speaker(full_name="王小明", nickname="小明")

# 建立聲紋
voiceprint_uuid = create_voiceprint(speaker_uuid, embedding_vector)

# 搜尋相似聲紋
best_id, best_name, distance, all_results = find_similar_voiceprints(embedding_vector)
```

注意：
- 本模組從 database.py 拆分而來，保留原有邏輯
- 所有函式都是同步的（不使用 async/await）
- UUID 驗證與格式化已內建
- 包含完整的錯誤處理與日誌記錄
"""

import logging
import re
import uuid as uuid_lib
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np

from weaviate.classes.query import Filter, MetadataQuery, QueryReference
from modules.database.weaviate_connection import get_weaviate_client, ensure_connection

logger = logging.getLogger(__name__)

# ===========================
# 常數與輔助函數
# ===========================

# RFC3339 格式時間處理
def format_rfc3339(dt: Optional[datetime] = None) -> str:
    """
    將 datetime 轉換為 RFC3339 格式字串，用於 Weaviate 時間戳記
    
    Args:
        dt: 要格式化的 datetime 物件，若 None 則使用當前台北時間
        
    Returns:
        str: RFC3339 格式的時間字串
    """
    if dt is None:
        # 使用台北時間 (UTC+8)
        taipei_tz = timezone(timedelta(hours=8))
        dt = datetime.now(taipei_tz)
    elif dt.tzinfo is None:
        # 如果沒有時區資訊，假設為台北時間
        taipei_tz = timezone(timedelta(hours=8))
        dt = dt.replace(tzinfo=taipei_tz)
    return dt.isoformat()


# UUID 驗證工具
UUID_PATTERN = re.compile(r"^[0-9a-fA-F-]{36}$")

def valid_uuid(value) -> bool:
    """
    檢查值是否為有效 UUID 格式
    
    Args:
        value: 待檢查的值，可以是字串或 Weaviate UUID 物件
        
    Returns:
        bool: 是否為有效的 UUID 格式
    """
    if not value:
        return False
    
    try:
        uuid_str = str(value)
        return bool(UUID_PATTERN.match(uuid_str))
    except Exception:
        return False


# 預設常數
DEFAULT_SPEAKER_NAME = "未命名語者"
DEFAULT_FULL_NAME_PREFIX = "n"  # 預設 full_name 前綴


# Collection 名稱常數
SPEAKER_CLASS = "Speaker"
VOICEPRINT_CLASS = "VoicePrint"


# ===========================
# Speaker CRUD
# ===========================

def list_all_speakers() -> List[Dict[str, Any]]:
    """
    列出所有語者及其基本資訊
    
    Returns:
        List[Dict[str, Any]]: 語者列表，每個項目包含 uuid、speaker_id、full_name、nickname 等資訊
    
    範例：
    ```python
    speakers = list_all_speakers()
    for speaker in speakers:
        print(f"{speaker['full_name']} ({speaker['nickname']})")
        print(f"  UUID: {speaker['uuid']}")
        print(f"  聲紋數: {speaker['voiceprint_count']}")
    ```
    """
    try:
        client = ensure_connection()
        results = (
            client.collections.get(SPEAKER_CLASS)
            .query.fetch_objects()
        )
        speakers = []
        for obj in results.objects:
            voiceprint_ids = obj.properties.get("voiceprint_ids", [])
            
            # 處理時間欄位，確保轉換為字串格式
            created_at = obj.properties.get("created_at")
            last_active_at = obj.properties.get("last_active_at")
            
            # 如果是 datetime 對象，轉換為 ISO 格式字串
            if hasattr(created_at, 'isoformat'):
                created_at = created_at.isoformat()
            elif created_at is None:
                created_at = "未知"
                
            if hasattr(last_active_at, 'isoformat'):
                last_active_at = last_active_at.isoformat()
            elif last_active_at is None:
                last_active_at = "未知"
            
            speakers.append({
                "uuid": str(obj.uuid),
                "speaker_id": obj.properties.get("speaker_id", -1),
                "full_name": obj.properties.get("full_name", "未命名"),
                "nickname": obj.properties.get("nickname") or "",
                "gender": obj.properties.get("gender") or "",
                "created_at": created_at,
                "last_active_at": last_active_at,
                "meet_count": obj.properties.get("meet_count"),
                "meet_days": obj.properties.get("meet_days"),
                "first_audio": obj.properties.get("first_audio") or "",
                "voiceprint_count": len(voiceprint_ids),
                "voiceprint_ids": voiceprint_ids,
            })
        speakers.sort(key=lambda s: s["last_active_at"], reverse=True)
        return speakers
    except Exception as exc:
        logger.error(f"❌ 列出語者時發生錯誤: {exc}")
        return []


def get_speaker(speaker_uuid: str) -> Optional[Any]:
    """
    獲取特定語者的詳細資訊（使用 UUID）
    
    Args:
        speaker_uuid: 語者的 UUID
        
    Returns:
        Optional[Any]: 語者物件，若找不到則返回 None
    
    範例：
    ```python
    speaker = get_speaker("123e4567-e89b-12d3-a456-426614174000")
    if speaker:
        print(f"語者名稱: {speaker.properties['full_name']}")
        print(f"聲紋數量: {len(speaker.properties.get('voiceprint_ids', []))}")
    ```
    """
    try:
        if not valid_uuid(speaker_uuid):
            logger.error(f"❌ 無效的語者 UUID 格式: {speaker_uuid}")
            return None
        
        client = ensure_connection()
        return (
            client.collections.get(SPEAKER_CLASS)
            .query.fetch_object_by_id(uuid=speaker_uuid)
        )
    except Exception as exc:
        logger.error(f"❌ 獲取語者詳細資訊時發生錯誤: {exc}")
        return None


def get_speaker_by_id(speaker_id: int) -> Optional[Any]:
    """
    獲取特定語者的詳細資訊（使用 speaker_id）
    
    Args:
        speaker_id: 語者的 speaker_id (INT)
        
    Returns:
        Optional[Any]: 語者物件，若找不到則返回 None
    
    範例：
    ```python
    speaker = get_speaker_by_id(1)
    if speaker:
        print(f"UUID: {speaker.uuid}")
    ```
    """
    try:
        client = ensure_connection()
        results = (
            client.collections.get(SPEAKER_CLASS)
            .query.fetch_objects(
                filters=Filter.by_property("speaker_id").equal(speaker_id),
                limit=1
            )
        )
        
        return results.objects[0] if results.objects else None
    except Exception as exc:
        logger.error(f"❌ 根據 speaker_id 獲取語者詳細資訊時發生錯誤: {exc}")
        return None


def get_speaker_by_name(full_name: str) -> Optional[Any]:
    """
    根據 full_name 獲取語者
    
    Args:
        full_name: 語者全名
        
    Returns:
        Optional[Any]: 語者物件，若找不到則返回 None
    
    範例：
    ```python
    speaker = get_speaker_by_name("王小明")
    if speaker:
        print(f"找到語者: {speaker.uuid}")
    ```
    """
    try:
        client = ensure_connection()
        results = (
            client.collections.get(SPEAKER_CLASS)
            .query.fetch_objects(
                filters=Filter.by_property("full_name").equal(full_name),
                limit=1
            )
        )
        
        return results.objects[0] if results.objects else None
    except Exception as exc:
        logger.error(f"❌ 根據名稱獲取語者時發生錯誤: {exc}")
        return None


def _get_next_speaker_id() -> int:
    """
    獲取下一個可用的 speaker_id（從 1 開始）
    
    Returns:
        int: 下一個 speaker_id
    """
    try:
        client = get_weaviate_client()
        results = (
            client.collections.get(SPEAKER_CLASS)
            .query.fetch_objects(
                return_properties=["speaker_id"],
                limit=1000  # 假設不會超過 1000 個語者
            )
        )
        
        # 提取所有現有的 speaker_id
        existing_ids = []
        for obj in results.objects:
            speaker_id = obj.properties.get("speaker_id")
            if speaker_id is not None:
                existing_ids.append(speaker_id)
        
        # 找出下一個可用的 ID
        next_id = max(existing_ids) + 1 if existing_ids else 1
        return next_id
        
    except Exception as e:
        logger.error(f"❌ 獲取下一個 speaker_id 時發生錯誤: {e}")
        return 1


def create_speaker(
    full_name: Optional[str] = None,
    nickname: Optional[str] = None,
    gender: Optional[str] = None,
    first_audio: Optional[str] = None
) -> str:
    """
    創建新的語者
    
    Args:
        full_name: 語者全名，若為 None 則自動生成（如 n1, n2, ...）
        nickname: 語者暱稱，可為 None
        gender: 語者性別，可為 None
        first_audio: 第一次生成該語者時使用的音檔來源
        
    Returns:
        str: 新建立的語者 UUID，若建立失敗則返回空字串
    
    範例：
    ```python
    speaker_uuid = create_speaker(
        full_name="王小明",
        nickname="小明",
        gender="male"
    )
    print(f"建立語者成功: {speaker_uuid}")
    ```
    """
    try:
        # 生成新的 UUID 和 speaker_id
        speaker_uuid = str(uuid_lib.uuid4())
        speaker_id = _get_next_speaker_id()
        
        # 如果未提供 full_name，自動生成
        if not full_name:
            full_name = f"{DEFAULT_FULL_NAME_PREFIX}{speaker_id}"
        
        # 創建語者
        client = ensure_connection()
        speaker_collection = client.collections.get(SPEAKER_CLASS)
        
        properties = {
            "speaker_id": speaker_id,
            "full_name": full_name,
            "nickname": nickname or "",
            "gender": gender or "",
            "created_at": format_rfc3339(),
            "last_active_at": format_rfc3339(),
            "meet_count": None,
            "meet_days": None,
            "voiceprint_ids": [],  # 初始時沒有聲紋向量
            "first_audio": first_audio or ""
        }
        
        speaker_collection.data.insert(
            properties=properties,
            uuid=speaker_uuid
        )
        
        logger.info(f"✅ 已建立新語者 {full_name} (UUID: {speaker_uuid}, ID: {speaker_id})")
        return speaker_uuid
        
    except Exception as e:
        logger.error(f"❌ 創建新語者時發生錯誤: {e}")
        return ""


def update_speaker_name(
    speaker_uuid: str,
    new_full_name: Optional[str] = None,
    new_nickname: Optional[str] = None
) -> bool:
    """
    更改語者名稱，並同步更新所有該語者底下聲紋的 speaker_name
    
    Args:
        speaker_uuid: 語者 UUID
        new_full_name: 新的全名，若為 None 則不更新
        new_nickname: 新的暱稱，若為 None 則不更新
        
    Returns:
        bool: 是否更新成功
    
    範例：
    ```python
    success = update_speaker_name(
        speaker_uuid="123e4567-e89b-12d3-a456-426614174000",
        new_full_name="王大明",
        new_nickname="大明"
    )
    ```
    """
    try:
        if not valid_uuid(speaker_uuid):
            logger.error(f"❌ 無效的語者 UUID 格式: {speaker_uuid}")
            return False
        
        # 準備要更新的屬性
        update_properties = {}
        if new_full_name is not None:
            update_properties["full_name"] = new_full_name
        if new_nickname is not None:
            update_properties["nickname"] = new_nickname
            
        if not update_properties:
            logger.warning("⚠️ 沒有提供任何要更新的名稱")
            return False
            
        client = ensure_connection()
        
        # 1. 先更新 Speaker 本身
        sp_col = client.collections.get(SPEAKER_CLASS)
        sp_col.data.update(uuid=speaker_uuid, properties=update_properties)

        # 2. 拿回這個 Speaker 物件，讀出 voiceprint_ids 和更新後的 full_name
        sp_obj = sp_col.query.fetch_object_by_id(uuid=speaker_uuid)
        if not sp_obj:
            logger.error(f"❌ 找不到語者 (UUID: {speaker_uuid})")
            return False
            
        vp_ids = sp_obj.properties.get("voiceprint_ids", [])
        updated_full_name = sp_obj.properties.get("full_name", "未命名")

        # 3. 逐一更新每支 VoicePrint 的 speaker_name
        vp_col = client.collections.get(VOICEPRINT_CLASS)
        for vp_id in vp_ids:
            vp_col.data.update(
                uuid=vp_id,
                properties={"speaker_name": updated_full_name}
            )

        logger.info(f"✅ 已更新語者 {speaker_uuid} 的名稱")
        return True
    except Exception as exc:
        logger.error(f"❌ 更改語者名稱時發生錯誤: {exc}")
        return False


def update_speaker_last_active(speaker_uuid: str, timestamp: Optional[datetime] = None) -> bool:
    """
    更新語者的最後活動時間
    
    Args:
        speaker_uuid: 語者 UUID
        timestamp: 時間戳記，若為 None 則使用當前時間
        
    Returns:
        bool: 是否更新成功
    
    範例：
    ```python
    success = update_speaker_last_active("123e4567-e89b-12d3-a456-426614174000")
    ```
    """
    try:
        if not valid_uuid(speaker_uuid):
            logger.error(f"❌ 無效的語者 UUID 格式: {speaker_uuid}")
            return False
            
        time_str = format_rfc3339(timestamp) if timestamp else format_rfc3339()
        
        client = ensure_connection()
        sp_col = client.collections.get(SPEAKER_CLASS)
        sp_col.data.update(
            uuid=speaker_uuid,
            properties={"last_active_at": time_str}
        )
        
        return True
    except Exception as exc:
        logger.error(f"❌ 更新語者最後活動時間時發生錯誤: {exc}")
        return False


def update_speaker(speaker_uuid: str, update_fields: Dict[str, Any]) -> bool:
    """
    通用更新語者資料的方法
    如果更新了 full_name，會同步更新所有關聯聲紋的 speaker_name
    
    Args:
        speaker_uuid: 語者 UUID
        update_fields: 包含要更新的欄位與值的字典
    
    Returns:
        bool: 是否更新成功
    
    範例：
    ```python
    success = update_speaker(
        speaker_uuid="123e4567-e89b-12d3-a456-426614174000",
        update_fields={
            "full_name": "王大明",
            "gender": "male",
            "meet_count": 5
        }
    )
    ```
    """
    try:
        if not valid_uuid(speaker_uuid):
            logger.error(f"❌ 無效的語者 UUID 格式: {speaker_uuid}")
            return False

        if not update_fields:
            logger.warning("⚠️ 沒有提供任何要更新的欄位")
            return True  # 沒有東西要更新，不算錯誤

        client = ensure_connection()
        sp_col = client.collections.get(SPEAKER_CLASS)

        # 1. 更新 Speaker 物件本身
        sp_col.data.update(uuid=speaker_uuid, properties=update_fields)
        logger.info(f"✅ 已更新語者 {speaker_uuid} 的屬性: {list(update_fields.keys())}")

        # 2. 如果 full_name 被更新，需要同步更新所有 VoicePrint 的 speaker_name
        if "full_name" in update_fields:
            # 拿回這個 Speaker 物件，讀出 voiceprint_ids 和更新後的 full_name
            sp_obj = sp_col.query.fetch_object_by_id(uuid=speaker_uuid)
            if not sp_obj:
                logger.error(f"❌ 找不到語者 (UUID: {speaker_uuid})，無法同步更新聲紋")
                return False

            vp_ids = sp_obj.properties.get("voiceprint_ids", [])
            updated_full_name = sp_obj.properties.get("full_name", "未命名")

            # 逐一更新每支 VoicePrint 的 speaker_name
            if vp_ids:
                vp_col = client.collections.get(VOICEPRINT_CLASS)
                for vp_id in vp_ids:
                    try:
                        vp_col.data.update(
                            uuid=vp_id,
                            properties={"speaker_name": updated_full_name}
                        )
                    except Exception as vp_exc:
                        logger.error(f"❌ 同步更新聲紋 {vp_id} 的 speaker_name 時失敗: {vp_exc}")
                logger.info(f"✅ 已同步更新 {len(vp_ids)} 個關聯聲紋的 speaker_name 為 '{updated_full_name}'")

        return True
    except Exception as exc:
        logger.error(f"❌ 更新語者 {speaker_uuid} 時發生錯誤: {exc}")
        return False


def delete_speaker(speaker_uuid: str) -> bool:
    """
    刪除語者，同時刪除該語者底下的所有聲紋
    
    Args:
        speaker_uuid: 語者 UUID
        
    Returns:
        bool: 是否刪除成功
    
    範例：
    ```python
    success = delete_speaker("123e4567-e89b-12d3-a456-426614174000")
    if success:
        print("語者已刪除")
    ```
    """
    try:
        if not valid_uuid(speaker_uuid):
            logger.error(f"❌ 無效的語者 UUID 格式: {speaker_uuid}")
            return False
        
        client = ensure_connection()
        
        # 1. 獲取語者的所有聲紋
        speaker_collection = client.collections.get(SPEAKER_CLASS)
        speaker_obj = speaker_collection.query.fetch_object_by_id(
            uuid=speaker_uuid,
            return_properties=["full_name", "speaker_id", "voiceprint_ids"]
        )
        
        if not speaker_obj:
            logger.error(f"❌ 找不到語者 (UUID: {speaker_uuid})")
            return False
            
        speaker_name = speaker_obj.properties.get("full_name", "未命名")
        speaker_id = speaker_obj.properties.get("speaker_id", "未知")
        voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
        
        # 2. 刪除語者的所有聲紋
        deleted_count = 0
        voiceprint_collection = client.collections.get(VOICEPRINT_CLASS)
        for vp_id in voiceprint_ids:
            try:
                voiceprint_collection.data.delete_by_id(uuid=vp_id)
                deleted_count += 1
            except Exception as vp_exc:
                logger.error(f"❌ 刪除聲紋 {vp_id} 時發生錯誤: {vp_exc}")
        
        # 3. 刪除語者本身
        speaker_collection.data.delete_by_id(uuid=speaker_uuid)
        
        logger.info(f"✅ 已刪除語者 {speaker_name} (UUID: {speaker_uuid}, ID: {speaker_id}) 及其 {deleted_count} 個聲紋")
        return True
    except Exception as exc:
        logger.error(f"❌ 刪除語者時發生錯誤: {exc}")
        return False


def batch_get_speakers(speaker_uuids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    批次獲取多個語者的資訊
    
    Args:
        speaker_uuids: 語者 UUID 列表
        
    Returns:
        Dict[str, Dict[str, Any]]: UUID → 語者資訊的對應表
    
    範例：
    ```python
    uuids = ["uuid1", "uuid2", "uuid3"]
    speakers = batch_get_speakers(uuids)
    for uuid, info in speakers.items():
        print(f"{info['full_name']} - {info['nickname']}")
    ```
    """
    result = {}
    for speaker_uuid in speaker_uuids:
        speaker = get_speaker(speaker_uuid)
        if speaker:
            result[speaker_uuid] = {
                "full_name": speaker.properties.get("full_name", "未命名"),
                "nickname": speaker.properties.get("nickname", ""),
                "is_authenticated": speaker.properties.get("gender") != ""  # 簡單判斷
            }
    return result


# ===========================
# VoicePrint CRUD
# ===========================

def create_voiceprint(
    speaker_uuid: str,
    embedding: np.ndarray,
    audio_source: str = "",
    timestamp: Optional[datetime] = None,
    quality_score: Optional[float] = None
) -> str:
    """
    為語者創建新的聲紋向量
    
    Args:
        speaker_uuid: 語者 UUID
        embedding: 聲紋嵌入向量
        audio_source: 音訊來源描述
        timestamp: 時間戳記，用於設定聲紋的創建時間和更新時間
        quality_score: 聲紋品質評分，可為 None
        
    Returns:
        str: 新建立的聲紋向量 UUID，若創建失敗則返回空字串
    
    範例：
    ```python
    embedding = np.array([0.1, 0.2, 0.3, ...])  # 向量
    voiceprint_uuid = create_voiceprint(
        speaker_uuid="123e4567-e89b-12d3-a456-426614174000",
        embedding=embedding,
        audio_source="meeting_2024.wav",
        quality_score=0.95
    )
    ```
    """
    try:
        if not valid_uuid(speaker_uuid):
            logger.error(f"❌ 無效的語者 UUID 格式: {speaker_uuid}")
            return ""
        
        client = ensure_connection()
        
        # 獲取語者資訊
        speaker_collection = client.collections.get(SPEAKER_CLASS)
        speaker_obj = speaker_collection.query.fetch_object_by_id(
            uuid=speaker_uuid,
            return_properties=["full_name", "voiceprint_ids"]
        )
        
        if not speaker_obj:
            logger.error(f"❌ 找不到語者 (UUID: {speaker_uuid})")
            return ""
            
        speaker_name = speaker_obj.properties.get("full_name", DEFAULT_SPEAKER_NAME)
        voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
        
        # 格式化時間或使用當前時間
        time_str = format_rfc3339(timestamp) if timestamp else format_rfc3339()
        
        # 創建新的聲紋向量
        voiceprint_collection = client.collections.get(VOICEPRINT_CLASS)
        voiceprint_uuid = str(uuid_lib.uuid4())
        
        voiceprint_collection.data.insert(
            properties={
                "created_at": time_str,
                "updated_at": time_str,
                "update_count": 1,
                "sample_count": None,
                "quality_score": quality_score,
                "speaker_name": speaker_name,
            },
            uuid=voiceprint_uuid,
            vector=embedding.tolist(),
            references={
                "speaker": [speaker_uuid]
            }
        )
        
        # 更新語者的聲紋列表
        voiceprint_ids.append(voiceprint_uuid)
        speaker_collection.data.update(
            uuid=speaker_uuid,
            properties={
                "voiceprint_ids": voiceprint_ids,
                "last_active_at": time_str
            }
        )
        
        logger.info(f"✅ 已為語者 {speaker_name} 創建新的聲紋向量 (UUID: {voiceprint_uuid})")
        return voiceprint_uuid
        
    except Exception as e:
        logger.error(f"❌ 創建聲紋向量時發生錯誤: {e}")
        return ""


def get_voiceprint(
    voiceprint_uuid: str,
    include_vector: bool = False,
    include_refs: bool = False
) -> Optional[Any]:
    """
    獲取特定聲紋向量的詳細資訊
    
    Args:
        voiceprint_uuid: 聲紋向量 UUID
        include_vector: 是否包含向量數據
        include_refs: 是否包含引用的語者 UUID
        
    Returns:
        Optional[Any]: 聲紋向量物件，若找不到則返回 None
    
    範例：
    ```python
    voiceprint = get_voiceprint(
        "123e4567-e89b-12d3-a456-426614174001",
        include_vector=True
    )
    if voiceprint:
        print(f"更新次數: {voiceprint.properties['update_count']}")
    ```
    """
    if not valid_uuid(voiceprint_uuid):
        logger.error(f"❌ 無效的 VoicePrint UUID: {voiceprint_uuid}")
        return None

    try:
        client = ensure_connection()
        coll = client.collections.get(VOICEPRINT_CLASS)

        return coll.query.fetch_object_by_id(
            uuid=voiceprint_uuid,
            include_vector=include_vector,
            return_references=(
                QueryReference(link_on="speaker") if include_refs else None
            ),
        )
    except Exception as e:
        logger.error(f"❌ 抓取 VoicePrint {voiceprint_uuid[:8]} 失敗: {e}")
        return None


def get_voiceprint_properties(voiceprint_uuid: str, properties: List[str]) -> Optional[Dict[str, Any]]:
    """
    獲取聲紋向量的特定屬性（不包含向量）
    
    Args:
        voiceprint_uuid: 聲紋向量 UUID
        properties: 需要獲取的屬性列表
        
    Returns:
        Optional[Dict[str, Any]]: 屬性字典，若找不到則返回 None
    
    範例：
    ```python
    props = get_voiceprint_properties(
        "123e4567-e89b-12d3-a456-426614174001",
        ["speaker_name", "update_count", "quality_score"]
    )
    if props:
        print(f"語者名稱: {props['speaker_name']}")
    ```
    """
    try:
        if not valid_uuid(voiceprint_uuid):
            logger.error(f"❌ 無效的聲紋向量 UUID 格式: {voiceprint_uuid}")
            return None
            
        client = ensure_connection()
        voiceprint_collection = client.collections.get(VOICEPRINT_CLASS)
        result = voiceprint_collection.query.fetch_object_by_id(
            uuid=voiceprint_uuid,
            return_properties=properties
        )
        
        if not result:
            return None
            
        return result.properties
        
    except Exception as e:
        logger.error(f"❌ 獲取聲紋向量屬性時發生錯誤: {e}")
        return None


def list_voiceprints_by_speaker(speaker_uuid: str, include_vectors: bool = False) -> List[Dict[str, Any]]:
    """
    獲取語者的所有聲紋向量
    
    Args:
        speaker_uuid: 語者 UUID
        include_vectors: 是否包含向量數據
        
    Returns:
        List[Dict[str, Any]]: 聲紋向量列表
    
    範例：
    ```python
    voiceprints = list_voiceprints_by_speaker("123e4567-e89b-12d3-a456-426614174000")
    print(f"該語者有 {len(voiceprints)} 個聲紋")
    ```
    """
    try:
        if not valid_uuid(speaker_uuid):
            logger.error(f"❌ 無效的語者 UUID 格式: {speaker_uuid}")
            return []
        
        # 獲取語者的聲紋列表
        speaker_obj = get_speaker(speaker_uuid)
        if not speaker_obj:
            logger.error(f"❌ 找不到語者 (UUID: {speaker_uuid})")
            return []
            
        voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
        if not voiceprint_ids:
            return []
        
        client = ensure_connection()
        
        # 獲取每個聲紋向量的詳細資訊
        voiceprint_collection = client.collections.get(VOICEPRINT_CLASS)
        voiceprints = []
        
        for vp_id in voiceprint_ids:
            try:
                vp_obj = voiceprint_collection.query.fetch_object_by_id(
                    uuid=vp_id,
                    include_vector=include_vectors
                )
                
                if vp_obj:
                    # 處理時間欄位
                    created_at = vp_obj.properties.get("created_at")
                    updated_at = vp_obj.properties.get("updated_at")
                    
                    if hasattr(created_at, 'isoformat'):
                        created_at = created_at.isoformat()
                    elif created_at is None:
                        created_at = None
                        
                    if hasattr(updated_at, 'isoformat'):
                        updated_at = updated_at.isoformat()
                    elif updated_at is None:
                        updated_at = None
                    
                    vp_data = {
                        "uuid": vp_obj.uuid,
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "update_count": vp_obj.properties.get("update_count", -1),
                        "sample_count": vp_obj.properties.get("sample_count"),
                        "quality_score": vp_obj.properties.get("quality_score"),
                        "speaker_name": vp_obj.properties.get("speaker_name")
                    }
                    
                    if include_vectors:
                        vec_dict = vp_obj.vector
                        vp_data["vector"] = vec_dict["default"] if isinstance(vec_dict, dict) else vec_dict
                        
                    voiceprints.append(vp_data)
            except Exception as e:
                logger.error(f"❌ 獲取聲紋向量 {vp_id} 時發生錯誤: {e}")
        
        return voiceprints
        
    except Exception as e:
        logger.error(f"❌ 獲取語者聲紋向量列表時發生錯誤: {e}")
        return []


def update_voiceprint(
    voiceprint_uuid: str,
    new_embedding: np.ndarray,
    update_count: Optional[int] = None,
    quality_score: Optional[float] = None
) -> int:
    """
    使用加權移動平均更新現有的聲紋向量
    
    Args:
        voiceprint_uuid: 聲紋向量 UUID
        new_embedding: 新的嵌入向量
        update_count: 更新次數，若為 None 則從資料庫讀取並 +1
        quality_score: 新的品質分數，若為 None 則不更新
        
    Returns:
        int: 更新後的更新次數，若更新失敗則返回 0
    
    範例：
    ```python
    new_embedding = np.array([0.15, 0.25, 0.35, ...])
    updated_count = update_voiceprint(
        "123e4567-e89b-12d3-a456-426614174001",
        new_embedding,
        quality_score=0.98
    )
    print(f"已更新 {updated_count} 次")
    ```
    """
    try:
        voiceprint_uuid = str(voiceprint_uuid)
        
        if not valid_uuid(voiceprint_uuid):
            logger.error(f"❌ 無效的聲紋向量 UUID 格式: {voiceprint_uuid}")
            return 0
        
        client = ensure_connection()
        
        # 獲取現有的嵌入向量
        voiceprint_collection = client.collections.get(VOICEPRINT_CLASS)
        existing_object = voiceprint_collection.query.fetch_object_by_id(
            uuid=voiceprint_uuid,
            include_vector=True
        )
        
        if not existing_object:
            logger.error(f"❌ 找不到 UUID 為 {voiceprint_uuid} 的聲紋向量")
            return 0
        
        # 如果未提供 update_count，則從資料庫讀取並 +1
        if update_count is None:
            current_update_count = existing_object.properties.get("update_count")
            if current_update_count is None:
                logger.error(f"❌ 無法獲取聲紋向量 {voiceprint_uuid} 的更新次數")
                return 0
            update_count = current_update_count + 1
        
        # 獲取現有的嵌入向量
        vec_dict = existing_object.vector
        raw_old = vec_dict["default"] if isinstance(vec_dict, dict) else vec_dict
        old_embedding = np.array(raw_old, dtype=float)
        
        # 使用加權移動平均更新嵌入向量
        weight_old = update_count - 1
        updated_embedding = (old_embedding * weight_old + new_embedding) / update_count
        
        # 準備更新屬性
        update_properties = {
            "updated_at": format_rfc3339(),
            "update_count": update_count
        }
        
        # sample_count 獨立維護，保持原值
        if existing_object.properties.get("sample_count") is not None:
            update_properties["sample_count"] = existing_object.properties.get("sample_count")
        
        # 如果提供了 quality_score 則更新
        if quality_score is not None:
            update_properties["quality_score"] = quality_score
        
        # 更新資料庫中的向量
        voiceprint_collection.data.update(
            uuid=voiceprint_uuid,
            properties=update_properties,
            vector=updated_embedding.tolist()
        )
        
        logger.info(f"✅ 已更新聲紋向量 {voiceprint_uuid}，新的更新次數: {update_count}")
        return update_count
        
    except Exception as e:
        logger.error(f"❌ 更新聲紋向量時發生錯誤: {e}")
        return 0


def delete_voiceprint(voiceprint_uuid: str) -> bool:
    """
    刪除聲紋向量，並從相關語者的列表中移除
    
    Args:
        voiceprint_uuid: 聲紋向量 UUID
        
    Returns:
        bool: 是否刪除成功
    
    範例：
    ```python
    success = delete_voiceprint("123e4567-e89b-12d3-a456-426614174001")
    ```
    """
    try:
        if not valid_uuid(voiceprint_uuid):
            logger.error(f"❌ 無效的聲紋向量 UUID 格式: {voiceprint_uuid}")
            return False
        
        # 獲取聲紋關聯的語者 UUID
        speaker_uuid = get_speaker_uuid_from_voiceprint(voiceprint_uuid)
        
        client = ensure_connection()
        
        # 如果找到關聯的語者，從語者的聲紋列表中移除
        if speaker_uuid:
            speaker_collection = client.collections.get(SPEAKER_CLASS)
            speaker_obj = speaker_collection.query.fetch_object_by_id(
                uuid=speaker_uuid,
                return_properties=["voiceprint_ids"]
            )
            
            if speaker_obj:
                voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
                if voiceprint_uuid in voiceprint_ids:
                    voiceprint_ids.remove(voiceprint_uuid)
                    
                    speaker_collection.data.update(
                        uuid=speaker_uuid,
                        properties={"voiceprint_ids": voiceprint_ids}
                    )
        
        # 刪除聲紋向量
        voiceprint_collection = client.collections.get(VOICEPRINT_CLASS)
        voiceprint_collection.data.delete_by_id(uuid=voiceprint_uuid)
        
        logger.info(f"✅ 已刪除聲紋向量 {voiceprint_uuid}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 刪除聲紋向量時發生錯誤: {e}")
        return False


def get_speaker_uuid_from_voiceprint(voiceprint_uuid: str) -> str:
    """
    根據聲紋向量 UUID 獲取關聯的語者 UUID
    
    Args:
        voiceprint_uuid: 聲紋向量 UUID
        
    Returns:
        str: 語者 UUID，若找不到則返回空字串
    
    範例：
    ```python
    speaker_uuid = get_speaker_uuid_from_voiceprint("123e4567-e89b-12d3-a456-426614174001")
    ```
    """
    try:
        if not valid_uuid(voiceprint_uuid):
            logger.error(f"❌ 無效的聲紋向量 UUID 格式: {voiceprint_uuid}")
            return ""
        
        client = ensure_connection()
        voiceprint_collection = client.collections.get(VOICEPRINT_CLASS)
        qr = QueryReference(
            link_on="speaker",
            return_properties=["uuid"]
        )
        
        voiceprint_obj = voiceprint_collection.query.fetch_object_by_id(
            uuid=voiceprint_uuid,
            return_references=qr
        )
        
        if not voiceprint_obj:
            return ""
            
        refs = voiceprint_obj.references.get("speaker", []).objects
        if not refs:
            return ""
            
        return refs[0].uuid
    except Exception as e:
        logger.error(f"❌ 獲取聲紋關聯的語者 UUID 時發生錯誤: {e}")
        return ""


def get_speaker_voiceprints(speaker_uuid: str, include_vectors: bool = False) -> List[Dict[str, Any]]:
    """
    獲取語者的所有聲紋向量（list_voiceprints_by_speaker 的別名，用於向後兼容）
    
    Args:
        speaker_uuid: 語者 UUID
        include_vectors: 是否包含向量數據
        
    Returns:
        List[Dict[str, Any]]: 聲紋向量列表
    
    範例：
    ```python
    voiceprints = get_speaker_voiceprints("123e4567-e89b-12d3-a456-426614174000")
    ```
    """
    return list_voiceprints_by_speaker(speaker_uuid, include_vectors)


def search_similar_voiceprints(
    embedding: np.ndarray,
    limit: int = 3
) -> Tuple[Optional[str], Optional[str], float, List[Tuple[str, str, float, int]]]:
    """
    搜尋相似的聲紋向量（向量搜尋）
    
    Args:
        embedding: 待比較的嵌入向量
        limit: 返回結果的數量限制
        
    Returns:
        Tuple: (最佳匹配 ID, 最佳匹配語者名稱, 最小距離, 所有距離列表)
    
    範例：
    ```python
    embedding = np.array([0.1, 0.2, 0.3, ...])
    best_id, best_name, distance, all_results = search_similar_voiceprints(embedding, limit=5)
    
    if best_id:
        print(f"最佳匹配: {best_name}, 距離: {distance:.4f}")
        for vp_id, name, dist, count in all_results:
            print(f"  - {name}: {dist:.4f} (更新 {count} 次)")
    ```
    """
    try:
        client = ensure_connection()
        voiceprint_collection = client.collections.get(VOICEPRINT_CLASS)
        
        # 計算新向量與資料庫中所有向量的距離
        results = voiceprint_collection.query.near_vector(
            near_vector=embedding.tolist(),
            limit=limit,
            return_properties=["speaker_name", "update_count", "created_at", "updated_at"],
            return_metadata=MetadataQuery(distance=True)
        )
        
        # 如果沒有找到任何結果
        if not results.objects:
            logger.info("⚠️ 資料庫中尚無任何嵌入向量")
            return None, None, float('inf'), []
        
        # 處理結果，計算距離
        distances = []
        for obj in results.objects:
            # 距離信息
            distance = None
            if hasattr(obj, 'metadata') and hasattr(obj.metadata, 'distance'):
                distance = obj.metadata.distance
            
            if distance is None:
                distance = -1
                logger.warning(f"⚠️ 無法從結果中獲取距離信息，使用預設值 {distance}")
            
            object_id = obj.uuid
            speaker_name = obj.properties.get("speaker_name")
            update_count = obj.properties.get("update_count")
            
            logger.debug(f"比對 - 語者: {speaker_name}, 更新次數: {update_count}, 餘弦距離: {distance:.4f}")
            
            # 保存距離資訊
            distances.append((object_id, speaker_name, distance, update_count))
        
        # 找出最小距離
        if distances:
            best_match = min(distances, key=lambda x: x[2])
            best_id, best_name, best_distance, _ = best_match
            return best_id, best_name, best_distance, distances
        else:
            logger.warning("⚠️ 未能獲取有效的距離信息")
            return None, None, float('inf'), []
        
    except Exception as e:
        logger.error(f"❌ 比對嵌入向量時發生錯誤: {e}")
        return None, None, float('inf'), []


# 別名：保持向後相容
find_similar_voiceprints = search_similar_voiceprints


def batch_create_voiceprints(
    speaker_uuid: str,
    embeddings: List[np.ndarray],
    audio_sources: Optional[List[str]] = None
) -> List[str]:
    """
    批次建立多個聲紋向量
    
    Args:
        speaker_uuid: 語者 UUID
        embeddings: 嵌入向量列表
        audio_sources: 音訊來源列表（可選）
        
    Returns:
        List[str]: 建立的聲紋 UUID 列表
    
    範例：
    ```python
    embeddings = [np.array([...]), np.array([...]), np.array([...])]
    voiceprint_uuids = batch_create_voiceprints(speaker_uuid, embeddings)
    print(f"建立了 {len(voiceprint_uuids)} 個聲紋")
    ```
    """
    voiceprint_uuids = []
    
    for i, embedding in enumerate(embeddings):
        audio_source = audio_sources[i] if audio_sources and i < len(audio_sources) else ""
        vp_uuid = create_voiceprint(speaker_uuid, embedding, audio_source=audio_source)
        if vp_uuid:
            voiceprint_uuids.append(vp_uuid)
    
    return voiceprint_uuids


# ===========================
# Speaker ↔ VoicePrint 關聯管理
# ===========================

def add_voiceprint_to_speaker(speaker_uuid: str, voiceprint_uuid: str) -> bool:
    """
    將聲紋向量添加到語者的聲紋列表中
    
    Args:
        speaker_uuid: 語者 UUID
        voiceprint_uuid: 聲紋向量 UUID
        
    Returns:
        bool: 是否添加成功
    """
    try:
        if not valid_uuid(speaker_uuid) or not valid_uuid(voiceprint_uuid):
            logger.error(f"❌ 無效的 UUID 格式: speaker_uuid={speaker_uuid}, voiceprint_uuid={voiceprint_uuid}")
            return False
        
        client = ensure_connection()
        
        # 獲取語者的聲紋列表
        speaker_collection = client.collections.get(SPEAKER_CLASS)
        speaker_obj = speaker_collection.query.fetch_object_by_id(
            uuid=speaker_uuid,
            return_properties=["voiceprint_ids", "full_name"]
        )
        
        if not speaker_obj:
            logger.error(f"❌ 找不到語者 (UUID: {speaker_uuid})")
            return False
            
        # 更新語者的聲紋列表
        voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
        if voiceprint_uuid not in voiceprint_ids:
            voiceprint_ids.append(voiceprint_uuid)
            
            speaker_collection.data.update(
                uuid=speaker_uuid,
                properties={
                    "voiceprint_ids": voiceprint_ids,
                    "last_active_at": format_rfc3339()
                }
            )
            
            # 更新聲紋的語者名稱
            speaker_name = speaker_obj.properties.get("full_name", DEFAULT_SPEAKER_NAME)
            voiceprint_collection = client.collections.get(VOICEPRINT_CLASS)
            voiceprint_collection.data.update(
                uuid=voiceprint_uuid,
                properties={"speaker_name": speaker_name},
                references={"speaker": [speaker_uuid]}
            )
            
            logger.info(f"✅ 已將聲紋 {voiceprint_uuid} 添加到語者 {speaker_uuid} 的聲紋列表")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 添加聲紋到語者時發生錯誤: {e}")
        return False


def transfer_voiceprints(
    source_uuid: str,
    dest_uuid: str,
    voiceprint_uuids: Optional[List[str]] = None
) -> bool:
    """
    將聲紋從一個語者轉移到另一個語者
    
    Args:
        source_uuid: 來源語者 UUID
        dest_uuid: 目標語者 UUID
        voiceprint_uuids: 要轉移的聲紋 UUID 列表，若為 None 則轉移全部
        
    Returns:
        bool: 是否轉移成功
    """
    try:
        if not valid_uuid(source_uuid) or not valid_uuid(dest_uuid):
            logger.error(f"❌ 無效的語者 UUID 格式: source_uuid={source_uuid}, dest_uuid={dest_uuid}")
            return False
        
        client = ensure_connection()
        collection = client.collections.get(SPEAKER_CLASS)
        src_obj = collection.query.fetch_object_by_id(uuid=source_uuid)
        dest_obj = collection.query.fetch_object_by_id(uuid=dest_uuid)
        
        if not src_obj or not dest_obj:
            logger.warning("⚠️ 來源或目標語者不存在")
            return False
        
        src_vps = set(src_obj.properties.get("voiceprint_ids", []))
        dest_vps = set(dest_obj.properties.get("voiceprint_ids", []))
        move_set = set(src_vps) if voiceprint_uuids is None else set(voiceprint_uuids).intersection(src_vps)
        dest_vps.update(move_set)
        src_vps.difference_update(move_set)
        
        # 更新來源與目標語者的聲紋
        collection.data.update(uuid=source_uuid, properties={"voiceprint_ids": list(src_vps)})
        collection.data.update(uuid=dest_uuid, properties={"voiceprint_ids": list(dest_vps)})
        
        # 取得目標語者名稱
        dest_name = dest_obj.properties.get("full_name", "未命名")
        
        # 批次更新被轉移聲紋的 speaker_uuid 與 speaker_name
        vp_collection = client.collections.get(VOICEPRINT_CLASS)
        for vp_id in move_set:
            try:
                vp_collection.data.update(uuid=vp_id, properties={
                    "speaker_name": dest_name
                }, references={"speaker": [dest_uuid]})
            except Exception as e:
                logger.error(f"❌ 轉移聲紋 {vp_id} 時發生錯誤: {e}")
        
        # 若來源語者已無聲紋，自動刪除
        if not src_vps:
            try:
                collection.data.delete_by_id(uuid=source_uuid)
                logger.info(f"✅ 來源語者 {source_uuid} 已無聲紋，自動刪除")
            except Exception as del_exc:
                logger.error(f"❌ 自動刪除來源語者時發生錯誤: {del_exc}")
                
        logger.info(f"✅ 已成功將 {len(move_set)} 個聲紋從語者 {source_uuid} 轉移到語者 {dest_uuid}")
        return True
    except Exception as exc:
        logger.error(f"❌ 轉移聲紋時發生錯誤: {exc}")
        return False


# ===========================
# 測試程式
# ===========================

if __name__ == "__main__":
    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def test():
        print("=" * 60)
        print("Weaviate CRUD 測試")
        print("=" * 60)
        print()
        
        # 測試 1: 列出所有語者
        print("【測試 1】列出所有語者...")
        speakers = list_all_speakers()
        print(f"✅ 共有 {len(speakers)} 位語者")
        if speakers:
            print("前 3 位語者：")
            for speaker in speakers[:3]:
                print(f"  - {speaker['full_name']} (UUID: {speaker['uuid'][:8]}...)")
        
        # 測試 2: 建立語者
        print("\n【測試 2】建立新語者...")
        speaker_uuid = create_speaker(full_name="測試語者", nickname="測試")
        if speaker_uuid:
            print(f"✅ 建立成功: {speaker_uuid}")
        else:
            print("❌ 建立失敗")
            return
        
        # 測試 3: 查詢語者
        print("\n【測試 3】查詢語者...")
        speaker = get_speaker(speaker_uuid)
        if speaker:
            print(f"✅ 查詢成功: {speaker.properties['full_name']}")
        
        # 測試 4: 建立聲紋
        print("\n【測試 4】建立聲紋...")
        test_embedding = np.random.rand(192)  # 假設是 192 維向量
        vp_uuid = create_voiceprint(speaker_uuid, test_embedding, quality_score=0.95)
        if vp_uuid:
            print(f"✅ 建立成功: {vp_uuid}")
        
        # 測試 5: 搜尋相似聲紋
        print("\n【測試 5】搜尋相似聲紋...")
        best_id, best_name, distance, all_results = search_similar_voiceprints(test_embedding, limit=3)
        if best_id:
            print(f"✅ 最佳匹配: {best_name}, 距離: {distance:.4f}")
        
        # 測試 6: 清理測試資料
        print("\n【測試 6】清理測試資料...")
        success = delete_speaker(speaker_uuid)
        if success:
            print("✅ 清理成功")
        
        print("\n" + "=" * 60)
        print("測試完成！")
        print("=" * 60)
    
    # 執行測試
    test()
