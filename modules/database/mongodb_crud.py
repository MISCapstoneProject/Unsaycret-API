"""
MongoDB CRUD 操作模組

功能：
1. Session CRUD（會議工作階段）
2. Transcript CRUD（逐字稿）
3. AISummary CRUD（AI 摘要）

使用方式：
```python
from modules.database.mongodb_crud import (
    create_session, get_session, list_sessions,
    create_transcript, get_transcript,
    create_ai_summary, list_ai_summaries_by_session
)

# 建立會議
session = await create_session(
    title="專案討論",
    description="討論 Q1 目標",
    participants_snapshot=[
        {"speaker_uuid": "uuid-123", "name": "張三", "is_authenticated": True}
    ]
)

# 建立逐字稿
transcript = await create_transcript(
    session_id=session.id,
    segments=[
        {"speaker_uuid": "uuid-123", "text": "大家好", "start_time": 0.0, "end_time": 1.5}
    ]
)

# 建立 AI 摘要
summary = await create_ai_summary(
    session_id=session.id,
    summary="討論了 Q1 目標與時程",
    key_points=["時程規劃", "資源分配"]
)
```

注意：
- 所有函式都是 async，需要在 async context 中呼叫
- Session.id 是 MongoDB ObjectId，可用於關聯查詢
- Transcript 會自動從 Session.participants_snapshot 取得說話者姓名
- 建議使用 refresh_participants_snapshot() 定期更新 Session 快照
"""

import logging
from typing import Optional, Any
from datetime import datetime
from bson import ObjectId

from modules.database.mongodb_connection import get_database
from modules.database.models import Session, Transcript, TranscriptSegment, AISummary

logger = logging.getLogger(__name__)


# ===========================
# Session CRUD
# ===========================

async def create_session(
    title: str,
    description: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    participants_snapshot: Optional[list[dict[str, Any]]] = None,
    metadata: Optional[dict[str, Any]] = None
) -> Session:
    """
    建立新的會議工作階段
    
    Args:
        title: 會議標題
        description: 會議描述
        start_time: 開始時間（預設為當前時間）
        end_time: 結束時間
        participants_snapshot: 參與者快照列表
            格式：[{"speaker_uuid": "uuid-123", "name": "張三", "is_authenticated": True}]
        metadata: 額外的元資料
    
    Returns:
        Session: 建立的會議物件
    
    Raises:
        ValueError: 如果參數無效
        Exception: 如果資料庫操作失敗
    
    範例：
    ```python
    session = await create_session(
        title="產品規劃會議",
        description="討論 Q2 產品路線圖",
        participants_snapshot=[
            {"speaker_uuid": "uuid-456", "name": "李四", "is_authenticated": True},
            {"speaker_uuid": "uuid-789", "name": "王五", "is_authenticated": False}
        ]
    )
    print(f"建立會議 ID: {session.id}")
    ```
    """
    try:
        # 驗證必填欄位
        if not title:
            raise ValueError("title 為必填欄位")
        
        # 建立 Session 物件
        session = Session(
            title=title,
            description=description,
            start_time=start_time or datetime.utcnow(),
            end_time=end_time,
            participants_snapshot=participants_snapshot or [],
            metadata=metadata or {}
        )
        
        # 儲存到 MongoDB
        await session.insert()
        logger.info(f"✅ 建立 Session 成功: id={session.id}, title='{title}'")
        return session
        
    except ValueError as ve:
        logger.error(f"❌ 參數驗證失敗: {ve}")
        raise
    except Exception as e:
        logger.error(f"❌ 建立 Session 失敗: {e}")
        raise


async def get_session(session_id: str | ObjectId) -> Optional[Session]:
    """
    根據 ID 查詢單一會議
    
    Args:
        session_id: MongoDB ObjectId（字串或 ObjectId 物件）
    
    Returns:
        Optional[Session]: 找到的會議物件，不存在則返回 None
    
    範例：
    ```python
    session = await get_session("507f1f77bcf86cd799439011")
    if session:
        print(f"會議標題: {session.title}")
        print(f"參與者: {session.get_all_participant_names()}")
    else:
        print("找不到該會議")
    ```
    """
    try:
        # 確保 session_id 是 ObjectId
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
        
        session = await Session.get(session_id)
        
        if session:
            logger.debug(f"✅ 查詢 Session 成功: id={session_id}")
        else:
            logger.warning(f"⚠️ Session 不存在: id={session_id}")
        
        return session
        
    except Exception as e:
        logger.error(f"❌ 查詢 Session 失敗: {e}")
        return None


async def list_sessions(
    skip: int = 0,
    limit: int = 100,
    sort_by: str = "-start_time"
) -> list[Session]:
    """
    查詢多個會議（分頁 + 排序）
    
    Args:
        skip: 跳過前 N 筆
        limit: 最多返回 N 筆
        sort_by: 排序欄位（加 '-' 為降序）
            例如："-start_time" 表示最新的在前
    
    Returns:
        list[Session]: 會議列表
    
    範例：
    ```python
    # 取得最新的 10 筆會議
    sessions = await list_sessions(skip=0, limit=10, sort_by="-start_time")
    for session in sessions:
        print(f"{session.title} - {session.start_time}")
    
    # 分頁查詢（第 2 頁，每頁 20 筆）
    page2_sessions = await list_sessions(skip=20, limit=20)
    ```
    """
    try:
        query = Session.find()
        
        # 排序
        if sort_by:
            query = query.sort(sort_by)
        
        # 分頁
        query = query.skip(skip).limit(limit)
        
        sessions = await query.to_list()
        logger.debug(f"✅ 查詢 Sessions 成功: 取得 {len(sessions)} 筆")
        return sessions
        
    except Exception as e:
        logger.error(f"❌ 查詢 Sessions 失敗: {e}")
        return []


async def update_session(
    session_id: str | ObjectId,
    **update_fields
) -> Optional[Session]:
    """
    更新會議欄位
    
    Args:
        session_id: MongoDB ObjectId
        **update_fields: 要更新的欄位（key=value 格式）
    
    Returns:
        Optional[Session]: 更新後的會議物件，不存在則返回 None
    
    範例：
    ```python
    # 更新會議標題和描述
    session = await update_session(
        session_id="507f1f77bcf86cd799439011",
        title="【已結束】產品規劃會議",
        end_time=datetime.utcnow()
    )
    
    # 更新 metadata
    session = await update_session(
        session_id=session.id,
        metadata={"recording_url": "https://..."}
    )
    ```
    """
    try:
        session = await get_session(session_id)
        if not session:
            logger.warning(f"⚠️ Session 不存在，無法更新: id={session_id}")
            return None
        
        # 更新欄位
        for key, value in update_fields.items():
            if hasattr(session, key):
                setattr(session, key, value)
            else:
                logger.warning(f"⚠️ Session 沒有欄位 '{key}'，忽略")
        
        # 儲存變更
        await session.save()
        logger.info(f"✅ 更新 Session 成功: id={session_id}, 更新欄位={list(update_fields.keys())}")
        return session
        
    except Exception as e:
        logger.error(f"❌ 更新 Session 失敗: {e}")
        return None


async def delete_session(session_id: str | ObjectId) -> bool:
    """
    刪除會議（同時刪除關聯的 Transcript 和 AISummary）
    
    Args:
        session_id: MongoDB ObjectId
    
    Returns:
        bool: 是否成功刪除
    
    注意：
        此操作會級聯刪除：
        1. Session 本身
        2. 關聯的 Transcript
        3. 關聯的 AISummary
    
    範例：
    ```python
    success = await delete_session("507f1f77bcf86cd799439011")
    if success:
        print("會議及所有關聯資料已刪除")
    else:
        print("刪除失敗")
    ```
    """
    try:
        session = await get_session(session_id)
        if not session:
            logger.warning(f"⚠️ Session 不存在，無法刪除: id={session_id}")
            return False
        
        # 確保 session_id 是 ObjectId
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
        
        # 刪除關聯的 Transcript
        transcript_result = await Transcript.find(
            Transcript.session_id == session_id
        ).delete()
        logger.info(f"  刪除了 {transcript_result.deleted_count} 筆 Transcript")
        
        # 刪除關聯的 AISummary
        summary_result = await AISummary.find(
            AISummary.session_id == session_id
        ).delete()
        logger.info(f"  刪除了 {summary_result.deleted_count} 筆 AISummary")
        
        # 刪除 Session
        await session.delete()
        logger.info(f"✅ 刪除 Session 成功: id={session_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 刪除 Session 失敗: {e}")
        return False


async def refresh_participants_snapshot(session_id: str | ObjectId) -> Optional[Session]:
    """
    從 Weaviate 刷新參與者快照（定期同步最新姓名）
    
    功能：
    1. 從 Session.participants_snapshot 取得所有 speaker_uuid
    2. 查詢 Weaviate Speaker collection 取得最新 name
    3. 更新 Session.participants_snapshot
    
    Args:
        session_id: MongoDB ObjectId
    
    Returns:
        Optional[Session]: 更新後的會議物件
    
    使用時機：
    - 定期任務（每日/每週）
    - Speaker name 更新後
    - 需要確保姓名最新時
    
    範例：
    ```python
    # 刷新特定會議的參與者資訊
    session = await refresh_participants_snapshot("507f1f77bcf86cd799439011")
    if session:
        print("參與者快照已更新")
        for p in session.participants_snapshot:
            print(f"  - {p['name']} ({p['speaker_uuid']})")
    ```
    
    注意：
        此函式需要與 Weaviate CRUD 整合（Step 2 完成後實作）
        目前只是預留介面，實際邏輯待後續補充
    """
    try:
        session = await get_session(session_id)
        if not session:
            logger.warning(f"⚠️ Session 不存在，無法刷新: id={session_id}")
            return None
        
        # TODO: 在 Step 2 完成 Weaviate CRUD 後，整合以下邏輯：
        # 1. 取得所有 speaker_uuid
        # speaker_uuids = [p.get("speaker_uuid") for p in session.participants_snapshot]
        #
        # 2. 查詢 Weaviate Speaker collection
        # from modules.database.weaviate_crud import batch_get_speakers
        # speakers = await batch_get_speakers(speaker_uuids)
        #
        # 3. 更新 participants_snapshot
        # for participant in session.participants_snapshot:
        #     speaker_uuid = participant["speaker_uuid"]
        #     speaker = speakers.get(speaker_uuid)
        #     if speaker:
        #         participant["name"] = speaker.name
        #         participant["is_authenticated"] = speaker.is_authenticated
        #
        # 4. 儲存變更
        # await session.save()
        
        logger.info(f"⚠️ refresh_participants_snapshot 尚未完整實作（需 Step 2）")
        logger.info(f"  當前 Session: id={session_id}, 參與者數={len(session.participants_snapshot)}")
        
        return session
        
    except Exception as e:
        logger.error(f"❌ 刷新 participants_snapshot 失敗: {e}")
        return None


# ===========================
# Transcript CRUD
# ===========================

async def create_transcript(
    session_id: str | ObjectId,
    segments: list[dict[str, Any]],
    language: str = "zh-TW",
    update_session_snapshot: bool = False
) -> Transcript:
    """
    建立新的逐字稿（meeting-level storage）
    
    Args:
        session_id: 關聯的 Session ID
        segments: 逐字稿片段列表
            格式：[{
                "speaker_uuid": "uuid-123",
                "text": "大家好",
                "start_time": 0.0,
                "end_time": 1.5,
                "confidence": 0.95
            }]
        language: 語言代碼（預設為繁體中文）
        update_session_snapshot: 是否同時更新 Session.participants_snapshot
    
    Returns:
        Transcript: 建立的逐字稿物件
    
    Raises:
        ValueError: 如果參數無效或 Session 不存在
        Exception: 如果資料庫操作失敗
    
    範例：
    ```python
    transcript = await create_transcript(
        session_id="507f1f77bcf86cd799439011",
        segments=[
            {"speaker_uuid": "uuid-456", "text": "歡迎大家", "start_time": 0.0, "end_time": 2.0},
            {"speaker_uuid": "uuid-789", "text": "謝謝主持人", "start_time": 2.5, "end_time": 4.0}
        ],
        update_session_snapshot=True  # 自動更新 Session 快照
    )
    print(f"逐字稿總長度: {transcript.get_total_duration()} 秒")
    ```
    """
    try:
        # 確保 session_id 是 ObjectId
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
        
        # 驗證 Session 是否存在
        session = await get_session(session_id)
        if not session:
            raise ValueError(f"Session 不存在: id={session_id}")
        
        # 驗證 segments
        if not segments:
            raise ValueError("segments 不可為空")
        
        # 建立 TranscriptSegment 物件
        segment_objects = [
            TranscriptSegment(**seg) for seg in segments
        ]
        
        # 建立 Transcript 物件
        transcript = Transcript(
            session_id=session_id,
            segments=segment_objects,
            language=language
        )
        
        # 儲存到 MongoDB
        await transcript.insert()
        logger.info(f"✅ 建立 Transcript 成功: id={transcript.id}, session_id={session_id}, 片段數={len(segments)}")
        
        # 可選：更新 Session.participants_snapshot
        if update_session_snapshot:
            # 收集所有出現的 speaker_uuid
            speaker_uuids = list(set(seg.speaker_uuid for seg in segment_objects))
            logger.info(f"  準備更新 Session 快照，發現 {len(speaker_uuids)} 位說話者")
            
            # TODO: 在 Step 2 完成後，整合 Weaviate 查詢邏輯
            # 現階段暫時不執行實際更新
            logger.warning(f"  ⚠️ update_session_snapshot 功能需 Step 2 完成後實作")
        
        return transcript
        
    except ValueError as ve:
        logger.error(f"❌ 參數驗證失敗: {ve}")
        raise
    except Exception as e:
        logger.error(f"❌ 建立 Transcript 失敗: {e}")
        raise


async def get_transcript(transcript_id: str | ObjectId) -> Optional[Transcript]:
    """
    根據 ID 查詢單一逐字稿
    
    Args:
        transcript_id: MongoDB ObjectId
    
    Returns:
        Optional[Transcript]: 找到的逐字稿物件，不存在則返回 None
    
    範例：
    ```python
    transcript = await get_transcript("507f191e810c19729de860ea")
    if transcript:
        # 取得完整逐字稿
        full_text = await transcript.get_full_transcript()
        print(full_text)
        
        # 取得純文字（無說話者資訊）
        plain_text = transcript.get_plain_text()
        print(plain_text)
    else:
        print("找不到該逐字稿")
    ```
    """
    try:
        # 確保 transcript_id 是 ObjectId
        if isinstance(transcript_id, str):
            transcript_id = ObjectId(transcript_id)
        
        transcript = await Transcript.get(transcript_id)
        
        if transcript:
            logger.debug(f"✅ 查詢 Transcript 成功: id={transcript_id}")
        else:
            logger.warning(f"⚠️ Transcript 不存在: id={transcript_id}")
        
        return transcript
        
    except Exception as e:
        logger.error(f"❌ 查詢 Transcript 失敗: {e}")
        return None


async def get_transcript_by_session(session_id: str | ObjectId) -> Optional[Transcript]:
    """
    根據 Session ID 查詢逐字稿
    
    Args:
        session_id: MongoDB ObjectId
    
    Returns:
        Optional[Transcript]: 找到的逐字稿物件，不存在則返回 None
    
    範例：
    ```python
    transcript = await get_transcript_by_session("507f1f77bcf86cd799439011")
    if transcript:
        duration = transcript.get_total_duration()
        print(f"會議長度: {duration:.2f} 秒")
    ```
    """
    try:
        # 確保 session_id 是 ObjectId
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
        
        transcript = await Transcript.find_one(Transcript.session_id == session_id)
        
        if transcript:
            logger.debug(f"✅ 查詢 Transcript 成功: session_id={session_id}")
        else:
            logger.warning(f"⚠️ 該 Session 沒有 Transcript: session_id={session_id}")
        
        return transcript
        
    except Exception as e:
        logger.error(f"❌ 查詢 Transcript 失敗: {e}")
        return None


async def update_transcript(
    transcript_id: str | ObjectId,
    **update_fields
) -> Optional[Transcript]:
    """
    更新逐字稿欄位
    
    Args:
        transcript_id: MongoDB ObjectId
        **update_fields: 要更新的欄位
    
    Returns:
        Optional[Transcript]: 更新後的逐字稿物件
    
    範例：
    ```python
    # 更新語言
    transcript = await update_transcript(
        transcript_id="507f191e810c19729de860ea",
        language="en-US"
    )
    
    # 更新片段（完整替換）
    transcript = await update_transcript(
        transcript_id=transcript.id,
        segments=[
            {"speaker_uuid": "uuid-456", "text": "更新後的內容", "start_time": 0.0, "end_time": 2.0}
        ]
    )
    ```
    """
    try:
        transcript = await get_transcript(transcript_id)
        if not transcript:
            logger.warning(f"⚠️ Transcript 不存在，無法更新: id={transcript_id}")
            return None
        
        # 更新欄位
        for key, value in update_fields.items():
            if key == "segments" and isinstance(value, list):
                # 特殊處理 segments（需要轉換成 TranscriptSegment 物件）
                segment_objects = [TranscriptSegment(**seg) if isinstance(seg, dict) else seg for seg in value]
                setattr(transcript, key, segment_objects)
            elif hasattr(transcript, key):
                setattr(transcript, key, value)
            else:
                logger.warning(f"⚠️ Transcript 沒有欄位 '{key}'，忽略")
        
        # 儲存變更
        await transcript.save()
        logger.info(f"✅ 更新 Transcript 成功: id={transcript_id}, 更新欄位={list(update_fields.keys())}")
        return transcript
        
    except Exception as e:
        logger.error(f"❌ 更新 Transcript 失敗: {e}")
        return None


async def delete_transcript(transcript_id: str | ObjectId) -> bool:
    """
    刪除逐字稿
    
    Args:
        transcript_id: MongoDB ObjectId
    
    Returns:
        bool: 是否成功刪除
    
    範例：
    ```python
    success = await delete_transcript("507f191e810c19729de860ea")
    if success:
        print("逐字稿已刪除")
    ```
    """
    try:
        transcript = await get_transcript(transcript_id)
        if not transcript:
            logger.warning(f"⚠️ Transcript 不存在，無法刪除: id={transcript_id}")
            return False
        
        await transcript.delete()
        logger.info(f"✅ 刪除 Transcript 成功: id={transcript_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 刪除 Transcript 失敗: {e}")
        return False


# ===========================
# AISummary CRUD
# ===========================

async def create_ai_summary(
    session_id: str | ObjectId,
    summary: str,
    key_points: Optional[list[str]] = None,
    action_items: Optional[list[str]] = None,
    sentiment: Optional[dict[str, Any]] = None,
    participant_analysis: Optional[list[dict[str, Any]]] = None,
    topics: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None
) -> AISummary:
    """
    建立新的 AI 摘要
    
    Args:
        session_id: 關聯的 Session ID
        summary: 摘要文字
        key_points: 重點列表
        action_items: 行動事項
        sentiment: 情緒分析（格式：{"overall": "positive", "score": 0.8}）
        participant_analysis: 參與者分析（格式：[{"speaker_uuid": "...", "speaking_time": 120.5}]）
        topics: 主題標籤
        metadata: 額外元資料
    
    Returns:
        AISummary: 建立的 AI 摘要物件
    
    Raises:
        ValueError: 如果參數無效或 Session 不存在
        Exception: 如果資料庫操作失敗
    
    範例：
    ```python
    summary = await create_ai_summary(
        session_id="507f1f77bcf86cd799439011",
        summary="本次會議討論了 Q1 目標與資源分配，決定優先開發功能 A。",
        key_points=["Q1 目標確認", "資源分配方案", "功能 A 優先開發"],
        action_items=["王五負責功能 A 設計", "李四準備資源評估報告"],
        sentiment={"overall": "positive", "score": 0.85},
        topics=["產品規劃", "資源管理"]
    )
    print(f"AI 摘要 ID: {summary.id}")
    ```
    """
    try:
        # 確保 session_id 是 ObjectId
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
        
        # 驗證 Session 是否存在
        session = await get_session(session_id)
        if not session:
            raise ValueError(f"Session 不存在: id={session_id}")
        
        # 驗證 summary
        if not summary:
            raise ValueError("summary 為必填欄位")
        
        # 建立 AISummary 物件
        ai_summary = AISummary(
            session_id=session_id,
            summary=summary,
            key_points=key_points or [],
            action_items=action_items or [],
            sentiment=sentiment or {},
            participant_analysis=participant_analysis or [],
            topics=topics or [],
            metadata=metadata or {}
        )
        
        # 儲存到 MongoDB
        await ai_summary.insert()
        logger.info(f"✅ 建立 AISummary 成功: id={ai_summary.id}, session_id={session_id}")
        return ai_summary
        
    except ValueError as ve:
        logger.error(f"❌ 參數驗證失敗: {ve}")
        raise
    except Exception as e:
        logger.error(f"❌ 建立 AISummary 失敗: {e}")
        raise


async def get_ai_summary(summary_id: str | ObjectId) -> Optional[AISummary]:
    """
    根據 ID 查詢單一 AI 摘要
    
    Args:
        summary_id: MongoDB ObjectId
    
    Returns:
        Optional[AISummary]: 找到的 AI 摘要物件，不存在則返回 None
    
    範例：
    ```python
    summary = await get_ai_summary("507f191e810c19729de860eb")
    if summary:
        print(f"摘要: {summary.summary}")
        print(f"情緒: {summary.get_overall_sentiment()}")
        print(f"最活躍參與者: {summary.get_most_active_participant()}")
    ```
    """
    try:
        # 確保 summary_id 是 ObjectId
        if isinstance(summary_id, str):
            summary_id = ObjectId(summary_id)
        
        ai_summary = await AISummary.get(summary_id)
        
        if ai_summary:
            logger.debug(f"✅ 查詢 AISummary 成功: id={summary_id}")
        else:
            logger.warning(f"⚠️ AISummary 不存在: id={summary_id}")
        
        return ai_summary
        
    except Exception as e:
        logger.error(f"❌ 查詢 AISummary 失敗: {e}")
        return None


async def list_ai_summaries_by_session(session_id: str | ObjectId) -> list[AISummary]:
    """
    查詢特定 Session 的所有 AI 摘要
    
    Args:
        session_id: MongoDB ObjectId
    
    Returns:
        list[AISummary]: AI 摘要列表（按建立時間降序）
    
    範例：
    ```python
    summaries = await list_ai_summaries_by_session("507f1f77bcf86cd799439011")
    print(f"該會議有 {len(summaries)} 個 AI 摘要")
    for summary in summaries:
        print(f"  - {summary.created_at}: {summary.summary[:50]}...")
    ```
    """
    try:
        # 確保 session_id 是 ObjectId
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
        
        summaries = await AISummary.find(
            AISummary.session_id == session_id
        ).sort("-created_at").to_list()
        
        logger.debug(f"✅ 查詢 AISummaries 成功: session_id={session_id}, 取得 {len(summaries)} 筆")
        return summaries
        
    except Exception as e:
        logger.error(f"❌ 查詢 AISummaries 失敗: {e}")
        return []


async def update_ai_summary(
    summary_id: str | ObjectId,
    **update_fields
) -> Optional[AISummary]:
    """
    更新 AI 摘要欄位
    
    Args:
        summary_id: MongoDB ObjectId
        **update_fields: 要更新的欄位
    
    Returns:
        Optional[AISummary]: 更新後的 AI 摘要物件
    
    範例：
    ```python
    # 更新摘要內容
    summary = await update_ai_summary(
        summary_id="507f191e810c19729de860eb",
        summary="【已修正】本次會議...",
        key_points=["新增的重點 1", "新增的重點 2"]
    )
    
    # 更新情緒分析
    summary = await update_ai_summary(
        summary_id=summary.id,
        sentiment={"overall": "neutral", "score": 0.5}
    )
    ```
    """
    try:
        ai_summary = await get_ai_summary(summary_id)
        if not ai_summary:
            logger.warning(f"⚠️ AISummary 不存在，無法更新: id={summary_id}")
            return None
        
        # 更新欄位
        for key, value in update_fields.items():
            if hasattr(ai_summary, key):
                setattr(ai_summary, key, value)
            else:
                logger.warning(f"⚠️ AISummary 沒有欄位 '{key}'，忽略")
        
        # 儲存變更
        await ai_summary.save()
        logger.info(f"✅ 更新 AISummary 成功: id={summary_id}, 更新欄位={list(update_fields.keys())}")
        return ai_summary
        
    except Exception as e:
        logger.error(f"❌ 更新 AISummary 失敗: {e}")
        return None


async def delete_ai_summary(summary_id: str | ObjectId) -> bool:
    """
    刪除 AI 摘要
    
    Args:
        summary_id: MongoDB ObjectId
    
    Returns:
        bool: 是否成功刪除
    
    範例：
    ```python
    success = await delete_ai_summary("507f191e810c19729de860eb")
    if success:
        print("AI 摘要已刪除")
    ```
    """
    try:
        ai_summary = await get_ai_summary(summary_id)
        if not ai_summary:
            logger.warning(f"⚠️ AISummary 不存在，無法刪除: id={summary_id}")
            return False
        
        await ai_summary.delete()
        logger.info(f"✅ 刪除 AISummary 成功: id={summary_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 刪除 AISummary 失敗: {e}")
        return False


# ===========================
# 測試程式
# ===========================

if __name__ == "__main__":
    import asyncio
    from modules.database.init_mongodb import initialize_mongodb, close_mongodb
    
    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test():
        print("=" * 60)
        print("MongoDB CRUD 測試")
        print("=" * 60)
        print()
        
        # 初始化 MongoDB
        print("【初始化】連線 MongoDB...")
        await initialize_mongodb()
        print()
        
        try:
            # ===== 測試 Session CRUD =====
            print("=" * 60)
            print("【測試 1】Session CRUD")
            print("=" * 60)
            
            # 建立 Session
            print("\n1.1 建立 Session...")
            session = await create_session(
                title="測試會議",
                description="這是一個測試會議",
                participants_snapshot=[
                    {"speaker_uuid": "test-uuid-1", "name": "測試者 A", "is_authenticated": True},
                    {"speaker_uuid": "test-uuid-2", "name": "測試者 B", "is_authenticated": False}
                ]
            )
            print(f"✅ 建立成功: {session.id}")
            print(f"   標題: {session.title}")
            print(f"   參與者: {session.get_all_participant_names()}")
            
            # 查詢 Session
            print("\n1.2 查詢 Session...")
            found_session = await get_session(session.id)
            print(f"✅ 查詢成功: {found_session.title}")
            
            # 列出 Sessions
            print("\n1.3 列出所有 Sessions...")
            all_sessions = await list_sessions(limit=5)
            print(f"✅ 共有 {len(all_sessions)} 筆 Session")
            
            # 更新 Session
            print("\n1.4 更新 Session...")
            updated_session = await update_session(
                session.id,
                title="【已更新】測試會議"
            )
            print(f"✅ 更新成功: {updated_session.title}")
            
            # ===== 測試 Transcript CRUD =====
            print("\n" + "=" * 60)
            print("【測試 2】Transcript CRUD")
            print("=" * 60)
            
            # 建立 Transcript
            print("\n2.1 建立 Transcript...")
            transcript = await create_transcript(
                session_id=session.id,
                segments=[
                    {"speaker_uuid": "test-uuid-1", "text": "大家好", "start_time": 0.0, "end_time": 1.5},
                    {"speaker_uuid": "test-uuid-2", "text": "你好", "start_time": 2.0, "end_time": 3.0}
                ]
            )
            print(f"✅ 建立成功: {transcript.id}")
            print(f"   片段數: {len(transcript.segments)}")
            print(f"   總長度: {transcript.get_total_duration()} 秒")
            
            # 查詢 Transcript
            print("\n2.2 查詢 Transcript...")
            found_transcript = await get_transcript(transcript.id)
            print(f"✅ 查詢成功")
            print(f"   純文字: {found_transcript.get_plain_text()}")
            
            # 根據 Session 查詢
            print("\n2.3 根據 Session 查詢 Transcript...")
            session_transcript = await get_transcript_by_session(session.id)
            print(f"✅ 查詢成功: {session_transcript.id}")
            
            # ===== 測試 AISummary CRUD =====
            print("\n" + "=" * 60)
            print("【測試 3】AISummary CRUD")
            print("=" * 60)
            
            # 建立 AISummary
            print("\n3.1 建立 AISummary...")
            summary = await create_ai_summary(
                session_id=session.id,
                summary="這是一個測試摘要",
                key_points=["重點 1", "重點 2"],
                action_items=["待辦事項 1"],
                sentiment={"overall": "positive", "score": 0.8}
            )
            print(f"✅ 建立成功: {summary.id}")
            print(f"   摘要: {summary.summary}")
            print(f"   情緒: {summary.get_overall_sentiment()}")
            
            # 查詢 AISummary
            print("\n3.2 查詢 AISummary...")
            found_summary = await get_ai_summary(summary.id)
            print(f"✅ 查詢成功")
            print(f"   重點數: {len(found_summary.key_points)}")
            
            # 根據 Session 列出
            print("\n3.3 列出 Session 的所有 AISummaries...")
            session_summaries = await list_ai_summaries_by_session(session.id)
            print(f"✅ 共有 {len(session_summaries)} 筆")
            
            # ===== 測試刪除 =====
            print("\n" + "=" * 60)
            print("【測試 4】刪除測試")
            print("=" * 60)
            
            # 刪除 Session（級聯刪除）
            print("\n4.1 刪除 Session（會同時刪除 Transcript 和 AISummary）...")
            success = await delete_session(session.id)
            print(f"✅ 刪除{'成功' if success else '失敗'}")
            
            # 驗證刪除
            print("\n4.2 驗證刪除...")
            deleted_session = await get_session(session.id)
            deleted_transcript = await get_transcript(transcript.id)
            deleted_summary = await get_ai_summary(summary.id)
            print(f"Session 存在: {deleted_session is not None}")
            print(f"Transcript 存在: {deleted_transcript is not None}")
            print(f"AISummary 存在: {deleted_summary is not None}")
            
            print("\n" + "=" * 60)
            print("測試完成！")
            print("=" * 60)
            
        finally:
            # 關閉連線
            print("\n【清理】關閉 MongoDB 連線...")
            await close_mongodb()
    
    # 執行測試
    asyncio.run(test())
