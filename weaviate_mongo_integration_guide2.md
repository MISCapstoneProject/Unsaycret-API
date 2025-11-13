# Weaviate + MongoDB 整合規格文件 v2.0

本文件說明如何在專題專案中，將 **Weaviate（向量資料庫）** 與 **MongoDB（文檔資料庫）** 有效整合，並描述架構、任務拆分、API 層邏輯、資料模型與注意事項。

**文件更新日期**: 2024-11-14  
**版本**: v2.0 - 三層架構 + Transcript 重新設計

---

# 📋 目錄
1. [系統整體目標](#1-系統整體目標)
2. [架構概述](#2-架構概述)
3. [三層架構設計](#3-三層架構設計-新增)
4. [MongoDB 資料模型](#4-mongodb-資料模型)
5. [Weaviate 資料模型](#5-weaviate-資料模型)
6. [API 層責任](#6-api-層責任)
7. [已完成的工作](#7-已完成的工作)
8. [待執行的工作](#8-待執行的工作)
9. [檔案結構規劃](#9-檔案結構規劃)
10. [資料流設計](#10-資料流設計)
11. [實作步驟](#11-實作步驟)
12. [測試計劃](#12-測試計劃)

---

# 1. 系統整體目標

本專案需要同時管理：
- **語者與聲紋**（向量搜尋、聲紋比對）→ 使用 *Weaviate*
- **會議資料、逐字稿、AI 摘要** → 使用 *MongoDB*

兩個資料庫需要透過**三層架構**整合：
```
外層：services/data_facade.py（對外統一接口）
     ↓
中間層：modules/database/database_interface.py（協調層）
     ↓
底層：weaviate_crud.py + mongodb_crud.py（資料庫專屬操作）
      weaviate_connection.py + mongodb_connection.py（連線管理）
```

---

# 2. 架構概述

## 2.1 資料分層
### **Weaviate（向量資料庫）**
負責：
- Speaker (語者)
- VoicePrint (聲紋向量)
- 向量搜尋
- Speaker ↔ VoicePrint 的 reference

### **MongoDB（文檔資料庫）**
負責：
- Session（會議）
- **Transcript（逐字稿）** - 取代原本的 SpeechLog
- AISummary（AI 摘要）
- metadata（非向量、長文資料）

---

# 3. 三層架構設計（新增）

## 3.1 架構圖
```
┌─────────────────────────────────────────────────────┐
│  外層 (External API Layer)                          │
│  services/data_facade.py                            │
│  - 提供給 api/api.py, VID_manager.py 等調用         │
│  - 統一的資料存取接口                                │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  中間層 (Database Interface Layer)                   │
│  modules/database/database_interface.py             │
│  - 協調 Weaviate 和 MongoDB                         │
│  - 實現跨資料庫的整合邏輯（JOIN）                     │
└─────────────────────────────────────────────────────┘
                         ↓
        ┌────────────────┴────────────────┐
        ↓                                  ↓
┌──────────────────┐            ┌──────────────────┐
│  Weaviate 底層    │            │  MongoDB 底層     │
├──────────────────┤            ├──────────────────┤
│ weaviate_        │            │ mongodb_         │
│ connection.py    │            │ connection.py    │
│ - 連線管理        │            │ - 連線管理        │
│ - 集合檢查        │            │ - Beanie 初始化   │
│ - ping           │            │ - ping           │
├──────────────────┤            ├──────────────────┤
│ weaviate_crud.py │            │ mongodb_crud.py  │
│ - Speaker CRUD   │            │ - Session CRUD   │
│ - VoicePrint     │            │ - Transcript     │
│   CRUD           │            │   CRUD           │
│ - 向量搜尋        │            │ - AISummary      │
│                  │            │   CRUD           │
└──────────────────┘            └──────────────────┘
```

## 3.2 每層的職責

### 🔹 外層 (data_facade.py)
- **對外暴露的統一接口**
- 業務邏輯層調用此層
- 簡化複雜的資料庫操作
- 示例方法：
  ```python
  async def get_session_with_participants(session_id: str) -> dict
  async def get_transcript_with_speakers(transcript_id: str) -> dict
  async def create_speaker_with_voiceprint(name: str, embedding: np.ndarray) -> str
  ```

### 🔹 中間層 (database_interface.py)
- **資料庫協調者**
- 決定操作該打到哪個資料庫
- 實現跨資料庫 JOIN 邏輯
- 調用 integration_service 進行資料整合
- 示例方法：
  ```python
  async def get_session(session_id: str) -> Session
  async def enrich_session_with_speakers(session: Session) -> dict
  def get_speaker(speaker_uuid: str) -> dict
  ```

### 🔹 底層 (CRUD + Connection)
- **純資料庫操作**
- 不含業務邏輯
- 每個資料庫分為兩個檔案：
  - `*_connection.py`: 連線管理、初始化、健康檢查
  - `*_crud.py`: 純 CRUD 操作

---

# 4. MongoDB 資料模型

## 4.1 Session Collection（重新設計）
```python
from beanie import Document
from pydantic import Field
from typing import Optional, List, Dict
from datetime import datetime


class Session(Document):
    """
    會議資料模型
    
    ⚠️ 重要設計決策（2025-01-14）：
    - **使用 MongoDB ObjectId 作為主鍵**（Beanie 自動生成的 `id` 欄位）
    - **不使用自訂 session_id**（避免刪除 Session 時產生 ID 間隙）
    - 理由：使用者不會看到 Session ID，前端也不需要依賴特定 ID 格式
    
    關聯關係：
    - 一個 Session 對應一個 Transcript（1:1 關係）
    - Transcript.session_id 儲存此 Session 的 ObjectId 字串（如 "507f1f77bcf86cd799439011"）
    - participants 儲存 Weaviate Speaker UUID 列表
    - participants_snapshot 儲存會議當時的參與者名稱快照
    
    設計決策：
    - 參與者資訊存在 Session 而非 Transcript
    - 理由：Session 定義「誰參加會議」，Transcript 只記錄「誰說了什麼」
    - 好處：資料正規化，更新方便，符合邏輯
    
    參與者快照 (participants_snapshot)：
    - 記錄「會議開始時」每個參與者的名字
    - 格式：{"uuid": {"full_name": "王小明", "nickname": "小明"}}
    - 用途：
      1. 顯示「當時名字」（預設模式）
      2. 語者改名後，仍可追溯歷史記錄
      3. 避免查詢 Weaviate 即可顯示參與者名單
    
    參與者快照更新策略（2025-01-14 確認）：
    
    自動更新和手動更新一定要實作

    1️⃣ **自動更新**：Transcript 儲存時（第一次建立時）
       - 首次儲存 Transcript 時，自動將 participants_snapshot 更新到 Session
       - 確保快照包含完整的參與者資訊
    
    2️⃣ **手動更新**：前端觸發
       - 前端在 Session 詳情頁提供「重新整理參與者名單」按鈕
       - 呼叫 `PATCH /sessions/{session_id}/participants` 更新快照為最新資料
    
    3️⃣ **可選更新**：語者改名時
       - 預設：保留歷史快照（不自動更新）
       - 可選：提供 `update_historical_sessions=true` 參數，批次更新所有相關 Session
    
    顯示模式：
    - **快照模式**（預設）：顯示 Session.participants_snapshot（歷史記錄）
    """
    # 注意：Beanie 會自動加入 `id: PydanticObjectId` 欄位（對應 MongoDB _id）
    # 不需要手動定義 session_id
    
    session_type: str = Field(default="", description="會議類型")
    title: str = Field(default="", description="會議標題")
    start_time: Optional[datetime] = Field(default=None, description="開始時間")
    end_time: Optional[datetime] = Field(default=None, description="結束時間")
    summary: Optional[str] = Field(default=None, description="會議摘要")
    
    # 參與者資訊
    participants: List[str] = Field(
        default_factory=list, 
        description="參與者 UUID 列表（來自 Weaviate Speaker）"
    )
    
    # 參與者快照（記錄會議開始時的名字）
    participants_snapshot: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="會議參與者快照 {'uuid': {'full_name': '...', 'nickname': '...'}}"
    )
    
    class Settings:
        name = "sessions"
        indexes = [
            "start_time",
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                # Beanie 自動生成的 _id (ObjectId)
                "id": "507f1f77bcf86cd799439011",
                "session_type": "meeting",
                "title": "專案討論會議",
                "start_time": "2025-01-15T10:00:00",
                "end_time": "2025-01-15T11:30:00",
                "summary": "討論專案進度與下一步計畫",
                "participants": [
                    "550e8400-e29b-41d4-a716-446655440001",
                    "550e8400-e29b-41d4-a716-446655440002"
                ],
                "participants_snapshot": {
                    "550e8400-e29b-41d4-a716-446655440001": {
                        "full_name": "王小明",
                        "nickname": "小明"
                    },
                    "550e8400-e29b-41d4-a716-446655440002": {
                        "full_name": "李小華",
                        "nickname": "小華"
                    }
                }
            }
        }
```

## 4.2 Transcript Collection（重新設計，取代 SpeechLog）

### ⚠️ 重大變更：從句子級改為會議級儲存

**舊設計（SpeechLog）**：
- ❌ 一句話一個 SpeechLog 物件
- ❌ 即時錄音時每句話都立即儲存到資料庫
- ❌ 每句話都有 speaker_uuid, content, timestamp
- ❌ 需要查詢多筆資料才能取得完整會議記錄

**新設計（Transcript）**：
- ✅ 一個會議一個 Transcript 物件
- ✅ **錄音時只在前端即時顯示，不儲存資料庫**
- ✅ **錄完後使用者按「儲存」才一次性寫入資料庫**
- ✅ 不儲存完整文字稿（full_transcript）
- ✅ 使用 `segments` 陣列儲存所有段落
- ✅ 每個 segment **只儲存 UUID**，不儲存 speaker_name

### 🎯 **設計決策理由**

**為什麼選擇會議級儲存？**
1. **符合使用場景**：使用者是「錄完整個會議後才想儲存」，不是「邊錄邊存」
2. **減少資料庫壓力**：一次寫入取代 N 次寫入（N = 句子數量）
3. **資料完整性**：整個會議是一個完整的文件單位，不該被切碎
4. **效能優化**：查詢一筆資料取代查詢 N 筆資料
5. **前端友好**：前端可以繼續即時顯示，但不需要每次都呼叫 API

**為什麼 segment 不儲存 speaker_name？**
1. **資料一致性**：語者改名後，歷史逐字稿可選擇顯示「最新名字」或「當時名字」
2. **減少冗餘**：避免在每個 segment 重複儲存相同的名字
3. **靈活性**：可以輕鬆切換顯示模式（最新 vs 當時）

**效能考量**：
- ❓ **疑問**：會議列表頁面會讀取所有 Transcript 嗎？
- ✅ **答案**：不會！列表頁只查 Session，點開才讀 Transcript
- 🚀 **Mapping 開銷**：UUID → name 的 dict lookup 是 O(1)，可忽略不計

```python
from pydantic import BaseModel, Field
from beanie import Document
from typing import Optional, List, Dict
from datetime import datetime


class TranscriptSegment(BaseModel):
    """
    逐字稿段落 - Pydantic 模型（嵌入在 Transcript 中，不獨立成 collection）
    
    ⚠️ 注意：這不是 Beanie Document，而是 Pydantic BaseModel
    目的：提供型別安全的資料驗證
    
    每個 segment 代表一段連續的語音內容
    
    設計原則：
    - 只儲存 speaker_uuid，不儲存 speaker_name
    - speaker_name 透過 participants_snapshot 或即時查詢 Weaviate 取得
    """
    speaker_uuid: str = Field(..., description="說話者的 UUID（對應 Weaviate Speaker）")
    content: str = Field(..., description="該段落的文字內容")
    timestamp: float = Field(..., description="從會議開始計算的秒數")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="語音辨識信心度 (0.0-1.0)")
    duration: Optional[float] = Field(None, description="該段落的時長（秒）")
    language: str = Field(default="zh-TW", description="語言代碼")


class Transcript(Document):
    """
    逐字稿模型 - 會議級儲存
    
    關聯關係：
    - 一個 Transcript 對應一個 Session（1:1 關係）
    - session_id 用於關聯到 Session
    
    時間戳記說明：
    - created_at: 首次建立時間（第一次錄音開始）
    - last_recorded_at: 最後錄音時間（支援暫停後再錄音）
    - last_edited_at: 最後手動編輯時間（使用者修改文字內容）
    
    完整文字稿組合方式：
    - 需要先從 Session 查詢 participants_snapshot 取得 speaker_uuid → speaker_name 對應表
    - 然後組合：'\n'.join([f"{name}：{seg.content}" for seg in segments])
    
    顯示模式：
    - snapshot（預設）：顯示會議當時的名字（使用 Session.participants_snapshot）
    - current：顯示最新的名字（即時查詢 Weaviate Speaker）
    
    前端傳來的 JSON 格式範例：
    {
        "session_id": "507f1f77bcf86cd799439011",
        "segments": [
            {
                "speaker_uuid": "123e4567-e89b-12d3-a456-426614174000",
                "content": "大家好，歡迎來到今天的會議。",
                "timestamp": 0.0,
                "confidence": 0.95,
                "duration": 3.5,
                "language": "zh-TW"
            }
        ],
        "created_at": "2024-11-14T10:00:00Z",
        "last_recorded_at": "2024-11-14T10:30:00Z"
    }
    """
    session_id: str = Field(..., description="關聯的 Session.id（MongoDB ObjectId 字串，如 '507f1f77bcf86cd799439011'）")
    segments: List[TranscriptSegment] = Field(default_factory=list, description="逐字稿段落陣列")
    
    # 時間戳記
    created_at: datetime = Field(default_factory=datetime.utcnow, description="首次建立時間（第一次錄音）")
    last_recorded_at: Optional[datetime] = Field(None, description="最後錄音時間")
    last_edited_at: Optional[datetime] = Field(None, description="最後編輯時間")
    
    class Settings:
        name = "transcripts"
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "507f1f77bcf86cd799439011",
                "segments": [
                    {
                        "speaker_uuid": "123e4567-e89b-12d3-a456-426614174000",
                        "content": "大家好，歡迎來到今天的會議。",
                        "timestamp": 0.0,
                        "confidence": 0.95,
                        "duration": 3.5,
                        "language": "zh-TW"
                    }
                ],
                "created_at": "2024-11-14T10:00:00Z",
                "last_recorded_at": "2024-11-14T10:30:00Z",
                "last_edited_at": None
            }
        }
    
    # 輔助方法
    def get_full_transcript(self, speaker_name_map: Dict[str, str]) -> str:
        """
        組合所有 segments 成為完整文字稿
        
        Args:
            speaker_name_map: 語者 UUID → 名字的對應表（來自 Session.participants_snapshot）
                              格式：{"uuid": "王小明", ...}
        
        Returns:
            str: 完整的文字稿內容，格式為：
                 "王小明：大家好，歡迎來到今天的會議。\n李小華：謝謝主持人，很高興參加這次會議。"
        """
        lines = []
        for seg in self.segments:
            speaker_name = speaker_name_map.get(seg.speaker_uuid, "未知")
            lines.append(f"{speaker_name}：{seg.content}")
        return '\n'.join(lines)
    
    def get_plain_text(self) -> str:
        """
        取得純文字稿（不含語者名稱）
        
        Returns:
            str: 純文字內容，格式為：
                 "大家好，歡迎來到今天的會議。\n謝謝主持人，很高興參加這次會議。"
        """
        return '\n'.join([seg.content for seg in self.segments])
```

### 📊 資料結構對比

| 項目 | 舊設計 (SpeechLog) | 新設計 (Transcript) |
|------|-------------------|-------------------|
| 儲存粒度 | 句子級（一句一筆） | 會議級（一會議一筆） |
| 完整文字稿 | 需要查詢多筆後組合 | 在 segments 陣列中 |
| Session 關係 | 1:N | 1:1 |
| speaker_name | 每句都儲存 | 不儲存，使用 participants_snapshot |
| 時間戳記 | 只有 timestamp | created_at + last_recorded_at + last_edited_at |
| 錄音支援 | 不支援多次錄音 | 支援暫停後再錄音 |
| 編輯追蹤 | 無 | last_edited_at 追蹤手動編輯 |
| 改名處理 | 無法追溯歷史 | 可選「最新」或「當時」名字 |

## 4.3 AISummary Collection
```python
class AISummary(Document):
    """
    AI 生成的會議摘要
    
    關聯關係：
    - 一個 AISummary 對應一個 Session
    - session_id 儲存 Session 的 ObjectId 字串（如 "507f1f77bcf86cd799439011"）
    
    查詢方式：
    - 可透過 session_id 查詢該場合的所有摘要
    - 建議：在 session_id 欄位加上索引以提升查詢效能
    """
    session_id: str = Field(..., description="關聯的 Session.id（MongoDB ObjectId 字串）")
    summary: str = Field(..., description="會議摘要內容")
    sentiment: dict = Field(default_factory=dict, description="情感分析結果")
    key_points: List[str] = Field(default_factory=list, description="重點摘要")
    action_items: List[str] = Field(default_factory=list, description="待辦事項")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="建立時間")
    
    class Settings:
        name = "ai_summaries"
        indexes = [
            "session_id",  # 加速依 session_id 查詢
            "created_at",
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "507f1f77bcf86cd799439011",
                "summary": "會議討論了專案進度，確認了下階段目標",
                "sentiment": {"positive": 0.7, "neutral": 0.2, "negative": 0.1},
                "key_points": [
                    "確認專案時程",
                    "分配任務責任"
                ],
                "action_items": [
                    "王小明：完成需求文件",
                    "李小華：準備設計稿"
                ],
                "created_at": "2025-01-15T12:00:00"
            }
        }
```

---

# 5. Weaviate 資料模型

## 5.1 Speaker Collection（V2 版本，保持不變）
```python
{
    "class": "Speaker",
    "properties": [
        {"name": "speaker_id", "dataType": ["int"]},
        {"name": "full_name", "dataType": ["text"]},
        {"name": "nickname", "dataType": ["text"]},
        {"name": "gender", "dataType": ["text"]},
        {"name": "created_at", "dataType": ["date"]},
        {"name": "last_active_at", "dataType": ["date"]},
        {"name": "meet_count", "dataType": ["int"]},
        {"name": "meet_days", "dataType": ["int"]},
        {"name": "voiceprint_ids", "dataType": ["text[]"]},
        {"name": "first_audio", "dataType": ["text"]}
    ]
}
```

## 5.2 VoicePrint Collection（V2 版本，保持不變）
```python
{
    "class": "VoicePrint",
    "vectorizer": "none",
    "properties": [
        {"name": "created_at", "dataType": ["date"]},
        {"name": "updated_at", "dataType": ["date"]},
        {"name": "update_count", "dataType": ["int"]},
        {"name": "sample_count", "dataType": ["int"]},
        {"name": "quality_score", "dataType": ["number"]},
        {"name": "speaker_name", "dataType": ["text"]},
    ],
    "references": [
        {
            "name": "speaker",
            "target": "Speaker"
        }
    ]
}
```

---

# 6. API 層責任

## 6.1 跨資料庫 JOIN 邏輯

### Session → Speakers JOIN
```python
# 在 integration_service.py 中實現
async def enrich_session_with_speakers(session: Session) -> dict:
    """
    為 Session 補上完整的 Speaker 資料
    
    輸入：Session 物件（含 participants: List[str] 的 UUID）
    輸出：完整的資料結構
    """
    speakers_data = []
    for speaker_uuid in session.participants:
        speaker = database_service.get_speaker(speaker_uuid)
        if speaker:
            speakers_data.append({
                "uuid": speaker_uuid,
                "full_name": speaker.properties.get("full_name"),
                "nickname": speaker.properties.get("nickname"),
                ...
            })
    
    return {
        "session": session.dict(),
        "participants": speakers_data
    }
```

### Transcript → Speakers JOIN（支援顯示模式）
```python
async def enrich_transcript_with_speakers(
    transcript: Transcript, 
    name_mode: str = "snapshot"
) -> dict:
    """
    為 Transcript 的所有 segments 補上 Speaker 資料
    
    Args:
        transcript: Transcript 物件
        name_mode: 顯示模式
            - "snapshot" (預設): 顯示會議當時的名字（使用 participants_snapshot）
            - "current": 顯示最新的名字（即時查詢 Weaviate）
    
    批次優化（僅當 name_mode="current" 時）：
    1. 收集所有不重複的 speaker_uuid
    2. 一次查詢所有需要的 Speakers
    3. 建立 uuid -> speaker_data 的 mapping
    4. 為每個 segment 附加最新的 speaker_name, speaker_nickname
    
    效能分析：
    - snapshot 模式：O(1) dict lookup，極快
    - current 模式：需查詢 Weaviate，但使用批次查詢優化
    """
    
    # 根據顯示模式選擇資料來源
    if name_mode == "current":
        # 查詢 Weaviate 取得最新名字
        unique_speaker_uuids = set(seg.speaker_uuid for seg in transcript.segments)
        
        speaker_map = {}
        for uuid in unique_speaker_uuids:
            speaker = database_service.get_speaker(uuid)
            if speaker:
                speaker_map[uuid] = {
                    "full_name": speaker.properties.get("full_name"),
                    "nickname": speaker.properties.get("nickname")
                }
    else:
        # 使用快照（會議當時的名字）
        # 需要先從 Session 查詢 participants_snapshot
        session = await database_interface.get_session(transcript.session_id)
        speaker_map = session.participants_snapshot
    
    # 組合結果
    enriched_segments = []
    for seg in transcript.segments:
        speaker_info = speaker_map.get(seg.speaker_uuid, {})
        enriched_segments.append({
            "content": seg.content,
            "speaker_uuid": seg.speaker_uuid,
            "speaker_name": speaker_info.get("full_name", "未知"),
            "speaker_nickname": speaker_info.get("nickname", ""),
            "timestamp": seg.timestamp,
            "confidence": seg.confidence,
            "duration": seg.duration,
            "language": seg.language
        })
    
    return {
        "transcript_id": str(transcript.id),
        "session_id": transcript.session_id,
        "segments": enriched_segments,
        "name_mode": name_mode,  # 標記使用的顯示模式
        "created_at": transcript.created_at,
        "last_recorded_at": transcript.last_recorded_at,
        "last_edited_at": transcript.last_edited_at
    }
```

## 6.2 完整 API 端點規範（2025-01-14 新增）

### 🔹 Session API

#### POST /sessions（建立場合）
**用途**：建立新的會議場合

**Request Body**：
```json
{
  "session_type": "meeting",
  "title": "專案討論會議",
  "start_time": "2025-01-15T10:00:00Z"
  // participants_snapshot: 選填，預設 {}（通常由前端在儲存 Transcript 時提供）
}
```

**Response**：
```json
{
  "success": true,
  "data": {
    "session_id": "507f1f77bcf86cd799439011",  // MongoDB ObjectId
    "session_type": "meeting",
    "title": "專案討論會議",
    "start_time": "2025-01-15T10:00:00Z",
    "participants": [],  // 初始為空
    "participants_snapshot": {}  // 初始為空
  }
}
```

---

#### GET /sessions（列出所有場合）
**用途**：取得所有會議場合列表

**Query Parameters**：
- `skip`: int（分頁偏移量，預設 0）
- `limit`: int（每頁數量，預設 20）
- `sort_by`: str（排序欄位，預設 "start_time"）
- `order`: str（"asc" 或 "desc"，預設 "desc"）

**Response**：
```json
{
  "success": true,
  "data": {
    "sessions": [
      {
        "session_id": "507f1f77bcf86cd799439011",
        "session_type": "meeting",
        "title": "專案討論會議",
        "start_time": "2025-01-15T10:00:00Z",
        "end_time": "2025-01-15T11:30:00Z",
        "summary": "討論專案進度",
        "participants": [
          "550e8400-e29b-41d4-a716-446655440001",
          "550e8400-e29b-41d4-a716-446655440002"
        ],
        "participants_snapshot": {
          "550e8400-e29b-41d4-a716-446655440001": {
            "full_name": "王小明",
            "nickname": "小明"
          },
          "550e8400-e29b-41d4-a716-446655440002": {
            "full_name": "李小華",
            "nickname": "小華"
          }
        }
      }
    ],
    "total": 42,
    "skip": 0,
    "limit": 20
  }
}
```

⚠️ **注意**：前端需要 `participants_snapshot` 來顯示參與者名單！

---

#### GET /sessions/{session_id}（取得場合詳情）
**用途**：取得單一會議場合的完整資料

**Response**：
```json
{
  "success": true,
  "data": {
    "session_id": "507f1f77bcf86cd799439011",
    "session_type": "meeting",
    "title": "專案討論會議",
    "start_time": "2025-01-15T10:00:00Z",
    "end_time": "2025-01-15T11:30:00Z",
    "summary": "討論專案進度",
    "participants": [
      "550e8400-e29b-41d4-a716-446655440001",
      "550e8400-e29b-41d4-a716-446655440002"
    ],
    "participants_snapshot": {
      "550e8400-e29b-41d4-a716-446655440001": {
        "full_name": "王小明",
        "nickname": "小明"
      },
      "550e8400-e29b-41d4-a716-446655440002": {
        "full_name": "李小華",
        "nickname": "小華"
      }
    }
  }
}
```

---

#### PATCH /sessions/{session_id}/participants（更新參與者快照）
**用途**：手動重新整理參與者名單（將快照更新為最新資料）

**Request Body**：
```json
{
  "action": "refresh"  // 重新從 Weaviate 拉取最新語者資料
}
```

**Response**：
```json
{
  "success": true,
  "message": "參與者快照已更新",
  "data": {
    "updated_participants": {
      "550e8400-e29b-41d4-a716-446655440001": {
        "full_name": "王大明",  // 更新後的名字
        "nickname": "大明"
      }
    }
  }
}
```

**後端邏輯**：
```python
async def refresh_session_participants(session_id: str):
    """重新整理 Session 的參與者快照為最新資料"""
    session = await Session.get(session_id)
    
    # 從 Weaviate 查詢所有參與者的最新資料
    updated_snapshot = {}
    for speaker_uuid in session.participants:
        speaker = database_service.get_speaker(speaker_uuid)
        if speaker:
            updated_snapshot[speaker_uuid] = {
                "full_name": speaker.properties.get("full_name"),
                "nickname": speaker.properties.get("nickname")
            }
    
    session.participants_snapshot = updated_snapshot
    await session.save()
    return updated_snapshot
```

---

### 🔹 Transcript API

#### POST /transcript（儲存逐字稿）
**用途**：錄音完成後，一次性儲存整個會議的逐字稿

**重要行為**：
1. 儲存 Transcript
2. 根據 `update_session_snapshot` 參數決定是否更新 Session 的 participants_snapshot
3. 預設為 `true`（第一次儲存時應更新）

**Request Body**：
```json
{
  "session_id": "507f1f77bcf86cd799439011",
  "segments": [
    {
      "speaker_uuid": "550e8400-e29b-41d4-a716-446655440001",
      "content": "大家好，歡迎來到今天的會議。",
      "timestamp": 0.0,
      "confidence": 0.95,
      "duration": 3.5,
      "language": "zh-TW"
    },
    {
      "speaker_uuid": "550e8400-e29b-41d4-a716-446655440002",
      "content": "謝謝主持人，很高興參加這次會議。",
      "timestamp": 3.5,
      "confidence": 0.92,
      "duration": 4.2,
      "language": "zh-TW"
    }
  ],
  "participants_snapshot": {
    "550e8400-e29b-41d4-a716-446655440001": {
      "full_name": "王小明",
      "nickname": "小明"
    },
    "550e8400-e29b-41d4-a716-446655440002": {
      "full_name": "李小華",
      "nickname": "小華"
    }
  },
  "update_session_snapshot": true,  // 是否更新 Session 的 participants_snapshot（預設 true）
  "created_at": "2025-01-15T10:00:00Z",
  "last_recorded_at": "2025-01-15T10:30:00Z"
}
```

**Response**：
```json
{
  "success": true,
  "data": {
    "transcript_id": "67359abc123def456789",
    "session_id": "507f1f77bcf86cd799439011",
    "segments_count": 2,
    "session_updated": true  // Session.participants_snapshot 已更新
  }
}
```

**後端邏輯**：
```python
async def save_transcript(request: TranscriptCreateRequest):
    """儲存逐字稿，並根據參數決定是否更新 Session"""
    # 1. 建立 Transcript
    transcript = Transcript(
        session_id=request.session_id,
        segments=request.segments,
        created_at=request.created_at,
        last_recorded_at=request.last_recorded_at
    )
    await transcript.save()
    
    # 2. 根據參數決定是否更新 Session 的 participants_snapshot
    session_updated = False
    if request.update_session_snapshot:
        session = await Session.get(request.session_id)
        session.participants = list(request.participants_snapshot.keys())
        session.participants_snapshot = request.participants_snapshot
        await session.save()
        session_updated = True
    
    return {
        "transcript_id": str(transcript.id),
        "session_id": transcript.session_id,
        "segments_count": len(transcript.segments),
        "session_updated": session_updated
    }
```

---

#### GET /transcript/{transcript_id}（取得逐字稿）
**用途**：取得單一逐字稿的完整資料，並附加語者名稱

**Query Parameters**：
- `name_mode`: str（顯示模式，預設 "snapshot"）
  - `"snapshot"`：顯示會議當時的名字（使用 Session.participants_snapshot）
  - `"current"`：顯示最新的名字（即時查詢 Weaviate）

**Response**：
```json
{
  "success": true,
  "data": {
    "transcript_id": "67359abc123def456789",
    "session_id": "507f1f77bcf86cd799439011",
    "segments": [
      {
        "speaker_uuid": "550e8400-e29b-41d4-a716-446655440001",
        "speaker_name": "王小明",  // 已經附加 speaker_name
        "speaker_nickname": "小明",
        "content": "大家好，歡迎來到今天的會議。",
        "timestamp": 0.0,
        "confidence": 0.95,
        "duration": 3.5,
        "language": "zh-TW"
      },
      {
        "speaker_uuid": "550e8400-e29b-41d4-a716-446655440002",
        "speaker_name": "李小華",
        "speaker_nickname": "小華",
        "content": "謝謝主持人，很高興參加這次會議。",
        "timestamp": 3.5,
        "confidence": 0.92,
        "duration": 4.2,
        "language": "zh-TW"
      }
    ],
    "name_mode": "snapshot",  // 使用的顯示模式
    "created_at": "2025-01-15T10:00:00Z",
    "last_recorded_at": "2025-01-15T10:30:00Z",
    "last_edited_at": null
  }
}
```

---

#### PUT /transcript/{transcript_id}（更新逐字稿）
**用途**：編輯逐字稿內容（手動修正文字、調整時間戳記等）

**Request Body**：
```json
{
  "segments": [
    {
      "speaker_uuid": "550e8400-e29b-41d4-a716-446655440001",
      "content": "大家好，歡迎來到今天的會議。（已修正）",  // 修改內容
      "timestamp": 0.0,
      "confidence": 0.95,
      "duration": 3.5,
      "language": "zh-TW"
    }
  ]
}
```

**Response**：
```json
{
  "success": true,
  "message": "逐字稿已更新",
  "data": {
    "transcript_id": "67359abc123def456789",
    "last_edited_at": "2025-01-15T14:20:00Z"  // 記錄編輯時間
  }
}
```

---

#### DELETE /transcript/{transcript_id}（刪除逐字稿）
**用途**：刪除逐字稿（注意：不會刪除關聯的 Session）

**Response**：
```json
{
  "success": true,
  "message": "逐字稿已刪除"
}
```

---

### 🔹 Speaker API（更新語者時的快照處理）

#### PUT /speakers/{uuid}（更新語者資訊）
**用途**：更新語者名稱、綽號等資訊

**新增參數**：`update_historical_sessions`（是否更新所有歷史 Session 的快照）

**Request Body**：
```json
{
  "full_name": "王大明",  // 改名
  "nickname": "大明",
  "update_historical_sessions": false  // 預設 false（保留歷史快照）
}
```

**Response**：
```json
{
  "success": true,
  "message": "語者資訊已更新",
  "data": {
    "speaker_uuid": "550e8400-e29b-41d4-a716-446655440001",
    "full_name": "王大明",
    "nickname": "大明",
    "sessions_updated": 0  // 更新的 Session 數量（如果 update_historical_sessions=true）
  }
}
```

**後端邏輯**：
```python
async def update_speaker(speaker_uuid: str, request: SpeakerUpdateRequest):
    """更新語者資訊，並根據參數決定是否更新所有相關 Session"""
    # 1. 更新 Weaviate 中的 Speaker
    weaviate_service.update_speaker(speaker_uuid, {
        "full_name": request.full_name,
        "nickname": request.nickname
    })
    
    # 2. 根據參數決定是否更新所有相關 Session 的快照
    sessions_updated = 0
    if request.update_historical_sessions:
        # 查詢所有包含此語者的 Session
        sessions = await Session.find(
            {"participants": speaker_uuid}
        ).to_list()
        
        for session in sessions:
            # 更新快照中的語者資訊
            if speaker_uuid in session.participants_snapshot:
                session.participants_snapshot[speaker_uuid] = {
                    "full_name": request.full_name,
                    "nickname": request.nickname
                }
                await session.save()
                sessions_updated += 1
    
    return {
        "speaker_uuid": speaker_uuid,
        "full_name": request.full_name,
        "nickname": request.nickname,
        "sessions_updated": sessions_updated
    }
```

---

## 6.3 完整的使用者流程範例

### 情境：使用者錄音並儲存逐字稿

#### **步驟 1：建立場合**
```http
POST /sessions
Content-Type: application/json

{
  "session_type": "meeting",
  "title": "專案討論會議",
  "start_time": "2025-01-15T10:00:00Z"
}

Response:
{
  "success": true,
  "data": {
    "session_id": "507f1f77bcf86cd799439011",  // ✅ MongoDB ObjectId
    "session_type": "meeting",
    "title": "專案討論會議",
    "participants": [],  // 初始為空
    "participants_snapshot": {}  // 初始為空
  }
}
```

#### **步驟 2：開始錄音**
- 前端建立 WebSocket 連線
- 即時辨識語音 + 語者
- **前端暫存字幕資料**（不呼叫 API，只在前端即時顯示）

#### **步驟 3：錄音完成，儲存逐字稿**
```http
POST /transcript
Content-Type: application/json

{
  "session_id": "507f1f77bcf86cd799439011",
  "segments": [
    {
      "speaker_uuid": "550e8400-e29b-41d4-a716-446655440001",
      "content": "大家好，歡迎來到今天的會議。",
      "timestamp": 0.0,
      "confidence": 0.95,
      "duration": 3.5,
      "language": "zh-TW"
    }
  ],
  "participants_snapshot": {
    "550e8400-e29b-41d4-a716-446655440001": {
      "full_name": "王小明",
      "nickname": "小明"
    }
  },
  "update_session_snapshot": true,  // ✅ 第一次儲存時更新 Session
  "created_at": "2025-01-15T10:00:00Z",
  "last_recorded_at": "2025-01-15T10:30:00Z"
}

Response:
{
  "success": true,
  "data": {
    "transcript_id": "67359abc123def456789",
    "session_id": "507f1f77bcf86cd799439011",
    "segments_count": 1,
    "session_updated": true  // ✅ Session.participants_snapshot 已更新
  }
}
```

#### **步驟 4：查看逐字稿**
```http
GET /transcript/67359abc123def456789?name_mode=snapshot

Response:
{
  "success": true,
  "data": {
    "transcript_id": "67359abc123def456789",
    "session_id": "507f1f77bcf86cd799439011",
    "segments": [
      {
        "speaker_uuid": "550e8400-e29b-41d4-a716-446655440001",
        "speaker_name": "王小明",  // ✅ 已經附加 speaker_name
        "speaker_nickname": "小明",
        "content": "大家好，歡迎來到今天的會議。",
        "timestamp": 0.0,
        "confidence": 0.95,
        "duration": 3.5,
        "language": "zh-TW"
      }
    ],
    "name_mode": "snapshot",
    "created_at": "2025-01-15T10:00:00Z",
    "last_recorded_at": "2025-01-15T10:30:00Z"
  }
}
```

---

# 7. 專案狀態與決策

## 7.1 已確認的設計決策（2025-01-14）

### ✅ **Q1: 資料遷移策略**
- **決定**：直接清空重來，不保留舊資料
- **影響**：無需考慮 SpeechLog → Transcript 遷移
- **理由**：全新架構，簡化實作

### ✅ **Q2: 前端相依性**
- **決定**：全部改動，不需向後相容
- **影響**：API 端點可以自由重新設計
- **理由**：大規模重構，追求最佳設計

### ✅ **Q3: speaker_name 儲存策略**
- **決定**：採用「參與者快照」模式，儲存在 **Session** 而非 Transcript
- **設計**：
  - TranscriptSegment 只儲存 `speaker_uuid`（不儲存 speaker_name）
  - **Session** 儲存 `participants_snapshot: Dict[str, Dict[str, str]]`
  - Transcript **不儲存** participants_snapshot（避免冗餘）
  - 支援兩種顯示模式：
    - `snapshot`（預設）：顯示會議當時的名字（查詢 Session.participants_snapshot）
    - `current`：顯示最新的名字（即時查詢 Weaviate）
- **為什麼存在 Session？**
  1. **邏輯正確性**：Session 定義「誰參加會議」，Transcript 只記錄「誰說了什麼」
  2. **資料正規化**：單一資料源（Session），避免 Transcript 重複儲存
  3. **更新便利性**：語者改名 → 只需更新 Session
  4. **符合 1:1 關係**：既然 Session ↔ Transcript 是 1:1，參與者資訊自然屬於 Session
- **優點**：
  1. 減少資料冗餘（單一資料源）
  2. 支援語者改名後的追溯
  3. 效能開銷可忽略（O(1) dict lookup）
  4. 更新方便（只需改 Session，不需改每個 Transcript）
- **效能分析**：
  - 會議列表頁：只讀 Session，極快 ⚡
  - 逐字稿詳情頁：
    - 查詢 Transcript（取得 segments）
    - 查詢 Session（取得 participants_snapshot）
    - snapshot 模式：O(1) mapping，極快 ⚡
    - current 模式：需查詢 Weaviate，但使用批次查詢優化 ✅

### ✅ **Q4: 部署環境**
- **決定**：MongoDB 使用 Docker Compose
- **理由**：
  1. 環境一致性（開發、測試、CI/CD）
  2. 一鍵啟動（docker-compose up）
  3. 版本控制（docker-compose.yml）
  4. 易於團隊協作
- **替代方案對比**：
  - ❌ `mongodb-community`（Homebrew）：環境不一致，不推薦
  - ✅ Docker Compose：推薦用於開發環境
  - ✅ MongoDB Atlas：推薦用於生產環境

### ✅ **Q4: Session ID 型別策略**（2025-01-14 確認）
- **決定**：使用 MongoDB ObjectId 作為主鍵
- **設計**：
  - 不使用自訂 `session_id` 欄位
  - 使用 Beanie 自動生成的 `id` (PydanticObjectId)
  - Transcript.session_id 儲存 ObjectId 字串（如 "507f1f77bcf86cd799439011"）
  - AISummary.session_id 同樣儲存 ObjectId 字串
- **理由**：
  1. 避免刪除 Session 時產生 ID 間隙（如果用自增 "1", "2", "3"）
  2. 使用者不會直接看到 Session ID
  3. 前端也不需要依賴特定 ID 格式
  4. ObjectId 已有良好的索引效能
- **影響**：
  - API 回傳的 session_id 為 ObjectId 字串
  - 前端以此 ObjectId 查詢 Session

### ✅ **Q5: participants_snapshot 更新策略**（2025-01-14 確認）
- **決定**：提供三種更新時機，預設保留歷史快照
- **設計**：
  
  **1️⃣ 自動更新**：Transcript 儲存時（第一次建立時）
  - 首次儲存 Transcript 時，自動將 participants_snapshot 更新到 Session
  - API 參數：`update_session_snapshot=true`（預設）
  
  **2️⃣ 可選更新**：語者改名時
  - 預設：保留歷史快照（不自動更新）
  - 可選：提供 `update_historical_sessions=true` 參數，批次更新所有相關 Session
  
  **3️⃣ 手動更新**：前端觸發
  - 前端在 Session 詳情頁提供「重新整理參與者名單」按鈕
  - 呼叫 `PATCH /sessions/{session_id}/participants` 更新快照為最新資料
  
- **理由**：
  1. 保留歷史：符合「記錄當時情況」的需求
  2. 彈性更新：支援「統一更新所有記錄」的需求
  3. 手動控制：讓使用者自行決定何時更新
- **前端顯示模式**：
  - **快照模式**（預設）：顯示 Session.participants_snapshot（歷史記錄）
  - **即時模式**（可選）：即時查詢 Weaviate 最新語者資料

### ✅ **Q6: Weaviate Session/SpeechLog 遷移策略**（2025-01-14 確認）
- **決定**：立即切換策略（不保留並存期）
- **步驟**：
  
  **Step 0: 標記舊代碼為 DEPRECATED**
  1. 在 `modules/database/database.py` 頂部加上註解：
     ```python
     """
     ⚠️ DEPRECATED - 此檔案為開發參考用，請勿修改
     
     本檔案包含舊版 Weaviate Session/SpeechLog 實作。
     新版本請使用：
     - weaviate_crud.py (Speaker/VoicePrint)
     - mongodb_crud.py (Session/Transcript/AISummary)
     
     保留原因：
     - 提供 Speaker/VoicePrint CRUD 的參考實作
     - 幫助理解舊版資料結構
     
     請勿在新代碼中使用此檔案！
     """
     ```
  
  **Step 1: 建立 init_v3_weaviate.py**
  - 替代 `init_v2_collections.py`
  - 只初始化 Speaker 和 VoicePrint
  - 移除 Session 和 SpeechLog
  
  **Step 2: 更新 main.py**
  - 移除舊的 `ensure_weaviate_collections()` 呼叫（如果使用 init_v2_collections.py）
  - 加入新的 Weaviate 初始化（使用 init_v3_weaviate.py）
  - 加入 MongoDB 初始化（使用 init_mongodb.py）
  
  **Step 3: 建立新的 MongoDB 模組**
  - 建立 Session/Transcript/AISummary models
  - 建立 mongodb_crud.py
  - 建立 mongodb_connection.py
  
  **Step 4: 更新 API**
  - 修改 api/api.py，使用新的 MongoDB CRUD
  - 更新 data_facade.py

- **理由**：
  1. 舊資料不需要保留（已確認清空重來）
  2. 避免並存期的複雜性
  3. 加速開發進度
- **風險管理**：
  - 保留 database.py 作為參考（標記 DEPRECATED）
  - 先完成 MongoDB 實作，再修改 API
  - 測試完成後才刪除舊代碼

## 7.2 待執行的工作

### 第一優先級（基礎設施）✅ 已完成
1. ✅ 更新 `docker-compose.yml`（加入 MongoDB）
2. ✅ 安裝 MongoDB 相關套件（pymongo, motor, beanie）

### 第二優先級（底層實作）

**⚠️ 實作前準備**：
- **Step 0**: 標記 `database.py` 為 DEPRECATED（加上註解說明）
- **Step 0**: 建立 `init_v3_weaviate.py`（只初始化 Speaker/VoicePrint）

**正式實作**：
3. 📝 創建 `modules/database/init_mongodb.py`（MongoDB 初始化）
4. 📝 創建 `mongodb_connection.py`（連線管理 + Beanie 初始化）
5. 📝 創建 `modules/database/models/session.py`（Session 模型）
6. 📝 創建 `modules/database/models/transcript.py`（Transcript 模型）
7. 📝 創建 `modules/database/models/ai_summary.py`（AISummary 模型）
8. 📝 創建 `mongodb_crud.py`（Transcript, Session, AISummary CRUD）
9. 📝 更新 `main.py`（使用 init_v3_weaviate.py + init_mongodb.py）

### 第三優先級（架構重構）
9. 📝 創建 `weaviate_connection.py`（從 database.py 拆分）
10. 📝 創建 `weaviate_crud.py`（從 database.py 拆分）
11. 📝 創建 `database_interface.py`（中間層）
12. 📝 創建 `integration_service.py`（整合邏輯）

### 第四優先級（外層接口）
13. 📝 更新 `services/data_facade.py`（加入 Transcript 相關方法）
14. 📝 更新 `api/api.py`（新增 Transcript API 端點）

### 第五優先級（測試與文件）
15. 📝 創建測試檔案
16. 📝 更新文件

### 第六優先級（前端協作優化）
17. 📝 改進會議列表顯示方式
    - **問題**：目前前端顯示會議列表時，Session 沒有語者名稱，每次都需要查詢
    - **解決方案**：利用 `participants_snapshot` 讓前端直接取得參與者名單
    - **API 更新**：GET /sessions 回傳時包含 participants_snapshot
    - **前端更新**：會議列表直接顯示參與者名稱（來自快照），無需額外查詢
    - **效能提升**：減少前端對 Weaviate 的查詢次數，加快頁面載入速度
    - **實作細節**：
      ```python
      # 後端 API 回傳格式
      {
        "session_id": "1",
        "title": "專案討論",
        "start_time": "2024-11-14T10:00:00Z",
        "participants": ["uuid1", "uuid2"],
        "participants_snapshot": {
          "uuid1": {"full_name": "王小明", "nickname": "小明"},
          "uuid2": {"full_name": "李小華", "nickname": "小華"}
        }
      }
      ```

---

# 9. 檔案結構規劃

## 9.1 最終檔案結構
```
Unsaycret-API/
├── modules/
│   └── database/
│       ├── __init__.py
│       ├── database.py  ⚠️ DEPRECATED（保留以向後相容）
│       │
│       ├── models/
│       │   ├── __init__.py  ✏️ 更新導出
│       │   ├── session.py  📝 新建
│       │   ├── transcript.py  📝 新建
│       │   ├── speechlog.py  ❌ 已刪除（舊架構）
│       │   └── ai_summary.py  📝 新建
│       │
│       ├── weaviate_connection.py  📝 新建（連線管理）
│       ├── weaviate_crud.py  📝 新建（CRUD 操作）
│       │
│       ├── mongodb_connection.py  📝 新建（連線管理）
│       ├── mongodb_crud.py  ✏️ 重構（CRUD 操作）
│       ├── mongo_service.py  ❌ 刪除（已被 mongodb_connection 取代）
│       │
│       ├── database_interface.py  📝 新建（中間層）
│       └── integration_service.py  ✏️ 更新（整合邏輯）
│
├── services/
│   └── data_facade.py  📝 新建（外層接口）
│
├── api/
│   └── api.py  ✏️ 更新（使用 data_facade）
│
├── main.py  ✏️ 更新（初始化邏輯）
│
├── tests/  📁 新建測試目錄
│   ├── test_transcript_crud.py  📝 新建
│   ├── test_three_layer_architecture.py  📝 新建
│   ├── test_mongodb_integration.py  ✏️ 更新
│   └── test_performance.py  📝 新建
│
└── docs/  📁 文件目錄
    ├── DATABASE_MIGRATION_GUIDE.md  ✏️ 更新
    ├── QUICKSTART.md  ✏️ 更新
    ├── MONGODB_REFACTOR_SUMMARY.md  ✏️ 更新
    └── ARCHITECTURE.md  📝 新建
```

### 圖例說明：
- ✅ 保持不變
- ✏️ 需要更新
- 📝 需要新建
- ❌ 需要刪除
- ⚠️ 標記為 DEPRECATED

---

# 10. 資料流設計

## 10.1 Session 建立流程
```
1. 前端選擇場合/建立新場合
   ↓
2. 前端呼叫 POST /session（建立 Session，但不建立 Transcript）
   ↓
3. api.py → data_facade.create_session()
   ↓
4. data_facade → database_interface.create_session()
   ↓
5. database_interface → mongodb_crud.create_session()
   ↓
6. 返回 {session_id}
```

## 10.2 即時錄音與顯示流程（不儲存資料庫）
```
1. 前端開始錄音，建立 WebSocket 連線
   ↓
2. 音訊資料即時傳送到後端
   ↓
3. 後端處理：語音辨識 + 語者識別
   ↓
4. 後端透過 WebSocket 即時回傳字幕資料給前端：
   {
     "speaker_uuid": "...",
     "speaker_name": "...",  // 從 Weaviate 即時查詢
     "content": "...",
     "timestamp": 0.0
   }
   ↓
5. 前端即時顯示字幕（暫存在前端 state，不呼叫資料庫 API）
   ↓
6. 使用者繼續錄音，重複步驟 2-5
   ↓
7. 使用者按「停止錄音」
   ↓
8. 前端顯示「儲存」按鈕
```

## 10.3 儲存 Transcript 流程（錄完後才儲存）

### 📤 **前端準備資料**
```typescript
// 前端暫存的字幕資料結構（TypeScript）
interface SubtitleItem {
  id: string;
  speaker_uuid: string;
  speaker_name: string;  // 即時顯示用，不傳送給後端
  content: string;
  timestamp: number;  // 秒數
  confidence?: number;
  duration?: number;
  language?: string;
}

// 使用者按「儲存」時，前端組裝 JSON
const saveTranscript = async () => {
  // 建立參與者快照（收集所有不重複的語者）
  const participantsMap: Record<string, { full_name: string; nickname: string }> = {};
  subtitles.forEach(sub => {
    if (!participantsMap[sub.speaker_uuid]) {
      participantsMap[sub.speaker_uuid] = {
        full_name: sub.speaker_name,
        nickname: ""  // 如果前端有 nickname 就填入，否則留空
      };
    }
  });
  
  const transcriptData = {
    session_id: selectedSession.uuid,  // 當前場合的 UUID
    segments: subtitles.map(sub => ({
      speaker_uuid: sub.speaker_uuid,
      // ⚠️ 不傳送 speaker_name
      content: sub.content,
      timestamp: sub.timestamp,
      confidence: sub.confidence,
      duration: sub.duration,
      language: sub.language || "zh-TW"
    })),
    participants_snapshot: participantsMap,  // 新增：參與者快照
    created_at: recordingStartTime.toISOString(),
    last_recorded_at: recordingEndTime.toISOString()
  };
  
  // 呼叫後端 API
  const response = await fetch('/transcript', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transcriptData)
  });
  
  const result = await response.json();
  console.log('儲存成功:', result.transcript_id);
};
```

### 🔄 **後端處理流程**
```
1. 使用者按「儲存文字稿」按鈕
   ↓
2. 前端收集所有暫存的字幕資料（subtitles state）
   ↓
3. 前端組裝 JSON（包含 session_id, segments, participants_snapshot, 時間戳記）
   ↓
4. 前端呼叫 POST /transcript（傳送完整 JSON）
   ↓
5. api.py 接收請求：
   - 使用 Pydantic 自動驗證 JSON 格式
   - TranscriptSegment 自動進行型別檢查
   ↓
6. api.py → data_facade.create_transcript()
   ↓
7. data_facade → database_interface.create_transcript()
   ↓
8. database_interface 執行兩個操作：
   
   📝 操作一：儲存 Transcript（只含 segments）
   - mongodb_crud.create_transcript()
   - 寫入：session_id, segments, created_at, last_recorded_at
   - ⚠️ 不儲存 participants_snapshot（避免冗餘）
   
   📝 操作二：更新 Session（加入 participants_snapshot）
   - mongodb_crud.update_session()
   - 更新 Session.participants_snapshot 欄位
   - 如果 Session 已有 participants_snapshot，可選擇：
     - 保留原快照（預設）：維持「當時」的歷史記錄
     - 更新為最新：使用最新的 speaker_name
   ↓
9. MongoDB 返回插入的 transcript_id
   ↓
11. 後端返回 JSON：
    {
      "success": true,
      "transcript_id": "67359abc123def456789",
      "message": "文字稿已成功儲存"
    }
   ↓
12. 前端顯示「儲存成功」提示
   ↓
13. 前端可選：清空暫存的字幕資料或保留以便繼續編輯
```

### 📊 **實際 JSON 範例**

**前端傳送的 JSON（POST /transcript）**：
```json
{
  "session_id": "507f1f77bcf86cd799439011",
  "segments": [
    {
      "speaker_uuid": "123e4567-e89b-12d3-a456-426614174000",
      "content": "大家好，歡迎來到今天的會議。",
      "timestamp": 0.0,
      "confidence": 0.95,
      "duration": 3.5,
      "language": "zh-TW"
    },
    {
      "speaker_uuid": "123e4567-e89b-12d3-a456-426614174001",
      "content": "謝謝主持人，很高興參加這次會議。",
      "timestamp": 3.5,
      "confidence": 0.92,
      "duration": 4.2,
      "language": "zh-TW"
    },
    {
      "speaker_uuid": "123e4567-e89b-12d3-a456-426614174000",
      "content": "今天我們要討論的議題有三個。",
      "timestamp": 7.7,
      "confidence": 0.93,
      "duration": 2.8,
      "language": "zh-TW"
    }
  ],
  "participants_snapshot": {
    "123e4567-e89b-12d3-a456-426614174000": {
      "full_name": "王小明",
      "nickname": "小明"
    },
    "123e4567-e89b-12d3-a456-426614174001": {
      "full_name": "李小華",
      "nickname": "小華"
    }
  },
  "created_at": "2024-11-14T10:00:00Z",
  "last_recorded_at": "2024-11-14T10:30:15Z"
}
```

**⚠️ 注意**：
- `participants_snapshot` 會被後端儲存到 **Session** 而非 Transcript
- Transcript 只儲存 segments（每個 segment 只含 speaker_uuid）
- 這樣設計是為了資料正規化，避免冗餘

**後端返回的 JSON**：
```json
{
  "success": true,
  "data": {
    "transcript_id": "67359abc123def456789",
    "session_id": "507f1f77bcf86cd799439011",
    "segment_count": 3,
    "total_duration": 10.5,
    "created_at": "2024-11-14T10:00:00Z",
    "last_recorded_at": "2024-11-14T10:30:15Z"
  },
  "message": "文字稿已成功儲存"
}
```

## 10.4 編輯與更新 Transcript 流程
```
1. 使用者在前端手動編輯某段文字
   ↓
2. 前端暫存編輯內容（不立即儲存）
   ↓
3. 使用者按「儲存修改」
   ↓
4. 前端呼叫 PUT /transcript/{id}
   ↓
5. api.py → data_facade.update_transcript()
   ↓
6. database_interface → mongodb_crud.update_transcript()
   ↓
7. 更新 segments 陣列內容
   ↓
8. 自動更新 last_edited_at 時間戳記
   ↓
9. 返回更新結果
```

## 10.5 查詢 Transcript 並附加 Speaker 資料流程

### **情境 A：會議列表頁（不讀取 Transcript）**
```
1. 前端呼叫 GET /sessions
   ↓
2. api.py → data_facade.list_sessions()
   ↓
3. data_facade → mongodb_crud.list_sessions()
   ↓
4. 返回 Session 列表（只包含基本資訊）
   {
     "sessions": [
       {
         "session_id": "1",
         "title": "專案討論",
         "start_time": "2024-11-14T10:00:00Z",
         "participants": ["uuid1", "uuid2"]
       }
     ]
   }
   ↓
5. ✅ 效能優化：不讀取 Transcript，極快
```

### **情境 B：點開會議，讀取逐字稿（預設顯示當時名字）**
```
1. 前端呼叫 GET /transcript/{id} 或 GET /transcript/{id}?name_mode=snapshot
   ↓
2. api.py → data_facade.get_transcript()
   ↓
3. data_facade → database_interface.get_transcript_with_speakers(name_mode="snapshot")
   ↓
4. database_interface → mongodb_crud.get_transcript_by_id()
   ↓ (取得 Transcript 物件)
5. 使用 participants_snapshot 進行 mapping
   ↓ (O(1) dict lookup，極快)
6. 為每個 segment 附加 speaker_name, speaker_nickname
   ↓
7. 返回完整的資料結構
   {
     "transcript_id": "...",
     "segments": [
       {
         "speaker_name": "王小明",  // 會議當時的名字
         "content": "...",
         ...
       }
     ],
     "name_mode": "snapshot"
   }
```

### **情境 C：顯示最新名字（語者已改名）**
```
1. 前端呼叫 GET /transcript/{id}?name_mode=current
   ↓
2. api.py → data_facade.get_transcript()
   ↓
3. data_facade → database_interface.get_transcript_with_speakers(name_mode="current")
   ↓
4. database_interface → mongodb_crud.get_transcript_by_id()
   ↓ (取得 Transcript 物件)
5. database_interface → integration_service.enrich_transcript_with_speakers()
   ↓
6. integration_service 收集所有不重複的 speaker_uuid
   ↓
7. 批次查詢 → weaviate_crud.get_speaker() × N
   ↓
8. 建立 speaker_map（最新名字）
   ↓
9. 為每個 segment 附加最新的 speaker_name, speaker_nickname
   ↓
10. 返回完整的資料結構
   {
     "transcript_id": "...",
     "segments": [
       {
         "speaker_name": "王大明",  // 最新的名字（已改名）
         "content": "...",
         ...
       }
     ],
     "name_mode": "current"
   }
```

## 10.6 語者辨識與聲紋更新流程
```
1. 錄音系統取得新的 embedding
   ↓
2. data_facade.find_or_create_speaker(embedding, threshold)
   ↓
3. database_interface.find_similar_voiceprints()
   ↓
4. weaviate_crud.find_similar_voiceprints()
   ↓
5. 判斷：
   - 距離 < threshold → 使用現有 Speaker
   - 距離 >= threshold → 建立新 Speaker
   ↓
6. 如果是現有 Speaker：
   database_interface.update_voiceprint()
   ↓
7. 返回 speaker_uuid
```

---

# 11. 實作步驟

## 11.1 建議的實作順序

### 第零階段：遷移前準備（半天）⚠️ **必須先完成**

#### Step 0.1: 標記 database.py 為 DEPRECATED

在 `modules/database/database.py` 檔案頂部加上以下註解：

```python
"""
⚠️ DEPRECATED - 此檔案為開發參考用，請勿修改

本檔案包含舊版 Weaviate Session/SpeechLog 實作。

新版本請使用：
- weaviate_crud.py (Speaker/VoicePrint)
- mongodb_crud.py (Session/Transcript/AISummary)

保留原因：
- 提供 Speaker/VoicePrint CRUD 的參考實作
- 幫助理解舊版資料結構

請勿在新代碼中使用此檔案！

計劃：
- 新版本穩定後，此檔案將被刪除
- 預計保留 2-4 週作為參考
"""
```

#### Step 0.2: 建立 init_v3_weaviate.py（替代 init_v2_collections.py）

**目的**：只初始化 Speaker 和 VoicePrint，移除 Session 和 SpeechLog

**位置**：`modules/database/init_v3_weaviate.py`

**完整實作範例**：

```python
"""
Weaviate V3 Collection 初始化

⚠️ 重要變更（2025-01-14）：
- 只初始化 Speaker 和 VoicePrint
- 移除 Session 和 SpeechLog（已遷移到 MongoDB）

替代檔案：
- 舊版：init_v2_collections.py（包含 Session/SpeechLog）
- 新版：init_v3_weaviate.py（只有 Speaker/VoicePrint）
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType, ReferenceProperty
import logging

logger = logging.getLogger(__name__)


def init_v3_collections(client: weaviate.Client) -> bool:
    """
    初始化 Weaviate V3 Collections（只包含 Speaker 和 VoicePrint）
    
    Args:
        client: Weaviate 客戶端
    
    Returns:
        bool: 初始化是否成功
    
    注意：
    - Session 和 SpeechLog 已遷移到 MongoDB
    - 此函式只負責 Speaker 和 VoicePrint
    """
    try:
        # 取得所有現有 collections
        existing_collections = set()
        try:
            schema = client.schema.get()
            existing_collections = {cls["class"] for cls in schema.get("classes", [])}
            logger.info(f"現有 collections: {existing_collections}")
        except Exception as e:
            logger.warning(f"無法取得現有 schema: {e}")
        
        # ===== Speaker Collection =====
        if "Speaker" not in existing_collections:
            logger.info("建立 Speaker collection...")
            client.collections.create(
                name="Speaker",
                vectorizer_config=Configure.Vectorizer.none(),  # 不使用向量化器
                properties=[
                    Property(name="speaker_id", data_type=DataType.INT),
                    Property(name="full_name", data_type=DataType.TEXT),
                    Property(name="nickname", data_type=DataType.TEXT),
                    Property(name="gender", data_type=DataType.TEXT),
                    Property(name="created_at", data_type=DataType.DATE),
                    Property(name="last_active_at", data_type=DataType.DATE),
                    Property(name="meet_count", data_type=DataType.INT),
                    Property(name="meet_days", data_type=DataType.INT),
                    Property(name="voiceprint_ids", data_type=DataType.TEXT_ARRAY),
                    Property(name="first_audio", data_type=DataType.TEXT),
                ]
            )
            logger.info("✅ Speaker collection 建立完成")
        else:
            logger.info("✅ Speaker collection 已存在")
        
        # ===== VoicePrint Collection =====
        if "VoicePrint" not in existing_collections:
            logger.info("建立 VoicePrint collection...")
            client.collections.create(
                name="VoicePrint",
                vectorizer_config=Configure.Vectorizer.none(),  # 手動提供向量
                properties=[
                    Property(name="created_at", data_type=DataType.DATE),
                    Property(name="updated_at", data_type=DataType.DATE),
                    Property(name="update_count", data_type=DataType.INT),
                    Property(name="sample_count", data_type=DataType.INT),
                    Property(name="quality_score", data_type=DataType.NUMBER),
                    Property(name="speaker_name", data_type=DataType.TEXT),
                ],
                references=[
                    ReferenceProperty(name="speaker", target_collection="Speaker")
                ]
            )
            logger.info("✅ VoicePrint collection 建立完成")
        else:
            logger.info("✅ VoicePrint collection 已存在")
        
        # ===== 檢查是否有舊的 Session/SpeechLog =====
        if "Session" in existing_collections:
            logger.warning("⚠️ 發現舊的 Session collection，建議手動刪除或清空資料")
        if "SpeechLog" in existing_collections:
            logger.warning("⚠️ 發現舊的 SpeechLog collection，建議手動刪除或清空資料")
        
        logger.info("✅ Weaviate V3 初始化完成（Speaker + VoicePrint）")
        return True
        
    except Exception as e:
        logger.error(f"❌ Weaviate V3 初始化失敗: {e}", exc_info=True)
        return False


def ensure_weaviate_v3_collections() -> bool:
    """
    確保 Weaviate V3 collections 存在
    
    注意：此函式會在 main.py 的 initialize_system() 中被呼叫
    
    Returns:
        bool: 初始化是否成功
    """
    try:
        from utils.env_config import WEAVIATE_URL
        
        # 建立 Weaviate 客戶端
        client = weaviate.connect_to_local(
            host=WEAVIATE_URL.replace("http://", "").replace(":8080", ""),
            port=8080,
        )
        
        # 初始化 collections
        success = init_v3_collections(client)
        
        # 關閉連線
        client.close()
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 無法連線到 Weaviate: {e}")
        return False
```

**使用方式（在 main.py）**：

```python
# main.py
from modules.database.init_v3_weaviate import ensure_weaviate_v3_collections
from modules.database.init_mongodb import initialize_mongodb

def initialize_system() -> bool:
    """系統初始化"""
    try:
        # 1. 初始化 Weaviate（只有 Speaker + VoicePrint）
        logger.info("初始化 Weaviate...")
        if not ensure_weaviate_v3_collections():
            logger.error("Weaviate 初始化失敗")
            return False
        
        # 2. 初始化 MongoDB（Session + Transcript + AISummary）
        logger.info("初始化 MongoDB...")
        if not asyncio.run(initialize_mongodb()):
            logger.error("MongoDB 初始化失敗")
            return False
        
        logger.info("✅ 系統初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 系統初始化失敗: {e}", exc_info=True)
        return False
```

---

### 第一階段：MongoDB 模型與底層（2-3 天）
1. 創建 `modules/database/init_mongodb.py`（MongoDB 初始化函式）
2. 創建 `modules/database/models/session.py`（Session 模型）
3. 創建 `modules/database/models/transcript.py`（Transcript 模型）
4. 創建 `modules/database/models/ai_summary.py`（AISummary 模型）
5. 創建 `mongodb_connection.py`（連線管理 + Beanie 初始化）
6. 創建 `mongodb_crud.py`（完整的 CRUD 操作）
   - ⚠️ **重要設計**：
     - Transcript CRUD 是「會議級儲存」，不是「逐句新增」
     - `create_transcript(session_id, segments, participants_snapshot)` - 接收完整 JSON
     - 後端需將 `participants_snapshot` 儲存到 **Session** 而非 Transcript
     - `update_transcript(transcript_id, segments)` - 更新整個 segments 陣列
6. 測試 MongoDB 底層（test_transcript_crud.py, test_session_crud.py）

### 第二階段：底層重構（1-2 天）
5. 創建 `weaviate_connection.py`（從 database.py 拆分）
6. 創建 `weaviate_crud.py`（從 database.py 拆分）
7. 標記 `database.py` 為 DEPRECATED
8. 測試 Weaviate 底層

### 第三階段：中間層建立（1 天）
9. 創建 `database_interface.py`
10. 更新 `integration_service.py`（新增 Transcript 整合）
11. 測試中間層

### 第四階段：外層接口（1 天）
12. 創建 `services/data_facade.py`
13. 測試外層接口
14. 測試三層架構完整流程（test_three_layer_architecture.py）

### 第五階段：API 層更新（1-2 天）
15. 更新 `main.py` 初始化邏輯
16. 更新 `api/api.py`（使用 data_facade）
17. 更新 `modules/VID_manager.py`（如需要）
18. 測試 API 端點

### 第六階段：文件與測試（1 天）
19. 更新所有文件
20. 創建效能測試
21. 完整的整合測試

### 第七階段：前端協作優化（1 天）
22. 改進會議列表顯示方式
    - 更新 GET /sessions API，回傳時包含 participants_snapshot
    - 前端利用 participants_snapshot 直接顯示參與者名單
    - 減少對 Weaviate 的查詢次數

### 第八階段：清理與收尾（半天）
23. 清理 Weaviate 中的舊資料（Session/SpeechLog collection）
24. 清理 MongoDB 中的舊資料（如果有）
25. 程式碼審查與優化
26. 最終文件更新

**總計預估時間：7-9 天**

**⚠️ 注意**：
- 本專案已決定「不保留舊資料」，無需實作資料遷移
- API 端點可以自由重新設計，無需向後相容
- 優先實作核心功能，測試可以後補

---

# 12. 測試計劃

## 12.1 單元測試

### MongoDB 層測試
```python
# test_transcript_crud.py
async def test_create_transcript()
async def test_add_segment()
async def test_update_segment_content()
async def test_get_full_transcript()
async def test_last_recorded_at_update()
async def test_last_edited_at_update()
```

### Weaviate 層測試
```python
# test_weaviate_crud.py
def test_create_speaker()
def test_create_voiceprint()
def test_find_similar_voiceprints()
def test_speaker_voiceprint_relationship()
```

## 12.2 整合測試

### 中間層測試
```python
# test_database_interface.py
async def test_get_session_with_participants()
async def test_get_transcript_with_speakers()
async def test_cross_database_query()
```

### 外層測試
```python
# test_data_facade.py
async def test_create_session_with_transcript()
async def test_add_speech_to_session()
async def test_find_or_create_speaker()
```

## 12.3 效能測試

### 對比測試
```python
# test_performance.py
async def test_speechlog_vs_transcript_query_performance():
    """
    對比：
    - 舊：查詢 1000 筆 SpeechLog
    - 新：查詢 1 筆 Transcript（含 1000 segments）
    """

async def test_speaker_join_performance():
    """
    測試批次查詢 Speaker 的效能優化
    """
```

---

# 13. 注意事項與最佳實踐

## 13.1 程式碼規範
- ✅ 所有新檔案都要有完整的中文註解
- ✅ 每個函式都要有 docstring（說明參數、返回值、功能）
- ✅ 類別要有類別級別的註解（說明用途、關聯關係）
- ✅ 複雜邏輯要有行內註解

## 13.2 錯誤處理
- ✅ 所有資料庫操作都要有 try-except
- ✅ 使用 logger 記錄錯誤與關鍵操作
- ✅ 返回明確的錯誤訊息

## 13.3 效能優化
- ✅ 批次查詢 Speaker 而非逐一查詢
- ✅ 使用 async/await 處理 MongoDB 操作
- ✅ Transcript segments 按時間順序排列以加速查詢

## 13.4 資料一致性
- ✅ Session 與 Transcript 保持 1:1 關係
- ✅ speaker_uuid 必須對應到 Weaviate 中存在的 Speaker
- ✅ 時間戳記統一使用 UTC 時區

## 13.5 專案特定決策
- ✅ 不需向後相容（已確認）
- ✅ 不需資料遷移（直接清空重來）
- ✅ database.py 將被標記為 DEPRECATED（保留一段時間後刪除）

---

# 14. docker-compose 配置

## 14.1 完整配置（Weaviate + MongoDB）

**部署決策**：
- ✅ 使用 Docker Compose 管理兩個資料庫
- ✅ 開發環境：本機 Docker
- ✅ 生產環境：建議 MongoDB Atlas（雲端）

```yaml
version: '3.8'

services:
  # Weaviate 向量資料庫
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8080'
    - --scheme
    - http
    image: cr.weaviate.io/semitechnologies/weaviate:1.30.0
    ports:
    - 8080:8080
    - 50051:50051
    volumes:
    - weaviate_data:/var/lib/weaviate
    - weaviate_backups:/var/lib/weaviate/backups
    restart: unless-stopped
    healthcheck:          
      test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:8080/v1/.well-known/ready || exit 1"]
      interval: 30s       
      timeout: 5s
      retries: 3
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: 'backup-filesystem'
      BACKUP_FILESYSTEM_PATH: '/var/lib/weaviate/backups'
      CLUSTER_HOSTNAME: 'node1'

  # Weaviate Console (可選，方便查看資料)
  weaviate_console:
    image: semitechnologies/weaviate-console:latest
    platform: linux/amd64
    ports:
      - "8081:80"
    depends_on:
      - weaviate
    environment:
      - WEAVIATE_URL=http://weaviate:8080

  # MongoDB 文檔資料庫
  mongodb:
    image: mongodb/mongodb-community-server:latest
    restart: unless-stopped
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: root
      MONGO_INITDB_ROOT_PASSWORD: admin123
    volumes:
      - mongodb_data:/data/db
      - mongodb_config:/data/configdb
    healthcheck:
      # ⚠️ 注意：需要確認 MongoDB 映像檔包含 mongosh
      # MongoDB 6+ 官方映像檔通常包含 mongosh
      # 如果沒有 mongosh，可使用替代方案（見下方說明）
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Mongo Express (可選，方便查看資料)
  mongo_express:
    image: mongo-express:latest
    restart: unless-stopped
    ports:
      - "8082:8081"
    environment:
      ME_CONFIG_MONGODB_ADMINUSERNAME: root
      ME_CONFIG_MONGODB_ADMINPASSWORD: admin123
      ME_CONFIG_MONGODB_URL: mongodb://root:admin123@mongodb:27017/
      ME_CONFIG_BASICAUTH: false
    depends_on:
      - mongodb

volumes:
  weaviate_data:
  weaviate_backups:
  mongodb_data:
  mongodb_config:
```

**服務說明**：
- `weaviate`: 主要的向量資料庫（必須）
- `weaviate_console`: Web UI 查看 Weaviate 資料（可選，http://localhost:8081）
- `mongodb`: 主要的文檔資料庫（必須）
- `mongo_express`: Web UI 查看 MongoDB 資料（可選，http://localhost:8082）

**啟動指令**：
```bash
# 啟動所有服務
docker-compose up -d

# 只啟動必要服務（不含 UI）
docker-compose up -d weaviate mongodb

# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f mongodb

# 停止所有服務
docker-compose down

# 停止並刪除資料（⚠️ 危險操作）
docker-compose down -v
```

---

## 14.2 MongoDB Healthcheck 替代方案

### ⚠️ 為什麼需要替代方案？

`mongosh` 是 MongoDB 6+ 的新版 Shell，取代了舊的 `mongo` 命令。但有些情況下可能需要替代方案：
- 使用舊版 MongoDB 映像檔（不含 mongosh）
- 容器啟動時 mongosh 尚未就緒
- 想要更輕量的健康檢查方式

### 📋 Healthcheck 方案對比

| 方案 | 優點 | 缺點 | 適用場景 |
|------|------|------|---------|
| **mongosh** | 真正檢查 MongoDB 服務 | 需要 MongoDB 6+ | ✅ 推薦（預設） |
| **mongo** | 舊版 MongoDB 都有 | MongoDB 6+ 已移除 | MongoDB 5 以下 |
| **TCP 端口檢查** | 不依賴 MongoDB 工具 | 只檢查端口，不檢查服務 | 簡單場景 |
| **無 healthcheck** | 最簡單 | 無法檢測服務狀態 | 不推薦 |

### ✅ 方案 1：使用 mongosh（推薦，預設）

```yaml
mongodb:
  image: mongo:6  # 或 mongodb/mongodb-community-server:latest
  healthcheck:
    test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 40s
```

**測試方式**：
```bash
# 測試 healthcheck 命令
docker exec mongodb mongosh --eval "db.adminCommand('ping')"

# 查看健康狀態
docker inspect mongodb --format='{{.State.Health.Status}}'
```

---

### 方案 2：使用 mongo（舊版 MongoDB）

**適用**：MongoDB 5 以下

```yaml
mongodb:
  image: mongo:5
  healthcheck:
    test: ["CMD", "mongo", "--eval", "db.adminCommand('ping')"]
    interval: 30s
    timeout: 10s
    retries: 3
```

---

### 方案 3：使用 TCP 端口檢查（最簡單）

**不依賴 MongoDB 工具**，只檢查 27017 端口是否開放

```yaml
mongodb:
  image: mongo:6
  healthcheck:
    test: ["CMD-SHELL", "nc -z localhost 27017 || exit 1"]
    interval: 30s
    timeout: 10s
    retries: 3
```

**缺點**：只檢查端口，不檢查 MongoDB 是否真正可用

**測試方式**：
```bash
docker exec mongodb nc -z localhost 27017 && echo "✅ 端口開放" || echo "❌ 端口關閉"
```

---

### 方案 4：不設 healthcheck（最保險但不推薦）

```yaml
mongodb:
  image: mongo:6
  restart: unless-stopped
  # 移除 healthcheck
```

**缺點**：
- 無法知道 MongoDB 是否準備好
- 依賴 Docker 的 restart policy
- 可能導致應用啟動時連不到資料庫

---

### 🔍 如何檢查你的 MongoDB 映像檔支援哪些工具？

```bash
# 1. 啟動容器
docker-compose up -d mongodb

# 2. 檢查可用工具
docker exec mongodb bash -c "which mongosh mongo nc curl"

# 3. 測試 healthcheck 命令
docker exec mongodb mongosh --eval "db.adminCommand('ping')"

# 4. 查看健康狀態
docker inspect mongodb --format='{{.State.Health.Status}}'
```

---

### 📌 推薦設定（根據映像檔版本）

| MongoDB 映像檔 | 推薦 Healthcheck |
|---------------|-----------------|
| `mongo:6` 或 `mongo:latest` | ✅ `mongosh` |
| `mongodb/mongodb-community-server:latest` | ✅ `mongosh` |
| `mongo:5` | ✅ `mongo`（舊版命令）|
| `mongo:4` 或更舊 | ✅ `mongo`（舊版命令）|

---

# 15. 環境變數配置

## 15.1 .env 檔案
```bash
# Weaviate 設定
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_SCHEME=http
WEAVIATE_MAX_RETRIES=3
WEAVIATE_CONNECTION_TIMEOUT=30

# MongoDB 設定
MONGO_URL=mongodb://root:admin123@localhost:27017
MONGO_DB_NAME=unsaycret
MONGO_MAX_POOL_SIZE=10
MONGO_MIN_POOL_SIZE=1

# 其他設定
DEFAULT_SPEAKER_NAME=未命名
DEFAULT_FULL_NAME_PREFIX=n
```

## 15.2 Docker 環境變數調整

如果在 Docker 容器中運行 API，需要調整網路配置：

```bash
# Docker 內部網路（容器間通訊）
WEAVIATE_HOST=weaviate  # 不是 localhost
MONGO_URL=mongodb://root:admin123@mongodb:27017  # 不是 localhost
```

---

# 15. 初始化實作範例

## 15.1 init_mongodb.py 完整實作

**位置**：`modules/database/init_mongodb.py`

```python
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

if not asyncio.run(initialize_mongodb()):
    logger.error("MongoDB 初始化失敗")
    sys.exit(1)
```

注意：
- MongoDB 是 schema-less，collections 會在第一次插入資料時自動建立
- 但 Beanie 需要先初始化，才能使用 Document 模型
- 索引會在初始化時自動建立（根據 Document.Settings.indexes）
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

# 匯入所有 Document 模型
from modules.database.models.session import Session
from modules.database.models.transcript import Transcript
from modules.database.models.ai_summary import AISummary

logger = logging.getLogger(__name__)


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
    try:
        # 1. 讀取環境變數
        from utils.env_config import MONGO_URL, MONGO_DB_NAME
        
        logger.info(f"正在連線到 MongoDB: {MONGO_DB_NAME}")
        
        # 2. 建立 MongoDB 客戶端（async）
        client = AsyncIOMotorClient(MONGO_URL)
        
        # 3. 測試連線（ping）
        try:
            await client.admin.command('ping')
            logger.info("✅ MongoDB 連線成功")
        except Exception as e:
            logger.error(f"❌ MongoDB 連線失敗: {e}")
            return False
        
        # 4. 初始化 Beanie（註冊所有 Document 模型）
        database = client[MONGO_DB_NAME]
        await init_beanie(
            database=database,
            document_models=[
                Session,     # 會議資料
                Transcript,  # 逐字稿
                AISummary,   # AI 摘要
            ]
        )
        
        logger.info("✅ Beanie ODM 初始化完成")
        
        # 5. 顯示已註冊的 collections
        collections = await database.list_collection_names()
        logger.info(f"現有 collections: {collections}")
        
        # 6. 檢查索引是否建立
        # Beanie 會自動建立索引，但我們可以驗證
        for collection_name in ["sessions", "transcripts", "ai_summaries"]:
            if collection_name in collections:
                indexes = await database[collection_name].index_information()
                logger.info(f"Collection '{collection_name}' 索引: {list(indexes.keys())}")
        
        logger.info("✅ MongoDB 初始化完成")
        return True
        
    except ImportError as e:
        logger.error(f"❌ 無法匯入模型或環境變數: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB 初始化失敗: {e}", exc_info=True)
        return False


async def test_mongodb_connection() -> bool:
    """
    測試 MongoDB 連線（不初始化 Beanie）
    
    用途：在不需要完整初始化的情況下，快速檢查 MongoDB 是否可用
    
    Returns:
        bool: 連線是否成功
    """
    try:
        from utils.env_config import MONGO_URL
        
        client = AsyncIOMotorClient(MONGO_URL)
        await client.admin.command('ping')
        logger.info("✅ MongoDB ping 成功")
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB ping 失敗: {e}")
        return False


# 如果直接執行此檔案，進行測試
if __name__ == "__main__":
    import asyncio
    
    # 測試連線
    print("測試 MongoDB 連線...")
    if asyncio.run(test_mongodb_connection()):
        print("✅ 連線測試成功")
    else:
        print("❌ 連線測試失敗")
    
    # 完整初始化
    print("\n執行完整初始化...")
    if asyncio.run(initialize_mongodb()):
        print("✅ 初始化成功")
    else:
        print("❌ 初始化失敗")
```

---

## 15.2 main.py 更新範例

**位置**：`main.py`

**更新內容**：加入 MongoDB 初始化，使用 init_v3_weaviate.py

```python
"""
應用程式主入口

系統初始化流程（2025-01-14 更新）：
1. 初始化 Weaviate（只有 Speaker + VoicePrint）
2. 初始化 MongoDB（Session + Transcript + AISummary）
3. 啟動 FastAPI 應用
"""

import asyncio
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

# 匯入初始化函式
from modules.database.init_v3_weaviate import ensure_weaviate_v3_collections
from modules.database.init_mongodb import initialize_mongodb

# 設定 logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unsaycret API", version="2.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    應用程式生命週期管理
    
    startup: 初始化資料庫
    shutdown: 清理資源
    """
    # ===== Startup =====
    logger.info("🚀 應用程式啟動中...")
    
    # 1. 初始化 Weaviate（只有 Speaker + VoicePrint）
    logger.info("正在初始化 Weaviate...")
    if not ensure_weaviate_v3_collections():
        logger.error("❌ Weaviate 初始化失敗")
        raise RuntimeError("Weaviate 初始化失敗")
    logger.info("✅ Weaviate 初始化完成")
    
    # 2. 初始化 MongoDB（Session + Transcript + AISummary）
    logger.info("正在初始化 MongoDB...")
    if not await initialize_mongodb():
        logger.error("❌ MongoDB 初始化失敗")
        raise RuntimeError("MongoDB 初始化失敗")
    logger.info("✅ MongoDB 初始化完成")
    
    logger.info("✅ 所有資料庫初始化完成")
    logger.info("🎉 應用程式啟動成功")
    
    yield
    
    # ===== Shutdown =====
    logger.info("👋 應用程式關閉中...")
    # 清理資源（如需要）
    logger.info("✅ 應用程式已關閉")


app = FastAPI(lifespan=lifespan)


# ===== 健康檢查端點 =====
@app.get("/")
async def root():
    """根端點 - 檢查 API 是否運行"""
    return {
        "status": "ok",
        "message": "Unsaycret API v2.0",
        "databases": {
            "weaviate": "Speaker + VoicePrint",
            "mongodb": "Session + Transcript + AISummary"
        }
    }


@app.get("/health")
async def health_check():
    """健康檢查端點"""
    try:
        # 檢查 Weaviate（可選）
        # weaviate_ok = check_weaviate_connection()
        
        # 檢查 MongoDB（可選）
        # mongodb_ok = await test_mongodb_connection()
        
        return {
            "status": "healthy",
            "weaviate": "ok",
            "mongodb": "ok"
        }
    except Exception as e:
        logger.error(f"健康檢查失敗: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ===== 匯入其他路由 =====
# from api.api import router as api_router
# app.include_router(api_router, prefix="/api/v2")


if __name__ == "__main__":
    import uvicorn
    
    # 執行開發伺服器
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 開發模式下啟用自動重載
        log_level="info"
    )
```

**執行方式**：

```bash
# 開發模式（自動重載）
python main.py

# 或使用 uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 生產模式
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**測試初始化**：

```bash
# 啟動應用後，測試健康檢查
curl http://localhost:8000/

# 預期回應：
# {
#   "status": "ok",
#   "message": "Unsaycret API v2.0",
#   "databases": {
#     "weaviate": "Speaker + VoicePrint",
#     "mongodb": "Session + Transcript + AISummary"
#   }
# }
```

---

## 15.3 初始化流程說明

### MongoDB 初始化過程

**Q: MongoDB 初始化時會做什麼？**

A: 執行 `initialize_mongodb()` 時，會進行以下操作：

1. **建立連線**
   ```python
   client = AsyncIOMotorClient(MONGO_URL)
   # 連線到 mongodb://root:admin123@localhost:27017
   ```

2. **初始化 Beanie ODM**
   ```python
   await init_beanie(
       database=client["unsaycret"],
       document_models=[Session, Transcript, AISummary]
   )
   ```
   - 註冊所有 Document 模型
   - 建立 model → collection 的對應關係
   - **自動建立索引**（根據 `Document.Settings.indexes`）

3. **自動建立索引**
   - Beanie 會讀取 `Document.Settings.indexes`
   - 自動在 MongoDB 中建立對應的索引
   - 例如：`Session.Settings.indexes = ["start_time"]`
     → MongoDB 會建立 `start_time` 索引

4. **Collections 自動建立**
   - MongoDB 是 schema-less（無固定結構）
   - Collections 會在**第一次插入資料時**自動建立
   - 不需要手動 `CREATE TABLE`

### Weaviate 初始化過程

**Q: Weaviate 初始化時會做什麼？**

A: 執行 `ensure_weaviate_v3_collections()` 時，會進行以下操作：

1. **檢查現有 Collections**
   ```python
   schema = client.schema.get()
   existing_collections = {cls["class"] for cls in schema.get("classes", [])}
   ```

2. **建立 Speaker Collection**（如果不存在）
   ```python
   client.collections.create(
       name="Speaker",
       properties=[...]
   )
   ```

3. **建立 VoicePrint Collection**（如果不存在）
   ```python
   client.collections.create(
       name="VoicePrint",
       properties=[...],
       references=[ReferenceProperty(name="speaker", target_collection="Speaker")]
   )
   ```

4. **警告舊資料**
   - 如果發現舊的 Session/SpeechLog collections
   - 提示使用者手動刪除或清空

### 常見問題

**Q1: 初始化失敗怎麼辦？**

A: 檢查以下項目：
```bash
# 1. 確認 Docker Compose 已啟動
docker-compose ps

# 2. 確認 MongoDB 健康狀態
docker-compose logs mongodb

# 3. 確認 Weaviate 健康狀態
docker-compose logs weaviate

# 4. 測試 MongoDB 連線
docker exec -it mongodb mongosh -u root -p admin123 --eval "db.adminCommand('ping')"

# 5. 測試 Weaviate 連線
curl http://localhost:8080/v1/.well-known/ready
```

**Q2: 是否需要手動建立資料庫？**

A: 不需要！
- MongoDB：資料庫和 collections 會自動建立
- Weaviate：collections 由 `init_v3_weaviate.py` 自動建立

**Q3: 如何清空所有資料重新開始？**

A: 清空方式：
```bash
# 方法 1：刪除所有資料（包含 volumes）
docker-compose down -v
docker-compose up -d

# 方法 2：只清空 MongoDB 資料
docker exec -it mongodb mongosh -u root -p admin123 --eval "use unsaycret; db.dropDatabase()"

# 方法 3：只清空 Weaviate 資料
# 使用 Weaviate Console (http://localhost:8081) 手動刪除 collections
```

---

# 16. 總結

## 16.1 架構優勢
✅ **清晰的職責分離**：每層都有明確的職責  
✅ **易於維護**：修改底層不影響外層  
✅ **可測試性高**：每層都可以獨立測試  
✅ **效能優化**：Transcript 會議級儲存 + participants_snapshot 減少查詢次數  
✅ **擴展性好**：可輕鬆新增其他資料庫或服務  
✅ **資料一致性**：語者改名後可追溯歷史記錄

## 16.2 關鍵設計決策
🔄 **SpeechLog → Transcript**：從句子級改為會議級儲存  
🔄 **單層 → 三層**：data_facade → database_interface → CRUD  
🔄 **單一檔案 → 分離**：連線管理 + CRUD 分離  
✨ **參與者快照**：支援「當時名字」與「最新名字」兩種顯示模式  
🚀 **效能優化**：會議列表不讀 Transcript，O(1) dict lookup

## 16.3 與原始問題的對應

| 問題 | 決策 | 理由 |
|------|------|------|
| Q1: 資料遷移 | 直接清空重來 | 簡化實作，全新架構 |
| Q2: 向後相容 | 不需要 | 大規模重構，追求最佳設計 |
| Q3: speaker_name | 參與者快照模式 | 減少冗餘，支援改名追溯，效能佳 |
| Q4: 部署環境 | Docker Compose | 環境一致，易於協作 |

## 16.4 下一步行動

### **✅ 已完成（基礎設施）**
1. ✅ 更新 `docker-compose.yml`（加入 MongoDB 服務，包含 volumes 和 healthcheck）
2. ✅ 在 `requirements-base.txt` 加入 MongoDB 相關套件：
   - pymongo==4.9.1
   - motor==3.5.1
   - beanie==1.27.0
3. ✅ 完整的文件規劃與設計決策確認

### **📝 實作前準備（必須先完成）**
**Step 0: 標記舊代碼與建立新初始化模組**
1. 標記 `modules/database/database.py` 為 DEPRECATED（加上註解說明）
2. 建立 `modules/database/init_v3_weaviate.py`（只初始化 Speaker/VoicePrint，參考第 11 章範例）
3. 建立 `modules/database/init_mongodb.py`（MongoDB 初始化，參考第 15 章範例）

### **📝 第一階段：MongoDB 模型與底層（2-3 天）**
4. 創建 `modules/database/models/session.py`（Session 模型）
   - 使用 MongoDB ObjectId 作為主鍵（Beanie 的 `id` 欄位）
   - 包含 `participants_snapshot: Dict[str, Dict[str, str]]`
5. 創建 `modules/database/models/transcript.py`（Transcript + TranscriptSegment 模型）
   - TranscriptSegment 只儲存 `speaker_uuid`（不儲存 speaker_name）
   - Transcript.session_id 儲存 ObjectId 字串
6. 創建 `modules/database/models/ai_summary.py`（AISummary 模型）
   - session_id 儲存 ObjectId 字串（不是 session_uuid）
7. 創建 `modules/database/mongodb_connection.py`（連線管理 + Beanie 初始化）
8. 創建 `modules/database/mongodb_crud.py`（完整的 CRUD 操作）
   - ⚠️ **特別注意**：
     - `create_transcript()` 接收 `update_session_snapshot` 參數
     - 如果 `update_session_snapshot=True`，更新 Session.participants_snapshot
     - 參考第 6.2 章的 API 邏輯
9. 更新 `main.py`（使用 init_v3_weaviate.py + init_mongodb.py，參考第 15.2 章範例）
10. 測試 MongoDB 連接與 CRUD

### **📝 第二階段：底層重構（1-2 天）**
11. 創建 `weaviate_connection.py`（從 database.py 拆分）
12. 創建 `weaviate_crud.py`（從 database.py 拆分）
13. 測試 Weaviate 底層

### **📝 第三階段：中間層建立（1 天）**
14. 創建 `database_interface.py`
15. 更新 `integration_service.py`（實作 enrich_transcript_with_speakers，支援 name_mode 參數）
16. 測試中間層

### **📝 第四階段：外層接口（1 天）**
17. 更新 `services/data_facade.py`（加入 Transcript 相關方法）
18. 測試外層接口
19. 測試三層架構完整流程

### **📝 第五階段：API 層更新（1-2 天）**
20. 更新 `api/api.py`（實作第 6.2 章的完整 API 端點）
   - POST /sessions
   - GET /sessions（回傳 participants_snapshot）
   - PATCH /sessions/{session_id}/participants（手動更新快照）
   - POST /transcript（支援 update_session_snapshot 參數）
   - GET /transcript/{transcript_id}（支援 name_mode 參數）
   - PUT /transcript/{transcript_id}
   - PUT /speakers/{uuid}（支援 update_historical_sessions 參數）
21. 測試 API 端點

### **📝 第六階段：測試與優化（1 天）**
22. 創建效能測試（對比舊版 SpeechLog 查詢效能）
23. 完整的整合測試
24. 前端協作測試（WebSocket + 儲存流程）

### **📝 第七階段：清理與收尾（半天）**
25. 清理 Weaviate 中的舊資料（Session/SpeechLog collections）
26. 程式碼審查與優化
27. 最終文件更新

**總計預估時間：7-9 天**

---

**文件狀態**：✅ 已根據 2025-01-14 的設計決策完整更新  
**下次更新**：實作過程中如有新發現或調整，隨時更新本文件  
**聯絡方式**：如有任何疑問，請隨時提出

---

**📌 重要提醒**：
- ⚠️ 實作前請先完成「立即執行」的 3 項任務
- 📖 參考第 11 章的實作順序，不要跳過步驟
- 🧪 每個階段完成後都要測試
- 💾 記得定期 commit 程式碼

**文件結束**
