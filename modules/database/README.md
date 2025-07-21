# Database Module (Weaviate V2)

**版本**：v0.4.0  
**作者**：CYouuu  
**最後更新者**：CYouuu  
**最後更新**：2025-07-21

⚠️ **重要變更** ⚠️  
本版本已升級為 Weaviate V2 資料庫結構，與 V1 版本不相容！

## 🚀 V2 版本重大更新

### Speaker 集合新增欄位
- `speaker_id` (INT): 從 1 開始遞增的序號 ID
- `full_name`: 主要名稱
- `nickname`: 暱稱 (可為空值)
- `gender`: 性別 (可為空值)
- `meet_count`: 見面次數 (可為空值)
- `meet_days`: 見面天數 (可為空值)

### VoicePrint 集合優化
- 移除冗餘的 `voiceprint_id`，直接使用 Weaviate UUID
- 新增 `sample_count`: 樣本數量 (預留欄位)
- 新增 `quality_score`: 品質分數 (可為None)

### 時間欄位重命名
- `create_time` → `created_at`
- `updated_time` → `updated_at`

## 📁 模組結構

```
modules/database/
├── database.py            # DatabaseService V2 類別實作
├── init_v2_collections.py # V2 集合初始化工具
└── README.md              # 本文檔
```

## 🛠 安裝與配置

### 前置需求
```bash
# 安裝基本相依套件
pip install -r requirements-base.txt

# GPU 支援 (可選)
pip install -r requirements-gpu.txt
```

### 環境配置
```bash
# 複製配置範例
cp .env.example .env

# 編輯 .env 檔案
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_SCHEME=http
```

### 啟動 Weaviate
```bash
docker-compose up -d
```

## 🏗 V2 資料庫結構

### Speaker 集合
```python
{
    "speaker_id": int,           # 序號ID (從1開始遞增)
    "full_name": str,           # 主要名稱
    "nickname": Optional[str],   # 暱稱 (可為None)
    "gender": Optional[str],     # 性別 (可為None)
    "created_at": datetime,      # 建立時間 (RFC3339格式)
    "last_active_at": datetime,  # 最後活動時間
    "meet_count": Optional[int], # 見面次數 (可為None)
    "meet_days": Optional[int],  # 見面天數 (可為None)
    "voiceprint_ids": List[str], # 關聯聲紋UUID列表
    "first_audio": str          # 首個音檔路徑
}
```

### VoicePrint 集合
```python
{
    "created_at": datetime,      # 建立時間 (RFC3339格式)
    "updated_at": datetime,      # 更新時間
    "update_count": int,         # 更新次數
    "sample_count": Optional[int], # 樣本數量 (預留欄位)
    "quality_score": Optional[float], # 品質分數 (可為None)
    "speaker_name": str,         # 語者名稱 (相容性)
    "speaker": Reference         # 關聯到Speaker集合
}
```

### Session 集合
```python
{
    "session_id": str,           # 會議/對話ID
    "session_type": str,         # 會議類型
    "title": str,               # 會議標題 (支援語意搜尋)
    "start_time": datetime,      # 開始時間
    "end_time": datetime,        # 結束時間
    "summary": str,              # 會議摘要 (支援語意搜尋)
    "participants": Reference[]  # 參與語者列表 (關聯到Speaker)
}
```

### SpeechLog 集合
```python
{
    "content": str,              # 語音轉錄文字 (支援語意搜尋)
    "timestamp": datetime,       # 發言時間戳
    "confidence": float,         # ASR 信心值
    "duration": float,           # 語音長度(秒)
    "language": str,             # 語言類型
    "speaker": Reference,        # 發言語者 (關聯到Speaker)
    "session": Reference         # 所屬會議 (關聯到Session)
}
```

## 💾 DatabaseService V2 核心類別

### 設計模式
- **單例模式**: 確保全域只有一個 DatabaseService 實例
- **連線管理**: 自動管理 Weaviate client 連線與重試機制
- **集合檢查**: 初始化時檢查必要的 V2 集合是否存在

### 主要方法

#### 語者管理 (Speaker Operations)
```python
# 列出所有語者
speakers: List[Dict] = db.list_all_speakers()

# 透過 UUID 查詢語者
speaker: Optional[Dict] = db.get_speaker(speaker_uuid)

# 透過序號 ID 查詢語者 (V2 新功能)
speaker: Optional[Dict] = db.get_speaker_by_id(speaker_id: int)

# 創建新語者
speaker_uuid: str = db.create_speaker(
    full_name="王小明",
    nickname="小明", 
    gender="男性"
)

# 更新語者名稱
success: bool = db.update_speaker_name(
    speaker_uuid, 
    new_full_name="王大明",
    new_nickname="大明"
)

# 更新最後活動時間
db.update_speaker_last_active(speaker_uuid, timestamp)

# 更新統計資訊
db.update_speaker_stats(speaker_uuid, meet_count=5, meet_days=3)

# 刪除語者
success: bool = db.delete_speaker(speaker_uuid)
```

#### 聲紋管理 (VoicePrint Operations)
```python
# 建立聲紋
voiceprint_uuid: str = db.create_voiceprint(
    speaker_uuid, 
    embedding_vector, 
    audio_source="path/to/audio.wav"
)

# 查詢聲紋
voiceprint: Optional[Dict] = db.get_voiceprint(voiceprint_uuid, include_vector=True)

# 更新聲紋
success: bool = db.update_voiceprint(voiceprint_uuid, new_embedding, update_count)

# 刪除聲紋  
success: bool = db.delete_voiceprint(voiceprint_uuid)

# 獲取語者的所有聲紋
voiceprints: List[Dict] = db.get_speaker_voiceprints(speaker_uuid, include_vectors=False)
```

#### 會議管理 (Session Operations)
```python
# 建立會議
session_uuid: str = db.create_session(
    session_id="meeting_001",
    session_type="會議",
    title="專案討論",
    start_time=datetime.now(),
    participant_uuids=["speaker_uuid1", "speaker_uuid2"]
)

# 查詢會議
session: Optional[Dict] = db.get_session(session_uuid)

# 結束會議
success: bool = db.end_session(session_uuid, end_time=datetime.now(), summary="會議總結")
```

#### 語音記錄管理 (SpeechLog Operations)
```python
# 建立語音記錄
speechlog_uuid: str = db.create_speechlog(
    content="這是一段語音轉錄內容",
    timestamp=datetime.now(),
    confidence=0.95,
    duration=5.2,
    speaker_uuid="speaker_uuid",
    session_uuid="session_uuid"
)

# 查詢語音記錄
speechlog: Optional[Dict] = db.get_speechlog(speechlog_uuid)

# 語意搜尋語音內容
results: List[Dict] = db.search_speech_content("關鍵字搜尋", limit=10)
```

#### 向量搜索 (Vector Search)
```python
# 搜索相似聲紋
results: List[Dict] = db.find_similar_voiceprints(embedding_vector, limit=5)

# 結果格式
# {
#     "uuid": "聲紋UUID",
#     "distance": 0.234,          # 餘弦距離
#     "speaker_uuid": "語者UUID",
#     "speaker_name": "語者名稱",
#     "properties": {...}         # 聲紋屬性
# }
```

#### 關聯管理
```python
# 添加聲紋到語者
db.add_voiceprint_to_speaker(speaker_uuid, voiceprint_uuid)

# 轉移聲紋
transferred_count: int = db.transfer_voiceprints(
    source_uuid, 
    dest_uuid, 
    voiceprint_uuids
)

# 獲取聲紋關聯的語者
speaker_uuid: Optional[str] = db.get_speaker_uuid_from_voiceprint(voiceprint_uuid)
```

## 🔧 V2 初始化與遷移

### 初始化 V2 集合
```bash
python -m modules.database.init_v2_collections
```

### 從 V1 遷移資料
```bash
# 1. 備份現有資料
python weaviate_study/tool_backup.py

# 2. 初始化 V2 集合
python -m modules.database.init_v2_collections

# 3. 匯入現有聲紋資料
python weaviate_study/npy_to_weaviate.py
```

## 🧪 使用範例

### 基本使用
```python
from modules.database.database import DatabaseService

# 取得單例實例
db = DatabaseService()

# 檢查資料庫連接
if not db.check_database_connection():
    print("資料庫連接失敗")
    exit(1)

# 創建新語者
speaker_uuid = db.create_speaker(
    full_name="測試語者",
    nickname="測試",
    gender="未知"
)

# 列出所有語者
speakers = db.list_all_speakers()
for speaker in speakers:
    print(f"ID: {speaker['speaker_id']}, 名稱: {speaker['full_name']}")
```

### 聲紋比對
```python
import numpy as np

# 假設有語音特徵向量
embedding = np.random.rand(192).tolist()

# 建立聲紋
voiceprint_uuid = db.create_voiceprint(
    speaker_uuid,
    embedding,
    audio_source="test.wav"
)

# 搜尋相似聲紋
similar_results = db.find_similar_voiceprints(embedding, limit=3)
for result in similar_results:
    print(f"相似度: {result['distance']:.3f}, 語者: {result['speaker_name']}")
```

## 🔍 疑難排解

### 常見問題
1. **連接失敗**: 確認 Weaviate 服務運行且配置正確
2. **集合不存在**: 執行 `init_v2_collections.py` 初始化
3. **向量維度錯誤**: 確保嵌入向量維度與模型一致 (192維)

### 除錯
```python
# 啟用詳細日誌
import logging
logging.basicConfig(level=logging.DEBUG)

# 檢查集合狀態
db.check_collection_exists("Speaker")
db.check_collection_exists("VoicePrint")
```

## 📊 效能優化

- 使用批次操作提高大量資料處理效率
- 向量搜尋限制結果數量避免記憶體溢出
- 定期清理無用的聲紋資料釋放儲存空間

## 🔗 相關文檔

- [Weaviate 官方文檔](https://weaviate.io/developers/weaviate)
- [CONFIG_README.md](../../CONFIG_README.md) - 配置說明
- [weaviate_study/README.md](../../weaviate_study/README.md) - V2 工具說明

## 📋 V2 集合驗證

系統啟動時會自動驗證以下 4 個 V2 集合是否存在：

| 集合名稱 | 用途說明 |
|---------|---------|
| `Speaker` | 語者基本資料，支援雙ID系統 |
| `VoicePrint` | 聲紋向量，支援餘弦相似度搜尋 |
| `Session` | 會議/對話紀錄，支援語意搜尋 |
| `SpeechLog` | 語音轉錄記錄，支援內容搜尋 |

如需手動驗證集合狀態：
```python
from modules.database.init_v2_collections import WeaviateV2CollectionManager

manager = WeaviateV2CollectionManager()
manager.connect()
results = manager.verify_v2_collections()
print(results)  # {'Speaker': True, 'VoicePrint': True, 'Session': True, 'SpeechLog': True}
```

