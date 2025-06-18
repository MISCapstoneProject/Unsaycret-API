# Database Module (Weaviate)

**版本**：v1.0.1  
**作者**：Gino
**最後更新**：2025-05-19  

## 安裝前置需求
pip install -r requirements.txt
如果有GPU
pip install -r requirements-gpu.txt


### 目錄結構
modules/database/
├── VID_database.py        # DatabaseService 類別實作
└── README.md              # 模組說明（本檔）

### 核心類別
DatabaseService

單例模式：確保整個程式只有一個 DatabaseService 實例，內部管理 Weaviate client 連線。
連線檢查：初始化時會呼叫 weaviate.connect_to_local()，並檢查必須的兩個 collection（Speaker、VoicePrint）是否存在。

### 主要方法

Speaker 管理

list_all_speakers() → List[Dict]
列出所有語者及統計資訊。

get_speaker(speaker_id) → Dict or None
依 UUID 取回單一語者物件。

create_speaker(name: str) → str
建立一個新的 Speaker，回傳 UUID。

update_speaker_name(speaker_id, new_name) → bool
同步更新 Speaker 本身 & 底下所有其 voiceprint 的 speaker_name 屬性。

update_speaker_last_active(speaker_id, timestamp) → bool
更新最後活動時間欄位。

delete_speaker(speaker_id) → bool
刪除 Speaker 並一併刪除其所有聲紋向量。

VoicePrint 管理

create_voiceprint(speaker_id, embedding, …) → str
建立新的聲紋向量並回傳其 UUID。

get_voiceprint(id, include_vector=False) → Dict or None
取回單支向量（可選擇是否要回傳向量值）。

update_voiceprint(id, new_embedding, update_count) → int
以「加權移動平均」更新向量，並回傳新的更新次數。

delete_voiceprint(id) → bool
刪除向量並從所屬 Speaker 的列表中移除。

get_speaker_voiceprints(speaker_id, include_vectors=False) → List[Dict]
取回某語者所有聲紋資訊。

向量搜尋

find_similar_voiceprints(embedding, limit=3) → (best_id, best_name, best_dist, all_list)
對比輸入向量與資料庫中所有向量，依距離排序並回傳最相近結果。

工具方法

check_database_connection() → bool
快速驗證 Weaviate client 連線是否正常。

check_collection_exists(name) → bool
檢查指定的 collection（class）是否存在。

database_cleanup() → bool / Dict
執行多步檢查＆修復（如移除「孤兒」聲紋、修正錯誤 reference）。

### 使用範例

from modules.database.VID_database import DatabaseService

# 初始化（單例）
db = DatabaseService()

# 1. 創建並查詢 Speaker
sid = db.create_speaker("Alice")
all_sp = db.list_all_speakers()

# 2. 為 Alice 加入一筆聲紋
import numpy as np
vec = np.random.rand(512)
vp_id = db.create_voiceprint(sid, vec, audio_source="test.wav")

# 3. 搜尋最相似
best_id, best_name, dist, all_scores = db.find_similar_voiceprints(vec, limit=5)





###### 重點功能速查 (Quick Reference)

以下為最常用的場景與對應函式

| 場景說明                       | 函式名稱

| **列出所有語者**                | `list_all_speakers()`
| **查詢單一語者詳細資訊**         | `get_speaker(speaker_id: str)`
| **新增新語者**                  | `create_speaker(speaker_name: str)`
| **更新語者名稱**                | `update_speaker_name(speaker_id: str, new_name: str)` 
| **刪除語者及其所有聲紋**         | `delete_speaker(speaker_id: str)` 
| **新增聲紋向量**                | `create_voiceprint(speaker_id: str, embedding: np.ndarray, …)` 
| **查詢聲紋向量**                | `get_voiceprint(voiceprint_id: str, include_vector: bool=False)`
| **更新聲紋向量（加權移動平均）**  | `update_voiceprint(voiceprint_id: str, new_embedding: np.ndarray)` 
| **刪除聲紋向量**                | `delete_voiceprint(voiceprint_id: str)` 
| **列出某語者所有聲紋**           | `get_speaker_voiceprints(speaker_id: str, include_vectors: bool=False)` 
| **搜尋最相似聲紋**              | `find_similar_voiceprints(embedding: np.ndarray, limit: int=3)` 
| **檢查資料庫連線**              | `check_database_connection()` 
| **資料庫清理 & 修復**           | `database_cleanup()` 

> **示例**：  
> 如果你想查詢目前所有語者，就直接：
> ```python
> from modules.database.VID_database import DatabaseService
> db = DatabaseService()
> speakers = db.list_all_speakers()
> ```

