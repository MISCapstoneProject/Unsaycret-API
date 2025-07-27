# Unsaycret API 文檔

本項目提供語音處理與語者管理的 RESTful API 服務。

**最後更新**: 2025-07-28

## 🎯 API 端點總覽

### 🧑‍💼 語者管理 API (Speakers)
- `GET /speakers` - 列出所有語者
- `GET /speakers/{speaker_id}` - 查詢特定語者資訊
- `PATCH /speakers/{speaker_id}` - 更新語者資料
- `DELETE /speakers/{speaker_id}` - 刪除語者及其聲紋

### 🔧 語者操作 API (Speaker Actions)
- `POST /speakers/verify` - 語音驗證識別
- `POST /speakers/transfer` - 聲紋轉移

### 📅 會議管理 API (Sessions)
- `GET /sessions` - 列出所有會議
- `POST /sessions` - 建立新會議
- `GET /sessions/{session_id}` - 查詢特定會議資訊
- `PATCH /sessions/{session_id}` - 更新會議資料
- `DELETE /sessions/{session_id}` - 刪除會議

### 💬 語音記錄 API (SpeechLogs)
- `GET /speechlogs` - 列出所有語音記錄
- `POST /speechlogs` - 建立新語音記錄
- `GET /speechlogs/{speechlog_id}` - 查詢特定語音記錄
- `PATCH /speechlogs/{speechlog_id}` - 更新語音記錄
- `DELETE /speechlogs/{speechlog_id}` - 刪除語音記錄

### 🔗 關聯查詢 API (Nested Resources)
- `GET /speakers/{speaker_id}/sessions` - 語者參與的會議
- `GET /speakers/{speaker_id}/speechlogs` - 語者的語音記錄
- `GET /sessions/{session_id}/speechlogs` - 會議中的語音記錄

### ⚙️ 語音處理 API (Core Processing)
- `POST /transcribe` - 語音轉錄與語者識別
- `POST /transcribe_dir` - 批次轉錄
- `WS /ws/stream` - 即時串流轉錄


## 📋 API 詳細說明

### 1. 語音轉錄功能

**端點**: `POST /transcribe`

**功能**: 上傳音訊檔案進行語音分離、語者識別與語音轉錄

**請求格式**: `multipart/form-data`
- **file** (required): 音訊檔案 (支援 `.wav`, `.mp3`, `.m4a` 等格式)

**回應格式**:
```json
{
  "segments": [
    {
      "speaker_id": "81d60ed8-3c8b-43b8-808d-2dd4409ca814",
      "speaker_name": "王小明",
      "start_time": 0.0,
      "end_time": 3.5,
      "text": "你好，今天天氣很好",
      "confidence": 0.95
    }
  ],
  "pretty": "王小明: 你好，今天天氣很好",
  "stats": {
    "total_duration": 7.2,
    "speakers_detected": 2
  }
}
```

---

### 2. 語者管理 API

#### 2.1 列出所有語者

**端點**: `GET /speakers`

**功能**: 獲取所有語者的完整資訊列表

**回應格式**:
```json
[
  {
    "uuid": "81d60ed8-3c8b-43b8-808d-2dd4409ca814",
    "speaker_id": 1,
    "full_name": "王小明",
    "nickname": null,
    "gender": null,
    "created_at": "2025-01-27T14:30:00Z",
    "last_active_at": "2025-01-27T16:45:00Z",
    "meet_count": 5,
    "meet_days": 3,
    "voiceprint_ids": ["vp-001", "vp-002"],
    "first_audio": "audio-001"
  }
]
```

#### 2.2 查詢特定語者

**端點**: `GET /speakers/{speaker_id}`

**功能**: 獲取指定語者的詳細資訊

**路徑參數**:
- `speaker_id` (required): 語者的 UUID

**回應格式**: 同上

#### 2.3 更新語者資料

**端點**: `PATCH /speakers/{speaker_id}`

**功能**: 部分更新語者資料

**請求格式**:
```json
{
  "full_name": "王大明",
  "nickname": "大明",
  "gender": "男"
}
```

**回應格式**:
```json
{
  "success": true,
  "message": "語者資料更新成功",
  "data": {
    "speaker_id": "81d60ed8-3c8b-43b8-808d-2dd4409ca814",
    "updated_fields": ["full_name", "nickname", "gender"]
  }
}
```

#### 2.4 刪除語者

**端點**: `DELETE /speakers/{speaker_id}`

**功能**: 刪除指定語者及其所有關聯的聲紋資料

**回應格式**:
```json
{
  "success": true,
  "message": "成功刪除語者及其聲紋資料",
  "data": {
    "speaker_id": "81d60ed8-3c8b-43b8-808d-2dd4409ca814",
    "deleted_voiceprint_count": 3
  }
}
```

---

### 3. 語者操作 API

#### 3.1 語音驗證

**端點**: `POST /speakers/verify`

**功能**: 識別音檔中的語者身份（純讀取操作）

**請求格式**: `multipart/form-data`
- **file** (required): 要驗證的音檔
- **threshold** (optional): 比對閾值，預設 0.4
- **max_results** (optional): 最大結果數量，預設 3

**回應格式**:
```json
{
  "success": true,
  "message": "語音驗證完成",
  "is_known_speaker": true,
  "best_match": {
    "voiceprint_uuid": "vp-001",
    "speaker_name": "王小明",
    "distance": 0.23,
    "is_match": true
  },
  "all_candidates": [...],
  "threshold": 0.4,
  "total_candidates": 2
}
```

#### 3.2 聲紋轉移

**端點**: `POST /speakers/transfer`

**功能**: 將一個語者的所有聲紋轉移到另一個語者

**請求格式**:
```json
{
  "source_speaker_id": "source-uuid",
  "source_speaker_name": "來源語者",
  "target_speaker_id": "target-uuid",
  "target_speaker_name": "目標語者"
}
```

---

### 4. 會議管理 API

#### 4.1 列出所有會議

**端點**: `GET /sessions`

**回應格式**:
```json
[
  {
    "uuid": "session-uuid",
    "session_id": "S001",
    "session_type": "會議",
    "title": "項目討論會",
    "start_time": "2025-01-27T09:00:00Z",
    "end_time": "2025-01-27T10:30:00Z",
    "summary": "討論項目進度",
    "participants": ["speaker-uuid-1", "speaker-uuid-2"]
  }
]
```

#### 4.2 建立新會議

**端點**: `POST /sessions`

**請求格式**:
```json
{
  "session_type": "會議",
  "title": "項目討論會",
  "start_time": "2025-01-27T09:00:00Z",
  "participants": ["speaker-uuid-1"]
}
```

#### 4.3 查詢特定會議

**端點**: `GET /sessions/{session_id}`

#### 4.4 更新會議資料

**端點**: `PATCH /sessions/{session_id}`

#### 4.5 刪除會議

**端點**: `DELETE /sessions/{session_id}`

**功能**: 刪除指定會議及其所有關聯的語音記錄（級聯刪除）

**回應格式**:
```json
{
  "success": true,
  "message": "成功刪除 Session 會議名稱 及其 3 個關聯記錄",
  "data": {
    "uuid": "session-uuid",
    "session_name": "會議名稱",
    "deleted_speechlogs": 3
  }
}
```

**重要說明**: 
- 刪除會議時會自動刪除所有關聯的語音記錄
- 此操作不可逆，請謹慎使用

---

### 5. 語音記錄 API

#### 5.1 列出所有語音記錄

**端點**: `GET /speechlogs`

**回應格式**:
```json
[
  {
    "uuid": "speechlog-uuid",
    "content": "你好，今天天氣很好",
    "timestamp": "2025-01-27T10:15:30Z",
    "confidence": 0.95,
    "duration": 3.5,
    "language": "zh-TW",
    "speaker": "speaker-uuid",
    "session": "session-uuid"
  }
]
```

#### 5.2 建立新語音記錄

**端點**: `POST /speechlogs`

**請求格式**:
```json
{
  "content": "語音內容",
  "confidence": 0.95,
  "duration": 3.5,
  "speaker": "speaker-uuid",
  "session": "session-uuid"
}
```

#### 5.3 查詢特定語音記錄

**端點**: `GET /speechlogs/{speechlog_id}`

#### 5.4 更新語音記錄

**端點**: `PATCH /speechlogs/{speechlog_id}`

#### 5.5 刪除語音記錄

**端點**: `DELETE /speechlogs/{speechlog_id}`

---

### 6. 關聯查詢 API

#### 6.1 語者參與的會議

**端點**: `GET /speakers/{speaker_id}/sessions`

**功能**: 取得指定語者參與的所有會議

#### 6.2 語者的語音記錄

**端點**: `GET /speakers/{speaker_id}/speechlogs`

**功能**: 取得指定語者的所有語音記錄

#### 6.3 會議中的語音記錄

**端點**: `GET /sessions/{session_id}/speechlogs`

**功能**: 取得指定會議中的所有語音記錄（按時間排序）


## ⚠️ 錯誤處理

所有API都會返回統一格式的錯誤回應：

### HTTP狀態碼
- **200 OK**: 請求成功
- **400 Bad Request**: 參數錯誤或驗證失敗
- **404 Not Found**: 找不到指定的資源
- **500 Internal Server Error**: 伺服器內部錯誤

### 錯誤回應格式
```json
{
  "detail": "錯誤描述訊息"
}
```

### 常見錯誤情況

#### 輸入驗證錯誤 (400)
1. **空值參數**: ID參數不能為空
2. **格式錯誤**: 請求格式不符合要求
3. **必要參數缺失**: 缺少必要的請求參數

#### 資源不存在錯誤 (404)
1. **語者不存在**: 當提供的 `speaker_id` 無法在資料庫中找到
2. **會議不存在**: 當提供的 `session_id` 無法在資料庫中找到
3. **語音記錄不存在**: 當提供的 `speechlog_id` 無法在資料庫中找到

#### 業務邏輯錯誤 (400/409)
1. **名稱不匹配**: 聲紋轉移時提供的語者名稱與資料庫中的名稱不符
2. **相同ID操作**: 聲紋轉移時來源和目標語者為同一人
3. **數據衝突**: 嘗試建立重複的資源

#### 伺服器錯誤 (500)
1. **資料庫連接錯誤**: Weaviate 連接失敗
2. **內部處理錯誤**: 系統內部邏輯錯誤

## 🚀 部署與啟動

### 1. 環境準備
```bash
# 安裝依賴套件
pip install -r requirements.txt

# 啟動Weaviate資料庫
docker-compose up -d
```

### 2. 啟動API服務
```bash
# 方法一：使用 main.py
python main.py

# 方法二：使用 uvicorn 命令
uvicorn api.api:app --reload

# 方法三：指定主機和端口
uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 訪問API文檔
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc


## 🔐 安全注意事項

1. **身份驗證**: 
   - 聲紋轉移會驗證提供的語者名稱是否與資料庫匹配
   - UUID驗證確保資源存在於資料庫中
   - 輸入參數驗證防止空值和格式錯誤

2. **操作安全**:
   - 聲紋轉移操作會刪除來源語者，**無法自動撤銷**
   - 語者刪除操作會同時刪除所有關聯的聲紋資料，**不可逆**
   - **會議刪除會級聯刪除所有關聯的語音記錄，不可逆**
   - 建議在執行重要操作前先使用查詢API確認資料

3. **資料完整性**:
   - 所有修改操作都有原子性保證
   - 操作失敗時不會留下不一致的資料狀態
   - 級聯刪除確保不會產生孤立記錄
   - 輸入驗證防止無效資料進入系統

4. **日誌監控**:
   - 所有操作都會記錄在系統日誌中
   - 包含操作時間、參數和結果，便於審計和除錯

## 🧪 測試與除錯

### 測試檔案
```bash
# 基礎功能測試
python examples/test_all_apis.py

# 綜合測試（推薦）- 包含併發、邊界條件、錯誤恢復等
python examples/test_comprehensive_apis.py

# 語者API專項測試
python examples/test_speaker_api.py
```

### API測試工具
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 除錯建議
1. 檢查日誌檔案 `system_output.log` 獲取詳細錯誤資訊
2. 確認 Weaviate 資料庫連接正常
3. 驗證請求參數格式是否正確
4. 使用綜合測試驗證系統完整性

## ❓ 常見問題

**Q: 列表API返回的 `first_audio_id` 有什麼用途？**
A: `first_audio_id` 是該語者的第一個聲紋ID，可用於參考或快速訪問該語者的聲紋樣本。

**Q: 聲紋轉移會影響語音識別的準確性嗎？**
A: 轉移操作只是將聲紋資料重新關聯，不會改變聲紋向量本身，因此不會影響識別準確性。

**Q: 如何恢復已刪除的語者或會議？**
A: 刪除操作不可逆。會議刪除會級聯刪除所有關聯的語音記錄。建議在執行刪除前備份重要資料。

**Q: API支持批次操作嗎？**
A: 目前版本不支持批次操作，每次請求只能處理一個資源。如需批次操作，請多次調用API。

**Q: 系統如何處理併發操作？**
A: 系統經過併發測試驗證，支持多執行緒同時操作。資料庫操作具有原子性保證。

**Q: 轉移操作失敗後如何處理？**
A: 如果轉移失敗，原始資料會保持不變。檢查錯誤訊息並確認參數正確後重新嘗試。
