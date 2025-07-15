# Unsaycret API 文檔

本項目提供語音處理與語者管理的API服務。

## 🎯 API 端點總覽

### 語音處理API
- `POST /transcribe` - 語音轉錄與語者識別

### 語者管理API
- `GET /speakers` - 列出所有語者
- `GET /speaker/{speaker_id}` - 查詢特定語者資訊
- `POST /speaker/rename` - 語者改名
- `POST /speaker/transfer` - 聲紋轉移
- `DELETE /speaker/{speaker_id}` - 刪除語者及其聲紋


## 📋 API詳細說明

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
    },
    {
      "speaker_id": "a372ca19-8531-4f3d-bee2-a7580989acf6",
      "speaker_name": "李小華",
      "start_time": 3.6,
      "end_time": 7.2,
      "text": "是的，很適合出去走走",
      "confidence": 0.92
    }
  ],
  "pretty": "王小明: 你好，今天天氣很好\n李小華: 是的，很適合出去走走"
}
```

**使用範例**:
```bash
# 使用 curl 上傳音訊檔案
curl -X POST "http://localhost:18000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_audio_file.wav"
```

### 2. 語者列表查詢

**端點**: `GET /speakers`

**功能**: 獲取所有語者的完整資訊列表

**請求參數**: 無

**回應格式**:
```json
[
  {
    "speaker_id": "81d60ed8-3c8b-43b8-808d-2dd4409ca814",
    "name": "王小明",
    "first_audio_id": "9a123520-db14-4b98-b0f5-c27470632946",
    "created_at": "2025-06-29T22:00:22.752980+08:00",
    "updated_at": "2025-06-29T22:51:08.300915+08:00",
    "voiceprint_ids": [
      "9a123520-db14-4b98-b0f5-c27470632946",
      "f9d94903-6748-4cef-b89e-7f2d359e764b",
      "9d8334c2-262d-4029-9e89-80a9908f34eb"
    ]
  },
  {
    "speaker_id": "a372ca19-8531-4f3d-bee2-a7580989acf6",
    "name": "李小華",
    "first_audio_id": "908cd3c8-0bf1-4041-8f85-88f40c2adc31",
    "created_at": "2025-06-29T22:00:22.203953+08:00",
    "updated_at": "2025-06-29T22:00:14.715988+08:00",
    "voiceprint_ids": [
      "908cd3c8-0bf1-4041-8f85-88f40c2adc31",
      "f45a57cb-5dc6-4799-a3eb-2fac92ed6aba"
    ]
  }
]
```

**欄位說明**:
- `speaker_id`: 語者的唯一識別碼 (UUID)
- `name`: 語者名稱
- `first_audio_id`: 第一個聲紋ID（用於參考）
- `created_at`: 語者創建時間 (ISO 8601格式)
- `updated_at`: 最後活動時間 (ISO 8601格式)
- `voiceprint_ids`: 該語者的所有聲紋ID列表

**使用範例**:
```bash
curl -X GET "http://localhost:18000/speakers" \
     -H "accept: application/json"
```

### 3. 單一語者資訊查詢

**端點**: `GET /speaker/{speaker_id}`

**功能**: 獲取指定語者的詳細資訊

**路徑參數**:
- `speaker_id` (required): 語者的UUID

**回應格式**:
```json
{
  "speaker_id": "81d60ed8-3c8b-43b8-808d-2dd4409ca814",
  "speaker_name": "王小明",
  "created_time": "2025-06-29T22:00:22.752980+08:00",
  "last_active_time": "2025-06-29T22:51:08.300915+08:00",
  "voiceprint_count": 3
}
```

**使用範例**:
```bash
curl -X GET "http://localhost:18000/speaker/81d60ed8-3c8b-43b8-808d-2dd4409ca814" \
     -H "accept: application/json"
```

### 4. 語者改名功能

**端點**: `POST /speaker/rename`

**功能**: 更改指定語者的名稱

**請求格式**:
```json
{
  "speaker_id": "語者的UUID",
  "current_name": "當前名稱",
  "new_name": "新名稱"
}
```

**回應格式**:
```json
{
  "success": true,
  "message": "成功將語者 '舊名稱' 更名為 '新名稱'",
  "data": {
    "speaker_id": "語者的UUID",
    "old_name": "舊名稱",
    "new_name": "新名稱"
  }
}
```

**使用範例**:
```bash
curl -X POST "http://localhost:18000/speaker/rename" \
     -H "Content-Type: application/json" \
     -d '{
       "speaker_id": "123e4567-e89b-12d3-a456-426614174000",
       "current_name": "王小明",
       "new_name": "王大明"
     }'
```

### 5. 聲紋轉移功能

**端點**: `POST /speaker/transfer`

**功能**: 將一個語者的所有聲紋轉移到另一個語者，並刪除來源語者

**請求格式**:
```json
{
  "source_speaker_id": "來源語者的UUID",
  "source_speaker_name": "來源語者名稱",
  "target_speaker_id": "目標語者的UUID",
  "target_speaker_name": "目標語者名稱"
}
```

**回應格式**:
```json
{
  "success": true,
  "message": "成功將語者 '來源名稱' 的所有聲紋轉移到 '目標名稱' 並刪除來源語者",
  "data": {
    "source_speaker_id": "來源語者UUID",
    "source_speaker_name": "來源語者名稱",
    "target_speaker_id": "目標語者UUID",
    "target_speaker_name": "目標語者名稱"
  }
}
```

**使用範例**:
```bash
curl -X POST "http://localhost:18000/speaker/transfer" \
     -H "Content-Type: application/json" \
     -d '{
       "source_speaker_id": "123e4567-e89b-12d3-a456-426614174000",
       "source_speaker_name": "錯誤識別的語者",
       "target_speaker_id": "987fcdeb-51d3-12a4-b567-426614174111",
       "target_speaker_name": "正確的語者"
     }'
```

### 6. 語者刪除功能

**端點**: `DELETE /speaker/{speaker_id}`

**功能**: 刪除指定語者及其所有關聯的聲紋資料

**路徑參數**:
- `speaker_id` (required): 語者的UUID

**回應格式**:
```json
{
  "success": true,
  "message": "成功刪除語者 '王小明' 及其 3 個聲紋",
  "data": {
    "speaker_id": "81d60ed8-3c8b-43b8-808d-2dd4409ca814",
    "speaker_name": "王小明",
    "deleted_voiceprint_count": 3
  }
}
```

**使用範例**:
```bash
curl -X DELETE "http://localhost:18000/speaker/81d60ed8-3c8b-43b8-808d-2dd4409ca814" \
     -H "accept: application/json"
```

**⚠️ 重要提醒**:
- 此操作為**不可逆**操作
- 會同時刪除語者本身和其所有聲紋資料
- 請謹慎使用此功能

### 7. 語音身份驗證

**端點**: `POST /speaker/verify`

**功能**: 查詢聲音是否為資料庫已有的語者 (純讀取操作 - 上傳音檔驗證語者身份，不會對資料庫進行任何修改或新增操作)

**請求格式**: `multipart/form-data`
- **file** (required): 要驗證的音檔 (支援 `.wav` 格式)
- **threshold** (optional): 比對閾值，距離小於此值才認為是匹配到語者，預設 0.4 (範圍 0.0-1.0)
- **max_results** (optional): 返回最相似的結果數量，預設 3 (範圍 1-10)

**回應格式**:
```json
{
  "success": true,
  "message": "語音驗證完成",
  "is_known_speaker": true,
  "best_match": {
    "voiceprint_id": "9a123520-db14-4b98-b0f5-c27470632946",
    "speaker_name": "王小明",
    "distance": 0.23,
    "is_match": true
  },
  "all_candidates": [
    {
      "voiceprint_id": "9a123520-db14-4b98-b0f5-c27470632946",
      "speaker_name": "王小明",
      "distance": 0.23,
      "update_count": 5,
      "is_match": true
    },
    {
      "voiceprint_id": "f9d94903-6748-4cef-b89e-7f2d359e764b",
      "speaker_name": "李小華",
      "distance": 0.68,
      "update_count": 3,
      "is_match": false
    }
  ],
  "threshold": 0.4,
  "total_candidates": 2
}
```

**回應欄位說明**:
- `is_known_speaker`: 是否為已知語者（基於比對閾值）
- `best_match`: 最佳匹配結果，包含語者資訊和相似度
- `all_candidates`: 所有候選結果，按相似度排序
- `distance`: 餘弦距離（越小越相似，0.0-1.0）
- `threshold`: 比對閾值，距離小於此值才認為是匹配
- `is_match`: 該候選是否超過比對閾值
- `update_count`: 該聲紋的更新次數
- `total_candidates`: 候選結果總數

**使用範例**:
```bash
# 基本驗證
curl -X POST "http://localhost:18000/speaker/verify" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@voice_to_verify.wav"

# 自訂閾值和結果數量
curl -X POST "http://localhost:18000/speaker/verify" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@voice_to_verify.wav" \
     -F "threshold=0.3" \
     -F "max_results=5"
```

**特點說明**:
- ✅ **純讀取操作**: 不會修改、新增或刪除任何資料庫內容
- ✅ **即時驗證**: 快速判斷音檔中的語者身份
- ✅ **多格式支援**: 支援常見的音檔格式
- ✅ **靈活閾值**: 可調整比對嚴格程度
- ✅ **詳細結果**: 提供完整的比對信息和候選列表


## ⚠️ 錯誤處理

所有API都包含完整的錯誤處理機制：

### HTTP狀態碼
- **200 OK**: 請求成功
- **400 Bad Request**: 參數錯誤或驗證失敗
- **404 Not Found**: 找不到指定的語者
- **500 Internal Server Error**: 伺服器內部錯誤

### 錯誤回應格式
```json
{
  "detail": "錯誤描述訊息"
}
```

### 常見錯誤情況
1. **語者不存在**: 當提供的 `speaker_id` 無法在資料庫中找到
2. **名稱不匹配**: 改名時提供的 `current_name` 與資料庫中的名稱不符
3. **參數驗證失敗**: 必要參數為空或格式不正確
4. **相同ID操作**: 聲紋轉移時來源和目標語者為同一人

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
uvicorn services.api:app --reload

# 方法三：指定主機和端口
uvicorn services.api:app --host 0.0.0.0 --port 18000 --reload
```

### 3. 訪問API文檔
- **Swagger UI**: http://localhost:18000/docs
- **ReDoc**: http://localhost:18000/redoc


## 🔐 安全注意事項

1. **身份驗證**: 
   - 語者改名和聲紋轉移會驗證提供的當前名稱是否與資料庫匹配
   - UUID驗證確保語者存在於資料庫中

2. **操作安全**:
   - 聲紋轉移操作會刪除來源語者，**無法自動撤銷**
   - 語者刪除操作會同時刪除所有關聯的聲紋資料，**不可逆**
   - 建議在執行重要操作前先使用查詢API確認資料

3. **資料完整性**:
   - 所有修改操作都有原子性保證
   - 操作失敗時不會留下不一致的資料狀態

4. **日誌監控**:
   - 所有操作都會記錄在系統日誌中
   - 包含操作時間、參數和結果，便於審計和除錯

## 🧪 測試與除錯

### 功能測試
```bash
# 執行語者API測試腳本
python examples/test_speaker_api.py
```

### API測試工具
- **Swagger UI**: http://localhost:18000/docs
- **ReDoc**: http://localhost:18000/redoc

### 除錯建議
1. 檢查日誌檔案 `system_output.log` 獲取詳細錯誤資訊
2. 確認 Weaviate 資料庫連接正常
3. 驗證請求參數格式是否正確

## ❓ 常見問題

**Q: 列表API返回的 `first_audio_id` 有什麼用途？**
A: `first_audio_id` 是該語者的第一個聲紋ID，可用於參考或快速訪問該語者的聲紋樣本。

**Q: 聲紋轉移會影響語音識別的準確性嗎？**
A: 轉移操作只是將聲紋資料重新關聯，不會改變聲紋向量本身，因此不會影響識別準確性。

**Q: 如何恢復已刪除的語者？**
A: 語者刪除操作不可逆。建議在執行刪除前使用 `GET /speakers` 備份重要資料。

**Q: API支持批次操作嗎？**
A: 目前版本不支持批次操作，每次請求只能處理一個語者。如需批次操作，請多次調用API。

**Q: 轉移操作失敗後如何處理？**
A: 如果轉移失敗，原始資料會保持不變。檢查錯誤訊息並確認參數正確後重新嘗試。
