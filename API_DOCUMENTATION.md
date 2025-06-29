# Unsaycret API 文檔

本項目提供語音處理與說話者管理的API服務。

### 語音處理API
- `POST /transcribe` - 語音轉錄與說話者識別

### 說話者管理API
- `POST /speaker/rename` - 說話者改名
- `POST /speaker/transfer` - 聲紋轉移
- `GET /speaker/{speaker_id}` - 說話者資訊查詢


## 📋 API詳細說明

### 1. 語音轉錄功能

**端點**: `POST /transcribe`

**功能**: 上傳音訊檔案進行語音分離、說話者識別與語音轉錄

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
curl -X POST "http://localhost:8000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_audio_file.wav"
```

### 1. 說話者改名功能

**端點**: `POST /speaker/rename`

**功能**: 更改指定說話者的名稱

**請求格式**:
```json
{
  "speaker_id": "說話者的UUID",
  "current_name": "當前名稱",
  "new_name": "新名稱"
}
```

**回應格式**:
```json
{
  "success": true,
  "message": "成功將說話者 '舊名稱' 更名為 '新名稱'",
  "data": {
    "speaker_id": "說話者的UUID",
    "old_name": "舊名稱",
    "new_name": "新名稱"
  }
}
```

**使用範例**:
```bash
curl -X POST "http://localhost:8000/speaker/rename" \
     -H "Content-Type: application/json" \
     -d '{
       "speaker_id": "123e4567-e89b-12d3-a456-426614174000",
       "current_name": "王小明",
       "new_name": "王大明"
     }'
```

### 2. 聲紋轉移功能

**端點**: `POST /speaker/transfer`

**功能**: 將一個說話者的所有聲紋轉移到另一個說話者，並刪除來源說話者

**請求格式**:
```json
{
  "source_speaker_id": "來源說話者的UUID",
  "source_speaker_name": "來源說話者名稱",
  "target_speaker_id": "目標說話者的UUID",
  "target_speaker_name": "目標說話者名稱"
}
```

**回應格式**:
```json
{
  "success": true,
  "message": "成功將說話者 '來源名稱' 的所有聲紋轉移到 '目標名稱' 並刪除來源說話者",
  "data": {
    "source_speaker_id": "來源說話者UUID",
    "source_speaker_name": "來源說話者名稱",
    "target_speaker_id": "目標說話者UUID",
    "target_speaker_name": "目標說話者名稱"
  }
}
```

**使用範例**:
```bash
curl -X POST "http://localhost:8000/speaker/transfer" \
     -H "Content-Type: application/json" \
     -d '{
       "source_speaker_id": "123e4567-e89b-12d3-a456-426614174000",
       "source_speaker_name": "錯誤識別的說話者",
       "target_speaker_id": "987fcdeb-51d3-12a4-b567-426614174111",
       "target_speaker_name": "正確的說話者"
     }'
```

### 3. 說話者資訊查詢

**端點**: `GET /speaker/{speaker_id}`

**功能**: 獲取指定說話者的詳細資訊

**回應格式**:
```json
{
  "speaker_id": "說話者的UUID",
  "speaker_name": "說話者名稱",
  "created_time": "創建時間",
  "last_active_time": "最後活躍時間",
  "voiceprint_count": 5
}
```

## 錯誤處理

所有API都包含完整的錯誤處理機制：

- **400 Bad Request**: 參數錯誤或驗證失敗
- **404 Not Found**: 找不到指定的說話者
- **500 Internal Server Error**: 伺服器內部錯誤

錯誤回應格式：
```json
{
  "detail": "錯誤描述"
}
```

## 啟動服務

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
uvicorn services.api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 訪問API文檔
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc


## 安全注意事項

1. **名稱驗證**: API會驗證提供的當前名稱是否與資料庫中的名稱匹配，以防止意外操作
2. **UUID驗證**: 會檢查提供的Speaker ID是否存在於資料庫中
3. **操作不可逆**: 聲紋轉移操作會刪除來源說話者，請謹慎使用
4. **日誌記錄**: 所有操作都會記錄在系統日誌中，便於追蹤和除錯

## 測試

可以使用提供的測試腳本進行功能測試：
```bash
python examples/test_speaker_api.py
```

## 常見問題

**Q: 聲紋轉移會影響語音識別的準確性嗎？**
A: 轉移操作只是將聲紋資料從一個說話者關聯到另一個說話者，不會改變聲紋向量本身，因此不會影響識別準確性。

**Q: 如果提供錯誤的當前名稱會怎樣？**
A: API會回傳400錯誤，並提示名稱不匹配，操作不會執行。

**Q: 轉移操作可以撤銷嗎？**
A: 無法自動撤銷，但您可以手動將聲紋再次轉移回原來的說話者（需要重新創建說話者）。
