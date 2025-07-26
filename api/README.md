# API 模組

FastAPI 應用程式層，提供完整的 HTTP REST API 和 WebSocket 介面。

**版本**: v0.4.1  
**最後更新者**: CYouuu  
**最後更新**: 2025-07-27

## 📁 模組結構

```
api/
├── api.py              # FastAPI 應用程式主體
├── README.md           # 本文檔
```
services/
├── data_facade.py      # 資料存取門面（統一對外資料操作介面）

## 🚀 主要功能

### HTTP REST API
- **語音轉錄**: 單檔和批次音訊處理（分離+辨識+ASR）
- **語者管理**: CRUD 操作，支援 UUID 和序號 ID 雙重識別系統
- **聲紋管理**: 語音驗證、改名、轉移功能
- **資料查詢**: 完整的語者與聲紋資訊查詢

### WebSocket 即時處理  
- 即時語音串流處理
- 支援多執行緒背景處理
- 非同步結果傳送

## 🛠 啟動方式

### 標準啟動 (推薦)
```bash
python main.py
```
> 自動初始化 Weaviate V2 資料庫並啟動 API 服務

### 直接啟動 uvicorn
```bash
uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
```

## 📡 API 端點總覽

### 語音處理
- `POST /transcribe` - 單檔轉錄（分離+辨識+ASR）
- `POST /transcribe_dir` - 批次轉錄（目錄/ZIP）
- `WS /ws/stream` - 即時語音處理

### 語者管理  
- `GET /speakers` - 列出所有語者
- `GET /speaker/{id}` - 獲取語者資訊（支援 UUID/序號ID）
- `DELETE /speaker/{id}` - 刪除語者

### 聲紋管理
- `POST /speaker/verify` - 語音身份驗證
- `POST /speaker/rename` - 語者改名  
- `POST /speaker/transfer` - 聲紋轉移

## 🗄️ API 回應模型 (V2)

### SpeakerInfo (語者資訊)
```python
{
    "uuid": str,              # Weaviate UUID
    "speaker_id": int,        # 序號ID (從1開始)
    "full_name": str,         # 主要名稱
    "nickname": str,          # 暱稱 (可為None)
    "gender": str,            # 性別 (可為None)  
    "created_at": str,        # 建立時間 (ISO格式)
    "last_active_at": str,    # 最後活動時間 (ISO格式)
    "meet_count": int,        # 見面次數 (可為None)
    "meet_days": int,         # 見面天數 (可為None)
    "voiceprint_ids": List[str], # 關聯聲紋UUID列表
    "first_audio": str        # 首個音檔路徑
}
```

### VoiceVerificationResponse (語音驗證回應)
```python
{
    "success": bool,
    "message": str,
    "is_known_speaker": bool,
    "best_match": {           # 最佳匹配結果
        "voiceprint_uuid": str,
        "speaker_name": str,
        "distance": float,
        "is_match": bool
    },
    "all_candidates": [...],  # 所有候選者列表
    "threshold": float,       # 使用的比對閾值
    "total_candidates": int   # 總候選者數量
}
```

## ⚙️ 配置參數

API 服務使用環境變數配置，主要參數：

```bash
# .env 檔案
API_HOST=0.0.0.0                    # API 服務主機
API_PORT=8000                       # API 服務端口  
API_DEBUG=false                     # 除錯模式
API_LOG_LEVEL=info                  # 日誌等級
```

演算法參數定義於 `utils/constants.py`：

```python
# API 預設值
API_DEFAULT_VERIFICATION_THRESHOLD = 0.4  # 語音驗證閾值
API_DEFAULT_MAX_RESULTS = 3              # 預設最大結果數
API_MAX_WORKERS = 2                      # API 最大工作執行緒
```

## 🔧 資料存取門面（Data Facade）

### DataFacade
位於 `services/data_facade.py`，負責：

- **語者資料管理**: CRUD 操作，支援雙 ID 查詢
- **聲紋處理**: 語音驗證、改名、轉移等資料層邏輯
- **錯誤處理**: 統一的異常處理與 HTTP 錯誤回應
- **資料轉換**: 資料庫物件轉換為 API 回應格式

### 使用範例
```python
from services.data_facade import DataFacade

data_facade = DataFacade()

# 查詢語者 (支援 UUID 或序號ID)
speaker = data_facade.get_speaker_info("1")  # 序號ID
speaker = data_facade.get_speaker_info("uuid-string")  # UUID

# 列出所有語者
speakers = data_facade.list_all_speakers()

# 語音驗證
result = data_facade.verify_speaker_voice(
    audio_file_path="/path/to/audio.wav",
    threshold=0.4,
    max_results=3
)
```

## 📝 開發指南

### 添加新的 API 端點
1. 在 `api.py` 中定義路由和 Pydantic 模型
2. 在 `services/data_facade.py` 中實作資料存取邏輯
3. 更新 API 文檔

### 錯誤處理原則
- 使用 `HTTPException` 回傳標準 HTTP 錯誤
- 記錄詳細錯誤訊息到日誌
- 回傳使用者友善的錯誤訊息

### 測試
```bash
# 測試 API 端點
python examples/test_speaker_api.py
python examples/test_voice_verification.py
```

## 🔗 相關文檔

- [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) - 完整 API 文檔
- [CONFIG_README.md](../CONFIG_README.md) - 配置說明
- `services/data_facade.py` 內部文檔
