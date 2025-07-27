# API 測試指南
**最後更新**: 2025-07-28

## 🧪 測試檔案說明

### 基礎測試
`test_all_apis.py` - 基本的 API 功能測試，驗證所有端點的正確性

### 綜合測試  
`test_comprehensive_apis.py` - 全面的系統測試，包含併發、邊界條件、錯誤恢復等進階測試

## 🚀 使用方法

### 1. 環境準備

```bash
# 確保 API 伺服器運行中
python main.py
# 或
uvicorn api.api:app --reload

# 確保 Weaviate 資料庫運行中
docker-compose up -d
```

### 2. 執行測試

#### 基礎測試
```bash
# 基本使用（預設 localhost:8000）
python examples/test_all_apis.py

# 指定不同的伺服器地址
python examples/test_all_apis.py http://localhost:8080
```

#### 綜合測試
```bash
# 執行完整的綜合測試（推薦）
python examples/test_comprehensive_apis.py

# 指定不同的伺服器地址
python examples/test_comprehensive_apis.py http://localhost:8080
```

### 3. 測試覆蓋範圍

#### 📋 基礎測試 (`test_all_apis.py`)

##### ✅ Sessions API
- `GET /sessions` - 列出所有會議
- `POST /sessions` - 建立新會議  
- `GET /sessions/{id}` - 查詢特定會議
- `PATCH /sessions/{id}` - 更新會議資料
- `DELETE /sessions/{id}` - 刪除會議

##### ✅ SpeechLogs API
- `GET /speechlogs` - 列出所有語音記錄
- `POST /speechlogs` - 建立新語音記錄
- `GET /speechlogs/{id}` - 查詢特定語音記錄
- `PATCH /speechlogs/{id}` - 更新語音記錄
- `DELETE /speechlogs/{id}` - 刪除語音記錄

##### ✅ Speakers API（模擬測試）
- `GET /speakers` - 列出所有語者
- `GET /speakers/{id}` - 查詢特定語者
- `PATCH /speakers/{id}` - 更新語者資料
- 錯誤處理測試

##### ✅ 關聯查詢 API
- `GET /speakers/{id}/sessions` - 語者的會議
- `GET /speakers/{id}/speechlogs` - 語者的語音記錄
- `GET /sessions/{id}/speechlogs` - 會議的語音記錄

##### ✅ 錯誤處理
- 無效 UUID 測試
- 無效 JSON 資料測試
- 伺服器連接測試

#### 🔥 綜合測試 (`test_comprehensive_apis.py`)

##### ✅ 數據庫重置測試
- 自動重置 Weaviate 資料庫
- 驗證初始狀態清空

##### ✅ 有資料狀態測試
- 建立多個 Sessions 和 SpeechLogs
- 驗證資料查詢和排序
- 測試 Session-SpeechLog 關聯查詢

##### ✅ 語音驗證功能
- 空資料庫語音驗證測試
- 未知語者識別驗證

##### ✅ 併發操作測試
- 多執行緒建立 Sessions
- 併發查詢操作壓力測試
- 競爭條件檢測

##### ✅ 邊界條件測試
- 超長字串處理（1000+ 字符）
- 特殊字符注入測試（XSS、SQL注入）
- 空值和 null 處理

##### ✅ 錯誤恢復機制
- 無效 UUID 格式處理
- 空字串 UUID 重定向測試
- 不存在資源查詢
- 惡意輸入防護

##### ✅ 數據完整性測試
- 級聯刪除驗證（Session 刪除時自動清理 SpeechLogs）
- 資料清理功能完整性
- 孤立記錄檢測

### 4. 測試流程

#### 基礎測試流程
1. **健康檢查** - 驗證 API 伺服器狀態
2. **CRUD 測試** - 完整的建立、讀取、更新、刪除流程
3. **關聯測試** - 測試資源間的關聯查詢
4. **錯誤測試** - 驗證錯誤處理機制
5. **清理資料** - 刪除測試過程中建立的資料
6. **生成報告** - 輸出詳細的測試結果

#### 綜合測試流程
1. **資料庫重置** - 自動重置到乾淨狀態
2. **功能完整性測試** - 有資料狀態下的所有功能驗證
3. **語音模組測試** - 語音驗證功能測試
4. **併發壓力測試** - 多執行緒操作測試
5. **邊界條件測試** - 極限輸入和特殊字符測試
6. **錯誤恢復測試** - 異常情況處理驗證
7. **完整清理** - 級聯刪除和資料完整性驗證

### 5. 測試報告

#### 基礎測試報告
- **控制台輸出** - 即時的測試結果
- **test_report.json** - 詳細的 JSON 格式報告

#### 綜合測試報告
- **控制台輸出** - 彩色即時測試結果
- **comprehensive_test_report.json** - 完整的測試報告（包含時間戳、錯誤詳情）
- **成功率統計** - 總測試數量、通過率、失敗測試列表

## ⚠️ 注意事項

### 資料庫狀態
- **基礎測試**: 測試前請確保資料庫是**空白狀態**
- **綜合測試**: 會自動重置資料庫到乾淨狀態
- 測試會自動清理建立的測試資料
- 如果測試中斷，可能需要手動清理殘留資料

### 音檔相關測試
由於需要實際的音檔，以下功能僅進行**模擬測試**：
- `POST /speakers/verify` - 語音驗證
- `POST /speakers/transfer` - 聲紋轉移
- `POST /transcribe` - 語音轉錄

### 測試選擇建議
- **開發階段**: 使用 `test_all_apis.py` 快速驗證基本功能
- **部署前**: 使用 `test_comprehensive_apis.py` 進行完整驗證
- **CI/CD**: 建議同時運行兩個測試確保完整覆蓋

### 依賴套件
```bash
pip install requests
```

## 🐛 故障排除

### 常見問題

1. **連接失敗**
   ```
   請求失敗: GET http://localhost:8000/docs - Connection refused
   ```
   **解決**: 確保 API 伺服器正在運行

2. **資料庫錯誤**
   ```
   狀態碼: 500, 內容: Internal Server Error
   ```
   **解決**: 確保 Weaviate 資料庫正常運行

3. **測試失敗**
   ```
   ❌ FAIL 建立 Session: 建立失敗
   ```
   **解決**: 檢查 `test_report.json` 獲取詳細錯誤信息

### 手動清理資料庫

如果需要手動清理測試資料：

```bash
# 進入 API 伺服器控制台
python -c "
from modules.database.database import DatabaseService
db = DatabaseService()
# 根據需要清理特定資料
"
```

## 📊 預期結果

### 基礎測試預期結果
- **總測試數量**: ~25-30 個測試
- **成功率**: >90%
- **失敗測試**: 主要是語音相關的模擬測試

### 綜合測試預期結果
- **總測試數量**: ~23 個測試
- **成功率**: 100%（所有功能正常）
- **測試時間**: 約 1-2 分鐘

### 成功指標
- ✅ Sessions CRUD 完全正常
- ✅ SpeechLogs CRUD 完全正常  
- ✅ 關聯查詢功能正常
- ✅ 錯誤處理機制正常
- ✅ 資料清理功能正常
- ✅ 併發操作穩定
- ✅ 邊界條件處理正確
- ✅ 級聯刪除功能正常

測試通過代表你的 API 重構成功，所有端點都能正常工作！🎉
