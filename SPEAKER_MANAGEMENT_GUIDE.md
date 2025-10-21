# 語者管理功能使用指南

本指南說明 Unsaycret API 中的兩個重要語者管理功能的使用方式。

## 📋 功能總覽

### 1. ✅ 語者轉移功能 (已實作)
- **API 端點**: `POST /speakers/transfer`
- **功能**: 將誤生成的語者合併到正確的語者
- **使用場景**: 修正自動識別錯誤，整理重複的語者檔案

### 2. ✅ 手動新增語者功能 (新實作)
- **API 端點**: `POST /speakers/create`
- **功能**: 手動上傳音檔建立新語者檔案
- **使用場景**: 快速建檔，不需透過即時錄音功能

---

## 🔄 語者轉移功能

### 使用方式
當系統誤將同一人識別為多個不同語者時，可以使用此功能合併。

#### API 呼叫範例
```bash
curl -X POST "http://localhost:8000/speakers/transfer" \
     -H "Content-Type: application/json" \
     -d '{
       "source_speaker_id": "錯誤語者的UUID",
       "source_speaker_name": "錯誤語者名稱",
       "target_speaker_id": "正確語者的UUID", 
       "target_speaker_name": "正確語者名稱"
     }'
```

#### 前端整合要點
1. **取得語者列表**: 先呼叫 `GET /speakers` 取得所有語者
2. **選擇來源和目標**: 讓用戶選擇要合併的語者
3. **確認操作**: 顯示清楚的確認對話框
4. **執行轉移**: 呼叫轉移 API
5. **更新介面**: 轉移完成後重新載入語者列表

#### 操作結果
- 來源語者的所有聲紋會轉移到目標語者
- 來源語者會被自動刪除
- 目標語者的聲紋數量會增加

---

## ➕ 手動新增語者功能

### 使用方式
當需要快速建立新語者檔案時，可上傳10秒左右的音檔。

#### API 呼叫範例 (multipart/form-data)
```bash
curl -X POST "http://localhost:8000/speakers/create" \
     -F "file=@recording.wav" \
     -F "full_name=王小明" \
     -F "nickname=小明" \
     -F "gender=男性"
```

#### 前端整合要點
1. **錄音或上傳**: 提供錄音按鈕或檔案上傳功能
2. **表單填寫**: 語者基本資料（姓名、暱稱、性別）
3. **音檔格式**: 僅支援 WAV
4. **建議長度**: 10秒左右的清晰語音
5. **即時回饋**: 顯示建立進度和結果

#### 支援參數
- **file** (必填): 音檔檔案
- **full_name** (必填): 語者全名 (最多50字元)
- **nickname** (選填): 語者暱稱 (最多30字元)  
- **gender** (選填): 性別資訊 (不限制選項，可填任何值)

#### 回應格式
```json
{
  "success": true,
  "message": "成功建立語者 '王小明' 並加入聲紋特徵",
  "data": {
    "speaker_uuid": "新語者的UUID",
    "speaker_id": 15,
    "full_name": "王小明",
    "nickname": "小明", 
    "gender": "男性",
    "voiceprint_uuid": "聲紋UUID",
    "voiceprint_count": 1
  }
}
```

---

## 🧪 測試方式

### 1. 啟動 API 服務
```bash
cd /Users/cyouuu/Desktop/Unsaycret/Unsaycret-API
python main.py
```

### 2. 測試手動建立語者
```bash
python test_create_speaker_api.py test_audio_me/output_001.wav
```

### 3. 檢查結果
- 訪問 http://localhost:8000/speakers 查看新建立的語者
- 確認聲紋特徵已正確建立

---

## 🚨 注意事項

### 語者轉移
- **不可逆操作**: 轉移完成後無法復原
- **名稱驗證**: 系統會驗證語者名稱是否匹配
- **自動清理**: 空的來源語者會被自動刪除

### 手動新增語者
- **音檔品質**: 建議使用清晰、無雜音的音檔
- **長度適中**: 10秒左右最佳，太短可能特徵不足
- **格式限制**: 僅支援常見音訊格式
- **重複檢查**: 系統不會自動檢查是否為重複語者

---

## 🔗 相關 API 端點

- `GET /speakers` - 列出所有語者
- `GET /speakers/{id}` - 查看語者詳細資訊
- `POST /speakers/verify` - 語音驗證識別
- `DELETE /speakers/{id}` - 刪除語者
- `GET /docs` - API 互動式文檔 (Swagger)

---

**文檔版本**: v0.4.2  
**最後更新**: 2025-01-27