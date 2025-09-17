# 語音處理測試使用範例

## 📁 資料夾音檔測試

如果你有一個資料夾包含多個連續的音檔片段，可以這樣使用：

### 1. 基本使用（資料夾）
```bash
# 測試整個資料夾的音檔
python model_comparison_test.py --audio_path /path/to/your/audio/folder

# 指定不同的採樣率
python model_comparison_test.py --audio_path /path/to/your/audio/folder --sample_rate 22050

# 調整chunk大小
python model_comparison_test.py --audio_path /path/to/your/audio/folder --chunk_size_ms 3000
```

### 2. 資料夾結構範例
```
your_audio_folder/
├── part_001.wav  # 第1段
├── part_002.wav  # 第2段
├── part_003.wav  # 第3段
├── part_004.mp3  # 第4段（支援不同格式）
└── part_005.wav  # 第5段
```

**重要**: 檔案會按檔名排序，請確保檔名能正確反映順序！

### 3. 支援的音檔格式
- WAV (.wav)
- MP3 (.mp3) 
- FLAC (.flac)
- M4A (.m4a)
- AAC (.aac)
- OGG (.ogg)

### 4. 自動處理功能
✅ **自動重採樣**: 不同採樣率的音檔會自動重採樣到16kHz  
✅ **自動合併**: 多個音檔按順序合併成連續音訊  
✅ **格式轉換**: 自動轉換為16-bit PCM格式  
✅ **單聲道轉換**: 立體聲音檔會自動轉為單聲道  

## 🎵 單一檔案測試

```bash
# 測試單一音檔
python model_comparison_test.py --audio_path your_audio.wav

# 指定輸出檔案
python model_comparison_test.py --audio_path your_audio.wav --output my_test_result.json
```

## 📊 測試報告

測試會生成詳細的JSON報告，包含：

```json
{
  "test_info": {
    "audio_path": "/path/to/audio",
    "is_directory": true,
    "chunk_size_ms": 6000,
    "target_sample_rate": 16000,
    "test_time": "2025-09-07 15:30:45",
    "session_uuid": "uuid-here"
  },
  "audio_info": {
    "total_files": 5,
    "files": [
      {
        "filename": "part_001.wav",
        "original_sr": 44100,
        "original_duration": 10.5,
        "samples": 462465
      }
    ],
    "original_duration": 52.3,
    "resampled_duration": 52.3,
    "original_sample_rates": [44100, 22050],
    "target_sample_rate": 16000
  },
  "websocket_results": [...],
  "file_api_result": {...},
  "comparison": {...}
}
```

## 🔧 進階使用

### 測試不同模型配置
1. 修改 `.env` 或 `utils/constants.py` 中的模型設定
2. 重新啟動API服務
3. 執行測試並比較結果

### 批次測試範例
```bash
#!/bin/bash
# 測試不同chunk大小的效果

for chunk_size in 3000 6000 9000; do
    echo "測試 chunk_size: ${chunk_size}ms"
    python model_comparison_test.py \
        --audio_path /path/to/audio/folder \
        --chunk_size_ms $chunk_size \
        --output "test_chunk_${chunk_size}ms.json"
done

echo "所有測試完成！"
```

## ⚠️ 注意事項

1. **檔案順序**: 確保檔名能正確排序（如 001, 002, 003...）
2. **API服務**: 確保API服務（`python main.py`）正在運行
3. **Weaviate**: 確保Weaviate資料庫（`docker-compose up -d`）正在運行
4. **記憶體**: 大型音檔可能需要較多記憶體
5. **網路**: WebSocket測試需要穩定的網路連線

## 💡 最佳實踐

1. **標準化檔名**: 使用一致的命名規則，如 `segment_001.wav`, `segment_002.wav`
2. **分段長度**: 建議每段5-30秒，便於測試即時處理效果
3. **音質統一**: 盡量使用相同的音質和格式
4. **備份原檔**: 測試前備份原始音檔
