# 🎯 VAD工具 - 簡潔版

## ✅ 已整合完成

現在只有**2個工具**，簡潔易用：

### 📁 工具列表
```
├── quick_vad.py      # 🚀 快速工具 (日常使用)
├── vad_module.py     # 🔧 完整模組 (進階功能)
├── test_vad.py       # 🧪 測試工具
└── VAD_GUIDE.md      # 📚 詳細使用指南
```

### 🗑️ 已刪除的重複工具
- ❌ `advanced_vad.py` (功能已整合到 `vad_module.py`)
- ❌ `silence_remover.py` (功能已整合到 `quick_vad.py`)
- ❌ `vad_configs.py` (配置已整合到 `vad_module.py`)

---

## 🚀 快速使用

### 1️⃣ 日常快速處理
```bash
# 處理單一檔案 (敏感模式)
python quick_vad.py audio.wav

# 批次處理目錄
python quick_vad.py audio_folder/

# 使用其他模式
python quick_vad.py audio.wav --strict    # 嚴格
python quick_vad.py audio.wav --ultra     # 超敏感
```

### 2️⃣ 進階功能
```bash
# 查看所有配置
python vad_module.py --list-presets

# 自訂參數
python vad_module.py input.wav output.wav --preset sensitive --energy-threshold -52

# 批次處理 + 報告
python vad_module.py input_folder/ output_folder/ --preset sensitive --report
```

### 3️⃣ 程式碼中使用
```python
from vad_module import VADProcessor

# 創建處理器
vad = VADProcessor(preset="sensitive")

# 處理檔案
output_path, stats = vad.process_file("input.wav")
```

---

## ⚙️ 預設模式說明

| 模式 | 保留率 | 適用場景 |
|------|--------|----------|
| `strict` | 40-60% | 高品質錄音，噪音多 |
| `normal` | 60-75% | 一般品質錄音 |
| `sensitive` | 85-95% | 重要內容，不想遺漏 ⭐ |
| `ultra_sensitive` | 95-98% | 低音量，微弱語音 |
| `music` | 70-85% | 歌聲和音樂內容 |
| `phone` | 80-90% | 電話錄音，低品質 |

**推薦**: 大多數情況使用 `sensitive` 模式

---

## 💡 使用建議

1. **新手**: 先用 `quick_vad.py`，簡單直接
2. **進階**: 用 `vad_module.py` 自訂參數和生成報告  
3. **批次**: 先測試少量檔案，確認效果後批次處理
4. **調參**: 如果語音被刪太多，用更敏感的模式；如果噪音太多，用更嚴格的模式

---

## 📊 整合效果

✅ **簡化了文件結構** - 從 6 個工具減少到 2 個主要工具  
✅ **統一了介面** - 所有功能通過統一的 `VADProcessor` 類別  
✅ **保留了功能** - 所有原有功能都完整保留  
✅ **改善了易用性** - 更清晰的使用方式和文檔  
✅ **向下相容** - 原有的參數調整功能完全保留
