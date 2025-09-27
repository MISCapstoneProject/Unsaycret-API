# 🎯 VAD模組使用指南

## 📋 工具概覽

現在只有**2個簡潔的VAD工具**：

### 1. 🚀 **quick_vad.py** - 快速工具 (推薦日常使用)
最簡單的使用方式，適合快速處理

### 2. 🔧 **vad_module.py** - 完整模組 (推薦進階使用)
完整功能，支援自訂參數、批次處理和詳細報告

---

## 🚀 快速開始

### 最簡單的使用方式：

```bash
# 處理單一檔案 (自動使用敏感模式)
python quick_vad.py audio.wav

# 批次處理整個目錄
python quick_vad.py test_audio/

# 使用不同模式
python quick_vad.py audio.wav --strict      # 嚴格模式
python quick_vad.py audio.wav --ultra       # 超敏感模式
```

---

## 🔧 進階使用

### 1. 預設配置模式

| 模式 | 適用場景 | 特點 |
|------|----------|------|
| `strict` | 高品質錄音清理 | 只保留清晰語音 |
| `normal` | 一般使用 | 平衡處理 |
| `sensitive` | 重要內容 | 保留更多語音 ⭐ |
| `ultra_sensitive` | 低音量錄音 | 保留微弱語音 |
| `music` | 歌聲偵測 | 適合音樂內容 |
| `phone` | 電話錄音 | 適合低品質音訊 |

### 2. 使用完整模組

```bash
# 查看所有可用預設
python vad_module.py --list-presets

# 使用預設配置
python vad_module.py input.wav output.wav --preset sensitive

# 批次處理並生成報告
python vad_module.py test_audio/ processed_audio/ --preset sensitive --report

# 自訂參數
python vad_module.py input.wav output.wav --energy-threshold -50 --min-duration 0.2

# 組合預設和自訂參數
python vad_module.py input.wav output.wav --preset sensitive --energy-threshold -52
```

### 3. 程式碼中使用

```python
# 導入模組
from vad_module import VADProcessor, BatchVADProcessor

# 創建處理器
vad = VADProcessor(preset="sensitive")

# 處理單一檔案
output_path, stats = vad.process_file("input.wav", "output.wav")

# 批次處理
processor = BatchVADProcessor(vad)
result = processor.process_directory("input_folder/", "output_folder/")
```

---

## ⚙️ 參數調整指南

### 核心參數說明

| 參數名稱 | 作用 | 範圍 | 調整建議 |
|----------|------|------|----------|
| `energy_threshold` | 最低能量閾值 | -60 ~ -30 | 降低→更敏感 |
| `loud_threshold` | 響亮聲音閾值 | -50 ~ -30 | 降低→接受更小聲 |
| `zcr_min` | 最小過零率 | 0.005 ~ 0.05 | 降低→接受更單調 |
| `zcr_max` | 最大過零率 | 0.20 ~ 0.40 | 提高→接受更嘈雜 |
| `min_duration` | 最短語音長度 | 0.1 ~ 0.8 | 縮短→保留短語音 |
| `padding` | 前後保留時間 | 0.03 ~ 0.1 | 增加→避免切斷 |

### 常見問題解決

**🔴 語音被誤刪？**
```bash
# 使用更敏感的模式
python quick_vad.py audio.wav --ultra

# 或調整參數
python unified_vad.py audio.wav output.wav --energy-threshold -52 --min-duration 0.15
```

**🔴 保留太多噪音？**
```bash
# 使用更嚴格的模式
python quick_vad.py audio.wav --strict

# 或調整參數
python unified_vad.py audio.wav output.wav --energy-threshold -42 --min-duration 0.4
```

**🔴 語音斷斷續續？**
```bash
# 增加前後保留時間
python unified_vad.py audio.wav output.wav --padding 0.1 --zcr-max 0.35
```

---

### 批次處理並生成報告

```bash
# 批次處理並生成CSV/JSON報告
python vad_module.py test_audio/ processed_audio/ --preset sensitive --report
```

報告包含：
- 📄 **JSON報告**: 完整配置和統計數據
- 📈 **CSV報告**: 適合Excel分析的表格數據
- 📊 **總體統計**: 壓縮率、語音段落統計等

### 報告內容範例

```
📊 總體統計:
   總原始時長: 150.2s
   總語音時長: 142.8s  
   平均壓縮率: 95.1%
   平均語音段落: 3.2 個/檔案
```

---

## 🎯 實用範例

### 日常使用場景

```bash
# 1. 清理會議錄音 (保留重要內容)
python quick_vad.py meeting.wav --sensitive

# 2. 處理播客 (移除長時間停頓)
python quick_vad.py podcast.wav --normal

# 3. 清理低品質錄音 (電話/遠距)
python vad_module.py phone_call.wav clean.wav --preset phone

# 4. 批次處理大量檔案
python quick_vad.py audio_archive/

# 5. 精細調整參數
python vad_module.py interview.wav clean.wav \
    --preset sensitive \
    --energy-threshold -48 \
    --padding 0.08 \
    --report
```

---

## 🗂️ 文件結構

```
Unsaycret-API/
├── quick_vad.py          # 🚀 快速工具 (日常使用)
├── vad_module.py         # 🔧 完整模組 (進階功能)
├── test_vad.py           # 🧪 測試工具
├── VAD_GUIDE.md          # � 使用指南
└── utils/
    └── voice_activity_detection.py  # 核心VAD功能
```

---

## 💡 使用建議

### 🎯 **新手推薦流程：**

1. **先試快速工具**: `python quick_vad.py audio.wav --sensitive`
2. **如果效果不理想**: 試試其他模式 (`--strict`, `--ultra`)
3. **需要批次處理**: `python quick_vad.py audio_folder/`
4. **需要詳細控制**: 使用 `vad_module.py` 自訂參數

### 🔧 **進階用戶建議：**

1. **使用完整模組**: `vad_module.py` 提供最完整功能
2. **生成報告**: 加上 `--report` 參數分析處理效果
3. **參數調整**: 基於報告結果微調參數
4. **程式碼整合**: 直接導入 `VADProcessor` 類別使用

### 📊 **批次處理建議：**

1. **先測試少量檔案**: 確認參數效果
2. **生成報告**: 分析整體處理效果
3. **根據統計調整**: 如果平均壓縮率過低/過高，調整參數
4. **保存處理記錄**: 報告可用於後續分析和改進

---

## ⚡ 效能參考

根據之前的測試結果：

| 模式 | 平均保留率 | 適用場景 |
|------|------------|----------|
| `strict` | 40-60% | 高品質錄音，噪音較多 |
| `normal` | 60-75% | 一般品質錄音 |
| `sensitive` | 85-95% | 重要內容，不想遺漏 ⭐ |
| `ultra_sensitive` | 95-98% | 低音量，微弱語音 |

**推薦**: 大多數情況下使用 `sensitive` 模式，既保留了重要語音，又移除了明顯的靜音。
