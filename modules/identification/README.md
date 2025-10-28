# Identification (聲紋識別) 模組

**版本**: v5.3.1  
**作者**: CYouuu  
**最後更新**: 2025-10-28

聲紋識別模組提供「輸入任何音檔 → 輸出說話者身分」的一站式能力，支援自動聲紋更新、新語者建立，並即時同步至 Weaviate V2 資料庫。

## ⭐ 核心架構

| 類別 | 功能 | 說明 |
|------|------|------|
| **AudioProcessor** | 音訊處理與 embedding 提取 | 支援 8/16/44.1 kHz 自動重採樣 → 16 kHz<br>使用 **Wespeaker** wespeaker-voxceleb-resnet293-LM 模型 |
| **WeaviateRepository** | 向量資料庫操作 (Weaviate V2) | 包辦比對、加權平均更新、新增聲紋、建新語者 |
| **SpeakerIdentifier** | 單例入口、策略判斷 | 依距離閾值自動選擇「跳過 / 更新 / 新增 / 建新語者」 |

## 快速開始

```bash
# 1. 安裝基礎依賴
pip install -r requirements-base.txt
# 2. 安裝 CPU / GPU 版本依賴，視硬體選擇其一
#  CPU 版本
pip install -r requirements-cpu.txt
#  GPU 版本
pip install -r requirements-gpu.txt

# 2. 啟動 Weaviate（Docker）
docker-compose -f weaviate_study/docker-compose.yml up -d
```

```python
from modules.identification.VID_identify_v5 import SpeakerIdentifier

idtf = SpeakerIdentifier()

# 1. 單檔識別（自動更新資料庫）
speaker_id, speaker_name, distance = idtf.process_audio_file("samples/voice.wav")

# 2. 僅提取聲紋向量（不更新資料庫）
embedding = idtf.audio_processor.extract_embedding("samples/voice.wav")
# 回傳 numpy array，shape: (192,) for Wespeaker

# 3. 比對聲紋與資料庫（不自動更新）
best_id, best_name, best_distance, all_distances = idtf.database.compare_embedding(embedding)

# 4. 批次處理資料夾
stats = idtf.process_audio_directory("samples/")

# 5. 已知語者新增聲紋
success = idtf.add_voiceprint_to_speaker("voice.wav", speaker_uuid)
```

## 閾值策略（Wespeaker 優化版）

| 距離範圍 | 動作 | 說明 |
|---------|------|------|
| < 0.11 | ⏭️ **跳過** | 距離極小，認定同一檔案 |
| 0.11 – 0.22 | 🔄 **更新** | 更新現有聲紋（加權平均） |
| 0.22 – 0.39 | ➕ **新增聲紋** | 新增聲紋至同語者 |
| > 0.39 | 🆕 **新語者** | 建立新的 Speaker & VoicePrint |

### 自訂閾值：
```python
idtf.threshold_low = 0.11      # 跳過閾值
idtf.threshold_update = 0.22   # 更新閾值
idtf.threshold_new = 0.39      # 新語者閾值
```

## API 使用指南

| 使用場景 | 對應方法 | 說明 | 回傳值 |
|---------|---------|------|-------|
| **單檔辨識** | `process_audio_file(audio_path)` | 處理音檔並自動更新資料庫 | `(speaker_id, speaker_name, distance)` |
| **僅提取向量** | `audio_processor.extract_embedding(audio_path)` | 只取得 embedding，不碰資料庫 | `numpy.ndarray (192,)` |
| **僅比對不更新** | `database.compare_embedding(embedding)` | 比對相似度但不更新 | `(best_id, best_name, best_distance, all_distances)` |
| **批次處理** | `process_audio_directory(dir_path)` | 處理整個資料夾 | `dict` 統計結果 |
| **手動新增聲紋** | `add_voiceprint_to_speaker(audio_path, speaker_id)` | 將音檔加到指定語者 | `bool` 成功與否 |

### ⚠️ 重要提醒
- ❌ `process_audio_stream()` 和 `extract_embedding_from_stream()` 目前已註解（Wespeaker 不支援音流版本）
- ✅ Orchestrator 使用 `audio_processor.extract_embedding()` + `process_audio_file()` 組合
- ✅ 單例模式：全域只會初始化一次模型，節省記憶體



## 前置需求

- Python 3.9+
- **Wespeaker** 模型（自動下載 wespeaker-voxceleb-resnet293-LM）
- Weaviate 向量資料庫（Docker 一鍵啟動）
- NumPy / PyTorch / SoundFile / SciPy

```bash
# 完整安裝
pip install -r requirements-base.txt

# 或分別安裝
pip install wespeaker weaviate-client numpy scipy soundfile torch
pip install git+https://github.com/wenet-e2e/wespeaker.git
```

## 注意事項

1. ✅ 確保 Weaviate 已啟動，且 `Speaker` / `VoicePrint` V2 Schema 已建立  
   執行：`python -m modules.database.init_v2_collections`

2. ✅ 最佳效果建議輸入 **16 kHz 單聲道** 音檔

3. ⚠️ 大量批次匯入時可調寬 `threshold_new`，避免產生過多新語者

4. ⚠️ Wespeaker 模型不支援音流版本，請使用檔案版本的 API

## 技術細節

- **模型**：Wespeaker wespeaker-voxceleb-resnet293-LM
- **向量維度**：192
- **資料庫**：Weaviate V2（支援 named vectors）
- **更新策略**：加權移動平均（Weighted Moving Average）
- **單例模式**：全域共用一個模型實例

## 相關文件

- 詳細 API 說明：見 `VID_identify_v5.py` 頂部文檔
- 資料庫結構：見 `modules/database/init_v2_collections.py`
- Orchestrator 整合：見 `pipelines/orchestrator_v2.py`