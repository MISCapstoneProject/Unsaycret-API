# Modules 核心模組總覽

**版本**：v0.4.0  
**作者**: CYouuu  
**最後更新者**: CYouuu  
**最後更新**: 2025-07-21

此資料夾包含專案的核心業務邏輯模組，採用模組化設計，各司其職：

## 📁 模組結構

```
modules/
├── asr/                    # 語音辨識 (Faster-Whisper)
│   ├── asr_model.py           # 模型載入與管理
│   ├── text_utils.py          # 文字處理工具
│   └── whisper_asr.py         # ASR 主流程
├── database/               # Weaviate V2 資料庫操作
│   ├── database.py            # DatabaseService V2 實作
│   └── init_v2_collections.py # V2 集合初始化
├── identification/         # 語者識別 (ECAPA-TDNN)
│   └── VID_identify_v5.py     # 語者識別引擎 V2
├── management/             # 語者管理
│   └── VID_manager.py         # 語者與聲紋管理
└── separation/             # 語者分離 (Sepformer/ConvTasNet)
    ├── separator.py           # 分離主流程
    └── RSS_3_v1.py           # 即時分離系統
```

## 🚀 主要功能模組

### 🎤 ASR (語音辨識)
- **技術**: Faster-Whisper
- **功能**: 語音轉文字、逐詞時間戳、信心值
- **支援**: GPU/CPU 動態切換、多語言辨識
- **配置**: 使用 `constants.py` 中的模型參數

### 🧠 Database (資料庫)
- **技術**: Weaviate V2 向量資料庫
- **功能**: 語者/聲紋 CRUD、向量相似度搜尋
- **特色**: 雙 ID 系統 (UUID + 序號ID)、單例模式
- **新特性**: V2 資料結構、時間欄位標準化

### 🗣 Identification (語者識別)
- **技術**: SpeechBrain ECAPA-TDNN
- **功能**: 聲紋提取、語者比對、自動更新
- **閾值**: 使用 `constants.py` 定義的識別閾值
- **輸出**: 192維聲紋向量

### 👥 Management (語者管理)
- **功能**: 語者資料管理、聲紋轉移、改名
- **整合**: 與資料庫模組緊密配合
- **API**: 提供完整的管理介面

### 🎭 Separation (語者分離)
- **技術**: SpeechBrain Sepformer (2人) / ConvTasNet (3人)
- **功能**: 多人語音分離、即時處理
- **配置**: 動態模型選擇、音訊品質優化

## 💻 使用範例

### 語音辨識
```python
from modules.asr.whisper_asr import WhisperASR
from utils.constants import DEFAULT_WHISPER_MODEL

asr = WhisperASR(model_name=DEFAULT_WHISPER_MODEL, gpu=True)
text, confidence, words = asr.transcribe("audio.wav")
```

### 語者識別
```python
from modules.identification.VID_identify_v5 import SpeakerIdentifier

identifier = SpeakerIdentifier()
speaker_name, distance = identifier.process_audio_file("audio.wav")
```

### 資料庫操作
```python
from modules.database.database import DatabaseService

db = DatabaseService()
speakers = db.list_all_speakers()
speaker = db.get_speaker_by_id(1)  # V2: 支援序號ID查詢
```

### 語者分離
```python
from modules.separation.separator import AudioSeparator
from utils.constants import DEFAULT_SEPARATION_MODEL

separator = AudioSeparator(model_type=DEFAULT_SEPARATION_MODEL)
separated_files = separator.separate_and_save(audio_tensor, output_dir)
```

## ⚙️ 配置整合

### 環境變數 (env_config.py)
```python
from utils.env_config import WEAVIATE_HOST, MODELS_BASE_DIR, FORCE_CPU
```

### 應用常數 (constants.py)  
```python
from utils.constants import (
    THRESHOLD_LOW, THRESHOLD_UPDATE, THRESHOLD_NEW,
    DEFAULT_WHISPER_MODEL, SPEECHBRAIN_SPEAKER_MODEL
)
```

## 🔧 V2 版本更新

### 重大變更
1. **統一配置管理**: 所有模組使用環境變數和常數系統
2. **V2 資料庫結構**: 升級為 Weaviate V2，不相容於 V1
3. **雙 ID 系統**: 支援 UUID 和序號 ID 雙重識別  
4. **模型路徑統一**: 使用 `get_model_save_dir()` 管理模型路徑
5. **錯誤處理增強**: 統一的日誌系統和異常處理

### 遷移指南
1. 更新配置檔案 (`.env` 和 `constants.py`)
2. 初始化 V2 資料庫集合
3. 更新匯入路徑以使用新的配置系統
4. 測試各模組功能確保正常運作

## 🧪 測試

### 模組測試
```bash
# ASR 測試
python examples/test_asr.py

# 語者識別測試  
python examples/test_modules.py

# 語者 API 測試
python examples/test_speaker_api.py
```

### 整合測試
```bash
# 完整流程測試
python examples/run_orchestrator.py

# API 模型測試
python examples/test_api_models.py
```

## 📚 詳細文檔

各模組都有獨立的 README 檔案提供詳細說明：

- [asr/README.md](asr/README.md) - 語音辨識模組
- [database/README.md](database/README.md) - 資料庫模組 
- [identification/README.md](identification/README.md) - 語者識別模組
- [management/README.md](management/README.md) - 語者管理模組
- [separation/README.md](separation/README.md) - 語者分離模組

## 🔗 相關資源

- [CONFIG_README.md](../CONFIG_README.md) - 配置系統說明
- [pipelines/README.md](../pipelines/README.md) - 處理流程文檔
- [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) - API 完整文檔
