# ASR (語音辨識) 模組

**版本**: v0.4.0  
**作者**: Gino  
**最後更新者**: CYouuu  
**最後更新**: 2025-07-21

ASR (Automatic Speech Recognition) 模組負責將音檔轉換成文字，並提供逐詞時間戳與信心值。採用 Faster-Whisper 技術，支援 GPU/CPU 動態切換。

## 📁 模組結構

```
modules/asr/
├── asr_model.py        # Faster-Whisper 模型載入與管理
├── text_utils.py       # 文字處理與時間戳合併工具
├── whisper_asr.py      # ASR 主流程，單檔與批次處理
└── README.md           # 本文檔
```

## 🚀 主要功能

- **語音轉文字**: 使用 Faster-Whisper 進行高精度轉錄
- **逐詞時間戳**: 提供詞級別的時間標記
- **信心值評估**: 每個詞都有對應的信心分數
- **多語言支援**: 自動語言偵測或指定語言
- **GPU/CPU 切換**: 根據硬體環境動態選擇
- **批次處理**: 支援目錄級別的批次轉錄

## ⚙️ 配置系統 (V2)

### 模型配置 (constants.py)
```python
# 模型名稱常數
DEFAULT_WHISPER_MODEL = "medium"           # 預設模型
DEFAULT_WHISPER_BEAM_SIZE = 5             # 預設 beam size
WHISPER_MODEL_CACHE_DIR = "models/faster-whisper"  # 模型快取目錄

# 模型對照表
WHISPER_MODEL_MAP = {
    "tiny": "guillaumekln/faster-whisper-tiny",
    "base": "guillaumekln/faster-whisper-base", 
    "small": "guillaumekln/faster-whisper-small",
    "medium": "guillaumekln/faster-whisper-medium",
    "large-v2": "guillaumekln/faster-whisper-large-v2",
    "large-v3": "guillaumekln/faster-whisper-large-v3",
}
```

### 環境變數 (.env)
```bash
MODELS_BASE_DIR=./models          # 模型基礎目錄
FORCE_CPU=false                   # 強制使用 CPU
```

## 💻 使用方式

### 基本使用
```python
from modules.asr.whisper_asr import WhisperASR
from utils.constants import DEFAULT_WHISPER_MODEL

# 初始化 ASR (使用預設配置)
asr = WhisperASR(gpu=True)  # 自動使用 DEFAULT_WHISPER_MODEL

# 指定模型
asr = WhisperASR(model_name="large-v2", gpu=True, beam=8)

# 單檔轉錄
text, confidence, words = asr.transcribe("audio.wav")
print(f"文字: {text}")
print(f"平均信心值: {confidence:.3f}")
```

### 批次處理
```python
# 批次轉錄目錄下所有 wav 檔案
json_path = asr.transcribe_dir("audio_folder", "output_id")
print(f"結果儲存於: {json_path}")
```

### 詞級時間戳
```python
text, confidence, words = asr.transcribe("audio.wav")

for word_info in words:
    print(f"詞: {word_info['word']}")
    print(f"開始時間: {word_info['start']:.2f}s")
    print(f"結束時間: {word_info['end']:.2f}s") 
    print(f"信心值: {word_info['probability']:.3f}")
    print("-" * 20)
```

## 🔧 模型管理 (asr_model.py)

### 模型載入
```python
from modules.asr.asr_model import load_model
from utils.constants import WHISPER_MODEL_CACHE_DIR

# 載入模型（使用常數配置）
model = load_model(
    model_name="medium",           # 或使用 DEFAULT_WHISPER_MODEL
    gpu=True,
    cache=WHISPER_MODEL_CACHE_DIR  # 使用常數定義的快取目錄
)
```

### 支援的模型
- `tiny`: 最快速，精度較低
- `base`: 平衡速度與精度
- `small`: 良好的精度  
- `medium`: 推薦使用，精度與速度平衡
- `large-v2`: 高精度，需要更多計算資源
- `large-v3`: 最新版本，最高精度

### 自動設備選擇
```python
# 系統會自動根據硬體選擇最佳設定：
# GPU 可用: device="cuda", compute_type="float16"
# CPU 模式: device="cpu", compute_type="int8"
```

## 📝 文字處理 (text_utils.py)

### 字元到詞的時間戳合併
Whisper 預設回傳字元級時間戳，本模組使用結巴分詞將其合併為詞級：

```python
from modules.asr.text_utils import merge_char_to_word

# 字元級時間戳 (Whisper 原始輸出)
char_words = [
    {"start": 0.0, "end": 0.3, "word": "你", "probability": 0.95},
    {"start": 0.3, "end": 0.6, "word": "好", "probability": 0.92}
]

# 合併為詞級時間戳
word_level = merge_char_to_word("你好", char_words)
# 結果: [{"start": 0.0, "end": 0.6, "word": "你好", "probability": 0.935}]
```

### 依賴套件
```bash
# 優先使用 jieba-fast (更快)
pip install jieba-fast

# 備用 jieba
pip install jieba
```

## 📊 輸出格式

### 單檔轉錄輸出
```python
# transcribe() 回傳格式
text = "這是一段測試語音"                    # str: 完整轉錄文字
confidence = 0.852                          # float: 平均信心值
words = [                                   # List[Dict]: 詞級時間戳
    {
        "word": "這是",
        "start": 0.0,
        "end": 0.4, 
        "probability": 0.89
    },
    {
        "word": "一段",
        "start": 0.4,
        "end": 0.8,
        "probability": 0.91
    },
    ...
]
```

### 批次處理輸出 JSON
```json
{
    "identity": "speaker_001",
    "timestamp": "2025-07-21 10:30:45",
    "results": {
        "audio1.wav": {
            "text": "轉錄文字內容",
            "confidence": 0.876,
            "words": [...]
        },
        "audio2.wav": {
            "text": "另一段語音",
            "confidence": 0.823,
            "words": [...]
        }
    },
    "summary": {
        "total_files": 2,
        "avg_confidence": 0.849,
        "processing_time": "12.34s"
    }
}
```

## 🔄 與其他模組整合

### 與 Separation 模組
```python
# 語音分離後的多軌道轉錄
from modules.separation.separator import SeparationService
from modules.asr.whisper_asr import WhisperASR

separator = SeparationService()
asr = WhisperASR(model_name="medium", gpu=True)

# 1. 分離多說話者語音
separated_files = separator.separate("mixed_audio.wav", "output_dir")

# 2. 分別轉錄每個軌道
for track_file in separated_files:
    text, confidence, words = asr.transcribe(track_file)
    print(f"{track_file}: {text} (信心值: {confidence:.3f})")
```

### 與 Database 模組 (V2)
```python
# 轉錄結果儲存至 Weaviate V2
from modules.database.database import DatabaseService
from modules.asr.whisper_asr import WhisperASR

db = DatabaseService()
asr = WhisperASR()

text, confidence, words = asr.transcribe("speaker.wav")

# V2 schema 儲存語音轉錄記錄
transcription_data = {
    "text": text,
    "confidence": confidence, 
    "word_timestamps": words,
    "audio_filename": "speaker.wav",
    "transcription_timestamp": datetime.now()
}
# 可與 Speaker 資料建立關聯
```

## 🛠️ 效能調整

### 模型選擇建議
- **開發/測試**: `tiny` 或 `base` (快速驗證)
- **生產環境**: `medium` (推薦，平衡精度與速度)
- **高精度需求**: `large-v2` 或 `large-v3`

### GPU 記憶體優化
```python
# 大檔案處理時的記憶體管理
asr = WhisperASR(model_name="medium", gpu=True)

# 分段處理長音檔 (例如 >30分鐘)
def transcribe_long_audio(audio_path, chunk_duration=300):  # 5分鐘分段
    # 實作音檔切割與合併邏輯
    pass
```

### 批次處理最佳化
```python
# 大量檔案處理時的資源管理
import gc
import torch

def batch_transcribe_optimized(file_list):
    asr = WhisperASR(model_name="medium", gpu=True)
    
    for i, audio_file in enumerate(file_list):
        text, conf, words = asr.transcribe(audio_file)
        # 處理結果...
        
        # 每 10 個檔案清理一次記憶體
        if i % 10 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
```

## 🐛 常見問題

### Q: 模型下載失敗
```bash
# 手動下載模型到快取目錄
mkdir -p models/faster-whisper
# 檢查網路連線，或使用代理
```

### Q: GPU 記憶體不足
```python
# 強制使用 CPU
asr = WhisperASR(model_name="medium", gpu=False)

# 或設定環境變數
export FORCE_CPU=true
```

### Q: 中文分詞效果不佳
```python
# 嘗試不同的分詞模式
# 在 text_utils.py 中可調整 jieba.cut() 參數
```

## 📚 相關文檔

- [主專案 README](../../README.md)
- [Database V2 文檔](../database/README.md)  
- [Separation 模組](../separation/README.md)
- [API 接口文檔](../../api/README.md)
- [配置系統說明](../../utils/README.md)

## 📝 文字處理 (text_utils.py)

### 字元到詞的時間戳合併
Whisper 預設回傳字元級時間戳，本模組使用結巴分詞將其合併為詞級：

```python
from modules.asr.text_utils import merge_char_to_word

# 字元級時間戳 (Whisper 原始輸出)
char_words = [
    {"start": 0.0, "end": 0.3, "word": "你", "probability": 0.95},
    {"start": 0.3, "end": 0.6, "word": "好", "probability": 0.92}
]

# 合併為詞級時間戳
word_level = merge_char_to_word("你好", char_words)
# 結果: [{"start": 0.0, "end": 0.6, "word": "你好", "probability": 0.935}]
```

### 依賴套件
```bash
# 優先使用 jieba-fast (更快)
pip install jieba-fast

# 備用 jieba
pip install jieba
```

char_words：Whisper 輸出的每個字詞時間戳清單，每項包含 start、end、word、probability。

回傳值：詞級合併結果，每項包含 word、start、end、probability。

本模組可讓輸出結果更適合用來做後續的字幕呈現、語意分析、或資料庫比對。




🔹 whisper_asr.py — ASR 核心封裝與流程控制
此檔是 asr_model 與 text_utils 的整合封裝，提供兩大功能：

transcribe：處理單一音檔，回傳轉寫結果與信心值。

transcribe_dir：批次處理資料夾內所有 .wav 檔，並輸出 asr.json。

建立時會傳入 model_name、gpu 等參數，內部會自動呼叫 load_model 取得 Whisper 模型實例。

類別初始化

WhisperASR(model_name="medium", gpu=False, beam=5, lang="auto")
model_name：Whisper 模型大小。

gpu：是否啟用 GPU。

beam：Beam search 大小，數值越大結果越穩定但速度較慢。

lang：語言，可設為 "auto" 或指定語系（如 "zh"、"en"）。

主要方法


def transcribe(wav_path: str) -> tuple[str, float, list[dict]]
傳入單一 wav 檔路徑，回傳：

文字轉寫

平均信心值

詞級時間戳清單


def transcribe_dir(input_dir: str, output_id: str) -> str
處理整個資料夾內所有 .wav，轉寫後輸出為 data/{output_id}/asr.json，回傳輸出檔案路徑。