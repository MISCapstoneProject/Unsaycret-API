## 📁 modules/asr（語音辨識模組）
 
**作者**：Gino
**最後更新**：2025-05-19  

此資料夾負責將音檔轉成文字，並提供詞級時間戳。主要檔案包括：

- **asr_model.py**：載入並快取 Faster-Whisper 模型  
- **text_utils.py**：將字元層級時間戳合併成詞級時間戳  
- **whisper_asr.py**：封裝單檔與批次轉寫流程，輸出 JSON

---

### 安裝與初始化

```bash
# 確認已安裝 faster-whisper
pip install faster-whisper

# 也請先安裝 jieba_fast 或 jieba
pip install jieba_fast      # 若無法安裝，pip install jieba


from modules.asr.asr_model import load_model

# 載入 medium 模型到 GPU，權重快取在 models/faster-whisper
model = load_model(model_name="medium", gpu=True, cache="models/faster-whisper")


主要函式
def load_model(model_name: str = "medium", gpu: bool = False, cache: str = "models/faster-whisper") -> WhisperModel

model_name：可使用 tiny、base、small、medium、large-v2、large-v3，或完整 repo 名稱。

gpu：是否啟用 GPU，決定載入的裝置與精度。

cache：模型快取目錄，避免重複下載。

此模組單純負責回傳 WhisperModel 實例，供其他模組使用。




text_utils.py — 時間戳合併工具
Whisper 回傳的是字元層級的時間戳（例如「你」的開始時間是 0.0 秒，「好」是 0.3 秒），但人類閱讀與實務應用常需要「詞級」時間範圍，因此本模組使用結巴分詞（jieba 或 jieba_fast）來將連續字元合併為詞。

主要函式

def merge_char_to_word(full_txt: str, char_words: List[Dict]) -> List[Dict]
full_txt：ASR 全部轉出的文字內容。

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