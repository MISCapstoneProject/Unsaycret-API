# Agent 指南: asr

- `load_model` 載入 Faster-Whisper 模型
- `merge_char_to_word` 將字級時間戳合併成詞級
- `WhisperASR` 提供單檔或資料夾轉錄

```python
from modules.asr import WhisperASR
asr = WhisperASR(gpu=True)
text, conf, words = asr.transcribe('a.wav')
```
