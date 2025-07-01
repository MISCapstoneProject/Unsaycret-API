# Agent 指南: separation

`AudioSeparator` 使用 SpeechBrain Sepformer 將混合語音分離，輸出分段檔案。

```python
from modules.separation import AudioSeparator
sep = AudioSeparator()
segments = sep.separate_and_save(waveform, 'out')
```
