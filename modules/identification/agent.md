# Agent 指南: identification

`SpeakerIdentifier` 處理音檔並回傳辨識結果，內部自動更新聲紋。

```python
from modules.identification import SpeakerIdentifier
idtf = SpeakerIdentifier()
spk, name, dist = idtf.process_audio_file('voice.wav')
```
