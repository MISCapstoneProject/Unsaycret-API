# Agent 指南: modules

此目錄集合多個核心子模組，供其他程式匯入使用。
主要子模組與入口：

- **asr** : `WhisperASR`, `load_model`, `merge_char_to_word`
- **database** : `DatabaseService`
- **identification** : `SpeakerIdentifier`
- **management** : `SpeakerManager`
- **separation** : `AudioSeparator`

匯入範例：
```python
from modules.identification import SpeakerIdentifier
```
