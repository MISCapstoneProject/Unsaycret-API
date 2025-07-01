# modules 模組總覽

此資料夾收錄專案的核心功能模組，依照功能區分多個子模組：

- `asr/` 語音辨識（Whisper）
- `database/` Weaviate 資料庫介面
- `identification/` 語者辨識
- `management/` 資料管理工具
- `separation/` 語者分離

各子目錄皆有獨立 README 介紹具體 API。匯入範例：

```python
from modules.asr import WhisperASR
```
