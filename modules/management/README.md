# modules/management

管理 Weaviate 中的語者與聲紋資料，提供 CLI 與程式介面。
核心類別 `SpeakerManager` 封裝了新增、修改與轉移聲紋等動作。

基本用法：
```python
from modules.management import SpeakerManager
mgr = SpeakerManager()
print(mgr.list_all_speakers())
```
