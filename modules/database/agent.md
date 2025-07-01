# Agent 指南: database

集中管理 Weaviate 資料庫存取，使用 `DatabaseService` 單例。

```python
from modules.database import DatabaseService
DB = DatabaseService()
print(DB.list_all_speakers())
```
