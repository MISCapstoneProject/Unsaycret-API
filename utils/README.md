# utils

共用工具，例如集中式 `get_logger` 函式，以及資料同步腳本。
基本使用：
```python
from utils import get_logger
log = get_logger(__name__)
log.info("hello")
```
