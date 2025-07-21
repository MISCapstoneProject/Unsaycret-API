# Utils (工具) 模組

**版本**: v0.4.0
**最後更新**: 2025-07-21

共用工具，包含環境配置、日誌系統、常數定義等。
提供集中式 `get_logger` 函式與配置管理工具。
基本使用：
```python
from utils import get_logger
log = get_logger(__name__)
log.info("hello")
```
