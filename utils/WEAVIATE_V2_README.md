# Weaviate V2 集合管理

## 檔案結構

### `utils/init_v2_collections.py` - 核心模組
**用途**：負責建立和管理 Weaviate V2 資料庫結構
**功能**：
- 建立 4 個正規化集合（Speaker, Session, SpeechLog, VoicePrint）
- 配置向量化策略
- 提供模組化的集合管理功能

**使用方法**：
```python
# 作為模組導入使用
from utils.init_v2_collections import ensure_weaviate_v2_collections, WeaviateV2CollectionManager

# 確保集合存在
success = ensure_weaviate_v2_collections(host="localhost", port=8080)

# 使用管理器
with WeaviateV2CollectionManager("localhost", 8080) as manager:
    # 檢查集合狀態
    status = manager.verify_v2_collections()
    print(status)
```

**命令列執行**：
```bash
# 建立集合
python -m utils.init_v2_collections

# 指定主機和端口
python -m utils.init_v2_collections --host localhost --port 8080
```

### `utils/test_init_v2_collections.py` - 測試模組
**用途**：提供完整的測試和驗證功能
**功能**：
- 插入測試資料
- 執行複雜查詢驗證
- 語義搜尋測試
- 關聯性測試

**使用方法**：
```bash
# 執行完整測試套件（包含集合建立 + 測試資料 + 驗證）
python -m utils.test_init_v2_collections

# 跳過集合建立（假設已存在）
python -m utils.test_init_v2_collections --skip-setup

# 只執行測試驗證（不插入資料）
python -m utils.test_init_v2_collections --test-only

# 指定主機和端口
python -m utils.test_init_v2_collections --host localhost --port 8080
```

## 資料庫結構

### 集合設計
1. **Speaker V2**: 說話者主檔（包含 speaker_id INT）
2. **Session**: 對話場景（取代 Meeting）
3. **SpeechLog**: 語音記錄（正規化）
4. **VoicePrint V2**: 聲紋特徵庫（改進版）

### 向量化策略
- **Session**: `title`, `summary` 可語義搜尋
- **SpeechLog**: `content` 可語義搜尋
- **Speaker, VoicePrint**: 無向量化（關聯查詢即可）

## 重要警告
⚠️ 本次重構將大幅改變資料庫結構，會導致現有資料不相容！
執行前請務必備份現有 Weaviate 資料庫。

## 工作流程建議

1. **初次設置**：
   ```bash
   python -m utils.init_v2_collections
   ```

2. **驗證和測試**：
   ```bash
   python -m utils.test_init_v2_collections
   ```

3. **日常使用**：
   ```python
   from utils.init_v2_collections import WeaviateV2CollectionManager
   # 在您的應用程式中使用管理器
   ```
