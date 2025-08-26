# AS-Norm Cohort 資料庫管理模組

專門用於管理 AS-Norm 背景模型資料庫的完整解決方案。

## 📁 檔案結構

```
modules/database/
├── cohort_manager.py     # 核心 Cohort 管理模組
└── README.md            # 本文件

examples/
├── cohort_cli.py        # 快速命令列工具
└── cohort_manager.py    # 舊版本（已廢棄，自動重導向）

COHORT_DATABASE_README.md # 詳細技術文件
```

## 🚀 快速開始

### 1. 初始化 Cohort 資料庫

```bash
# 建立 cohort collection
python examples/cohort_cli.py init
```

### 2. 導入音頻資料

```bash
# 從資料夾導入音頻檔案
python examples/cohort_cli.py import /path/to/cohort/audio

# 指定資料集名稱
python examples/cohort_cli.py import /path/to/cohort/audio my_cohort_dataset
```

### 3. 查看統計信息

```bash
# 檢查資料庫狀態
python examples/cohort_cli.py stats
```

### 4. 重置資料庫

```bash
# 清空所有 cohort 資料
python examples/cohort_cli.py reset
```

## 🛠️ 程式化使用

### 基本使用

```python
from modules.database.cohort_manager import CohortDatabaseManager

# 建立管理器
manager = CohortDatabaseManager()

try:
    # 初始化資料庫
    manager.initialize_cohort_collection()
    
    # 導入音頻資料夾
    results = manager.import_audio_folder(
        folder_path="/path/to/cohort/audio",
        source_dataset="my_cohort",
        chunk_length=3.0,    # 3秒切片
        overlap=0.5,         # 50%重疊
        metadata={
            "language": "zh-TW",
            "gender": "mixed"
        }
    )
    
    print(f"成功導入 {results['total_embeddings']} 個聲紋")
    
finally:
    manager.close()
```

### 高級功能

```python
# 單檔案導入
embeddings_count = manager.import_audio_file(
    audio_path="/path/to/audio.wav",
    source_dataset="single_file",
    metadata={"speaker_id": "cohort_001"}
)

# 獲取統計信息
stats = manager.get_cohort_statistics()
print(f"資料庫包含 {stats['total_count']} 個聲紋")

# 匯出資料庫信息
info_file = manager.export_cohort_info("cohort_info.json")
```

## ⚙️ 配置參數

### 切片設定

- **chunk_length**: 音頻切片長度（秒），預設 3.0
- **overlap**: 切片重疊比例（0-1），預設 0.5

### 音頻格式支援

- `.wav` - 首選格式
- `.mp3` - 常用格式
- `.flac` - 無損格式
- `.m4a`, `.aac` - Apple 格式
- `.ogg` - 開源格式

### 元數據欄位

- **cohort_id**: 唯一識別碼（自動生成）
- **source_dataset**: 來源資料集名稱
- **gender**: 語者性別（可選）
- **language**: 語音語言（預設 zh-TW）
- **description**: 描述信息

## 🎯 最佳實踐

### 1. Cohort 資料選擇

✅ **建議使用**：
- 公開語音資料集（VoxCeleb, LibriSpeech）
- 不會在實際系統中出現的語者
- 多樣化的語音特徵（不同性別、年齡、口音）

❌ **避免使用**：
- 系統中實際語者的聲音
- 品質不佳的音頻
- 過於相似的語音特徵

### 2. 資料庫維護

```bash
# 定期檢查資料庫狀態
python examples/cohort_cli.py stats

# 匯出備份信息
python -c "
from modules.database.cohort_manager import CohortDatabaseManager
manager = CohortDatabaseManager()
manager.export_cohort_info('backup_info.json')
manager.close()
"
```

### 3. 性能優化

- **批量導入**：一次處理整個資料夾而非單個檔案
- **適當切片**：3-5秒片段通常效果最好
- **合理重疊**：50%重疊提供更多樣本但增加儲存需求

## 🔧 故障排除

### 常見問題

1. **模組導入錯誤**
   ```bash
   ModuleNotFoundError: No module named 'modules'
   ```
   **解決**：確保在專案根目錄執行命令

2. **Weaviate 連接失敗**
   ```bash
   ConnectionError: Could not connect to Weaviate
   ```
   **解決**：檢查 Weaviate 服務是否運行

3. **音頻檔案讀取失敗**
   ```bash
   Warning: 提取聲紋失敗: audio.wav
   ```
   **解決**：檢查音頻格式和檔案完整性

### 調試模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 這樣可以看到詳細的處理過程
manager = CohortDatabaseManager()
```

## 📊 監控與分析

### 資料庫統計

```python
stats = manager.get_cohort_statistics()

# 檢查重要指標
print(f"總聲紋數: {stats['total_count']}")
print(f"資料集分佈: {stats['source_datasets']}")
print(f"性別分佈: {stats['genders']}")
print(f"語言分佈: {stats['languages']}")
```

### 品質評估

- 監控聲紋向量的分佈
- 檢查是否有異常值
- 確保資料多樣性

## 🔄 遷移指南

### 從舊版本遷移

如果您之前使用 `examples/cohort_manager.py`：

```bash
# 舊方式（已廢棄）
python examples/cohort_manager.py --action init

# 新方式
python examples/cohort_cli.py init
```

### 重要變更

1. **模組位置**：從 `examples/` 移至 `modules/database/`
2. **命令格式**：從 `--action` 參數改為直接命令
3. **功能增強**：新增音頻處理和批量導入功能

## 📚 相關文件

- [COHORT_DATABASE_README.md](../COHORT_DATABASE_README.md) - 技術設計文件
- [AS_NORM_README.md](../AS_NORM_README.md) - AS-Norm 功能說明
- [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) - API 文件

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request 來改進此模組。

### 開發環境設置

```bash
# 安裝開發依賴
pip install -r requirements-cpu.txt

# 執行測試
python -m pytest tests/

# 程式碼檢查
python -m py_compile modules/database/cohort_manager.py
```
