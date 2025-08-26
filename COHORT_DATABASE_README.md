# AS-Norm Cohort 資料庫設計與使用指南

## 🎯 問題描述

在 AS-Norm（Adaptive Score Normalization）實作中，cohort（背景模型）的選擇至關重要。**cohort 應該是固定的、不會在實際辨識中出現的語音集合**，用來計算正規化統計量。

### ⚠️ 原實作的問題

之前的實作從主要的 `VoicePrint` collection 中隨機選取 impostor 作為 cohort，這會導致：

1. **背景模型污染**：新加入的真實說話者會被當作背景模型使用
2. **不穩定的正規化**：每次新增語者後，背景統計量會改變
3. **邏輯錯誤**：真實語者不應該作為其他語者的背景模型

## 🏗️ 解決方案：專門的 Cohort 資料庫

### 設計原則

1. **獨立性**：cohort 資料與實際語者資料完全分離
2. **穩定性**：cohort 集合固定，不會因為新增語者而改變
3. **專用性**：cohort 僅用於 AS-Norm 計算，不參與實際辨識

### 資料庫結構

```
CohortVoicePrint Collection:
├── Properties:
│   ├── create_time (DATE)       # 建立時間
│   ├── cohort_id (TEXT)         # 背景模型識別碼
│   ├── source_dataset (TEXT)    # 來源資料集
│   ├── gender (TEXT)            # 性別（可選）
│   ├── language (TEXT)          # 語言（可選）
│   └── description (TEXT)       # 描述
└── Vector: 語音嵌入向量 (192 維)
```

## 🔧 配置參數

在 `utils/constants.py` 中新增：

```python
# AS-Norm Cohort 資料庫配置
AS_NORM_COHORT_COLLECTION = "CohortVoicePrint"  # 專用的背景模型資料庫
AS_NORM_USE_DEDICATED_COHORT = True  # 是否使用專門的cohort資料庫
```

## 📊 實作細節

### 1. Collection 建立

- 擴展 `WeaviateCollectionManager` 類別
- 新增 `create_cohort_voiceprint_collection()` 方法
- 在 `create_all_collections()` 中包含 cohort collection

### 2. AS-Norm 處理器修改

```python
# 根據配置選擇資料來源
if AS_NORM_USE_DEDICATED_COHORT:
    collection_name = AS_NORM_COHORT_COLLECTION
    # 檢查專門的cohort collection是否存在
    if not self.database.client.collections.exists(collection_name):
        logger.warning(f"專門的cohort collection '{collection_name}' 不存在，回退到主資料庫")
        collection_name = "VoicePrint"
        use_where_filter = True
    else:
        use_where_filter = False  # cohort資料庫中沒有目標語者，不需要過濾
else:
    collection_name = "VoicePrint"
    use_where_filter = True
```

### 3. 自動回退機制

如果專門的 cohort collection 不存在，系統會自動回退到從主資料庫中選取，確保向後兼容性。

## 🛠️ 使用方法

### 1. 初始化 Cohort Collection

```bash
# 建立 cohort collection
python examples/cohort_manager.py --action init
```

### 2. 生成測試用的 Cohort 資料

```bash
# 生成 100 個隨機 cohort 嵌入向量
python examples/cohort_manager.py --action generate --count 100 --dim 192
```

### 3. 查看 Cohort 統計信息

```bash
# 檢查 cohort 資料庫狀態
python examples/cohort_manager.py --action stats
```

### 4. 清空 Cohort Collection

```bash
# 清空 cohort collection
python examples/cohort_manager.py --action clear
```

### 5. 在程式中使用

```python
from modules.identification.VID_identify_v5 import SpeakerIdentifier

# 建立語者辨識器（會自動根據配置使用 cohort 資料庫）
identifier = SpeakerIdentifier()

# 啟用 AS-Norm
identifier.set_as_norm_enabled(True)

# 正常進行語者辨識，AS-Norm 會使用專門的 cohort 資料
results = identifier.identify_speaker(audio_data)
```

## 📈 效果與優勢

### 1. 背景模型隔離

- ✅ 新加入的語者不會影響背景模型
- ✅ cohort 集合保持穩定
- ✅ 正規化結果更可靠

### 2. 性能提升

- ✅ 不需要過濾目標語者（因為 cohort 中本來就沒有）
- ✅ 查詢更簡單高效
- ✅ 可預先準備專門的背景語音

### 3. 靈活配置

- ✅ 可開關專門 cohort 資料庫功能
- ✅ 自動回退到原有邏輯
- ✅ 向後兼容

## 🔄 遷移步驟

### 1. 現有系統升級

```bash
# 1. 更新程式碼（已完成）
git pull

# 2. 初始化 cohort collection
python examples/cohort_manager.py --action init

# 3. 生成測試用 cohort 資料
python examples/cohort_manager.py --action generate --count 100

# 4. 驗證配置
python examples/cohort_manager.py --action stats
```

### 2. 配置開關

```python
# 啟用專門的 cohort 資料庫
AS_NORM_USE_DEDICATED_COHORT = True

# 如果遇到問題，可暫時回退
AS_NORM_USE_DEDICATED_COHORT = False
```

## 🚀 未來擴展

### 1. 真實 Cohort 資料

可以使用以下來源建立更真實的 cohort：

- **公開語音資料集**：VoxCeleb, LibriSpeech 等
- **多語言語音**：涵蓋不同語言和口音
- **性別平衡**：確保男女聲比例適當

### 2. Cohort 品質控制

- **向量品質檢查**：確保嵌入向量的品質
- **多樣性分析**：保證 cohort 的多樣性
- **定期更新**：根據需求更新 cohort 集合

### 3. 動態 Cohort 選擇

- **條件篩選**：根據語言、性別等條件選擇適當的 cohort
- **自適應大小**：根據系統負載動態調整 cohort 大小
- **分層 Cohort**：建立不同級別的 cohort 集合

## 📝 注意事項

1. **資料隱私**：確保 cohort 資料不包含敏感信息
2. **版權合規**：使用公開資料集時注意版權問題
3. **資源管理**：cohort 資料會占用額外的儲存空間
4. **維護計劃**：定期檢查和更新 cohort 資料品質

## 🔍 測試驗證

使用 `examples/test_as_norm.py` 來驗證新的 cohort 資料庫功能：

```bash
# 測試 AS-Norm 功能（會自動使用新的 cohort 資料庫）
python examples/test_as_norm.py
```

這個設計解決了您提出的重要問題，確保背景模型不會被新加入的語者污染，提供更穩定可靠的 AS-Norm 正規化效果。
