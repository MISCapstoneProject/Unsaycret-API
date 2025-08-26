# AS-Norm 語者識別正規化功能

## 概述

AS-Norm (Adaptive Score Normalization) 是一種語者識別系統中的分數正規化技術，旨在改善識別的穩定性和準確性，減少環境變異和條件差異對識別結果的影響。

## 功能特性

### 🔧 正規化技術

#### 1. T-Norm (Test Normalization)
- **原理**: 使用 impostor 語者的分數統計進行正規化
- **公式**: `(score - mean_impostor) / std_impostor`
- **適用**: 減少測試條件變異的影響

#### 2. Z-Norm (Zero Normalization)
- **原理**: 使用所有已知語者的分數統計進行正規化
- **公式**: `(score - mean_all) / std_all`
- **適用**: 標準化語者模型的分數分布

#### 3. S-Norm (Symmetric Normalization)
- **原理**: 結合 T-Norm 和 Z-Norm 的優點
- **公式**: `alpha * t_norm + (1-alpha) * z_norm`
- **適用**: 平衡測試和模型端的正規化效果

### ⚙️ 可配置參數

| 參數 | 說明 | 預設值 | 範圍 |
|------|------|--------|------|
| `ENABLE_AS_NORM` | AS-Norm 總開關 | `False` | `True/False` |
| `ENABLE_T_NORM` | T-Norm 開關 | `True` | `True/False` |
| `ENABLE_Z_NORM` | Z-Norm 開關 | `True` | `True/False` |
| `ENABLE_S_NORM` | S-Norm 開關 | `True` | `True/False` |
| `AS_NORM_COHORT_SIZE` | Cohort 大小 | `100` | `10-500` |
| `AS_NORM_TOP_K` | Top-K impostor | `10` | `5-50` |
| `AS_NORM_ALPHA` | S-Norm 權重 | `0.9` | `0.0-1.0` |

## 使用方法

### 基本使用

```python
from modules.identification.VID_identify_v5 import SpeakerIdentifier

# 創建語者識別器
identifier = SpeakerIdentifier()

# 啟用 AS-Norm
identifier.set_as_norm_enabled(True)

# 處理音檔（自動應用 AS-Norm）
result = identifier.process_audio_file("path/to/audio.wav")
```

### 自訂配置

```python
# 配置 AS-Norm 參數
identifier.configure_as_norm(
    t_norm=True,      # 啟用 T-Norm
    z_norm=True,      # 啟用 Z-Norm  
    s_norm=True,      # 啟用 S-Norm
    cohort_size=50,   # 使用 50 個 impostor 語者
    top_k=10,         # 使用前 10 個最相似的 impostor
    alpha=0.8         # S-Norm 權重參數
)
```

### 狀態查詢

```python
# 查看當前 AS-Norm 設定
status = identifier.get_as_norm_status()
print("AS-Norm 狀態:", status)
```

### 不同正規化方法測試

```python
# 僅使用 T-Norm
identifier.configure_as_norm(t_norm=True, z_norm=False, s_norm=False)

# 僅使用 Z-Norm  
identifier.configure_as_norm(t_norm=False, z_norm=True, s_norm=False)

# 使用 T-Norm + Z-Norm 組合
identifier.configure_as_norm(t_norm=True, z_norm=True, s_norm=False)

# 使用完整 S-Norm
identifier.configure_as_norm(t_norm=True, z_norm=True, s_norm=True, alpha=0.8)
```

## 測試腳本

我們提供了專門的測試腳本來評估 AS-Norm 功能：

```bash
python examples/test_as_norm.py
```

### 測試功能

1. **配置演示**: 展示不同 AS-Norm 配置的效果
2. **效能比較**: 比較正規化前後的識別結果
3. **參數調優**: 測試不同參數組合的影響
4. **統計分析**: 提供詳細的統計分析報告

## 配置文件控制

### constants.py 設定

```python
# AS-Norm 總開關
ENABLE_AS_NORM = False  # 改為 True 啟用

# 各種正規化方法開關
ENABLE_T_NORM = True
ENABLE_Z_NORM = True  
ENABLE_S_NORM = True

# 參數配置
AS_NORM_COHORT_SIZE = 100  # Cohort 大小
AS_NORM_TOP_K = 10         # Top-K impostor
AS_NORM_ALPHA = 0.9        # S-Norm 權重
```

### 環境變數控制

也可以透過環境變數控制（優先度較低）：

```bash
export ENABLE_AS_NORM=true
export AS_NORM_COHORT_SIZE=50
export AS_NORM_ALPHA=0.8
```

## 效能影響

### 計算開銷

- **T-Norm**: 輕微增加（需額外計算 impostor 分數）
- **Z-Norm**: 中等增加（需計算所有語者分數）
- **S-Norm**: 較高增加（結合 T-Norm 和 Z-Norm）

### 記憶體使用

- Cohort Size 越大，記憶體使用越多
- 建議根據系統資源調整 `AS_NORM_COHORT_SIZE`

### 準確性提升

- 在環境雜訊較多的情況下效果顯著
- 對於乾淨錄音環境提升有限
- 建議先在測試集上評估效果

## 最佳實踐

### 1. 參數調優

```python
# 開發/測試環境：較小的 cohort
identifier.configure_as_norm(cohort_size=30, top_k=5)

# 生產環境：較大的 cohort
identifier.configure_as_norm(cohort_size=100, top_k=15)

# 高精度需求：完整配置
identifier.configure_as_norm(cohort_size=200, top_k=20, alpha=0.85)
```

### 2. 資料集考量

- **小資料集** (< 50 語者): 使用較小的 `cohort_size`
- **大資料集** (> 500 語者): 可使用完整配置
- **多環境錄音**: 建議啟用完整 S-Norm

### 3. 效能平衡

```python
# 快速識別模式
identifier.set_as_norm_enabled(False)

# 平衡模式
identifier.configure_as_norm(t_norm=True, z_norm=False, s_norm=False, cohort_size=50)

# 高精度模式  
identifier.configure_as_norm(t_norm=True, z_norm=True, s_norm=True, cohort_size=100)
```

## 故障排除

### 常見問題

1. **導入錯誤**
   ```python
   # 確保正確導入
   from modules.identification.VID_identify_v5 import SpeakerIdentifier
   ```

2. **Weaviate 連接問題**
   ```bash
   # 確保 Weaviate 服務運行
   docker-compose -f weaviate_study/docker-compose.yml up -d
   ```

3. **記憶體不足**
   ```python
   # 減少 cohort_size
   identifier.configure_as_norm(cohort_size=30)
   ```

### 除錯模式

```python
# 啟用詳細輸出
identifier.set_verbose(True)

# 查看 AS-Norm 狀態
status = identifier.get_as_norm_status()
print("除錯資訊:", status)
```

## 技術細節

### 演算法流程

1. **向量比對**: 首先進行標準的向量相似度計算
2. **Impostor 選擇**: 隨機選擇非目標語者作為 impostor
3. **統計計算**: 計算 impostor 和全體語者的分數統計
4. **正規化應用**: 根據配置應用對應的正規化方法
5. **結果輸出**: 返回正規化後的相似度分數

### 資料流

```
音訊輸入 → 特徵提取 → 向量比對 → AS-Norm 正規化 → 最終分數
                                    ↑
                        Impostor 統計 + 全體統計
```

## 參考文獻

1. Reynolds, D.A. (2003). "Channel robust speaker verification via feature mapping"
2. Auckenthaler, R. et al. (2000). "Score normalization for text-independent speaker verification systems"
3. Sturim, D.E. et al. (2002). "Speaker adaptive cohort selection for Tnorm in text-independent speaker verification"

## 版本歷史

- **v1.0.0** (2025-08-25): 初始版本，支援 T-Norm、Z-Norm、S-Norm
- 未來計畫：增加更多正規化方法、效能最佳化

## 授權

本功能屬於 Unsaycret-API 專案的一部分，遵循專案的授權條款。
