# 🔬 VAD 參數詳細指南

## 📊 參數含義詳解

### 1. Energy Threshold (能量閾值)
**範圍：** -60 ~ -30 dBFS
**含義：** 音量門檻，決定多小的聲音被認為是「有聲音」

```
-30 dBFS  ████████████████ 很大聲才算 (嚴格)
-40 dBFS  ████████████     中等音量
-50 dBFS  ████████         小聲音也算  
-60 dBFS  ████             很小聲也算 (敏感)
```

### 2. ZCR (Zero Crossing Rate) - 過零率
**範圍：** 0.005 ~ 0.40
**含義：** 聲音波形的「變化頻率」指標

#### 🎵 不同聲音的 ZCR 特徵：
- **人聲語音：** 0.03 ~ 0.20 (適中變化)
- **純音/音樂：** 0.005 ~ 0.05 (規律變化) 
- **噪音/風聲：** 0.25 ~ 0.40 (快速變化)
- **靜音：** 接近 0 (無變化)

#### 🔬 技術原理：
ZCR 測量聲音波形「穿越零點」的頻率
```
波形示例：
語音:  /\  /\    /\  /\     ← ZCR ≈ 0.08 (適中)
噪音:  /\/\/\/\/\/\/\/\    ← ZCR ≈ 0.35 (很高)
純音:  /    \  /    \      ← ZCR ≈ 0.02 (很低)
```

### 3. 完整參數對照表

| 參數 | 範圍 | 嚴格模式 | 敏感模式 | 實際效果 |
|------|------|----------|----------|----------|
| **energy_threshold** | -60~-30 | -40 | -55 | 音量門檻 |
| **loud_threshold** | -50~-30 | -35 | -50 | 確認門檻 |
| **zcr_min** | 0.005~0.05 | 0.03 | 0.005 | 過濾單調聲 |
| **zcr_max** | 0.20~0.40 | 0.20 | 0.35 | 過濾嘈雜聲 |
| **min_voice_duration** | 0.1~0.8 | 0.4 | 0.15 | 最短語音 |
| **padding** | 0.03~0.1 | 0.03 | 0.08 | 安全邊距 |

## 🎯 預設配置解析

### Custom (自訂模式)
```python
{
    "energy_threshold": -45.0,    # 中等敏感度
    "loud_threshold": -44.0,      # 接近能量閾值
    "zcr_min": 0.03,              # 過濾單調聲音
    "zcr_max": 0.20,              # 過濾雜音
    "min_voice_duration": 0.4,    # 保留較長語音
    "padding": 0.04               # 最小保護邊距
}
```
**策略：** 平衡精確度和保留率，適合語音識別前處理

### Strict (嚴格模式) 
```python
{
    "energy_threshold": -40.0,    # 只要較大聲音
    "loud_threshold": -35.0,      # 確認門檻更高
    "zcr_min": 0.03,              # 標準語音範圍
    "zcr_max": 0.20,              # 不接受雜音
    "min_voice_duration": 0.4,    # 過濾短音
    "padding": 0.03               # 最小邊距
}
```
**策略：** 只保留清晰語音，大量移除雜音

### Sensitive (敏感模式)
```python
{
    "energy_threshold": -50.0,    # 小聲音也保留
    "loud_threshold": -45.0,      # 確認門檻較低
    "zcr_min": 0.01,              # 接受更單調聲音
    "zcr_max": 0.30,              # 接受一些雜音
    "min_voice_duration": 0.2,    # 保留短語音
    "padding": 0.05               # 更多保護
}
```
**策略：** 盡量保留所有可能的語音內容

### Phone (電話模式)
```python
{
    "energy_threshold": -52.0,    # 適應電話音質
    "loud_threshold": -47.0,      # 考慮壓縮失真
    "zcr_min": 0.015,             # 電話頻寬限制
    "zcr_max": 0.28,              # 允許壓縮雜音
    "min_voice_duration": 0.25,   # 電話對話節奏
    "padding": 0.08               # 更多保護
}
```
**策略：** 專門針對電話錄音的特殊音質調整

## 🛠️ 自訂參數建議

### 根據音檔品質調整：
- **高品質錄音室：** energy_threshold = -40, zcr_max = 0.15
- **手機錄音：** energy_threshold = -48, zcr_max = 0.25  
- **老舊錄音：** energy_threshold = -55, zcr_max = 0.35

### 根據使用目的調整：
- **語音識別前處理：** 使用 custom 或 strict
- **保留完整對話：** 使用 sensitive 或 ultra_sensitive
- **音樂人聲分離：** 使用 music 模式

### 常見問題與解決：
1. **切掉語音開頭：** 增加 padding 值
2. **保留太多雜音：** 降低 energy_threshold 或 zcr_max
3. **切掉短詞：** 降低 min_voice_duration
4. **漏掉小聲語音：** 降低 energy_threshold

## 💡 實戰調優步驟

1. **先選擇最接近的預設模式**
2. **測試一小段音檔**
3. **根據結果微調關鍵參數：**
   - 切太多 → 降低 energy_threshold
   - 留太多雜音 → 提高 energy_threshold  
   - 切掉開頭結尾 → 增加 padding
   - 短音被切 → 降低 min_voice_duration

## 🔬 進階技巧

### 參數相互影響：
- `energy_threshold` 和 `loud_threshold` 差距影響檢測穩定性
- `zcr_min/max` 範圍太窄會過度過濾
- `padding` 太大會保留過多靜音

### 最佳實踐：
```python
# 保守調整策略
base_config = VADConfig.get_config("sensitive")
base_config.update({
    "energy_threshold": base_config["energy_threshold"] - 2,  # 微調
    "padding": base_config["padding"] + 0.01                  # 增加保護
})
```
