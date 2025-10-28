# WebSocket 停止信號測試指南

## 修復說明

### 問題
前端發送停止信號後，後端無法立即響應，需要手動強制結束伺服器。

### 根本原因
1. **主循環阻塞**：先處理 `result_q.get(timeout=0.1)`，再處理 `ws.receive()`，是**循序執行**
2. **佇列積壓**：當 `result_q` 一直有資料時，`ws.receive()` 被延遲執行
3. **停止信號被忽略**：pipeline 持續處理片段，停止信號在 WebSocket 接收佇列中等待

### 修復方案

#### 1. **改用非阻塞並行處理**
```python
# 舊的循序處理
seg = result_q.get(timeout=0.1)  # 阻塞 0.1 秒
msg = await ws.receive()          # 再阻塞等待

# 新的並行處理
seg = result_q.get(timeout=0.01)  # 只阻塞 0.01 秒
if ws_receive_task.done():        # 非阻塞檢查
    msg = ws_receive_task.result()
```

#### 2. **立即清空音訊佇列**
```python
if t == "stop":
    # 清空所有待處理的音訊
    while not raw_q.empty():
        raw_q.get_nowait()
    # 發送結束標記
    raw_q.put_nowait(b"")
```

#### 3. **使用 asyncio.create_task**
- WebSocket 接收變成背景任務，不阻塞主循環
- 每次循環都檢查任務是否完成（非阻塞）
- 主循環添加 `await asyncio.sleep(0.001)` 避免 CPU 空轉

### 關鍵改進

| 項目 | 修改前 | 修改後 |
|------|--------|--------|
| 結果佇列 timeout | 0.1 秒 | 0.01 秒 |
| WebSocket 接收方式 | 阻塞等待 | 非阻塞檢查 |
| 停止信號響應 | 需等待當前循環完成 | 立即處理 |
| 音訊佇列處理 | 繼續處理 | 立即清空 |
| CPU 使用 | 可能空轉 | 休眠 1ms |

## 測試步驟

### 測試 1: 立即停止
1. 啟動錄音
2. **立即**點擊停止（< 1 秒）
3. **預期結果**：
   - 看到 `📝 收到文字訊息: 'stop'`
   - 看到 `🛑 收到停止信號，開始優雅關閉`
   - 看到 `🧹 已清空音訊佇列`
   - 1-2 秒內完全停止

### 測試 2: 處理中停止
1. 啟動錄音
2. 等待 5-10 秒（讓系統處理幾個片段）
3. 點擊停止
4. **預期結果**：
   - 立即看到停止相關日誌
   - 不再處理新的音訊片段
   - 完成剩餘片段處理後停止

### 測試 3: 無聲音停止
1. 啟動錄音但保持靜音
2. 等待幾秒
3. 點擊停止
4. **預期結果**：
   - 即使沒有語音活動也能立即停止
   - 看到 `segment X 無 speaker wav` 的警告是正常的

### 測試 4: 異常斷線
1. 啟動錄音
2. 直接關閉瀏覽器 tab
3. **預期結果**：
   - 後端偵測到斷線：`🔌 前端主動斷線`
   - 3 秒內自動清理資源
   - 看到 `✅ 背景處理線程已正常結束`

## 關鍵日誌

正常停止流程應該看到：
```
[時間] [api.api] INFO: 📝 收到文字訊息: 'stop'
[時間] [api.api] INFO: 🛑 收到停止信號，開始優雅關閉
[時間] [api.api] INFO: 🧹 已清空音訊佇列
[時間] [api.api] INFO: 📊 已標記停止，將繼續處理剩餘結果
[時間] [pipelines.orchestrator] INFO: 🛑 recorder_from_queue: 收到結束標記
[時間] [pipelines.orchestrator] INFO: 🧹 開始清理 pipeline 資源
[時間] [pipelines.orchestrator] INFO: ✅ recorder 線程已結束
[時間] [api.api] INFO: ✅ 背景處理線程已正常結束
[時間] [api.api] INFO: 🏁 WebSocket 會話 {uuid} 完全結束
```

## 故障排除

### 如果還是沒有日誌
1. 檢查前端是否真的發送了 `"stop"` 文字訊息
2. 在瀏覽器開發者工具查看 WebSocket 面板
3. 確認 WebSocket 連線狀態

### 如果停止很慢（> 5 秒）
- 可能是 GPU 處理延遲
- 檢查 `max_workers` 設定
- 查看是否有片段處理卡住

### 如果資料遺失
- 檢查 `result_q` 是否被正確處理完
- 確認 SpeechLog 是否都已儲存
- 查看 Session 時間範圍是否更新

## 程式碼位置

- **API 主循環**：`api/api.py` 第 485-740 行
- **Pipeline 處理**：`pipelines/orchestrator.py` 第 505-640 行
- **停止信號處理**：`api/api.py` 第 652-670 行
