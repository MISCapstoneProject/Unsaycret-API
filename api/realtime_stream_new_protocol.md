# 即時語音串流功能 - 開發教學與訊息格式規格文件

## 1. 概要

本功能使用 **WebSocket** 建立「雙向即時連線」，讓前端可以持續傳送語音片段，後端即時處理語音並回傳字幕。

* **前端 → 後端**：傳輸音訊資料、控制訊號 (stop)。
* **後端 → 前端**：回傳逐步產生的字幕結果（包含 segmentId、speakerId、語者辨識距離等）。

---

## 2. WebSocket 端點

```
ws://<server-host>:<port>/ws/stream?session=<UUID>
```

* `session` (query param) 必填，用於辨識會議 Session UUID。

---

## 3. 訊息格式 (Message Protocol Spec)

所有訊息均為 **JSON 格式**，且必須包含欄位 `type` 以區分用途。

---

### 3.1 前端 → 後端

#### (A) 傳送音訊片段

```json
{
  "type": "audio",
  "timestamp": 1694581200,       // 選填，音訊開始時間 (epoch 秒)
  "data": "<base64音訊或ArrayBuffer>"  // 音訊資料
}
```

#### (B) 傳送停止訊號

```json
{
  "type": "stop"
}
```

---

### 3.2 後端 → 前端

#### (A) 傳送字幕結果

```json
{
  "type": "subtitle",
  "segmentId": "seg_1",         // 字幕段落 ID (唯一值)
  "speakerId": "uuid",          // 語者唯一 ID
  "speakerName": "n1",          // 語者名稱 (ex: n1, n2...)
  "distance": 0.2,              // 語者辨識距離 (越小越準確)
  "text": "今天心情很好",          // 字幕內容
  "isFinal": true               // 是否定稿 (true=不再更新)
}
```

---

## 4. 前端開發教學

### 建立連線

```js
const ws = new WebSocket("ws://localhost:8000/ws/stream?session=1234");

ws.onopen = () => {
  console.log("WebSocket connected");
};

ws.onclose = () => {
  console.log("WebSocket disconnected");
};
```

### 傳送音訊

```js
function sendAudio(base64Audio) {
  ws.send(JSON.stringify({
    type: "audio",
    timestamp: Date.now(),
    data: base64Audio
  }));
}
```

### 停止錄音

```js
function stopStream() {
  ws.send(JSON.stringify({ type: "stop" }));
}
```

### 接收字幕

```js
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === "subtitle") {
    // 根據 segmentId 更新字幕
    const existing = document.getElementById(msg.segmentId);
    if (existing) {
      existing.innerText = `[${msg.speakerName}] ${msg.text}` + (msg.isFinal ? " ✅" : "");
    } else {
      const el = document.createElement("div");
      el.id = msg.segmentId;
      el.innerText = `[${msg.speakerName}] ${msg.text}`;
      document.getElementById("subtitles").appendChild(el);
    }
  }
};
```

---

## 5. 後端處理流程

1. **接收前端音訊 (`type: audio`)** → 放入 `raw_q` 佇列
2. **STT pipeline** → 持續讀取 `raw_q`，產生字幕，放入 `result_q`
3. **取出 `result_q` 的字幕結果** → 包裝成 `subtitle` JSON，傳給前端
4. **接收停止訊號 (`type: stop`)** → 停止 pipeline，傳送最後字幕，關閉 WebSocket

---

## 6. 範例訊息交換流程 (時序圖)

```plaintext
前端 (瀏覽器)                       後端 (FastAPI)

send {type:"audio",...}   ──────▶   raw_q ← 音訊片段
send {type:"audio",...}   ──────▶   raw_q ← 音訊片段

                             STT pipeline → 文字結果 → result_q

receive {type:"subtitle", segmentId:"seg_1", text:"今天心"} ◀─────
receive {type:"subtitle", segmentId:"seg_1", text:"今天心情很好", isFinal:true} ◀─────

send {type:"stop"}        ──────▶   停止 pipeline → flush 剩餘字幕
```

---

## 7. 注意事項

* 前端傳送音訊必須與後端 STT 模型的編碼格式一致 (ex: PCM16, 16kHz)。
* `segmentId` 對應的是「一句話」，若 `isFinal=false` → 前端應更新同一行。
* 收到 `isFinal=true` → 前端可將該字幕鎖定，開啟下一行顯示新字幕。
* `speakerId` 與 `speakerName` 可用來分辨不同講者，UI 上可以用不同顏色或標籤。

---

## ✅ 總結

* **前端傳入**：`audio` (音訊 + timestamp) / `stop`
* **後端回傳**：`subtitle` (segmentId, speakerId, speakerName, distance, text, isFinal)
* **雙向 WebSocket**，前端可即時錄音上傳，後端即時回傳字幕。
