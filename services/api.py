# services/api.py
from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import asyncio, threading, queue, json
from pipelines.orchestrator import run_pipeline_FILE, run_pipeline_STREAM
import tempfile, shutil, os

app = FastAPI(title="SpeechProject API")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # 1. 存暫存 wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # 2. 跑 pipeline，拿 raw + pretty
    raw, pretty = run_pipeline_FILE(tmp_path)

    # 3. 刪暫存檔
    os.remove(tmp_path)

    # 4. 回傳 JSON（同時給 raw 與 pretty）
    return {
        "segments": raw,       # 機器可讀
        "pretty":   pretty     # Demo 時人類易讀 👍
    }

@app.get("/stream/sse")
def stream_sse(
    request: Request,          # <- 監聽 client 斷線
    chunk: float = 6.0,
    workers: int = 2,
    seconds: float | None = None,   # None => 無限，否則錄到秒數就結束
):
    q: queue.Queue[dict | None] = queue.Queue()
    stop_evt = threading.Event()

    def bg():
        # 把 run_pipeline_STREAM 跑在背景，塞資料進 queue
        run_pipeline_STREAM(
            chunk_secs=chunk,
            max_workers=workers,
            record_secs=seconds,     # 可選上限
            queue_out=q,
        )
        q.put(None)                  # 送結束旗標

    threading.Thread(target=bg, daemon=True).start()

    async def event_gen():
        loop = asyncio.get_event_loop()
        try:
            while True:
                # 如果瀏覽器關閉連線 -> request.is_disconnected() 會變 True
                if await request.is_disconnected():
                    stop_evt.set()  # 通知背景 thread 停止
                    break
                item = await loop.run_in_executor(None, q.get)

                if item is None:     # 後端自然結束
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            # client 斷線或 generator 結束 -> 通知背景 thread
            stop_evt.set()
            q.put(None)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
