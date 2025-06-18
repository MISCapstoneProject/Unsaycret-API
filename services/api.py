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
        "pretty":   pretty,     # Demo 時人類易讀 👍
        "stats":    stats,
    }


@app.post("/transcribe_dir")
async def transcribe_dir(path: str = Form(None), zip_file: UploadFile = File(None)):
    """Transcribe all audio files in a directory or uploaded ZIP."""
    if path is None and zip_file is None:
        raise HTTPException(status_code=400, detail="Provide directory path or ZIP file")

    if zip_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, zip_file.filename or "input.zip")
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(zip_file.file, f)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)
            summary_path = run_pipeline_DIR(tmpdir)
    else:
        summary_path = run_pipeline_DIR(path)

    return {"summary_tsv": summary_path}


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()

    raw_q    = queue.Queue()   # 前端上送的音訊
    result_q = queue.Queue()   # 新增：後端要推給前端的結果
    stop_evt = threading.Event()

    # ---------------- 背景 thread ---------------- #
    def backend():
        run_pipeline_STREAM(
            chunk_secs=6,
            max_workers=2,
            record_secs=None,
            in_bytes_queue=raw_q,   # ← 改成讀前端送來的 bytes
            queue_out=result_q,     # ★ 把結果塞進 result_q
            stop_event=stop_evt,
        )
        result_q.put(None)          # 通知主線程「我結束了」

    threading.Thread(target=backend, daemon=True).start()

    # -------------- 主收/發 loop -------------- #
    try:
        while True:
            # 1) 先把後端產生的結果 non-blocking 取出、推給前端
            try:
                seg = result_q.get_nowait()
                if seg is None:          # backend 完成
                    break
                await ws.send_text(json.dumps(seg, ensure_ascii=False))
            except queue.Empty:
                pass

            # 2) 再 non-blocking 收前端的音訊
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=0.05)
            except asyncio.TimeoutError:
                continue

            if "bytes" in data:
                raw_q.put(data["bytes"])                 # 給後端
            elif "text" in data and data["text"] == "stop":
                stop_evt.set()
                break
    except WebSocketDisconnect:
        stop_evt.set()
    finally:
        await ws.close()