# services/api.py
"""
Unsaycret API 主要服務入口

此模組定義了 FastAPI 應用程式的 HTTP 路由，
負責處理客戶端請求並委託給相應的業務邏輯處理器。
"""
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import asyncio, threading, queue, json
from pipelines.orchestrator import (
    run_pipeline_FILE,
    run_pipeline_STREAM,
    run_pipeline_DIR,
)
from services.handlers.speaker_handler import SpeakerHandler
import tempfile, shutil, os, zipfile

app = FastAPI(title="Unsaycret API")

# 初始化處理器
speaker_handler = SpeakerHandler()

# Pydantic 模型定義
class SpeakerRenameRequest(BaseModel):
    """語者改名請求模型"""
    speaker_id: str
    current_name: str
    new_name: str

class SpeakerTransferRequest(BaseModel):
    """聲紋轉移請求模型"""
    source_speaker_id: str
    source_speaker_name: str
    target_speaker_id: str
    target_speaker_name: str

class SpeakerInfo(BaseModel):
    speaker_id: str
    name: str
    first_audio_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    voiceprint_ids: Optional[List[str]] = None

class ApiResponse(BaseModel):
    """統一API回應模型"""
    success: bool
    message: str
    data: Optional[dict] = None

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # 1. 存暫存 wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # 2. 跑 pipeline，拿 raw + pretty
    raw, pretty, stats = run_pipeline_FILE(tmp_path)

    # 3. 刪暫存檔
    os.remove(tmp_path)

    # 4. 回傳 JSON（同時給 raw 與 pretty）
    return {
        "segments": raw,       # 機器可讀
        "pretty":   pretty,     # Demo 時人類易讀 👍
        "stats":    stats,
    }
    

@app.post("/speaker/rename", response_model=ApiResponse)
async def rename_speaker(request: SpeakerRenameRequest):
    """
    更改語者名稱的API端點
    
    Args:
        request: 包含speaker_id、current_name和new_name的請求
        
    Returns:
        ApiResponse: 包含操作結果的回應
    """
    result = speaker_handler.rename_speaker(
        speaker_id=request.speaker_id,
        current_name=request.current_name,
        new_name=request.new_name
    )
    return ApiResponse(**result)

@app.post("/speaker/transfer", response_model=ApiResponse)
async def transfer_voiceprints(request: SpeakerTransferRequest):
    """
    將聲紋從來源語者轉移到目標語者的API端點
    
    Args:
        request: 包含來源和目標語者資訊的請求
        
    Returns:
        ApiResponse: 包含操作結果的回應
    """
    result = speaker_handler.transfer_voiceprints(
        source_speaker_id=request.source_speaker_id,
        source_speaker_name=request.source_speaker_name,
        target_speaker_id=request.target_speaker_id,
        target_speaker_name=request.target_speaker_name
    )
    return ApiResponse(**result)

@app.get("/speaker/{speaker_id}")
async def get_speaker_info(speaker_id: str):
    """
    獲取語者資訊的輔助API（用於前端驗證）
    """
    return speaker_handler.get_speaker_info(speaker_id)

@app.get("/speakers", response_model=List[SpeakerInfo])
async def list_speakers():
    """
    列出所有語者與完整資訊
    """
    return speaker_handler.list_all_speakers()

@app.delete("/speaker/{speaker_id}", response_model=ApiResponse)
async def delete_speaker(speaker_id: str):
    """
    刪除語者及其底下的所有 voiceprints
    """
    result = speaker_handler.delete_speaker(speaker_id)
    return ApiResponse(**result)


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