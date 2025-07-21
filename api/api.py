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
from api.handlers.speaker_handler import SpeakerHandler
import tempfile, shutil, os, zipfile
from utils.constants import (
    API_DEFAULT_VERIFICATION_THRESHOLD, API_DEFAULT_MAX_RESULTS,
    WEBSOCKET_CHUNK_SECS, WEBSOCKET_TIMEOUT, WEBSOCKET_MAX_WORKERS,
    API_MAX_WORKERS
)

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

class VoiceCandidate(BaseModel):
    """語音驗證候選者模型"""
    voiceprint_uuid: str  # 使用 UUID 作為識別符
    speaker_name: str
    distance: float
    update_count: int
    is_match: bool

class VoiceMatch(BaseModel):
    """語音匹配結果模型"""
    voiceprint_uuid: str  # 使用 UUID 作為識別符
    speaker_name: str
    distance: float
    is_match: bool

class VoiceVerificationResponse(BaseModel):
    """語音驗證響應模型"""
    success: bool
    message: str
    is_known_speaker: bool
    best_match: Optional[VoiceMatch] = None
    all_candidates: List[VoiceCandidate] = []
    threshold: float
    total_candidates: int

class SpeakerInfo(BaseModel):
    """V2 資料庫完整語者資訊模型"""
    uuid: str  # Weaviate UUID
    speaker_id: int  # 序號ID (從1開始)
    full_name: Optional[str] = None
    nickname: Optional[str] = None
    gender: Optional[str] = None
    created_at: Optional[str] = None
    last_active_at: Optional[str] = None
    meet_count: Optional[int] = None
    meet_days: Optional[int] = None
    voiceprint_ids: Optional[List[str]] = None
    first_audio: Optional[str] = None

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

@app.get("/speaker/{speaker_id}", response_model=SpeakerInfo)
async def get_speaker_info(speaker_id: str):
    """
    獲取語者資訊的輔助API（用於前端驗證）
    回傳 V2 資料庫完整結構
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

@app.post("/speaker/verify", response_model=VoiceVerificationResponse)
async def verify_speaker_voice(
    file: UploadFile = File(...),
    max_results: int = Form(API_DEFAULT_MAX_RESULTS),
    threshold: float = Form(API_DEFAULT_VERIFICATION_THRESHOLD)
):
    """
    語音驗證API端點 - 純讀取操作，判斷音檔中的語者身份
    
    Args:
        file: 要驗證的音檔
        max_results: 返回最相似的結果數量 (預設 {API_DEFAULT_MAX_RESULTS})
        threshold: 比對閾值，距離小於此值才認為是匹配到語者 (預設 {API_DEFAULT_VERIFICATION_THRESHOLD})
        
    Returns:
        VoiceVerificationResponse: 包含驗證結果的回應
    """
    # 1. 驗證檔案類型
    if not file.filename or not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(
            status_code=400, 
            detail="不支援的音檔格式，請使用 WAV、MP3、FLAC 或 M4A 格式"
        )
    
    # 2. 驗證參數範圍
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(
            status_code=400, 
            detail="比對閾值必須在 0.0 到 1.0 之間"
        )
    
    if not 1 <= max_results <= 10:
        raise HTTPException(
            status_code=400, 
            detail="最大結果數量必須在 1 到 10 之間"
        )
    
    # 3. 儲存暫存檔案
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # 4. 執行語音驗證
        result = speaker_handler.verify_speaker_voice(
            audio_file_path=tmp_path,
            threshold=threshold,
            max_results=max_results
        )
        
        return VoiceVerificationResponse(**result)
        
    finally:
        # 5. 清理暫存檔案
        try:
            os.remove(tmp_path)
        except:
            pass  # 忽略刪除暫存檔案的錯誤


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
            chunk_secs=WEBSOCKET_CHUNK_SECS,
            max_workers=API_MAX_WORKERS,
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
                data = await asyncio.wait_for(ws.receive(), timeout=WEBSOCKET_TIMEOUT)
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