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
from datetime import datetime
from pipelines.orchestrator import (
    run_pipeline_FILE,
    run_pipeline_STREAM,
    run_pipeline_DIR,
)
from services.data_facade import DataFacade
import tempfile, shutil, os, zipfile
from utils.constants import (
    API_DEFAULT_VERIFICATION_THRESHOLD, API_DEFAULT_MAX_RESULTS,
    WEBSOCKET_CHUNK_SECS, WEBSOCKET_TIMEOUT, WEBSOCKET_MAX_WORKERS,
    API_MAX_WORKERS
)
from utils.logger import get_logger
import re

# 創建日誌器
logger = get_logger(__name__)

# UUID 驗證正則表達式
UUID_PATTERN = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

def validate_id_parameter(id_value: str, param_name: str = "ID") -> str:
    """驗證並標準化ID參數"""
    # 檢查空字串或None
    if not id_value or not id_value.strip():
        raise HTTPException(status_code=400, detail=f"{param_name}參數不能為空")
    
    if re.match(UUID_PATTERN, id_value):
        return id_value
    else:
        # 非標準UUID格式，原樣返回以便後續處理
        return id_value

app = FastAPI(title="Unsaycret API")

# 初始化資料存取接口
data_facade = DataFacade()

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

class SpeakerUpdateRequest(BaseModel):
    """語者資料更新請求模型（僅允許部分欄位可選）"""
    full_name: Optional[str] = None
    nickname: Optional[str] = None
    gender: Optional[str] = None
    created_at: Optional[str] = None
    last_active_at: Optional[str] = None
    meet_count: Optional[int] = None
    meet_days: Optional[int] = None

class ApiResponse(BaseModel):
    """統一API回應模型"""
    success: bool
    message: str
    data: Optional[dict] = None

# ----------------------------------------------------------------------------
# RESTful API 路由設計
# 統一使用複數形式的資源名稱，遵循 REST 最佳實務
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Health Check API - 系統健康檢查
# ----------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """系統健康檢查端點"""
    return {
        "status": "healthy",
        "message": "Unsaycret API is running",
        "timestamp": datetime.now().isoformat()
    }

# ----------------------------------------------------------------------------
# Sessions API - 會議/場次管理
# ----------------------------------------------------------------------------
class SessionCreateRequest(BaseModel):
    session_type: Optional[str] = None
    title: Optional[str] = None
    start_time: Optional[str] = None  # ISO 格式字串，預設為當下時間
    end_time: Optional[str] = None    # ISO 格式字串
    summary: Optional[str] = None
    participants: Optional[List[str]] = None  # 語者 UUID 列表

class SessionUpdateRequest(BaseModel):
    session_type: Optional[str] = None
    title: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    summary: Optional[str] = None
    participants: Optional[List[str]] = None

class SessionInfo(BaseModel):
    uuid: str
    session_id: str
    session_type: Optional[str] = None
    title: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    summary: Optional[str] = None
    participants: Optional[List[str]] = []

@app.post("/sessions", response_model=ApiResponse)
async def create_session(request: SessionCreateRequest) -> ApiResponse:
    """新增 Session 記錄"""
    result = data_facade.create_session(request)
    return ApiResponse(**result)

@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions() -> List[SessionInfo]:
    """列出所有 Session"""
    return data_facade.list_sessions()

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str) -> SessionInfo:
    """取得單一 Session 資訊"""
    try:
        # 驗證並清理 session_id
        session_id = validate_id_parameter(session_id, "Session ID")
        
        result = data_facade.get_session_info(session_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"找不到ID為 {session_id} 的Session")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取Session資訊時發生內部錯誤: {str(e)}")

@app.patch("/sessions/{session_id}", response_model=ApiResponse)
async def update_session(session_id: str, request: SessionUpdateRequest) -> ApiResponse:
    """部分更新 Session"""
    try:
        session_id = validate_id_parameter(session_id, "Session ID")
        update_data = request.model_dump(exclude_unset=True)
        result = data_facade.update_session(session_id, update_data)
        return ApiResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新Session時發生內部錯誤: {str(e)}")

@app.delete("/sessions/{session_id}", response_model=ApiResponse)
async def delete_session(session_id: str) -> ApiResponse:
    """刪除 Session"""
    try:
        session_id = validate_id_parameter(session_id)
        result = data_facade.delete_session(session_id)
        return ApiResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除Session時發生內部錯誤: {str(e)}")

# ----------------------------------------------------------------------------
# SpeechLogs API - 語音記錄管理  
# ----------------------------------------------------------------------------
class SpeechLogCreateRequest(BaseModel):
    content: Optional[str] = None
    timestamp: Optional[str] = None  # ISO 格式字串，預設為當下時間
    confidence: Optional[float] = None
    duration: Optional[float] = None
    language: Optional[str] = None
    speaker: Optional[str] = None  # 語者 UUID
    session: Optional[str] = None  # Session UUID

class SpeechLogUpdateRequest(BaseModel):
    content: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    session: Optional[str] = None

class SpeechLogInfo(BaseModel):
    uuid: str
    content: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    session: Optional[str] = None

@app.post("/speechlogs", response_model=ApiResponse)
async def create_speechlog(request: SpeechLogCreateRequest) -> ApiResponse:
    """新增 SpeechLog 記錄"""
    result = data_facade.create_speechlog(request)
    return ApiResponse(**result)

@app.get("/speechlogs", response_model=List[SpeechLogInfo])
async def list_speechlogs() -> List[SpeechLogInfo]:
    """列出所有 SpeechLog"""
    return data_facade.list_speechlogs()

@app.get("/speechlogs/{speechlog_id}", response_model=SpeechLogInfo)
async def get_speechlog_info(speechlog_id: str) -> SpeechLogInfo:
    """取得單一 SpeechLog 資訊"""
    try:
        # 驗證並清理 speechlog_id
        speechlog_id = validate_id_parameter(speechlog_id, "SpeechLog ID")
        
        result = data_facade.get_speechlog_info(speechlog_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"找不到ID為 {speechlog_id} 的SpeechLog")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取SpeechLog資訊時發生內部錯誤: {str(e)}")

@app.patch("/speechlogs/{speechlog_id}", response_model=ApiResponse)
async def update_speechlog(speechlog_id: str, request: SpeechLogUpdateRequest) -> ApiResponse:
    """部分更新 SpeechLog"""
    try:
        speechlog_id = validate_id_parameter(speechlog_id, "SpeechLog ID")
        update_data = request.model_dump(exclude_unset=True)
        result = data_facade.update_speechlog(speechlog_id, update_data)
        return ApiResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新SpeechLog時發生內部錯誤: {str(e)}")

@app.delete("/speechlogs/{speechlog_id}", response_model=ApiResponse)
async def delete_speechlog(speechlog_id: str) -> ApiResponse:
    """刪除 SpeechLog"""
    try:
        speechlog_id = validate_id_parameter(speechlog_id, "SpeechLog ID")
        result = data_facade.delete_speechlog(speechlog_id)
        return ApiResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除SpeechLog時發生內部錯誤: {str(e)}")

# ----------------------------------------------------------------------------
# Core Processing APIs - 核心處理功能
# ----------------------------------------------------------------------------

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """轉錄音檔"""
    tmp_path = None
    try:
        # 驗證檔案
        if not file.filename:
            raise HTTPException(status_code=400, detail="未提供檔案名稱")
        
        # 1. 存暫存 wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # 2. 跑 pipeline，拿 raw + pretty
        raw, pretty, stats = run_pipeline_FILE(tmp_path)

        # 4. 回傳 JSON（同時給 raw 與 pretty）
        return {
            "segments": raw,       # 機器可讀
            "pretty":   pretty,     # Demo 時人類易讀 👍
            "stats":    stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"轉錄處理時發生錯誤: {str(e)}")
    finally:
        # 3. 確保刪除暫存檔
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_error:
                logger.warning(f"清理暫存檔案失敗: {cleanup_error}")

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
    """WebSocket即時語音處理"""
    raw_q = queue.Queue()   # 前端上送的音訊
    result_q = queue.Queue()   # 後端要推給前端的結果
    stop_evt = threading.Event()
    backend_thread = None

    # 讀取並驗證 Session UUID
    session_uuid = ws.query_params.get("session")
    if not session_uuid or not UUID_PATTERN.match(session_uuid):
        await ws.close(code=1008, reason="Missing or invalid session UUID")
        return

    # 取得 Session 既有參與者
    session_info = data_facade.get_session_info(session_uuid) or {}
    session_participants = set(session_info.get("participants") or [])

    try:
        await ws.accept()

        # ---------------- 背景 thread ---------------- #
        def backend():
            try:
                run_pipeline_STREAM(
                    chunk_secs=WEBSOCKET_CHUNK_SECS,
                    max_workers=API_MAX_WORKERS,
                    record_secs=None,
                    in_bytes_queue=raw_q,   # ← 改成讀前端送來的 bytes
                    queue_out=result_q,     # ★ 把結果塞進 result_q
                    stop_event=stop_evt,
                )
            except Exception as e:
                logger.error(f"WebSocket背景處理發生錯誤: {e}")
            finally:
                result_q.put(None)          # 通知主線程「我結束了」

        backend_thread = threading.Thread(target=backend, daemon=True)
        backend_thread.start()

        # -------------- 主收/發 loop -------------- #
        while True:
            # 1) 先把後端產生的結果 non-blocking 取出、推給前端
            try:
                seg = result_q.get_nowait()
                if seg is None:          # backend 完成
                    break

                # 儲存 SpeechLog 並更新 Session 參與者
                for sp in seg.get("speakers", []):
                    speaker_id = sp.get("speaker_id")
                    if speaker_id:
                        sl_req = SpeechLogCreateRequest(
                            content=sp.get("text"),
                            confidence=sp.get("confidence"),
                            duration=(seg.get("end", 0) - seg.get("start", 0)),
                            speaker=speaker_id,
                            session=session_uuid,
                        )
                        data_facade.create_speechlog(sl_req)

                        if speaker_id not in session_participants:
                            session_participants.add(speaker_id)
                            data_facade.update_session(
                                session_uuid,
                                {"participants": list(session_participants)},
                            )

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
        logger.info("WebSocket客戶端斷線")
    except Exception as e:
        logger.error(f"WebSocket處理發生錯誤: {e}")
    finally:
        # 確保資源清理
        stop_evt.set()
        
        # 等待背景線程結束
        if backend_thread and backend_thread.is_alive():
            backend_thread.join(timeout=5)  # 最多等5秒
            
        # 清空佇列
        try:
            while not raw_q.empty():
                raw_q.get_nowait()
        except:
            pass
            
        try:
            while not result_q.empty():
                result_q.get_nowait()
        except:
            pass
        
        # 關閉WebSocket連接
        try:
            await ws.close()
        except:
            pass

# ----------------------------------------------------------------------------
# Speakers API - 語者管理
# ----------------------------------------------------------------------------
    
@app.get("/speakers", response_model=List[SpeakerInfo])
async def list_speakers():
    """列出所有語者"""
    return data_facade.list_all_speakers()

@app.get("/speakers/{speaker_id}", response_model=SpeakerInfo)
async def get_speaker(speaker_id: str):
    """取得單一語者資訊"""
    try:
        # 驗證並清理 speaker_id
        speaker_id = validate_id_parameter(speaker_id, "語者ID")
        
        result = data_facade.get_speaker_info(speaker_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取語者資訊時發生內部錯誤: {str(e)}")

@app.patch("/speakers/{speaker_id}", response_model=ApiResponse)
async def update_speaker(speaker_id: str, request: SpeakerUpdateRequest) -> ApiResponse:
    """更新語者資料"""
    try:
        speaker_id = validate_id_parameter(speaker_id, "語者ID")
        forbidden_fields = {"voiceprint_ids", "first_audio"}
        update_data = request.model_dump(exclude_unset=True)
        update_fields = {k: v for k, v in update_data.items() if k not in forbidden_fields and v is not None}
        if not update_fields:
            return ApiResponse(success=False, message="未提供可更新的欄位", data=None)
        
        result = data_facade.update_speaker(
            speaker_id=speaker_id,
            update_fields=update_fields
        )
        return ApiResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        return ApiResponse(success=False, message=f"更新失敗: {str(e)}", data=None)

@app.delete("/speakers/{speaker_id}", response_model=ApiResponse)
async def delete_speaker(speaker_id: str):
    """刪除語者及其所有聲紋"""
    try:
        # 驗證並清理 speaker_id
        speaker_id = validate_id_parameter(speaker_id, "語者ID")
        
        result = data_facade.delete_speaker(speaker_id)
        return ApiResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除語者時發生內部錯誤: {str(e)}")

# ----------------------------------------------------------------------------
# Speaker Actions - 語者相關操作
# ----------------------------------------------------------------------------

@app.post("/speakers/verify", response_model=VoiceVerificationResponse)
async def verify_speaker_voice(
    file: UploadFile = File(...),
    max_results: int = Form(API_DEFAULT_MAX_RESULTS),
    threshold: float = Form(API_DEFAULT_VERIFICATION_THRESHOLD)
):
    """語音驗證 - 識別音檔中的語者身份"""
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
        result = data_facade.verify_speaker_voice(
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

@app.post("/speakers/transfer", response_model=ApiResponse)
async def transfer_voiceprints(request: SpeakerTransferRequest):
    """聲紋轉移 - 將聲紋從來源語者轉移到目標語者"""
    result = data_facade.transfer_voiceprints(
        source_speaker_id=request.source_speaker_id,
        source_speaker_name=request.source_speaker_name,
        target_speaker_id=request.target_speaker_id,
        target_speaker_name=request.target_speaker_name
    )
    return ApiResponse(**result)

# ----------------------------------------------------------------------------
# Nested Resource APIs - 巢狀資源查詢
# RESTful 設計：/resource/{id}/sub-resource
# ----------------------------------------------------------------------------

@app.get("/speakers/{speaker_id}/sessions", response_model=List[SessionInfo])
async def get_speaker_sessions(speaker_id: str) -> List[SessionInfo]:
    """取得語者參與的所有會議"""
    return data_facade.get_speaker_sessions(speaker_id)

@app.get("/speakers/{speaker_id}/speechlogs", response_model=List[SpeechLogInfo])
async def get_speaker_speechlogs(speaker_id: str) -> List[SpeechLogInfo]:
    """取得語者的所有語音記錄"""
    return data_facade.get_speaker_speechlogs(speaker_id)

@app.get("/sessions/{session_id}/speechlogs", response_model=List[SpeechLogInfo])
async def get_session_speechlogs(session_id: str) -> List[SpeechLogInfo]:
    """取得會議中的所有語音記錄"""
    return data_facade.get_session_speechlogs(session_id)