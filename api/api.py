# services/api.py
"""
Unsaycret API 主要服務入口

此模組定義了 FastAPI 應用程式的 HTTP 路由，
負責處理客戶端請求並委託給相應的業務邏輯處理器。
"""
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import asyncio, threading, queue, json, time
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

# 添加 CORS 支援
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該限制為特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class SpeakerCreateResponse(BaseModel):
    """手動建立語者的回應模型"""
    speaker_uuid: str
    speaker_id: int
    full_name: str
    nickname: Optional[str] = None
    gender: Optional[str] = None
    voiceprint_uuid: str
    voiceprint_count: int

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
    session_type: str  # 必填
    title: str         # 必填
    # start_time: Optional[str] = None  # 自動從第一個 SpeechLog 設定
    # end_time: Optional[str] = None    # 自動從最後一個 SpeechLog 設定
    # summary: Optional[str] = None
    # participants: Optional[List[str]] = None  # 語者 UUID 列表

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

@app.post("/sessions/{session_id}/recalculate-timerange", response_model=ApiResponse)
async def recalculate_session_timerange(session_id: str) -> ApiResponse:
    """手動重新計算 Session 的時間範圍"""
    try:
        session_id = validate_id_parameter(session_id, "Session ID")
        result = data_facade.recalculate_session_timerange(session_id)
        return ApiResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新計算時間範圍時發生內部錯誤: {str(e)}")

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
    speaker: Optional[str] = None  # Speaker UUID (保留相容性)
    # ✅ 新增欄位: Speaker 的名字和暱稱
    speaker_name: Optional[str] = None      # Speaker 的全名
    speaker_nickname: Optional[str] = None  # Speaker 的暱稱
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

# ============================================================================
# 即時串流 WebSocket API - 詳細註釋版本
# 根據 realtime_stream_new_protocol.md 規格實作
# ============================================================================

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """
    WebSocket 即時語音處理 - 雙向通訊
    
    📝 功能說明:
    - 前端持續發送音訊片段
    - 後端即時轉換成文字並識別語者
    - 自動儲存 SpeechLog 到資料庫
    - 更新 Session 參與者清單
    
    🔄 訊息格式規格:
    前端 → 後端:
    - WebSocket bytes: 原始音訊 bytes 資料
    - WebSocket text: "stop" (停止信號)
    
    後端 → 前端:
    - {"type": "subtitle", "segmentId": "seg_1", "speakerId": "uuid", "speakerName": "n1", "distance": 0.2, "text": "今天心情很好", "isFinal": true}
    """
    
    # ========== 初始化階段 ==========
    logger.info("開始 WebSocket 連線建立程序")
    
    # 建立雙向通訊佇列
    raw_q = queue.Queue()       # 📥 前端→後端: 儲存音訊片段等待處理
    result_q = queue.Queue()    # 📤 後端→前端: 儲存轉錄結果等待發送
    stop_evt = threading.Event()  # 🛑 停止信號: 協調主線程與背景線程
    backend_thread = None       # 🔧 背景處理線程引用
    
    # ========== Session 驗證階段 ==========
    logger.info("驗證 Session UUID")
    session_uuid = ws.query_params.get("session")
    if not session_uuid or not UUID_PATTERN.match(session_uuid):
        logger.error(f"Session UUID 無效或缺失: {session_uuid}")
        await ws.close(code=1008, reason="Missing or invalid session UUID")
        return
    
    logger.info(f"Session UUID 驗證通過: {session_uuid}")
    
    # ========== Session 參與者管理初始化 ==========
    logger.info("載入 Session 既有參與者清單")
    session_info = data_facade.get_session_info(session_uuid) or {}
    session_participants = set(session_info.get("participants") or [])
    logger.info(f"既有參與者數量: {len(session_participants)}")

    try:
        # ========== WebSocket 連線建立 ==========
        await ws.accept()
        logger.info(f"✅ WebSocket 連線成功建立: {session_uuid}")

        # ========== 背景處理線程啟動 ==========
        def backend():
            """
            🔧 背景處理線程 - 負責音訊轉文字的核心工作
            
            工作流程:
            1. 從 raw_q 讀取音訊片段
            2. 執行 STT (語音轉文字) 處理
            3. 執行語者辨識 (Speaker Identification)
            4. 將結果放入 result_q 供主線程發送
            5. 收到停止信號時優雅結束
            """
            logger.info("🚀 背景處理線程啟動")
            try:
                run_pipeline_STREAM(
                    chunk_secs=WEBSOCKET_CHUNK_SECS,        # 音訊切片長度
                    max_workers=API_MAX_WORKERS,             # 最大併發處理數
                    in_bytes_queue=raw_q,                    # 📥 輸入: 音訊佇列
                    queue_out=result_q,                      # 📤 輸出: 結果佇列
                    stop_event=stop_evt,                     # 🛑 停止信號
                )
                logger.info("✅ STT pipeline 正常結束")
            except Exception as e:
                logger.error(f"❌ STT pipeline 發生錯誤: {e}")
            finally:
                # 🏁 無論成功或失敗都要通知主線程結束
                result_q.put(None)
                logger.info("📡 已發送結束信號給主線程")

        # 啟動背景處理線程
        backend_thread = threading.Thread(target=backend, daemon=True)
        backend_thread.start()
        logger.info("🔄 背景處理線程已啟動")

        # 處理狀態標誌
        processing_complete = False  # 📊 追蹤背景處理是否完成
        frontend_connected = True    # 📡 追蹤前端連線狀態
        websocket_broken = False     # 🔌 WebSocket 連線是否已斷開

        # 建立非同步任務用於並行處理
        ws_receive_task = None
        last_ws_receive_time = time.time()

        # ========== 主處理迴圈 - 雙向通訊核心 ==========
        logger.info("🔄 進入主處理迴圈 - 開始雙向通訊")
        while True:
            
            # ========== 步驟 1: 處理背景結果 (字幕發送) ==========
            try:
                # 📥 從結果佇列取得處理完的語音片段 (非常短的等待時間以提升響應性)
                seg = result_q.get(timeout=0.01)

                # 🏁 檢查是否為結束信號
                if seg is None:
                    processing_complete = True
                    logger.info("✅ 背景處理完全結束，準備關閉連線")
                    break

                segment_id = seg.get('segment', 'unknown')
                logger.info(f"📝 收到新的轉錄結果: segment {segment_id}")

                # ========== 資料庫儲存階段 (SpeechLog 管理) ==========
                logger.info(f"💾 開始處理 segment {segment_id} 的資料庫儲存")
                speechlog_created = False
                # 🆔 儲存每個語者的 SpeechLog UUID（用於前端追蹤和編輯）
                speaker_speechlog_uuids = {}
                
                # 遍歷此音訊片段中的所有識別到的語者
                speakers = seg.get("speakers", [])
                logger.info(f"👥 此片段識別到 {len(speakers)} 個語者")
                
                for speaker_idx, sp in enumerate(speakers):
                    speaker_id = sp.get("speaker_id")
                    speaker_text = sp.get("text", "")
                    
                    if speaker_id and speaker_text.strip():
                        logger.info(f"🗣️  處理語者 {speaker_idx + 1}: {speaker_id}")
                        
                        # 🔍 【診斷】驗證語者是否存在於資料庫
                        try:
                            speaker_exists = data_facade.get_speaker_info(speaker_id)
                            if speaker_exists:
                                logger.debug(f"✅ 語者驗證通過: {speaker_exists.get('full_name', 'Unknown')}")
                            else:
                                logger.error(f"❌ 語者不存在於資料庫: {speaker_id}")
                        except HTTPException as he:
                            if he.status_code == 404:
                                logger.error(f"❌ 語者不存在於資料庫: {speaker_id}")
                                logger.error("   此字幕將無法儲存！請檢查 pipeline 的語者識別邏輯")
                            else:
                                logger.warning(f"⚠️  語者驗證時發生錯誤: {he.detail}")
                        except Exception as ve:
                            logger.warning(f"⚠️  語者驗證異常: {ve}")
                        
                        # 📅 時間戳處理 - 使用絕對時間而非相對時間
                        absolute_start_time = sp.get("absolute_start_time")
                        start_time = seg.get("start", 0)
                        end_time = seg.get("end", 0)
                        duration = end_time - start_time
                        
                        logger.info(f"⏰ 時間資訊: {absolute_start_time}, 長度: {duration:.2f}秒")
                        
                        # 🏗️  建立 SpeechLog 記錄
                        sl_req = SpeechLogCreateRequest(
                            content=speaker_text,
                            confidence=sp.get("confidence"),
                            timestamp=absolute_start_time,
                            duration=duration,
                            speaker=speaker_id,
                            session=session_uuid,
                        )
                        
                        # 💾 嘗試儲存到資料庫
                        try:
                            result = data_facade.create_speechlog(sl_req)
                            if result.get("success"):
                                # 🆔 捕獲 SpeechLog UUID 供前端使用
                                speechlog_uuid = result.get("data", {}).get("uuid")
                                if speechlog_uuid:
                                    speaker_speechlog_uuids[speaker_id] = speechlog_uuid
                                    logger.info(f"✅ SpeechLog 儲存成功 (UUID: {speechlog_uuid}): {speaker_id} - \"{speaker_text[:50]}...\"")
                                else:
                                    logger.warning(f"⚠️  SpeechLog 儲存成功但未取得 UUID: {speaker_id}")
                                speechlog_created = True
                            else:
                                # 🚨 【Bug #2 增強】詳細記錄失敗原因
                                error_msg = result.get('message', '未知錯誤')
                                logger.error(f"❌ SpeechLog 儲存失敗: {error_msg}")
                                logger.error(f"   語者ID: {speaker_id}")
                                logger.error(f"   Session: {session_uuid}")
                                logger.error(f"   內容: \"{speaker_text[:50]}...\"")
                                
                                # 如果是語者不存在的錯誤，特別標記
                                if "語者不存在" in error_msg or "不存在" in error_msg:
                                    logger.critical(f"🚨 語者 {speaker_id} 不存在於資料庫！")
                                    logger.critical("   可能需要檢查語者識別邏輯或資料庫同步問題")
                        except Exception as e:
                            logger.error(f"💥 SpeechLog 儲存發生異常: {e}")
                            logger.error(f"   語者ID: {speaker_id}")
                            logger.error(f"   異常類型: {type(e).__name__}")

                        # 👥 Session 參與者管理 - 新語者自動加入
                        if speaker_id not in session_participants:
                            logger.info(f"🆕 發現新參與者: {speaker_id}")
                            session_participants.add(speaker_id)
                            try:
                                data_facade.update_session(
                                    session_uuid,
                                    {"participants": list(session_participants)},
                                )
                                logger.info(f"✅ Session 參與者更新成功: {speaker_id}")
                            except Exception as e:
                                logger.error(f"❌ Session 參與者更新失敗: {e}")
                    else:
                        logger.debug(f"⏭️  跳過空白語者資料: speaker_id={speaker_id}, text=\"{speaker_text}\"")

                # 📊 儲存結果統計
                if not speechlog_created and speakers:
                    logger.warning(f"⚠️  segment {segment_id} 有語者資料但未能儲存任何 SpeechLog")
                elif speechlog_created:
                    logger.info(f"📊 segment {segment_id} 成功儲存 SpeechLog")

                # ========== 前端通訊階段 (多語者 subtitle 格式) ==========
                # 📡 根據 realtime_stream_new_protocol.md 規格轉換格式
                logger.info(f"📡 準備發送 subtitle 訊息給前端")
                
                # 🎯 為每個語者發送獨立的字幕訊息
                # 📋 說明：每個語者都會收到獨立的字幕訊息，前端可以選擇如何顯示
                speakers = seg.get("speakers", [])
                total_speakers = len(speakers)
                
                if speakers and frontend_connected:
                    logger.info(f"👥 此片段有 {total_speakers} 個語者，將分別發送字幕")
                    
                    for speaker_idx, speaker in enumerate(speakers):
                        speaker_id = speaker.get("speaker_id")
                        speaker_text = speaker.get("text", "")
                        
                        # ⚠️ 【修復】統一過濾條件：必須同時有 speaker_id 和 text 才發送
                        # 這樣可以確保前端顯示的字幕都有對應的資料庫記錄
                        if not speaker_id or not speaker_text.strip():
                            logger.debug(f"⏭️  跳過無效語者資料: speaker_id={speaker_id}, text=\"{speaker_text}\"")
                            continue
                        
                        # 🆔 取得這個語者對應的 SpeechLog UUID
                        speechlog_uuid = speaker_speechlog_uuids.get(speaker_id)
                        
                        # 🚨 【Bug #1 修復】只發送已成功儲存到資料庫的字幕
                        # 如果該語者的 SpeechLog 儲存失敗（字典中沒有 UUID），則跳過發送
                        if not speechlog_uuid:
                            logger.error(f"❌ 跳過發送：語者 {speaker_id} 的字幕未成功儲存到資料庫")
                            logger.error(f"   內容: \"{speaker_text[:50]}...\"")
                            continue
                            
                        # 🏗️  組裝標準 subtitle 訊息格式 (含完整時間資訊 + SpeechLog UUID)
                        subtitle_msg = {
                            "type": "subtitle",                                    # 🏷️  訊息類型標識
                            "segmentId": seg.get("segment", "unknown"),           # 🆔 片段唯一識別碼
                            "speakerId": speaker_id,                              # 👤 語者 UUID
                            "speakerName": speaker.get("speaker", "Unknown"),     # 📛 語者顯示名稱
                            "speechLogUuid": speechlog_uuid,                      # 🆔 SpeechLog UUID (供前端編輯/刪除)
                            "distance": speaker.get("distance", None),           # 📏 識別信心距離
                            "text": speaker_text,                                # 💬 轉錄文字內容（使用已驗證的變數）
                            "confidence": speaker.get("confidence", None),       # 🎯 ASR 信心度
                            "startTime": speaker.get("start", None),             # ⏰ 語者開始時間 (相對)
                            "endTime": speaker.get("end", None),                 # ⏰ 語者結束時間 (相對)
                            "absoluteStartTime": speaker.get("absolute_start_time", None),  # 📅 絕對開始時間
                            "absoluteEndTime": speaker.get("absolute_end_time", None),      # 📅 絕對結束時間
                            "isFinal": True,                                      # ✅ 串流模式都是最終版本
                            "segment": {                                          # 📦 片段資訊
                                "totalSpeakers": total_speakers,                 # 👥 此片段總語者數
                                "speakerIndex": speaker_idx,                     # 📍 當前語者在片段中的索引
                                "segmentStart": seg.get("start", None),          # ⏰ 片段開始時間
                                "segmentEnd": seg.get("end", None)               # ⏰ 片段結束時間
                            }
                        }

                        # 📤 發送 JSON 訊息給前端
                        try:
                            await ws.send_text(json.dumps(subtitle_msg, ensure_ascii=False))
                            logger.info(f"✅ 已發送字幕 [{speaker_idx+1}/{total_speakers}]: segment={segment_id}, speaker={speaker_id}, text=\"{speaker_text[:30]}...\"")
                        except Exception as send_error:
                            logger.warning(f"⚠️  發送字幕時發生錯誤: {send_error}")
                            frontend_connected = False  # 標記前端已斷線
                            break  # 停止發送剩餘字幕
                
                elif not speakers:
                    logger.warning(f"⚠️  segment {segment_id} 沒有識別到任何語者")
                elif not frontend_connected:
                    logger.info(f"📝 前端已斷線，僅儲存不發送: segment={segment_id}")

            except queue.Empty:
                # 😴 結果佇列暫時為空，繼續等待
                pass

            # ========== 步驟 2: 接收前端訊息 (音訊輸入處理) - 使用非阻塞方式 ==========
            # 🔌 如果 WebSocket 已斷開，跳過接收步驟，只處理剩餘結果
            if not websocket_broken:
                # 創建或檢查 WebSocket 接收任務
                if ws_receive_task is None:
                    # 創建新的接收任務（非阻塞）
                    try:
                        ws_receive_task = asyncio.create_task(ws.receive())
                        last_ws_receive_time = time.time()
                    except Exception as e:
                        logger.error(f"創建 WebSocket 接收任務失敗: {e}")
                        websocket_broken = True
                        stop_evt.set()
                        ws_receive_task = None
                
                # 檢查接收任務是否完成（非阻塞檢查）
                if ws_receive_task is not None and ws_receive_task.done():
                    try:
                        msg = ws_receive_task.result()
                        ws_receive_task = None  # 重置任務以便下次創建新的
                        
                        mtype = msg.get("type")
                        
                        if mtype == "websocket.receive":
                            t = msg.get("text")
                            b = msg.get("bytes")

                            # 先處理文字，確保 "stop" 不會被 bytes 分支吃掉
                            if t is not None:
                                if t == "stop" or t.strip() == "stop":
                                    logger.info("🛑 收到停止信號，開始優雅關閉")
                                    stop_evt.set()
                                    # 立即清空音訊佇列
                                    cleared_count = 0
                                    try:
                                        while not raw_q.empty():
                                            raw_q.get_nowait()
                                            cleared_count += 1
                                        logger.info(f"已清空音訊佇列 ({cleared_count} 個項目)")
                                    except Exception:
                                        pass
                                    # 發送結束標記喚醒 pipeline
                                    try:
                                        raw_q.put_nowait(b"")
                                    except Exception:
                                        pass
                                    frontend_connected = False
                                    websocket_broken = True
                                    # 回 ACK
                                    try:
                                        await ws.send_text(json.dumps({"type": "status", "event": "stopping"}))
                                    except Exception:
                                        pass

                            elif b is not None:
                                if len(b) > 0:
                                    raw_q.put(b)

                            else:
                                logger.warning(f"websocket.receive 但 text/bytes 皆為 None")

                        elif mtype == "websocket.disconnect":
                            code = msg.get("code")
                            logger.info(f"🔌 客戶端斷線，code={code}")
                            frontend_connected = False
                            websocket_broken = True
                            stop_evt.set()
                            # 清空音訊佇列
                            try:
                                while not raw_q.empty():
                                    raw_q.get_nowait()
                                raw_q.put_nowait(b"")
                            except Exception:
                                pass

                        else:
                            logger.warning(f"❓ 未知訊息: {msg}")
                            
                    except WebSocketDisconnect:
                        # 🔌 前端主動斷線 - 但不立即結束，先完成背景處理
                        logger.info("🔌 前端主動斷線，但繼續完成背景處理以避免資料遺失")
                        frontend_connected = False
                        websocket_broken = True
                        stop_evt.set()
                        try:
                            while not raw_q.empty():
                                raw_q.get_nowait()
                            raw_q.put_nowait(b"")
                        except Exception:
                            pass
                        
                    except Exception as e:
                        # 💥 前端通訊錯誤
                        error_msg = str(e)
                        if "disconnect" in error_msg.lower() or "receive" in error_msg.lower():
                            logger.info("🔌 檢測到前端斷線，停止接收新音訊")
                            frontend_connected = False
                            websocket_broken = True
                            stop_evt.set()
                            try:
                                while not raw_q.empty():
                                    raw_q.get_nowait()
                                raw_q.put_nowait(b"")
                            except Exception:
                                pass
                        else:
                            logger.warning(f"� 前端通訊錯誤: {e}")
                            stop_evt.set()
                else:
                    # WebSocket 接收任務尚未完成，檢查是否超時
                    if time.time() - last_ws_receive_time > 10.0:
                        logger.warning("⏰ WebSocket 接收超時 (10秒)，可能連線已斷開")
                        websocket_broken = True
                        stop_evt.set()
                        try:
                            ws_receive_task.cancel()
                        except Exception:
                            pass
                
            # ========== 步驟 3: 檢查是否該結束 ==========
            # ========== 步驟 3: 檢查是否該結束 ==========
            if processing_complete:
                # ✅ 背景處理已完成，可以安全結束
                logger.info("🏁 背景處理完成，準備結束 WebSocket 連線")
                break
            elif stop_evt.is_set() and websocket_broken:
                # 🛑 已收到停止信號且連線已斷開，檢查是否該超時退出
                # 給予最多 3 秒時間完成剩餘處理
                if not hasattr(stop_evt, '_stop_time'):
                    stop_evt._stop_time = time.time()  # type: ignore
                    logger.info("⏰ 開始計時，最多等待 3 秒完成剩餘處理")
                elif time.time() - stop_evt._stop_time > 3.0:  # type: ignore
                    logger.warning("⏰ 超時：停止信號後 3 秒仍未完成，強制結束")
                    processing_complete = True
                    break
            elif stop_evt.is_set():
                # ⏳ 已收到停止信號但連線未斷開，繼續等待
                pass
            
            # 短暫休眠避免 CPU 空轉
            await asyncio.sleep(0.001)  # 1ms，讓出 CPU 時間

        # ========== 最後檢查 - 僅作為調試驗證 ==========
        logger.info("🔍 主循環結束，驗證佇列狀態")
        try:
            remaining_result = result_q.get_nowait()
            if remaining_result is not None:
                logger.warning(f"⚠️  發現未處理的結果，這可能表示程式邏輯有問題: {remaining_result.get('segment', 'unknown')}")
                # 不處理，只記錄警告
            else:
                logger.debug("佇列中只有結束信號，正常")
        except queue.Empty:
            logger.info("✅ 結果佇列已清空，正常結束")
        
        logger.info("主循環處理完畢")

    # ========== 異常處理區塊 ==========
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket 客戶端主動斷線")
    except Exception as e:
        logger.error(f"💥 WebSocket 處理過程發生未預期錯誤: {e}")
        
    # ========== 資源清理階段 (無論成功或失敗都會執行) ==========
    finally:
        logger.info("🧹 開始資源清理程序")
        
        # 🛑 確保停止信號被設置 (防止背景線程繼續運行)
        stop_evt.set()
        logger.info("📡 已設置停止信號")
        
        # ⏳ 等待背景線程優雅結束
        if backend_thread and backend_thread.is_alive():
            logger.info("⏳ 等待背景處理線程結束...")
            backend_thread.join(timeout=5)  # 最多等待 5 秒
            if backend_thread.is_alive():
                logger.warning("⚠️  背景線程未在時限內結束")
            else:
                logger.info("✅ 背景處理線程已正常結束")

        # 📅 Session 時間範圍自動更新 (重要！)
        logger.info("📅 開始更新 Session 時間範圍")
        try:
            logger.info(f"🔄 重新計算 Session {session_uuid} 的時間範圍...")
            result = data_facade.recalculate_session_timerange(session_uuid)
            if result.get("success"):
                logger.info(f"✅ Session {session_uuid} 時間範圍更新成功")
            else:
                logger.warning(f"⚠️  Session {session_uuid} 時間範圍更新失敗: {result.get('message')}")
        except Exception as e:
            logger.error(f"💥 更新 Session 時間範圍時發生錯誤: {e}")

        # 🔌 關閉 WebSocket 連線
        try:
            logger.info("🔌 關閉 WebSocket 連線")
            await ws.close()
        except Exception as e:
            logger.debug(f"關閉 WebSocket 時的輕微錯誤: {e}")
            pass  # 忽略關閉時的錯誤
            
        logger.info(f"🏁 WebSocket 會話 {session_uuid} 完全結束")

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

@app.post("/speakers/create", response_model=ApiResponse)
async def create_speaker_with_voice(
    file: UploadFile = File(...),
    full_name: str = Form(...),  # 必填
    nickname: Optional[str] = Form(None),
    gender: Optional[str] = Form(None)
):
    """手動建立語者 - 上傳音檔並建立新語者檔案"""
    # 1. 驗證檔案類型（僅支援 WAV）
    if not file.filename or not file.filename.lower().endswith('.wav'):
        raise HTTPException(
            status_code=400, 
            detail="不支援的音檔格式，請使用 WAV 格式"
        )
    
    # 2. 驗證全名（必填且不能為空）
    if not full_name or not full_name.strip():
        raise HTTPException(
            status_code=400, 
            detail="語者全名為必填欄位，不能為空"
        )
    
    if len(full_name.strip()) > 50:
        raise HTTPException(
            status_code=400, 
            detail="語者全名不能超過50個字元"
        )
    
    # 3. 驗證暱稱長度（選填）
    if nickname and len(nickname.strip()) > 30:
        raise HTTPException(
            status_code=400, 
            detail="語者暱稱不能超過30個字元"
        )
    
    # 4. 性別不做限制，可以是任何值或空值
    # 4. 性別不做限制，可以是任何值或空值
    
    # 5. 儲存暫存檔案
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # 6. 執行建立語者邏輯
        result = data_facade.create_speaker_with_voice(
            audio_file_path=tmp_path,
            full_name=full_name.strip(),  # 必填，已驗證不為空
            nickname=nickname.strip() if nickname else None,
            gender=gender.strip() if gender else None
        )
        
        return ApiResponse(**result)
        
    finally:
        # 7. 清理暫存檔案
        try:
            os.remove(tmp_path)
        except:
            pass  # 忽略刪除暫存檔案的錯誤

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