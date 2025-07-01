# services/api.py
"""
Unsaycret API 主要服務入口

此模組定義了 FastAPI 應用程式的 HTTP 路由，
負責處理客戶端請求並委託給相應的業務邏輯處理器。
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pipelines.orchestrator import run_pipeline
from services.handlers.speaker_handler import SpeakerHandler
import tempfile, shutil, os

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
    raw, pretty = run_pipeline(tmp_path)

    # 3. 刪暫存檔
    os.remove(tmp_path)

    # 4. 回傳 JSON（同時給 raw 與 pretty）
    return {
        "segments": raw,       # 機器可讀
        "pretty":   pretty     # Demo 時人類易讀
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
