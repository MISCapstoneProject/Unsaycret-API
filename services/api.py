# services/api.py
from fastapi import FastAPI, UploadFile, File
from pipelines.orchestrator import run_pipeline_file#, run_pipeline_record
import tempfile, shutil, os

app = FastAPI(title="SpeechProject API")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # 1. 存暫存 wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # 2. 跑 pipeline，拿 raw + pretty
    raw, pretty = run_pipeline_file(tmp_path)

    # 3. 刪暫存檔
    os.remove(tmp_path)

    # 4. 回傳 JSON（同時給 raw 與 pretty）
    return {
        "segments": raw,       # 機器可讀
        "pretty":   pretty     # Demo 時人類易讀 👍
    }


# @app.post("/record")
# async def record_endpoint(duration: int = 5):
#     """Record audio from microphone for a fixed duration and transcribe."""
#     raw, pretty = run_pipeline_record(duration)
#     return {
#         "segments": raw,
#         "pretty": pretty,
#     }