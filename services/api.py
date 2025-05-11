# services/api.py
from fastapi import FastAPI, UploadFile, File
from pipelines.orchestrator import run_pipeline
import tempfile, shutil, os

app = FastAPI(title="SpeechProject API")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # 1. 存成暫存 wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # 2. 呼叫管線，拿到 segments list
    segments = run_pipeline(tmp_path)

    # 3. 刪除暫存檔
    os.remove(tmp_path)

    # 4. 回傳 JSON
    return {"segments": segments}
