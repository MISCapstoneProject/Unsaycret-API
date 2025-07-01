# services/api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pipelines.orchestrator import run_pipeline_file, run_pipeline_dir  #, run_pipeline_record
import tempfile, shutil, os, zipfile

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
            summary_path = run_pipeline_dir(tmpdir)
    else:
        summary_path = run_pipeline_dir(path)

    return {"summary_tsv": summary_path}


# @app.post("/record")
# async def record_endpoint(duration: int = 5):
#     """Record audio from microphone for a fixed duration and transcribe."""
#     raw, pretty = run_pipeline_record(duration)
#     return {
#         "segments": raw,
#         "pretty": pretty,
#     }