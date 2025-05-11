# asr_model.py
from faster_whisper import WhisperModel

# 常用型號對照表
_MODEL_MAP = {
    "tiny":  "guillaumekln/faster-whisper-tiny",
    "base":  "guillaumekln/faster-whisper-base",
    "small": "guillaumekln/faster-whisper-small",
    "medium":"guillaumekln/faster-whisper-medium",
    "large-v2":"guillaumekln/faster-whisper-large-v2",
    "large-v3":"guillaumekln/faster-whisper-large-v3",
}

def load_model(model_name: str = "medium",
               gpu: bool = False,
               cache: str = "ct2_models") -> WhisperModel:
    """
    model_name 可用簡名 (tiny…large-v3) 或完整 repo 名。
    """
    repo = _MODEL_MAP.get(model_name, model_name)
    device = "cuda" if gpu else "cpu"

    # ctype  = "float16" if gpu else "int8_float16"
    ctype  = "float16" if gpu else "int8"
    
    return WhisperModel(repo,
                        device=device,
                        compute_type=ctype,
                        download_root=cache)
