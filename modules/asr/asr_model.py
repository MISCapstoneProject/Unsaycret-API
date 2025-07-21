# asr_model.py
from faster_whisper import WhisperModel
from utils.constants import WHISPER_MODEL_CACHE_DIR, WHISPER_MODEL_MAP, DEFAULT_WHISPER_MODEL

def load_model(model_name: str = DEFAULT_WHISPER_MODEL,
               gpu: bool = False,
               cache: str = None) -> WhisperModel:
    """
    model_name 可用簡名 (tiny…large-v3) 或完整 repo 名。
    """
    repo = WHISPER_MODEL_MAP.get(model_name, model_name)
    device = "cuda" if gpu else "cpu"

    # ctype  = "float16" if gpu else "int8_float16"
    ctype  = "float16" if gpu else "int8"
    
    # 使用常數定義的快取目錄
    cache_dir = cache if cache is not None else WHISPER_MODEL_CACHE_DIR
    
    return WhisperModel(repo,
                        device=device,
                        compute_type=ctype,
                        download_root=cache_dir)
