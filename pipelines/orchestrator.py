# pipelines/orchestrator.py
# 在文件最顶部
import torch, os
from utils.logger import get_logger

logger = get_logger(__name__)

logger.info("🖥  GPU available: %s", torch.cuda.is_available())
if torch.cuda.is_available():
    logger.info("   Device: %s", torch.cuda.get_device_name(0))
# 然後再 import 你的模块

import threading
import json, uuid, pathlib, datetime as dt
import torchaudio
from concurrent.futures import ThreadPoolExecutor
from modules.separation.separator import AudioSeparator
from modules.identification.VID_identify_v5 import SpeakerIdentifier
from modules.asr.whisper_asr import WhisperASR

# ---------- 1. 自動偵測 GPU ----------
use_gpu = torch.cuda.is_available()
logger.info(f"🚀 使用設備: {'cuda' if use_gpu else 'cpu'}")


sep = AudioSeparator()
spk = SpeakerIdentifier()
asr = WhisperASR(model_name="medium", gpu=use_gpu)

# ---------- 3. 處理單一片段的函式 ----------

def process_segment(seg_path, t0, t1):
    logger.info(f"🔧 執行緒 {threading.get_ident()} 正在處理 segment ({t0:.2f} - {t1:.2f})")

    speaker_id, name, dist = spk.process_audio_file(seg_path)
    text, conf, words       = asr.transcribe(seg_path)
    return {
        "start": round(t0, 2),
        "end":   round(t1, 2),
        "speaker": name,
        "distance": round(float(dist), 3),
        "text": text,
        "confidence": round(conf, 2),
        "words": words,
    }

# ---------- 4. 主 pipeline 函式 ----------
def make_pretty(seg: dict) -> dict:
    """把一段 segment 轉成易讀格式"""
    return {
        "time": f"{seg['start']:.2f}s → {seg['end']:.2f}s",
        "speaker": seg["speaker"],
        "similarity": f"{seg['distance']:.3f}",
        "confidence": f"{seg['confidence']*100:.1f}%",
        "text": seg["text"],
        "word_count": len(seg["words"]),
    }

def run_pipeline(raw_wav: str, max_workers: int = 1):
    # (保持和你一樣，只有 1 條執行緒)
    waveform, sr = torchaudio.load(raw_wav)
    out_dir = pathlib.Path("work_output") / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = sep.separate_and_save(waveform, str(out_dir), segment_index=0)

    logger.info(f"🔄 處理 {len(segments)} 段... (max_workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        bundle = list(ex.map(lambda s: process_segment(*s), segments))

    bundle.sort(key=lambda x: x["start"])

    # -------- 新增 prettified bundle --------
    pretty_bundle = [make_pretty(s) for s in bundle]

    json_path = out_dir / "output.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": bundle}, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Pipeline finished → {json_path}")
    return bundle, pretty_bundle

if __name__ == "__main__":
    import sys
    run_pipeline(sys.argv[1])

