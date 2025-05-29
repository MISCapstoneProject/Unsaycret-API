# pipelines/orchestrator.py
# 在文件最顶部
import torch, os
print("🖥  GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("   Device:", torch.cuda.get_device_name(0))
# 然后再 import 你的模块

import threading
import json, uuid, pathlib, datetime as dt
import torchaudio
from concurrent.futures import ThreadPoolExecutor
from modules.separation.separator import AudioSeparator
from modules.speaker_id.v4 import SpeakerIdentifier
from modules.asr.whisper_asr import WhisperASR

# ---------- 1. 自動偵測 GPU ----------
use_gpu = torch.cuda.is_available()
print(f"🚀 使用設備: {'cuda' if use_gpu else 'cpu'}")


sep = AudioSeparator()
spk = SpeakerIdentifier()
asr = WhisperASR(model_name="medium", gpu=use_gpu)

# ---------- 3. 處理單一片段的函式 ----------

def process_segment(seg_path, t0, t1):
    print(f"🔧 執行緒 {threading.get_ident()} 正在處理 segment ({t0:.2f} - {t1:.2f})")

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
def run_pipeline(raw_wav: str, max_workers: int = 4):
    # 載入音訊檔
    waveform, sr = torchaudio.load(raw_wav)

    # 建立輸出資料夾
    out_dir = pathlib.Path("work_output") / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 語音分離 → 得到多個音檔段落
    segments = sep.separate_and_save(waveform, str(out_dir), segment_index=0)
    # segments: [(seg_path, t0, t1), ...]

    print(f"🔄 處理 {len(segments)} 個音檔片段...")

    # ---------- 5. 多執行緒處理每一段 ----------
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        bundle = list(executor.map(lambda s: process_segment(*s), segments))

    # ---------- 6. 排序（保險） ----------
    bundle.sort(key=lambda x: x["start"])

    # ---------- 7. 寫入 JSON ----------
    json_path = out_dir / "output.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": bundle}, f, ensure_ascii=False, indent=2)

    print(f"✅ Pipeline finished → {json_path}")
    return bundle

if __name__ == "__main__":
    import sys
    run_pipeline(sys.argv[1])

