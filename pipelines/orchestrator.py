"""Orchestrate separation, speaker ID and ASR."""

import argparse
import json
import os
import pathlib
import tempfile
import threading
import time
import uuid
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import torch
import torchaudio
import pyaudio  # type: ignore
import wave

from utils.logger import get_logger
from modules.separation.separator import AudioSeparator
from modules.identification.VID_identify_v5 import SpeakerIdentifier
from modules.asr.whisper_asr import WhisperASR

logger = get_logger(__name__)

logger.info("🖥 GPU available: %s", torch.cuda.is_available())
if torch.cuda.is_available():
    logger.info("   Device: %s", torch.cuda.get_device_name(0))


# ---------- 1. 自動偵測 GPU ----------
use_gpu = torch.cuda.is_available()
logger.info(f"🚀 使用設備: {'cuda' if use_gpu else 'cpu'}")

sep = AudioSeparator()
spk = SpeakerIdentifier()
asr = WhisperASR(model_name="medium", gpu=use_gpu)


def process_segment(seg_path: str, t0: float, t1: float) -> dict:
    """Process a single separated segment."""
    logger.info(
        f"🔧 執行緒 {threading.get_ident()} 處理 ({t0:.2f}-{t1:.2f}) → {os.path.basename(seg_path)}"
    )

    start = time.perf_counter()

    t_spk0 = time.perf_counter()
    speaker_id, name, dist = spk.process_audio_file(seg_path)
    t_spk1 = time.perf_counter()
    logger.info(f"⏱ SpeakerID 耗時 {t_spk1 - t_spk0:.3f}s")

    t_asr0 = time.perf_counter()
    text, conf, words = asr.transcribe(seg_path)
    t_asr1 = time.perf_counter()
    logger.info(f"⏱ ASR 耗時 {t_asr1 - t_asr0:.3f}s")

    total = time.perf_counter() - start
    logger.info(f"⏱ segment 總耗時 {total:.3f}s")

    return {
        "start": round(t0, 2),
        "end": round(t1, 2),
        "speaker": name,
        "distance": round(float(dist), 3),
        "text": text,
        "confidence": round(conf, 2),
        "words": words,
    }


def make_pretty(seg: dict) -> dict:
    """Convert a segment dict to human friendly format."""
    return {
        "time": f"{seg['start']:.2f}s → {seg['end']:.2f}s",
        "speaker": seg["speaker"],
        "similarity": f"{seg['distance']:.3f}",
        "confidence": f"{seg['confidence']*100:.1f}%",
        "text": seg["text"],
        "word_count": len(seg["words"]),
    }


def run_pipeline_file(raw_wav: str, max_workers: int = 3):
    """Run pipeline on an existing wav file."""
    total_start = time.perf_counter()

    waveform, sr = torchaudio.load(raw_wav)
    out_dir = pathlib.Path("work_output") / dt.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    sep_start = time.perf_counter()
    segments = sep.separate_and_save(waveform, str(out_dir), segment_index=0)
    sep_end = time.perf_counter()
    logger.info(f"⏱ 分離耗時 {sep_end - sep_start:.3f}s, 共 {len(segments)} 段")
    
    logger.info(f"🔄 處理 {len(segments)} 段... (max_workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        bundle = list(ex.map(lambda s: process_segment(*s), segments))

    bundle.sort(key=lambda x: x["start"])
    pretty_bundle = [make_pretty(s) for s in bundle]

    json_path = out_dir / "output.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": bundle}, f, ensure_ascii=False, indent=2)

    total_end = time.perf_counter()
    logger.info(f"✅ Pipeline finished → {json_path} (總耗時 {total_end - total_start:.3f}s)")
    return bundle, pretty_bundle


# def record_to_wav(path: str, duration: int = 5, rate: int = 44100) -> None:
#     """Record microphone input for a fixed duration and save to wav."""
#     pa = pyaudio.PyAudio()
#     stream = pa.open(
#         format=pyaudio.paInt16,
#         channels=1,
#         rate=rate,
#         input=True,
#         frames_per_buffer=1024,
#     )

#     frames = []
#     for _ in range(int(rate / 1024 * duration)):
#         frames.append(stream.read(1024))

#     stream.stop_stream()
#     stream.close()
#     pa.terminate()

#     wf = wave.open(path, "wb")
#     wf.setnchannels(1)
#     wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
#     wf.setframerate(rate)
#     wf.writeframes(b"".join(frames))
#     wf.close()


# def run_pipeline_record(duration: int = 5, max_workers: int = 4):
#     """Record audio then run the standard pipeline."""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         record_to_wav(tmp.name, duration)
#         tmp_path = tmp.name

#     try:
#         return run_pipeline_file(tmp_path, max_workers)
#     finally:
#         os.remove(tmp_path)


# Backwards compatible name
run_pipeline = run_pipeline_file


def main():
    parser = argparse.ArgumentParser(description="Speech pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_file = sub.add_parser("file", help="process existing wav file")
    p_file.add_argument("path")
    p_file.add_argument("--workers", type=int, default=4)

    # p_rec = sub.add_parser("record", help="record from microphone")
    # p_rec.add_argument("--duration", type=int, default=5)
    # p_rec.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    if args.mode == "file":
        run_pipeline_file(args.path, args.workers)
    # else:
    #     run_pipeline_record(args.duration, args.workers)


if __name__ == "__main__":
    main()

