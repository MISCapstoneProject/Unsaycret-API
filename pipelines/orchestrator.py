import argparse
import json
import os
import pathlib
import tempfile
import threading
import time
import queue
from pathlib import Path
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor, Future

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

    # 1) 分離
    sep_start = time.perf_counter()
    segments = sep.separate_and_save(waveform, str(out_dir), segment_index=0)
    if not segments:                           # ← 新增
        logger.error("🚨 語者分離失敗：回傳空值 / None")
        raise RuntimeError("Speaker separation failed – no segments returned")
    sep_end = time.perf_counter()
    logger.info(f"⏱ 分離耗時 {sep_end - sep_start:.3f}s, 共 {len(segments)} 段")

    # 2) 多執行緒處理所有段
    logger.info(f"🔄 處理 {len(segments)} 段... (max_workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        bundle = [r for r in ex.map(lambda s: process_segment(*s), segments) if r]

    # 3) 輸出結果
    bundle.sort(key=lambda x: x["start"])
    pretty_bundle = [make_pretty(s) for s in bundle]

    json_path = out_dir / "output.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": bundle}, f, ensure_ascii=False, indent=2)

    total_end = time.perf_counter()
    logger.info(f"✅ Pipeline finished → {json_path} (總耗時 {total_end - total_start:.3f}s)")
    return bundle, pretty_bundle


def run_pipeline_stream(
    chunk_secs: float = 6.0,
    rate: int = 16000,
    channels: int = 1,
    frames_per_buffer: int = 1024,
    max_workers: int = 2,
    record_secs: float | None = None,
    queue_out: "queue.Queue[dict] | None" = None,
    stop_event: threading.Event | None = None
):
    """連續錄音，每 `chunk_secs` 秒切片，對**三位說話人**都做 SpeakerID + ASR。"""

    total_start = time.perf_counter()
    out_root = Path("stream_output") / dt.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    # ── 1) 執行緒池 ────────────────────────────────────────────────
    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
    futures: list[Future] = []

    def process_chunk(raw_bytes: bytes, idx: int):
        """對 chunk 內的所有 speaker wav 做 SpeakerID + ASR，回傳段層級 dict"""
        t0 = idx * chunk_secs
        t1 = t0 + chunk_secs

        # 1) Tensor
        waveform = (
            torch.frombuffer(raw_bytes, dtype=torch.int16).float().view(1, -1) / 32768.0
        )

        # 2) 輸出資料夾
        seg_dir = out_root / f"segment_{idx:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        mix_path = seg_dir / "mix.wav"
        torchaudio.save(mix_path.as_posix(), waveform, rate)

        # 3) 語者分離
        sep.separate_and_save(waveform, seg_dir.as_posix(), segment_index=idx)
        speaker_paths = sorted(seg_dir.glob("speaker*.wav"))
        if not speaker_paths:
            logger.warning("segment %d 無 speaker wav", idx)
            return None

        # 4) 每位 speaker 做辨識 + ASR
        speaker_results = []
        for sp_idx, wav_path in enumerate(speaker_paths, 1):
            res = process_segment(str(wav_path), t0, t1)

            # --- B. 跳過空白文字 / 低信心 ---
            if not res["text"].strip() or res["confidence"] < 0.1:
                continue


            res["speaker_index"] = sp_idx
            speaker_results.append(res)
        
        # --- C. 依 speaker name 去重複，只留最高信心 ---
        unique = {}
        for item in speaker_results:
            name = item["speaker"]
            if name not in unique or item["confidence"] > unique[name]["confidence"]:
                unique[name] = item
        speaker_results = list(unique.values())

        seg_dict = {
            "segment": idx,
            "start": round(t0, 2),
            "end": round(t1, 2),
            "mix": mix_path.as_posix(),
            "sources": [str(p) for p in speaker_paths],
            "speakers": speaker_results,
        }

        with open(seg_dir / "output.json", "w", encoding="utf-8") as f:
            json.dump(seg_dict, f, ensure_ascii=False, indent=2)

        if queue_out is not None:   # 即時推送給 WS
            queue_out.put(seg_dict)
        return seg_dict

    # ── 2) 錄音執行緒 ────────────────────────────────────────────
    q: queue.Queue[tuple[bytes, int]] = queue.Queue(maxsize=max_workers * 2)
    stop_flag = threading.Event()

    def recorder():
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=frames_per_buffer,
        )
        frames_needed = int(rate * chunk_secs)
        buf = bytearray()
        idx = 0
        start_time = time.time()
        try:
            while not stop_flag.is_set():
                if stop_event and stop_event.is_set():
                    break
                if record_secs is not None and time.time() - start_time >= record_secs:
                    stop_flag.set(); break
                buf.extend(stream.read(frames_per_buffer, exception_on_overflow=False))
                if len(buf) // 2 >= frames_needed:
                    raw = bytes(buf[: frames_needed * 2]); buf = buf[frames_needed * 2 :]
                    q.put((raw, idx)); idx += 1
        finally:
            stream.stop_stream(); stream.close(); pa.terminate()

    threading.Thread(target=recorder, daemon=True).start()
    logger.info("🎙 開始錄音 (%s) ...", "Ctrl‑C" if record_secs is None else f"{record_secs}s")

    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                raw, idx = q.get(timeout=0.1)
            except queue.Empty:
                # 如果錄音執行緒已停，且佇列也空，就離開迴圈
                if stop_flag.is_set():
                    break
                continue
            futures.append(executor.submit(process_chunk, raw, idx))
    except KeyboardInterrupt:
        logger.info("🛑 Ctrl‑C 偵測到使用者手動停止")
    finally:
        stop_flag.set()
        executor.shutdown(wait=True)

    # ── 3) 聚合結果 ────────────────────────────────────────────
    bundle = [f.result() for f in futures if f.done() and f.result()]
    bundle.sort(key=lambda x: x["start"])

    # pretty version：把 speaker list 攤平成行
    pretty_bundle: list[dict] = []
    for seg in bundle:
        for sp in seg["speakers"]:
            pretty_bundle.append(make_pretty(sp))

    # json_path = out_root / "output.json" # 全域 segment
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump({"segments": bundle}, f, ensure_ascii=False, indent=2)

    logger.info(
        "🚩 stream 結束，共 %d 段，總耗時 %.3fs → %s",
        len(bundle), time.perf_counter() - total_start, out_root,
    )
    return bundle, pretty_bundle


# Backwards compatible name
run_pipeline_FILE = run_pipeline_file
run_pipeline_STREAM = run_pipeline_stream


def main():
    parser = argparse.ArgumentParser(description="Speech pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_file = sub.add_parser("file", help="process existing wav file")
    p_file.add_argument("path")
    p_file.add_argument("--workers", type=int, default=4)

    p_stream = sub.add_parser("stream", help="live stream from microphone")
    p_stream.add_argument("--chunk", type=float, default=6.0, help="seconds per chunk")
    p_stream.add_argument("--workers", type=int, default=2)

    args = parser.parse_args()

    if args.mode == "file":
        run_pipeline_file(args.path, args.workers)
    elif args.mode == "stream":
        run_pipeline_stream(chunk_secs=args.chunk, max_workers=args.workers)

if __name__ == "__main__":
    main()