import argparse
import json
import os
import pathlib
import tempfile
import threading
import time
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
    """Process a single separated segment and return metrics."""
    logger.info(
        f"🔧 執行緒 {threading.get_ident()} 處理 ({t0:.2f}-{t1:.2f}) → {os.path.basename(seg_path)}"
    )
    
    start = time.perf_counter()
    
    t_spk0 = time.perf_counter()
    speaker_id, name, dist = spk.process_audio_file(seg_path)
    t_spk1 = time.perf_counter()
    spk_time = t_spk1 - t_spk0
    logger.info(f"⏱ SpeakerID 耗時 {spk_time:.3f}s")

    t_asr0 = time.perf_counter()
    text, conf, words = asr.transcribe(seg_path)
    t_asr1 = time.perf_counter()
    asr_time = t_asr1 - t_asr0
    logger.info(f"⏱ ASR 耗時 {asr_time:.3f}s")

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
        "spk_time": spk_time,
        "asr_time": asr_time,
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
    audio_len = waveform.shape[1] / sr
    out_dir = pathlib.Path("work_output") / dt.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 分離
    sep_start = time.perf_counter()
    segments = sep.separate_and_save(waveform, str(out_dir), segment_index=0)
    sep_end = time.perf_counter()
    logger.info(f"⏱ 分離耗時 {sep_end - sep_start:.3f}s, 共 {len(segments)} 段")

    # 2) 多執行緒處理所有段
    logger.info(f"🔄 處理 {len(segments)} 段... (max_workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        bundle = [r for r in ex.map(lambda s: process_segment(*s), segments) if r]
    spk_time = sum(s.get("spk_time", 0.0) for s in bundle)
    asr_time = sum(s.get("asr_time", 0.0) for s in bundle)

    # 3) 輸出結果
    bundle.sort(key=lambda x: x["start"])
    pretty_bundle = [make_pretty(s) for s in bundle]

    json_path = out_dir / "output.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": bundle}, f, ensure_ascii=False, indent=2)

    total_end = time.perf_counter()
    total_time = total_end - total_start
    logger.info(f"✅ Pipeline finished → {json_path} (總耗時 {total_time:.3f}s)")

    stats = {
        "length": audio_len,
        "total": total_time,
        "separate": sep_end - sep_start,
        "speaker": spk_time,
        "asr": asr_time,
    }

    return bundle, pretty_bundle, stats


def run_pipeline_dir(dir_path: str, max_workers: int = 3) -> str:
    """
    Run pipeline on all audio files in a directory,
    then save a TSV summary with both file-level stats and per-segment details.

    Parameters
    ----------
    dir_path: str
        Directory containing audio files (.wav, .mp3, .flac, .ogg).
    max_workers: int
        Number of parallel workers per file.

    Returns
    -------
    str
        Path to the generated summary TSV file.
    """
    # 準備輸出目錄
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path("work_output") / f"batch_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.tsv"
    # 收集所有支援格式的音檔
    audio_paths = sorted(pathlib.Path(dir_path).rglob("*"))
    audio_files = [f for f in audio_paths if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]
    total_files = len(audio_files)

    with open(summary_path, "w", encoding="utf-8") as f:
        # 1) 檔案層級統計
        f.write("編號\t檔名\t音檔長度(s)\t總耗時(s)\t分離耗時(s)\tSpeakerID耗時(s)\tASR耗時(s)\n")
        for idx, audio in enumerate(audio_files, start=1):
            logger.info(f"===== 處理檔案 {audio.name} ({idx}/{total_files}) =====")
            bundle, pretty, stats = run_pipeline_file(str(audio), max_workers)
            f.write(
                f"{idx}\t{audio.name}\t"
                f"{stats['length']:.2f}\t{stats['total']:.2f}\t{stats['separate']:.2f}\t"
                f"{stats['speaker']:.2f}\t{stats['asr']:.2f}\n"
            )

        # 2) 段落層級詳情
        f.write("\n檔案\t開始(s)\t結束(s)\t說話者\tdistance\tconfidence\t文字\n")
        for audio in audio_files:
            # 再次呼叫以取得分段內容
            segments, pretty, stats = run_pipeline_file(str(audio), max_workers)
            for seg in segments:
                text = str(seg.get("text", "")).replace("\t", " ")
                f.write(
                    f"{audio.name}\t{seg['start']:.3f}\t{seg['end']:.3f}\t"
                    f"{seg['speaker']}\t{seg['distance']:.4f}\t{seg['confidence']:.4f}\t{text}\n"
                )

    logger.info(f"✅ Directory pipeline 完成 → {summary_path}")
    return str(summary_path)


def main():
    parser = argparse.ArgumentParser(description="Speech pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_file = sub.add_parser("file", help="process existing wav file")
    p_file.add_argument("path")
    p_file.add_argument("--workers", type=int, default=4)

    p_dir = sub.add_parser("dir", help="process all wav files in a directory")
    p_dir.add_argument("path")
    p_dir.add_argument("--workers", type=int, default=4)
    p_dir.add_argument("--out", default="summary.tsv")

    args = parser.parse_args()

    if args.mode == "file":
        run_pipeline_file(args.path, args.workers)
    elif args.mode == "dir":
        run_pipeline_dir(args.path, args.workers, args.out)

if __name__ == "__main__":
    main()
