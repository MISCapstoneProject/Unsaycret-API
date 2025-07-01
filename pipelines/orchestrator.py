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


def run_pipeline_dir(directory: str, max_workers: int = 3, out_path: str = "summary.tsv"):
    """Run pipeline on every wav file in a directory and save summary."""
    dir_path = pathlib.Path(directory)
    wav_files = sorted(dir_path.glob("*.wav"))
    results = []
    for idx, wav in enumerate(wav_files, start=1):
        logger.info(f"===== Processing {wav.name} ({idx}/{len(wav_files)}) =====")
        _, _, stats = run_pipeline_file(str(wav), max_workers)
        results.append((idx, stats))

    if results:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("編號\t音檔長度(s)\t總耗時(s)\t分離耗時(s)\tSpeakerID耗時(s)\tASR耗時(s)\n")
            for idx, st in results:
                f.write(
                    f"{idx}\t{st['length']:.2f}\t{st['total']:.2f}\t{st['separate']:.2f}\t{st['speaker']:.2f}\t{st['asr']:.2f}\n"
                )
        logger.info(f"📄 Summary saved to {out_path}")
    else:
        logger.warning("No wav files found in directory")
    return results

# Backwards compatible name
run_pipeline = run_pipeline_file


def run_pipeline_dir(dir_path: str, max_workers: int = 3) -> str:
    """Run pipeline on all audio files in a directory.

    Parameters
    ----------
    dir_path: str
        Directory containing audio files.
    max_workers: int
        Number of workers used for each file.

    Returns
    -------
    str
        Path to the summary TSV file aggregating results of all files.
    """
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path("work_output") / f"batch_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.tsv"
    with open(summary_path, "w", encoding="utf-8") as summary:
        summary.write("file\tstart\tend\tspeaker\tdistance\tconfidence\ttext\n")
        for audio in sorted(pathlib.Path(dir_path).glob("*")):
            if audio.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
                continue
            logger.info(f"🔄 Batch process {audio.name}")
            segments, _ = run_pipeline_file(str(audio), max_workers)
            for seg in segments:
                text = str(seg["text"]).replace("\t", " ")
                summary.write(
                    f"{audio.name}\t{seg['start']}\t{seg['end']}\t{seg['speaker']}\t{seg['distance']}\t{seg['confidence']}\t{text}\n"
                )

    logger.info(f"✅ Directory pipeline finished → {summary_path}")
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
