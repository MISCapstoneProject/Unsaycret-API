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

def _timed_call(func, *args):
    t0 = time.perf_counter()
    res = func(*args)
    return res, time.perf_counter() - t0


def process_segment(seg_path: str, t0: float, t1: float) -> dict:
    """Process a single separated segment."""
    logger.info(
        f"🔧 執行緒 {threading.get_ident()} 處理 ({t0:.2f}-{t1:.2f}) → {os.path.basename(seg_path)}"
    )
    
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as ex:
        spk_future = ex.submit(_timed_call, spk.process_audio_file, seg_path)
        asr_future = ex.submit(_timed_call, asr.transcribe, seg_path)

        # 安全地取得語者識別結果
        spk_result = spk_future.result()
        if spk_result is None:
            logger.error(f"語者識別失敗: {seg_path}")
            speaker_id, name, dist = "unknown", "Unknown", 999.0
            spk_time = 0.0
        else:
            (speaker_id, name, dist), spk_time = spk_result
            
        (text, conf, words), asr_time = asr_future.result()

    logger.info(f"⏱ SpeakerID 耗時 {spk_time:.3f}s")
    logger.info(f"⏱ ASR 耗時 {asr_time:.3f}s")

    # 調整每個詞的時間戳，使其對齊原始混音檔軸
    adjusted_words = []
    for w in words:
        w['start'] = w['start'] + t0
        w['end']   = w['end'] + t0
        adjusted_words.append(w)

    total = time.perf_counter() - start
    logger.info(f"⏱ segment 總耗時 {total:.3f}s")

    return {
        "start": round(t0, 2),
        "end": round(t1, 2),
        "speaker": name,
        "distance": round(float(dist), 3),
        "text": text,
        "confidence": round(conf, 2),
        "words": adjusted_words,
        "spk_time": spk_time,
        "asr_time": asr_time,
    }



def make_pretty(seg: dict) -> dict:
    """Convert a segment dict to human friendly format."""
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
    # ← 把 waveform 傳到 separator 設定的裝置 (cuda or cpu)
    waveform = waveform.to(sep.device)
    audio_len = waveform.shape[1] / sr
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

    spk_time = max((s.get("spk_time", 0.0) for s in bundle), default=0.0)
    asr_time = max((s.get("asr_time", 0.0) for s in bundle), default=0.0)
    pipeline_total = (sep_end - sep_start) + spk_time + asr_time

    stages_path = pathlib.Path("pipeline_stages.csv")
    if not stages_path.exists():
        stages_path.write_text("sep_max,sid_max,asr_max,pipeline_total\n", encoding="utf-8")
    with stages_path.open("a", encoding="utf-8") as f:
        f.write(f"{sep_end - sep_start:.3f},{spk_time:.3f},{asr_time:.3f},{pipeline_total:.3f}\n")

    # 3) 輸出結果
    # 3) 輸出結果
    bundle.sort(key=lambda x: x["start"])
    pretty_bundle = [make_pretty(s) for s in bundle]

    # --- ASR quality metrics ---
    valid = [s for s in bundle if s.get("text")]
    avg_conf = (
        sum(s.get("confidence", 0.0) for s in valid) / len(valid)
        if valid
        else 0.0
    )
    recog_text = " ".join(s.get("text", "") for s in valid)
    ref_text = None
    wav_path = pathlib.Path(raw_wav)
    for ext in (".txt", ".lab"):
        p = wav_path.with_suffix(ext)
        if p.exists():
            ref_text = p.read_text(encoding="utf-8").strip()
            break
    if ref_text is not None:
        wer = compute_wer(ref_text, recog_text)
        cer = compute_cer(ref_text, recog_text)
    else:
        wer = None
        cer = None

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
        "pipeline_total": pipeline_total,
        "avg_conf": avg_conf,
        "wer": wer,
        "cer": cer,
    }

    return bundle, pretty_bundle, stats


def run_pipeline_dir(dir_path: str, max_workers: int = 3) -> str:
    """一次處理資料夾內所有音檔，並將「檔案層級統計」+「段落層級詳情」寫入同一份 TSV。

    只呼叫 `run_pipeline_file()` 一次就同時拿到 `stats` 與 `segments`，避免重複運算。
    回傳值為生成之 `summary.tsv` 的路徑（字串）。
    """

    # === 準備輸出目錄 ===
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path("work_output") / f"batch_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.tsv"
    asr_report_path = out_dir / "asr_report.tsv"

    # === 收集所有支援格式之音檔（含子目錄） ===
    audio_files = [f for f in pathlib.Path(dir_path).rglob("*") if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]
    if not audio_files:
        logger.warning("⚠️  目錄內未找到支援格式音檔")
        return str(summary_path)

    # === 一次處理並暫存結果 ===
    file_results = []  # [(檔案索引, Path, stats, segments)]
    for idx, audio in enumerate(sorted(audio_files), start=1):
        logger.info(f"===== 處理檔案 {audio.name} ({idx}/{len(audio_files)}) =====")
        segments, pretty, stats = run_pipeline_file(str(audio), max_workers)
        file_results.append((idx, audio, stats, segments))

    # === 寫入 TSV ===
    with open(summary_path, "w", encoding="utf-8") as f_sum, open(asr_report_path, "w", encoding="utf-8") as f_asr:
        # -- 檔案層級統計 --
        f_sum.write("編號\t檔名\t音檔長度(s)\t總耗時(s)\t分離耗時(s)\tSpeakerID耗時(s)\tASR耗時(s)\n")
        f_asr.write("編號\t檔名\tASR耗時(s)\t總耗時(s)\t平均confidence\tWER\tCER\n")
        for idx, audio, stats, _ in file_results:
            f_sum.write(
                f"{idx}\t{audio.name}\t{stats['length']:.2f}\t{stats['total']:.2f}\t"
                f"{stats['separate']:.2f}\t{stats['speaker']:.2f}\t{stats['asr']:.2f}\n"
            )
            wer_str = f"{stats['wer']:.4f}" if stats['wer'] is not None else "NA"
            cer_str = f"{stats['cer']:.4f}" if stats['cer'] is not None else "NA"
            f_asr.write(
                f"{idx}\t{audio.name}\t{stats['asr']:.2f}\t{stats['total']:.2f}\t"
                f"{stats['avg_conf']:.4f}\t{wer_str}\t{cer_str}\n"
            )
        # 空行分段
        f_sum.write("\n檔案\t開始(s)\t結束(s)\t說話者\tdistance\tconfidence\t文字\n")
        # -- 段落層級詳情 --
        for _, audio, _, segments in file_results:
            for seg in segments:
                text = str(seg.get("text", "")).replace("\t", " ")
                f_sum.write(
                    f"{audio.name}\t{seg['start']:.3f}\t{seg['end']:.3f}\t"
                    f"{seg['speaker']}\t{seg['distance']:.4f}\t{seg['confidence']:.4f}\t{text}\n"
                )

    logger.info(f"✅ Directory pipeline 完成 → {summary_path}")
    logger.info(f"📊 ASR report → {asr_report_path}")
    return str(summary_path)


def run_pipeline_stream(
    chunk_secs: float = 6.0,
    rate: int = 16000,
    channels: int = 1,
    frames_per_buffer: int = 1024,
    max_workers: int = 2,
    record_secs: float | None = None,
    queue_out: "queue.Queue[dict] | None" = None,
    stop_event: threading.Event | None = None,
    in_bytes_queue: "queue.Queue[bytes] | None" = None,  # 新增：前端傳入的原始 bytes 佇列
):
    """連續錄音或接收前端 bytes，每 `chunk_secs` 秒切片，
    對三位說話人做 SpeakerID + ASR。

    * 若 ``in_bytes_queue`` 為 ``None`` → 使用本機麥克風。
    * 若提供 ``in_bytes_queue`` → 從該佇列讀取 16‑bit PCM bytes，
      適合 WebSocket/APP 上傳串流音訊的情境。
    """

    total_start = time.perf_counter()
    out_root = Path("stream_output") / dt.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    # ─────────────────── 執行緒池 ────────────────────
    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
    futures: list[Future] = []

    # -------------------------------------------------
    # 1. 對單一 chunk 進行分離、SpeakerID、ASR
    # -------------------------------------------------
    def process_chunk(raw_bytes: bytes, idx: int):
        t0 = idx * chunk_secs
        t1 = t0 + chunk_secs

        # 1) 轉 tensor [-1, 1]
        waveform = torch.frombuffer(raw_bytes, dtype=torch.int16).float() / 32768.0
        waveform = waveform.view(1, -1)

        # 2) 儲存混合音檔
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

        # 4) SpeakerID + ASR
        speaker_results: list[dict] = []
        for sp_idx, wav_path in enumerate(speaker_paths, 1):
            res = process_segment(str(wav_path), t0, t1)
            if not res["text"].strip() or res["confidence"] < 0.1:
                continue  # 跳過低信心或空白
            res["speaker_index"] = sp_idx
            speaker_results.append(res)

        # 去重複：同 speaker 名只留信心最高
        unique: dict[str, dict] = {}
        for item in speaker_results:
            n = item["speaker"]
            if n not in unique or item["confidence"] > unique[n]["confidence"]:
                unique[n] = item
        speaker_results = list(unique.values())

        seg_dict = {
            "segment": idx,
            "start": round(t0, 2),
            "end": round(t1, 2),
            "mix": mix_path.as_posix(),
            "sources": [str(p) for p in speaker_paths],
            "speakers": speaker_results,
        }

        # 每段寫一份 JSON
        with open(seg_dir / "output.json", "w", encoding="utf-8") as f:
            json.dump(seg_dict, f, ensure_ascii=False, indent=2)

        # 即時推送給外部佇列（WS/SSE）
        if queue_out is not None:
            queue_out.put(seg_dict)
        return seg_dict

    # -------------------------------------------------
    # 2. 錄音 / 讀取 bytes 的執行緒
    # -------------------------------------------------
    q: queue.Queue[tuple[bytes, int]] = queue.Queue(maxsize=max_workers * 2)
    stop_flag = threading.Event()

    def recorder_from_queue():
        """外部 bytes 來源：從 in_bytes_queue 讀資料"""
        frames_needed = int(rate * chunk_secs) * 2  # bytes
        buf = bytearray()
        idx = 0
        start_time = time.time()
        while not stop_flag.is_set():
            if stop_event and stop_event.is_set():
                break
            try:
                pkt = in_bytes_queue.get(timeout=0.1)
            except queue.Empty:
                # 檢查是否時間到
                if record_secs is not None and time.time() - start_time >= record_secs:
                    stop_flag.set(); break
                continue
            buf.extend(pkt)
            while len(buf) >= frames_needed:
                raw = bytes(buf[:frames_needed]); buf = buf[frames_needed:]
                q.put((raw, idx)); idx += 1

    def recorder_from_mic():
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=frames_per_buffer,
        )
        frames_needed = int(rate * chunk_secs)
        buf = bytearray(); idx = 0; start_time = time.time()
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

    # 啟動正確的 recorder
    rec_thread = threading.Thread(
        target=recorder_from_queue if in_bytes_queue else recorder_from_mic,
        daemon=True,
    )
    rec_thread.start()

    logger.info("🎙 開始錄音/接收 (%s) ...", "外部 bytes" if in_bytes_queue else ("Ctrl‑C" if record_secs is None else f"{record_secs}s"))

    # -------------------------------------------------
    # 3. 主迴圈：把 chunk 交給執行緒池
    # -------------------------------------------------
    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                raw, idx = q.get(timeout=0.1)
            except queue.Empty:
                if stop_flag.is_set():
                    break
                continue
            futures.append(executor.submit(process_chunk, raw, idx))
    except KeyboardInterrupt:
        logger.info("🛑 Ctrl‑C 偵測到使用者手動停止")
        if stop_event:
            stop_event.set()   # ← 讓外部 WebSocket loop 也能收尾
        stop_flag.set()        # ← 通知 recorder thread 結束
    finally:
        stop_flag.set(); rec_thread.join(timeout=1)
        executor.shutdown(wait=True)

    # -------------------------------------------------
    # 4. 聚合結果（可選）
    # -------------------------------------------------
    bundle = [f.result() for f in futures if f.done() and f.result()]
    bundle.sort(key=lambda x: x["start"])

    pretty_bundle: list[dict] = []
    for seg in bundle:
        for sp in seg["speakers"]:
            pretty_bundle.append(make_pretty(sp))

    logger.info("🚩 stream 結束，共 %d 段，耗時 %.3fs → %s", len(bundle), time.perf_counter() - total_start, out_root)
    return bundle, pretty_bundle



# Backwards compatible name
run_pipeline_FILE = run_pipeline_file
run_pipeline_STREAM = run_pipeline_stream
run_pipeline_DIR = run_pipeline_dir


def main():
    parser = argparse.ArgumentParser(description="Speech pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_file = sub.add_parser("file", help="process existing wav file")
    p_file.add_argument("path")
    p_file.add_argument("--workers", type=int, default=4)

    p_stream = sub.add_parser("stream", help="live stream from microphone")
    p_stream.add_argument("--chunk", type=float, default=6.0, help="seconds per chunk")
    p_stream.add_argument("--workers", type=int, default=2)
    p_stream.add_argument("--record_secs", type=float, default=18.0, help="total recording time in seconds (None for infinite)")

    args = parser.parse_args()

    if args.mode == "file":
        run_pipeline_file(args.path, args.workers)
    elif args.mode == "stream":
        run_pipeline_stream(chunk_secs=args.chunk, max_workers=args.workers)

if __name__ == "__main__":    main()