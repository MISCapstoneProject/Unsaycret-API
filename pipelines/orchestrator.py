import argparse
import json
import os
import pathlib
import threading
import time
import queue
from pathlib import Path
from datetime import datetime as dt, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, Future

# 修復 SVML 錯誤：在導入 PyTorch 之前設定環境變數
os.environ["MKL_DISABLE_FAST_MM"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchaudio
import pyaudio  # type: ignore
import numpy as np
from scipy.signal import resample_poly

from utils.logger import get_logger
from utils.constants import DEFAULT_WHISPER_MODEL,DEFAULT_WHISPER_BEAM_SIZE, USE_DIARIZATION_STREAMING
from utils.env_config import FORCE_CPU, CUDA_DEVICE_INDEX
from modules.separation.separator import AudioSeparator
from modules.identification.VID_identify_v5 import SpeakerIdentifier
from modules.asr.whisper_asr import WhisperASR
from modules.asr.text_utils import compute_cer, compute_wer

logger = get_logger(__name__)

logger.info("🖥 GPU available: %s", torch.cuda.is_available())
if torch.cuda.is_available():
    logger.info("   Device: %s", torch.cuda.get_device_name(0))

# ---------- 1. GPU/CPU 設備選擇 ----------
def init_pipeline_modules():
    """初始化 sep / spk / asr，考慮 CUDA 設定，並回傳模組實例們"""
    current_cuda_device = CUDA_DEVICE_INDEX  # 建立本地變數避免修改全域變數

    if FORCE_CPU:
        use_gpu = False
        logger.info("🔧 FORCE_CPU=true，強制使用 CPU")
    else:
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            if current_cuda_device < torch.cuda.device_count():
                torch.cuda.set_device(current_cuda_device)
                logger.info(f"🎯 設定 CUDA 設備索引: {current_cuda_device}")
                logger.info(f"   使用設備: {torch.cuda.get_device_name(current_cuda_device)}")
            else:
                logger.warning(f"⚠️  CUDA 設備索引 {current_cuda_device} 不存在，使用預設設備 0")
                current_cuda_device = 0
                torch.cuda.set_device(current_cuda_device)
                logger.info(f"   已設定為設備 0: {torch.cuda.get_device_name(0)}")

    logger.info(f"🚀 使用設備: {'cuda:' + str(current_cuda_device) if use_gpu else 'cpu'}")

    sep = AudioSeparator()
    spk = SpeakerIdentifier()
    asr = WhisperASR(model_name=DEFAULT_WHISPER_MODEL, gpu=use_gpu, beam=DEFAULT_WHISPER_BEAM_SIZE)

    return sep, spk, asr,use_gpu

def _timed_call(func, *args):
    t0 = time.perf_counter()
    res = func(*args)
    return res, time.perf_counter() - t0


def process_segment(seg_path: str, t0: float, t1: float, absolute_timestamp: float = None, *, sep=None, spk=None, asr=None) -> dict:
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

    result = {
        "start": round(t0, 2),
        "end": round(t1, 2),
        "speaker": name,
        "speaker_id": speaker_id,
        "distance": round(float(dist), 3),
        "text": text,
        "confidence": round(conf, 2),
        "words": adjusted_words,
        "spk_time": spk_time,
        "asr_time": asr_time,
        "path": seg_path
    }
    
    # 如果有絕對時間戳，加入到結果中
    if absolute_timestamp is not None:
        result["absolute_timestamp"] = absolute_timestamp
        taipei_tz = timezone(timedelta(hours=8))
        result["absolute_start_time"] = dt.fromtimestamp(absolute_timestamp, tz=taipei_tz).isoformat()
        result["absolute_end_time"] = dt.fromtimestamp(absolute_timestamp + (t1 - t0), tz=taipei_tz).isoformat()
    
    # 🆕 更新 output.json（加入 ASR 結果）
    try:
        segment_dir = os.path.dirname(seg_path)
        json_path = os.path.join(segment_dir, "output.json")
        
        if os.path.exists(json_path):
            # 讀取現有 metadata
            with open(json_path, "r", encoding="utf-8") as f:
                segment_meta = json.load(f)
            
            # 🆕 加入 ASR 和識別結果
            segment_meta.update({
                "speaker_id": speaker_id,
                "speaker_name": name,
                "speaker_distance": round(float(dist), 3),
                "transcription": {
                    "text": text,
                    "confidence": round(conf, 2),
                    "language": "zh",  # 可以從 ASR 取得
                    "words": [
                        {
                            "word": w["word"],
                            "start": round(w["start"], 3),
                            "end": round(w["end"], 3),
                            "probability": round(w.get("probability", 0.0), 3)
                        }
                        for w in adjusted_words
                    ]
                },
                "processing_time": {
                    "speaker_id_seconds": round(spk_time, 3),
                    "asr_seconds": round(asr_time, 3),
                    "total_seconds": round(total, 3)
                }
            })
            
            # 寫回檔案
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(segment_meta, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"已更新 {json_path} 加入 ASR 結果")
        else:
            logger.warning(f"找不到 output.json: {json_path}")
            
    except Exception as e:
        logger.error(f"更新 output.json 失敗: {e}")
    
    return result



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


def run_pipeline_file(raw_wav: str, max_workers: int = 3, sep=None, spk=None, asr=None):
    """Run pipeline on an existing wav file.

    If the separator / speaker identifier / ASR modules are not provided,
    they will be initialized automatically. This preserves backwards
    compatibility for callers that import :func:`run_pipeline_file` directly
    without using :func:`init_pipeline_modules` first (e.g. older API code).
    """

    # Allow legacy usage where modules are not injected explicitly.
    if sep is None or spk is None or asr is None:
        sep, spk, asr, _ = init_pipeline_modules()

    total_start = time.perf_counter()

    waveform, sr = torchaudio.load(raw_wav)
    
    # 確保單聲道：如果是多聲道，轉為單聲道（取平均）
    if waveform.shape[0] > 1:
        logger.info(f"🔄 多聲道音檔 ({waveform.shape[0]} 聲道) → 單聲道")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 如果採樣率不等於16000就重採樣（使用高品質 scipy resample_poly）
    if sr != 16000:
        logger.info(f"🔄 採樣率 {sr} ≠ 16000，進行重採樣")
        # 轉換為 numpy 進行高品質重採樣
        waveform_np = waveform.cpu().numpy()
        # 此時已確保是單聲道，直接處理
        resampled = resample_poly(waveform_np.squeeze(), 16000, sr)
        waveform = torch.from_numpy(resampled).unsqueeze(0)
        sr = 16000
    
    # ← 把 waveform 傳到 separator 設定的裝置 (cuda or cpu)
    waveform = waveform.to(sep.device)
    audio_len = waveform.shape[1] / sr
    out_dir = pathlib.Path("work_output") / dt.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 分離
    sep_start = time.perf_counter()
    file_start_time = dt.now()  # 記錄處理檔案的開始時間
    segments = sep.separate_and_save(waveform, str(out_dir), segment_index=0, absolute_start_time=file_start_time)
    if not segments:                           # ← 新增
        logger.error("🚨 語者分離失敗：回傳空值 / None")
        raise RuntimeError("Speaker separation failed – no segments returned")
    sep_end = time.perf_counter()
    logger.info(f"⏱ 分離耗時 {sep_end - sep_start:.3f}s, 共 {len(segments)} 段")

    # 2) 多執行緒處理所有段（同時支援含/不含 absolute_timestamp，並注入 sep/spk/asr）
    # 釋放分離步驟暫佔的 VRAM，避免後續 ASR/SpkID 爆顯存
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"torch.cuda.empty_cache skipped: {e}")

    logger.info(f"🔄 處理 {len(segments)} 段... (max_workers={max_workers})")


    def _run(seg):
        if len(seg) >= 4:
            # 新格式：(path, start, end, absolute_timestamp)
            return process_segment(seg[0], seg[1], seg[2], seg[3],
                                sep=sep, spk=spk, asr=asr)
        else:
            # 舊格式：(path, start, end)
            return process_segment(seg[0], seg[1], seg[2],
                                sep=sep, spk=spk, asr=asr)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        bundle = [
            r for r in ex.map(_run, segments) if r
        ]

    spk_time = max((s.get("spk_time", 0.0) for s in bundle), default=0.0)
    asr_time = max((s.get("asr_time", 0.0) for s in bundle), default=0.0)
    pipeline_total = (sep_end - sep_start) + spk_time + asr_time



    stages_path = pathlib.Path("pipeline_stages.csv")
    if not stages_path.exists():
        stages_path.write_text("sep_max,sid_max,asr_max,pipeline_total\n", encoding="utf-8")
    with stages_path.open("a", encoding="utf-8") as f:
        f.write(f"{sep_end - sep_start:.3f},{spk_time:.3f},{asr_time:.3f},{pipeline_total:.3f}\n")

    # 3) 輸出結果 + ASR 品質指標
    bundle.sort(key=lambda x: x["start"])
    pretty_bundle = [make_pretty(s) for s in bundle]

    # ### PATCH START: quality metrics ###
    valid = [s for s in bundle if s.get("text")]
    avg_conf = (sum(s.get("confidence", 0.0) for s in valid) / len(valid)) if valid else 0.0
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
    # ### PATCH END ###

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

# ───────────────────────── Dir Mode ─────────────────────────
def load_truth_map(path: str) -> dict[str, str]:
    """讀取 filename<TAB>transcript 的對照表。"""
    m: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            fname, txt = line.split("\t", 1)
            m[fname] = txt.strip()
    return m


def run_pipeline_dir(
    dir_path: str,
    truth_map_path: str = "truth_map.txt",
    max_workers: int = 3, sep=None, spk=None, asr=None
) -> str:
    """
    批次處理資料夾內所有音檔，輸出：
      - summary.tsv：檔案級統計 + 段落詳情
      - asr_report.tsv：ASR 指標 (avg_conf, WER, CER)
    """
    
    # Lazily initialize modules for legacy callers that do not provide them.
    if sep is None or spk is None or asr is None:
        sep, spk, asr, _ = init_pipeline_modules()

    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path("work_output") / f"batch_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.tsv"
    asr_report_path = out_dir / "asr_report.tsv"

    truth_map = load_truth_map(truth_map_path) if truth_map_path and os.path.exists(truth_map_path) else {}

    audio_files = [
        f for f in pathlib.Path(dir_path).rglob("*")
        if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
    ]
    if not audio_files:
        logger.warning("⚠️  目錄內未找到支援格式音檔")
        return str(summary_path)

    file_results: list[tuple[int, Path, dict, list[dict]]] = []
    for idx, audio in enumerate(sorted(audio_files), start=1):
        logger.info(f"===== 處理檔案 {audio.name} ({idx}/{len(audio_files)}) =====")
        segments, pretty, stats = run_pipeline_file(str(audio), max_workers , sep=sep, spk=spk, asr=asr)

        # 用 truth_map 覆寫 WER/CER
        gt = truth_map.get(audio.name)
        if gt:
            recog_text = " ".join(s.get("text", "") for s in segments if s.get("text"))
            stats["wer"] = compute_wer(gt, recog_text)
            stats["cer"] = compute_cer(gt, recog_text)

        file_results.append((idx, audio, stats, segments))

    with open(summary_path, "w", encoding="utf-8") as f_sum, \
         open(asr_report_path, "w", encoding="utf-8") as f_asr:

        # 檔案級統計
        f_sum.write("編號\t檔名\t音檔長度(s)\t總耗時(s)\t分離耗時(s)\tSpeakerID耗時(s)\tASR耗時(s)\n")
        # ASR 報表
        f_asr.write("編號\t檔名\tASR耗時(s)\t總耗時(s)\t平均confidence\tWER\tCER\n")

        for idx, audio, stats, segments in file_results:
            wer_str = f"{stats['wer']:.4f}" if stats.get("wer") is not None else "NA"
            cer_str = f"{stats['cer']:.4f}" if stats.get("cer") is not None else "NA"

            f_sum.write(
                f"{idx}\t{audio.name}\t{stats['length']:.2f}\t{stats['total']:.2f}\t"
                f"{stats['separate']:.2f}\t{stats['speaker']:.2f}\t{stats['asr']:.2f}\n"
            )
            f_asr.write(
                f"{idx}\t{audio.name}\t{stats['asr']:.2f}\t{stats['total']:.2f}\t"
                f"{stats.get('avg_conf', 0.0):.4f}\t{wer_str}\t{cer_str}\n"
            )

        # 段落詳情
        f_sum.write("\n檔案\t開始(s)\t結束(s)\t說話者\tdistance\tconfidence\t文字\n")
        for _, audio, _, segments in file_results:
            for seg in segments:
                text = seg.get("text", "").replace("\t", " ")
                f_sum.write(
                    f"{audio.name}\t{seg['start']:.3f}\t{seg['end']:.3f}\t"
                    f"{seg['speaker']}\t{seg.get('distance', 0.0):.4f}\t"
                    f"{seg['confidence']:.4f}\t{text}\n"
                )

    logger.info(f"✅ Directory pipeline 完成 → {summary_path}")
    logger.info(f"📊 ASR report → {asr_report_path}")
    return str(summary_path)

# ───────────────────────── Stream Mode ─────────────────────────
def run_pipeline_stream(
    chunk_secs: float = None,
    rate: int = 16000,
    channels: int = 1,
    frames_per_buffer: int = 1024,
    max_workers: int = 2,
    record_secs: float | None = None,
    queue_out: "queue.Queue[dict] | None" = None,
    stop_event: threading.Event | None = None,
    in_bytes_queue: "queue.Queue[bytes] | None" = None,
    sep=None, spk=None, asr=None
):
    """串流模式：每 chunk_secs 做一次分離/識別/ASR。"""

    # Initialize modules when not supplied to maintain backward compatibility
    if sep is None or spk is None or asr is None:
        sep, spk, asr, _ = init_pipeline_modules()

    # 初始化 VAD 驅動的串流狀態
    sep.reset_streaming_state(
        silence_threshold=2,      # 靜音 0.8 秒觸發處理
        max_buffer_duration=15.0,   # 最長 15 秒強制處理
        vad_threshold=0.02          # VAD 能量閾值
    )
    
    total_start = time.perf_counter()
    out_root = Path("stream_output") / dt.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
    futures: list[Future] = []
    # 使用台北時間作為串流開始時間 (UTC+8)
    taipei_tz = timezone(timedelta(hours=8))
    stream_start_time = dt.now(taipei_tz)

    all_results = []  # 累積所有處理結果

    def process_chunk_vad(raw_bytes: bytes, chunk_idx: int, chunk_time: dt):
        """處理單個音訊 chunk（VAD 驅動）"""
        audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 呼叫 VAD 驅動的處理方法
        results = sep.process_audio_chunk_streaming(
            audio_chunk=audio_data,
            output_dir=str(out_root),
            absolute_time=chunk_time
        )
        
        # 如果觸發了處理（有結果）
        if results:
            speaker_results = []
            for path, start_rel, end_rel, abs_ts in results:
                try:
                    # 🆕 讀取對應的 output.json
                    segment_dir = os.path.dirname(path)
                    json_path = os.path.join(segment_dir, "output.json")
                    
                    segment_meta = {}
                    if os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as f:
                            segment_meta = json.load(f)
                    
                    # 執行 Speaker ID + ASR
                    res = process_segment(
                        str(path), 
                        start_rel, 
                        end_rel, 
                        abs_ts,
                        sep=sep, spk=spk, asr=asr
                    )
                    
                    if res["text"].strip() and res["confidence"] >= 0.1:
                        # 🆕 合併 metadata
                        res.update({
                            "segment_index": segment_meta.get("segment_index"),
                            "speaker_label": segment_meta.get("speaker_label"),
                            "audio_file": os.path.basename(path),
                            "segment_dir": os.path.basename(segment_dir),
                        })
                        speaker_results.append(res)
                        
                except Exception as e:
                    logger.error(f"處理音檔 {path} 失敗: {e}")
            
            # 去重（同 speaker 保留最高信心）
            unique = {}
            for item in speaker_results:
                name = item["speaker"]
                if name not in unique or item["confidence"] > unique[name]["confidence"]:
                    unique[name] = item
            
            if unique:
                seg_dict = {
                    "chunk_index": chunk_idx,
                    "timestamp": chunk_time.isoformat(),
                    "speakers": list(unique.values()),
                }
                
                if queue_out is not None:
                    queue_out.put(seg_dict)
                
                return seg_dict
        
        return None

    # 錄音執行緒（逐 chunk 處理）
    q = queue.Queue(maxsize=max_workers * 2)
    stop_flag = threading.Event()

    def recorder_from_mic():
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=frames_per_buffer,
        )
        
        chunk_idx = 0
        start_time = time.time()
        
        try:
            while not stop_flag.is_set():
                if stop_event and stop_event.is_set():
                    break
                if record_secs is not None and time.time() - start_time >= record_secs:
                    stop_flag.set()
                    break
                
                raw = stream.read(frames_per_buffer, exception_on_overflow=False)
                chunk_time = dt.now(taipei_tz)
                q.put((raw, chunk_idx, chunk_time))
                chunk_idx += 1
                
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    rec_thread = threading.Thread(target=recorder_from_mic, daemon=True)
    rec_thread.start()

    logger.info("🎙 開始錄音（VAD 驅動斷句模式）...")

    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                raw, idx, chunk_time = q.get(timeout=0.1)
            except queue.Empty:
                if stop_flag.is_set():
                    break
                continue
            
            result = process_chunk_vad(raw, idx, chunk_time)
            if result:
                all_results.append(result)
                
    except KeyboardInterrupt:
        logger.info("🛑 Ctrl-C 偵測到使用者手動停止")
        if stop_event:
            stop_event.set()
        stop_flag.set()
    finally:
        stop_flag.set()
        rec_thread.join(timeout=1)
        
        # 🆕 強制處理剩餘緩衝區
        logger.info("處理剩餘緩衝區...")
        final_results = sep.force_process_buffer(
            output_dir=str(out_root),
            absolute_time=dt.now(taipei_tz)
        )

        if final_results:
            speaker_results = []
            for path, start_rel, end_rel, abs_ts in final_results:
                try:
                    # 🆕 讀取並更新 output.json
                    segment_dir = os.path.dirname(path)
                    json_path = os.path.join(segment_dir, "output.json")
                    
                    res = process_segment(str(path), start_rel, end_rel, abs_ts,
                                        sep=sep, spk=spk, asr=asr)
                    
                    if res["text"].strip() and res["confidence"] >= 0.1:
                        speaker_results.append(res)
                        
                except Exception as e:
                    logger.error(f"處理最終音檔 {path} 失敗: {e}")
            
            if speaker_results:
                final_dict = {
                    "chunk_index": "final",
                    "timestamp": dt.now(taipei_tz).isoformat(),
                    "speakers": speaker_results,
                }
                all_results.append(final_dict)
                if queue_out is not None:
                    queue_out.put(final_dict)
    
    logger.info(
        f"串流結束，共處理 {len(all_results)} 個片段，耗時 {time.perf_counter() - total_start:.3f}s → {out_root}"
    )
    
    # 生成統一格式的回傳
    pretty_bundle = []
    for result in all_results:
        for sp in result.get("speakers", []):
            pretty_bundle.append(make_pretty(sp))
    
    return all_results, pretty_bundle


# 兼容舊名稱
run_pipeline_FILE = run_pipeline_file
run_pipeline_STREAM = run_pipeline_stream
run_pipeline_DIR = run_pipeline_dir

# ───────────────────────── CLI ─────────────────────────
def main():
    sep, spk, asr, use_gpu = init_pipeline_modules()
    parser = argparse.ArgumentParser(description="Speech pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    # file
    p_file = sub.add_parser("file", help="process existing wav file")
    p_file.add_argument("path")
    p_file.add_argument("--workers", type=int, default=4)
    p_file.add_argument("--model", type=str, default=DEFAULT_WHISPER_MODEL,
                        help="Whisper model name (override constants)")
    p_file.add_argument("--beam", type=int, default=DEFAULT_WHISPER_BEAM_SIZE,
                        help="Beam size for Whisper")

    # stream
    p_stream = sub.add_parser("stream", help="live stream from microphone")
    p_stream.add_argument("--chunk", type=float, default=4.0, help="seconds per chunk")
    p_stream.add_argument("--workers", type=int, default=2)
    p_stream.add_argument("--record_secs", type=float, default=18.0,
                          help="total recording time in seconds (None for infinite)")
    p_stream.add_argument("--model", type=str, default=DEFAULT_WHISPER_MODEL)
    p_stream.add_argument("--beam", type=int, default=DEFAULT_WHISPER_BEAM_SIZE)

    # dir
    p_dir = sub.add_parser("dir", help="process all audio files in a directory")
    p_dir.add_argument("path")
    p_dir.add_argument("--workers", type=int, default=3)
    p_dir.add_argument("--truth_map", type=str, default="truth_map.txt")
    p_dir.add_argument("--model", type=str, default=DEFAULT_WHISPER_MODEL)
    p_dir.add_argument("--beam", type=int, default=DEFAULT_WHISPER_BEAM_SIZE)

    args = parser.parse_args()

    # 用 CLI 覆蓋 ASR 設定
    # 如果命令行 override 模型，就重新拿一个新的 asr
    asr = WhisperASR(model_name=args.model, gpu=use_gpu, beam=args.beam)

    if args.mode == "file":
        run_pipeline_file(args.path,
                          args.workers,
                          sep=sep, spk=spk, asr=asr)    
    elif args.mode == "stream":
        run_pipeline_stream(chunk_secs=args.chunk, max_workers=args.workers, sep=sep, spk=spk, asr=asr)
    elif args.mode == "dir":
        run_pipeline_dir(args.path, truth_map_path=args.truth_map, max_workers=args.workers, sep=sep, spk=spk, asr=asr)


if __name__ == "__main__":
    main()