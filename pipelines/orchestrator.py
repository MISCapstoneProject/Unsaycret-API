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
from utils.constants import DEFAULT_WHISPER_MODEL,DEFAULT_WHISPER_BEAM_SIZE
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
        # 使用台北時間戳轉換 (UTC+8)
        taipei_tz = timezone(timedelta(hours=8))
        result["absolute_start_time"] = dt.fromtimestamp(absolute_timestamp, tz=taipei_tz).isoformat()
        result["absolute_end_time"] = dt.fromtimestamp(absolute_timestamp + (t1 - t0), tz=taipei_tz).isoformat()
    
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
    chunk_secs: float = 4.0,
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

    total_start = time.perf_counter()
    out_root = Path("stream_output") / dt.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
    futures: list[Future] = []
    
    # 使用台北時間作為串流開始時間 (UTC+8)
    taipei_tz = timezone(timedelta(hours=8))
    stream_start_time = dt.now(taipei_tz)

    def process_chunk(raw_bytes: bytes, idx: int, src_sr: int, chunk_start_time: dt = None):
        try:
            t0 = idx * chunk_secs
            t1 = t0 + chunk_secs

            BYTES_PER_SAMPLE = 4
            if channels <= 0 or chunk_secs <= 0:
                raise ValueError(f"invalid channels={channels} or chunk_secs={chunk_secs}")

            # B) decode & 解交錯 → [C, T]
            n_samples_total = len(raw_bytes) // (BYTES_PER_SAMPLE * channels)
            if n_samples_total == 0:
                logger.warning(f"[proc] idx={idx} empty chunk: bytes={len(raw_bytes)}")
                return None

            waveform = torch.frombuffer(raw_bytes, dtype=torch.float32)
            waveform = waveform[: n_samples_total * channels].view(channels, n_samples_total)
            if channels > 1:
                waveform = waveform.mean(dim=0, keepdim=True)   # [1, T]
            # channels==1 時已是 [1, T]，不要再 unsqueeze

            # C) 重採樣到 16k 給模型
            MODEL_SR = 16000
            wave_for_model = waveform
            if src_sr != MODEL_SR:
                wave_for_model = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=MODEL_SR)(waveform)

            # D) 存兩份混音：一份 48k/44.1k 給耳朵聽（不會變慢），一份 16k 給模型檢查
            seg_dir = out_root / f"segment_{idx:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            torchaudio.save((seg_dir / "mix_playback.wav").as_posix(),
                            torch.clamp(waveform * 0.98, -1, 1), src_sr)
            torchaudio.save((seg_dir / "mix_16k.wav").as_posix(),
                            torch.clamp(wave_for_model * 0.98, -1, 1), MODEL_SR)
            mix_path = seg_dir / "mix.wav"
            if chunk_start_time is None:
                chunk_start_time = stream_start_time + timedelta(seconds=t0)

            # F) 分離（吃 16k）
            segments = sep.separate_and_save(
                wave_for_model, seg_dir.as_posix(), segment_index=idx, absolute_start_time=chunk_start_time
            )

            speaker_paths = sorted(seg_dir.glob("speaker*.wav"))
            if not speaker_paths:
                logger.warning("segment %d 無 speaker wav", idx)
                return None

            # --- G) 跑 SpkID + ASR
            speaker_results: list[dict] = []
            for sp_idx, wav_path in enumerate(speaker_paths, 1):
                if segments and len(segments) > sp_idx - 1 and len(segments[sp_idx - 1]) == 4:
                    absolute_timestamp = segments[sp_idx - 1][3]
                    res = process_segment(
                        str(wav_path), t0, t1, absolute_timestamp,
                        sep=sep, spk=spk, asr=asr
                    )
                else:
                    res = process_segment(
                        str(wav_path), t0, t1,
                        sep=sep, spk=spk, asr=asr
                    )

                if not res["text"].strip() or res["confidence"] < 0.1:
                    continue
                res["speaker_index"] = sp_idx
                speaker_results.append(res)

            # 去重：同 speaker 留最高信心
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

            with open(seg_dir / "output.json", "w", encoding="utf-8") as f:
                json.dump(seg_dict, f, ensure_ascii=False, indent=2)

            if queue_out is not None:
                queue_out.put(seg_dict)
            return seg_dict

        except Exception as e:
            logger.exception(f"[process_chunk] idx={idx} failed: {e}")
            return None


    # 錄音/接收執行緒
    q: queue.Queue[tuple[bytes, int]] = queue.Queue(maxsize=max_workers * 2)
    
    # 統一使用外部 stop_event，如果沒提供則創建一個
    if stop_event is None:
        stop_event = threading.Event()

    def recorder_from_queue():
        BYTES_PER_SAMPLE = 4  # Float32
        ch = max(1, int(channels))

        buf = bytearray()
        idx = 0
        start_time = time.time()

        # 先用參數當暫值；進線 0.3s 後用「牆鐘時間 + 累積 bytes」校準
        provisional_sr = max(1, int(rate))
        frames_needed = int(provisional_sr * chunk_secs) * BYTES_PER_SAMPLE * ch

        calibrated = False
        calib_t0 = time.time()
        calib_bytes = 0

        while not stop_event.is_set():
            try:
                pkt = in_bytes_queue.get(timeout=0.1)  # 縮短等待時間提升響應性
            except queue.Empty:
                if record_secs is not None and time.time() - start_time >= record_secs:
                    stop_event.set()  # 統一使用 stop_event
                    break
                # 每次 timeout 都檢查停止信號，提升響應性
                if stop_event.is_set():
                    break
                continue

            calib_bytes += len(pkt)    # 用來估計來源 sr
            buf.extend(pkt)

            # ===== 首包校準：用 bytes/sec 估計真實來源 sr，一次到位 =====
            if not calibrated:
                elapsed = max(1e-3, time.time() - calib_t0)
                if elapsed >= 0.30 and calib_bytes >= BYTES_PER_SAMPLE * ch * 4000:  # 至少 ~0.25s@16k 的量
                    bps = calib_bytes / elapsed
                    est_sr = bps / (BYTES_PER_SAMPLE * ch)   # ≈ 真實 sr
                    # 就近對齊到常見取樣率
                    candidates = [48000, 44100, 32000, 24000, 22050, 16000]
                    src_sr = min(candidates, key=lambda s: abs(s - est_sr))
                    frames_needed = int(src_sr * chunk_secs) * BYTES_PER_SAMPLE * ch
                    calibrated = True
                    logger.info(f"[calib] est≈{est_sr:.1f} → use {src_sr}, ch={ch}, frames_needed={frames_needed}")

            # ===== 切塊：校準完成後才切，確保每塊恰好是 chunk_secs 的「真實時間」 =====
            if not calibrated:
                continue  # 等校準完成再切，避免一開始就用錯的 16k 尺寸

            while len(buf) >= frames_needed:
                raw = bytes(buf[:frames_needed])
                del buf[:frames_needed]
                # 把「校準得到的來源 sr」一併傳下去
                q.put((raw, idx, src_sr))
                idx += 1


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
        buf = bytearray()
        idx = 0
        start_time = time.time()
        try:
            while not stop_event.is_set():
                if record_secs is not None and time.time() - start_time >= record_secs:
                    stop_event.set()  # 統一使用 stop_event
                    break
                buf.extend(stream.read(frames_per_buffer, exception_on_overflow=False))
                if len(buf) // 2 >= frames_needed:
                    raw = bytes(buf[: frames_needed * 2])
                    buf = buf[frames_needed * 2 :]
                    q.put((raw, idx))
                    idx += 1
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    rec_thread = threading.Thread(
        target=recorder_from_queue if in_bytes_queue else recorder_from_mic,
        daemon=True,
    )
    rec_thread.start()

    logger.info(
        "🎙 開始錄音/接收 (%s) ...",
        "外部 bytes" if in_bytes_queue else ("Ctrl‑C" if record_secs is None else f"{record_secs}s"),
    )

    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                raw, idx, src_sr = q.get(timeout=0.1)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue
            futures.append(executor.submit(process_chunk, raw, idx, src_sr))
    except KeyboardInterrupt:
        logger.info("🛑 Ctrl‑C 偵測到使用者手動停止")
        stop_event.set()  # 統一使用 stop_event
    finally:
        stop_event.set()  # 確保停止信號被設置
        rec_thread.join(timeout=1)
        executor.shutdown(wait=True)

    # 等待所有 Future 完成，確保不丟失任何處理結果
    bundle = []
    for f in futures:
        try:
            result = f.result()  # 等待 Future 完成
            if result:
                bundle.append(result)
        except Exception as e:
            logger.error(f"❌ Future 處理失敗: {e}")
    
    bundle.sort(key=lambda x: x["start"])

    pretty_bundle: list[dict] = []
    for seg in bundle:
        for sp in seg["speakers"]:
            pretty_bundle.append(make_pretty(sp))

    logger.info(
        "🚩 stream 結束，共 %d 段，耗時 %.3fs → %s",
        len(bundle), time.perf_counter() - total_start, out_root
    )
    return bundle, pretty_bundle


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