import argparse
import json
import os
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# optional mic deps
try:
    import pyaudio  # type: ignore
except ImportError:
    pyaudio = None  # we'll just disable stream mode mic if not available

# heavy deps
try:
    import torch
    import torchaudio
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore
    torchaudio = None  # type: ignore
    F = None  # type: ignore

from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly

# env / local modules
os.environ.setdefault("MKL_DISABLE_FAST_MM", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from utils.logger import get_logger
from utils.constants import DEFAULT_WHISPER_BEAM_SIZE, DEFAULT_WHISPER_MODEL
from utils.env_config import CUDA_DEVICE_INDEX, FORCE_CPU

from modules.separation.separator import (
    AudioSeparator,
    TARGET_RATE,
)
from modules.identification.VID_identify_v5 import SpeakerIdentifier
from modules.asr.whisper_asr import WhisperASR
from pipelines.fast_aggregator import FastAggregator


logger = get_logger(__name__)

# -----------------------------
# Tunables / heuristics
# -----------------------------

# 4 秒窗, 每 1 秒往前推 (和 sample 一致)
DEFAULT_WINDOW_LEN = 4.0
DEFAULT_STRIDE = 1.0

# 語言相關 (句子切分)
CJK_LANGS = {"zh", "ja", "ko"}
SENTENCE_GAP = 0.3  # 如果兩詞之間斷太久，就視為句子邊界

# 匈牙利追蹤
EMBED_WEIGHT = 1.0
TIME_DECAY = 0.05         # 隔越久成本越高，避免把一年前的人誤綁現在
NEW_TRACK_THRESHOLD = 0.85  # 低成本才會沿用舊 track，否則新 track
TRACK_TTL = 12.0            # 超過多久沒出現就回收 track

# RTF 監控 (只是 log 給你看，不做節流)
RTF_EMA_ALPHA = 0.2

# -----------------------------
# Data structures
# -----------------------------


@dataclass
class SentenceFragment:
    """一小段語句 (已經組好，不是逐字)"""
    text: str
    start: float   # 絕對時間 (全局，而不是片段內)
    end: float


@dataclass
class RawSourceResult:
    """
    單一 source (某位可能的講者) 在本 window 的辨識結果
    - embedding: 用來給匈牙利追蹤
    - sentences: 斷句後的句子 (CJK 合併好)
    - asr_segments: [{text, confidence, start, end}] 給 UI / summary
    - id_info: (speaker_id, speaker_name, distance) 來自 SpeakerIdentifier
               不參與 track 決策，只是 metadata
    """
    embedding: np.ndarray
    sentences: List[SentenceFragment]
    asr_segments: List[dict]
    id_info: Optional[Tuple[str, str, float]]


@dataclass
class WindowRawResult:
    """
    單一視窗 4s/1s 的完整成果 (還沒做匈牙利指派 track_id)
    """
    window_index: int
    t_start: float
    t_end: float
    sources: List[RawSourceResult]
    rtf: float
    seg_dir: Path       # segment_0000/ 這個資料夾路徑
    error: Optional[str] = None


@dataclass
class TrackState:
    """
    匈牙利追蹤器裡面維護的 track 狀態
    - embedding: 此 track 的代表向量（單位向量）
    - last_seen: 這個 track 最後一次出現 (秒，window 的 end)
    - ema_alpha: 更新代表向量的平滑係數
    - sticky_hits: 單一來源黏著命中的次數（僅供觀察/調參）
    """
    track_id: int
    embedding: np.ndarray
    last_seen: float
    ema_alpha: float = 0.15
    sticky_hits: int = 0



# -----------------------------
# 小工具 / 輔助函式
# -----------------------------


class SummaryWriter:
    """
    把每個 window 的摘要行寫進 <stem>_summary.jsonl
    """
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = path.open("w", encoding="utf-8")

    def write_line(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    餘弦距離：1 - cos_sim，越小越像
    若輸入非單位向量，仍做一次安全除法。
    """
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na > 0 and nb > 0:
        return 1.0 - (dot / (na * nb))
    # 回退：至少不會炸
    return 1.0 - dot



def _merge_cjk_sentences(
    words: List[dict],
    fallback_text: str,
    window_start: float,
    window_end: float,
) -> List[SentenceFragment]:
    """
    中文/日文/韓文的情況下，句子以停頓(SENTENCE_GAP)或句尾標點切。
    words: whisper 回傳的逐詞 (我們已經把時間全域對齊了)
    """
    if not words:
        clean = fallback_text.strip()
        if not clean:
            return []
        return [SentenceFragment(text=clean, start=window_start, end=window_end)]

    out: List[SentenceFragment] = []
    buf: List[str] = []
    seg_start: Optional[float] = None
    prev_end: Optional[float] = None

    for w in words:
        token = (w.get("word") or w.get("text") or "").strip()
        if not token:
            continue
        start = float(w.get("start") or 0.0)
        end = float(w.get("end") or start)

        # 句開始
        if seg_start is None:
            seg_start = start

        # 停頓太久 → 先收一段
        if prev_end is not None and (start - prev_end > SENTENCE_GAP) and buf:
            out.append(SentenceFragment("".join(buf), seg_start, prev_end))
            buf = []
            seg_start = start

        buf.append(token)
        prev_end = end

        # 碰到句尾標點就收
        if token.endswith(("。", "！", "？", ".", "!", "?")):
            out.append(SentenceFragment("".join(buf), seg_start, end))
            buf = []
            seg_start = None
            prev_end = None

    # 收尾
    if buf and seg_start is not None and prev_end is not None:
        out.append(SentenceFragment("".join(buf), seg_start, prev_end))

    return out


def _merge_default_sentence(
    words: List[dict],
    fallback_text: str,
    window_start: float,
    window_end: float,
) -> List[SentenceFragment]:
    """
    英文等：就把每個 word 用空白接成一句，當成一個 fragment
    """
    if words:
        toks = [
            (w.get("word") or w.get("text") or "").strip()
            for w in words
            if (w.get("word") or w.get("text"))
        ]
        clean = " ".join([t for t in toks if t]).strip()
        if clean:
            s0 = float(words[0].get("start", window_start))
            s1 = float(words[-1].get("end", window_end))
            return [SentenceFragment(text=clean, start=s0, end=s1)]
    clean_fb = fallback_text.strip()
    if not clean_fb:
        return []
    return [SentenceFragment(text=clean_fb, start=window_start, end=window_end)]


def _build_sentences(
    words: List[dict],
    raw_text: str,
    lang: str,
    window_start: float,
    window_end: float,
) -> List[SentenceFragment]:
    """
    幫某個 source 把 Whisper 的逐詞變成句子片段。
    - 中文：用 _merge_cjk_sentences
    - 其他語言：用 _merge_default_sentence
    """
    if lang.lower() in CJK_LANGS:
        return _merge_cjk_sentences(words, raw_text, window_start, window_end)
    return _merge_default_sentence(words, raw_text, window_start, window_end)


def _segments_from_sentences(sentences: List[SentenceFragment]) -> List[List[float]]:
    """
    給 summary.jsonl 用的小區間資訊
    [[start, end], [start, end], ...]
    """
    return [
        [round(seg.start, 3), round(seg.end, 3)]
        for seg in sentences
    ]


def _build_asr_segments(
    words: List[dict],
    fallback_text: str,
    window_start: float,
    window_end: float,
    avg_conf: float,
) -> List[dict]:
    """
    給 output.json 的 asr_segments
    """
    if not words:
        clean = fallback_text.strip()
        if not clean:
            return []
        return [{
            "text": clean,
            "confidence": round(float(avg_conf), 3),
            "start": round(window_start, 3),
            "end": round(window_end, 3),
        }]

    return [{
        "text": fallback_text.strip(),
        "confidence": round(float(avg_conf), 3),
        "start": round(float(words[0].get("start", window_start)), 3),
        "end": round(float(words[-1].get("end", window_end)), 3),
    }]


def _str_to_bool(val: str) -> bool:
    v = str(val).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean from {val!r}")


# -----------------------------
# 匈牙利追蹤器 (track manager)
# -----------------------------


class TrackManager:
    """
    做「連戲」字幕的重點：
    - 不把 SpeakerIdentifier 的結果塞進決策（避免 track 重命名/亂跳）
    - 只用 embedding + 時間距離 做匈牙利匹配
    - 向量以 EMA 平滑，以提升跨窗穩定度
    - 單一來源情境提供「黏著」保險：只有 1 個活躍 track 且 1 個 source 時，直接續用
    """

    def __init__(self) -> None:
        self.tracks: Dict[int, TrackState] = {}
        self.next_id = 0

    def _expire_old_tracks(self, current_t: float) -> None:
        stale = [tid for tid, st in self.tracks.items() if (current_t - st.last_seen) > TRACK_TTL]
        for tid in stale:
            self.tracks.pop(tid, None)

    def assign(
        self,
        window_start: float,
        window_end: float,
        embeddings: List[np.ndarray],
    ) -> Dict[int, int]:
        """
        回傳 {source_index: track_id} 並更新各個 track 的 embedding/last_seen
        """
        self._expire_old_tracks(window_start)
        assignments: Dict[int, int] = {}
        if not embeddings:
            return assignments

        track_ids = list(self.tracks.keys())

        # --- 單一來源黏著保險 ---
        if len(track_ids) == 1 and len(embeddings) == 1:
            tid = track_ids[0]
            assignments[0] = tid
            self._ema_update_track(tid, embeddings[0], window_end, sticky=True)
            return assignments

        if track_ids:
            cost = np.zeros((len(track_ids), len(embeddings)), dtype=np.float32)
            for r, tid in enumerate(track_ids):
                ref = self.tracks[tid]
                gap = max(0.0, window_start - ref.last_seen)
                for c, emb in enumerate(embeddings):
                    dist = _cosine_distance(ref.embedding, emb)
                    cval = EMBED_WEIGHT * dist + TIME_DECAY * gap
                    cost[r, c] = cval

            row_idx, col_idx = linear_sum_assignment(cost)

            used_sources = set()
            for r, c in zip(row_idx, col_idx):
                if r >= len(track_ids) or c >= len(embeddings):
                    continue
                cval = float(cost[r, c])
                if cval > NEW_TRACK_THRESHOLD:
                    # 成本太高 → 視為新來源
                    continue
                tid = track_ids[r]
                assignments[c] = tid
                used_sources.add(c)
                self._ema_update_track(tid, embeddings[c], window_end)

            # 尚未指派的 source → 新 track
            for s_idx in range(len(embeddings)):
                if s_idx in used_sources:
                    continue
                tid_new = self._create_track(embeddings[s_idx], window_end)
                assignments[s_idx] = tid_new
        else:
            # 系統剛啟動，沒有既有 track → 全部創新
            for s_idx, emb in enumerate(embeddings):
                tid_new = self._create_track(emb, window_end)
                assignments[s_idx] = tid_new

        return assignments

    def _create_track(self, emb: np.ndarray, last_seen: float) -> int:
        tid = self.next_id
        self.next_id += 1
        # emb 應已正規化；再保險一次
        emb = emb / max(float(np.linalg.norm(emb)), 1e-12)
        self.tracks[tid] = TrackState(track_id=tid, embedding=emb, last_seen=last_seen)
        return tid

    def _ema_update_track(self, tid: int, emb_new: np.ndarray, last_seen: float, sticky: bool = False) -> None:
        st = self.tracks.get(tid)
        if st is None:
            return
        # 兩邊皆為單位向量，做 EMA 後再正規化
        alpha = st.ema_alpha
        merged = (1.0 - alpha) * st.embedding + alpha * emb_new
        n = float(np.linalg.norm(merged))
        if n > 0.0 and np.isfinite(n):
            merged = merged / n
        st.embedding = merged
        st.last_seen = last_seen
        if sticky:
            st.sticky_hits += 1



# -----------------------------
# 初始化各模組 (分離 / 辨識 / ASR)
# -----------------------------


def init_pipeline_modules(
    load_separator: bool = True,
    load_identifier: bool = True,
    load_asr: bool = True,
    prefer_device: str = "auto",
) -> Tuple[Optional[AudioSeparator], Optional[SpeakerIdentifier], Optional[WhisperASR], bool]:
    """
    啟動 AudioSeparator, SpeakerIdentifier, WhisperASR
    回傳 (sep, identifier, asr, use_gpu)
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch/torchaudio not available.")

    # 決定要不要走 GPU
    current_cuda_device = CUDA_DEVICE_INDEX
    force_cpu = FORCE_CPU or (prefer_device == "cpu")
    if prefer_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; fallback to CPU.")
        force_cpu = True

    if force_cpu:
        use_gpu = False
        logger.info("Running on CPU.")
    else:
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            if current_cuda_device < torch.cuda.device_count():
                torch.cuda.set_device(current_cuda_device)
                logger.info("CUDA device %s (%s)", current_cuda_device, torch.cuda.get_device_name(current_cuda_device))
            else:
                logger.warning("CUDA device index %s invalid, defaulting to 0.", current_cuda_device)
                torch.cuda.set_device(0)
                logger.info("Using CUDA device 0: %s", torch.cuda.get_device_name(0))

    # 分離
    separator = AudioSeparator() if load_separator else None

    # 語者辨識
    identifier = None
    if load_identifier:
        try:
            identifier = SpeakerIdentifier()
            logger.info("SpeakerIdentifier loaded.")
        except Exception as exc:
            logger.warning("SpeakerIdentifier unavailable: %s", exc)
            identifier = None

    # ASR
    asr = None
    if load_asr:
        asr = WhisperASR(
            model_name=DEFAULT_WHISPER_MODEL,
            gpu=use_gpu,
            beam=DEFAULT_WHISPER_BEAM_SIZE,
        )

    return separator, identifier, asr, use_gpu


# -----------------------------
# 音訊切窗
# -----------------------------


def _load_audio_file(path: Path) -> Tuple[torch.Tensor, int]:
    """
    讀整支 wav, downmix 成單聲道, resample 到 TARGET_RATE
    回傳 (waveform[TorchTensor shape=(1,T)], sr)
    """
    waveform, sr = torchaudio.load(str(path))  # [ch, T]
    if waveform.ndim != 2:
        raise ValueError("Unexpected waveform shape from torchaudio.load")
    if waveform.shape[0] > 1:
        # 轉單聲道
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_RATE)
        sr = TARGET_RATE
    # 確保 contiguous + float32
    return waveform.contiguous().to(torch.float32), sr


def _generate_windows(
    waveform: torch.Tensor,
    sr: int,
    win_len: float,
    stride: float,
) -> Iterable[Tuple[int, float, float, torch.Tensor]]:
    """
    sliding window:
    - 每個 window 長 win_len 秒 (預設 4s)
    - 每次平移 stride 秒 (預設 1s)
    - 最後一段如果不夠長，用 zero pad 補齊到 win_len
    """
    assert waveform.ndim == 2 and waveform.shape[0] == 1
    total_samples = waveform.shape[1]

    win_samples = max(1, int(round(win_len * sr)))
    hop_samples = max(1, int(round(stride * sr)))

    start_sample = 0
    window_idx = 0

    while start_sample < total_samples:
        end_sample = start_sample + win_samples
        chunk = waveform[:, start_sample:end_sample]  # shape [1, T?<=win_samples]
        if chunk.shape[1] < win_samples:
            pad = win_samples - chunk.shape[1]
            if F is None:
                raise RuntimeError("torch.nn.functional (F.pad) missing.")
            chunk = F.pad(chunk, (0, pad))  # right pad zeros

        t0 = start_sample / sr
        t1 = t0 + win_len
        yield window_idx, t0, t1, chunk.contiguous()
        window_idx += 1
        start_sample += hop_samples


# -----------------------------
# 一個 window 的 heavy 工作
# -----------------------------


def _offset_words_global(words: List[dict], offset: float) -> List[dict]:
    """
    Whisper 回傳的 word 時間通常是相對於該音檔開頭(0s)。
    我們把它加上 window 的起點 t_start，變成全域時間軸。
    """
    adjusted = []
    for w in words or []:
        local_start = float(w.get("start") or 0.0)
        local_end = float(w.get("end") or w.get("start") or 0.0)
        adjusted.append(
            {
                "word": str(w.get("word") or w.get("text") or "").strip(),
                "text": str(w.get("text") or w.get("word") or "").strip(),
                "start": local_start + offset,
                "end": local_end + offset,
                "probability": float(w.get("probability") or w.get("prob") or 0.0),
            }
        )
    return adjusted


def _safe_extract_embedding(
    identifier: Optional[SpeakerIdentifier],
    wav_path: str,
    window_idx: int,
    local_idx: int,
) -> np.ndarray:
    """
    匈牙利需要 embedding 來比較 source 之間的相似度。
    正常路線：identifier.extract_embedding(wav_path) 或 identifier.audio_processor.extract_embedding(...)
    備援：deterministic random（也會 L2 正規化）
    """
    # 1) 嘗試從 v5 取真正的向量
    if identifier is not None:
        try:
            if hasattr(identifier, "extract_embedding"):
                emb = identifier.extract_embedding(wav_path)
            elif hasattr(identifier, "audio_processor") and hasattr(identifier.audio_processor, "extract_embedding"):
                emb = identifier.audio_processor.extract_embedding(wav_path)
            else:
                emb = None

            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                n = float(np.linalg.norm(emb))
                if np.isfinite(n) and n > 0.0:
                    emb = emb / n  # L2 normalize
                    return emb
        except Exception as exc:
            logger.debug("extract_embedding failed for %s: %s", wav_path, exc)

    # 2) 備援：可重現亂數 + 正規化
    rng = np.random.RandomState(window_idx * 7919 + local_idx * 104729)
    emb = rng.randn(192).astype(np.float32)
    emb /= max(float(np.linalg.norm(emb)), 1e-12)
    return emb



def _safe_speaker_meta(identifier: Optional[SpeakerIdentifier], wav_path: str) -> Optional[Tuple[str, str, float]]:
    """
    給 summary 用的額外資訊 (speaker_id, name, distance)
    不干擾 track 決策
    """
    if identifier is None:
        return None
    try:
        return identifier.process_audio_file(wav_path)
    except Exception as exc:
        logger.debug("SpeakerIdentifier.process_audio_file error: %s", exc)
        return None


def process_window(
    window_idx: int,
    chunk: torch.Tensor,          # shape [1, T] @ TARGET_RATE
    t_start: float,
    t_end: float,
    base_ts: datetime,
    session_dir: Path,
    separator: AudioSeparator,
    identifier: Optional[SpeakerIdentifier],
    asr: WhisperASR,
    lang: str,
) -> WindowRawResult:
    """
    真正重的 pipeline：
    1. 存 mix.wav
    2. 呼叫 separator.separate_and_save() → 輸出 speaker1.wav, speaker2.wav...
    3. 對每個 speakerX.wav:
        - speakerID (metadata)
        - embedding (for Hungarian)
        - Whisper ASR
        - 句子切分
    4. 統一打包回傳
    """

    seg_dir = session_dir / f"segment_{window_idx:04d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    mix_path = seg_dir / "mix.wav"
    # chunk shape [1, T], torchaudio.save expects shape [ch, T]
    torchaudio.save(str(mix_path), chunk.cpu(), TARGET_RATE)

    # base_ts 是整段開始錄製/處理時的 "真實世界時間"
    # 我們給 separator 用，因為它會存絕對 timestamp
    abs_ts = base_ts + timedelta(seconds=t_start)

    start_t = time.perf_counter()
    separated_info = separator.separate_and_save(
        audio_tensor=chunk,                 # torch.Tensor [1,T]
        output_dir=seg_dir.as_posix(),      # 存哪裡
        segment_index=window_idx,           # 第幾個窗
        absolute_start_time=abs_ts,         # 用來記錄時間
    )
    # separated_info 是 list[(wav_path, rel_start, rel_end, absolute_timestamp)]
    # (依你們 separator 的實作)

    window_sources: List[RawSourceResult] = []

    for local_idx, entry in enumerate(separated_info):
        wav_path, rel_t0, rel_t1, _abs_ts = entry  # 我們其實用不到 rel_t0/rel_t1/abs_ts 這邊

        # 1) speaker metadata / embedding
        spk_meta = _safe_speaker_meta(identifier, wav_path)
        emb_vec = _safe_extract_embedding(identifier, wav_path, window_idx, local_idx)

        # 2) Whisper ASR
        #    我們盡量不動 ASR 前處理 (不做奇怪正規化/Mask)
        text, avg_conf, words = asr.transcribe(
            wav_path,
            language=lang,
        )

        # 調整 words 時間軸成全域絕對時間 (不是4秒local)
        global_words = _offset_words_global(words, t_start)

        # 3) 句子切分/合併
        sentences = _build_sentences(
            global_words,
            text,
            lang,
            window_start=t_start,
            window_end=t_end,
        )

        # 4) asr_segments 格式化 (給 output.json / debug UI)
        asr_segments = _build_asr_segments(
            global_words,
            text,
            t_start,
            t_end,
            avg_conf,
        )

        window_sources.append(
            RawSourceResult(
                embedding=emb_vec,
                sentences=sentences,
                asr_segments=asr_segments,
                id_info=spk_meta,
            )
        )

    elapsed = time.perf_counter() - start_t
    win_dur = max(t_end - t_start, 1e-6)
    rtf = elapsed / win_dur  # 如果 >1 代表比即時還慢

    logger.info(
        "[RTF=%.2f] win=%d t0=%.2f srcs=%d text_len=%d",
        rtf,
        window_idx,
        t_start,
        len(window_sources),
        sum(len(seg.text) for src in window_sources for seg in src.sentences),
    )

    return WindowRawResult(
        window_index=window_idx,
        t_start=t_start,
        t_end=t_end,
        sources=window_sources,
        rtf=rtf,
        seg_dir=seg_dir,
        error=None,
    )


# -----------------------------
# 把 future 結果寫到檔案 & summary
# 這一步是「序列化 + 匈牙利追蹤 + FAST聚合」
# -----------------------------


def handle_window_result(
    result: WindowRawResult,
    track_manager: TrackManager,
    summary_writer: SummaryWriter,
    aggregator: Optional[FastAggregator],
    rtf_state: Dict[str, float],
) -> None:
    """
    必須照 window_index 的順序執行：
      1) 匈牙利分配 track_id
      2) 建立 output.json
      3) 追加一行到 summary.jsonl
      4) 把句子丟進 FAST 聚合器（跨窗合併）
    """
    # 1) 匈牙利：只用 embedding + 時間做追蹤，不混 SpeakerID
    embeddings = [src.embedding for src in result.sources]
    assignments = track_manager.assign(
        window_start=result.t_start,
        window_end=result.t_end,
        embeddings=embeddings,
    )

    tracks_payload_for_segment: List[dict] = []
    summary_tracks: List[dict] = []

    # 2) 組 per-window 的輸出內容
    for s_idx, src in enumerate(result.sources):
        tid = assignments.get(s_idx)
        if tid is None:
            # 理論上 assign() 會涵蓋所有 source；保險起見：
            tid = track_manager._create_track(src.embedding, result.t_end)

        speaker_label = f"S{tid}"

        # SpeakerID 只作為輸出 metadata（不參與追蹤決策）
        spk_id = None
        spk_name = None
        spk_dist = None
        if src.id_info:
            try:
                spk_id = src.id_info[0]
                spk_name = src.id_info[1]
                spk_dist = float(src.id_info[2]) if src.id_info[2] is not None else None
            except Exception:
                pass

        # CJK 已合併的句子
        joined_text = " ".join(seg.text for seg in src.sentences).strip()
        seg_spans = _segments_from_sentences(src.sentences)

        # 2-1) segment_XXXX/output.json 內的 tracks 欄位
        tracks_payload_for_segment.append({
            "track_id": tid,
            "speaker": speaker_label,
            "asr_segments": src.asr_segments,  # 這裡保留 ASR 的句段（供下游需要）
            "speaker_id": spk_id,
            "speaker_name": spk_name,
            "speaker_dist": round(spk_dist, 3) if isinstance(spk_dist, float) else None,
        })

        # 2-2) summary.jsonl 這一行要寫的 tracks 陣列
        summary_tracks.append({
            "track_id": tid,
            "speaker": speaker_label,
            "text": joined_text,
            "segments": seg_spans,
            "speaker_id": spk_id,
            "speaker_name": spk_name,
            "speaker_dist": round(spk_dist, 3) if isinstance(spk_dist, float) else None,
        })

        # 4) 丟給 FAST（跨窗合併）：以「句子片段」為 token
        if aggregator and src.sentences:
            tokens = [
                {
                    "text": seg_frag.text,
                    "start": seg_frag.start,
                    "end": seg_frag.end,
                }
                for seg_frag in src.sentences
            ]
            aggregator.append_window_tokens(
                track_id=tid,
                tokens=tokens,
                t_start=result.t_start,
                t_end=result.t_end,
            )

    # 3) 落地 output.json（本窗）
    segment_payload = {
        "window_index": result.window_index,
        "t_start": round(result.t_start, 3),
        "t_end": round(result.t_end, 3),
        "num_sources": len(result.sources),
        "tracks": tracks_payload_for_segment,
    }
    out_json_path = result.seg_dir / "output.json"
    out_json_path.write_text(
        json.dumps(segment_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 3) 追加一行到 summary.jsonl
    summary_writer.write_line({
    "window_index": result.window_index,
    "t_start": round(result.t_start, 3),
    "t_end": round(result.t_end, 3),
    "num_sources": len(result.sources),
    "tracks": summary_tracks,
    })


    # RTF EMA：監控延遲風險
    prev_ema = rtf_state.get("ema", 0.0)
    rtf_now = float(result.rtf)
    new_ema = (RTF_EMA_ALPHA * rtf_now) + (1.0 - RTF_EMA_ALPHA) * prev_ema
    rtf_state["ema"] = new_ema
    logger.debug(
        "RTF current=%.2f, EMA=%.2f, window=%d",
        rtf_now,
        new_ema,
        result.window_index,
    )

    # ⬇⬇⬇ 這是新增的部分 ⬇⬇⬇
    # 我們希望在每個 window flush 後，馬上把目前聚合好的 transcript 輸出到同一個 txt
    # 也可以同時把 snapshot 用 SSE broadcast 出去（如果你啟動了 SSE）。
    if aggregator is not None:
        try:
            # 1) 先 finalize() 目前累積的片段，讓最後一句在檔案裡是最新版本
            aggregator.finalize()
        except Exception:
            pass

        # 2) 這邊需要 transcript_path，file mode 跟 stream mode 都有，
        #    但 handle_window_result() 現在拿不到那個路徑本身。
        #    我們做一個簡單折衷：Aggregator 自己維護最後一次 export 的路徑。
        #    這表示我們要在 FastAggregator 裡加一個 set_export_path()。
        try:
            if hasattr(aggregator, "_export_path") and aggregator._export_path:
                aggregator.export_txt(aggregator._export_path)
        except Exception:
            pass

        # 3) SSE: 如果啟動了 SSE (aggregator.enable_sse_server 叫過)
        #    每次 window 更新後就 broadcast 最新 snapshot，
        #    這樣你的前端就可以即時看到修好的字幕
        try:
            if hasattr(aggregator, "broadcast_snapshot"):
                # 用 stem 當 session id，純顯示用
                aggregator.broadcast_snapshot(getattr(aggregator, "_session_stem", "session"))
        except Exception:
            pass



# -----------------------------
# 檔案模式：整支 wav -> sliding windows -> executor -> flush
# -----------------------------


def run_file_mode(
    wav_path: Path,
    outdir: Path,
    win_len: float,
    stride: float,
    workers: int,
    lang: str,
    enable_fast: bool,
    separator: AudioSeparator,
    identifier: Optional[SpeakerIdentifier],
    asr: WhisperASR,
) -> None:

    # 輸出檔名基底
    stem = wav_path.stem
    session_dir = outdir / stem
    summary_path = outdir / f"{stem}_summary.jsonl"
    transcript_path = outdir / f"{stem}.txt"

    session_dir.mkdir(parents=True, exist_ok=True)

    # 共用物件 (這些東西要跨 window 維持狀態)
    summary_writer = SummaryWriter(summary_path)
    track_manager = TrackManager()
    aggregator = FastAggregator() if enable_fast else None
    if aggregator:
        aggregator.set_export_path(str(transcript_path))
        aggregator.set_session_stem(stem)
        # 如果你想啟動 SSE server 讓你邊看邊長字幕，打開這行：
        aggregator.enable_sse_server(host="127.0.0.1", port=9877)
    rtf_state = {"ema": 0.0}

    # executor for heavy work
    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    pending: Dict[int, Future] = {}
    next_to_flush = 0

    # 時區時間 (做絕對 timestamp 用)
    tz = timezone(timedelta(hours=8))
    base_ts = datetime.now(tz)

    # 讀整隻音檔 & 切窗
    waveform, sr = _load_audio_file(wav_path)

    # 主要 loop：每個 window 丟進 executor
    try:
        for w_idx, t0, t1, chunk_tensor in _generate_windows(
            waveform, sr, win_len, stride
        ):
            # submit heavy job
            fut = executor.submit(
                process_window,
                w_idx,
                chunk_tensor,
                t0,
                t1,
                base_ts,
                session_dir,
                separator,
                identifier,
                asr,
                lang,
            )
            pending[w_idx] = fut

            # flush anything ready in order
            _flush_ready_results_in_order(
                pending, track_manager, summary_writer, aggregator, rtf_state, next_to_flush
            )
            # update pointer if we flushed
            while next_to_flush in pending and pending[next_to_flush].done():
                # after flush_ready_results_in_order we already handled them,
                # so we can pop and advance
                pending.pop(next_to_flush, None)
                next_to_flush += 1

    except KeyboardInterrupt:
        logger.info("File mode interrupted by user.")
    finally:
        # 完整收尾：等全部 future 跑完，強制 flush
        executor.shutdown(wait=True)

        # flush all remaining (force=True)
        _flush_all_remaining(
            pending, track_manager, summary_writer, aggregator, rtf_state
        )

    if aggregator:
        aggregator.finalize()
        aggregator.export_txt(str(transcript_path))
        summary_writer.close()

    logger.info("File processing complete. Summary -> %s", summary_path)


def _flush_ready_results_in_order(
    pending: Dict[int, Future],
    track_manager: TrackManager,
    summary_writer: SummaryWriter,
    aggregator: Optional[FastAggregator],
    rtf_state: Dict[str, float],
    next_to_flush: int,
) -> None:
    """
    嘗試從 next_to_flush 開始，依序處理已經完成的 future。
    注意：我們不會 block 等它完成；只處理「已經 done()」的。
    這等同於 sample 裡「邊跑邊寫」的感覺，避免整批卡在記憶體。
    """
    while next_to_flush in pending:
        fut = pending[next_to_flush]
        if not fut.done():
            break
        try:
            result = fut.result()
        except Exception as exc:
            logger.exception("Window %d failed: %s", next_to_flush, exc)
            # 如果一個 window 壞掉，我們還是寫一個空的 result
            fake_dir = Path()
            result = WindowRawResult(
                window_index=next_to_flush,
                t_start=float(next_to_flush),
                t_end=float(next_to_flush) + DEFAULT_WINDOW_LEN,
                sources=[],
                rtf=0.0,
                seg_dir=fake_dir,
                error=str(exc),
            )
        handle_window_result(
            result,
            track_manager,
            summary_writer,
            aggregator,
            rtf_state,
        )
        # 不能在這裡 pop，因為外面也會 pop & 進位 (避免重入衝突)
        next_to_flush += 1


def _flush_all_remaining(
    pending: Dict[int, Future],
    track_manager: TrackManager,
    summary_writer: SummaryWriter,
    aggregator: Optional[FastAggregator],
    rtf_state: Dict[str, float],
) -> None:
    """
    結束前保底：把所有 future.block 等到結束，然後依序寫出。
    """
    # 先拿所有 keys 排序
    keys_sorted = sorted(pending.keys())
    for idx in keys_sorted:
        fut = pending[idx]
        try:
            result = fut.result()
        except Exception as exc:
            logger.exception("Window %d failed (final flush): %s", idx, exc)
            fake_dir = Path()
            result = WindowRawResult(
                window_index=idx,
                t_start=float(idx),
                t_end=float(idx) + DEFAULT_WINDOW_LEN,
                sources=[],
                rtf=0.0,
                seg_dir=fake_dir,
                error=str(exc),
            )
        handle_window_result(
            result,
            track_manager,
            summary_writer,
            aggregator,
            rtf_state,
        )


# -----------------------------
# 直播模式(麥克風): 即時切窗 + executor + flush
# -----------------------------


def run_stream_mode(
    outdir: Path,
    win_len: float,
    stride: float,
    workers: int,
    lang: str,
    enable_fast: bool,
    separator: AudioSeparator,
    identifier: Optional[SpeakerIdentifier],
    asr: WhisperASR,
    rate: int,
    channels: int,
    frames_per_buffer: int,
) -> None:
    if pyaudio is None:
        raise RuntimeError("pyaudio is not available; can't run in stream mode.")

    # 以當下時間做 session 名稱
    tz = timezone(timedelta(hours=8))
    stem = datetime.now(tz).strftime("stream_%Y%m%d_%H%M%S")
    session_dir = outdir / stem
    summary_path = outdir / f"{stem}_summary.jsonl"
    transcript_path = outdir / f"{stem}.txt"
    session_dir.mkdir(parents=True, exist_ok=True)

    summary_writer = SummaryWriter(summary_path)
    track_manager = TrackManager()
    aggregator = FastAggregator() if enable_fast else None
    if aggregator:
        aggregator.set_export_path(str(transcript_path))
        aggregator.set_session_stem(stem)
        # 如果你想啟動 SSE server 讓你邊看邊長字幕，打開這行：
        aggregator.enable_sse_server(host="127.0.0.1", port=9877)
    rtf_state = {"ema": 0.0}

    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    pending: Dict[int, Future] = {}
    next_to_flush = 0

    base_ts = datetime.now(tz)

    # mic stream
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )

    # 我們會持續把麥克風資料 append 到 buffer
    # 然後用 ring buffer 方式每 stride 秒切一次 win_len 長度的窗
    buffer = np.zeros(0, dtype=np.float32)
    buffer_start_sample = 0  # 全域 buffer 的起點 index
    next_window_start = 0    # 下一個要切的窗的起點 (以 TARGET_RATE 為單位)
    win_samples = int(round(win_len * TARGET_RATE))
    hop_samples = int(round(stride * TARGET_RATE))

    win_idx = 0

    try:
        while True:
            data = stream.read(frames_per_buffer, exception_on_overflow=False)
            chunk = np.frombuffer(data, dtype=np.float32)

            # downmix 如果多聲道
            if channels > 1:
                chunk = chunk.reshape(-1, channels).mean(axis=1)

            # 重採樣到 TARGET_RATE
            if rate != TARGET_RATE:
                chunk = resample_poly(chunk, TARGET_RATE, rate).astype(np.float32)

            # append 進 buffer
            buffer = np.concatenate([buffer, chunk])

            # 只要 buffer 足夠，就切一個新 window
            while (next_window_start + win_samples) <= (buffer_start_sample + len(buffer)):
                # 把這一段 [next_window_start, next_window_start + win_samples) 抽出
                offset0 = next_window_start - buffer_start_sample
                seg = buffer[offset0: offset0 + win_samples]  # shape [samples]

                # 變成 torch.Tensor [1, T]
                seg_tensor = torch.from_numpy(seg).unsqueeze(0).to(torch.float32)

                t0 = next_window_start / TARGET_RATE
                t1 = t0 + win_len

                fut = executor.submit(
                    process_window,
                    win_idx,
                    seg_tensor,
                    t0,
                    t1,
                    base_ts,
                    session_dir,
                    separator,
                    identifier,
                    asr,
                    lang,
                )
                pending[win_idx] = fut

                # 推進到下一個窗
                win_idx += 1
                next_window_start += hop_samples

                # 丟一個 flush 檢查
                _flush_ready_results_in_order(
                    pending, track_manager, summary_writer, aggregator, rtf_state, next_to_flush
                )
                while next_to_flush in pending and pending[next_to_flush].done():
                    pending.pop(next_to_flush, None)
                    next_to_flush += 1

            # 丟掉太舊的 buffer (防止越囤越大)
            drop_until = next_window_start - win_samples
            if drop_until > buffer_start_sample:
                drop_n = drop_until - buffer_start_sample
                if drop_n > 0:
                    buffer = buffer[drop_n:]
                    buffer_start_sample += drop_n

    except KeyboardInterrupt:
        logger.info("Stream mode interrupted by user.")
    finally:
        # 停麥克風
        stream.stop_stream()
        stream.close()
        pa.terminate()

        executor.shutdown(wait=True)
        _flush_all_remaining(
            pending, track_manager, summary_writer, aggregator, rtf_state
        )

        if aggregator:
            aggregator.finalize()
            aggregator.export_txt(str(transcript_path))

        summary_writer.close()

    logger.info("Stream finished. Summary -> %s", summary_path)


# -----------------------------
# CLI / main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unsaycret orchestrator_v2 (clean, sample-style)."
    )
    p.add_argument("--mode", choices=["file", "stream"], default="file")
    p.add_argument("--wav", type=str, help="Input WAV path (file mode).")
    p.add_argument("--outdir", type=str, default="outputs")

    p.add_argument("--chunk", type=float, default=DEFAULT_WINDOW_LEN, help="window length seconds")
    p.add_argument("--stride", type=float, default=DEFAULT_STRIDE, help="window hop seconds")

    p.add_argument("--workers", type=int, default=2, help="thread pool size")
    p.add_argument("--lang", type=str, default="zh")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument("--enable-fast", type=str, default="true")

    # stream mode audio params
    p.add_argument("--rate", type=int, default=16000)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--frames-per-buffer", type=int, default=1024)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    enable_fast = _str_to_bool(args.enable_fast)

    # 啟動所有必要模組
    separator, identifier, asr, _use_gpu = init_pipeline_modules(
        load_separator=True,
        load_identifier=True,
        load_asr=True,
        prefer_device=args.device,
    )
    if separator is None or asr is None:
        raise RuntimeError("Failed to initialize core modules (separator/asr).")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "file":
        if not args.wav:
            raise ValueError("--wav is required in file mode.")
        wav_path = Path(args.wav)
        if not wav_path.exists():
            raise FileNotFoundError(f"Input file not found: {wav_path}")

        run_file_mode(
            wav_path=wav_path,
            outdir=outdir,
            win_len=args.chunk,
            stride=args.stride,
            workers=args.workers,
            lang=args.lang,
            enable_fast=enable_fast,
            separator=separator,
            identifier=identifier,
            asr=asr,
        )
    else:
        run_stream_mode(
            outdir=outdir,
            win_len=args.chunk,
            stride=args.stride,
            workers=args.workers,
            lang=args.lang,
            enable_fast=enable_fast,
            separator=separator,
            identifier=identifier,
            asr=asr,
            rate=args.rate,
            channels=args.channels,
            frames_per_buffer=args.frames_per_buffer,
        )


if __name__ == "__main__":
    main()
