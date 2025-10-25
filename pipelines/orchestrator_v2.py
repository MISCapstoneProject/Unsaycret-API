import argparse
import http.server
import importlib
import json
import logging
import math
import os
import queue
import socketserver
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
try:
    import pyaudio  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyaudio = None

# Lazy imports for optional/heavy dependencies
try:
    import torch
    import torchaudio
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore
    torchaudio = None  # type: ignore

try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    resample_poly = None  # type: ignore

os.environ.setdefault("MKL_DISABLE_FAST_MM", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from utils.logger import get_logger
from utils.constants import DEFAULT_WHISPER_BEAM_SIZE, DEFAULT_WHISPER_MODEL
from utils.env_config import CUDA_DEVICE_INDEX, FORCE_CPU

from modules.separation.separator import (
    AudioSeparator,
    MIN_ENERGY_THRESHOLD,
    TARGET_RATE,
)
from modules.identification.VID_identify_v5 import SpeakerIdentifier
from modules.identification.matching_hungarian import HungarianAB
from modules.asr.whisper_asr import WhisperASR


logger = get_logger(__name__)


@dataclass
class SeparationThresholds:
    min_voiced: float
    min_rms_db: float
    min_duration: float


@dataclass
class SeparationDSPConfig:
    pad_left: float
    pad_right: float
    mask_floor: float
    mixture_consistency: bool


@dataclass
class AsrGateConfig:
    min_voiced: float
    min_rms_db: float


@dataclass
class AsrRuntimeConfig:
    language: str = "zh"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    task: str = "transcribe"
    condition_on_previous_text: bool = False
    vad_filter: bool = True
    vad_parameters: dict = field(default_factory=lambda: {"min_silence_duration_ms": 200})
    no_speech_threshold: float = 0.5
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    suppress_tokens: str = "-1"


@dataclass
class AsrNormalizationConfig:
    target_rms_db: float
    peak_limit_db: float


# The rest of orchestrator implementation is injected below.


@dataclass
class FastComponents:
    Aggregator: Any
    Tok: Any
    tokens_to_text: Any
    write_srt: Any
    write_txt: Any
    write_srt_with_repair: Optional[Any]


def _load_fast_components() -> Optional[FastComponents]:
    module_candidates = ["modules.asr.Fast_Test", "Fast_Test"]
    for name in module_candidates:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        required = ("Aggregator", "Tok", "tokens_to_text", "write_srt")
        if not all(hasattr(mod, attr) for attr in required):
            continue
        write_txt_func = getattr(mod, "write_txt", None)
        if write_txt_func is None:
            def _fallback_write_txt(path: Path, toks: List[Any]) -> None:
                text = getattr(mod, "tokens_to_text")(toks)
                Path(path).write_text(text + "\n", encoding="utf-8")
            write_txt_func = _fallback_write_txt
        write_srt_with_repair = None
        for repair_mod in ("modules.asr.llm_repair", "llm_repair"):
            try:
                rep = importlib.import_module(repair_mod)
                if hasattr(rep, "write_srt_with_repair"):
                    write_srt_with_repair = getattr(rep, "write_srt_with_repair")
                    break
            except Exception:
                continue
        return FastComponents(
            Aggregator=getattr(mod, "Aggregator"),
            Tok=getattr(mod, "Tok"),
            tokens_to_text=getattr(mod, "tokens_to_text"),
            write_srt=getattr(mod, "write_srt"),
            write_txt=write_txt_func,
            write_srt_with_repair=write_srt_with_repair,
        )
    logger.warning("FAST aggregator modules not found; aggregation disabled.")
    return None


FAST_COMPONENTS = _load_fast_components()

def init_pipeline_modules(
    load_separator: bool = True,
    load_identifier: bool = True,
    load_asr: bool = True,
    id_backend: str = "auto",
) -> Tuple[Optional[AudioSeparator], Optional[SpeakerIdentifier], Optional[WhisperASR], bool]:
    """
    Initialize pipeline modules with graceful fallback for missing dependencies.
    
    Args:
        load_separator: Whether to load audio separation module
        load_identifier: Whether to load speaker identification module
        load_asr: Whether to load ASR module
        id_backend: Speaker ID backend ('auto', 'wespeaker', 'pyannote', 'speechbrain', 'none')
        
    Returns:
        Tuple of (separator, identifier, asr, use_gpu)
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required but not installed. Please install: pip install torch torchaudio")
    
    current_cuda_device = CUDA_DEVICE_INDEX

    if FORCE_CPU:
        use_gpu = False
        logger.info("FORCE_CPU=true; forcing CPU execution.")
    else:
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            if current_cuda_device < torch.cuda.device_count():
                torch.cuda.set_device(current_cuda_device)
                logger.info(
                    "CUDA device %s selected (%s)",
                    current_cuda_device,
                    torch.cuda.get_device_name(current_cuda_device),
                )
            else:
                logger.warning(
                    "CUDA index %s unavailable; defaulting to device 0.",
                    current_cuda_device,
                )
                current_cuda_device = 0
                torch.cuda.set_device(current_cuda_device)
                logger.info("Using CUDA device 0: %s", torch.cuda.get_device_name(0))

    logger.info("Pipeline device: %s", f"cuda:{current_cuda_device}" if use_gpu else "cpu")

    # Load separator with error handling
    separator = None
    if load_separator:
        try:
            separator = AudioSeparator()
            logger.info("[Separation] AudioSeparator loaded successfully")
        except Exception as exc:
            logger.warning("[Separation] Failed to load AudioSeparator: %s", exc)
            separator = None
    
    # Load speaker identifier with graceful fallback
    identifier = None
    if load_identifier and id_backend != "none":
        try:
            identifier = SpeakerIdentifier()
            logger.info("[Identification] SpeakerIdentifier loaded successfully")
        except ImportError as exc:
            missing_module = exc.name if hasattr(exc, 'name') else 'dependencies'
            logger.warning("[Identification] SpeakerIdentifier unavailable (missing %s) → fallback to none", missing_module)
            identifier = None
        except Exception as exc:
            logger.warning("[Identification] Failed to load SpeakerIdentifier: %s → fallback to none", exc)
            identifier = None
    elif id_backend == "none":
        logger.info("[Identification] Speaker identification explicitly disabled (--id-backend none)")
    
    # Load ASR with error handling
    asr = None
    if load_asr:
        try:
            asr = WhisperASR(
                model_name=DEFAULT_WHISPER_MODEL,
                gpu=use_gpu,
                beam=DEFAULT_WHISPER_BEAM_SIZE,
            )
            logger.info("[ASR] WhisperASR loaded successfully")
        except Exception as exc:
            logger.warning("[ASR] Failed to load WhisperASR: %s", exc)
            asr = None
            
    return separator, identifier, asr, use_gpu

def generate_sliding_windows(
    wav: np.ndarray,
    sr: int,
    win_len: float = 4.0,
    stride: float = 1.0,
) -> Iterable[Tuple[int, float, float, np.ndarray, int, int]]:
    if wav.ndim != 1:
        raise ValueError("Sliding window expects a mono waveform (1-D).")

    win_size = int(round(win_len * sr))
    hop = int(round(stride * sr))
    if win_size <= 0 or hop <= 0:
        raise ValueError("win_len and stride must be positive.")

    total = wav.shape[0]
    if total < win_size:
        return

    idx = 0
    start = 0
    while start + win_size <= total:
        end = start + win_size
        chunk = wav[start:end].astype(np.float32, copy=False)
        yield idx, start / sr, end / sr, chunk, start, end
        idx += 1
        start += hop


def compute_audio_metrics(audio: np.ndarray, sr: int) -> Dict[str, float]:
    if audio.size == 0 or sr <= 0:
        return {
            "rms_db": -120.0,
            "voiced_ratio": 0.0,
            "duration_s": 0.0,
            "peak_db": -120.0,
        }

    duration = float(audio.size) / float(sr)
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-9))
    peak_db = 20.0 * math.log10(max(peak, 1e-9))

    frame_len = max(1, int(round(0.02 * sr)))
    if audio.size < frame_len:
        voiced_ratio = 1.0 if peak > 1e-3 else 0.0
    else:
        trimmed = (audio.size // frame_len) * frame_len
        frames = audio[:trimmed].reshape(-1, frame_len)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
        threshold = max(1e-4, 0.25 * rms)
        voiced_ratio = float(np.mean(frame_rms > threshold))

    return {
        "rms_db": rms_db,
        "voiced_ratio": voiced_ratio,
        "duration_s": duration,
        "peak_db": peak_db,
    }


def should_drop_source(metrics: Dict[str, float], thresholds: SeparationThresholds) -> bool:
    return (
        metrics["voiced_ratio"] < thresholds.min_voiced
        or metrics["rms_db"] < thresholds.min_rms_db
        or metrics["duration_s"] < thresholds.min_duration
    )


def passes_asr_gate(metrics: Dict[str, float], gate: AsrGateConfig) -> bool:
    return metrics["voiced_ratio"] >= gate.min_voiced and metrics["rms_db"] >= gate.min_rms_db


def average_confidence(segments: List[Dict[str, Any]]) -> Optional[float]:
    if not segments:
        return None
    vals = [seg.get("confidence") for seg in segments if seg.get("confidence") is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _blend_words_to_segment(
    words: List[Dict[str, Any]],
    window_start: float,
    window_end: float,
    text: str,
    avg_conf: float,
) -> List[Dict[str, Any]]:
    if not words:
        clean_text = text.strip()
        if not clean_text:
            return []
        return [
            {
                "start": round(window_start, 3),
                "end": round(window_end, 3),
                "text": clean_text,
                "confidence": round(float(avg_conf), 3),
            }
        ]

    seg_start = window_start + float(words[0]["start"])
    seg_end = window_start + float(words[-1]["end"])
    return [
        {
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": text.strip(),
            "confidence": round(float(avg_conf), 3),
        }
    ]


def _decode_audio_bytes(raw_bytes: bytes, channels: int, bytes_per_sample: int) -> np.ndarray:
    if bytes_per_sample not in (2, 4):
        raise ValueError(f"Unsupported bytes_per_sample={bytes_per_sample}")

    dtype = np.float32 if bytes_per_sample == 4 else np.int16
    waveform = np.frombuffer(raw_bytes, dtype=dtype)
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(axis=1)
    if bytes_per_sample == 2:
        waveform = waveform.astype(np.float32) / 32768.0
    else:
        waveform = waveform.astype(np.float32, copy=False)
    return waveform


def _extract_padded_segment(
    waveform: np.ndarray,
    start: int,
    end: int,
    pad_left: int,
    pad_right: int,
) -> np.ndarray:
    total_len = pad_left + (end - start) + pad_right
    segment = np.zeros(total_len, dtype=np.float32)

    left = max(0, start - pad_left)
    right = min(waveform.shape[0], end + pad_right)
    dst_start = pad_left - max(0, start - left)
    segment[dst_start : dst_start + (right - left)] = waveform[left:right]
    return segment


def _segment_from_buffer(
    buffer: np.ndarray,
    buffer_start: int,
    start: int,
    end: int,
) -> np.ndarray:
    length = max(0, end - start)
    segment = np.zeros(length, dtype=np.float32)
    if length == 0:
        return segment

    available_start = max(start, buffer_start)
    available_end = min(end, buffer_start + buffer.shape[0])
    if available_end <= available_start:
        return segment

    src_offset = available_start - buffer_start
    dst_offset = available_start - start
    segment[dst_offset : dst_offset + (available_end - available_start)] = buffer[
        src_offset : src_offset + (available_end - available_start)
    ]
    return segment


def _bool_flag(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass
class AggregatorCLIConfig:
    enabled: bool
    track_limit: int
    guard_sec: float
    last_slack_sec: float
    protect_head_sec: float
    final_protect_sec: float
    commit_tail_sec: float
    epsilon: float
    fuse_back: float
    dedup_near_gap: float
    dedup_overlap_ratio: float
    dedup_repeat_gap: float
    dedup_bigram_gap: float
    back_overlap_tol: float
    cov_min_overlap_sec: float
    cov_min_cover_ratio: float
    srt_gap_break: float
    srt_max_line: int
    repair_enable: bool


@dataclass
class TrackAggregatorState:
    track_id: int
    aggregator: Any
    windows_seen: int = 0
    last_window_end: float = 0.0


class SSEBroker:
    def __init__(self) -> None:
        self.subscribers: List[queue.Queue] = []
        self.lock = threading.Lock()
        self.running = True

    def register(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unregister(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def publish(self, payload: str) -> None:
        with self.lock:
            targets = list(self.subscribers)
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass

    def stop(self) -> None:
        self.running = False
        with self.lock:
            targets = list(self.subscribers)
        for q in targets:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass


class AggregatorHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class AggregatorRequestHandler(http.server.BaseHTTPRequestHandler):
    server_version = "AggStream/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - debug noise
        logger.debug("agg_server: " + format, *args)

    def do_GET(self) -> None:  # pragma: no cover - I/O heavy
        if self.path.startswith("/stream"):
            broker: Optional[SSEBroker] = getattr(self.server, "broker", None)
            if broker is None or not broker.running:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "stream disabled")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = broker.register()
            try:
                while broker.running:
                    try:
                        payload = q.get(timeout=1.0)
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    if payload is None:
                        break
                    msg = f"data: {payload}\n\n".encode("utf-8")
                    self.wfile.write(msg)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                broker.unregister(q)
        elif self.path.startswith("/healthz"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{\"status\":\"ok\"}")
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def do_PATCH(self) -> None:  # pragma: no cover - network I/O
        if not self.path.startswith("/tracks/") or not self.path.endswith("/edits"):
            self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return
        manager = getattr(self.server, "manager", None)
        if manager is None or not getattr(manager, "enabled", False):
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "aggregator disabled")
            return
        parts = [p for p in self.path.strip("/").split("/") if p]
        if len(parts) < 3:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid track path")
            return
        try:
            track_id = int(parts[1])
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid track id")
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid json")
            return
        ok, resp = manager.apply_edit(track_id, payload)
        if not ok:
            self.send_error(HTTPStatus.BAD_REQUEST, resp.get("error", "edit failed"))
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))


class ViewUpdateServer:
    def __init__(self, manager: "FastAggregatorManager", host: str, port: int, enable: bool) -> None:
        self.manager = manager
        self.host = host
        self.port = port
        self.enable = enable
        self.httpd: Optional[AggregatorHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.broker: Optional[SSEBroker] = None
        if not enable:
            return
        try:
            self.broker = SSEBroker()
            self.httpd = AggregatorHTTPServer((host, port), AggregatorRequestHandler)
            self.httpd.manager = manager
            self.httpd.broker = self.broker
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()
            logger.info("Aggregator stream server listening on http://%s:%s/stream", host, port)
        except Exception as exc:
            logger.warning("Failed to start aggregator stream server (%s:%s): %s", host, port, exc)
            self.enable = False
            self.httpd = None
            self.broker = None

    def publish(self, event: Dict[str, Any]) -> None:
        if not self.enable or not self.broker:
            return
        payload = json.dumps(event, ensure_ascii=False)
        self.broker.publish(payload)

    def shutdown(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.broker:
            self.broker.stop()


class FastAggregatorManager:
    def __init__(
        self,
        components: Optional[FastComponents],
        config: AggregatorCLIConfig,
        track_limit: int,
        server_host: str,
        server_port: int,
        server_enable: bool,
        edits_log_path: Path,
    ) -> None:
        self.components = components
        self.config = config
        self.track_limit = max(0, track_limit)
        self.enabled = bool(components and config.enabled and self.track_limit > 0)
        self.lock = threading.Lock()
        self.tracks: Dict[int, TrackAggregatorState] = {}
        self.edits_log_path = edits_log_path
        if self.enabled:
            self.edits_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.server = ViewUpdateServer(self, server_host, server_port, enable=self.enabled and server_enable)

    def shutdown(self) -> None:
        if self.server:
            self.server.shutdown()

    def _ensure_state(self, track_id: int) -> TrackAggregatorState:
        state = self.tracks.get(track_id)
        if state is None:
            agg = self._build_aggregator()
            state = TrackAggregatorState(track_id=track_id, aggregator=agg)
            self.tracks[track_id] = state
        return state

    def _build_aggregator(self) -> Any:
        assert self.components is not None
        cfg = self.config
        agg = self.components.Aggregator(
            commit_tail_sec=cfg.commit_tail_sec,
            epsilon=cfg.epsilon,
            san_near_dup_gap=cfg.dedup_near_gap,
            san_back_overlap_tol=cfg.back_overlap_tol,
            san_overlap_ratio=cfg.dedup_overlap_ratio,
            cov_min_overlap_sec=cfg.cov_min_overlap_sec,
            cov_min_cover_ratio=cfg.cov_min_cover_ratio,
        )
        agg.fuse_back = cfg.fuse_back
        agg.dedup_near_gap = cfg.dedup_near_gap
        agg.dedup_overlap_ratio = cfg.dedup_overlap_ratio
        agg.dedup_repeat_gap = cfg.dedup_repeat_gap
        agg.dedup_bigram_gap = cfg.dedup_bigram_gap
        agg.front_grace_sec = cfg.guard_sec
        return agg

    def append_window_tokens(
        self,
        track_id: int,
        words: List[Dict[str, Any]],
        window_start: float,
        window_end: float,
        is_last_window: bool,
        window_index: int,
        rtf: float,
    ) -> None:
        if not self.enabled or self.components is None:
            return
        if track_id >= self.track_limit:
            return
        tokens = self._words_to_tokens(words)
        if not tokens:
            return
        with self.lock:
            state = self._ensure_state(track_id)
            filtered = self._filter_tokens(state, tokens, window_start, window_end, is_last_window)
            if not filtered and not is_last_window:
                state.windows_seen += 1
                state.last_window_end = window_end
                return
            if filtered:
                state.aggregator.append_fast(filtered)
            state.windows_seen += 1
            state.last_window_end = window_end
        self._publish_view_update(track_id, window_end, window_index, rtf)

    def finalize_all(self) -> None:
        if not self.enabled or self.components is None:
            return
        with self.lock:
            for state in self.tracks.values():
                state.aggregator.finalize()
        for track_id, state in self.tracks.items():
            self._publish_view_update(track_id, state.last_window_end, -1, 0.0)

    def export_transcripts(self, base_dir: Path, stem: str, repair_enable: bool, gap_break: float, max_line: int) -> None:
        if not self.enabled or self.components is None:
            return
        base_dir.mkdir(parents=True, exist_ok=True)
        for track_id, state in sorted(self.tracks.items()):
            toks = list(state.aggregator.state.committed)
            if not toks:
                continue
            txt_path = base_dir / f"{stem}_track{track_id}.txt"
            self.components.write_txt(txt_path, toks)
            srt_path = base_dir / f"{stem}_track{track_id}.srt"
            if repair_enable and self.components.write_srt_with_repair:
                dict_tokens = [
                    {"text": t.text, "start": t.start, "end": t.end, "prob": getattr(t, "prob", 0.0)}
                    for t in toks
                ]
                clean_text = self.components.tokens_to_text(toks)
                self.components.write_srt_with_repair(
                    str(srt_path),
                    dict_tokens,
                    clean_text,
                    gap_break=gap_break,
                    max_line=max_line,
                )
            else:
                self.components.write_srt(srt_path, toks, max_len=max_line, gap_break=gap_break)

    def apply_edit(self, track_id: int, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        if not self.enabled or self.components is None:
            return False, {"error": "aggregator disabled"}
        mode = str(payload.get("mode", "commit")).lower()
        if mode not in {"commit", "tail"}:
            return False, {"error": "mode must be 'commit' or 'tail'"}
        range_info = payload.get("range") or {}
        try:
            start = float(range_info.get("start"))
            end = float(range_info.get("end"))
        except (TypeError, ValueError):
            return False, {"error": "range.start/end required"}
        if end < start:
            start, end = end, start
        replace_text = str(payload.get("replace_text", "")).strip()
        user = str(payload.get("user", "anonymous"))
        with self.lock:
            state = self.tracks.get(track_id)
            if state is None:
                return False, {"error": "track not found"}
            tokens = state.aggregator.state.committed if mode == "commit" else state.aggregator.state.tail
            if not tokens:
                return False, {"error": "no tokens to edit"}
            replaced, old_text = self._replace_tokens(tokens, start, end, replace_text)
            if not replaced:
                return False, {"error": "no overlapping tokens"}
            self._log_edit(track_id, user, start, end, old_text, replace_text, mode)
        self._publish_view_update(track_id, self.tracks[track_id].last_window_end, -1, 0.0)
        return True, {"status": "ok"}

    # Internal helpers -----------------------------------------------------

    def _replace_tokens(
        self,
        tokens: List[Any],
        start: float,
        end: float,
        new_text: str,
    ) -> Tuple[Optional[Any], str]:
        overlapping = [t for t in tokens if not (t.end <= start or t.start >= end)]
        if not overlapping:
            return None, ""
        span_start = min(start, min(t.start for t in overlapping))
        span_end = max(end, max(t.end for t in overlapping))
        old_text = self.components.tokens_to_text(overlapping)
        remaining = [t for t in tokens if t not in overlapping]
        new_tok = self.components.Tok(text=new_text or "", start=span_start, end=span_end, prob=1.0)
        remaining.append(new_tok)
        remaining.sort(key=lambda t: (t.start, t.end))
        tokens[:] = remaining
        return new_tok, old_text

    def _words_to_tokens(self, words: List[Dict[str, Any]]) -> List[Any]:
        if not words:
            return []
        tokens = []
        for w in words:
            text = str(w.get("word") or w.get("text") or "").strip()
            if not text:
                continue
            start = float(w.get("start") or 0.0)
            end = float(w.get("end") or start)
            if end < start:
                end = start
            prob = float(w.get("probability") or w.get("prob") or 0.0)
            tokens.append(self.components.Tok(text=text, start=start, end=end, prob=prob))
        return tokens

    def _filter_tokens(
        self,
        state: TrackAggregatorState,
        tokens: List[Any],
        win_start: float,
        win_end: float,
        is_last_window: bool,
    ) -> List[Any]:
        agg = state.aggregator
        cfg = self.config
        eps = cfg.epsilon
        if state.windows_seen == 0 or win_start < cfg.guard_sec:
            accept_s = 0.0
        else:
            accept_s = win_start + cfg.guard_sec
        accept_s = max(accept_s, agg.state.last_committed_end - eps)

        def keep_token(tok: Any) -> bool:
            if tok.start < accept_s - eps:
                return False
            if is_last_window:
                return tok.end <= (win_end + cfg.last_slack_sec) + eps
            return tok.end <= (win_end - cfg.guard_sec) + eps

        kept = [t for t in tokens if keep_token(t)]
        if agg.state.tail and cfg.protect_head_sec > 0:
            tail_head = agg.state.tail[0].start
            cutoff = max(tail_head + cfg.protect_head_sec, agg.state.last_committed_end + 0.005, accept_s)
            seam_free = agg.state.tail[-1].end - agg.fuse_back
            filtered = []
            for t in kept:
                if t.start >= cutoff - agg.eps:
                    filtered.append(t)
                elif t.start >= seam_free - agg.eps:
                    filtered.append(t)
            kept = filtered

        if is_last_window and agg.state.tail and cfg.final_protect_sec > 0:
            tail_end = agg.state.tail[-1].end
            protect_start = max(accept_s, tail_end - cfg.final_protect_sec)
            kept = [t for t in kept if t.start >= protect_start - agg.eps]
        return kept

    def _publish_view_update(self, track_id: int, timestamp: float, window_index: int, rtf: float) -> None:
        if not self.enabled or self.components is None:
            return
        state = self.tracks.get(track_id)
        if state is None:
            return
        agg = state.aggregator
        committed_text = self.components.tokens_to_text(agg.state.committed)
        tail_text = self.components.tokens_to_text(agg.state.tail)
        tail_duration = agg.tail_duration()
        tail_probs = [getattr(t, "prob", 0.0) for t in agg.state.tail]
        avg_tail_prob = float(sum(tail_probs) / len(tail_probs)) if tail_probs else 0.0
        event = {
            "type": "view_update",
            "t": round(float(timestamp), 3),
            "track_id": track_id,
            "window_index": window_index,
            "committed_text": committed_text,
            "tail_text": tail_text,
            "tail_duration": tail_duration,
            "last_committed_end": agg.state.last_committed_end,
            "stats": {
                "rtf": float(rtf),
                "avg_tail_prob": avg_tail_prob,
            },
        }
        if self.server:
            self.server.publish(event)
            if agg.state.tail:
                tail_event = {
                    "type": "tail_tokens",
                    "track_id": track_id,
                    "tokens": [
                        {
                            "text": t.text,
                            "start": t.start,
                            "end": t.end,
                            "prob": getattr(t, "prob", 0.0),
                        }
                        for t in agg.state.tail
                    ],
                }
                self.server.publish(tail_event)

    def _log_edit(self, track_id: int, user: str, start: float, end: float, old: str, new: str, mode: str) -> None:
        if not self.enabled:
            return
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "track_id": track_id,
            "user": user,
            "mode": mode,
            "range": {"start": start, "end": end},
            "old_text": old,
            "new_text": new,
        }
        try:
            with self.edits_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Failed to write edit log entry", exc_info=True)

class WindowOrchestrator:
    def __init__(
        self,
        separator: AudioSeparator,
        identifier: SpeakerIdentifier,
        asr: Optional[WhisperASR],
        mode: str,
        sep_thresholds: SeparationThresholds,
        sep_dsp: SeparationDSPConfig,
        asr_gate: AsrGateConfig,
        asr_norm: AsrNormalizationConfig,
        asr_config: AsrRuntimeConfig,
        enable_asr: bool = True,
        debug: bool = False,
        debug_audio: bool = False,
        dump_audio: bool = True,
        dump_variants: Optional[List[str]] = None,
        dump_root: Optional[Path] = None,
        agg_manager: Optional[FastAggregatorManager] = None,
    ) -> None:
        self.sep = separator
        self.identifier = identifier
        self.asr = asr
        self.mode = mode
        self.sep_thresholds = sep_thresholds
        self.sep_dsp = sep_dsp
        self.asr_gate = asr_gate
        self.asr_norm = asr_norm
        self.asr_config = asr_config
        self.enable_asr = enable_asr and (mode == "pipeline")
        self.debug = debug
        self.debug_audio = debug_audio
        self.agg_manager = agg_manager

        self.matcher = HungarianAB()
        self._asr_kwargs = {
            "language": asr_config.language,
            "beam_size": asr_config.beam_size,
            "best_of": asr_config.best_of,
            "temperature": asr_config.temperature,
            "task": asr_config.task,
            "condition_on_previous_text": asr_config.condition_on_previous_text,
            "vad_filter": asr_config.vad_filter,
            "vad_parameters": asr_config.vad_parameters,
            "no_speech_threshold": asr_config.no_speech_threshold,
            "compression_ratio_threshold": asr_config.compression_ratio_threshold,
            "logprob_threshold": asr_config.logprob_threshold,
            "suppress_tokens": asr_config.suppress_tokens,
        }

        valid_variants = {"pre", "post", "norm"}
        variant_set = {v.lower() for v in (dump_variants or ["post"])}
        self.dump_variants = {v for v in variant_set if v in valid_variants}
        if not self.dump_variants:
            self.dump_variants = {"post"}

        dump_root_path = Path(dump_root) if dump_root is not None else None
        self.dump_root = dump_root_path if dump_audio and dump_root_path is not None else None
        self.dump_audio = self.dump_root is not None
        if self.dump_audio:
            self.dump_root.mkdir(parents=True, exist_ok=True)
    def process_window(
        self,
        window_idx: int,
        t_start: float,
        t_end: float,
        chunk: np.ndarray,
        chunk_sr: int,
        pad_segment: Optional[np.ndarray] = None,
        is_last_window: bool = False,
    ) -> Dict[str, Any]:
        if chunk.size == 0:
            return self._empty_result(window_idx, t_start, t_end, 0)

        chunk = chunk.astype(np.float32, copy=False)
        energy = float(np.mean(np.abs(chunk)))
        if energy < MIN_ENERGY_THRESHOLD:
            logger.info(
                "Window %03d [%.2f, %.2f] skipped due to low energy %.6f",
                window_idx,
                t_start,
                t_end,
                energy,
            )
            return self._empty_result(window_idx, t_start, t_end, 0)

        if chunk_sr != TARGET_RATE:
            chunk = resample_poly(chunk, TARGET_RATE, chunk_sr).astype(np.float32)
            chunk_sr = TARGET_RATE

        target_samples = chunk.shape[0]
        sr = chunk_sr
        pad_left_samples = int(round(self.sep_dsp.pad_left * sr))
        pad_right_samples = int(round(self.sep_dsp.pad_right * sr))

        if pad_segment is None:
            pad_segment = np.pad(
                chunk,
                (pad_left_samples, pad_right_samples),
                mode="constant",
                constant_values=0.0,
            )
        else:
            expected = pad_left_samples + target_samples + pad_right_samples
            if pad_segment.shape[0] != expected:
                padded = np.zeros(expected, dtype=np.float32)
                length = min(expected, pad_segment.shape[0])
                padded[:length] = pad_segment[:length]
                pad_segment = padded

        if self.debug:
            logger.debug(
                "Window %03d pad_left=%.3fs pad_right=%.3fs samples=%d",
                window_idx,
                self.sep_dsp.pad_left,
                self.sep_dsp.pad_right,
                pad_segment.shape[0],
            )

        central_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.sep.device)
        padded_tensor = torch.from_numpy(pad_segment).unsqueeze(0).to(self.sep.device)

        with torch.no_grad():
            detected = self.sep.spk_counter.count_with_refine(
                audio=central_tensor,
                sample_rate=sr,
                expected_min=1,
                expected_max=3,
                first_pass_range=(1, 3),
                allow_zero=True,
                debug=False,
            )
        if detected <= 0:
            with torch.no_grad():
                detected = self.sep.spk_counter.count_with_refine(
                    audio=central_tensor,
                    sample_rate=sr,
                    expected_min=1,
                    expected_max=3,
                    first_pass_range=(1, 3),
                    allow_zero=False,
                    debug=False,
                )
        if detected <= 0:
            logger.info(
                "Window %03d [%.2f, %.2f] contains no detectable speakers.",
                window_idx,
                t_start,
                t_end,
            )
            return self._empty_result(window_idx, t_start, t_end, 0)
        separated_sources, total_sources = self._separate_sources(
            padded_tensor,
            detected,
            window_idx,
            crop_start=pad_left_samples,
            crop_length=target_samples,
        )
        if not separated_sources:
            logger.warning(
                "Window %03d [%.2f, %.2f] separation returned no sources.",
                window_idx,
                t_start,
                t_end,
            )
            return self._empty_result(window_idx, t_start, t_end, 0)

        raw_sources = [src.copy() for src in separated_sources]
        processed_sources, mask_applied, mc_applied = self._apply_post_filters(
            separated_sources,
            chunk,
        )

        source_infos: List[Dict[str, Any]] = []
        for audio, raw in zip(processed_sources, raw_sources):
            metrics = compute_audio_metrics(audio, sr)
            dropped = should_drop_source(metrics, self.sep_thresholds)
            source_infos.append(
                {
                    "audio": audio,
                    "raw_audio": raw,
                    "metrics": metrics,
                    "dropped": dropped,
                }
            )

        kept_infos = [info for info in source_infos if not info["dropped"]]
        dropped_infos = [info for info in source_infos if info["dropped"]]

        if self.debug:
            metrics_table = [
                {
                    "rms_db": round(info["metrics"]["rms_db"], 2),
                    "voiced_ratio": round(info["metrics"]["voiced_ratio"], 3),
                    "duration_s": round(info["metrics"]["duration_s"], 3),
                    "peak_db": round(info["metrics"]["peak_db"], 2),
                    "dropped": info["dropped"],
                }
                for info in source_infos
            ]
            logger.debug("Window %03d source metrics: %s", window_idx, metrics_table)

        if not kept_infos:
            return self._build_result(
                window_idx,
                t_start,
                t_end,
                len(source_infos),                0,
                [],
                dropped_infos,
                mask_applied,
                mc_applied,
            )
        embeddings: List[np.ndarray] = []
        for info in kept_infos:
            try:
                if self.identifier and hasattr(self.identifier, 'audio_processor'):
                    emb = self.identifier.audio_processor.extract_embedding_from_stream(
                        info["audio"], sr
                    )
                else:
                    # Fallback: use random but consistent embeddings when identifier is unavailable
                    logger.debug("Speaker identifier unavailable, using fallback embedding")
                    emb = np.random.RandomState(window_idx).randn(192).astype(np.float32)
            except Exception as exc:  # pragma: no cover
                logger.warning("Embedding extraction failed: %s", exc)
                emb = np.zeros(192, dtype=np.float32)
            embeddings.append(emb)

        assignment = self.matcher.assign(embeddings)
        if self.debug:
            cm = self.matcher.last_debug.get("cost_matrix")
            if cm is not None:
                logger.debug("Window %03d cost matrix:\n%s", window_idx, cm)
            logger.debug("Window %03d assignment: %s", window_idx, assignment)

        track_entries: Dict[int, Dict[str, Any]] = {0: None, 1: None}  # type: ignore
        track_audio_post: Dict[int, np.ndarray] = {}
        track_metrics: Dict[int, Dict[str, float]] = {}

        for src_idx, track_id in assignment.items():
            info = kept_infos[src_idx]
            metrics = info["metrics"]
            entry = self._base_track_entry(track_id, metrics)
            entry["mask_floor_used"] = mask_applied
            entry["mc_projected"] = mc_applied
            entry["skip_reason"] = None
            entry["fallback_pass"] = False
            entry["pre_norm_rms_db"] = None
            entry["gain_db_applied"] = None
            entry["asr_segments"] = []
            entry["segments"] = 0
            entry["avg_conf"] = None
            entry["audio_paths"] = {}
            track_entries[track_id] = entry
            track_audio_post[track_id] = info["audio"]
            track_metrics[track_id] = metrics
            if self.dump_audio and "post" in self.dump_variants:
                path = self._dump_audio_variant(
                    window_idx,
                    track_id,
                    metrics,
                    "post",
                    info["audio"],
                    sr,
                    t_start,
                    t_end,
                )
                if path:
                    entry["audio_paths"]["post"] = str(path)
            if self.dump_audio and "pre" in self.dump_variants and info.get("raw_audio") is not None:
                path = self._dump_audio_variant(
                    window_idx,
                    track_id,
                    metrics,
                    "pre",
                    info["raw_audio"],
                    sr,
                    t_start,
                    t_end,
                )
                if path:
                    entry["audio_paths"]["pre"] = str(path)

        remaining_track_ids = [tid for tid in (0, 1) if track_entries[tid] is None]
        for info, track_id in zip(dropped_infos, remaining_track_ids):
            metrics = info["metrics"]
            entry = self._base_track_entry(track_id, metrics)
            entry["mask_floor_used"] = mask_applied
            entry["mc_projected"] = mc_applied
            entry["skip_reason"] = "silent_source_dropped"
            entry["fallback_pass"] = False
            entry["pre_norm_rms_db"] = None
            entry["gain_db_applied"] = None
            entry["asr_segments"] = []
            entry["segments"] = 0
            entry["avg_conf"] = None
            entry["audio_paths"] = {}
            track_entries[track_id] = entry

        for track_id in (0, 1):
            if track_entries[track_id] is None:
                entry = {
                    "track_id": track_id,
                    "rms_db": None,
                    "voiced_ratio": None,
                    "duration_s": None,
                    "peak_db": None,
                    "pre_norm_rms_db": None,
                    "gain_db_applied": None,
                    "segments": 0,
                    "avg_conf": None,
                    "skip_reason": "no_source_detected",
                    "fallback_pass": False,
                    "mask_floor_used": mask_applied,
                    "mc_projected": mc_applied,
                    "asr_segments": [],
                    "audio_paths": {},
                }
                track_entries[track_id] = entry

        norm_cache: Dict[int, Tuple[np.ndarray, float, float]] = {}
        need_norm = (self.enable_asr and self.asr and track_audio_post) or (
            self.dump_audio and "norm" in self.dump_variants
        )
        if need_norm:
            for track_id, audio in track_audio_post.items():
                normalized, pre_rms_db, gain_db = self._normalize_for_asr(audio)
                norm_cache[track_id] = (normalized, pre_rms_db, gain_db)
                track_entries[track_id]["pre_norm_rms_db"] = round(pre_rms_db, 3)
                track_entries[track_id]["gain_db_applied"] = round(gain_db, 3)
                if self.dump_audio and "norm" in self.dump_variants:
                    metrics = track_metrics[track_id]
                    path = self._dump_audio_variant(
                        window_idx,
                        track_id,
                        metrics,
                        "norm",
                        normalized,
                        sr,
                        t_start,
                        t_end,
                    )
                    if path:
                        track_entries[track_id]["audio_paths"]["norm"] = str(path)

        if self.enable_asr and self.asr and track_audio_post:
            gating_results: Dict[int, bool] = {}
            for track_id, metrics in track_metrics.items():
                passes_gate = passes_asr_gate(metrics, self.asr_gate)
                gating_results[track_id] = passes_gate
                if not passes_gate:
                    track_entries[track_id]["skip_reason"] = "asr_gated_low_voice"

            if gating_results and not any(gating_results.values()):
                chosen = max(
                    gating_results.items(),
                    key=lambda kv: track_entries[kv[0]]["voiced_ratio"],
                )[0]
                gating_results[chosen] = True
                track_entries[chosen]["skip_reason"] = None
                track_entries[chosen]["fallback_pass"] = True
                for track_id in gating_results.keys():
                    if track_id != chosen:
                        track_entries[track_id]["skip_reason"] = "both_failed_fallback"
                if self.debug:
                    logger.debug(
                        "Window %03d both tracks gated; fallback passing track %d",
                        window_idx,
                        chosen,
                    )

            for track_id, passed in gating_results.items():
                if not passed:
                    continue

                audio = track_audio_post[track_id]
                if track_id in norm_cache:
                    normalized, pre_rms_db, gain_db = norm_cache[track_id]
                else:
                    normalized, pre_rms_db, gain_db = self._normalize_for_asr(audio)
                    track_entries[track_id]["pre_norm_rms_db"] = round(pre_rms_db, 3)
                    track_entries[track_id]["gain_db_applied"] = round(gain_db, 3)
                    if self.dump_audio and "norm" in self.dump_variants:
                        metrics = track_metrics[track_id]
                        path = self._dump_audio_variant(
                            window_idx,
                            track_id,
                            metrics,
                            "norm",
                            normalized,
                            sr,
                            t_start,
                            t_end,
                        )
                        if path:
                            track_entries[track_id]["audio_paths"]["norm"] = str(path)

                segments, abs_words, asr_rtf = self._run_asr_on_source(
                    normalized,
                    sr,
                    t_start,
                    t_end,
                )
                track_entries[track_id]["asr_segments"] = segments
                track_entries[track_id]["segments"] = len(segments)
                track_entries[track_id]["avg_conf"] = (
                    round(average_confidence(segments) or 0.0, 3) if segments else None
                )
                if self.debug:
                    logger.debug(
                        "Window %03d track %d ASR segments=%d avg_conf=%s",
                        window_idx,
                        track_id,
                        len(segments),
                        track_entries[track_id]["avg_conf"],
                    )
                if self.agg_manager and abs_words:
                    self.agg_manager.append_window_tokens(
                        track_id=track_id,
                        words=abs_words,
                        window_start=t_start,
                        window_end=t_end,
                        is_last_window=is_last_window,
                        window_index=window_idx,
                        rtf=asr_rtf,
                    )

        elif self.enable_asr and not self.asr:
            logger.error("ASR module is unavailable; skipping ASR inference.")
            for track_id in track_audio_post.keys():
                track_entries[track_id]["skip_reason"] = "asr_module_missing"

        tracks_payload = [track_entries[0], track_entries[1]]
        kept_count = len(kept_infos)
        return {
            "window_index": window_idx,
            "t_start": round(float(t_start), 3),
            "t_end": round(float(t_end), 3),
            "num_sources_before_drop": len(source_infos),
            "num_sources_after_drop": kept_count,
            "num_sources": kept_count,
            "tracks": tracks_payload,
        }
    def _base_track_entry(self, track_id: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        return {
            "track_id": track_id,
            "rms_db": round(float(metrics["rms_db"]), 3),
            "voiced_ratio": round(float(metrics["voiced_ratio"]), 3),
            "duration_s": round(float(metrics["duration_s"]), 3),
            "peak_db": round(float(metrics["peak_db"]), 3),
            "audio_paths": {},
        }

    def _empty_result(
        self,
        window_idx: int,
        t_start: float,
        t_end: float,
        num_sources: int,
    ) -> Dict[str, Any]:
        template = {
            "track_id": 0,
            "rms_db": None,
            "voiced_ratio": None,
            "duration_s": None,
            "peak_db": None,
            "pre_norm_rms_db": None,
            "gain_db_applied": None,
            "segments": 0,
            "avg_conf": None,
            "skip_reason": "no_source_detected",
            "fallback_pass": False,
            "mask_floor_used": False,
            "mc_projected": False,
            "asr_segments": [],
            "audio_paths": {},
        }
        track0 = template.copy()
        track1 = template.copy()
        track1["track_id"] = 1
        return {
            "window_index": window_idx,
            "t_start": round(float(t_start), 3),
            "t_end": round(float(t_end), 3),
            "num_sources_before_drop": num_sources,
            "num_sources_after_drop": 0,
            "num_sources": 0,
            "tracks": [track0, track1],
        }

    def _separate_sources(
        self,
        audio_tensor: torch.Tensor,
        detected: int,
        window_idx: int,
        crop_start: int,
        crop_length: int,
    ) -> Tuple[List[np.ndarray], int]:
        with torch.no_grad():
            model, _ = self.sep._get_appropriate_model(int(max(1, detected)))
            prepared = audio_tensor
            if prepared.dim() == 3 and prepared.shape[1] == 1:
                prepared = prepared.squeeze(1)

            estimates = model.separate_batch(prepared)
            estimates, layout, _, _ = self.sep._normalize_estimates(estimates)

        if layout == "BST":
            est_st = estimates[0]
        elif layout == "BTS":
            est_st = estimates[0].transpose(0, 1)
        elif layout == "BT":
            est_st = estimates.unsqueeze(0) if estimates.dim() == 1 else estimates
        else:
            est_st = estimates

        if est_st.dim() == 1:
            est_st = est_st.unsqueeze(0)

        total_sources = est_st.shape[0]
        energy = est_st.pow(2).mean(dim=1)
        order = torch.argsort(energy, descending=True)
        top_count = min(2, total_sources)
        if total_sources > top_count:
            logger.warning(
                "Window %03d detected %d sources; selecting top-%d by energy.",
                window_idx,
                total_sources,
                top_count,
            )
        top_indices = order[:top_count]
        separated = []
        for idx in top_indices:
            source = est_st[idx].detach().cpu().numpy().astype(np.float32)
            source = source[crop_start : crop_start + crop_length]
            separated.append(source)
        return separated, total_sources

    def _apply_post_filters(
        self,
        sources: List[np.ndarray],
        mixture: np.ndarray,
    ) -> Tuple[List[np.ndarray], bool, bool]:
        if not sources:
            return sources, False, False

        processed = [src.copy() for src in sources]
        mask_applied = False
        if self.sep_dsp.mask_floor > 0.0:
            mask_applied = True
            mix_abs = np.abs(mixture) + 1e-9
            for idx, src in enumerate(processed):
                mask = np.abs(src) / mix_abs
                mask = np.clip(mask, self.sep_dsp.mask_floor, 1.0)
                processed[idx] = np.sign(src) * mask * mix_abs

        mc_applied = False
        if self.sep_dsp.mixture_consistency and processed:
            mc_applied = True
            stacked = np.stack(processed, axis=1)
            coeffs, _, _, _ = np.linalg.lstsq(stacked, mixture, rcond=None)
            processed = [(processed[i] * coeffs[i]).astype(np.float32) for i in range(stacked.shape[1])]

        return [src.astype(np.float32) for src in processed], mask_applied, mc_applied

    def _normalize_for_asr(self, audio: np.ndarray) -> Tuple[np.ndarray, float, float]:
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
        pre_rms_db = 20.0 * math.log10(max(rms, 1e-9))

        target_linear = 10.0 ** (self.asr_norm.target_rms_db / 20.0)
        gain = target_linear / max(rms, 1e-9)
        peak = float(np.max(np.abs(audio)) + 1e-9)
        peak_limit_linear = 10.0 ** (self.asr_norm.peak_limit_db / 20.0)
        if peak * gain > peak_limit_linear:
            gain = peak_limit_linear / peak
        gain_db = 20.0 * math.log10(max(gain, 1e-9))
        normalized = np.clip(audio * gain, -peak_limit_linear, peak_limit_linear).astype(np.float32)
        return normalized, pre_rms_db, gain_db

    def _offset_words(self, words: List[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
        adjusted: List[Dict[str, Any]] = []
        for w in words or []:
            start = float(w.get("start") or 0.0) + offset
            end = float(w.get("end") or w.get("start") or 0.0) + offset
            adjusted.append(
                {
                    "word": str(w.get("word") or w.get("text") or "").strip(),
                    "start": start,
                    "end": end,
                    "probability": float(w.get("probability") or w.get("prob") or 0.0),
                }
            )
        return adjusted

    def _run_asr_on_source(
        self,
        audio: np.ndarray,
        sr: int,
        window_start: float,
        window_end: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        if not self.asr or audio.size == 0:
            return [], [], 0.0

        tensor = torch.from_numpy(audio).unsqueeze(0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        start_time = time.perf_counter()
        try:
            torchaudio.save(tmp_path, tensor, sr)
            text, avg_conf, words = self.asr.transcribe(tmp_path, **self._asr_kwargs)
            segments = _blend_words_to_segment(words, window_start, window_end, text, avg_conf)
            abs_words = self._offset_words(words, window_start)
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass

        elapsed = time.perf_counter() - start_time
        duration = max(window_end - window_start, 1e-6)
        rtf = float(elapsed / duration)
        return segments, abs_words, rtf

    def _dump_audio_variant(
        self,
        window_idx: int,
        track_id: int,
        metrics: Dict[str, float],
        variant: str,
        audio: np.ndarray,
        sr: int,
        t_start: float,
        t_end: float,
    ) -> Optional[Path]:
        if not self.dump_audio or self.dump_root is None or audio.size == 0:
            return None
        window_dir = self.dump_root / f"win_{window_idx:04d}"
        window_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"track_{track_id}_t{t_start:.2f}-{t_end:.2f}"
            f"_rms{metrics.get('rms_db', 0.0):.1f}_vr{metrics.get('voiced_ratio', 0.0):.2f}_{variant}.wav"
        )
        path = window_dir / filename
        tensor = torch.from_numpy(audio).unsqueeze(0)
        torchaudio.save(str(path), tensor, sr)
        return path

    def _build_result(
        self,
        window_idx: int,
        t_start: float,
        t_end: float,
        sources_before: int,
        sources_after: int,
        kept_infos: List[Dict[str, Any]],
        dropped_infos: List[Dict[str, Any]],
        mask_applied: bool,
        mc_applied: bool,
    ) -> Dict[str, Any]:
        track_entries: List[Dict[str, Any]] = []
        all_infos = kept_infos + dropped_infos
        for track_id in (0, 1):
            entry = {
                "track_id": track_id,
                "rms_db": None,
                "voiced_ratio": None,
                "duration_s": None,
                "peak_db": None,
                "pre_norm_rms_db": None,
                "gain_db_applied": None,
                "segments": 0,
                "avg_conf": None,
                "skip_reason": "no_source_detected",
                "fallback_pass": False,
                "mask_floor_used": mask_applied,
                "mc_projected": mc_applied,
                "asr_segments": [],
                "audio_paths": {},
            }
            if track_id < len(all_infos):
                metrics = all_infos[track_id]["metrics"]
                entry.update(
                    {
                        "rms_db": round(float(metrics["rms_db"]), 3),
                        "voiced_ratio": round(float(metrics["voiced_ratio"]), 3),
                        "duration_s": round(float(metrics["duration_s"]), 3),
                        "peak_db": round(float(metrics["peak_db"]), 3),
                        "skip_reason": "silent_source_dropped" if all_infos[track_id]["dropped"] else None,
                    }
                )
            track_entries.append(entry)

        return {
            "window_index": window_idx,
            "t_start": round(float(t_start), 3),
            "t_end": round(float(t_end), 3),
            "num_sources_before_drop": sources_before,
            "num_sources_after_drop": sources_after,
            "num_sources": sources_after,
            "tracks": track_entries,
        }

def run_file_pipeline(
    wav_path: Path,
    out_path: Path,
    win_len: float,
    stride: float,
    mode: str,
    sep_thresholds: SeparationThresholds,
    sep_dsp: SeparationDSPConfig,
    asr_gate: AsrGateConfig,
    asr_norm: AsrNormalizationConfig,
    enable_asr: bool,
    debug: bool,
    debug_audio: bool,
    asr_config: AsrRuntimeConfig,
    dump_audio: bool,
    dump_variants: List[str],
    dump_root: Optional[Path],
    agg_manager: Optional[FastAggregatorManager],
    id_backend: str = "auto",
) -> None:
    if mode == "asr_only":
        raise RuntimeError("run_file_pipeline cannot be used in ASR-only mode.")

    load_asr = mode == "pipeline" and enable_asr
    sep, identifier, asr, use_gpu = init_pipeline_modules(
        load_separator=True,
        load_identifier=True,
        load_asr=load_asr,
        id_backend=id_backend,
    )

    if asr:
        asr.lang = asr_config.language

    if dump_audio and dump_root is not None:
        dump_root.mkdir(parents=True, exist_ok=True)

    orchestrator = WindowOrchestrator(
        separator=sep,
        identifier=identifier,
        asr=asr,
        mode=mode,
        sep_thresholds=sep_thresholds,
        sep_dsp=sep_dsp,
        asr_gate=asr_gate,
        asr_norm=asr_norm,
        asr_config=asr_config,
        enable_asr=enable_asr,
        debug=debug,
        debug_audio=debug_audio,
        dump_audio=dump_audio,
        dump_variants=dump_variants,
        dump_root=dump_root,
        agg_manager=agg_manager,
    )

    waveform, sr = _prepare_audio(wav_path)
    if sr != TARGET_RATE:
        raise RuntimeError("Audio preparation failed to resample properly.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pad_left_samples = int(round(sep_dsp.pad_left * sr))
    pad_right_samples = int(round(sep_dsp.pad_right * sr))

    windows = list(generate_sliding_windows(waveform, sr, win_len=win_len, stride=stride))
    if not windows:
        logger.warning("Audio shorter than window length (%.2fs); nothing to process.", win_len)
        return

    with out_path.open("a", encoding="utf-8") as handle:
        total_windows = len(windows)
        for order_idx, window in enumerate(windows):
            idx, t0, t1, chunk, start, end = window
            padded = _extract_padded_segment(
                waveform,
                start,
                end,
                pad_left_samples,
                pad_right_samples,
            )
            result = orchestrator.process_window(
                idx,
                t0,
                t1,
                chunk,
                sr,
                pad_segment=padded,
                is_last_window=(order_idx == total_windows - 1),
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    if agg_manager:
        agg_manager.finalize_all()
        agg_manager.export_transcripts(
            out_path.parent,
            wav_path.stem,
            agg_manager.config.repair_enable,
            agg_manager.config.srt_gap_break,
            agg_manager.config.srt_max_line,
        )

    if use_gpu and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def run_asr_only_mix(
    wav_path: Path,
    out_path: Path,
    asr: WhisperASR,
    asr_config: AsrRuntimeConfig,
    asr_norm: AsrNormalizationConfig,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    asr.lang = asr_config.language
    kwargs = {
        "language": asr_config.language,
        "beam_size": asr_config.beam_size,
        "best_of": asr_config.best_of,
        "temperature": asr_config.temperature,
        "task": asr_config.task,
        "condition_on_previous_text": asr_config.condition_on_previous_text,
        "vad_filter": asr_config.vad_filter,
        "vad_parameters": asr_config.vad_parameters,
        "no_speech_threshold": asr_config.no_speech_threshold,
        "compression_ratio_threshold": asr_config.compression_ratio_threshold,
        "logprob_threshold": asr_config.logprob_threshold,
        "suppress_tokens": asr_config.suppress_tokens,
    }
    text, avg_conf, words = asr.transcribe(str(wav_path), **kwargs)
    result = {
        "mode": "asr_only_mix",
        "input": str(wav_path),
        "text": text,
        "avg_conf": avg_conf,
        "words": words,
        "target_rms_db": asr_norm.target_rms_db,
        "peak_limit_db": asr_norm.peak_limit_db,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("ASR-only mix result written to %s", out_path)


def run_asr_only_files(
    pattern: str,
    out_path: Path,
    asr: WhisperASR,
    asr_config: AsrRuntimeConfig,
    asr_norm: AsrNormalizationConfig,
) -> None:
    paths = sorted(Path().glob(pattern))
    if not paths:
        logger.warning("No files matched pattern %s", pattern)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    asr.lang = asr_config.language
    kwargs = {
        "language": asr_config.language,
        "beam_size": asr_config.beam_size,
        "best_of": asr_config.best_of,
        "temperature": asr_config.temperature,
        "task": asr_config.task,
        "condition_on_previous_text": asr_config.condition_on_previous_text,
        "vad_filter": asr_config.vad_filter,
        "vad_parameters": asr_config.vad_parameters,
        "no_speech_threshold": asr_config.no_speech_threshold,
        "compression_ratio_threshold": asr_config.compression_ratio_threshold,
        "logprob_threshold": asr_config.logprob_threshold,
        "suppress_tokens": asr_config.suppress_tokens,
    }

    with out_path.open("a", encoding="utf-8") as handle:
        for path in paths:
            text, avg_conf, words = asr.transcribe(str(path), **kwargs)
            result = {
                "mode": "asr_only_files",
                "input": str(path),
                "text": text,
                "avg_conf": avg_conf,
                "words": words,
                "target_rms_db": asr_norm.target_rms_db,
                "peak_limit_db": asr_norm.peak_limit_db,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info("ASR-only files result written to %s", out_path)

def run_pipeline_stream_v2(
    chunk_secs: float = 4.0,
    rate: int = 16000,
    channels: int = 1,
    frames_per_buffer: int = 1024,
    max_workers: int = 2,
    record_secs: Optional[float] = None,
    queue_out: "queue.Queue[dict] | None" = None,
    stop_event: Optional[threading.Event] = None,
    in_bytes_queue: "queue.Queue[bytes] | None" = None,
    sep: Optional[AudioSeparator] = None,
    spk: Optional[SpeakerIdentifier] = None,
    asr: Optional[WhisperASR] = None,
    out_path: Optional[Path] = None,
    debug_audio: bool = False,
    mode: str = "pipeline",
    sep_thresholds: SeparationThresholds = SeparationThresholds(0.30, -45.0, 0.50),
    sep_dsp: SeparationDSPConfig = SeparationDSPConfig(0.5, 0.5, 0.05, True),
    asr_gate: AsrGateConfig = AsrGateConfig(0.45, -45.0),
    asr_norm: AsrNormalizationConfig = AsrNormalizationConfig(-20.0, -1.0),
    enable_asr: bool = True,
    debug: bool = False,
    asr_config: AsrRuntimeConfig = AsrRuntimeConfig(),
    dump_audio: bool = True,
    dump_variants: Optional[List[str]] = None,
    dump_root: Optional[Path] = None,
    agg_manager: Optional[FastAggregatorManager] = None,
) -> None:
    if mode == "asr_only":
        raise ValueError("Streaming ASR-only mode is not supported.")

    if sep is None or spk is None or (asr is None and mode == "pipeline" and enable_asr):
        load_asr = mode == "pipeline" and enable_asr
        sep, spk, asr, _ = init_pipeline_modules(
            load_separator=True,
            load_identifier=True,
            load_asr=load_asr,
            id_backend="auto",
        )

    if asr:
        asr.lang = asr_config.language

    if agg_manager and getattr(agg_manager, "enabled", False) and max_workers > 1:
        logger.info("FAST aggregator enabled; forcing max_workers=1 to preserve window ordering.")
        max_workers = 1

    taipei_tz = timezone(timedelta(hours=8))
    stream_start = datetime.now(taipei_tz)

    effective_variants = dump_variants or ["post"]
    if dump_audio:
        if dump_root is None:
            dump_root = Path("outputs") / "separated" / stream_start.strftime("stream_%Y%m%d_%H%M%S")
        dump_root.mkdir(parents=True, exist_ok=True)

    orchestrator = WindowOrchestrator(
        separator=sep,
        identifier=spk,
        asr=asr,
        mode=mode,
        sep_thresholds=sep_thresholds,
        sep_dsp=sep_dsp,
        asr_gate=asr_gate,
        asr_norm=asr_norm,
        asr_config=asr_config,
        enable_asr=enable_asr,
        debug=debug,
        debug_audio=debug_audio,
        dump_audio=dump_audio,
        dump_variants=effective_variants,
        dump_root=dump_root,
        agg_manager=agg_manager,
    )

    output_path = out_path or Path("outputs") / "v2_windows.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
    pending: Dict[int, Future] = {}
    next_to_write = 0

    result_queue: "queue.Queue[Tuple[bytes, int, int, int, int]]" = queue.Queue(maxsize=max_workers * 2)
    shared_stop = stop_event or threading.Event()

    buffer = np.zeros(0, dtype=np.float32)
    buffer_start = 0
    total_samples = 0

    sr_established = False
    stride_samples = None
    win_samples = None
    pad_left_samples = None
    pad_right_samples = None

    window_start_sample = 0
    window_idx = 0

    out_handle = output_path.open("a", encoding="utf-8")
    def submit_window(idx: int, t_start: float, t_end: float, chunk: np.ndarray, padded: np.ndarray, sr: int) -> None:
        future = executor.submit(
            orchestrator.process_window,
            idx,
            t_start,
            t_end,
            chunk,
            sr,
            pad_segment=padded,
        )
        pending[idx] = future
        flush_pending()

    def flush_pending(force: bool = False) -> None:
        nonlocal next_to_write
        while next_to_write in pending:
            future = pending[next_to_write]
            if not future.done():
                if not force:
                    break
                try:
                    future.result(timeout=10.0)
                except Exception as exc:
                    logger.error("Window %03d failed: %s", next_to_write, exc)
            try:
                result = future.result()
            except Exception as exc:
                logger.error("Window %03d processing error: %s", next_to_write, exc)
                result = orchestrator._empty_result(
                    next_to_write,
                    next_to_write,
                    next_to_write + chunk_secs,
                    0,
                )
            out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_handle.flush()
            if queue_out is not None:
                queue_out.put(result)
            del pending[next_to_write]
            next_to_write += 1
    def recorder_from_queue() -> None:
        nonlocal sr_established, stride_samples, win_samples, pad_left_samples, pad_right_samples
        ch = max(1, int(channels))
        buf = bytearray()
        idx = 0
        start_time = time.time()

        provisional_sr = max(1, int(rate))
        frames_needed = provisional_sr

        calibrated = False
        calib_t0 = time.time()
        calib_bytes = 0
        bytes_per_sample = 4

        while not shared_stop.is_set():
            try:
                pkt = in_bytes_queue.get(timeout=0.1)  # type: ignore[arg-type]
            except queue.Empty:
                if record_secs is not None and time.time() - start_time >= record_secs:
                    shared_stop.set()
                    break
                continue

            buf.extend(pkt)
            calib_bytes += len(pkt)

            if not calibrated:
                elapsed = max(1e-3, time.time() - calib_t0)
                if elapsed >= 0.30:
                    candidates = [48000, 44100, 32000, 24000, 22050, 16000]
                    best_score = float("inf")
                    best_choice = (provisional_sr, 4)
                    for bps in (2, 4):
                        est_sr = calib_bytes / (elapsed * bps * ch)
                        candidate = min(candidates, key=lambda s: abs(s - est_sr))
                        score = abs(candidate - est_sr)
                        if score < best_score:
                            best_score = score
                            best_choice = (int(candidate), bps)
                    stride_samples = int(round(1.0 * best_choice[0]))
                    win_samples = int(round(chunk_secs * best_choice[0]))
                    pad_left_samples = int(round(sep_dsp.pad_left * best_choice[0]))
                    pad_right_samples = int(round(sep_dsp.pad_right * best_choice[0]))
                    frames_needed = best_choice[0]
                    bytes_per_sample = best_choice[1]
                    sr_established = True
                    calibrated = True
                    logger.info(
                        "[calib] est_sr≈%.1fHz -> use %dHz (bytes/sample=%d)",
                        calib_bytes / (elapsed * ch * best_choice[1]),
                        best_choice[0],
                        best_choice[1],
                    )

            if not calibrated:
                continue

            required = frames_needed * bytes_per_sample * ch
            while len(buf) >= required:
                raw = bytes(buf[:required])
                del buf[:required]
                result_queue.put((raw, idx, frames_needed, bytes_per_sample, ch))
                idx += 1

    def recorder_from_mic() -> None:
        nonlocal sr_established, stride_samples, win_samples, pad_left_samples, pad_right_samples
        if pyaudio is None:
            raise RuntimeError("pyaudio is required for microphone streaming mode.")

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=frames_per_buffer,
        )
        bytes_per_sample = 2
        stride_samples = int(round(1.0 * rate))
        win_samples = int(round(chunk_secs * rate))
        pad_left_samples = int(round(sep_dsp.pad_left * rate))
        pad_right_samples = int(round(sep_dsp.pad_right * rate))
        sr_established = True

        buf = bytearray()
        idx = 0
        start_time = time.time()

        try:
            while not shared_stop.is_set():
                data = stream.read(frames_per_buffer, exception_on_overflow=False)
                buf.extend(data)

                if record_secs is not None and time.time() - start_time >= record_secs:
                    shared_stop.set()
                    break

                required = stride_samples * bytes_per_sample * channels
                while len(buf) >= required:
                    raw = bytes(buf[:required])
                    del buf[:required]
                    result_queue.put((raw, idx, rate, bytes_per_sample, channels))
                    idx += 1
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
    ingest_thread = threading.Thread(
        target=recorder_from_queue if in_bytes_queue else recorder_from_mic,
        daemon=True,
    )
    ingest_thread.start()

    start_time = time.time()
    try:
        while not shared_stop.is_set():
            if record_secs is not None and time.time() - start_time >= record_secs:
                shared_stop.set()
                break

            try:
                raw, idx, src_sr, bytes_per_sample, ch = result_queue.get(timeout=0.1)
            except queue.Empty:
                flush_pending()
                continue

            decoded = _decode_audio_bytes(raw, ch, bytes_per_sample)
            buffer = np.concatenate([buffer, decoded])
            total_samples += decoded.shape[0]

            if not sr_established:
                continue

            min_retain = max(0, window_start_sample - pad_left_samples)
            if min_retain > buffer_start:
                drop = min_retain - buffer_start
                if drop > 0:
                    buffer = buffer[drop:]
                    buffer_start += drop

            while total_samples >= window_start_sample + win_samples + pad_right_samples:
                chunk = _segment_from_buffer(
                    buffer,
                    buffer_start,
                    window_start_sample,
                    window_start_sample + win_samples,
                )
                padded = _segment_from_buffer(
                    buffer,
                    buffer_start,
                    window_start_sample - pad_left_samples,
                    window_start_sample + win_samples + pad_right_samples,
                )
                t0 = window_start_sample / src_sr
                t1 = t0 + chunk_secs
                submit_window(
                    window_idx,
                    t0,
                    t1,
                    chunk.copy(),
                    padded.copy(),
                    src_sr,
                )
                window_idx += 1
                window_start_sample += stride_samples

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user; shutting down.")
        shared_stop.set()
    finally:
        shared_stop.set()
        ingest_thread.join(timeout=1.0)
        flush_pending(force=True)
        executor.shutdown(wait=True)
        out_handle.close()
        if agg_manager:
            agg_manager.finalize_all()
            agg_manager.export_transcripts(
                output_path.parent,
                f"stream_{stream_start.strftime('%Y%m%d_%H%M%S')}",
                agg_manager.config.repair_enable,
                agg_manager.config.srt_gap_break,
                agg_manager.config.srt_max_line,
            )
        logger.info(
            "Stream finished. Results appended to %s (started at %s)",
            output_path,
            stream_start.isoformat(),
        )

def _prepare_audio(path: Path) -> Tuple[np.ndarray, int]:
    waveform, sr = torchaudio.load(str(path))
    if waveform.ndim != 2:
        raise ValueError("Unexpected waveform shape.")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
    else:
        waveform = waveform.squeeze(0)

    waveform_np = waveform.cpu().numpy()
    if sr != TARGET_RATE:
        waveform_np = resample_poly(waveform_np, TARGET_RATE, sr)
        sr = TARGET_RATE
    return waveform_np.astype(np.float32), sr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliding-window orchestrator v2.")
    parser.add_argument("--wav", type=str, help="Input WAV file path.")
    parser.add_argument("--out", type=str, help="Output path (mode-dependent).")
    parser.add_argument("--win-len", type=float, default=4.0, help="Window length in seconds.")
    parser.add_argument("--stride", type=float, default=1.0, help="Stride in seconds (offline mode).")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode.")
    parser.add_argument("--from-queue", action="store_true", help="Stream from external bytes queue (programmatic use).")
    parser.add_argument("--record-secs", type=float, default=None, help="Limit stream duration in seconds.")
    parser.add_argument("--workers", type=int, default=2, help="ThreadPoolExecutor worker count.")
    parser.add_argument("--rate", type=int, default=16000, help="Recorder sample rate (mic mode).")
    parser.add_argument("--channels", type=int, default=1, help="Recorder channel count.")
    parser.add_argument("--frames-per-buffer", type=int, default=1024, help="PyAudio frames per buffer.")
    parser.add_argument("--debug-audio", action="store_true", help="Save debug audio artifacts (placeholder).")
    parser.add_argument("--debug", action="store_true", help="Enable verbose DEBUG logging.")
    parser.add_argument("--id-backend", choices=["auto", "wespeaker", "pyannote", "speechbrain", "none"], 
                        default="auto", help="Speaker identification backend ('auto', 'wespeaker', 'pyannote', 'speechbrain', 'none').")
    parser.add_argument("--id-device", choices=["cuda", "cpu", "auto"], default="auto",
                        help="Device for speaker identification model.")

    parser.add_argument("--mode", choices=["pipeline", "asr_only", "sep_only"], default="pipeline", help="Processing mode.")
    parser.add_argument("--asr-target", default="mix", help="ASR-only target: 'mix' or 'files:<glob>'.")

    parser.add_argument("--sep-pad-left", type=float, default=0.5, help="Left padding before separation (seconds).")
    parser.add_argument("--sep-pad-right", type=float, default=0.5, help="Right padding before separation (seconds).")
    parser.add_argument("--sep-mask-floor", type=float, default=0.05, help="Lower bound applied to separation masks.")
    parser.add_argument(
        "--sep-mixture-consistency",
        type=str,
        default="true",
        help="Enable mixture consistency projection (true/false).",
    )
    parser.add_argument("--sep-min-voiced", type=float, default=0.30, help="Minimum voiced ratio before separation drop.")
    parser.add_argument("--sep-min-rms-db", type=float, default=-45.0, help="Minimum RMS dBFS before separation drop.")
    parser.add_argument("--sep-min-duration", type=float, default=0.50, help="Minimum voiced duration before separation drop.")

    parser.add_argument("--enable-asr", dest="enable_asr", action="store_true", help="Enable ASR stage.")
    parser.add_argument("--disable-asr", dest="enable_asr", action="store_false", help="Disable ASR stage.")
    parser.set_defaults(enable_asr=True)

    parser.add_argument("--asr-min-voiced", type=float, default=0.45, help="ASR gating minimum voiced ratio.")
    parser.add_argument("--asr-min-rms-db", type=float, default=-45.0, help="ASR gating minimum RMS dBFS.")
    parser.add_argument("--asr-target-rms", type=float, default=-20.0, help="Target RMS (dBFS) for ASR normalization.")
    parser.add_argument("--asr-peak-limit", type=float, default=-1.0, help="Peak limiter (dBFS) applied before ASR.")
    parser.add_argument("--asr-language", type=str, default="zh", help="ASR language code.")
    parser.add_argument("--asr-beam", type=int, default=5, help="ASR beam size.")
    parser.add_argument("--asr-best-of", type=int, default=5, help="ASR best_of parameter.")
    parser.add_argument("--asr-temperature", type=float, default=0.0, help="ASR temperature.")
    parser.add_argument("--asr-cond-prev-text", action="store_true", help="ASR condition on previous text.")
    parser.add_argument("--asr-vad", action="store_true", help="ASR VAD filter.")
    parser.set_defaults(asr_cond_prev_text=False, asr_vad=True)

    parser.add_argument("--agg-enable", type=str, default="true", help="Enable FAST aggregator integration (true/false).")
    parser.add_argument("--agg-track-limit", type=int, default=2, help="Maximum number of tracks to aggregate.")
    parser.add_argument("--guard-sec", type=float, default=0.4, help="Sliding guard duration applied to each window.")
    parser.add_argument("--last-slack-sec", type=float, default=0.8, help="Slack applied to the final window when committing tokens.")
    parser.add_argument("--protect-head-sec", type=float, default=0.35, help="Protected duration for the aggregator tail head.")
    parser.add_argument("--final-protect-sec", type=float, default=0.6, help="Tail protection applied on the last window.")
    parser.add_argument("--commit-tail-sec", type=float, default=1.8, help="Maximum editable tail duration before committing.")
    parser.add_argument("--fuse-back", type=float, default=0.16, help="Fuse-back duration when aligning overlapping tokens.")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Aggregator epsilon guard.")
    parser.add_argument("--dedup-near-gap", type=float, default=0.10, help="Near-duplicate gap for deduplication.")
    parser.add_argument("--dedup-overlap-ratio", type=float, default=0.6, help="Overlap ratio threshold for deduplication.")
    parser.add_argument("--dedup-repeat-gap", type=float, default=0.22, help="Repeat gap threshold.")
    parser.add_argument("--dedup-bigram-gap", type=float, default=0.30, help="Bigram repeat gap threshold.")
    parser.add_argument("--dedup-back-overlap", type=float, default=0.02, help="Backward overlap tolerance.")
    parser.add_argument("--cov-min-overlap-sec", type=float, default=0.02, help="Coverage minimum overlap seconds.")
    parser.add_argument("--cov-min-cover-ratio", type=float, default=0.6, help="Coverage minimum ratio.")
    parser.add_argument("--agg-server-host", type=str, default="127.0.0.1", help="Aggregator stream server host.")
    parser.add_argument("--agg-server-port", type=int, default=8899, help="Aggregator stream server port.")
    parser.add_argument("--agg-server-enable", type=str, default="true", help="Enable the aggregator SSE server (true/false).")
    parser.add_argument("--repair-enable", action="store_true", help="Enable LLM repair for final subtitles.")
    parser.add_argument("--srt-gap-break", type=float, default=0.5, help="Gap used to break SRT segments.")
    parser.add_argument("--srt-max-line", type=int, default=70, help="Maximum characters per SRT line.")
    parser.add_argument(
        "--dump-separated-audio",
        type=str,
        default="true",
        help="Enable dumping separated audio variants (true/false).",
    )
    parser.add_argument(
        "--dump-variants",
        nargs="*",
        help="Separated audio variants to store (choices: pre post norm). Defaults to post.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    sep_thresholds = SeparationThresholds(
        min_voiced=args.sep_min_voiced,
        min_rms_db=args.sep_min_rms_db,
        min_duration=args.sep_min_duration,
    )
    sep_dsp = SeparationDSPConfig(
        pad_left=args.sep_pad_left,
        pad_right=args.sep_pad_right,
        mask_floor=args.sep_mask_floor,
        mixture_consistency=_bool_flag(args.sep_mixture_consistency, True),
    )
    asr_gate = AsrGateConfig(
        min_voiced=args.asr_min_voiced,
        min_rms_db=args.asr_min_rms_db,
    )
    asr_norm = AsrNormalizationConfig(
        target_rms_db=args.asr_target_rms,
        peak_limit_db=args.asr_peak_limit,
    )
    asr_config = AsrRuntimeConfig(
        language=args.asr_language,
        beam_size=args.asr_beam,
        best_of=args.asr_best_of,
        temperature=args.asr_temperature,
        condition_on_previous_text=args.asr_cond_prev_text,
        vad_filter=args.asr_vad,
    )

    agg_enabled = _bool_flag(getattr(args, "agg_enable", "true"), True) and args.mode == "pipeline" and args.enable_asr
    agg_config = AggregatorCLIConfig(
        enabled=agg_enabled,
        track_limit=max(0, getattr(args, "agg_track_limit", 2)),
        guard_sec=args.guard_sec,
        last_slack_sec=args.last_slack_sec,
        protect_head_sec=args.protect_head_sec,
        final_protect_sec=args.final_protect_sec,
        commit_tail_sec=args.commit_tail_sec,
        epsilon=args.epsilon,
        fuse_back=args.fuse_back,
        dedup_near_gap=args.dedup_near_gap,
        dedup_overlap_ratio=args.dedup_overlap_ratio,
        dedup_repeat_gap=args.dedup_repeat_gap,
        dedup_bigram_gap=args.dedup_bigram_gap,
        back_overlap_tol=args.dedup_back_overlap,
        cov_min_overlap_sec=args.cov_min_overlap_sec,
        cov_min_cover_ratio=args.cov_min_cover_ratio,
        srt_gap_break=args.srt_gap_break,
        srt_max_line=args.srt_max_line,
        repair_enable=args.repair_enable,
    )
    agg_manager: Optional[FastAggregatorManager] = None
    if agg_config.enabled:
        if FAST_COMPONENTS is None:
            logger.warning("FAST aggregator modules unavailable; disabling aggregation.")
            agg_config.enabled = False
        else:
            agg_manager = FastAggregatorManager(
                FAST_COMPONENTS,
                agg_config,                agg_config.track_limit,
                getattr(args, "agg_server_host", "127.0.0.1"),
                getattr(args, "agg_server_port", 8899),
                _bool_flag(getattr(args, "agg_server_enable", "true"), True),
                Path("outputs") / "edits.jsonl",
            )

    dump_audio = _bool_flag(args.dump_separated_audio, True)
    dump_variants = [v.lower() for v in (args.dump_variants or ["post"])]

    if args.mode == "asr_only":
        if args.stream:
            raise ValueError("Streaming mode is incompatible with --mode asr_only.")

        _, _, asr, _ = init_pipeline_modules(
            load_separator=False,
            load_identifier=False,
            load_asr=True,
            id_backend=getattr(args, 'id_backend', 'none'),
        )
        asr_output = Path(args.out) if args.out else None

        target = args.asr_target
        if target == "mix":
            if not args.wav:
                raise ValueError("ASR-only mix mode requires --wav PATH.")
            wav_path = Path(args.wav)
            if not wav_path.exists():
                raise FileNotFoundError(f"WAV path does not exist: {wav_path}")
            out_path = asr_output or Path("outputs") / "asr_only_mix.json"
            run_asr_only_mix(wav_path, out_path, asr, asr_config, asr_norm)
        elif target.startswith("files:"):
            pattern = target.split(":", 1)[1]
            out_path = asr_output or Path("outputs") / "asr_only_files.jsonl"
            run_asr_only_files(pattern, out_path, asr, asr_config, asr_norm)
        else:
            raise ValueError(f"Unsupported --asr-target: {target}")
        return

    out_path = Path(args.out) if args.out else Path("outputs") / "v2_windows.jsonl"

    try:
        if args.stream:
            run_pipeline_stream_v2(
                chunk_secs=args.win_len,
                rate=args.rate,
                channels=args.channels,
                frames_per_buffer=args.frames_per_buffer,
                max_workers=args.workers,
                record_secs=args.record_secs,
                queue_out=None,
                stop_event=None,
                in_bytes_queue=None,
                out_path=out_path,
                debug_audio=args.debug_audio,
                mode=args.mode,
                sep_thresholds=sep_thresholds,
                sep_dsp=sep_dsp,
                asr_gate=asr_gate,
                asr_norm=asr_norm,
                enable_asr=args.enable_asr,
                debug=args.debug,
                asr_config=asr_config,
                dump_audio=dump_audio,
                dump_variants=dump_variants,
                dump_root=None,                agg_manager=agg_manager,
            )
            return

        if not args.wav:
            raise ValueError("Offline mode requires --wav PATH.")

        wav_path = Path(args.wav)
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV path does not exist: {wav_path}")

        dump_root = None
        if dump_audio:
            dump_root = Path("outputs") / "separated" / wav_path.stem

        run_file_pipeline(
            wav_path=wav_path,
            out_path=out_path,
            win_len=args.win_len,
            stride=args.stride,
            mode=args.mode,
            sep_thresholds=sep_thresholds,
            sep_dsp=sep_dsp,
            asr_gate=asr_gate,
            asr_norm=asr_norm,
            enable_asr=args.enable_asr,
            debug=args.debug,
            debug_audio=args.debug_audio,
            asr_config=asr_config,
            dump_audio=dump_audio,
            dump_variants=dump_variants,
            dump_root=dump_root,
            agg_manager=agg_manager,
            id_backend=getattr(args, 'id_backend', 'auto'),
        )
        logger.info("Processing complete. Results appended to %s", out_path)
    finally:
        if agg_manager:
            agg_manager.shutdown()


if __name__ == "__main__":
    main()
