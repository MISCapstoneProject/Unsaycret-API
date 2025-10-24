import argparse
import glob
import json
import logging
import math
import os
import queue
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from scipy.signal import resample_poly

try:
    import pyaudio  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyaudio = None

# Align numeric behaviour with v1 orchestrator.
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
    log_prob_threshold: float = -1.0
    suppress_tokens: str = "-1"

def init_pipeline_modules(
    load_separator: bool = True,
    load_identifier: bool = True,
    load_asr: bool = True,
) -> Tuple[Optional[AudioSeparator], Optional[SpeakerIdentifier], Optional[WhisperASR], bool]:
    """
    Initialise the heavy pipeline modules while respecting the v1 GPU/CPU policy.

    Args:
        load_separator: Instantiate the separation module if ``True``.
        load_identifier: Instantiate the speaker identification module if ``True``.
        load_asr: Instantiate the ASR wrapper if ``True``.

    Returns:
        Tuple of (separator, identifier, asr, use_gpu_flag).
        The module entries will be ``None`` when their respective ``load_*`` flag is ``False``.
    """
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

    separator = AudioSeparator() if load_separator else None
    identifier = SpeakerIdentifier() if load_identifier else None
    asr = (
        WhisperASR(
            model_name=DEFAULT_WHISPER_MODEL,
            gpu=use_gpu,
            beam=DEFAULT_WHISPER_BEAM_SIZE,
        )
        if load_asr
        else None
    )
    return separator, identifier, asr, use_gpu


def generate_sliding_windows(
    wav: np.ndarray,
    sr: int,
    win_len: float = 4.0,
    stride: float = 1.0,
) -> Iterable[Tuple[int, float, float, np.ndarray]]:
    if wav.ndim != 1:
        raise ValueError("Sliding window expects a mono waveform (1-D).")

    win_size = int(win_len * sr)
    hop = int(stride * sr)
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
        yield idx, start / sr, end / sr, chunk
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
    rms = float(math.sqrt(np.mean(np.square(audio)) + 1e-12))
    peak = float(np.max(np.abs(audio)) if audio.size else 0.0)
    rms_db = 20.0 * math.log10(max(rms, 1e-9))
    peak_db = 20.0 * math.log10(max(peak, 1e-9))

    frame_len = max(1, int(0.02 * sr))
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

class WindowOrchestrator:
    """
    Sliding-window orchestrator that reuses the v1 modules while offering new modes.
    """

    def __init__(
        self,
        separator: Optional[AudioSeparator],
        identifier: Optional[SpeakerIdentifier],
        asr: Optional[WhisperASR],
        mode: str,
        win_len: float = 4.0,
        stride: float = 1.0,
        separation_thresholds: SeparationThresholds = SeparationThresholds(0.30, -45.0, 0.50),
        asr_gate: AsrGateConfig = AsrGateConfig(0.60, -45.0),
        enable_asr: bool = True,
        debug: bool = False,
        debug_audio: bool = False,
        asr_config: AsrRuntimeConfig = AsrRuntimeConfig(),
    ) -> None:
        if separator is None:
            raise ValueError("AudioSeparator instance is required for sliding-window processing.")
        if identifier is None:
            raise ValueError("SpeakerIdentifier instance is required for sliding-window processing.")

        self.sep = separator
        self.identifier = identifier
        self.asr = asr
        self.mode = mode
        self.win_len = win_len
        self.stride = stride
        self.sep_thresholds = separation_thresholds
        self.asr_gate = asr_gate
        self.enable_asr = enable_asr and (mode == "pipeline")
        self.debug = debug
        self.debug_audio = debug_audio
        self.asr_config = asr_config

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
            "log_prob_threshold": asr_config.log_prob_threshold,
            "suppress_tokens": asr_config.suppress_tokens,
        }

    def process_window(
        self,
        window_idx: int,
        t_start: float,
        t_end: float,
        chunk: np.ndarray,
        chunk_sr: int = TARGET_RATE,
    ) -> Dict[str, Any]:
        if self.mode == "asr_only":
            raise RuntimeError("process_window is not applicable in ASR-only mode.")

        if chunk.size == 0:
            logger.warning(
                "Window %03d [%.2f, %.2f] received empty audio chunk.",
                window_idx,
                t_start,
                t_end,
            )
            return self._empty_result(window_idx, t_start, t_end)

        chunk = np.asarray(chunk, dtype=np.float32)
        energy = float(np.mean(np.abs(chunk)))
        if energy < MIN_ENERGY_THRESHOLD:
            logger.warning(
                "Window %03d [%.2f, %.2f] low energy (%.4f); skipping.",
                window_idx,
                t_start,
                t_end,
                energy,
            )
            return self._empty_result(window_idx, t_start, t_end)

        if chunk_sr != TARGET_RATE:
            chunk = resample_poly(chunk, TARGET_RATE, chunk_sr).astype(np.float32)
            chunk_sr = TARGET_RATE

        tensor_input = torch.from_numpy(chunk).unsqueeze(0).to(self.sep.device)

        with torch.no_grad():
            detected = self.sep.spk_counter.count_with_refine(
                audio=tensor_input,
                sample_rate=TARGET_RATE,
                expected_min=1,
                expected_max=3,
                first_pass_range=(1, 3),
                allow_zero=True,
                debug=False,
            )

        if detected <= 0:
            with torch.no_grad():
                detected = self.sep.spk_counter.count_with_refine(
                    audio=tensor_input,
                    sample_rate=TARGET_RATE,
                    expected_min=1,
                    expected_max=3,
                    first_pass_range=(1, 3),
                    allow_zero=False,
                    debug=False,
                )

        if detected <= 0:
            logger.info(
                "Window %03d [%.2f, %.2f] contains no active speakers.",
                window_idx,
                t_start,
                t_end,
            )
            return self._empty_result(window_idx, t_start, t_end)
        separated_sources, total_sources = self._separate_sources(
            tensor_input, detected, window_idx
        )
        if not separated_sources:
            logger.warning(
                "Window %03d [%.2f, %.2f] separation returned no sources.",
                window_idx,
                t_start,
                t_end,
            )
            return self._empty_result(window_idx, t_start, t_end, num_sources=0)

        source_infos: List[Dict[str, Any]] = []
        for idx, audio in enumerate(separated_sources):
            metrics = compute_audio_metrics(audio, TARGET_RATE)
            dropped = should_drop_source(metrics, self.sep_thresholds)
            source_infos.append(
                {
                    "index": idx,
                    "audio": audio,
                    "metrics": metrics,
                    "dropped": dropped,
                }
            )

        kept = [info for info in source_infos if not info["dropped"]]
        dropped = [info for info in source_infos if info["dropped"]]

        if self.debug:
            logger.debug(
                "Window %03d metrics: %s",
                window_idx,
                [
                    {
                        "idx": info["index"],
                        "rms_db": round(info["metrics"]["rms_db"], 2),
                        "voiced_ratio": round(info["metrics"]["voiced_ratio"], 3),
                        "duration_s": round(info["metrics"]["duration_s"], 3),
                        "dropped": info["dropped"],
                    }
                    for info in source_infos
                ],
            )

        embeddings: List[np.ndarray] = []
        for info in kept:
            try:
                emb_audio = self._select_embedding_region(info["audio"], TARGET_RATE)
                embedding = self.identifier.audio_processor.extract_embedding_from_stream(
                    emb_audio, TARGET_RATE
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Window %03d embedding extraction failed for source %d: %s",
                    window_idx,
                    info["index"],
                    exc,
                )
                embedding = np.zeros(192, dtype=np.float32)
            embeddings.append(embedding)
            info["embedding"] = embedding

        assignment: Dict[int, int] = {}
        if embeddings:
            assignment = self.matcher.assign(embeddings)

        cost_matrix = self.matcher.last_debug.get("cost_matrix")
        if self.debug and cost_matrix is not None:
            logger.debug("Window %03d cost matrix:\n%s", window_idx, cost_matrix)
            logger.debug("Window %03d assignment: %s", window_idx, assignment)

        track_entries: Dict[int, Dict[str, Any]] = {0: None, 1: None}  # type: ignore

        for src_idx, track_id in assignment.items():
            info = kept[src_idx]
            metrics = info["metrics"]
            track_entry = self._base_track_entry(track_id, metrics)

            if self.mode == "sep_only":
                track_entry["skip_reason"] = "sep_only_mode"
            elif not self.enable_asr:
                track_entry["skip_reason"] = "asr_disabled"
            else:
                if not self.asr:
                    logger.error("ASR module is unavailable; skipping ASR inference.")
                    track_entry["skip_reason"] = "asr_module_missing"
                elif not passes_asr_gate(metrics, self.asr_gate):
                    track_entry["skip_reason"] = "asr_gated_low_voice"
                else:
                    segments = self._run_asr_on_source(info["audio"], t_start, t_end)
                    track_entry["asr_segments"] = segments
                    track_entry["segments"] = len(segments)
                    track_entry["avg_conf"] = average_confidence(segments)
                    if self.debug:
                        logger.debug(
                            "Window %03d track %d ASR segments=%d avg_conf=%s",
                            window_idx,
                            track_id,
                            track_entry["segments"],
                            track_entry["avg_conf"],
                        )
            track_entries[track_id] = track_entry

        remaining_track_ids = [tid for tid in (0, 1) if track_entries[tid] is None]
        for info, track_id in zip(dropped, remaining_track_ids):
            metrics = info["metrics"]
            entry = self._base_track_entry(track_id, metrics)
            entry["skip_reason"] = "silent_source_dropped"
            track_entries[track_id] = entry

        for track_id in (0, 1):
            if track_entries[track_id] is None:
                track_entries[track_id] = {
                    "track_id": track_id,
                    "rms_db": None,
                    "voiced_ratio": None,
                    "duration_s": None,
                    "peak_db": None,
                    "segments": 0,
                    "avg_conf": None,
                    "asr_segments": [],
                    "skip_reason": "no_source_detected",
                }

        tracks_payload = sorted(track_entries.values(), key=lambda item: item["track_id"])
        num_after_drop = sum(1 for entry in tracks_payload if entry.get("skip_reason") != "silent_source_dropped")

        logger.info(
            "Window %03d [%.2f, %.2f] -> detected=%d separated=%d kept=%d",
            window_idx,
            t_start,
            t_end,
            detected,
            len(source_infos),
            len(kept),
        )

        return {
            "window_index": window_idx,
            "t_start": round(float(t_start), 3),
            "t_end": round(float(t_end), 3),
            "num_sources_before_drop": len(source_infos),
            "num_sources_after_drop": len(kept),
            "num_sources": num_after_drop,
            "tracks": tracks_payload,
        }
    def _base_track_entry(self, track_id: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        return {
            "track_id": track_id,
            "rms_db": round(float(metrics["rms_db"]), 3),
            "voiced_ratio": round(float(metrics["voiced_ratio"]), 3),
            "duration_s": round(float(metrics["duration_s"]), 3),
            "peak_db": round(float(metrics["peak_db"]), 3),
            "segments": 0,
            "avg_conf": None,
            "asr_segments": [],
        }

    def _empty_result(
        self,
        window_idx: int,
        t_start: float,
        t_end: float,
        num_sources: int = 0,
    ) -> Dict[str, Any]:
        return {
            "window_index": window_idx,
            "t_start": round(float(t_start), 3),
            "t_end": round(float(t_end), 3),
            "num_sources_before_drop": num_sources,
            "num_sources_after_drop": 0,
            "num_sources": 0,
            "tracks": [
                {
                    "track_id": 0,
                    "rms_db": None,
                    "voiced_ratio": None,
                    "duration_s": None,
                    "peak_db": None,
                    "segments": 0,
                    "avg_conf": None,
                    "asr_segments": [],
                    "skip_reason": "no_source_detected",
                },
                {
                    "track_id": 1,
                    "rms_db": None,
                    "voiced_ratio": None,
                    "duration_s": None,
                    "peak_db": None,
                    "segments": 0,
                    "avg_conf": None,
                    "asr_segments": [],
                    "skip_reason": "no_source_detected",
                },
            ],
        }

    def _separate_sources(
        self,
        tensor_input: torch.Tensor,
        detected: int,
        window_idx: int,
    ) -> Tuple[List[np.ndarray], int]:
        with torch.no_grad():
            model, _ = self.sep._get_appropriate_model(int(max(1, detected)))
            prepared = tensor_input
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
        separated = [
            est_st[idx].detach().cpu().numpy().astype(np.float32) for idx in top_indices
        ]

        return separated, total_sources

    def _select_embedding_region(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if audio.size == 0:
            return audio

        target = min(audio.shape[0], max(int(sr), int(2 * sr)))
        if audio.shape[0] <= target:
            return audio

        frame = max(1, int(0.25 * sr))
        trimmed = (audio.shape[0] // frame) * frame
        if trimmed == 0:
            return audio[-target:]

        frames = audio[:trimmed].reshape(-1, frame)
        energy = np.mean(np.square(frames), axis=1)
        best_idx = int(np.argmax(energy))
        start = min(best_idx * frame, audio.shape[0] - target)
        start = max(0, start)
        return audio[start : start + target]

    def _run_asr_on_source(
        self,
        audio: np.ndarray,
        window_start: float,
        window_end: float,
    ) -> List[Dict[str, Any]]:
        if not self.asr or audio.size == 0:
            return []

        tensor = torch.from_numpy(audio).unsqueeze(0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torchaudio.save(tmp_path, tensor, TARGET_RATE)
            text, avg_conf, words = self.asr.transcribe(tmp_path, **self._asr_kwargs)
            return _blend_words_to_segment(words, window_start, window_end, text, avg_conf)
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:  # pragma: no cover - defensive cleanup
                pass

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

def run_file_pipeline(
    wav_path: Path,
    out_path: Path,
    win_len: float,
    stride: float,
    mode: str,
    separation_thresholds: SeparationThresholds,
    asr_gate: AsrGateConfig,
    enable_asr: bool,
    debug: bool,
    debug_audio: bool,
    asr_config: AsrRuntimeConfig,
) -> None:
    if mode == "asr_only":
        raise RuntimeError("run_file_pipeline cannot be used in ASR-only mode.")

    load_asr = mode == "pipeline" and enable_asr
    sep, identifier, asr, use_gpu = init_pipeline_modules(
        load_separator=True,
        load_identifier=True,
        load_asr=load_asr,
    )

    if asr:
        asr.lang = asr_config.language

    orchestrator = WindowOrchestrator(
        separator=sep,
        identifier=identifier,
        asr=asr,
        mode=mode,
        win_len=win_len,
        stride=stride,
        separation_thresholds=separation_thresholds,
        asr_gate=asr_gate,
        enable_asr=enable_asr,
        debug=debug,
        debug_audio=debug_audio,
        asr_config=asr_config,
    )

    waveform, sr = _prepare_audio(wav_path)
    if sr != TARGET_RATE:
        raise RuntimeError("Audio preparation failed to resample properly.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    windows = list(generate_sliding_windows(waveform, sr, win_len=win_len, stride=stride))
    if not windows:
        logger.warning(
            "Audio shorter than window length (%.2fs); nothing to process.", win_len
        )
        return

    logger.info("Processing %d windows from %s", len(windows), wav_path)

    with out_path.open("a", encoding="utf-8") as handle:
        for idx, t0, t1, chunk in windows:
            result = orchestrator.process_window(idx, t0, t1, chunk, chunk_sr=TARGET_RATE)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    if use_gpu and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive cleanup
            pass

def run_asr_only_mix(
    wav_path: Path,
    out_path: Path,
    asr: WhisperASR,
    asr_config: AsrRuntimeConfig,
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
        "log_prob_threshold": asr_config.log_prob_threshold,
        "suppress_tokens": asr_config.suppress_tokens,
    }
    text, avg_conf, words = asr.transcribe(str(wav_path), **kwargs)
    result = {
        "mode": "asr_only_mix",
        "input": str(wav_path),
        "text": text,
        "avg_conf": avg_conf,
        "words": words,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("ASR-only mix result written to %s", out_path)


def run_asr_only_files(
    pattern: str,
    out_path: Path,
    asr: WhisperASR,
    asr_config: AsrRuntimeConfig,
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
        "log_prob_threshold": asr_config.log_prob_threshold,
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
    separation_thresholds: SeparationThresholds = SeparationThresholds(0.30, -45.0, 0.50),
    asr_gate: AsrGateConfig = AsrGateConfig(0.60, -45.0),
    enable_asr: bool = True,
    debug: bool = False,
    asr_config: AsrRuntimeConfig = AsrRuntimeConfig(),
) -> None:
    if mode == "asr_only":
        raise ValueError("Streaming ASR-only mode is not supported.")

    if sep is None or spk is None or (asr is None and mode == "pipeline" and enable_asr):
        load_asr = mode == "pipeline" and enable_asr
        sep, spk, asr, _ = init_pipeline_modules(
            load_separator=True,
            load_identifier=True,
            load_asr=load_asr,
        )

    if asr:
        asr.lang = asr_config.language

    orchestrator = WindowOrchestrator(
        separator=sep,
        identifier=spk,
        asr=asr,
        mode=mode,
        win_len=chunk_secs,
        stride=1.0,
        separation_thresholds=separation_thresholds,
        asr_gate=asr_gate,
        enable_asr=enable_asr,
        debug=debug,
        debug_audio=debug_audio,
        asr_config=asr_config,
    )

    output_path = out_path or Path("outputs") / "v2_windows.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    taipei_tz = timezone(timedelta(hours=8))
    stream_start = datetime.now(taipei_tz)

    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
    pending: Dict[int, Future] = {}
    next_to_write = 0

    results_queue: "queue.Queue[Tuple[bytes, int, int, int, int]]" = queue.Queue(maxsize=max_workers * 2)
    shared_stop = stop_event or threading.Event()

    ingest_chunks = deque(maxlen=max(1, int(round(chunk_secs / 1.0))))
    out_handle = output_path.open("a", encoding="utf-8")

    ring_sr: Optional[int] = None
    bytes_per_sample_hint: Optional[int] = None

    running = True

    def submit_window(
        idx: int,
        t_start: float,
        t_end: float,
        audio_chunk: np.ndarray,
        src_sr: int,
    ) -> None:
        future = executor.submit(
            orchestrator.process_window,
            idx,
            t_start,
            t_end,
            audio_chunk,
            src_sr,
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
            except Exception as exc:  # pragma: no cover - already logged
                logger.error("Window %03d processing error: %s", next_to_write, exc)
                result = orchestrator._empty_result(
                    next_to_write,
                    next_to_write,
                    next_to_write + chunk_secs,
                )

            out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_handle.flush()

            if queue_out is not None:
                queue_out.put(result)
            del pending[next_to_write]
            next_to_write += 1
    def recorder_from_queue() -> None:
        nonlocal ring_sr, bytes_per_sample_hint
        ch = max(1, int(channels))

        buf = bytearray()
        idx = 0
        start_time = time.time()

        provisional_sr = max(1, int(rate))
        frames_needed = int(provisional_sr * 1.0) * ch

        calibrated = False
        calib_t0 = time.time()
        calib_bytes = 0

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
                    candidates_sr = [48000, 44100, 32000, 24000, 22050, 16000]
                    best_score = float("inf")
                    best_choice = (provisional_sr, 4)
                    for bps in (2, 4):
                        est_sr = calib_bytes / (elapsed * bps * ch)
                        candidate = min(candidates_sr, key=lambda s: abs(s - est_sr))
                        score = abs(candidate - est_sr)
                        if score < best_score:
                            best_score = score
                            best_choice = (int(candidate), bps)
                    ring_sr, bytes_per_sample_hint = best_choice
                    frames_needed = int(ring_sr * 1.0) * bytes_per_sample_hint * ch
                    calibrated = True
                    logger.info(
                        "[calib] est_sr≈%.1fHz -> use %dHz, bytes/sample=%d",
                        calib_bytes / (elapsed * ch * best_choice[1]),
                        ring_sr,
                        bytes_per_sample_hint,
                    )

            if not calibrated:
                continue

            required = int(ring_sr * 1.0) * bytes_per_sample_hint * ch
            while len(buf) >= required:
                raw = bytes(buf[:required])
                del buf[:required]
                results_queue.put((raw, idx, ring_sr, bytes_per_sample_hint, ch))
                idx += 1

    def recorder_from_mic() -> None:
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
        frames_needed = int(rate * 1.0) * bytes_per_sample * channels
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

                while len(buf) >= frames_needed:
                    raw = bytes(buf[:frames_needed])
                    del buf[:frames_needed]
                    results_queue.put((raw, idx, rate, bytes_per_sample, channels))
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

    loop_start = time.time()
    window_idx = 0

    try:
        while running:
            if shared_stop.is_set():
                break

            if record_secs is not None and time.time() - loop_start >= record_secs:
                shared_stop.set()
                break

            try:
                raw, idx, src_sr, bytes_per_sample, ch = results_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            ring_sr = src_sr
            bytes_per_sample_hint = bytes_per_sample

            decoded = _decode_audio_bytes(raw, ch, bytes_per_sample)
            if decoded.size == 0:
                logger.warning("Stride chunk %d empty; skipping.", idx)
                continue

            ingest_chunks.append(decoded)
            if len(ingest_chunks) < ingest_chunks.maxlen:
                continue

            window_audio = np.concatenate(list(ingest_chunks), axis=0)
            t_start = window_idx * 1.0
            t_end = t_start + chunk_secs

            submit_window(window_idx, t_start, t_end, window_audio, ring_sr)
            window_idx += 1
    except KeyboardInterrupt:
        logger.info("Stream interrupted by user; shutting down.")
        shared_stop.set()
    finally:
        shared_stop.set()
        ingest_thread.join(timeout=1.0)
        flush_pending(force=True)
        executor.shutdown(wait=True)
        out_handle.close()
        logger.info(
            "Stream finished. Results appended to %s (started at %s)",
            output_path,
            stream_start.isoformat(),
        )

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

    parser.add_argument("--mode", choices=["pipeline", "asr_only", "sep_only"], default="pipeline", help="Processing mode.")
    parser.add_argument("--asr-target", default="mix", help="ASR-only target: 'mix' or 'files:<glob>'.")

    parser.add_argument("--sep-min-voiced", type=float, default=0.30, help="Minimum voiced ratio before separation drop.")
    parser.add_argument("--sep-min-rms-db", type=float, default=-45.0, help="Minimum RMS dBFS before separation drop.")
    parser.add_argument("--sep-min-duration", type=float, default=0.50, help="Minimum voiced duration before separation drop.")

    parser.add_argument("--enable-asr", dest="enable_asr", action="store_true", help="Enable ASR stage.")
    parser.add_argument("--disable-asr", dest="enable_asr", action="store_false", help="Disable ASR stage.")
    parser.set_defaults(enable_asr=True)

    parser.add_argument("--asr-min-voiced", type=float, default=0.45, help="ASR gating minimum voiced ratio.")
    parser.add_argument("--asr-min-rms-db", type=float, default=-45.0, help="ASR gating minimum RMS dBFS.")
    parser.add_argument("--asr-language", type=str, default="zh", help="ASR language code.")
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
    asr_gate = AsrGateConfig(
        min_voiced=args.asr_min_voiced,
        min_rms_db=args.asr_min_rms_db,
    )
    asr_config = AsrRuntimeConfig(language=args.asr_language)

    if args.mode == "asr_only":
        if args.stream:
            raise ValueError("Streaming mode is incompatible with --mode asr_only.")

        _, _, asr, _ = init_pipeline_modules(
            load_separator=False,
            load_identifier=False,
            load_asr=True,
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
            run_asr_only_mix(wav_path, out_path, asr, asr_config)
        elif target.startswith("files:"):
            pattern = target.split(":", 1)[1]
            out_path = asr_output or Path("outputs") / "asr_only_files.jsonl"
            run_asr_only_files(pattern, out_path, asr, asr_config)
        else:
            raise ValueError(f"Unsupported --asr-target: {target}")
        return

    out_path = Path(args.out) if args.out else Path("outputs") / "v2_windows.jsonl"

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
            separation_thresholds=sep_thresholds,
            asr_gate=asr_gate,
            enable_asr=args.enable_asr,
            debug=args.debug,
            asr_config=asr_config,
        )
        return

    if not args.wav:
        raise ValueError("Offline mode requires --wav PATH.")

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV path does not exist: {wav_path}")

    run_file_pipeline(
        wav_path=wav_path,
        out_path=out_path,
        win_len=args.win_len,
        stride=args.stride,
        mode=args.mode,
        separation_thresholds=sep_thresholds,
        asr_gate=asr_gate,
        enable_asr=args.enable_asr,
        debug=args.debug,
        debug_audio=args.debug_audio,
        asr_config=asr_config,
    )
    logger.info("Processing complete. Results appended to %s", out_path)


if __name__ == "__main__":
    main()
