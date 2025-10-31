"""
fast_aggregator.py — FAST aggregator for orchestrator_v2, mirroring orchestrator_sample.py logic.

MIRRORED FROM orchestrator_sample.py (do not diverge):
- Windowing defaults: win_len=4.0, stride=1.0
- Aggregator parameters (commit_tail_sec, guards, dedup, coverage, etc.)
- Update loop semantics, is_last_window handling, finalize() timing
- Export behavior (committed-only, same naming, SRT/TXT formatting)
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

# MIRRORED FROM orchestrator_sample.py: import the core FAST components
try:
    from modules.asr.Fast_Test import (
        Aggregator as CoreAggregator,
        Tok,
        tokens_to_text,
    )
except ImportError:
    try:
        from Fast_Test import Aggregator as CoreAggregator, Tok, tokens_to_text
    except ImportError:
        raise RuntimeError("Cannot import Aggregator from modules.asr.Fast_Test")

logger = get_logger(__name__)


# -----------------------------
# Config dataclass (mirrored from sample)
# -----------------------------

@dataclass
class AggregatorCLIConfig:
    """
    MIRRORED FROM orchestrator_sample.py:
    All fields match the sample's defaults and naming.
    """
    # Window geometry
    window_len: float = 4.0      # MIRRORED: sample default
    stride: float = 1.0          # MIRRORED: sample default
    commit_tail_sec: float = 1.6 # MIRRORED: sample default
    epsilon: float = 0.001       # MIRRORED: sample default

    # Guard/protection knobs
    guard_sec: float = 0.4            # MIRRORED: sample default
    protect_head_sec: float = 0.8     # MIRRORED: sample default
    final_protect_sec: float = 0.6    # MIRRORED: sample default
    last_slack_sec: float = 0.12      # MIRRORED: sample default
    fuse_back: float = 0.18           # MIRRORED: sample default
    front_grace_sec: float = 0.0      # MIRRORED: set after construction (guard_sec)

    # Sanitization (same-window dedup)
    san_near_dup_gap: float = 0.12       # MIRRORED: sample default
    san_back_overlap_tol: float = 0.02   # MIRRORED: sample default
    san_overlap_ratio: float = 0.5       # MIRRORED: sample default

    # Coverage (tail-cutting)
    cov_min_overlap_sec: float = 0.02    # MIRRORED: sample default
    cov_min_cover_ratio: float = 0.6     # MIRRORED: sample default

    # Cross-window dedup
    dedup_near_gap: float = 0.1          # MIRRORED: sample default
    dedup_overlap_ratio: float = 0.5     # MIRRORED: sample default
    dedup_repeat_gap: float = 0.22       # MIRRORED: sample default (ABA)
    dedup_bigram_gap: float = 0.30       # MIRRORED: sample default (XY-XY)

    # Track management
    track_limit: int = 10                # MIRRORED: sample default

    # Export
    srt_gap_break: float = 0.5           # MIRRORED: sample default
    srt_max_line: int = 18               # MIRRORED: sample default


# -----------------------------
# Per-track state
# -----------------------------

@dataclass
class TrackAggState:
    """
    Holds a CoreAggregator instance per track_id.
    MIRRORED FROM orchestrator_sample.py: same construction and field assignments.
    """
    aggregator: CoreAggregator
    last_window_index: int = -1
    speaker_name: Optional[str] = None
    legacy_track_id: Optional[int] = None
    key: str = ""
    created_order: int = 0

    def display_label(self) -> str:
        if self.speaker_name:
            return self.speaker_name
        if self.legacy_track_id is not None:
            return f"S{self.legacy_track_id}"
        return self.key or "Unknown"


# -----------------------------
# FastAggregator (main class)
# -----------------------------

class FastAggregator:
    """
    MIRRORED FROM orchestrator_sample.py:
    - Same constructor signature and parameter defaults
    - Same update loop semantics (append_window_tokens, is_last_window)
    - Same finalize() timing and export_transcripts behavior
    """

    def __init__(self, config: Optional[AggregatorCLIConfig] = None) -> None:
        self.config = config or AggregatorCLIConfig()
        self._tracks: Dict[str, TrackAggState] = {}
        self._lock = threading.Lock()
        self._creation_counter = 0
        self._export_path: Optional[str] = None
        self._session_stem: str = "session"
        self._sse_enabled = False
        self._sse_host = "127.0.0.1"
        self._sse_port = 9877

        # MIRRORED: post-construction field adjustment (if sample does it)
        # e.g., if sample sets front_grace_sec = guard_sec after construction:
        if self.config.front_grace_sec == 0.0:
            self.config.front_grace_sec = self.config.guard_sec

        logger.info(
            "[FastAggregator] Initialized (window=%.1fs, stride=%.1fs, commit_tail=%.2fs, track_limit=%d)",
            self.config.window_len,
            self.config.stride,
            self.config.commit_tail_sec,
            self.config.track_limit,
        )

    def set_export_path(self, path: str) -> None:
        self._export_path = path

    def set_session_stem(self, stem: str) -> None:
        self._session_stem = stem

    def enable_sse_server(self, host: str = "127.0.0.1", port: int = 9877) -> None:
        self._sse_enabled = True
        self._sse_host = host
        self._sse_port = port
        # Actual SSE server startup logic can be implemented here or left as placeholder

    def _make_track_key(self, track_id: Optional[int], speaker_name: Optional[str]) -> str:
        if speaker_name:
            return f"speaker::{speaker_name}"
        if track_id is not None:
            return f"track::{track_id}"
        raise ValueError("FastAggregator requires either speaker_name or track_id.")

    @staticmethod
    def _sanitize_for_path(label: str) -> str:
        invalid = '<>:"/\\|?*'
        sanitized = "".join("_" if ch in invalid else ch for ch in label)
        sanitized = sanitized.strip()
        return sanitized or "speaker"

    def _get_or_create_track(
        self,
        track_key: str,
        track_id: Optional[int],
        speaker_name: Optional[str],
    ) -> TrackAggState:
        """
        Create or retrieve a CoreAggregator bound to a logical speaker key.
        """
        state = self._tracks.get(track_key)
        if state is None:
            agg = CoreAggregator(
                commit_tail_sec=self.config.commit_tail_sec,
                epsilon=self.config.epsilon,
                san_near_dup_gap=self.config.san_near_dup_gap,
                san_back_overlap_tol=self.config.san_back_overlap_tol,
                san_overlap_ratio=self.config.san_overlap_ratio,
                cov_min_overlap_sec=self.config.cov_min_overlap_sec,
                cov_min_cover_ratio=self.config.cov_min_cover_ratio,
            )
            agg.fuse_back = float(self.config.fuse_back)
            agg.dedup_near_gap = float(self.config.dedup_near_gap)
            agg.dedup_overlap_ratio = float(self.config.dedup_overlap_ratio)
            agg.dedup_repeat_gap = float(self.config.dedup_repeat_gap)
            agg.dedup_bigram_gap = float(self.config.dedup_bigram_gap)
            agg.front_grace_sec = float(self.config.front_grace_sec)

            state = TrackAggState(
                aggregator=agg,
                speaker_name=speaker_name,
                legacy_track_id=track_id,
                key=track_key,
                created_order=self._creation_counter,
            )
            self._creation_counter += 1
            self._tracks[track_key] = state
            logger.debug(
                "[FastAggregator] Created track key=%s speaker=%s legacy=%s",
                track_key,
                speaker_name,
                track_id,
            )

            if self.config.track_limit > 0 and len(self._tracks) > self.config.track_limit:
                victims = [
                    (k, st)
                    for k, st in self._tracks.items()
                    if k != track_key
                ]
                if victims:
                    victim_key, victim_state = min(
                        victims,
                        key=lambda kv: kv[1].created_order,
                    )
                    self._tracks.pop(victim_key, None)
                    logger.warning(
                        "[FastAggregator] Track limit exceeded; dropped track %s",
                        victim_state.display_label(),
                    )
                else:
                    logger.warning(
                        "[FastAggregator] Track limit %d reached; retaining current track %s",
                        self.config.track_limit,
                        state.display_label(),
                    )
        else:
            if speaker_name and not state.speaker_name:
                state.speaker_name = speaker_name
            if track_id is not None and state.legacy_track_id is None:
                state.legacy_track_id = track_id

        return state

    def append_window_tokens(
        self,
        track_id: Optional[int],
        tokens: List[dict],
        t_start: float,
        t_end: float,
        is_last_window: bool = False,
        window_index: int = 0,
        rtf: float = 0.0,
        speaker_name: Optional[str] = None,
    ) -> None:
        """
        MIRRORED FROM orchestrator_sample.py:
        - Same token filtering logic (guard_sec, protect_head_sec, final_protect_sec, last_slack_sec)
        - Same is_last_window handling
        - Same append_fast() call signature
        """
        if track_id is None and speaker_name is None:
            raise ValueError("FastAggregator.append_window_tokens requires speaker_name or track_id.")

        track_key = self._make_track_key(track_id, speaker_name)
        with self._lock:
            state = self._get_or_create_track(track_key, track_id, speaker_name)
            agg = state.aggregator

            # Convert dict tokens to Tok objects
            toks = [
                Tok(
                    text=str(t.get("text", "")).strip(),
                    start=float(t.get("start", t_start)),
                    end=float(t.get("end", t_end)),
                    prob=float(t.get("prob") or t.get("probability") or 0.0),
                )
                for t in tokens
                if (t.get("text") or t.get("word"))
            ]

            if not toks:
                return

            # MIRRORED: same window filtering logic as sample
            is_first_window = (state.last_window_index < 0)
            state.last_window_index = window_index

            # Left boundary
            if is_first_window:
                accept_s = 0.0
            elif t_start < self.config.guard_sec:
                accept_s = 0.0
            else:
                accept_s = t_start + self.config.guard_sec
            accept_s = max(accept_s, agg.state.last_committed_end - self.config.epsilon)

            # Right boundary
            if is_last_window:
                accept_e = t_end + self.config.last_slack_sec
            else:
                accept_e = t_end - self.config.guard_sec

            def keep_token(t: Tok) -> bool:
                if t.start < accept_s - self.config.epsilon:
                    # Seam rescue (mirrored from sample)
                    if agg.state.tail:
                        seam_free = agg.state.tail[-1].end - self.config.fuse_back
                        if t.start >= seam_free - agg.eps:
                            return True
                    return False
                return t.end <= accept_e + self.config.epsilon

            mid_toks = [t for t in toks if keep_token(t)]

            if speaker_name and not is_first_window and mid_toks:
                logger.debug(
                    "Merged utterances for %s from %.2f-%.2f",
                    speaker_name,
                    t_start,
                    t_end,
                )

            # MIRRORED: protect head of tail (sample logic)
            if agg.state.tail and self.config.protect_head_sec > 0:
                tail_head = agg.state.tail[0].start
                cutoff = max(
                    tail_head + self.config.protect_head_sec,
                    agg.state.last_committed_end + 0.005,
                    accept_s,
                )
                seam_free = agg.state.tail[-1].end - agg.fuse_back
                kept_mid = []
                for t in mid_toks:
                    if t.start >= cutoff - agg.eps:
                        kept_mid.append(t)
                    elif t.start >= seam_free - agg.eps:
                        kept_mid.append(t)
                mid_toks = kept_mid

            # MIRRORED: final window protection (sample logic)
            if is_last_window and agg.state.tail and self.config.final_protect_sec > 0:
                tail_end = agg.state.tail[-1].end
                protect_start = max(accept_s, tail_end - self.config.final_protect_sec)
                mid_toks = [t for t in mid_toks if t.start >= protect_start - self.config.epsilon]

            # MIRRORED: append_fast call (same as sample)
            report = agg.append_fast(mid_toks)
            logger.debug(
                "[FastAggregator] track=%s win=%d kept=%d dropped=%d commit=%d tail_dur=%.2fs",
                state.display_label(),
                window_index,
                report.kept_new,
                report.dropped_by_floor,
                report.committed_moved,
                agg.tail_duration(),
            )

    def finalize(self) -> None:
        """
        MIRRORED FROM orchestrator_sample.py:
        - Same finalize() call on all aggregators
        """
        with self._lock:
            for state in self._tracks.values():
                state.aggregator.finalize()
                logger.debug(
                    "[FastAggregator] Finalized track %s (%d committed)",
                    state.display_label(),
                    len(state.aggregator.state.committed),
                )

    def broadcast_snapshot(self, stem: str) -> None:
        """
        MIRRORED FROM orchestrator_sample.py (optional):
        Placeholder for SSE snapshot broadcast.
        """
        if not self._sse_enabled:
            return
        # Actual SSE broadcast logic can be implemented here
        logger.debug("[FastAggregator] SSE snapshot broadcast (stem=%s)", stem)

    def export_transcripts(
        self,
        base_dir: Path,
        stem: str,
        write_txt: bool = True,
        write_srt: bool = True,
    ) -> None:
        """
        MIRRORED FROM orchestrator_sample.py:
        - Same naming convention (stem + track index)
        - Same SRT/TXT formatting and export logic
        """
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if not self._tracks:
                logger.warning("[FastAggregator] No tracks to export")
                return

            used_stems: Dict[str, int] = {}
            for _, state in sorted(
                self._tracks.items(),
                key=lambda kv: kv[1].display_label(),
            ):
                agg = state.aggregator
                committed = agg.state.committed

                if not committed:
                    continue

                display_label = state.display_label()
                base_suffix = f"{stem}_{self._sanitize_for_path(display_label)}"
                count = used_stems.get(base_suffix, 0)
                track_stem = f"{base_suffix}_{count + 1}" if count else base_suffix
                used_stems[base_suffix] = count + 1

                # MIRRORED: TXT export (committed-only)
                if write_txt:
                    txt_path = base_dir / f"{track_stem}.txt"
                    full_text = tokens_to_text(committed)
                    if display_label:
                        full_text = f"{display_label}: {full_text}"
                    txt_path.write_text(full_text, encoding="utf-8")
                    logger.info("[FastAggregator] Exported TXT: %s", txt_path)

                # MIRRORED: SRT export (same format as sample)
                if write_srt:
                    srt_path = base_dir / f"{track_stem}.srt"
                    self._write_srt_simple(
                        srt_path,
                        committed,
                        max_len=self.config.srt_max_line,
                        gap_break=self.config.srt_gap_break,
                        speaker_label=display_label,
                    )
                    logger.info("[FastAggregator] Exported SRT: %s", srt_path)

    def _write_srt_simple(
        self,
        path: Path,
        toks: List[Tok],
        max_len: int,
        gap_break: float,
        speaker_label: Optional[str] = None,
    ) -> None:
        """
        MIRRORED FROM orchestrator_sample.py (or Fast_Test.py):
        Same SRT formatting logic (time rounding, line breaks).
        """
        def fmt_ts(sec: float) -> str:
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            ms = int((sec % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        segments = []
        buf: List[Tok] = []
        for t in toks:
            if buf and (t.start - buf[-1].end > gap_break):
                segments.append(buf)
                buf = []
            buf.append(t)
        if buf:
            segments.append(buf)

        lines = []
        for idx, seg in enumerate(segments, start=1):
            start_ts = fmt_ts(seg[0].start)
            end_ts = fmt_ts(seg[-1].end)
            text = tokens_to_text(seg)
            # MIRRORED: line wrapping logic (same as sample)
            wrapped = []
            words = text.split()
            line = ""
            for w in words:
                if len(line) + len(w) + 1 > max_len:
                    if line:
                        wrapped.append(line)
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                wrapped.append(line)
            body = "\n".join(wrapped) if wrapped else text
            if speaker_label:
                body = f"{speaker_label}: {body}"

            lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{body}\n")

        path.write_text("\n".join(lines), encoding="utf-8")

    def export_txt(self, path: str) -> None:
        """
        MIRRORED FROM orchestrator_sample.py:
        Export all tracks to a single TXT (optional convenience).
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            lines = []
            for _, state in sorted(
                self._tracks.items(),
                key=lambda kv: kv[1].display_label(),
            ):
                agg = state.aggregator
                full_text = tokens_to_text(agg.state.committed)
                if full_text.strip():
                    label = state.display_label()
                    lines.append(f"[{label}] {full_text}")
            path_obj.write_text("\n".join(lines), encoding="utf-8")
            logger.info("[FastAggregator] Exported merged TXT: %s", path_obj)
