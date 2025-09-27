
#!/usr/bin/env python3
"""
asr_user.py — User-facing FAST ASR with clean UI (per‑second updates + mic mode).

Highlights
- Minimal, friendly console UI with just two lines: COMMITTED (green) and TAIL (yellow).
- Per‑second dynamic updates (animation) with a simple time axis.
- File mode (streaming simulation) and real microphone mode.
- Optional subtitle export (SRT/TXT) and optional LLM Repair (same flags as test harness).

Dependencies
    pip install faster-whisper rich sounddevice numpy librosa

Assumptions
- Reuses core logic from your test harness for decoding/aggregation to ensure identical behavior.
- Expects you have `Fast_Test.py` available/importable in the working directory or PYTHONPATH.

Usage examples
    # 1) File mode (UI + SRT export)
    python asr_user.py --audio data/sample.wav --language zh --subtitle-format srt --out-dir out

    # 2) Live microphone mode (Ctrl+C to stop, then we finalize + export SRT)
    python asr_user.py --mic --language zh --subtitle-format srt --out-dir out

Notes
- Colors: COMMITTED = green, TAIL = yellow.
- Time axis ticks are in whole seconds; the UI refreshes once per stride (default 1s).
- This is a *quiet* build: no debugging tables, no drop/cut logs — just the live view.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# --- Console UI (colors & live refresh) ---
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich.box import ROUNDED
except Exception as e:
    print("[ERROR] rich is required for the UI. Try: pip install rich", file=sys.stderr)
    raise


# --- Core FAST components (reuse from your test harness for identical behavior) ---
try:
    # Prefer package-relative import when running as `python -m modules.asr.asr_user`
    from .Fast_Test import (
        Aggregator, Tok, tokens_to_text,
        FileStreamSimulator, FastDecoder, create_decoder, TARGET_SR
    )
except Exception:
    try:
        # Fallback if imported from project root
        from modules.asr.Fast_Test import (
            Aggregator, Tok, tokens_to_text,
            FileStreamSimulator, FastDecoder, create_decoder, TARGET_SR
        )
    except Exception:
        # Last resort: same-directory import
        from Fast_Test import (
            Aggregator, Tok, tokens_to_text,
            FileStreamSimulator, FastDecoder, create_decoder, TARGET_SR
        )

# --- Optional LLM repair (same as test harness) ---
try:
    from .llm_repair import llm_repair_pipeline, write_srt_with_repair
    HAS_REPAIR = True
except Exception:
    try:
        from modules.asr.llm_repair import llm_repair_pipeline, write_srt_with_repair
        HAS_REPAIR = True
    except Exception:
        HAS_REPAIR = False

try:
    from modules.asr.llm_repair import llm_repair_pipeline, write_srt_with_repair
    HAS_REPAIR = True
except Exception:
    HAS_REPAIR = False

# --- Subtitle writer (fallback to simple SRT/TXT if you don't enable repair) ---
def _fmt_ts(x: float) -> str:
    h = int(x // 3600); x -= h*3600
    m = int(x // 60);   x -= m*60
    s = int(x);         ms = int((x - s) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt_simple(path: Path, toks: List[Tok], max_len: int = 70, gap_break: float = 0.5) -> None:
    if not toks:
        Path(path).write_text("", encoding="utf-8")
        return
    # group by gap
    segments = []
    cur = []
    for t in toks:
        if not cur:
            cur = [t]; continue
        if (t.start - cur[-1].end) >= gap_break:
            segments.append(cur); cur = [t]
        else:
            cur.append(t)
    if cur: segments.append(cur)

    with open(path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            s = seg[0].start; e = seg[-1].end
            text = tokens_to_text(seg)
            # hard wrap by length
            lines = []
            while len(text) > max_len:
                lines.append(text[:max_len]); text = text[max_len:]
            if text: lines.append(text)
            f.write(f"{idx}\n{_fmt_ts(s)} --> {_fmt_ts(e)}\n")
            f.write("\n".join(lines) + "\n\n")
            idx += 1

# ---------------------------
# Per‑second minimal UI
# ---------------------------
class LiveSubUI:
    def __init__(self, width: int = 100) -> None:
        self.console = Console()
        self.width = width

    def _truncate(self, s: str) -> str:
        if len(s) <= self.width: return s
        keep = max(0, self.width - 3)
        left = keep // 2
        right = keep - left
        return s[:left] + "..." + s[-right:]

    def _time_axis(self, t: float, span: int = 20) -> Text:
        cur = int(math.floor(t + 1e-6))
        start = max(0, cur - span + 1)
        tx = Text("")
        for sec in range(start, cur + 1):
            mark = f"{sec:>2d}s"
            if sec == cur:
                tx.append(f" {mark} ", style="bold cyan")
            else:
                tx.append(f" {mark} ", style="dim")
        return tx

    def render(self, t: float, committed: str, tail: str) -> Panel:
        # 只靠顏色：committed(綠) 直接接 tail(黃)；沒有任何標籤或分隔
        committed_txt = Text(self._truncate(committed), style="green")
        tail_txt = Text(self._truncate(tail), style="yellow")

        body = Text()
        body.append("Time: ", style="bold")
        body.append(self._time_axis(t))
        body.append("\n\n")

        line = Text()
        line.append(committed_txt)  # 綠
        line.append(tail_txt)       # 黃（緊接在後）
        body.append(line)

        return Panel(body, title="view", border_style="cyan", box=ROUNDED)


# ---------------------------
# Token filtering (same logic as your test harness, minus prints)
# ---------------------------
def filter_window_tokens(
    toks: List[Tok],
    agg: Aggregator,
    win_start: float,
    win_end: float,
    guard_sec: float,
    last_slack_sec: float,
    epsilon: float,
    fuse_back: float,
    protect_head_sec: float,
    final_protect_sec: float,
    is_first_window: bool,
    is_last_window: bool,
) -> List[Tok]:
    
    # -------- 新增：min-prob 濾嘴（避免低置信抖動觸發剪尾） --------
    min_prob = float(getattr(agg, "min_prob", 0.0))
    if min_prob > 0.0:
        toks = [t for t in toks if getattr(t, "prob", 1.0) >= min_prob]
    # 1) left boundary (accept_s)
    if is_first_window:
        accept_s = 0.0
    elif win_start < guard_sec:
        accept_s = 0.0
    else:
        accept_s = win_start + guard_sec
    accept_s = max(accept_s, agg.state.last_committed_end - epsilon)

    # 2) right boundary
    base_e = win_end if is_last_window else (win_end - guard_sec)

    def keep_token(t: Tok) -> bool:
        # -------- 改良：左護欄例外「接縫撈救」 --------
        if t.start < accept_s - epsilon:
            # 只要它還貼近前一個尾巴的「可覆蓋區」，就放行
            if agg.state.tail:
                seam_free = agg.state.tail[-1].end - fuse_back
                if t.start >= seam_free - agg.eps:
                    return True
            return False
        # 右邊界（原本程式）
        if is_last_window:
            return (t.end <= (win_end + last_slack_sec) + epsilon)
        else:
            return (t.end <= (win_end - guard_sec) + epsilon)

    mid_toks = [t for t in toks if keep_token(t)]

    # 3) protect head of current tail (to avoid overcut at the seam)
    if agg.state.tail and protect_head_sec > 0:
        tail_head = agg.state.tail[0].start
        cutoff = max(tail_head + protect_head_sec, agg.state.last_committed_end + 0.005, accept_s)
        seam_free = agg.state.tail[-1].end - fuse_back
        kept_mid = []
        for t in mid_toks:
            if t.start >= cutoff - agg.eps:
                kept_mid.append(t)
            else:
                if t.start >= seam_free - agg.eps:
                    kept_mid.append(t)
        mid_toks = kept_mid

    # 4) final window protection (trim new additions to only touch the tail end)
    if is_last_window and agg.state.tail and final_protect_sec > 0:
        tail_end = agg.state.tail[-1].end
        protect_start = max(accept_s, tail_end - final_protect_sec)
        mid_toks = [t for t in mid_toks if t.start >= protect_start - epsilon]

    return mid_toks

# ---------------------------
# Runners
# ---------------------------
def run_file(args) -> None:
    ui = LiveSubUI(width=args.ui_width)
    console = ui.console

    dec = create_decoder(args)
    sim = FileStreamSimulator(Path(args.audio), args.window_len, args.stride, endpad_sec=args.endpad_sec)
    agg = Aggregator(
        commit_tail_sec=args.commit_tail_sec,
        epsilon=args.epsilon,
        san_near_dup_gap=args.san_near_dup_gap,
        san_back_overlap_tol=args.san_back_overlap_tol,
        san_overlap_ratio=args.san_overlap_ratio,
        cov_min_overlap_sec=args.cov_min_overlap_sec,
        cov_min_cover_ratio=args.cov_min_cover_ratio,
    )
    agg.fuse_back = float(args.fuse_back)
    agg.dedup_near_gap = float(args.dedup_near_gap)
    agg.dedup_overlap_ratio = float(args.dedup_overlap_ratio)
    agg.dedup_repeat_gap = float(args.dedup_repeat_gap)
    agg.dedup_bigram_gap = float(args.dedup_bigram_gap)

    sim_time = args.window_len

    with Live(refresh_per_second=max(1, int(1/args.stride)), console=console, transient=False) as live:
        while not sim.done():
            window = sim.tick()
            if window is None:
                break
            abs_start = max(0.0, sim_time - args.window_len)
            toks, rtf = dec.decode_window(window, abs_start)
            # 冷啟動：前 cold-start-sec 期間只允許較高置信 token；太吵就先不輸出
            if args.cold_start_sec > 0 and t < args.cold_start_sec:
                toks = [x for x in toks if getattr(x, "prob", 1.0) >= max(0.45, args.min_prob)]
                # 可選：若幾乎沒語音能量，直接跳過這窗
                if len(toks) == 0:
                    if args.real_time:
                        time.sleep(max(0.0, args.stride))
                    t += args.stride
                    continue

            win_start = max(0.0, sim_time - args.window_len)
            win_end = sim_time
            is_first_window = (win_start <= 1e-6)
            is_last_window = (sim_time >= sim.duration - 1e-3)
            
            MIN_PROB = 0.38   # 可微調 0.35~0.45
            toks = [t for t in toks if getattr(t, "prob", 1.0) >= MIN_PROB]
            
            mid_toks = filter_window_tokens(
                toks, agg, win_start, win_end,
                guard_sec=args.guard_sec, last_slack_sec=args.last_slack_sec,
                epsilon=args.epsilon, fuse_back=agg.fuse_back,
                protect_head_sec=args.protect_head_sec,
                final_protect_sec=args.final_protect_sec,
                is_first_window=is_first_window,
                is_last_window=is_last_window,
            )
            
            if jsonl_fp is not None:
                rec = {
                    "time": float(t),
                    "win_start": float(win_start),
                    "win_end": float(win_end),
                    "tokens": [{"text":tt.text, "start":tt.start, "end":tt.end, "prob":getattr(tt,"prob",None)} for tt in mid_toks]
                }
                import json; jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

            
            
            agg.append_fast(mid_toks)

            panel = ui.render(sim_time, agg.committed_text(), agg.tail_text())
            live.update(panel, refresh=True)
            if args.real_time:
                time.sleep(max(0.0, args.stride))
            sim_time += args.stride

    # finalize & export
    agg.finalize()
    committed = agg.state.committed
    os.makedirs(args.out_dir, exist_ok=True)
    stem = Path(args.audio).stem

    if args.subtitle_format == "srt":
        if args.repair_enable and HAS_REPAIR:
            toks_dict = [{"text": t.text, "start": t.start, "end": t.end, "prob": t.prob} for t in committed]
            res = llm_repair_pipeline(
                tokens=toks_dict,
                provider=args.repair_provider,
                model=args.repair_model,
                host=args.repair_host,
                api_key=args.repair_api_key or os.environ.get("GROQ_API_KEY"),
                opencc_config=args.opencc_config,
                whitelist_path=args.whitelist,
                glossary_path=args.glossary,
                gap_period=args.seed_period_gap,
                gap_comma=args.seed_comma_gap,
            )
            write_srt_with_repair(
                str(Path(args.out_dir) / f"{stem}.srt"),
                toks_dict, res.clean_text,
                gap_break=args.srt_gap_break,
                max_line=args.srt_max_line,
            )
        else:
            write_srt_simple(Path(args.out_dir) / f"{stem}.srt", committed, max_len=args.srt_max_line, gap_break=args.srt_gap_break)
    else:
        txt = tokens_to_text(committed)
        with open(Path(args.out_dir) / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write(txt + "\n")

    ui.console.print(f"[bold green]Saved →[/bold green] {Path(args.out_dir) / (stem + '.' + args.subtitle_format)}")

# ---- Microphone mode ----
class MicCapture:
    """Capture mono audio at TARGET_SR using sounddevice into a ring buffer."""
    def __init__(self, samplerate: int = TARGET_SR, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self.buffer = deque()  # list of numpy arrays
        self.n_samples = 0
        self.running = False
        self._lock = threading.Lock()

        try:
            import sounddevice as sd
        except Exception as e:
            print("[ERROR] sounddevice is required for --mic. Try: pip install sounddevice", file=sys.stderr)
            raise
        self._sd = sd
        self._stream = None

    def start(self):
        def cb(indata, frames, time_info, status):
            if status:
                # Avoid noisy prints; in user build we stay quiet.
                pass
            mono = indata[:, 0].astype(np.float32, copy=False)
            with self._lock:
                self.buffer.append(mono.copy())
                self.n_samples += len(mono)

        self._stream = self._sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=cb,
            blocksize=int(self.samplerate * 0.1),  # 100ms chunks
        )
        self._stream.start()
        self.running = True

    def stop(self):
        self.running = False
        try:
            if self._stream is not None:
                self._stream.stop(); self._stream.close()
        except Exception:
            pass

    def last_window(self, window_len: float) -> np.ndarray:
        """Return the last `window_len` seconds of audio (zero‑padded on the left)."""
        need = int(window_len * self.samplerate)
        with self._lock:
            if not self.buffer:
                return np.zeros(need, dtype=np.float32)
            # Concatenate minimally to get the last `need` samples
            chunks = list(self.buffer)
        arr = np.concatenate(chunks).astype(np.float32, copy=False)
        if len(arr) >= need:
            return arr[-need:]
        pad = np.zeros(need - len(arr), dtype=np.float32)
        return np.concatenate([pad, arr])

def run_mic(args) -> None:
    ui = LiveSubUI(width=args.ui_width)
    console = ui.console
    dec = create_decoder(args)

    agg = Aggregator(
        commit_tail_sec=args.commit_tail_sec,
        epsilon=args.epsilon,
        san_near_dup_gap=args.san_near_dup_gap,
        san_back_overlap_tol=args.san_back_overlap_tol,
        san_overlap_ratio=args.san_overlap_ratio,
        cov_min_overlap_sec=args.cov_min_overlap_sec,
        cov_min_cover_ratio=args.cov_min_cover_ratio,
    )
    agg.fuse_back = float(args.fuse_back)
    agg.dedup_near_gap = float(args.dedup_near_gap)
    agg.dedup_overlap_ratio = float(args.dedup_overlap_ratio)
    agg.dedup_repeat_gap = float(args.dedup_repeat_gap)
    agg.dedup_bigram_gap = float(args.dedup_bigram_gap)
    # 新增：把 CLI 參數掛到 agg，供過濾/剪尾用
    agg.min_prob = float(args.min_prob)
    agg.front_grace_sec = float(args.front_grace_sec)


    mic = MicCapture(samplerate=TARGET_SR, channels=1)
    mic.start()
    ui.console.print("[bold cyan]🎙️ Mic mode[/bold cyan] — speak! Press [bold]Ctrl+C[/bold] to stop.")
    t = args.window_len  # logical time (s), start after first full window
    
    jsonl_fp = None
    if args.dump_jsonl:
        os.makedirs(args.out_dir, exist_ok=True)
        jsonl_fp = open(Path(args.out_dir)/"mic_tokens.jsonl", "w", encoding="utf-8")

    try:
        with Live(refresh_per_second=max(1, int(1/args.stride)), console=console, transient=False) as live:
            while True:
                window = mic.last_window(args.window_len)
                abs_start = max(0.0, t - args.window_len)
                toks, rtf = dec.decode_window(window, abs_start)
                # 冷啟動：前 cold-start-sec 期間只允許較高置信 token；太吵就先不輸出
                if args.cold_start_sec > 0 and t < args.cold_start_sec:
                    toks = [x for x in toks if getattr(x, "prob", 1.0) >= max(0.45, args.min_prob)]
                    # 可選：若幾乎沒語音能量，直接跳過這窗
                    if len(toks) == 0:
                        if args.real_time:
                            time.sleep(max(0.0, args.stride))
                        t += args.stride
                        continue

                win_start = max(0.0, t - args.window_len)
                win_end = t
                is_first_window = (t <= args.window_len + 1e-6)
                is_last_window = False  # live stream; we don't know the end yet
                
                
                MIN_PROB = 0.38   # 可微調 0.35~0.45
                toks = [t for t in toks if getattr(t, "prob", 1.0) >= MIN_PROB]
            
                mid_toks = filter_window_tokens(
                    toks, agg, win_start, win_end,
                    guard_sec=args.guard_sec, last_slack_sec=args.last_slack_sec,
                    epsilon=args.epsilon, fuse_back=agg.fuse_back,
                    protect_head_sec=args.protect_head_sec,
                    final_protect_sec=args.final_protect_sec,
                    is_first_window=is_first_window,
                    is_last_window=is_last_window,
                )
                
                if jsonl_fp is not None:
                    rec = {
                        "time": float(t),
                        "win_start": float(win_start),
                        "win_end": float(win_end),
                        "tokens": [{"text":tt.text, "start":tt.start, "end":tt.end, "prob":getattr(tt,"prob",None)} for tt in mid_toks]
                    }
                    import json; jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

                
                agg.append_fast(mid_toks)

                panel = ui.render(t, agg.committed_text(), agg.tail_text())
                live.update(panel, refresh=True)

                if args.real_time:
                    time.sleep(max(0.0, args.stride))
                t += args.stride
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        if jsonl_fp is not None:
            jsonl_fp.close()

    # finalize & export（原本程式）...
    # 另外：dump 原始 mic
    if args.dump_mic:
        try:
            import soundfile as sf
            with mic._lock:
                raw = np.concatenate(list(mic.buffer)).astype(np.float32) if mic.buffer else np.zeros(1, np.float32)
            os.makedirs(args.out_dir, exist_ok=True)
            sf.write(str(Path(args.out_dir)/"mic_raw.wav"), raw, TARGET_SR)
            ui.console.print(f"[bold cyan]Saved raw mic →[/bold cyan] {Path(args.out_dir)/'mic_raw.wav'}")
        except Exception as e:
            ui.console.print(f"[yellow]WARN[/] dump-mic failed: {e}")


    # finalize & export
    agg.finalize()
    committed = agg.state.committed
    os.makedirs(args.out_dir, exist_ok=True)
    stem = datetime.now().strftime("mic_%Y%m%d_%H%M%S")

    if args.subtitle_format == "srt":
        if args.repair_enable and HAS_REPAIR:
            toks_dict = [{"text": t.text, "start": t.start, "end": t.end, "prob": t.prob} for t in committed]
            res = llm_repair_pipeline(
                tokens=toks_dict,
                provider=args.repair_provider,
                model=args.repair_model,
                host=args.repair_host,
                api_key=args.repair_api_key or os.environ.get("GROQ_API_KEY"),
                opencc_config=args.opencc_config,
                whitelist_path=args.whitelist,
                glossary_path=args.glossary,
                gap_period=args.seed_period_gap,
                gap_comma=args.seed_comma_gap,
            )
            write_srt_with_repair(
                str(Path(args.out_dir) / f"{stem}.srt"),
                toks_dict, res.clean_text,
                gap_break=args.srt_gap_break,
                max_line=args.srt_max_line,
            )
        else:
            write_srt_simple(Path(args.out_dir) / f"{stem}.srt", committed, max_len=args.srt_max_line, gap_break=args.srt_gap_break)
    else:
        txt = tokens_to_text(committed)
        with open(Path(args.out_dir) / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write(txt + "\n")

    ui.console.print(f"[bold green]Saved →[/bold green] {Path(args.out_dir) / (stem + '.' + args.subtitle_format)}")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="User-facing FAST ASR (clean UI + mic)")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--audio", type=str, help="Single audio file (wav/mp3/flac)")
    g.add_argument("--mic", action="store_true", help="Use live microphone")

    # Model
    p.add_argument("--model-name", type=str, default="medium", help="faster-whisper model (e.g., small, medium, large-v3)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    p.add_argument("--compute-type", type=str, default="float16", help="Compute type (e.g., float16, int8_float16, int8)")
    p.add_argument("--language", type=str, default="none", help='Language hint (e.g., zh, en). Use "none" to auto-detect')
    p.add_argument("--beam-size", type=int, default=3, help="Beam size")

    # Windowing
    p.add_argument("--window-len", type=float, default=3.0, help="Window length (s)")
    p.add_argument("--stride", type=float, default=1.0, help="Step per update (s)")
    p.add_argument("--endpad-sec", type=float, default=0.6, help="Extra padding (s) appended to the end of file-mode streams")
    p.add_argument("--commit-tail-sec", type=float, default=1.6, help="Editable tail length (s)")
    p.add_argument("--epsilon", type=float, default=0.001, help="Time epsilon")

    # Guards / protection
    p.add_argument("--guard-sec", type=float, default=0.4, help="Guard band near window edges (s)")
    p.add_argument("--protect-head-sec", type=float, default=0.8, help="Protect early tail tokens (s)")
    p.add_argument("--final-protect-sec", type=float, default=0.6, help="Protect very end in final window (s)")
    p.add_argument("--last-slack-sec", type=float, default=0.12, help="Extra slack for last window (s)")
    p.add_argument("--fuse-back", type=float, default=0.16, help="Pull new window back to cover seam (s)")
    p.add_argument("--min-prob", type=float, default=0.0,
               help="Drop tokens with prob below this before aggregation")
    p.add_argument("--cold-start-sec", type=float, default=0.8,
                help="Suppress low-confidence output during the first X seconds")
    p.add_argument("--front-grace-sec", type=float, default=0.08,
                help="Disable tail-cutting if new window starts within this gap after the old tail (s)")
    p.add_argument("--dump-mic", action="store_true",
                help="Save raw mic audio to out_dir/mic_raw.wav at the end")
    p.add_argument("--dump-jsonl", action="store_true",
                help="Write per-window tokens to out_dir/mic_tokens.jsonl for debugging")


    # Dedup
    p.add_argument("--san-near-dup-gap", type=float, default=0.12, help="Sanitize (same-window) near-dup gap")
    p.add_argument("--san-back-overlap-tol", type=float, default=0.02, help="Sanitize back-overlap tolerance")
    p.add_argument("--san-overlap-ratio", type=float, default=0.5, help="Sanitize overlap ratio")
    p.add_argument("--cov-min-overlap-sec", type=float, default=0.02, help="Minimum coverage overlap (s)")
    p.add_argument("--cov-min-cover-ratio", type=float, default=0.6, help="Minimum coverage ratio")
    p.add_argument("--dedup-near-gap", type=float, default=0.10, help="Cross-window near-dup gap")
    p.add_argument("--dedup-overlap-ratio", type=float, default=0.5, help="Cross-window overlap ratio")
    p.add_argument("--dedup-repeat-gap", type=float, default=0.22, help="Cross-window repeat gap (ABA)")
    p.add_argument("--dedup-bigram-gap", type=float, default=0.30, help="Cross-window bigram repeat gap")

    # Export
    p.add_argument("--out-dir", type=str, default="out", help="Output directory")
    p.add_argument("--subtitle-format", type=str, default="srt", choices=["srt", "txt"], help="Export subtitles or plain text")
    p.add_argument("--srt-gap-break", type=float, default=0.5, help="Gap threshold for SRT segmentation (s)")
    p.add_argument("--srt-max-line", type=int, default=18, help="Max characters per SRT line")

    # UI
    p.add_argument("--ui-width", type=int, default=120, help="Max characters to display per line")
    p.add_argument("--real-time", action="store_true", help="Sleep per stride to mimic real-time updates")

    # LLM Repair (optional; same as test harness)
    p.add_argument("--repair-enable", action="store_true", help="Enable LLM repair (post-processing)")
    p.add_argument("--repair-provider", type=str, default="ollama", choices=["ollama", "groq"], help="LLM provider")
    p.add_argument("--repair-host", type=str, default="http://localhost:11434", help="Ollama host if provider=ollama")
    p.add_argument("--repair-model", type=str, default="qwen3:4b", help="Model name for LLM provider")
    p.add_argument("--repair-api-key", type=str, default=None, help="API key (e.g., GROQ_API_KEY)")
    p.add_argument("--opencc-config", type=str, default="s2twp.json", help="OpenCC config")
    p.add_argument("--whitelist", type=str, default=None, help="Whitelist mapping for known confusions")
    p.add_argument("--glossary", type=str, default=None, help="Glossary for NER")
    p.add_argument("--seed-period-gap", type=float, default=0.5, help="gap≥x→period")
    p.add_argument("--seed-comma-gap", type=float, default=0.25, help="gap≥x→comma")

    return p.parse_args()

def main():
    args = parse_args()
    if isinstance(args.language, str) and args.language.lower() in {"none", "auto", "null"}:
        args.language = None
    if args.mic:
        run_mic(args)
    else:
        run_file(args)

if __name__ == "__main__":
    main()
