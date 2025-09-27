#!/usr/bin/env python3
"""
fast_test.py — Minimal FAST (file-mode only) with merge debugging.

- File-only streaming simulation: window_len=3s, stride=1s (default).
- Aggregator with COMMITTED/TAIL; time-based overlay; conservative dedup.
- 1Hz terminal view: [C], [T], [VIEW].
- Enhanced debug: show WHY tokens were dropped/cut, with coverage ratios.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import time
import json
import sys
import re

import numpy as np
import librosa

from modules.asr.llm_repair import llm_repair_pipeline, write_srt_with_repair

try:
    from faster_whisper import WhisperModel
except Exception:
    print('[ERROR] faster-whisper not installed. pip install faster-whisper', file=sys.stderr)
    raise

################################################################################
# Data structures
################################################################################

@dataclass
class Tok: #單一token (文字+時間區間+置信度)
    text: str
    start: float
    end: float
    prob: float

@dataclass
class AggregatorState: #儲存目前狀態 (committed已鎖定 + tail可變動區域 + last_committed_end 上次提交時間)
    committed: List[Tok]
    tail: List[Tok]
    last_committed_end: float
    def __init__(self) -> None:
        self.committed = []
        self.tail = []
        self.last_committed_end = 0.0

@dataclass
class CutDetail: #切除細節 (被切除的token + 覆蓋秒數 + 覆蓋比例) debug用
    tok: Tok
    cover_sec: float
    cover_ratio: float  # cover_sec / tok_len

@dataclass
class AppendReport:
    floor: float #地板時間 (last_committed_end - epsilon)
    kept_new: int #此次新增並保留的token數
    dropped_by_floor: int #此次被地板丟棄的token數
    # 以下三項若無剪除則為 None/0
    cut_tail_from: Optional[float] #此次剪尾起始時間 (cut_time) 
    cut_tail_count: int #此次剪尾數量
    committed_moved: int    #此次從tail移到committed的數量
    dropped_by_floor_tokens: List[Tok] #此次被地板丟棄的token清單
    cut_tail_tokens: List[Tok] #此次剪尾的token清單 判定原因(覆蓋達標)
    kept_tokens: List[Tok] #此次新增並保留的token清單
    cut_details: List[CutDetail] #此次剪尾的細節清單 (前3筆)

################################################################################
# Aggregator
################################################################################
class Aggregator:
    def __init__(self,
                commit_tail_sec: float = 1.8,
                epsilon: float = 1e-3,
                san_near_dup_gap: float = 0.08, # 同字、時間幾乎緊貼(≤這個gap) → 視為重覆，丟掉後者
                san_back_overlap_tol: float = 0.02, #若新token的start往「更早」倒退超過這個容忍 → 視為回捲，丟掉
                san_overlap_ratio: float = 0.5,    # 同字且重疊比例≥這門檻 → 視為重覆，丟掉後者
                cov_min_overlap_sec: float = 0.02,  # 最小覆蓋秒數 (通常0.02-0.03秒)
                cov_min_cover_ratio: float = 0.6  # 最小覆蓋比例 (通常0.55-0.7秒)
                ) -> None: 
        self.state = AggregatorState()
        self.commit_tail_sec = float(commit_tail_sec) #允許的可編輯 TAIL 長度 (常用範圍1.2~2.5秒)
        self.eps = float(epsilon) #容忍誤差 (通常0.001秒即可)
        self.fuse_back = 0.12  #將新字往前推的寬度 (秒) (通常0.08~0.16秒) 目的是讓新字能覆蓋到舊字    
        # knobs
        self.san_near_dup_gap = float(san_near_dup_gap)
        self.san_back_overlap_tol = float(san_back_overlap_tol)
        self.san_overlap_ratio = float(san_overlap_ratio)
        self.cov_min_overlap_sec = float(cov_min_overlap_sec)
        self.cov_min_cover_ratio = float(cov_min_cover_ratio)
        self.front_grace_sec = 0.0  # 新增：新窗第一顆token若離舊尾太近，短暫停用剪尾
    # ---- helpers ----
    def tail_duration(self) -> float: #計算目前 TAIL 長度 
        #無需調整
        s = self.state
        if not s.tail: return 0.0
        return max(0.0, s.tail[-1].end - s.tail[0].start)

    # 清理新 tokens（丟掉明顯重覆或回捲的）[僅限這一窗內]
    def _sanitize_new_tokens(self, new: List[Tok]) -> List[Tok]:
        """
        目的（白話）：
        清掉「這一窗內」明顯的重覆與時間回捲，留下乾淨的 token 序列再交給 Aggregator。
        規則只比「這一窗內的相鄰token」，不處理跨窗（跨窗在後面dedup）。

        規則：
        1) 同字 + 幾乎緊貼  → 刪掉後面那顆（避免你你）
        2) 同字 + 重疊很多  → 刪掉後面那顆（避免你你）
        3) 時間回捲(往回跑) → 刪掉這顆（避免時間軸倒退）

        參數如何理解：
        - near_dup_gap：兩顆同字，若第二顆的start離第一顆的end很近(≤gap)就當重覆。
        - overlap_ratio_thr：兩顆同字，若時間重疊比例很高(≥thr)就當重覆。
        - back_overlap_tol：這顆的start若比目前序列的「最後end」還更早超過tol → 當回捲。
        """
        def norm(s: str) -> str: return s.strip()#去除前後空白
        def overlap_ratio(a: Tok, b: Tok) -> float: #計算兩token的重疊比例
            inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
            den = max(1e-6, min(a.end - a.start, b.end - b.start))
            return inter / den
        out: List[Tok] = []
        last_end = -1e9
        for t in new:
            if out:
                prev = out[-1]
                same = (norm(t.text) == norm(prev.text))
                if same and ((t.start - prev.end) <= self.san_near_dup_gap + self.eps):  # 幾乎相接
                    continue
                if same and (overlap_ratio(t, prev) >= self.san_overlap_ratio):      # 高重疊
                    continue
            if t.start < (last_end - self.san_back_overlap_tol - self.eps):     # 明顯回捲
                continue
            out.append(t)
            last_end = max(last_end, t.end)
        return out

    # 覆蓋判定（允許剪舊 TAIL 的必要條件）
    def _covered_enough(self, tok: Tok, new_toks: List[Tok]) -> Tuple[bool, float, float]:
        cover = 0.0
        for n in new_toks:
            overlap = min(tok.end, n.end) - max(tok.start, n.start)
            if overlap > 0: cover += overlap
        tok_len = max(tok.end - tok.start, 1e-6)
        ratio = cover / tok_len
        ok = (cover >= (self.cov_min_overlap_sec - self.eps)) and (ratio >= self.cov_min_cover_ratio)
        return ok, cover, ratio

    def append_fast(self, toks: List[Tok]) -> AppendReport:
        s = self.state
        if not toks:
            return AppendReport(s.last_committed_end, 0, 0, None, 0, 0, [], [], [], [])

        toks = sorted(toks, key=lambda t: (t.start, t.end))
        # 只以 committed_end 當地板（保留覆蓋 TAIL 的能力）
        floor = s.last_committed_end - self.eps
        kept = [t for t in toks if t.start >= floor] #保留的token (≥地板時間)
        dropped_tokens = [t for t in toks if t.start < floor] #被丟掉的token (<地板時間)
        kept = self._sanitize_new_tokens(kept) #清理新tokens (去掉同窗內重覆/回捲)
        if not kept: #若清理後沒東西，直接回報
            return AppendReport(s.last_committed_end, 0, len(dropped_tokens),
                                None, 0, 0, dropped_tokens, [], [], [])

        first_new_start = kept[0].start #第一個新token的start
        # ===== 新增：前緣寬限，避免接縫誤剪 =====
        if self.front_grace_sec > 0 and self.state.tail:
            gap = first_new_start - self.state.tail[-1].end
            if gap <= (self.front_grace_sec + self.eps):
                # 這一窗不剪尾（把 cut_time 推到超大）
                cut_time = 1e9
        # =========================================
        cut_time = first_new_start - self.fuse_back #計算剪尾時間 (往前推 fuse_back 秒)

        # 只剪「被新窗充分覆蓋」且「靠近尾端」的 TAIL token
        to_cut: List[Tok] = []
        cut_details: List[CutDetail] = []
        for t in s.tail:
            if t.end >= cut_time - self.eps:
                ok, cover, ratio = self._covered_enough(t, kept) #判定是否覆蓋達標
                if ok: 
                    to_cut.append(t) #加入剪除清單
                    cut_details.append(CutDetail(tok=t, cover_sec=cover, cover_ratio=ratio))
        if to_cut:
            # 用物件身份 id() 當 key，避免需要 Tok 可 hash
            kill_ids = {id(t) for t in to_cut}
            s.tail = [t for t in s.tail if id(t) not in kill_ids]

        # 接上新字
        s.tail.extend(kept)

        # 邊界去重（跨窗）：同字近距/高重疊/片語重覆 → 視為重覆，丟掉後者
        def _norm(s: str) -> str: return s.strip()
        def _overlap_ratio(a: Tok, b: Tok) -> float:
            inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
            den = max(1e-6, min(a.end - a.start, b.end - b.start))
            return inter / den

        near_gap  = float(getattr(self, "dedup_near_gap", 0.08))
        high_thr  = float(getattr(self, "dedup_overlap_ratio", 0.5))
        rep_gap   = float(getattr(self, "dedup_repeat_gap", 0.22))     # ★ 新
        bi_gap    = float(getattr(self, "dedup_bigram_gap", 0.30))     # ★ 新

        dedup: List[Tok] = []
        for t in s.tail:
            if dedup:
                prev = dedup[-1]
                same = (_norm(t.text) == _norm(prev.text))
                gap  = t.start - prev.end

                # 規則A：同字且「幾乎連接」或「高重疊」
                near = (gap <= near_gap + self.eps)
                high = (_overlap_ratio(t, prev) >= high_thr)
                if same and (near or high):
                    continue

                # 規則B：同字「近距重覆」（無重疊但間隙很小，也視為重覆）
                if same and (gap >= -self.eps) and (gap <= rep_gap + self.eps):
                    continue

                # 規則C：雙字片語重覆（… X Y X Y …）
                # 只有在處理第二個 Y 時才有足夠上下文可判斷
                if len(dedup) >= 2:
                    prev2 = dedup[-2]      # 倒數第二顆（例：第一次的 Y）
                    # 檢查 (prev2, prev) 與 (prev, t) 是否構成 [X Y] [X Y] 的重覆
                    # 即：prev2.text == prev.text 且 prev.text == t.text，不對；要比「片語內容」相同：
                    # 正確條件： (dedup[-2].text, dedup[-1].text) == (prev.text, t.text) 其實等價於 dedup[-1] 就是 prev
                    # 但我們要比「前一對」（X,Y）和「現在這一對」（X,Y），因此在處理 t=第二個 Y 時，
                    # 前一對是 (dedup[-3], dedup[-2])，現在這一對是 (dedup[-1], t)。
                    if len(dedup) >= 3:
                        A1 = _norm(dedup[-3].text)  # 第一次的 X
                        A2 = _norm(dedup[-2].text)  # 第一次的 Y
                        B1 = _norm(dedup[-1].text)  # 第二次的 X
                        B2 = _norm(t.text)          # 第二次的 Y
                        if (A1 == B1) and (A2 == B2):
                            pair_gap = t.start - dedup[-2].end  # 第一次 Y → 第二次 Y 的間隙
                            if pair_gap <= bi_gap + self.eps:
                                continue
                    
                # === 新增：ABA（skip-back-2）去重，使用既有 dedup_near_gap 門檻 ===
                if len(dedup) >= 2:
                    prev2 = dedup[-2]
                    if _norm(t.text) == _norm(prev2.text):
                        gap_prev2 = t.start - prev2.end
                        rep_gap = float(getattr(self, "dedup_repeat_gap", 0.22))  # ← 用這個
                        # 允許中間夾一顆不同字（典型 ABA），但要求兩次同字距離不超過 repeat 門檻
                        if gap_prev2 <= rep_gap + self.eps:
                            continue



            dedup.append(t)

        s.tail = dedup

        # 控制 tail 長度 → 超過就往 committed 推
        commit_ops = 0
        while self.tail_duration() > self.commit_tail_sec and s.tail:
            tok = s.tail.pop(0)
            s.committed.append(tok)
            s.last_committed_end = max(s.last_committed_end, tok.end)
            commit_ops += 1

        # 這裡的 return 用 cut_time（若有剪才回，否則 None）
        return AppendReport(
            floor, len(kept), len(dropped_tokens),
            cut_time if to_cut else None,   # ← 這裡
            len(to_cut), commit_ops,
            dropped_tokens, to_cut, kept, cut_details
        )
        


    def finalize(self) -> None: #將剩餘的 TAIL 全部提交
        s = self.state
        while s.tail:
            tok = s.tail.pop(0)
            s.committed.append(tok)
            s.last_committed_end = max(s.last_committed_end, tok.end)

    # Views
    def view_text(self) -> str:
        return tokens_to_text(self.state.committed + self.state.tail)
    def committed_text(self) -> str:
        return tokens_to_text(self.state.committed)
    def tail_text(self) -> str:
        return tokens_to_text(self.state.tail)

################################################################################
# Token/Text utilities
################################################################################
#regex 在判定是否需要空白時會用到 (如果兩邊都是字母或數字) 
_re_alnum_end = re.compile(r"[A-Za-z0-9]$")
_re_alnum_start = re.compile(r"^[A-Za-z0-9]")

def needs_space(a: str, b: str) -> bool:
    return bool(_re_alnum_end.search(a)) and bool(_re_alnum_start.search(b))

def tokens_to_text(toks: List[Tok]) -> str:
    if not toks: return ""
    out: List[str] = []
    prev: Optional[Tok] = None
    for t in toks:
        if prev is None:
            out.append(t.text)
        else:
            if needs_space(prev.text, t.text):
                out.append(" ")
            out.append(t.text)
        prev = t
    return "".join(out)

################################################################################
# Audio helper: File-only streaming simulator
################################################################################

TARGET_SR = 16000

class FileStreamSimulator:
    def __init__(self, path: Path, window_len: float, stride: float, endpad_sec: float = 0.0) -> None:
        self.window_len = float(window_len)
        self.stride = float(stride)
        self.endpad = float(endpad_sec)
        self.y, _sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
        self.duration = len(self.y) / TARGET_SR
        self.t = self.window_len  # start at window_len

    def done(self) -> bool:
        # 讓時間走到 duration + endpad 才結束
        return self.t > (self.duration + self.endpad) + 1e-6

    def tick(self) -> Optional[np.ndarray]:
        if self.done(): return None
        start = max(0.0, self.t - self.window_len)
        end = min(self.duration, self.t)   # 音訊只取到真實結束，其餘用 0 補
        a = int(start * TARGET_SR); b = int(end * TARGET_SR)
        win = self.y[a:b]
        need = int(self.window_len * TARGET_SR)
        if len(win) < need:
            pad = np.zeros(need - len(win), dtype=np.float32)  # 這段就是結尾靜音
            win = np.concatenate([pad, win.astype(np.float32)])
        else:
            win = win.astype(np.float32)
        self.t += self.stride
        return win


################################################################################
# Decoder (FAST)
################################################################################

class FastDecoder:
    def __init__(self, model_name: str, device: str, compute_type: str, beam_size: int, language: Optional[str]) -> None:
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        self.language = language

    def decode_window(self, audio_window: np.ndarray, abs_start_time: float) -> Tuple[List[Tok], float]:
        t0 = time.perf_counter()
        segments, _info = self.model.transcribe(
            audio_window,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=False,
            word_timestamps=True,
        )
        toks: List[Tok] = []
        for seg in segments:
            if getattr(seg, 'words', None):
                for w in seg.words:
                    ws = (w.start or 0.0) + abs_start_time
                    we = (w.end or (w.start or 0.0)) + abs_start_time
                    prob = float(getattr(w, 'probability', 0.0) or 0.0)
                    txt = (w.word or "")
                    toks.append(Tok(text=txt.strip(), start=float(ws), end=float(we), prob=prob))
            else:
                ws = (seg.start or 0.0) + abs_start_time
                we = (seg.end or (seg.start or 0.0)) + abs_start_time
                toks.append(Tok(text=(seg.text or "").strip(), start=float(ws), end=float(we), prob=0.0))
        t1 = time.perf_counter()
        rtf = (t1 - t0) / (len(audio_window) / TARGET_SR)
        toks.sort(key=lambda t: (t.start, t.end))
        return toks, rtf

################################################################################
# Minimal JSONL writer (optional)
################################################################################

def write_jsonl(path: Path, toks: List[Tok]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for t in toks:
            f.write(json.dumps({'text': t.text, 'start': t.start, 'end': t.end, 'prob': t.prob}, ensure_ascii=False) + "\n")

################################################################################
# 字幕與評估工具函式
################################################################################
import os
import glob
import csv

# ---------- simple text utils ----------
_cjk_rng = ('\u4e00', '\u9fff')
def is_cjk(ch: str) -> bool:
    return len(ch)==1 and _cjk_rng[0] <= ch <= _cjk_rng[1]

def normalize_text(s: str) -> str:
    # 低強度正規化：去首尾、把連續空白壓成單一空白
    s = re.sub(r'\s+', ' ', s.strip())
    return s

def tokenize_for_eval(s: str, unit: str) -> List[str]:
    s = normalize_text(s)
    if unit == 'char':
        return list(s.replace(' ', ''))
    elif unit == 'word':
        return s.split(' ')
    else:
        # auto: 如果包含大量 CJK，就用 char，否則 word
        cjk_cnt = sum(1 for ch in s if is_cjk(ch))
        return list(s.replace(' ', '')) if cjk_cnt >= max(1, len(s)//10) else s.split(' ')

def levenshtein(a: List[str], b: List[str]) -> int:
    # O(|a||b|) DP
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = 0 if ai == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[n][m]

def compute_error_rate(hyp: str, ref: str, unit: str = 'auto') -> Tuple[float, float]:
    hyp_tokens = tokenize_for_eval(hyp, unit)
    ref_tokens = tokenize_for_eval(ref, unit)
    dist = levenshtein(hyp_tokens, ref_tokens)
    denom = max(1, len(ref_tokens))
    err = dist / denom
    return err, dist

# ---------- subtitle writers ----------
def write_txt(path: Path, toks: List[Tok]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tokens_to_text(toks) + "\n")

def write_srt(path: Path, toks: List[Tok], max_len: int = 70, gap_break: float = 0.5) -> None:
    """
    極簡 SRT：以時間缺口(gap_break)分段；每段字數超過 max_len 就硬切。
    """
    if not toks:
        with open(path, 'w', encoding='utf-8') as f:
            pass
        return
    segments = []
    cur = []
    for i, t in enumerate(toks):
        if not cur:
            cur = [t]
            continue
        prev = cur[-1]
        gap = t.start - prev.end
        if gap >= gap_break:
            segments.append(cur)
            cur = [t]
        else:
            cur.append(t)
    if cur:
        segments.append(cur)

    # 輸出 SRT（簡單硬切字數）
    def fmt_ts(x: float) -> str:
        h = int(x // 3600); x -= h*3600
        m = int(x // 60);   x -= m*60
        s = int(x);         ms = int((x - s) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(path, 'w', encoding='utf-8') as f:
        idx = 1
        for seg in segments:
            s = seg[0].start; e = seg[-1].end
            text = tokens_to_text(seg)
            # 依長度分行
            lines = []
            while len(text) > max_len:
                lines.append(text[:max_len])
                text = text[max_len:]
            if text:
                lines.append(text)
            f.write(f"{idx}\n{fmt_ts(s)} --> {fmt_ts(e)}\n")
            f.write("\n".join(lines) + "\n\n")
            idx += 1
#######################
def create_decoder(args) -> "FastDecoder":
    device = detect_device() if args.device == 'auto' else args.device
    compute_type = args.compute_type
    if device == 'cpu' and compute_type == 'float16':
        compute_type = 'int8'  # 或 'int8_float32'
    return FastDecoder(args.model_name, device, compute_type, args.beam_size, args.language)

# ---------- single-file runner (no per-window logs) ----------
def run_one_file(audio_path: Path, args, dec: "FastDecoder") -> dict:
    print(f"[RUN] {audio_path}")

    sim = FileStreamSimulator(audio_path, args.window_len, args.stride, endpad_sec=args.endpad_sec)
    agg = Aggregator(
        commit_tail_sec=args.commit_tail_sec,
        epsilon=args.epsilon,
        san_near_dup_gap=args.san_near_dup_gap,
        san_back_overlap_tol=args.san_back_overlap_tol,
        san_overlap_ratio=args.san_overlap_ratio,
        cov_min_overlap_sec=args.cov_min_overlap_sec,
        cov_min_cover_ratio=args.cov_min_cover_ratio,
    )
    # ← 新增這行，把 fuse_back 改用 CLI
    agg.fuse_back = float(args.fuse_back)
    agg.dedup_near_gap = float(args.dedup_near_gap)
    agg.dedup_overlap_ratio = float(args.dedup_overlap_ratio)
    agg.dedup_repeat_gap = float(args.dedup_repeat_gap)      # ★ 新增
    agg.dedup_bigram_gap = float(args.dedup_bigram_gap)      # ★ 新增
    sim_time = args.window_len
    while not sim.done():
        window = sim.tick()
        if window is None: break

        abs_start = max(0.0, sim_time - args.window_len)
        toks, rtf = dec.decode_window(window, abs_start)

        win_start = max(0.0, sim_time - args.window_len)
        win_end   = sim_time
        is_last_window = (sim_time >= sim.duration - 1e-3)  # 給較寬 margin，避免浮點誤差
        is_first_window = (win_start <= 1e-6)

        if is_first_window:
            accept_s = 0.0
        elif win_start < args.guard_sec:
            accept_s = 0.0
        else:
            accept_s = win_start + args.guard_sec
        accept_s = max(accept_s, agg.state.last_committed_end - args.epsilon)
                
        eps = args.epsilon
        is_last_window = (sim_time >= sim.duration - 1e-3)  # 比舊的更穩定

        base_e = win_end if is_last_window else (win_end - args.guard_sec)
        hard_e = base_e if not is_last_window else (win_end + args.last_slack_sec)

        def keep_token(t):
            if t.start < accept_s - eps:
                return False
            if is_last_window:
                # 最後一窗：整顆只要「結尾」在 hard_e 之內就放行（允許起頭跨過 base_e）
                return t.end <= (win_end + args.last_slack_sec) + eps
            else:
                # 中間視窗：照舊，不能碰到右 guard
                return t.end <= (win_end - args.guard_sec) + eps


        mid_toks = [t for t in toks if keep_token(t)]
        window_drop_left  = [t for t in toks if t.start < accept_s - eps]
        window_drop_right = [t for t in toks if (t.start >= accept_s - eps) and (not keep_token(t))]



        if agg.state.tail and args.protect_head_sec > 0:
            tail_head = agg.state.tail[0].start
            cutoff = tail_head + args.protect_head_sec
            cutoff = max(cutoff, agg.state.last_committed_end + 0.005)
            cutoff = max(cutoff, accept_s)

            # 接縫放行：允許接縫附近的小段進來修補上一窗尾巴
            seam_free = agg.state.tail[-1].end - agg.fuse_back

            kept_mid = []
            for t in mid_toks:
                if t.start >= cutoff - agg.eps:
                    kept_mid.append(t)
                else:
                    if t.start >= seam_free - agg.eps:
                        kept_mid.append(t)
            mid_toks = kept_mid



        if is_last_window and agg.state.tail and args.final_protect_sec > 0:
            tail_end = agg.state.tail[-1].end
            protect_start = max(accept_s, tail_end - args.final_protect_sec)
            mid_toks = [t for t in mid_toks if t.start >= protect_start - args.epsilon]

        report = agg.append_fast(mid_toks)

        # ------ logs （保留原本逐窗觀測） ------
        C = agg.committed_text()
        T = agg.tail_text()
        V = agg.view_text()
        tail_dur = agg.tail_duration()
        last_c = agg.state.last_committed_end
        p = (sum(t.prob for t in agg.state.tail) / max(1, len(agg.state.tail))) if agg.state.tail else 0.0

        print(
            f"t={sim_time:5.1f}s | +{report.kept_new:02d}tok | "
            f"drop_win(L/R)={len(window_drop_left):02d}/{len(window_drop_right):02d} | "
            f"drop_floor={report.dropped_by_floor:02d} | "
            f"cut_tail={report.cut_tail_count:02d} | commit={report.committed_moved:02d} | "
            f"tail={tail_dur:4.2f}s | last={last_c:5.2f}s | RTF={rtf:4.2f} | p={p:4.2f}"
        )

        orig_text  = tokens_to_text(toks)
        text_dropL = tokens_to_text(window_drop_left)
        text_dropR = tokens_to_text(window_drop_right)
        text_floor = tokens_to_text(report.dropped_by_floor_tokens)
        text_cut   = tokens_to_text(report.cut_tail_tokens)

        print(f"{win_start:.0f}-{win_end:.0f} 原(視窗ASR): {truncate_middle(orig_text, args.view_width)}")
        print(f"       丟棄(視窗-左): {truncate_middle(text_dropL, args.view_width) if text_dropL else '（無）'}")
        print(f"       丟棄(視窗-右): {truncate_middle(text_dropR, args.view_width) if text_dropR else '（無）'}")
        print(f"       丟棄(地板/舊時間): {truncate_middle(text_floor, args.view_width) if text_floor else '（無）'}")
        if report.cut_tail_tokens:
            brief = ", ".join([f"{d.tok.text}(覆蓋{d.cover_sec:.2f}s/{d.cover_ratio:.2f})" for d in report.cut_details[:3]])
            print(f"       切除(尾端覆蓋達標): {truncate_middle(text_cut, args.view_width)}")
            print(f"       └ 覆蓋細節: {brief}" if brief else "       └ 覆蓋細節: -")
        else:
            print("       切除(尾端覆蓋達標): （無）")
        print(f"       合併後: {truncate_middle(V, args.view_width)}")

        print('[C] ' + truncate_middle(C, args.view_width))
        print('[T] ' + truncate_middle(T, args.view_width))
        print('[VIEW] ' + truncate_middle(V, args.view_width))
        print('-' * min(args.view_width, 120))

        if args.real_time:
            time.sleep(max(0.0, args.stride))

        sim_time += args.stride

    # finalize & write outputs（每檔各自輸出）
    agg.finalize()
    committed = agg.state.committed

    # 轉 token 給修字管線（不修也用得到）
    toks_dict = [{"text": t.text, "start": t.start, "end": t.end, "prob": t.prob} for t in committed]
    hyp_text = tokens_to_text(committed)

    # （A）是否啟用 LLM Repair
    if getattr(args, 'repair_enable', False):
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
        clean_text = res.clean_text
    else:
        clean_text = hyp_text

    # （B）輸出
    os.makedirs(args.out_dir, exist_ok=True)
    stem = audio_path.stem

    if args.subtitle_format == 'srt':
        if getattr(args, 'repair_enable', False):
            # 用修後句子 + 原始時間 gap 重切
            write_srt_with_repair(
                str(Path(args.out_dir) / f"{stem}.srt"),
                toks_dict,
                clean_text,
                gap_break=args.srt_gap_break,
                max_line=args.srt_max_line
            )
        else:
            # 走原本的極簡 SRT
            write_srt(Path(args.out_dir) / f"{stem}.srt", committed)
    else:
        # 純文字：若有修字就寫修後，否則原文
        with open(Path(args.out_dir) / f"{stem}.txt", 'w', encoding='utf-8') as f:
            f.write(clean_text + "\n")

    # （C）edits 審計日誌（有啟用修字時）
    if getattr(args, 'repair_enable', False):
        try:
            with open(Path(args.out_dir) / f"{stem}.edits.jsonl", 'w', encoding='utf-8') as ef:
                for e in res.edits:
                    ef.write(json.dumps(e.__dict__, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # （D）JSONL（沿用你的邏輯）
    if args.keep_jsonl and (not args.no_jsonl):
        write_jsonl(Path(args.out_dir) / f"{stem}.jsonl", committed)

    metrics = {
        'file': str(audio_path),
        'duration_sec': sim.duration,
        'tokens': len(committed),
        'avg_prob': (sum(t.prob for t in committed) / max(1, len(committed))) if committed else 0.0,
    }
    print(f"[OK]  {audio_path}")
    return {'text': clean_text if getattr(args, 'repair_enable', False) else hyp_text, 'metrics': metrics}




################################################################################
# Pretty terminal printing
################################################################################

def truncate_middle(s: str, max_len: int) -> str:
    if len(s) <= max_len: return s
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return s[:left] + '...' + s[-right:]

################################################################################
# Main (file-only)
################################################################################

def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available(): return 'cuda'
    except Exception:
        pass
    return 'cpu'



def run_file_mode(args) -> None:
    print(f"[INFO] Device: {detect_device() if args.device=='auto' else args.device}")
    device = detect_device() if args.device == 'auto' else args.device
    compute_type = args.compute_type
    if device == 'cpu' and compute_type == 'float16':
        compute_type = 'int8'  # 或 'int8_float32'
    dec = FastDecoder(args.model_name, device, compute_type, args.beam_size, args.language)

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
    # ← 新增這行，把 fuse_back 改用 CLI
    agg.fuse_back = float(args.fuse_back)
    agg.dedup_near_gap = float(args.dedup_near_gap)
    agg.dedup_overlap_ratio = float(args.dedup_overlap_ratio)
    agg.dedup_repeat_gap = float(args.dedup_repeat_gap)      # ★ 新增
    agg.dedup_bigram_gap = float(args.dedup_bigram_gap)      # ★ 新增

    sim_time = args.window_len

    while not sim.done():
        window = sim.tick()
        if window is None: break

        abs_start = max(0.0, sim_time - args.window_len)
        toks, rtf = dec.decode_window(window, abs_start)

        win_start = max(0.0, sim_time - args.window_len)
        win_end   = sim_time
        is_last_window = (sim_time + 1e-6 >= sim.duration)
        is_first_window = (win_start <= 1e-6)   # ★ 新增：首窗判定
        
        # 1) 先決定左界
        if is_first_window:
            accept_s = 0.0                      # ★ 首窗絕不切左邊
        elif win_start < args.guard_sec:
            accept_s = 0.0                      # 起始幾秒內放寬左界（漸進）
        else:
            accept_s = win_start + args.guard_sec
            
        # 2) 再做「不覆蓋已鎖定」的夾緊（但首窗 last_committed_end=0，不會動）
        accept_s = max(accept_s, agg.state.last_committed_end - args.epsilon)

        # 3) 右界：最後一窗不切尾，其餘切掉右側 guard
        eps = args.epsilon
        is_last_window = (sim_time >= sim.duration - 1e-3)  # 比舊的更穩定

        base_e = win_end if is_last_window else (win_end - args.guard_sec)
        hard_e = base_e if not is_last_window else (win_end + args.last_slack_sec)

        def keep_token(t):
            if t.start < accept_s - eps:
                return False
            if is_last_window:
                # 最後一窗：整顆只要「結尾」在 hard_e 之內就放行（允許起頭跨過 base_e）
                return t.end <= (win_end + args.last_slack_sec) + eps
            else:
                # 中間視窗：照舊，不能碰到右 guard
                return t.end <= (win_end - args.guard_sec) + eps


        mid_toks = [t for t in toks if keep_token(t)]
        window_drop_left  = [t for t in toks if t.start < accept_s - eps]
        window_drop_right = [t for t in toks if (t.start >= accept_s - eps) and (not keep_token(t))]



        # 5) 每窗的「保護 TAIL 頭段」—首窗 tail 是空，不會觸發
        if agg.state.tail and args.protect_head_sec > 0:
            tail_head = agg.state.tail[0].start
            cutoff = tail_head + args.protect_head_sec
            cutoff = max(cutoff, agg.state.last_committed_end + 0.005)
            cutoff = max(cutoff, accept_s)

            # 接縫放行：允許接縫附近的小段進來修補上一窗尾巴
            seam_free = agg.state.tail[-1].end - agg.fuse_back

            kept_mid = []
            for t in mid_toks:
                if t.start >= cutoff - agg.eps:
                    kept_mid.append(t)
                else:
                    if t.start >= seam_free - agg.eps:
                        kept_mid.append(t)
            mid_toks = kept_mid



        # 6) 最後一窗的尾端保護（只動尾巴）
        if is_last_window and agg.state.tail and args.final_protect_sec > 0:
            tail_end = agg.state.tail[-1].end
            protect_start = max(accept_s, tail_end - args.final_protect_sec)
            mid_toks = [t for t in mid_toks if t.start >= protect_start - args.epsilon]

        report = agg.append_fast(mid_toks)
        
                        
        # ------ Printing ------
        C = agg.committed_text()
        T = agg.tail_text()
        V = agg.view_text()
        tail_dur = agg.tail_duration()
        last_c = agg.state.last_committed_end
        p = 0.0
        if agg.state.tail:
            p = float(sum(t.prob for t in agg.state.tail) / max(1, len(agg.state.tail)))

        print(
            f"t={sim_time:5.1f}s | +{report.kept_new:02d}tok | "
            f"drop_win(L/R)={len(window_drop_left):02d}/{len(window_drop_right):02d} | "
            f"drop_floor={report.dropped_by_floor:02d} | "
            f"cut_tail={report.cut_tail_count:02d} | commit={report.committed_moved:02d} | "
            f"tail={tail_dur:4.2f}s | last={last_c:5.2f}s | RTF={rtf:4.2f} | p={p:4.2f}"
        )

        orig_text = tokens_to_text(toks)
        text_dropL = tokens_to_text(window_drop_left)
        text_dropR = tokens_to_text(window_drop_right)
        text_floor = tokens_to_text(report.dropped_by_floor_tokens)
        text_cut   = tokens_to_text(report.cut_tail_tokens)

        print(f"{win_start:.0f}-{win_end:.0f} 原(視窗ASR): {truncate_middle(orig_text, args.view_width)}")
        print(f"       丟棄(視窗-左): {truncate_middle(text_dropL, args.view_width) if text_dropL else '（無）'}")
        print(f"       丟棄(視窗-右): {truncate_middle(text_dropR, args.view_width) if text_dropR else '（無）'}")
        print(f"       丟棄(地板/舊時間): {truncate_middle(text_floor, args.view_width) if text_floor else '（無）'}")
        if report.cut_tail_tokens:
            # 顯示部分 cut 詳情（前 3 顆）
            brief = ", ".join([f"{d.tok.text}(覆蓋{d.cover_sec:.2f}s/{d.cover_ratio:.2f})" for d in report.cut_details[:3]])
            print(f"       切除(尾端覆蓋達標): {truncate_middle(text_cut, args.view_width)}")
            print(f"       └ 覆蓋細節: {brief}" if brief else "       └ 覆蓋細節: -")
        else:
            print("       切除(尾端覆蓋達標): （無）")
        print(f"       合併後: {truncate_middle(V, args.view_width)}")

        print('[C] ' + truncate_middle(C, args.view_width))
        print('[T] ' + truncate_middle(T, args.view_width))
        print('[VIEW] ' + truncate_middle(V, args.view_width))
        print('-' * min(args.view_width, 120))

        if args.real_time:
            time.sleep(max(0.0, args.stride))

        sim_time += args.stride

    # finalize & optional JSONL dump
    agg.finalize()
    if not args.no_jsonl:
        out = Path('fast_test_output.jsonl')
        write_jsonl(out, agg.state.committed)
        print(f"[DONE] wrote {out}")
    else:
        print("[DONE] (no JSONL dump)")

################################################################################
# CLI
################################################################################

def parse_args():
    p = argparse.ArgumentParser(description='Minimal FAST test (file-mode only)')

    # === 輸入路徑：三擇一 ===
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--audio', type=str, help='Single audio file (wav/mp3/flac)')
    group.add_argument('--inputs', nargs='+', help='Process multiple audio files (list)')
    group.add_argument('--input-glob', type=str, help='Glob pattern for input files, e.g., "data/*.wav"')

    # 基本模型參數
    p.add_argument('--model-name', type=str, default='medium', help='faster-whisper model (e.g., small, medium, large-v3)')
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device selection')
    p.add_argument('--compute-type', type=str, default='float16', help='Compute type (e.g., float16, int8_float16, int8)')
    p.add_argument('--language', type=str, default='none', help='Language hint (e.g., zh, en). Use "none" to auto-detect')
    p.add_argument('--beam-size', type=int, default=3, help='Beam size for decoding (FAST default=3)')

    # 視窗切片
    p.add_argument('--window-len', type=float, default=3.0, help='視窗長度 (秒)')
    p.add_argument('--stride', type=float, default=1.0, help='視窗步進 (秒)')
    p.add_argument('--commit-tail-sec', type=float, default=1.6, help='允許的可編輯 TAIL 長度 (秒)')
    p.add_argument('--epsilon', type=float, default=0.001, help='容忍誤差 (秒)')

    # 視窗篩選/保護
    p.add_argument('--view-width', type=int, default=120, help='終端單行顯示寬度')
    p.add_argument('--no-jsonl', action='store_true', help='是否不輸出 JSONL')
    p.add_argument('--guard-sec', type=float, default=0.4, help='視窗左右側保護區 (秒)')
    p.add_argument('--protect-head-sec', type=float, default=0.8, help='tail前段保護區 (秒)')
    p.add_argument('--final-protect-sec', type=float, default=0.6, help='最後一窗的尾端保護區 (秒)')
    p.add_argument('--real-time', action='store_true', help='模擬真實時間刷新的節奏 (每步進sleep stride秒)')

    # 同窗清理 knobs
    p.add_argument('--san-near-dup-gap', type=float, default=0.12, help='同字、時間幾乎緊貼(≤這個gap) → 視為重覆，丟掉後者')
    p.add_argument('--san-back-overlap-tol', type=float, default=0.02, help='若新token的start往「更早」倒退超過這個容忍 → 視為回捲，丟掉')
    p.add_argument('--san-overlap-ratio', type=float, default=0.5, help='同字且重疊比例≥這門檻 → 視為重覆，丟掉後者')

    # 覆蓋剪尾 knobs
    p.add_argument('--cov-min-overlap-sec', type=float, default=0.02, help='最小覆蓋秒數 (通常0.02-0.03秒)')
    p.add_argument('--cov-min-cover-ratio', type=float, default=0.6, help='最小覆蓋比例 (通常0.55-0.7)')

    # 批次與輸出
    p.add_argument('--out-dir', type=str, default='out', help='Output directory for transcripts/subtitles')
    p.add_argument('--subtitle-format', type=str, default='srt', choices=['srt','txt'], help='Subtitle output format')
    p.add_argument('--keep-jsonl', action='store_true', help='Also write JSONL of committed tokens per file')

    # 可選的正確率評估
    p.add_argument('--ref-dir', type=str, default=None, help='Directory of reference .txt (same basename) to compute WER/CER')
    p.add_argument('--eval-unit', type=str, default='auto', choices=['auto','word','char'], help='Evaluation unit')

    p.add_argument('--last-slack-sec', type=float, default=0.12,
                help='最後一窗的右側寬容(秒)：允許 token.end 超出視窗終點避免吞尾字')
    p.add_argument('--fuse-back', type=float, default=0.16,
                help='剪尾時往回回縮(秒)：新窗可向左「回拉」這麼多來覆蓋接縫')

    p.add_argument('--dedup-near-gap', type=float, default=0.1,
                help='Cross-window dedup: tokens with same text and gap <= this are treated as duplicates')
    p.add_argument('--dedup-overlap-ratio', type=float, default=0.5,
                help='Cross-window dedup: tokens with same text and overlap ratio >= this are treated as duplicates')

    p.add_argument('--endpad-sec', type=float, default=0.6,
               help='Add this much trailing silence for the last windows so the model can finalize the last tokens')
    p.add_argument('--dedup-repeat-gap', type=float, default=0.22,
        help='Cross-window dedup: same-token repeated within this gap (s) will be dropped')
    p.add_argument('--dedup-bigram-gap', type=float, default=0.30,
        help='Cross-window dedup: repeated 2-token phrase within this gap (s) will be dropped')
    
    # --- LLM Repair 開關 & 參數（預設關閉） ---
    p.add_argument('--repair-enable', action='store_true', help='Enable LLM repair (file-mode only)')
    p.add_argument('--repair-provider', type=str, default='ollama', choices=['ollama','groq'], help='LLM provider')
    p.add_argument('--repair-host', type=str, default='http://localhost:11434', help='Ollama host if provider=ollama')
    p.add_argument('--repair-model', type=str, default='qwen3:4b', help='Model name for LLM provider')
    p.add_argument('--repair-api-key', type=str, default=None, help='API key (e.g., GROQ_API_KEY)')

    p.add_argument('--opencc-config', type=str, default='s2twp.json', help='OpenCC config (e.g., s2twp.json)')
    p.add_argument('--whitelist', type=str, default=None, help='JSON mapping for known confusions')
    p.add_argument('--glossary', type=str, default=None, help='JSON/lines glossary for NER')

    p.add_argument('--seed-period-gap', type=float, default=0.5, help='gap≥x→句號（粗標點）')
    p.add_argument('--seed-comma-gap', type=float, default=0.25, help='gap≥x→逗點（粗標點）')

    p.add_argument('--srt-gap-break', type=float, default=0.5, help='SRT 以 gap 分段的門檻（秒）')
    p.add_argument('--srt-max-line', type=int, default=18, help='SRT 每行最大字數（CJK 約 10–18）')
    
    
    return p.parse_args()

def main():
    args = parse_args()
    if isinstance(args.language, str) and args.language.lower() in {'none', 'auto', 'null'}:
        args.language = None

    # 收集輸入檔
    files: List[Path] = []
    if args.inputs:
        files.extend([Path(p) for p in args.inputs])
    if args.input_glob:
        files.extend([Path(p) for p in glob.glob(args.input_glob)])
    if not files and getattr(args, 'audio', None):
        files = [Path(args.audio)]

    if not files:
        print("[ERROR] No input files. Use --audio or --inputs or --input-glob", file=sys.stderr)
        sys.exit(1)

    # 只建立一次 decoder，整批沿用
    dec = create_decoder(args)

    # 批次跑
    results = []
    for pth in files:
        try:
            out = run_one_file(pth, args, dec)
            row = {**out['metrics']}
            row['hyp'] = out['text']
            results.append(row)
        except Exception as e:
            print(f"[ERROR] {pth} failed: {e}", file=sys.stderr)
            # 不中斷，繼續下一檔
            results.append({
                'file': str(pth),
                'duration_sec': '',
                'tokens': '',
                'avg_prob': '',
                'WER': '',
                'CER': '',
            })

    # 可選評估
    if args.ref_dir:
        for row in results:
            if not row.get('hyp'):
                continue
            stem = Path(row['file']).stem
            refp = Path(args.ref_dir) / f"{stem}.txt"
            if refp.exists():
                with open(refp, 'r', encoding='utf-8') as f:
                    ref = f.read()
                unit = args.eval_unit
                wer, _ = compute_error_rate(row['hyp'], ref, unit='word' if unit in ('word','auto') else unit)
                cer, _ = compute_error_rate(row['hyp'], ref, unit='char' if unit in ('char','auto') else unit)
                row['WER'] = f"{wer:.4f}"
                row['CER'] = f"{cer:.4f}"
            else:
                row['WER'] = ''
                row['CER'] = ''

    # 寫 summary CSV
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = Path(args.out_dir) / "summary.csv"
    fieldnames = ['file', 'duration_sec', 'tokens', 'avg_prob', 'WER', 'CER']
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in fieldnames})

    print(f"[DONE] Wrote {summary_path}")



if __name__ == '__main__':
    main()
