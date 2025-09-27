#!/usr/bin/env python3
"""
fast_view.py — 極簡 UI 模擬器（每秒更新：時間 + 文字）

用途：
- 以與 fast_test.py 相同的解碼/聚合邏輯跑整段音檔
- 只顯示「時間」與「目前合併後文字（VIEW）」
- 不輸出任何除錯細節、JSONL、切除原因等
"""

from __future__ import annotations
import argparse
import time
import sys
import shutil
from pathlib import Path

# ==== 從 fast_test.py 匯入需要的元件（請確保同資料夾） ====
try:
    from modules.asr.Fast_Test import (
        Aggregator,
        FileStreamSimulator,
        FastDecoder,
        tokens_to_text,
    )
except Exception as e:
    print("[ERROR] 請將 fast_view.py 與 fast_test.py 放在同一個資料夾，或確認模組可被匯入。", file=sys.stderr)
    raise

# ---- 小工具：清畫面 ----
def clear_screen():
    # ANSI 清屏，Windows 10+ 也支援；若 PowerShell 沒開啟 VT，退化成多列換行
    try:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()
    except Exception:
        print("\n" * 50)

def detect_device_auto() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def parse_args():
    p = argparse.ArgumentParser(description="Minimal FAST viewer (time + text only)")
    p.add_argument("--audio", type=str, required=True, help="Path to audio file (wav/mp3/flac)")

    # 解碼與語言
    p.add_argument("--model-name", type=str, default="medium", help="faster-whisper model (e.g., small, medium, large-v3)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    p.add_argument("--compute-type", type=str, default="float16", help="Compute type (e.g., float16, int8_float16, int8)")
    p.add_argument("--language", type=str, default="zh", help='Language hint (e.g., zh, en). Use "none" to auto-detect')
    p.add_argument("--beam-size", type=int, default=3, help="Beam size (FAST default=3)")

    # FAST 參數
    p.add_argument("--window-len", type=float, default=3.0, help="Window length in seconds")
    p.add_argument("--stride", type=float, default=1.0, help="Stride (update interval) in seconds")
    p.add_argument("--commit-tail-sec", type=float, default=1.6, help="Visible tail duration in seconds")
    p.add_argument("--epsilon", type=float, default=0.001, help="Timestamp epsilon")

    # 視窗保護策略（保持與 fast_test 的視窗裁切一致，避免觀感落差）
    p.add_argument("--guard-sec", type=float, default=0.4, help="Accept only tokens inside [win_start+guard, win_end-guard]")
    p.add_argument("--protect-head-sec", type=float, default=0.6, help="Do NOT replace the first X seconds of current TAIL (each window)")
    p.add_argument("--final-protect-sec", type=float, default=0.6, help="Final window: protect earlier part of TAIL")

    # 螢幕輸出
    p.add_argument("--no-clear", action="store_true", help="不要清屏，只是逐行印出（除錯用）")
    p.add_argument("--fit-width", action="store_true", help="視窗自動截斷到終端寬度（避免過長換行）")
    return p.parse_args()

def main():
    args = parse_args()
    device = detect_device_auto() if args.device == "auto" else args.device
    lang = None if (isinstance(args.language, str) and args.language.lower() in {"none", "auto", "null"}) else args.language

    # 構建核心元件
    dec = FastDecoder(args.model_name, device, args.compute_type, args.beam_size, lang)
    sim = FileStreamSimulator(path=Path(args.audio), window_len=args.window_len, stride=args.stride)  # type: ignore
    agg = Aggregator(commit_tail_sec=args.commit_tail_sec, epsilon=args.epsilon)

    sim_time = args.window_len  # 與 fast_test 一致：從第一個完整窗結束時刻開始

    # 主迴圈（每秒更新）
    while not sim.done():
        window = sim.tick()
        if window is None:
            break

        # 計算視窗位置與邊界
        win_start = max(0.0, sim_time - args.window_len)
        win_end = sim_time
        is_last_window = (sim_time + 1e-6 >= sim.duration)
        is_first_window = (win_start <= 1e-6)

        # 左界
        if is_first_window:
            accept_s = 0.0
        elif win_start < args.guard_sec:
            accept_s = 0.0
        else:
            accept_s = win_start + args.guard_sec
        accept_s = max(accept_s, agg.state.last_committed_end - args.epsilon)

        # 右界
        accept_e = win_end if is_last_window else (win_end - args.guard_sec)

        # 解碼 + 視窗篩選
        toks, _rtf = dec.decode_window(window, abs_start_time=max(0.0, sim_time - args.window_len))
        mid_toks = [t for t in toks if (t.start >= accept_s - args.epsilon and t.end <= accept_e + args.epsilon)]

        # 保護 TAIL 頭段（避免剛接上的文字被新窗立刻覆蓋）
        if agg.state.tail and args.protect_head_sec > 0:
            tail_head = agg.state.tail[0].start
            cutoff = max(tail_head + args.protect_head_sec, agg.state.last_committed_end + 0.005, accept_s)
            mid_toks = [t for t in mid_toks if t.start >= cutoff - args.epsilon]

        # 最後一窗：只允許「尾巴」被替換
        if is_last_window and agg.state.tail and args.final_protect_sec > 0:
            tail_end = agg.state.tail[-1].end
            protect_start = max(accept_s, tail_end - args.final_protect_sec)
            mid_toks = [t for t in mid_toks if t.start >= protect_start - args.epsilon]

        # 追加並更新可視文字
        agg.append_fast(mid_toks)
        view_text = tokens_to_text(agg.state.committed + agg.state.tail)

        # 終端顯示（極簡）
        if not args.no_clear:
            clear_screen()

        # 若需要截斷以符合終端寬度
        if args.fit_width:
            cols = shutil.get_terminal_size((100, 20)).columns
            if cols > 10:
                # 保留一些前綴空間，例如「time」那行
                view_text = (view_text[:cols] + "…") if len(view_text) > cols else view_text

        print(f"Time: {sim_time:5.1f}s")
        print(view_text)

        # 節奏：每次 stride 更新一次（模擬 UI 1Hz）
        time.sleep(max(0.0, args.stride))
        sim_time += args.stride

    # 結束時把 tail 推進 committed，最後印一次完整文字
    agg.finalize()
    final_text = tokens_to_text(agg.state.committed)
    if not args.no_clear:
        clear_screen()
    print(f"Time: END ({sim_time:5.1f}s)")
    print(final_text)

if __name__ == "__main__":
    main()
