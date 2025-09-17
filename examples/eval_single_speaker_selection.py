#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import json
import time
import shutil
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torchaudio

# --- 導入你的分離器 ---
try:
    # 你原本的模組路徑（若有 packages 結構）
    from modules.separation.separator import AudioSeparator, SeparationModel, TARGET_RATE
except Exception:
    # 直接在專案根目錄有 separator.py 的情況
    from separator import AudioSeparator, SeparationModel, TARGET_RATE  # type: ignore

# ---------- 小工具 ----------

def to_mono(wav: torch.Tensor) -> torch.Tensor:
    # [C,T] or [1,T] -> [T]
    if wav.ndim == 2:
        if wav.size(0) > 1:
            wav = wav.mean(0)
        else:
            wav = wav.squeeze(0)
    elif wav.ndim == 1:
        pass
    else:
        raise ValueError(f"Unexpected wav shape: {tuple(wav.shape)}")
    return wav

def resample_if_needed(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return wav
    return torchaudio.functional.resample(wav, sr_in, sr_out)

def si_sdr_db(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    # Scale-Invariant SDR
    ref = ref - ref.mean()
    est = est - est.mean()
    dot = torch.dot(est, ref)
    s_ref = torch.dot(ref, ref) + eps
    proj = (dot / s_ref) * ref
    noise = est - proj
    num = proj.pow(2).sum() + eps
    den = noise.pow(2).sum() + eps
    return 10.0 * torch.log10(num / den).item()

# ---------- 關鍵：呼叫你給的 separate_and_save ----------

@torch.inference_mode()
def run_separator_with_save(
    sep: AudioSeparator,
    clean_wav_path: Path,
    tmp_out_dir: Path,
) -> Tuple[Path, float]:
    """
    呼叫 AudioSeparator.separate_and_save(...)
    回傳 (選中的輸出檔路徑, 該檔與乾淨音的 SI-SDR dB)
    """
    wav, sr = torchaudio.load(str(clean_wav_path))
    wav = to_mono(wav)
    wav = resample_if_needed(wav, sr, TARGET_RATE)

    # 你的 separate_and_save 期望輸入 [B,T]
    audio_tensor = wav.unsqueeze(0).to(torch.float32).to(sep.device)

    # 每個輸入建一個臨時資料夾，避免覆寫
    utt_out = tmp_out_dir / clean_wav_path.stem
    utt_out.mkdir(parents=True, exist_ok=True)

    # 以單段方式呼叫（segment_index=0）
    results = sep.separate_and_save(
        audio_tensor=audio_tensor,
        output_dir=str(utt_out),
        segment_index=0,
        absolute_start_time=None,  # 讓函式自行帶入台北時區時間
    )
    if not results:
        raise RuntimeError("separate_and_save 回傳空列表（未產生輸出）。")

    # 單人情境 + 你的單人選路器：最後只會存一個「speaker1.wav」
    out_spk1 = utt_out / "speaker1.wav"
    if not out_spk1.exists():
        # 容錯：若命名不同，抓第一個結果檔
        out_spk1 = Path(results[0][0])

    # 計算和乾淨音的 SI-SDR
    est, sr_est = torchaudio.load(str(out_spk1))
    est = to_mono(est)
    if sr_est != TARGET_RATE:
        est = resample_if_needed(est, sr_est, TARGET_RATE)

    # 長度對齊（以較短者為準）
    L = min(est.numel(), wav.numel())
    sdr = si_sdr_db(ref=wav[:L], est=est[:L])
    return out_spk1, sdr

# ---------- 主流程：批次測試 ----------

def iter_clean_wavs(root: Path) -> List[Path]:
    """
    走訪你的資料結構：
      clean/
        speaker1/speaker1_01.wav ... speaker1_20.wav
        speaker2/...
        ...
    會自動忽略資料夾中的非 wav 檔與「VAT」等子資料夾
    """
    wavs: List[Path] = []
    for spk_dir in sorted(root.glob("speaker*")):
        if not spk_dir.is_dir():
            continue
        for wav_path in sorted(spk_dir.glob("*.wav")):
            # 忽略 VAT 目錄下的 wav（若有）
            if "VAT" in wav_path.parts:
                continue
            wavs.append(wav_path)
    return wavs

def main(
    data_root: str = "clean",
    work_dir: str = "eval_outputs",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    si_sdr_threshold: float = 8.0,  # 超過門檻視為「選對」
    use_2spk_model: bool = True,    # 固定用 2 人模型；若你有動態模型也可改成 False
):
    data_root = Path(data_root).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()
    out_audio_dir = work_dir / "sep_wavs"
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    # 建立/載入你的分離器
    model_type = SeparationModel.SEPFORMER_2SPEAKER if use_2spk_model \
                 else SeparationModel.SEPFORMER_3SPEAKER
    sep = AudioSeparator(
        model_type=model_type,
        enable_dynamic_model=False,    # 單人測試，固定 2spk 較穩定
        enable_noise_reduction=True,   # 依你目前設定
    )

    wav_list = iter_clean_wavs(data_root)
    if not wav_list:
        print(f"[ERR] 在 {data_root} 找不到任何 wav。")
        sys.exit(1)

    print(f"[INFO] 將評測 {len(wav_list)} 段單人語音，裝置: {device}")
    ok, fail = 0, 0
    per_spk_stats: Dict[str, Dict[str, float]] = {}
    t0 = time.time()

    for idx, wav_path in enumerate(wav_list, 1):
        spk = wav_path.parent.name
        try:
            out_path, sdr = run_separator_with_save(sep, wav_path, out_audio_dir / spk)
            is_ok = sdr >= si_sdr_threshold
            ok += int(is_ok)
            fail += int(not is_ok)

            # 收斂每位說話者平均
            d = per_spk_stats.setdefault(spk, {"sum": 0.0, "n": 0})
            d["sum"] += sdr
            d["n"] += 1

            print(f"[{idx:03d}/{len(wav_list)}] {spk}/{wav_path.stem} -> "
                  f"SI-SDR={sdr:.2f} dB  => {'✅' if is_ok else '❌'}   {out_path.name}")

        except Exception as e:
            fail += 1
            print(f"[{idx:03d}/{len(wav_list)}] {spk}/{wav_path.stem} -> 失敗：{e}")

    dur = time.time() - t0
    acc = ok / (ok + fail) if (ok + fail) else 0.0

    # 每位說話者平均
    per_spk_avg = {k: (v["sum"] / max(1, v["n"])) for k, v in per_spk_stats.items()}

    # 存結果
    summary = {
        "data_root": str(data_root),
        "device": device,
        "model": str(model_type.value),
        "threshold_db": si_sdr_threshold,
        "total": ok + fail,
        "ok": ok,
        "fail": fail,
        "accuracy": acc,
        "per_speaker_avg_si_sdr_db": per_spk_avg,
        "seconds": dur,
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n====== 單人選路器 測試總結 ======")
    print(f"總檔數: {summary['total']} | 準確率: {summary['accuracy']*100:.1f}% "
          f"(OK={ok}, FAIL={fail}) | 門檻: {si_sdr_threshold:.1f} dB | 花費 {dur:.1f}s")
    for spk, val in sorted(per_spk_avg.items()):
        print(f"  - {spk}: 平均 SI-SDR {val:.2f} dB")
    print(f"\n詳細 JSON：{work_dir/'summary.json'}")
    print(f"分離後音檔：{out_audio_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Evaluate single-speaker auto-route on clean set")
    ap.add_argument("--data_root", type=str, default="data/clean", help="clean 資料夾根目錄")
    ap.add_argument("--work_dir", type=str, default="eval_outputs", help="輸出根目錄")
    ap.add_argument("--device", type=str, default="cuda", help="cuda / cuda:0 / cpu")
    ap.add_argument("--thr", type=float, default=5.0, help="SI-SDR 視為正確的門檻(dB)")
    ap.add_argument("--use_2spk_model", action="store_true", help="強制用 2 人分離模型")
    args = ap.parse_args()

    dev = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    main(
        data_root=args.data_root,
        work_dir=args.work_dir,
        device=dev,
        si_sdr_threshold=args.thr,
        use_2spk_model=args.use_2spk_model or True,  # 單人測試建議固定 2spk
    )
