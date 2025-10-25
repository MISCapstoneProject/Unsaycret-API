# convert_wav.py
# 批次把資料夾內的 .wav 轉成 16kHz / mono / PCM_16
# 1) 優先用 ffmpeg（若已安裝且在 PATH）
# 2) 找不到 ffmpeg 時，自動改用 librosa+soundfile 純 Python 轉

import os
import sys
import shutil
import subprocess
from pathlib import Path

import soundfile as sf
import numpy as np

# 只有在沒有 ffmpeg 時才會用到 librosa
try:
    import librosa
except Exception:
    librosa = None

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def convert_with_ffmpeg(src: Path, dst: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"⚠️ ffmpeg 轉檔失敗：{src.name} → {e}")
        return False

def convert_with_python(src: Path, dst: Path) -> bool:
    if librosa is None:
        print("❌ 沒安裝 librosa，且找不到 ffmpeg，無法轉檔。請先：pip install librosa soundfile")
        return False
    try:
        # 讀檔（讓 soundfile 把任意PCM/float/24bit都吃進來，再轉）
        y, sr = sf.read(str(src), always_2d=False, dtype="float32")
        # y 可能是多聲道
        if y.ndim == 2:
            y = y.mean(axis=1)  # 轉 mono
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
            sr = 16000
        # 正規化到安全區間（避免溢位）
        y = np.clip(y, -1.0, 1.0)
        # 寫成 PCM_16
        sf.write(str(dst), y, sr, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"❌ 純 Python 轉檔失敗：{src.name} → {e}")
        return False

def convert_wav_folder(input_dir=".", output_dir="fix", recursive=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.wav" if recursive else "*.wav"
    wav_files = list(input_dir.glob(pattern))
    if not wav_files:
        print(f"❌ 在 {input_dir.resolve()} 找不到 .wav 檔")
        return 1

    use_ffmpeg = has_ffmpeg()
    print(f"🔧 找到 {len(wav_files)} 個音檔，模式：{'ffmpeg' if use_ffmpeg else '純 Python'}\n")

    ok = 0
    for wav in wav_files:
        out_file = output_dir / f"{wav.stem}_16k_mono_s16.wav"
        try:
            if use_ffmpeg:
                succ = convert_with_ffmpeg(wav, out_file)
            else:
                succ = convert_with_python(wav, out_file)
            if succ:
                ok += 1
                print(f"✅ {wav.name} → {out_file.name}")
            else:
                print(f"⚠️ 失敗：{wav.name}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"⚠️ 例外：{wav.name} → {e}")

    print(f"\n🎉 完成。成功 {ok}/{len(wav_files)}，輸出：{output_dir.resolve()}")
    return 0 if ok == len(wav_files) else 2

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="將資料夾內 .wav 轉成 16kHz/mono/PCM_16")
    p.add_argument("--in_dir", default=".", help="輸入資料夾")
    p.add_argument("--out_dir", default="fix", help="輸出資料夾")
    p.add_argument("--recursive", action="store_true", help="是否遞迴處理子資料夾")
    args = p.parse_args()
    sys.exit(convert_wav_folder(args.in_dir, args.out_dir, args.recursive))
