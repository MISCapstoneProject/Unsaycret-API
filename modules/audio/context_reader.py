from __future__ import annotations

import os
import numpy as np
import soundfile as sf


def _load_wav_mono_float32(path: str, target_sr: int) -> np.ndarray:
    """Load wav, resample if needed, downmix to mono float32."""
    try:
        audio, sr = sf.read(path, always_2d=True)
        audio = audio.mean(axis=1)  # mono
        if sr != target_sr:
            # lightweight linear resample to avoid extra deps
            import numpy as _np

            x = _np.arange(len(audio))
            new_len = int(round(len(audio) * (target_sr / sr)))
            xp = _np.linspace(0, len(audio) - 1, new_len)
            audio = _np.interp(xp, x, audio).astype("float32")
        else:
            audio = audio.astype("float32")
        return audio
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _slice_or_pad(a: np.ndarray, start_samp: int, end_samp: int) -> np.ndarray:
    """Return a[start:end], padding zeros when indices exceed bounds."""
    n = len(a)
    left = max(0, start_samp)
    right = min(n, end_samp)
    mid = a[left:right] if right > left else np.zeros(0, dtype=np.float32)
    pad_l = max(0, -start_samp)
    pad_r = max(0, end_samp - n)
    if pad_l or pad_r:
        mid = np.pad(mid, (pad_l, pad_r), mode="constant")
    return mid.astype("float32")


def _read_seg(base_dir: str, seg_idx: int, speaker_file: str, sr: int) -> np.ndarray:
    """Load speaker wav under base_dir/segment_{idx}/speakerX.wav."""
    seg_dir = os.path.join(base_dir, f"segment_{seg_idx:03d}")
    path = os.path.join(seg_dir, speaker_file)
    return _load_wav_mono_float32(path, sr)


def fetch_audio_with_context(
    base_dir: str,
    seg_idx: int,
    speaker_file: str,
    win_sec: float,
    ctx_sec: float,
    sr: int,
) -> np.ndarray:
    """
    Assemble an audio buffer for decoding window with symmetric context.

    Logical window for segment seg_idx is [t0,t1) = [seg_idx*win_sec, (seg_idx+1)*win_sec).
    Returns audio for [t0-ctx_sec, t1+ctx_sec) by stitching neighboring segments.
    """
    import numpy as np

    cur = _read_seg(base_dir, seg_idx, speaker_file, sr)
    prev = (
        _read_seg(base_dir, seg_idx - 1, speaker_file, sr) if seg_idx > 0 else np.zeros(0, dtype=np.float32)
    )
    nxt = _read_seg(base_dir, seg_idx + 1, speaker_file, sr)

    win_samps = int(round(win_sec * sr))
    ctx_samps = int(round(ctx_sec * sr))

    prev_tail = (
        _slice_or_pad(prev, len(prev) - ctx_samps, len(prev)) if len(prev) > 0 else np.zeros(ctx_samps, dtype=np.float32)
    )
    cur_full = _slice_or_pad(cur, 0, win_samps)
    nxt_head = _slice_or_pad(nxt, 0, ctx_samps) if len(nxt) > 0 else np.zeros(ctx_samps, dtype=np.float32)

    buf = np.concatenate([prev_tail, cur_full, nxt_head], axis=0).astype("float32")

    target_len = ctx_samps + win_samps + ctx_samps
    if len(buf) != target_len:
        buf = _slice_or_pad(buf, 0, target_len)
    return buf
