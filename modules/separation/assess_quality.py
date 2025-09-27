import math
import torch
from typing import Dict, Any, Optional, Tuple

def assess_audio_quality(
    wave: torch.Tensor,
    sr: int,
    logger=None,
    vad_abs_dbfs_min: float = -46.0,   # 與 SpeakerCounter 對齊（可微調）
    vad_min_zcr: float = 0.02,
    vad_max_zcr: float = 0.25,
    loud_dbfs: float = -40.0,          # 用於「loud frames」統計
    frame_ms: float = 30.0,
    hop_ms: float = 15.0,
    ref_wave: Optional[torch.Tensor] = None,  # 若給參考，會另外回傳 si_sdr_db
) -> Dict[str, Any]:
    """
    根據單一路訊號估測音訊品質。輸入 wave 建議為 1D、float32、[-1,1] 範圍。
    回傳 metrics 與加權分數/等第，以及改善建議。
    """
    # --------- 準備與安全處理 ----------
    x = wave.detach().float().cpu()
    if x.ndim > 1:
        x = x.mean(dim=-1)
    n = x.numel()
    duration_s = n / max(1, sr)

    eps = 1e-10
    peak = float(x.abs().max().item()) if n else 0.0
    rms = float(torch.sqrt(torch.mean(x ** 2)).item()) if n else 0.0
    rms_dbfs = -120.0 if rms <= eps else 20.0 * math.log10(rms)  # 相對 full-scale=1.0
    crest_db = 0.0 if rms <= eps else 20.0 * math.log10(max(peak, eps) / rms)
    clipping_pct = float(((x >= 0.999) | (x <= -0.999)).float().mean().item()) if n else 0.0

    # --------- 逐幀特徵 ----------
    win = int(frame_ms / 1000.0 * sr)
    hop = int(hop_ms / 1000.0 * sr)
    if n < win:
        # 太短：回基本資訊
        base = {
            "duration_s": duration_s, "rms_dbfs": rms_dbfs, "peak": peak,
            "clipping_pct": clipping_pct, "crest_db": crest_db, "frames": 0
        }
        base.update({
            "snr_db_est": 0.0, "silence_ratio": 1.0, "zcr": 0.0,
            "spectral_flatness": 1.0, "tonality": 0.0,
            "spectral_centroid_hz": 0.0, "high_band_ratio": 0.0,
            "telephony_band_ratio": 0.0
        })
        score, grade, issues, suggestions = _score_quality(base)
        base.update({"quality_score": score, "grade": grade, "issues": issues, "suggestions": suggestions})
        if ref_wave is not None:
            base["si_sdr_db"] = _si_sdr(x, ref_wave, eps=eps)
        return base

    # STFT（center=False 與你現行管線一致）
    n_fft = 1
    while n_fft < win:
        n_fft <<= 1
    window = torch.hann_window(win)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                      window=window, center=False, return_complex=True)
    mag2 = (spec.real ** 2 + spec.imag ** 2)  # [F, T]
    F, T = mag2.shape

    # 頻率座標
    freqs = torch.linspace(0, sr / 2, steps=F, dtype=torch.float32)

    # 幀 RMS / ZCR / VAD
    frame_rms = []
    frame_zcr = []
    voiced = []
    loud_frames = 0
    for i in range(0, n - win + 1, hop):
        f = x[i:i+win]
        frms = float(torch.sqrt(torch.mean(f**2)).item())
        dbfs_abs = -120.0 if frms <= eps else 20.0 * math.log10(frms)
        z = float((f[1:] * f[:-1] < 0).float().mean().item())
        frame_rms.append(frms)
        frame_zcr.append(z)
        v = (dbfs_abs > vad_abs_dbfs_min) and (vad_min_zcr <= z <= vad_max_zcr)
        voiced.append(v)
        if dbfs_abs > loud_dbfs:
            loud_frames += 1
    frame_rms = torch.tensor(frame_rms)
    frame_zcr = torch.tensor(frame_zcr)
    voiced_mask = torch.tensor(voiced, dtype=torch.bool)
    total_frames = len(voiced)

    silence_ratio = float((~voiced_mask).float().mean().item()) if total_frames > 0 else 1.0
    mean_zcr = float(frame_zcr.mean().item()) if total_frames > 0 else 0.0

    # 估測 SNR（以分位數近似：P80 當語音、P20 當噪音）
    if total_frames > 0:
        speech_rms = float(torch.quantile(frame_rms, 0.80).item())
        noise_rms  = float(torch.quantile(frame_rms, 0.20).item())
        snr_db_est = 20.0 * math.log10(max(speech_rms, eps) / max(noise_rms, eps))
    else:
        snr_db_est = 0.0

    # 頻譜統計：Spectral Flatness（越低越「純音/尖銳」，越高越「雜訊/平」）
    pow_spec = mag2[:, :min(T, total_frames)] + 1e-12  # 避免 log(0)
    log_pow = torch.log(pow_spec)
    gm = torch.exp(log_pow.mean(dim=0))          # 幾何平均（逐幀）
    am = pow_spec.mean(dim=0)                    # 算術平均（逐幀）
    sfm = float((gm / (am + 1e-12)).mean().item())
    tonality = float(1.0 - sfm)                  # 直觀：越靠近 1 越有音高/結構

    # 頻譜質心 & 頻帶能量比
    # 只用正頻（已是），避免 DC 影響
    w_pow = pow_spec
    num = (w_pow.T * freqs).sum(dim=1)           # [T]，每幀 Σ f*P(f)
    den = w_pow.sum(dim=0) + 1e-12              # [T]
    centroid = float((num / den).mean().item())  # 平均頻譜質心 (Hz)

    # 頻帶能量：300–3400（電話帶）、>4k（高頻）
    def _band_ratio(f_lo: float, f_hi: float) -> float:
        idx = (freqs >= f_lo) & (freqs <= f_hi)
        band = w_pow[idx, :].sum()
        tot = w_pow.sum()
        return float((band / (tot + 1e-12)).item())
    telephony_band_ratio = _band_ratio(300.0, 3400.0)
    high_band_ratio      = _band_ratio(4000.0, sr/2.0)

    # 彙整
    metrics: Dict[str, Any] = {
        "duration_s": duration_s,
        "frames": total_frames,
        "rms_dbfs": rms_dbfs,
        "peak": peak,
        "clipping_pct": clipping_pct,
        "crest_db": crest_db,
        "snr_db_est": snr_db_est,
        "silence_ratio": silence_ratio,
        "zcr": mean_zcr,
        "spectral_flatness": sfm,
        "tonality": tonality,
        "spectral_centroid_hz": centroid,
        "high_band_ratio": high_band_ratio,
        "telephony_band_ratio": telephony_band_ratio,
        "loud_frac": float(loud_frames / max(1, total_frames)),
    }

    # 品質分數與等第
    score, grade, issues, suggestions = _score_quality(metrics)
    metrics.update({"quality_score": score, "grade": grade, "issues": issues, "suggestions": suggestions})

    # 可選：計 SI-SDR（若給參考）
    if ref_wave is not None:
        metrics["si_sdr_db"] = _si_sdr(x, ref_wave, eps=eps)

    if logger is not None:
        logger.debug(f"[AQ] grade={grade} score={score:.1f} issues={issues} "
                     f"| rms={rms_dbfs:.1f}dBFS snr={snr_db_est:.1f}dB "
                     f"crest={crest_db:.1f}dB clip={clipping_pct*100:.2f}% "
                     f"centroid={centroid:.0f}Hz tonality={tonality:.2f}")
    return metrics


# ----------------- 打分邏輯（可視語料微調） -----------------

def _score_quality(m: Dict[str, Any]) -> Tuple[float, str, list, list]:
    """
    以啟發式規則打分（100 滿分），並產生等第與建議。
    """
    score = 100.0
    issues = []
    sugg = []

    # 音量/能量
    if m["rms_dbfs"] < -50:
        score -= 25; issues.append("too_quiet"); sugg.append("靠近麥克風或提高輸入增益")
    elif m["rms_dbfs"] < -40:
        score -= 12; issues.append("quiet"); sugg.append("提高輸入增益或關閉自動增益限制器")

    # 剪波
    if m["clipping_pct"] > 0.02:
        score -= 30; issues.append("clipping"); sugg.append("降低輸入增益，避免紅燈/破音")
    elif m["clipping_pct"] > 0.005:
        score -= 10; issues.append("minor_clipping"); sugg.append("稍微降低輸入增益")

    # 動態範圍（crest factor）
    if m["crest_db"] < 6.0:
        score -= 10; issues.append("over_compressed"); sugg.append("關閉過度壓縮/限制器或降低降噪強度")
    elif m["crest_db"] > 24.0:
        score -= 6; issues.append("too_dynamic"); sugg.append("環境過於安靜，適度加入房間聲或調整增益")

    # 估測 SNR
    if m["snr_db_est"] < 8.0:
        score -= 20; issues.append("low_snr"); sugg.append("降低環境噪音或使用指向性麥克風")
    elif m["snr_db_est"] < 15.0:
        score -= 10; issues.append("mid_snr"); sugg.append("靠近收音源、調整麥距")

    # 靜音比例（以 VAD 粗估）
    if m["silence_ratio"] > 0.8 and m["duration_s"] >= 4.0:
        score -= 8; issues.append("mostly_silence"); sugg.append("確認錄音裝置/路徑是否正確")
    
    # 頻譜：清晰度/悶塞感
    if m["spectral_centroid_hz"] < 800:
        score -= 8; issues.append("muffled"); sugg.append("提高高頻（3–6 kHz）或調整麥克風角度")
    if m["high_band_ratio"] < 0.05:
        score -= 6; issues.append("lack_hf"); 
        if "提高高頻（3–6 kHz）或調整麥克風角度" not in sugg:
            sugg.append("提高高頻（3–6 kHz）或調整麥克風角度")
    if m["high_band_ratio"] > 0.45:
        score -= 6; issues.append("too_bright"); sugg.append("降低嘶聲/齒音（去齒器或高頻EQ）")

    # 口語帶有效能量（電話帶比）
    if m["telephony_band_ratio"] < 0.35:
        score -= 6; issues.append("narrow_band"); sugg.append("檢查頻寬受限的通訊鏈路或低通濾波")
    
    # ZCR/tonality（粗判噪音或失真）
    if m["zcr"] > 0.2:
        score -= 6; issues.append("noisy/fricatives"); sugg.append("使用降噪或換較低雜訊的環境")
    if m["tonality"] < 0.2:
        score -= 5; issues.append("no_tonality"); sugg.append("提高直達聲占比，減少混響")

    # 限制邊界
    score = max(0.0, min(100.0, score))
    grade = "good" if score >= 75 else ("ok" if score >= 55 else "poor")

    # 合併相同建議、保持順序
    seen = set(); uniq_sugg = []
    for s in sugg:
        if s not in seen:
            uniq_sugg.append(s); seen.add(s)

    return score, grade, issues, uniq_sugg


# -----------------（可選）SI-SDR 計算 -----------------

def _si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Scale-Invariant SDR（單通道）。若 est/ref 長度不同，截成一致最短。
    """
    e = est.detach().float().cpu()
    r = ref.detach().float().cpu()
    if e.ndim > 1: e = e.squeeze()
    if r.ndim > 1: r = r.squeeze()
    L = min(e.numel(), r.numel())
    if L == 0:
        return float("nan")
    e = e[:L]; r = r[:L]
    # 最小二乘投影
    alpha = torch.dot(e, r) / (torch.dot(r, r) + eps)
    s_target = alpha * r
    e_noise = e - s_target
    si_sdr = 10.0 * torch.log10((torch.sum(s_target**2) + eps) / (torch.sum(e_noise**2) + eps))
    return float(si_sdr.item())
