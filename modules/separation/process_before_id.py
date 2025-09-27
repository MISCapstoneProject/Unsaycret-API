import torch
import math
try:
    import torchaudio
    import torchaudio.functional as AF
except Exception:
    torchaudio, AF = None, None

def _to_mono_1d_cpu(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    if x.ndim > 1:
        x = x.mean(dim=-1)
    return x

def _fixed_rms_gain(x: torch.Tensor, target_dbfs: float = -23.0, eps: float = 1e-9) -> torch.Tensor:
    """把波形等比縮放到固定 dBFS（常數增益，不改動態）"""
    target_rms = 10.0 ** (target_dbfs / 20.0)  # 約 0.0708 @ -23 dBFS
    rms = torch.sqrt(torch.mean(x ** 2) + eps)
    return x * (target_rms / (rms + eps))

def _prep_id_audio(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """
    用於語者辨識的一致化前處理：
    - 單聲道、CPU
    - 高通 50 Hz、低通 7.6 kHz（保留語音主頻帶）
    - 固定 dBFS 正規化
    - 溫和限幅（peak 0.98）
    """
    # 1) 直流去除 + 輕度 pre-emphasis
    x = wav - wav.mean()
    x = torch.cat([x[:1], x[1:] - 0.97 * x[:-1]])  # pre-emphasis 係數 0.97

    # 2) 固定 RMS 到 ~ -23 dBFS（≈ 0.07），只做乘法常數縮放
    target_rms = 0.07
    rms = torch.sqrt(torch.mean(x**2) + 1e-12)
    x = x * (target_rms / max(rms, 1e-6))

    # 3) 安全限幅
    peak = x.abs().max().item()
    if peak > 0.99:
        x = x * (0.98 / peak)
    return x

def _gentle_blend(dry_sep: torch.Tensor, dry_mix: torch.Tensor, ratio: float = 0.08) -> torch.Tensor:
    """
    只給人聽的保真混合：分離結果 + 少量原混音。
    不用於語者辨識！
    """
    s = _to_mono_1d_cpu(dry_sep)
    m = _to_mono_1d_cpu(dry_mix)
    L = min(s.numel(), m.numel())
    y = (1.0 - ratio) * s[:L] + ratio * m[:L]
    # 限幅
    peak = y.abs().max()
    if peak > 0.98:
        y = y * (0.98 / peak)
    return y.contiguous()

def mixture_consistency_projection(estimates: torch.Tensor, mix: torch.Tensor, eps=1e-8):
    """
    estimates: [T, S] 或 [S, T] 皆可（自動偵測）
    mix:       [T]
    讓所有 source 的和，與 mix 一致：E' = E + (mix - sum(E)) / S
    """
    if estimates.ndim == 2 and estimates.shape[0] == mix.numel():
        # [T, S] -> [S, T]
        estimates = estimates.transpose(0, 1)
    # estimates: [S, T]
    resid = mix - estimates.sum(dim=0)
    estimates = estimates + resid.unsqueeze(0) / (estimates.shape[0] + eps)
    return estimates  # [S, T]

def stft_wiener_refine(estimates: torch.Tensor, mix: torch.Tensor,
                       n_fft=1024, hop=256, win_length=1024, wiener_p=0.8, eps=1e-8):
    """
    用 time-domain estimates 生成 soft mask，套到 MIX 的 STFT 上再 iSTFT。
    estimates: [S, T]，mix: [T]
    """
    if estimates.ndim == 2 and estimates.shape[1] == mix.numel():
        S, T = estimates.shape
    else:
        estimates = estimates.transpose(0, 1)  # [S, T]
        S, T = estimates.shape

    win = torch.hann_window(win_length, periodic=True, device=estimates.device)
    X = torch.stft(mix, n_fft=n_fft, hop_length=hop, win_length=win_length,
                   window=win, return_complex=True)  # [F, N]

    mags = []
    for s in range(S):
        Es = torch.stft(estimates[s], n_fft=n_fft, hop_length=hop, win_length=win_length,
                        window=win, return_complex=True)  # [F, N]
        mags.append(Es.abs().pow(wiener_p))
    mags = torch.stack(mags, dim=0)  # [S, F, N]
    denom = mags.sum(dim=0, keepdim=False) + eps  # [F, N]
    masks = mags / denom  # [S, F, N]

    Y = masks * X.unsqueeze(0)  # [S, F, N]
    outs = []
    for s in range(S):
        y = torch.istft(Y[s], n_fft=n_fft, hop_length=hop, win_length=win_length,
                        window=win, length=T)
        outs.append(y)
    outs = torch.stack(outs, dim=0)  # [S, T]

    # 最後再做一次 mixture consistency，避免疊加誤差
    outs_mc = mixture_consistency_projection(outs, mix)
    return outs_mc  # [S, T]

def fade_io(x: torch.Tensor, sr: int, fade_ms: float = 8.0):
    T = x.numel()
    L = max(1, int(sr * fade_ms / 1000.0))
    if L*2 >= T: 
        return x
    w = torch.linspace(0, 1, L)
    x[:L] *= w
    x[-L:] *= w.flip(0)
    return x

def tf_mask_refine(est_np: torch.Tensor,
                   mix_wave: torch.Tensor,
                   sr: int,
                   n_fft: int = 512,
                   hop: int = 128,
                   win_length: int = 512,
                   sharpen: float = 1.15,
                   mask_floor: float = 0.02,
                   iters: int = 2) -> torch.Tensor:
    """
    est_np: [S, T] (CPU)
    mix_wave: [T]  (CPU)
    回傳: [S, T] (CPU) —— TF 遮罩銳化 + 多次維納化後的波形
    """
    S, T = est_np.shape
    window = torch.hann_window(win_length)
    # mixture STFT
    X = torch.stft(mix_wave, n_fft=n_fft, hop_length=hop, win_length=win_length,
                   window=window, return_complex=True)
    # estimates STFT
    Y = []
    for s in range(S):
        Ys = torch.stft(est_np[s], n_fft=n_fft, hop_length=hop, win_length=win_length,
                        window=window, return_complex=True)
        Y.append(Ys)
    Y = torch.stack(Y, dim=0)  # [S, F, TT]

    # 反覆以 winner-take-most 收斂遮罩
    for _ in range(iters):
        mags = (Y.abs() ** sharpen)  # [S, F, TT]
        denom = mags.sum(dim=0, keepdim=True).clamp_min(1e-8)
        masks = (mags / denom).clamp(min=mask_floor, max=1.0)
        Y = masks * X  # 維納化更新為 mixture-consistent

    # iSTFT
    outs = []
    for s in range(S):
        ys = torch.istft(Y[s], n_fft=n_fft, hop_length=hop, win_length=win_length,
                         window=window, length=T)
        outs.append(ys)
    return torch.stack(outs, dim=0)  # [S, T]


def framewise_dominance_gate(est_np: torch.Tensor,
                             frame: int = 320,  # 20ms@16k
                             hop: int = 160,    # 10ms hop
                             rel_ratio: float = 0.2,  # 其他說話者低於主導者 20%（≈ -14 dB）就壓制
                             min_floor: float = 0.1,  # 壓制到 10%（≈ -20 dB）
                             fade: int = 80) -> torch.Tensor:
    """
    對每個短窗，將非主導說話者能量壓到 floor，並做輕微交疊平滑。
    est_np: [S, T] (CPU)
    回傳:   [S, T] (CPU)
    """
    S, T = est_np.shape
    out = est_np.clone()
    for t0 in range(0, T, hop):
        t1 = min(t0 + frame, T)
        seg = est_np[:, t0:t1]                      # [S, L]
        rms = torch.sqrt((seg**2).mean(dim=1) + 1e-12)  # [S]
        dom = torch.argmax(rms)
        dom_val = rms[dom].item() + 1e-12
        for s in range(S):
            if s == dom:
                continue
            if rms[s].item() < dom_val * rel_ratio:
                # 壓到 floor 並加個入/出淡化
                gain = min_floor
                seg_s = out[s, t0:t1]
                # 淡入淡出
                L = seg_s.numel()
                if L > 2*fade:
                    w = torch.ones(L)
                    ramp = torch.linspace(0, 1, steps=fade)
                    w[:fade] = torch.maximum(w[:fade], ramp)
                    w[-fade:] = torch.maximum(w[-fade:], ramp.flip(0))
                else:
                    w = torch.ones(L)
                out[s, t0:t1] = seg_s * (gain + (1-gain)*w)
    return out

def gentle_spectral_gate(wave: torch.Tensor, sr: int,
                         n_fft=1024, hop=256, win_length=1024,
                         floor=0.15, attn_db=6.0):
    # floor: 最低通過比例；attn_db: 估計噪聲時的保留量（愈小愈保守）
    window = torch.hann_window(win_length)
    S = torch.stft(wave, n_fft=n_fft, hop_length=hop, win_length=win_length,
                   window=window, return_complex=True)
    mag = S.abs()
    # 以時間 10% 分位數當噪聲底（min-statistics 的近似）
    noise = torch.quantile(mag, 0.10, dim=-1, keepdim=True)
    # 採用平滑 mask，避免顫抖
    mask = (mag - noise) / (mag + 1e-8)
    mask = mask.clamp(0, 1)
    gain = floor + (1.0 - floor) * mask
    # 也避免高頻細顆粒：對 6kHz 以上增一點點衰減
    freqs = torch.linspace(0, sr/2, mag.size(0), device=mag.device)
    hf = (freqs > 6000).float().unsqueeze(-1)
    gain = gain * (1 - hf * (1 - 10**(-attn_db/20.0)))
    S_f = S * gain
    out = torch.istft(S_f, n_fft=n_fft, hop_length=hop, win_length=win_length,
                      window=window, length=wave.numel())
    return out

def crosstalk_suppress(est: torch.Tensor, frame=320, hop=160, attn=0.25, smooth=5):
    # est: [S, T] in CPU
    S, T = est.shape
    gains = torch.ones(S, T)
    for t0 in range(0, T - frame, hop):
        sl = slice(t0, t0 + frame)
        e = est[:, sl].pow(2).mean(dim=1)  # [S]
        winner = int(torch.argmax(e))
        # 其他通道溫和衰減
        for s in range(S):
            if s != winner:
                gains[s, sl] *= attn
    # 時域平滑（中值濾波）
    if smooth > 1:
        import torch.nn.functional as F
        k = torch.ones(1, 1, smooth) / smooth
        for s in range(S):
            g = gains[s].unsqueeze(0).unsqueeze(0)
            gains[s] = F.conv1d(g, k, padding=smooth//2).squeeze()
    return est * gains.clamp(0.1, 1.0)

def _soft_spectral_floor(spec: torch.Tensor, floor_db: float = -34.0, knee_db: float = 10.0):
    """
    對 STFT 複數頻譜做「柔性地板」：把每一幀中低於 floor_db 的微弱雜訊柔和壓下，
    以避免硬閾值造成顆粒感。spec: [F, T] complex.
    """
    mag = spec.abs().clamp_min(1e-8)
    # 以每一幀的最大值作為參考（避免整段增益漂移）
    ref = mag.amax(dim=0, keepdim=True).clamp_min(1e-8)  # [1, T]
    rel_db = 20.0 * torch.log10(mag / ref)               # [F, T]
    # 在 floor_db ~ floor_db+knee_db 之間用平滑曲線過渡
    alpha = torch.clamp((rel_db - floor_db) / knee_db, 0.0, 1.0)
    alpha = alpha * alpha  # 稍微再柔一點
    return spec * alpha

def _hf_hiss_suppress(spec: torch.Tensor, sr: int, cutoff_hz: float = 6200.0, max_reduct_db: float = 8.0):
    """
    高頻自適應抑制：如果高頻能量相對中頻偏高（常見的分離殘留沙沙聲），
    就用一個很柔的高頻棚式遮罩，把 cutoff 以上逐步衰減（最多 max_reduct_db）。
    spec: [F, T] complex.
    """
    F, T = spec.shape
    device = spec.device
    # 頻率座標
    freqs = torch.linspace(0.0, sr/2, F, device=device)
    mag2 = (spec.real**2 + spec.imag**2) + 1e-12

    # 粗略估計整段「高頻/中頻」能量比例（用整段的平均，避免抽動）
    mid_band = (freqs >= 300.0) & (freqs <= 3400.0)
    hi_band  = (freqs >= cutoff_hz)
    mid_e = mag2[mid_band, :].mean()
    hi_e  = mag2[hi_band, :].mean()
    ratio = float((hi_e / mid_e).clamp_min(1e-8))

    # 比例高才啟動抑制；0.35 左右是人聲常見上限，再高就多半是殘留噪聲
    strength = max(0.0, min(1.0, (ratio - 0.35) / 0.35))  # 0→關閉, 1→滿額
    if strength <= 0.0:
        return spec

    reduct_db = strength * max_reduct_db
    # 造一個「非常柔」的高頻棚形狀（tanh 過渡）
    # 0 (低頻) → 1 (高頻)
    shelf = 0.5 * (1.0 + torch.tanh((freqs - cutoff_hz) / 1000.0))  # [F]
    # 將 dB 轉成線性增益
    g = torch.pow(10.0, (-reduct_db / 20.0))
    # 頻帶依比例漸進地逼近 g（低頻 ~1，高頻 ~g）
    band_gain = (1.0 - shelf) + shelf * g                          # [F]
    return spec * band_gain.unsqueeze(1)                             # [F,1]→[F,T]

def _tpdf_dither(x: torch.Tensor, level_db: float = -90.0):
    # x: [T] or [1,T]
    amp = 10.0 ** (level_db / 20.0)
    d = (torch.rand_like(x) - 0.5 + torch.rand_like(x) - 0.5) * amp
    return (x + d).clamp(-1.0, 1.0)