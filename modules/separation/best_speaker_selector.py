from __future__ import annotations
import math
from typing import Iterable, List, Tuple, Dict

import torch


class SingleSpeakerSelector:
    """
    單人情境（實際只有 1 位說話者、但模型輸出 2 條以上聲道）之自動選路器（V2）。

    與第一版不同，本版以 **SI-SDR 對 mix 的投影分數** 當主特徵，
    輔以：
      • band_ratio : 300–3400 Hz 的能量佔比（人聲帶寬）
      • tonality   : 1 - spectral_flatness（越接近 1 代表越有音調/非白噪）
      • zcr_penalty: 零交越率過高的懲罰（抑制嘶聲、寬頻噪）

    綜合分數：
        score = 0.6 * sig_sisdr + 0.25 * band_ratio + 0.15 * tonality - 0.10 * zcr_penalty
    其中 sig_sisdr = tanh(si_sdr_db / 10) ，把 dB 壓到 [0,1) 範圍。

    使用：
        selector = SingleSpeakerSelector(sr=16000)
        idx, best, stats = selector.select([s1, s2], mix, return_stats=True)

    注意：若環境極度安靜/吵雜，可微調 zcr 與 band_ratio 權重。
    """

    def __init__(
        self,
        sr: int = 16000,
        frame_ms: int = 20,
        hop_ms: int = 10,
        alpha: float = 1.5,      # 僅供能量式 VAD 使用（統計 & zcr 時用到）
        min_rms: float = 1e-6,
        w_sisdr: float = 0.6,
        w_band: float = 0.25,
        w_tonality: float = 0.15,
        w_zcr_penalty: float = 0.10,
        tie_tol: float = 0.02,
    ) -> None:
        self.sr = sr
        self.frame_len = max(1, int(sr * frame_ms / 1000))
        self.hop = max(1, int(sr * hop_ms / 1000))
        self.alpha = alpha
        self.min_rms = min_rms
        self.w_sisdr = w_sisdr
        self.w_band = w_band
        self.w_tonality = w_tonality
        self.w_zcr_penalty = w_zcr_penalty
        self.tie_tol = tie_tol

        # 頻帶遮罩（300–3400 Hz）
        self._band_lo = 300.0
        self._band_hi = 3400.0

    # ---------------------------
    # public API
    # ---------------------------
    @torch.no_grad()
    def select(
        self,
        candidates: Iterable[torch.Tensor],
        mix: torch.Tensor,
        return_stats: bool = False,
    ) -> Tuple[int, torch.Tensor, List[Dict[str, float]] | None]:
        cands: List[torch.Tensor] = [self._ensure_1d_cpu_f32(c) for c in candidates]
        mix = self._ensure_1d_cpu_f32(mix)
        if len(cands) == 0:
            raise ValueError("candidates 為空，無法選擇")
        if len(cands) == 1:
            return 0, cands[0], ([self._metrics(cands[0], mix)] if return_stats else None)

        # 對齊長度
        min_len = min([c.numel() for c in cands] + [mix.numel()])
        cands = [c[:min_len] for c in cands]
        mix = mix[:min_len]

        # 靜音/崩壞先排除
        rmss = [self._rms(c) for c in cands]
        valid = [i for i, r in enumerate(rmss) if r >= self.min_rms]
        if len(valid) == 1:
            idx = valid[0]
            return idx, cands[idx], ([self._metrics(cands[idx], mix)] if return_stats else None)
        if len(valid) == 0:
            idx = int(torch.tensor(rmss).argmax().item())
            return idx, cands[idx], ([self._metrics(cands[idx], mix)] if return_stats else None)

        # 計分
        stats: List[Dict[str, float]] = []
        scores: List[float] = []
        for c in cands:
            m = self._metrics(c, mix)
            stats.append(m)
            score = (
                self.w_sisdr * m["sig_sisdr"]
                + self.w_band * m["band_ratio"]
                + self.w_tonality * m["tonality"]
                - self.w_zcr_penalty * m["zcr_penalty"]
            )
            m["score"] = float(score)
            scores.append(float(score))

        best_idx = int(torch.tensor(scores).argmax().item())

        # 平手處理
        sorted_scores = sorted(scores, reverse=True)
        if len(sorted_scores) >= 2 and math.isfinite(sorted_scores[0]) and math.isfinite(sorted_scores[1]):
            if abs(sorted_scores[0] - sorted_scores[1]) <= self.tie_tol:
                # 若分數相近，取 SI-SDR 較高者，仍平手則用 RMS
                sisdrs = [s["si_sdr_db"] for s in stats]
                best_idx = int(torch.tensor(sisdrs).argmax().item())
                if len({sisdrs[i] for i in range(len(sisdrs))}) == 1:
                    best_idx = int(torch.tensor(rmss).argmax().item())

        return best_idx, cands[best_idx], (stats if return_stats else None)

    # ---------------------------
    # internals
    # ---------------------------
    def _ensure_1d_cpu_f32(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 1:
            x = x.squeeze()
        if x.is_cuda:
            x = x.cpu()
        return x.to(dtype=torch.float32, copy=False)

    @staticmethod
    def _rms(x: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(x ** 2) + 1e-12))

    # --- 窗口分幀、能量 VAD（供統計/特徵 gating 用） ---
    def _frame_signal(self, x: torch.Tensor) -> torch.Tensor:
        T = x.numel()
        if T < self.frame_len:
            pad = torch.zeros(self.frame_len - T, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
            T = x.numel()
        n_frames = 1 + max(0, (T - self.frame_len) // self.hop)
        if n_frames <= 0:
            n_frames = 1
        idx = torch.arange(self.frame_len).unsqueeze(0) + torch.arange(n_frames).unsqueeze(1) * self.hop
        return x[idx]

    def _short_time_energy(self, x: torch.Tensor) -> torch.Tensor:
        frames = self._frame_signal(x)
        ste = (frames ** 2).mean(dim=1) + 1e-10
        return ste

    def _speech_mask(self, ste: torch.Tensor) -> torch.Tensor:
        thr = torch.median(ste) * self.alpha
        return ste > thr

    # --- SI-SDR vs mix（投影法） ---
    def _sisdr_to_mix_db(self, s: torch.Tensor, mix: torch.Tensor) -> float:
        eps = 1e-10
        dot = torch.dot(mix, s)
        s_energy = torch.dot(s, s) + eps
        proj = (dot / s_energy) * s  # s 的最佳縮放投影到 mix
        e_noise = mix - proj
        num = torch.sum(proj ** 2)
        den = torch.sum(e_noise ** 2) + eps
        si_sdr = 10.0 * torch.log10((num + eps) / den)
        return float(si_sdr)

    # --- 頻帶能量佔比（300–3400 Hz） ---
    def _band_energy_ratio(self, x: torch.Tensor) -> float:
        # 使用 rFFT，估計整段能量分布
        X = torch.fft.rfft(x)
        power = (X.real ** 2 + X.imag ** 2)
        freqs = torch.fft.rfftfreq(x.numel(), d=1.0 / self.sr)
        band_mask = (freqs >= self._band_lo) & (freqs <= self._band_hi)
        num = float(power[band_mask].sum() + 1e-12)
        den = float(power.sum() + 1e-12)
        return num / den

    # --- 光譜平坦度（Wiener entropy），取 1 - flatness 作為 tonality ---
    def _tonality(self, x: torch.Tensor, speech_mask: torch.Tensor | None) -> float:
        frames = self._frame_signal(x)
        if speech_mask is not None:
            # 以能量式 VAD 的語音幀為主
            ste = (frames ** 2).mean(dim=1) + 1e-12
            thr = torch.median(ste) * self.alpha
            keep = ste > thr
            if keep.any():
                frames = frames[keep]
        # rFFT 每幀
        X = torch.fft.rfft(frames, dim=1)
        power = (X.real ** 2 + X.imag ** 2) + 1e-12
        log_power = torch.log(power)
        gm = torch.exp(log_power.mean(dim=1))          # geometric mean
        am = power.mean(dim=1)                          # arithmetic mean
        flat = torch.clamp(gm / am, min=1e-6, max=1.0) # [0,1]
        tonality = 1.0 - float(flat.mean())
        return tonality

    # --- 零交越率（ZCR）懲罰 ---
    def _zcr_penalty(self, x: torch.Tensor) -> float:
        frames = self._frame_signal(x)
        # 中心化避免 DC 影響
        frames = frames - frames.mean(dim=1, keepdim=True)
        signs = torch.sign(frames)
        signs[signs == 0] = 1
        zc = (signs[:, 1:] * signs[:, :-1] < 0).float().sum(dim=1) / (frames.shape[1] - 1 + 1e-9)
        # 取語音幀（能量式 VAD）中的中位數 ZCR
        ste = (frames ** 2).mean(dim=1)
        mask = ste > (torch.median(ste) * self.alpha)
        if mask.any():
            zc = zc[mask]
        med = float(zc.median()) if zc.numel() > 0 else float(zc.mean())
        # 期望語音 ZCR 大致 0.05~0.15；高於 0.2 視為嘶聲/寬頻噪。
        if med <= 0.12:
            return 0.0
        # 線性映射到 [0,1] 的懲罰
        return float(min(1.0, max(0.0, (med - 0.12) / 0.3)))

    def _metrics(self, c: torch.Tensor, mix: torch.Tensor) -> Dict[str, float]:
        # 語音偵測用的能量遮罩（僅用作 tonality、zcr 的 gating）
        ste = self._short_time_energy(c)
        sp_mask = self._speech_mask(ste)

        si_sdr_db = self._sisdr_to_mix_db(c, mix)             # 可能 < 0 ~ 20+ dB
        sig_sisdr = float(torch.tanh(torch.tensor(si_sdr_db / 10.0)))  # 壓縮到 (0,1)
        band_ratio = self._band_energy_ratio(c)               # [0,1]
        tonality = self._tonality(c, sp_mask)                 # 越大越像人聲
        zcr_penalty = self._zcr_penalty(c)                    # 越大越像嘶聲/噪

        return {
            "si_sdr_db": float(si_sdr_db),
            "sig_sisdr": float(sig_sisdr),
            "band_ratio": float(band_ratio),
            "tonality": float(tonality),
            "zcr_penalty": float(zcr_penalty),
            "rms": self._rms(c),
        }
