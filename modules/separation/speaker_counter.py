from __future__ import annotations
import math
import logging
import torch
from typing import Optional, Tuple, List, Dict

try:
    from pyannote.audio import Pipeline
except Exception:
    Pipeline = None  # 延遲導入處理


class SpeakerCounter:
    """
    基於 pyannote 說話者分離管道構建的強健說話者計數工具。

    功能特色
    --------
    - 管道快取與明確的設備配置
    - 形狀安全的單聲道轉換 + 自動正規化（處理 int16/float）
    - 重疊感知的後處理來過濾虛假標籤
    - 雙重策略：寬鬆通過 → （可選）強制 2 說話者重試
    - 輕量級 VAD（相對於峰值的 dBFS 中的 RMS + ZCR）用於備用決策
    - **DEBUG 日誌掛鉤** 來檢查每個決策步驟
    """

    # 預設閾值（根據您的語料庫調整）
    MIN_ABS_DUR = 0.60            # 秒；捨棄短於此時間的標籤
    MIN_REL_DUR = 0.10            # 總語音聯集持續時間的比例
    MIN_NON_OVERLAP = 0.25        # 在組合中作為軟性保護
    SECOND_PASS_NON_OVERLAP = 0.22

    # 重疊信用
    OVERLAP_ALPHA = 0.50          # 有效 = 非重疊 + ALPHA * 重疊
    REL_MIN_STRONG = 0.18         # 有效相對值的強重疊接受度

    # VAD 閾值
    VAD_MIN_DBFS = -46.0          # 相對於峰值的 dBFS
    VAD_MIN_ZCR = 0.02
    VAD_MAX_ZCR = 0.25
    
    # 絕對 dBFS 門檻（對 full-scale 1.0）
    VAD_ABS_DBFS_MIN = -46.0      # 幀/全段的最低聲能
    VAD_LOUD_DBFS    = -40.0      # 至少若干幀須超過此能量，避免底噪誤判

    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        pipeline_name: str = "pyannote/speaker-diarization-3.1",
        pipeline: Optional[object] = None,
        logger: Optional[logging.Logger] = None,
        log_prefix: str = "SpeakerCounter",
    ) -> None:
        self.hf_token = hf_token
        self.device = device
        self.pipeline_name = pipeline_name
        self._pipeline = pipeline  # 允許依賴注入/重複使用
        self.logger = logger or logging.getLogger(log_prefix)
        self.log_prefix = log_prefix

    # -------------------------- 公開 API --------------------------
    def count(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        debug: bool = False,
    ) -> int:
        diar = self._run_diarization(audio, sample_rate, min_speakers, max_speakers, debug)
        if diar is None or not hasattr(diar, "labels"):
            self._dbg(debug, "count: 分離返回 None/無效 → 0")
            return 0
        n = int(len(set(list(diar.labels()))))
        self._dbg(debug, f"count: 原始不同標籤 = {n}")
        return n

    def count_with_refine(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        expected_min: Optional[int] = None,
        expected_max: Optional[int] = None,
        first_pass_range: Tuple[int, int] | None = None,
        debug: bool = False,
        allow_zero: bool = False,
        min_voiced_ratio: float = 0.06,   # 至少 6% 的語音幀
        min_voiced_union: float = 0.30,   # 語音聯集至少 0.30 秒
    ) -> int:
        """
        雙重通過計數，包含重疊感知優化和必要時的強制 2 說話者重試。
        """
        if first_pass_range is None:
            first_pass_range = (1, 3)

        # --- 靜音閘控（只有允許 0 人時才啟用） ---
        if allow_zero:
            ok, vad = self._has_voice(audio, sample_rate, debug=debug, return_metrics=True)
            self._dbg(debug, f"VAD 閘控: ok={ok}, dbfs={vad['dbfs']:.1f}, zcr={vad['zcr']:.3f}, "
                             f"ratio={vad['voiced_ratio']:.3f}, union={vad['voiced_union']:.2f}s")
            if (not ok) or (vad["voiced_ratio"] < min_voiced_ratio) or (vad["voiced_union"] < min_voiced_union):
                self._dbg(debug, "→ 靜音: 返回 0（跳過分離）")
                return 0

        self._dbg(debug, f"count_with_refine: first_pass_range={first_pass_range}, expected=({expected_min},{expected_max})")

        # 1) 第一次通過（寬鬆）
        min_spk = max(1, first_pass_range[0])
        max_spk = first_pass_range[1]
        diar = self._run_diarization(audio, sample_rate, min_spk, max_spk, debug=debug)
        if diar is None:
            has_voice = self._has_voice(audio, sample_rate, debug=debug)
            self._dbg(debug, f"第一次通過 diar=None, has_voice={has_voice}")
            return 1 if has_voice else 0

        keep, stats = self._refine_labels_by_overlap(diar, debug=debug)
        self._dbg(debug, f"優化後: keep={keep} | labels={list(stats.keys())}")

        # NEW: 如果第一次只看到兩個人，但形態很像「全程重疊的多人」
        raw_distinct = len(set(list(diar.labels()))) if hasattr(diar, "labels") else None
        def _eff(s):  # 有效時長
            return s["non_overlap"] + self.OVERLAP_ALPHA * s["overlap"]

        if (expected_max or 3) >= 3 and raw_distinct is not None and raw_distinct < 3 and len(keep) == 2:
            # 兩人的有效佔比約各一半，且 non_overlap 幾乎為 0 → 疑似漏了第 3 人
            eff_map = {k: _eff(v) for k, v in stats.items()}
            eff_sum = sum(eff_map.values()) + 1e-8
            top2 = sorted(keep, key=lambda k: eff_map[k], reverse=True)[:2]
            rels = [eff_map[k] / eff_sum for k in top2]
            nonovs = [stats[k]["non_overlap"] for k in top2]
            suspicious_three = (0.35 <= min(rels) <= max(rels) <= 0.65) and (max(nonovs) <= 0.10)
            self._dbg(debug, f"可能是 3語者? rels={rels}, nonov(top2)={nonovs} -> {suspicious_three}")

            if suspicious_three:
                accepted_retry = False
                diar3 = self._run_diarization(audio, sample_rate, 3, 3, debug=debug)
                if diar3 is not None:
                    keep3, stats3 = self._refine_labels_by_overlap(diar3, debug=debug)
                    if len(keep3) >= 3:
                        keep, stats = keep3, stats3
                        self._dbg(debug, "可能是 3語者? → retry(3,3) accepted: 3 labels kept")
                        accepted_retry = True
                    else:
                        eff_items = sorted(stats3.items(), key=lambda kv: _eff(kv[1]), reverse=True)
                        if len(eff_items) >= 3:
                            eff_total3 = sum(_eff(s) for _, s in eff_items) + 1e-8
                            third_eff_rel = _eff(eff_items[2][1]) / eff_total3
                            if third_eff_rel >= self.REL_MIN_STRONG:  # 例如 0.18
                                keep = [eff_items[0][0], eff_items[1][0], eff_items[2][0]]
                                stats = {k: v for k, v in stats3.items()}
                                self._dbg(debug, f"可能是 3語者? → retry(3,3) accepted by rel3={third_eff_rel:.3f}")
                                accepted_retry = True

                # ★ 就在這裡接 KMeans(3) 後備（等價於你問的那段 else 的位置）
                if not accepted_retry:
                    # (3,3) 仍只有 2 或 diar3 失敗 → 嘗試 KMeans(3) 後備
                    k3 = self._kmeans_3_fallback(audio, sample_rate, debug=debug)
                    if k3 == 3:
                        self._dbg(debug, "fallback KMeans(3) → set n=3")
                        return 3  # 直接回傳 3，結束流程

        # --- 有效分數和相對值的輔助函數
        def eff_of(k: str) -> float:
            s = stats[k]
            return s["non_overlap"] + self.OVERLAP_ALPHA * s["overlap"]

        total_union = sum((stats[k]["total"] for k in stats))  # 僅用於日誌；真正的聯集在優化內部使用
        
        # 如果所有項目都被過濾但有語音 → 保留有效的前 2 名（加上基本總時長保護）
        if not keep and stats and self._has_voice(audio, sample_rate, debug=debug):
            ranked = sorted(stats.items(), key=lambda kv: (kv[1]["non_overlap"] + self.OVERLAP_ALPHA * kv[1]["overlap"]), reverse=True)
            keep = [lab for lab, s in ranked[:2] if (s["total"] >= self.MIN_ABS_DUR)]
            self._dbg(debug, f"按有效值（前2名）的備用保留: keep={keep}")

        # 如果只剩 1 個但我們仍然清楚有語音 → 按有效值填充到 2
        if len(keep) == 1 and stats and self._has_voice(audio, sample_rate, debug=debug):
            ranked = sorted(stats.items(), key=lambda kv: (kv[1]["non_overlap"] + self.OVERLAP_ALPHA * kv[1]["overlap"]), reverse=True)
            for lab, s in ranked:
                if lab not in keep and s["total"] >= self.MIN_ABS_DUR:
                    keep.append(lab)
                    break
            self._dbg(debug, f"按有效值填充到 2: keep={keep}")

        # 如果資料集已知為 2 說話者，按有效值（不僅僅是 non_overlap）限制為 2
        if expected_max == 2 and len(keep) > 2:
            keep = sorted(keep, key=lambda k: eff_of(k), reverse=True)[:2]
            self._dbg(debug, f"按有效值限制為 2: keep={keep}")

        # 2) 如果只剩 1 個但預期 ≥2 → 強制 2 說話者重試
        if (expected_min or 0) >= 2 and len(keep) < 2:
            self._dbg(debug, "使用 (2,2) 強制重試")
            diar2 = self._run_diarization(audio, sample_rate, 2, 2, debug=debug)
            if diar2 is not None:
                keep2, stats2 = self._refine_labels_by_overlap(diar2, debug=debug)

                # 重試備用：有效值前 2 名
                if len(keep2) < 2 and stats2:
                    ranked2 = sorted(stats2.items(), key=lambda kv: (kv[1]["non_overlap"] + self.OVERLAP_ALPHA * kv[1]["overlap"]), reverse=True)
                    keep2 = [lab for lab, s in ranked2[:2] if s["total"] >= self.MIN_ABS_DUR]
                    self._dbg(debug, f"重試備用按有效值（前2名）保留: keep2={keep2}")

                if len(keep2) >= 2:
                    # 如果第二個有足夠的 non_overlap 或強有效相對值則接受
                    # 用來自 diar2 的所有項目聯集計算有效相對值
                    total_union2 = sum(v["total"] for v in stats2.values()) + 1e-8
                    top2 = sorted(keep2, key=lambda k: (stats2[k]["non_overlap"] + self.OVERLAP_ALPHA * stats2[k]["overlap"]), reverse=True)[:2]
                    second = top2[1]
                    eff2 = stats2[second]["non_overlap"] + self.OVERLAP_ALPHA * stats2[second]["overlap"]
                    rel2 = eff2 / total_union2
                    if (stats2[second]["non_overlap"] >= self.SECOND_PASS_NON_OVERLAP) or (rel2 >= self.REL_MIN_STRONG):
                        keep = top2
                        stats = stats2
                        self._dbg(debug, f"重試接受 top2={top2}（第二個: nonov={stats2[second]['non_overlap']:.2f}s, rel_eff={rel2:.3f}）")
                    else:
                        self._dbg(debug, f"重試拒絕: 第二個 nonov={stats2[second]['non_overlap']:.2f}s < {self.SECOND_PASS_NON_OVERLAP} 且 rel_eff={rel2:.3f} < {self.REL_MIN_STRONG}")

        n = len(keep)
        if expected_max is not None and n > expected_max:
            keep = sorted(keep, key=lambda k: eff_of(k), reverse=True)[:expected_max]
            n = len(keep)
            self._dbg(debug, f"裁剪到 expected_max={expected_max}: keep={keep}")
        if expected_min is not None and n < expected_min:
            self._dbg(debug, f"提高到 expected_min={expected_min}（原有 {n}）")
            n = expected_min

        self._dbg(debug, f"最終: n={n}")
        return int(n)

    # ------------------------- 內部方法 ----------------------------
    def _lazy_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if Pipeline is None:
            raise RuntimeError("pyannote.audio 不可用")
        if not self.hf_token:
            raise RuntimeError("pyannote 管道缺少 HF 令牌")
        pipe = Pipeline.from_pretrained(self.pipeline_name, use_auth_token=self.hf_token)
        if self.device:
            pipe.to(torch.device(self.device))
        self._pipeline = pipe
        return self._pipeline

    def _run_diarization(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        debug: bool = False,
    ):
        if sample_rate is None:
            raise ValueError("需要 sample_rate")
        wav = self._to_mono_1d(audio, debug=debug)
        dur = wav.numel() / float(sample_rate)
        self._dbg(debug, f"執行分離: dur={dur:.2f}s, min={min_speakers}, max={max_speakers}")
        if dur <= 0.0:
            return None
        pipe = self._lazy_pipeline()
        with torch.no_grad():
            diar = pipe({"waveform": wav.unsqueeze(0).cpu().float(), "sample_rate": int(sample_rate)},
                        min_speakers=min_speakers, max_speakers=max_speakers)
        if hasattr(diar, "labels"):
            labs = list(diar.labels())
            self._dbg(debug, f"執行分離: 原始標籤={labs}（不同={len(set(labs))}）")
        else:
            self._dbg(debug, "執行分離: diar 沒有 labels()")
        return diar

    def _to_mono_1d(self, x: torch.Tensor, debug: bool = False, for_vad: bool = False) -> torch.Tensor:
        x = x.detach().to(dtype=torch.float32, device="cpu")
        pre_peak = float(torch.max(torch.abs(x)).item()) if x.numel() > 0 else 0.0
        if x.ndim == 2:
            x = x.mean(dim=0) if x.shape[0] <= x.shape[1] else x.mean(dim=1)
        elif x.ndim != 1:
            raise ValueError(f"Unsupported waveform shape: {tuple(x.shape)}")

        max_abs = float(torch.max(torch.abs(x))) if x.numel() > 0 else 0.0
        scale = 1.0
        if max_abs > 2.0:  # int16 → float
            scale = 32768.0
            x = x / scale
        elif (0.0 < max_abs < 0.01) and (not for_vad):
            # 只有在「非 VAD」路徑才會做極小訊號增益
            scale = max_abs
            x = x / scale

        x = x - x.mean()          # 去 DC
        x = torch.clamp(x, -1.0, 1.0)
        post_peak = float(torch.max(torch.abs(x)).item()) if x.numel() > 0 else 0.0
        self._dbg(debug, f"to_mono: pre_peak={pre_peak:.4f}, norm_scale={scale:.4f}, post_peak={post_peak:.4f}{' [VAD]' if for_vad else ''}")
        return x

    # ---------------------- 後處理 ------------------------
    def _refine_labels_by_overlap(self, diar, debug: bool = False):
        """使用區間聯集數學計算精確非重疊的標籤統計。
        返回 (保留標籤, 統計字典)，其中 stats[標籤] 有鍵值：total, non_overlap, overlap。
        """
        from pyannote.core import Segment
        by_lab: Dict[str, List[Segment]] = {}
        for seg, _, lab in diar.itertracks(yield_label=True):
            by_lab.setdefault(lab, []).append(seg)

        def _merge_and_length(intervals: List[Tuple[float, float]]) -> float:
            if not intervals:
                return 0.0
            intervals = sorted(intervals)
            cur_s, cur_e = intervals[0]
            total = 0.0
            for s, e in intervals[1:]:
                if s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    total += (cur_e - cur_s)
                    cur_s, cur_e = s, e
            total += (cur_e - cur_s)
            return max(0.0, total)

        all_intervals: List[Tuple[float, float]] = []
        for segs in by_lab.values():
            for seg in segs:
                all_intervals.append((float(seg.start), float(seg.end)))
        total_speech_union = _merge_and_length(all_intervals) + 1e-8

        stats: Dict[str, Dict[str, float]] = {}
        for lab, segs in by_lab.items():
            total = 0.0
            non_overlap = 0.0
            for seg in segs:
                s_i, e_i = float(seg.start), float(seg.end)
                dur_i = max(0.0, e_i - s_i)
                total += dur_i
                overlaps: List[Tuple[float, float]] = []
                for other_lab, other_segs in by_lab.items():
                    if other_lab == lab:
                        continue
                    for seg_j in other_segs:
                        s_j, e_j = float(seg_j.start), float(seg_j.end)
                        s = max(s_i, s_j)
                        e = min(e_i, e_j)
                        if e > s:
                            overlaps.append((s, e))
                overlapped = _merge_and_length(overlaps)
                non_overlap += max(0.0, dur_i - overlapped)
            overlap = max(0.0, total - non_overlap)
            stats[lab] = {"total": total, "non_overlap": non_overlap, "overlap": overlap}

        keep: List[str] = []
        for lab, s in stats.items():
            effective = s["non_overlap"] + self.OVERLAP_ALPHA * s["overlap"]
            rel = effective / total_speech_union

            # 基本：總時長至少 self.MIN_ABS_DUR
            if s["total"] < self.MIN_ABS_DUR:
                self._dbg(debug, f"標籤={lab}: 捨棄 | total<{self.MIN_ABS_DUR:.2f}s")
                continue

            # 主要判斷：有效相對時長達標即可保留（強重疊情境）。
            # 若不足，則用較弱條件：rel >= MIN_REL_DUR 且 non_overlap >= 0.05s
            if (rel >= self.REL_MIN_STRONG) or (rel >= self.MIN_REL_DUR and s["non_overlap"] >= 0.05):
                keep.append(lab)
                self._dbg(debug, f"標籤={lab}: 保留 | total={s['total']:.2f}s nonov={s['non_overlap']:.2f}s ov={s['overlap']:.2f}s eff={effective:.2f}s rel={rel:.3f}")
            else:
                self._dbg(debug, f"標籤={lab}: 捨棄 | rel={rel:.3f} < {self.REL_MIN_STRONG:.2f} 且 nonov<0.05s")

        return keep, stats

    # ---------------------- 輕量級 VAD ------------------------
    @staticmethod
    def _rms_dbfs_peak(x: torch.Tensor) -> float:
        x = x.detach().float()
        if x.ndim > 1:
            x = x.mean(dim=-1)
        x = x - x.mean()
        peak = float(torch.max(torch.abs(x)).item()) if x.numel() > 0 else 0.0
        if peak < 1e-8:
            return -120.0
        rms = float(torch.sqrt(torch.mean(x ** 2)).item())
        rms = max(rms, 1e-12)
        return 20.0 * math.log10(rms / peak)

    @staticmethod
    def _zcr(x: torch.Tensor, frame_len: int, hop: int) -> float:
        x = x.detach().float()
        if x.ndim > 1:
            x = x.mean(dim=-1)
        n = x.numel()
        if n < frame_len + 1:
            return 0.0
        total = 0.0
        cnt = 0
        for i in range(0, n - frame_len, hop):
            f = x[i:i+frame_len]
            total += float((f[1:] * f[:-1] < 0).float().mean().item())
            cnt += 1
        return total / max(1, cnt)

    def _rms_dbfs_abs(self, x: torch.Tensor) -> float:
        """相對 full-scale 的絕對 dBFS（x ∈ [-1,1]）。"""
        x = x.detach().float()
        if x.ndim > 1: x = x.mean(dim=-1)
        if x.numel() == 0: return -120.0
        rms = float(torch.sqrt(torch.mean(x ** 2)).item())
        if rms <= 0.0: return -120.0
        return 20.0 * math.log10(rms)  # full-scale 1.0

    def _has_voice(self, audio: torch.Tensor, sr: int, debug: bool = False, return_metrics: bool = False):
        # 重要：VAD 專用，不做小訊號增益
        w = self._to_mono_1d(audio, debug=debug, for_vad=True)

        # 全段特徵
        rms_db_abs = self._rms_dbfs_abs(w)
        zcr = self._zcr(w, frame_len=int(0.025*sr), hop=int(0.010*sr))

        # 逐幀統計（30ms/15ms）
        frame = int(0.030 * sr)
        hop = int(0.015 * sr)
        n = w.numel()
        voiced_frames = 0
        loud_frames = 0
        for i in range(0, max(0, n - frame + 1), hop):
            f = w[i:i+frame]
            if f.numel() == 0: continue
            # 絕對 dBFS（不再相對峰值）
            frms = float(torch.sqrt(torch.mean(f**2)).item())
            dbfs_abs = -120.0 if frms <= 1e-12 else 20.0 * math.log10(frms)
            z = float((f[1:] * f[:-1] < 0).float().mean().item())
            if (dbfs_abs > self.VAD_ABS_DBFS_MIN) and (self.VAD_MIN_ZCR <= z <= self.VAD_MAX_ZCR):
                voiced_frames += 1
            if dbfs_abs > self.VAD_LOUD_DBFS:
                loud_frames += 1

        total_frames = max(1, (n - frame) // hop + 1)
        voiced_ratio = voiced_frames / total_frames
        voiced_union = voiced_frames * (hop / sr)  # 近似語音聯集
        loud_frac = loud_frames / total_frames

        ratio_union_ok = (voiced_ratio >= 0.20) and (voiced_union >= 1.00)
        
        energy_ok = (rms_db_abs > self.VAD_ABS_DBFS_MIN) and (loud_frames >= 3)
        ok = (self.VAD_MIN_ZCR <= zcr <= self.VAD_MAX_ZCR) and (energy_ok or ratio_union_ok)

        self._dbg(debug, f"VAD(abs): {rms_db_abs:.1f} dBFS, ZCR={zcr:.3f}, voiced_ratio={voiced_ratio:.3f}, "
                        f"voiced_union={voiced_union:.2f}s, loud_frac={loud_frac:.3f} -> {'VOICE' if ok else 'NOISE/SIL'}")

        if return_metrics:
            return ok, {"dbfs": rms_db_abs, "zcr": zcr, "voiced_ratio": voiced_ratio, "voiced_union": voiced_union, "loud_frac": loud_frac}
        return ok

    # ---------------------- KMeans(3) fallback ----------------------
    def _frame_vad_mask(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        """逐幀 VAD：回傳 [T] 的 0/1 mask（True=有聲）。對齊 30ms/15ms，center=False。"""
        frame = int(0.030 * sr)  # 30ms
        hop = int(0.015 * sr)    # 15ms
        n = w.numel()
        if n < frame:
            return torch.zeros(0, dtype=torch.bool, device=w.device)

        masks = []
        for i in range(0, n - frame + 1, hop):  # ← 重點：+1，包含最後一窗
            f = w[i:i+frame]
            if f.numel() == 0:
                masks.append(False); continue
            peak = float(torch.max(torch.abs(f)).item())
            dbfs = -120.0 if peak < 1e-8 else 20.0 * math.log10(float(torch.sqrt(torch.mean(f**2)).item()) / peak)
            z = float((f[1:] * f[:-1] < 0).float().mean().item())
            ok = (dbfs > self.VAD_MIN_DBFS) and (self.VAD_MIN_ZCR <= z <= self.VAD_MAX_ZCR)
            masks.append(ok)
        return torch.tensor(masks, dtype=torch.bool, device=w.device)

    def _stft_logmel(self, w: torch.Tensor, sr: int, n_mels: int = 40) -> torch.Tensor:
        """回傳 [T, n_mels] 的 log-mel 特徵（對齊 30ms/15ms 幀，center=False）。"""
        win_length = int(0.030 * sr)   # 30ms
        hop_length = int(0.015 * sr)   # 15ms
        n_fft = 1
        while n_fft < win_length:      # 取最近的 2 次冪
            n_fft <<= 1

        window = torch.hann_window(win_length, device=w.device, dtype=w.dtype)
        spec = torch.stft(
            w, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=False, return_complex=True  # ← 重點：center=False
        )
        mag = (spec.real**2 + spec.imag**2)          # [freq, T]
        mag = mag.transpose(0, 1).contiguous()       # [T, freq]

        # --- 建 mel 濾波器（三角形），全程用相同 device/dtype ---
        def hz_to_mel(f): return 2595.0 * math.log10(1.0 + f / 700.0)
        def mel_to_hz(m): return 700.0 * (10.0**(m / 2595.0) - 1.0)

        fmin, fmax = 20.0, sr / 2.0
        mmin, mmax = hz_to_mel(fmin), hz_to_mel(fmax)
        m_pts = torch.linspace(mmin, mmax, n_mels + 2, device=mag.device, dtype=torch.float32)
        f_pts = torch.tensor([mel_to_hz(m) for m in m_pts.tolist()], device=mag.device, dtype=torch.float32)

        bins = torch.floor(((n_fft // 2) * f_pts / fmax)).long().clamp(0, n_fft // 2).tolist()

        fb = torch.zeros(n_mels, n_fft // 2 + 1, device=mag.device, dtype=mag.dtype)
        for i in range(n_mels):
            left  = int(bins[i])
            center = int(bins[i + 1])
            right = int(bins[i + 2])

            if center <= left:
                center = left + 1
            if right <= center:
                right = center + 1

            rise_steps = center - left
            fb[i, left:center] = torch.linspace(0.0, 1.0, steps=rise_steps, device=fb.device, dtype=fb.dtype)

            fall_steps = right - center
            fb[i, center:right] = torch.linspace(1.0, 0.0, steps=fall_steps, device=fb.device, dtype=fb.dtype)

        mel = torch.matmul(mag[:, : n_fft // 2 + 1], fb.t())  # [T, n_mels]
        mel = torch.clamp(mel, min=1e-10)
        logmel = mel.log()
        mean = logmel.mean(dim=0, keepdim=True)
        std = logmel.std(dim=0, keepdim=True).clamp_min(1e-5)
        logmel = (logmel - mean) / std
        return logmel

    def _kmeans_torch(self, X: torch.Tensor, k: int, iters: int = 20, seed: int = 0):
        """簡易 KMeans：輸入 [N, D]，回傳 (centroids[k,D], labels[N])。"""
        g = torch.Generator().manual_seed(seed)
        N = X.size(0)
        # 初始化：隨機挑 k 個樣本
        idx = torch.randperm(N, generator=g)[:k]
        C = X[idx].clone()  # [k, D]
        for _ in range(iters):
            # 指派
            d2 = torch.cdist(X, C, p=2)  # [N, k]
            lbl = torch.argmin(d2, dim=1)
            # 更新
            newC = []
            for j in range(k):
                sel = X[lbl == j]
                if sel.numel() == 0:
                    # 若空群，隨機重抽一個點
                    ridx = torch.randint(0, N, (1,), generator=g).item()
                    newC.append(X[ridx: ridx+1])
                else:
                    newC.append(sel.mean(dim=0, keepdim=True))
            newC = torch.cat(newC, dim=0)
            if torch.allclose(newC, C, atol=1e-4):
                C = newC; break
            C = newC
        return C, lbl

    def _kmeans_3_fallback(self, audio: torch.Tensor, sr: int, debug: bool = False) -> int:
        """
        在可疑三人時的最終備援：
        - 取有聲幀的 log-mel 特徵 → KMeans(3)
        - 若第三群佔比 >= 0.15 且分群品質達標，回傳 3，否則 0（不介入）
        """
        w = self._to_mono_1d(audio, debug=debug)

        # 先算 log-mel（center=False），再算 VAD mask（同樣窗口/跳步）
        logmel = self._stft_logmel(w, sr)       # [T, n_mels]
        mask = self._frame_vad_mask(w, sr)      # [T]

        T_logmel = logmel.size(0)
        T_vad = mask.numel()
        if T_logmel != T_vad:
            L = min(T_logmel, T_vad)
            self._dbg(debug, f"kmeans3: align frames logmel_T={T_logmel}, vad_T={T_vad} -> {L}")
            logmel = logmel[:L, :]
            mask = mask[:L]
            T_logmel = L

        if T_logmel < 30:
            self._dbg(debug, "kmeans3: too few frames → skip")
            return 0
        if mask.float().mean().item() < 0.2:
            self._dbg(debug, "kmeans3: too sparse voiced frames → skip")
            return 0

        X = logmel[mask]
        if X.size(0) < 30:
            self._dbg(debug, "kmeans3: too few voiced frames after mask → skip")
            return 0

        # 分 3 群
        C, y = self._kmeans_torch(X, k=3, iters=25, seed=42)
        counts = torch.stack([(y == j).sum() for j in range(3)]).float()
        props = (counts / counts.sum()).tolist()

        # 簡單品質：最小質心距離 / 平均群內標準差
        with torch.no_grad():
            dcc = torch.cdist(C, C, p=2) + torch.eye(3, device=C.device) * 1e9
            min_centroid_dist = float(dcc.min().item())
            intra = []
            for j in range(3):
                sel = X[y == j]
                intra.append(float(sel.std(dim=0).norm().item()) + 1e-6 if sel.numel() else 1e-6)
            avg_intra = sum(intra) / 3.0
            quality = min_centroid_dist / avg_intra

        self._dbg(debug, f"kmeans3: props={props}, quality={quality:.3f}")

        third_prop = sorted(props, reverse=True)[2]
        if third_prop >= 0.15 and quality >= 0.12:
            self._dbg(debug, "kmeans3: accept → return 3")
            return 3
        self._dbg(debug, "kmeans3: reject")
        return 0
    
    # -------------------------- 日誌記錄 ----------------------------
    def _dbg(self, debug: bool, msg: str) -> None:
        if debug and self.logger:
            self.logger.debug(f"[{self.log_prefix}] {msg}")
