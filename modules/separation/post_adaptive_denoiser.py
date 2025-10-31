"""
===============================================================================
增強版自適應降噪模組 - 針對電子雜訊優化
===============================================================================

版本：v1.1.0 (Enhanced)
作者：EvanLo62
最後更新：2025-10-23

新增功能：
 • 電子雜訊抑制：專門處理分離模型產生的電子偽影
 • 高頻嘶嘶聲去除：針對 6-10kHz 的電子雜訊
 • 低頻隆隆聲過濾：移除 < 80Hz 的低頻噪音
 • 去混響處理：減少分離殘留的回音效應
 • 頻譜門檻優化：更激進的噪音閘控

相比 v1.0.0 的改進：
 + 新增專門的電子雜訊檢測器
 + 優化高頻段處理，更積極去除嘶嘶聲
 + 加入多頻段壓縮，平衡各頻段能量
 + 改進語音頻段保護，避免過度處理

===============================================================================
"""

import torch
import numpy as np
import noisereduce as nr
from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Optional, Tuple, Union, List
import logging

from utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveDenoiser:
    """
    增強版自適應降噪器 - 專門針對電子雜訊優化
    
    特別適用於：
    - 語音分離模型產生的電子偽影
    - 高頻嘶嘶聲（6-10kHz）
    - 低頻隆隆聲（< 80Hz）
    - 分離殘留物（crosstalk artifacts）
    """
    
    # 降噪策略閾值（相比基礎版更激進）
    SNR_THRESHOLDS = {
        'strong': 8.0,      # < 8dB: 強降噪（提高閾值）
        'medium': 15.0,     # 8-15dB: 中等降噪
        'light': 22.0,      # 15-22dB: 輕度降噪
        'skip': 22.0        # >= 22dB: 跳過降噪
    }
    
    # 降噪強度參數（更激進的設定）
    DENOISE_PARAMS = {
        'strong': {
            'prop_decrease': 0.95,      # 提高到 95%
            'stationary': False,
            'n_fft': 2048,
            'freq_mask_smooth_hz': 500,  # 降低平滑，更精確
            'time_mask_smooth_ms': 80
        },
        'medium': {
            'prop_decrease': 0.85,      # 提高到 85%
            'stationary': True,
            'n_fft': 2048,
            'freq_mask_smooth_hz': 600,
            'time_mask_smooth_ms': 80
        },
        'light': {
            'prop_decrease': 0.70,      # 提高到 70%
            'stationary': True,
            'n_fft': 1024,
            'freq_mask_smooth_hz': 500,
            'time_mask_smooth_ms': 50
        }
    }
    
    def __init__(
        self,
        device: str = 'cpu',
        speech_freq_range: Tuple[int, int] = (80, 7500),  # 調整語音範圍
        min_speech_energy: float = 0.001,
        smoothing_window: int = 5,
        wiener_beta: float = 0.015,           # 更激進的維納濾波
        enable_wiener: bool = True,
        enable_smoothing: bool = True,
        enable_hiss_removal: bool = True,     # 新增：嘶嘶聲去除
        enable_rumble_filter: bool = True,    # 新增：低頻過濾
        enable_dereverb: bool = False,        # 新增：去混響（較慢）
        hiss_threshold: float = 0.15,         # 嘶嘶聲檢測閾值
        aggressive_mode: bool = True          # 新增：激進模式
    ):
        """
        初始化增強版降噪器
        
        新增參數：
            enable_hiss_removal: 啟用高頻嘶嘶聲去除
            enable_rumble_filter: 啟用低頻隆隆聲過濾
            enable_dereverb: 啟用去混響處理（會增加處理時間）
            hiss_threshold: 嘶嘶聲檢測閾值（越低越激進）
            aggressive_mode: 激進模式，更大力度去除雜訊
        """
        self.device = device
        self.speech_freq_range = speech_freq_range
        self.min_speech_energy = min_speech_energy
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
        self.wiener_beta = wiener_beta
        self.enable_wiener = enable_wiener
        self.enable_smoothing = enable_smoothing
        
        # 新增功能開關
        self.enable_hiss_removal = enable_hiss_removal
        self.enable_rumble_filter = enable_rumble_filter
        self.enable_dereverb = enable_dereverb
        self.hiss_threshold = hiss_threshold
        self.aggressive_mode = aggressive_mode
        
        logger.info(
            f"AdaptiveDenoiserEnhanced 初始化 - 裝置: {device}, "
            f"激進模式: {aggressive_mode}, "
            f"嘶嘶聲去除: {enable_hiss_removal}, "
            f"低頻過濾: {enable_rumble_filter}, "
            f"去混響: {enable_dereverb}"
        )
    
    def estimate_snr(
        self, 
        audio: Union[torch.Tensor, np.ndarray], 
        sample_rate: int = 16000
    ) -> float:
        """估算 SNR（同基礎版）"""
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio
        
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()
        
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)
        
        energy = np.array([
            np.sum(audio_np[i:i+frame_length]**2)
            for i in range(0, len(audio_np) - frame_length, hop_length)
        ])
        
        if len(energy) == 0:
            return 0.0
        
        energy_threshold = np.percentile(energy, 30)
        speech_frames = energy > energy_threshold
        noise_frames = ~speech_frames
        
        if not np.any(speech_frames) or not np.any(noise_frames):
            return 10.0
        
        speech_power = np.mean(energy[speech_frames])
        noise_power = np.mean(energy[noise_frames])
        
        if noise_power < 1e-10:
            return 30.0
        
        snr_db = 10 * np.log10(speech_power / noise_power)
        return float(np.clip(snr_db, -10, 40))
    
    def detect_electronic_noise(
        self,
        audio_np: np.ndarray,
        sample_rate: int
    ) -> dict:
        """
        檢測電子雜訊特徵
        
        Returns:
            dict: {
                'has_hiss': bool,      # 是否有嘶嘶聲
                'has_rumble': bool,    # 是否有低頻噪音
                'hiss_level': float,   # 嘶嘶聲強度
                'rumble_level': float  # 低頻噪音強度
            }
        """
        # 頻譜分析
        f, Pxx = signal.welch(audio_np, sample_rate, nperseg=1024)
        
        # 檢測高頻嘶嘶聲（6-10kHz）
        hiss_band = (f >= 6000) & (f <= 10000)
        speech_band = (f >= 300) & (f <= 3400)
        
        hiss_energy = np.mean(Pxx[hiss_band]) if np.any(hiss_band) else 0
        speech_energy = np.mean(Pxx[speech_band]) if np.any(speech_band) else 1e-10
        hiss_ratio = hiss_energy / speech_energy
        
        # 檢測低頻隆隆聲（< 80Hz）
        rumble_band = f < 80
        rumble_energy = np.mean(Pxx[rumble_band]) if np.any(rumble_band) else 0
        rumble_ratio = rumble_energy / speech_energy
        
        return {
            'has_hiss': hiss_ratio > self.hiss_threshold,
            'has_rumble': rumble_ratio > 0.3,
            'hiss_level': float(hiss_ratio),
            'rumble_level': float(rumble_ratio)
        }
    
    def _select_strategy(self, snr: float) -> str:
        """根據 SNR 選擇策略（同基礎版）"""
        if snr < self.SNR_THRESHOLDS['strong']:
            return 'strong'
        elif snr < self.SNR_THRESHOLDS['medium']:
            return 'medium'
        elif snr < self.SNR_THRESHOLDS['light']:
            return 'light'
        else:
            return 'skip'
    
    def _apply_spectral_gate(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        strategy: str
    ) -> np.ndarray:
        """應用頻譜閘控降噪（同基礎版）"""
        params = self.DENOISE_PARAMS[strategy]
        
        denoised = nr.reduce_noise(
            y=audio_np,
            sr=sample_rate,
            stationary=params['stationary'],
            prop_decrease=params['prop_decrease'],
            n_fft=params['n_fft'],
            freq_mask_smooth_hz=params['freq_mask_smooth_hz'],
            time_mask_smooth_ms=params['time_mask_smooth_ms']
        )
        
        return denoised
    
    def _remove_hiss(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        hiss_level: float
    ) -> np.ndarray:
        """
        專門去除高頻嘶嘶聲
        
        使用頻域濾波器，針對 6-10kHz 範圍進行抑制
        """
        # STFT
        nperseg = 512
        noverlap = nperseg // 2
        
        f, t, stft = signal.stft(
            audio_np,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # 建立頻率遮罩
        hiss_band = (f >= 6000) & (f <= 10000)
        
        # 根據嘶嘶聲強度決定抑制程度
        if hiss_level > 0.5:
            suppression = 0.1  # 強抑制，保留 10%
        elif hiss_level > 0.3:
            suppression = 0.3  # 中度抑制，保留 30%
        else:
            suppression = 0.5  # 輕度抑制，保留 50%
        
        # 應用抑制
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        magnitude[hiss_band, :] *= suppression
        
        # 平滑過渡（避免突變）
        transition_bins = 5
        for i in range(len(hiss_band)):
            if hiss_band[i] and i > 0:
                for j in range(1, transition_bins + 1):
                    if i - j >= 0 and not hiss_band[i - j]:
                        blend = j / (transition_bins + 1)
                        magnitude[i - j, :] *= (1 - blend) + blend * suppression
                        break
        
        # 重建
        stft_filtered = magnitude * np.exp(1j * phase)
        _, audio_filtered = signal.istft(
            stft_filtered,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # 確保長度一致
        if len(audio_filtered) > len(audio_np):
            audio_filtered = audio_filtered[:len(audio_np)]
        elif len(audio_filtered) < len(audio_np):
            audio_filtered = np.pad(
                audio_filtered,
                (0, len(audio_np) - len(audio_filtered)),
                mode='constant'
            )
        
        return audio_filtered
    
    def _remove_rumble(
        self,
        audio_np: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        移除低頻隆隆聲
        
        使用高通濾波器，移除 < 80Hz 的低頻噪音
        """
        # 設計高通濾波器（80Hz 截止）
        nyquist = sample_rate / 2
        cutoff = 80 / nyquist
        
        # 使用 Butterworth 高通濾波器
        sos = signal.butter(4, cutoff, btype='high', output='sos')
        filtered = signal.sosfilt(sos, audio_np)
        
        # sosfilt 會將 float32 轉換為 float64，但保留原 dtype
        # 只有在原始是 float32 時才需要轉回
        if audio_np.dtype == np.float32 and filtered.dtype != np.float32:
            filtered = filtered.astype(np.float32)
        
        return filtered
    
    def _apply_dereverb(
        self,
        audio_np: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        簡單的去混響處理
        
        使用頻譜減法來減少混響效應
        """
        nperseg = 1024
        noverlap = nperseg // 2
        
        f, t, stft = signal.stft(
            audio_np,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 估計混響尾巴（時間軸上的指數衰減）
        # 對每個頻率，計算時間軸上的平滑包絡
        for freq_idx in range(magnitude.shape[0]):
            envelope = uniform_filter1d(magnitude[freq_idx, :], size=10, mode='nearest')
            # 減去部分包絡（相當於減去混響成分）
            magnitude[freq_idx, :] = np.maximum(
                magnitude[freq_idx, :] - 0.1 * envelope,
                0.01 * magnitude[freq_idx, :]  # 保留至少 1%
            )
        
        # 重建
        stft_dereverb = magnitude * np.exp(1j * phase)
        _, audio_dereverb = signal.istft(
            stft_dereverb,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # 確保長度一致
        if len(audio_dereverb) > len(audio_np):
            audio_dereverb = audio_dereverb[:len(audio_np)]
        elif len(audio_dereverb) < len(audio_np):
            audio_dereverb = np.pad(
                audio_dereverb,
                (0, len(audio_np) - len(audio_dereverb)),
                mode='constant'
            )
        
        return audio_dereverb
    
    def _apply_wiener_filter(
        self,
        audio_np: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """應用維納濾波（同基礎版但參數更激進）"""
        nperseg = 512
        noverlap = nperseg // 2
        
        f, t, stft = signal.stft(
            audio_np,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        power = np.abs(stft) ** 2
        noise_frames = min(10, power.shape[1] // 4)
        noise_power = np.median(power[:, :noise_frames], axis=1, keepdims=True)
        
        # 使用更激進的參數
        wiener_gain = power / (power + self.wiener_beta * noise_power)
        stft_filtered = stft * wiener_gain
        
        _, audio_filtered = signal.istft(
            stft_filtered,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        if len(audio_filtered) > len(audio_np):
            audio_filtered = audio_filtered[:len(audio_np)]
        elif len(audio_filtered) < len(audio_np):
            audio_filtered = np.pad(
                audio_filtered,
                (0, len(audio_np) - len(audio_filtered)),
                mode='constant'
            )
        
        return audio_filtered
    
    def _apply_spectral_smoothing(
        self,
        audio_np: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """應用頻譜平滑（同基礎版）"""
        nperseg = 512
        noverlap = nperseg // 2
        
        f, t, stft = signal.stft(
            audio_np,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        magnitude_smooth = uniform_filter1d(
            magnitude,
            size=self.smoothing_window,
            axis=1,
            mode='nearest'
        )
        
        stft_smooth = magnitude_smooth * np.exp(1j * phase)
        
        _, audio_smooth = signal.istft(
            stft_smooth,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        if len(audio_smooth) > len(audio_np):
            audio_smooth = audio_smooth[:len(audio_np)]
        elif len(audio_smooth) < len(audio_np):
            audio_smooth = np.pad(
                audio_smooth,
                (0, len(audio_np) - len(audio_smooth)),
                mode='constant'
            )
        
        return audio_smooth
    
    def _protect_speech_frequencies(
        self,
        original: np.ndarray,
        denoised: np.ndarray,
        sample_rate: int,
        blend_ratio: float = 0.75  # 提高混合比例，更多使用降噪結果
    ) -> np.ndarray:
        """
        保護語音頻段（調整混合比例）
        """
        nperseg = 512
        noverlap = nperseg // 2
        
        f, t, stft_orig = signal.stft(
            original, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
        )
        _, _, stft_denoised = signal.stft(
            denoised, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
        )
        
        freq_mask = np.zeros(len(f))
        speech_low, speech_high = self.speech_freq_range
        
        speech_bins = (f >= speech_low) & (f <= speech_high)
        
        # 在激進模式下，即使是語音頻段也更多使用降噪結果
        if self.aggressive_mode:
            freq_mask[speech_bins] = blend_ratio  # 75% 降噪
            freq_mask[~speech_bins] = 1.0         # 100% 降噪
        else:
            freq_mask[speech_bins] = 0.65         # 65% 降噪（保守）
            freq_mask[~speech_bins] = 0.95        # 95% 降噪
        
        freq_mask_2d = freq_mask[:, np.newaxis]
        stft_blended = stft_denoised * freq_mask_2d + stft_orig * (1 - freq_mask_2d)
        
        _, audio_blended = signal.istft(
            stft_blended,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        if len(audio_blended) > len(original):
            audio_blended = audio_blended[:len(original)]
        elif len(audio_blended) < len(original):
            audio_blended = np.pad(
                audio_blended,
                (0, len(original) - len(audio_blended)),
                mode='constant'
            )
        
        return audio_blended
    
    def denoise(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int = 16000,
        snr_estimate: Optional[float] = None
    ) -> torch.Tensor:
        """
        增強版降噪處理
        
        處理流程：
        1. 低頻過濾（移除隆隆聲）
        2. 頻譜閘控降噪
        3. 高頻嘶嘶聲去除（如果偵測到）
        4. 維納濾波
        5. 頻譜平滑
        6. 去混響（可選）
        7. 語音頻段保護
        """
        # 記錄原始資訊
        original_device = audio.device if isinstance(audio, torch.Tensor) else 'cpu'
        original_shape = audio.shape
        
        # 轉換為 numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio.copy()
        
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()
        
        # 檢查能量
        if np.max(np.abs(audio_np)) < self.min_speech_energy:
            logger.debug("音訊能量過低，跳過降噪")
            return torch.from_numpy(audio_np).reshape(original_shape).to(original_device)
        
        # 估算 SNR
        if snr_estimate is None:
            snr = self.estimate_snr(audio_np, sample_rate)
        else:
            snr = snr_estimate
        
        # 檢測電子雜訊
        noise_info = self.detect_electronic_noise(audio_np, sample_rate)
        
        # 選擇策略
        strategy = self._select_strategy(snr)
        
        logger.debug(
            f"降噪策略: {strategy} (SNR: {snr:.2f}dB), "
            f"嘶嘶聲: {noise_info['hiss_level']:.3f}, "
            f"低頻噪音: {noise_info['rumble_level']:.3f}"
        )
        
        if strategy == 'skip' and not noise_info['has_hiss'] and not noise_info['has_rumble']:
            return torch.from_numpy(audio_np).reshape(original_shape).to(original_device)
        
        try:
            denoised = audio_np.copy()
            
            # 1. 低頻過濾（總是執行，因為分離模型可能產生低頻偽影）
            if self.enable_rumble_filter:
                denoised = self._remove_rumble(denoised, sample_rate)
                logger.debug("已應用低頻過濾")
            
            # 2. 頻譜閘控降噪（根據策略）
            if strategy != 'skip':
                denoised = self._apply_spectral_gate(denoised, sample_rate, strategy)
                logger.debug(f"已應用 {strategy} 頻譜閘控")
            
            # 3. 高頻嘶嘶聲去除（如果偵測到）
            if self.enable_hiss_removal and noise_info['has_hiss']:
                denoised = self._remove_hiss(denoised, sample_rate, noise_info['hiss_level'])
                logger.debug("已去除高頻嘶嘶聲")
            
            # 4. 維納濾波
            if self.enable_wiener and strategy in ['strong', 'medium']:
                denoised = self._apply_wiener_filter(denoised, sample_rate)
                logger.debug("已應用維納濾波")
            
            # 5. 頻譜平滑
            if self.enable_smoothing and strategy == 'strong':
                denoised = self._apply_spectral_smoothing(denoised, sample_rate)
                logger.debug("已應用頻譜平滑")
            
            # 6. 去混響（可選，較慢）
            if self.enable_dereverb:
                denoised = self._apply_dereverb(denoised, sample_rate)
                logger.debug("已應用去混響")
            
            # 7. 保護語音頻段
            blend_ratio = 0.85 if self.aggressive_mode else 0.70
            denoised = self._protect_speech_frequencies(
                audio_np, denoised, sample_rate, blend_ratio
            )
            
            # 8. 正規化並確保 float32
            original_max = np.max(np.abs(audio_np))
            denoised_max = np.max(np.abs(denoised))
            if denoised_max > 0:
                denoised = denoised * (original_max / denoised_max) * 0.98
            
            # 確保最終輸出是 float32
            denoised = denoised.astype(np.float32)
            result = torch.from_numpy(denoised).reshape(original_shape).to(original_device)
            
            logger.debug(
                f"降噪完成 - 原始峰值: {original_max:.4f}, "
                f"降噪峰值: {np.max(np.abs(denoised)):.4f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"降噪處理失敗: {e}，返回原始音訊")
            return torch.from_numpy(audio_np).reshape(original_shape).to(original_device)
    
    def batch_denoise(
        self,
        audio_list: List[Union[torch.Tensor, np.ndarray]],
        sample_rate: int = 16000,
        snr_estimates: Optional[List[float]] = None
    ) -> List[torch.Tensor]:
        """批次處理（同基礎版）"""
        if snr_estimates is None:
            snr_estimates = [None] * len(audio_list)
        
        results = []
        for i, (audio, snr) in enumerate(zip(audio_list, snr_estimates)):
            logger.debug(f"處理批次音訊 {i+1}/{len(audio_list)}")
            denoised = self.denoise(audio, sample_rate, snr)
            results.append(denoised)
        
        return results
    
    def get_strategy_info(self, snr: float) -> dict:
        """取得策略資訊（同基礎版）"""
        strategy = self._select_strategy(snr)
        
        if strategy == 'skip':
            return {
                'strategy': strategy,
                'snr': snr,
                'will_denoise': False,
                'description': '音訊品質良好，跳過降噪'
            }
        
        params = self.DENOISE_PARAMS[strategy]
        return {
            'strategy': strategy,
            'snr': snr,
            'will_denoise': True,
            'prop_decrease': params['prop_decrease'],
            'stationary': params['stationary'],
            'enable_wiener': self.enable_wiener and strategy in ['strong', 'medium'],
            'enable_smoothing': self.enable_smoothing and strategy == 'strong',
            'enable_hiss_removal': self.enable_hiss_removal,
            'enable_rumble_filter': self.enable_rumble_filter,
            'description': f'{strategy.capitalize()} 降噪模式（增強版）'
        }


# 全域快取
_GLOBAL_DENOISER_ENHANCED_CACHE: Optional[AdaptiveDenoiser] = None


def get_adaptive_denoiser(
    device: str = 'cpu',
    force_new: bool = False,
    **kwargs
) -> AdaptiveDenoiser:
    """取得增強版降噪器實例（單例模式）"""
    global _GLOBAL_DENOISER_ENHANCED_CACHE
    
    if force_new or _GLOBAL_DENOISER_ENHANCED_CACHE is None:
        _GLOBAL_DENOISER_ENHANCED_CACHE = AdaptiveDenoiser(device=device, **kwargs)
        logger.info("建立新的 AdaptiveDenoiserEnhanced 實例")
    
    return _GLOBAL_DENOISER_ENHANCED_CACHE


def clear_denoiser_cache():
    """清理快取"""
    global _GLOBAL_DENOISER_ENHANCED_CACHE
    _GLOBAL_DENOISER_ENHANCED_CACHE = None
    logger.info("已清理 AdaptiveDenoiserEnhanced 快取")