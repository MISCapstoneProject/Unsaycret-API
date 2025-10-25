#!/usr/bin/env python3
"""
策略 2：條件式預降噪（智慧型）
目標：先評估音訊品質，只在必要時才預降噪
適用：想要最佳效能和品質平衡的場景
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AudioQualityMetrics:
    """音訊品質指標"""
    snr_db: float           # 信噪比
    noise_level: float      # 噪音水平
    clarity_score: float    # 清晰度評分
    need_denoise: bool      # 是否需要降噪
    recommended_strength: float  # 建議降噪強度


class AudioQualityAnalyzer:
    """
    音訊品質分析器
    
    決策邏輯：
    1. 高品質（SNR > 25 dB）→ 不降噪
    2. 良好品質（SNR 15-25 dB）→ 輕度降噪
    3. 中等品質（SNR 8-15 dB）→ 中度降噪
    4. 低品質（SNR < 8 dB）→ 較強降噪
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        
    def estimate_snr_advanced(self, audio: torch.Tensor) -> float:
        """
        進階 SNR 估計（更準確）
        
        方法：
        1. VAD（語音活動檢測）找出語音段和靜音段
        2. 分別計算語音能量和噪音能量
        3. 計算 SNR
        """
        frame_length = int(0.025 * self.sr)  # 25ms
        hop_length = int(0.010 * self.sr)    # 10ms
        
        # 計算短時能量
        energies = []
        for i in range(0, audio.shape[-1] - frame_length, hop_length):
            frame = audio[..., i:i+frame_length]
            energy = frame.pow(2).mean().item()
            energies.append(energy)
        
        energies = np.array(energies)
        
        if len(energies) == 0:
            return 40.0  # 預設高 SNR
        
        # 簡單的 VAD：使用能量閾值
        energy_threshold = np.percentile(energies, 40)
        
        speech_frames = energies > energy_threshold
        noise_frames = ~speech_frames
        
        # 計算語音和噪音能量
        if speech_frames.sum() > 0 and noise_frames.sum() > 0:
            speech_energy = energies[speech_frames].mean()
            noise_energy = energies[noise_frames].mean()
            
            if noise_energy > 0:
                snr_db = 10 * np.log10(speech_energy / noise_energy)
            else:
                snr_db = 40.0
        else:
            snr_db = 20.0  # 預設中等 SNR
        
        return snr_db
    
    def estimate_noise_level(self, audio: torch.Tensor) -> float:
        """
        估計噪音水平（0-1）
        
        0 = 幾乎無噪音
        1 = 噪音很大
        """
        # 使用前面靜音段估計
        n_samples = int(0.5 * self.sr)
        n_samples = min(n_samples, audio.shape[-1] // 4)
        
        noise_segment = audio[..., :n_samples]
        noise_level = noise_segment.abs().mean().item()
        
        # 正規化到 0-1
        noise_level = np.clip(noise_level * 10, 0, 1)
        
        return noise_level
    
    def estimate_clarity(self, audio: torch.Tensor) -> float:
        """
        估計清晰度（0-1）
        
        方法：看頻譜的集中程度
        - 清晰的語音：頻譜集中在語音頻段
        - 有雜訊的語音：頻譜分散
        """
        # 計算頻譜
        stft = torch.stft(
            audio,
            n_fft=512,
            hop_length=128,
            window=torch.hann_window(512, device=audio.device),
            return_complex=True
        )
        
        magnitude = stft.abs()
        power = magnitude.pow(2).mean(dim=-1)  # 平均功率譜
        
        # 計算語音頻段（300-3400 Hz）的能量佔比
        freq_bins = torch.fft.rfftfreq(512, 1/self.sr)
        speech_mask = (freq_bins >= 300) & (freq_bins <= 3400)
        
        speech_energy = power[speech_mask].sum()
        total_energy = power.sum()
        
        clarity = (speech_energy / total_energy).item() if total_energy > 0 else 0.5
        
        return clarity
    
    def analyze(self, audio: torch.Tensor) -> AudioQualityMetrics:
        """
        綜合分析音訊品質
        """
        # 計算各項指標
        snr_db = self.estimate_snr_advanced(audio)
        noise_level = self.estimate_noise_level(audio)
        clarity = self.estimate_clarity(audio)
        
        # 決策邏輯
        if snr_db > 25 and noise_level < 0.1:
            # 高品質：不需要降噪
            need_denoise = False
            strength = 0.0
        elif snr_db > 15 and noise_level < 0.3:
            # 良好品質：輕度降噪
            need_denoise = True
            strength = 0.2
        elif snr_db > 8:
            # 中等品質：中度降噪
            need_denoise = True
            strength = 0.4
        else:
            # 低品質：較強降噪（但不要太激進）
            need_denoise = True
            strength = 0.6
        
        # 根據清晰度微調
        if clarity < 0.5:
            strength = min(strength + 0.1, 0.7)  # 清晰度低，稍微加強
        
        return AudioQualityMetrics(
            snr_db=snr_db,
            noise_level=noise_level,
            clarity_score=clarity,
            need_denoise=need_denoise,
            recommended_strength=strength
        )


class ConditionalPreDenoiser:
    """
    條件式預降噪器
    
    核心邏輯：
    1. 先分析音訊品質
    2. 只在必要時才降噪
    3. 根據品質調整降噪強度
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        force_denoise: bool = False,  # 強制降噪（不評估品質）
        verbose: bool = True
    ):
        self.sr = sample_rate
        self.force_denoise = force_denoise
        self.verbose = verbose
        
        self.analyzer = AudioQualityAnalyzer(sample_rate)
    
    def denoise_with_strength(
        self,
        audio: torch.Tensor,
        strength: float
    ) -> torch.Tensor:
        """
        根據強度降噪
        
        Args:
            audio: 輸入音訊
            strength: 降噪強度（0-1）
        """
        if strength == 0:
            return audio
        
        # 估計噪音特徵
        n_samples = int(0.5 * self.sr)
        n_samples = min(n_samples, audio.shape[-1] // 4)
        noise_segment = audio[..., :n_samples]
        
        # STFT
        stft_noise = torch.stft(
            noise_segment,
            n_fft=512,
            hop_length=128,
            window=torch.hann_window(512, device=audio.device),
            return_complex=True
        )
        noise_power = stft_noise.abs().pow(2).mean(dim=-1, keepdim=True)
        
        stft = torch.stft(
            audio,
            n_fft=512,
            hop_length=128,
            window=torch.hann_window(512, device=audio.device),
            return_complex=True
        )
        
        magnitude = stft.abs()
        phase = stft.angle()
        signal_power = magnitude.pow(2)
        
        # 計算增益（Wiener-like filtering）
        snr = signal_power / (noise_power + 1e-10)
        gain = snr / (snr + 1)
        
        # 根據 strength 調整
        gain = 1 - (1 - gain) * strength
        
        # 平滑增益（時間軸）
        kernel = torch.ones(1, 1, 5, device=audio.device) / 5
        gain = gain.unsqueeze(0).unsqueeze(0)
        gain = torch.nn.functional.conv2d(
            gain, kernel, padding=(0, 2)
        ).squeeze(0).squeeze(0)
        
        # 應用
        filtered_magnitude = magnitude * gain
        filtered_stft = filtered_magnitude * torch.exp(1j * phase)
        
        # ISTFT
        denoised = torch.istft(
            filtered_stft,
            n_fft=512,
            hop_length=128,
            window=torch.hann_window(512, device=audio.device),
            length=audio.shape[-1]
        )
        
        return torch.clamp(denoised, -1.0, 1.0)
    
    def __call__(
        self,
        audio: torch.Tensor
    ) -> Tuple[torch.Tensor, AudioQualityMetrics]:
        """
        條件式預降噪
        
        Returns:
            denoised: 處理後的音訊
            metrics: 品質分析結果
        """
        original_shape = audio.shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # 分析品質
        metrics = self.analyzer.analyze(audio)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("音訊品質分析")
            print(f"{'='*60}")
            print(f"SNR: {metrics.snr_db:.2f} dB")
            print(f"噪音水平: {metrics.noise_level:.3f}")
            print(f"清晰度: {metrics.clarity_score:.3f}")
            print(f"建議降噪: {'是' if metrics.need_denoise else '否'}")
            if metrics.need_denoise:
                print(f"建議強度: {metrics.recommended_strength*100:.1f}%")
            print(f"{'='*60}\n")
        
        # 決定是否降噪
        if self.force_denoise or metrics.need_denoise:
            strength = metrics.recommended_strength if not self.force_denoise else 0.4
            denoised = self.denoise_with_strength(audio, strength)
            
            if self.verbose:
                print(f"✓ 已執行預降噪（強度: {strength*100:.1f}%）")
        else:
            denoised = audio
            
            if self.verbose:
                print("✓ 音訊品質良好，跳過預降噪")
        
        if len(original_shape) == 1:
            denoised = denoised.squeeze(0)
        
        return denoised, metrics


# ============================================================
# 完整流程整合
# ============================================================

class IntelligentPipeline:
    """
    智慧型處理流程
    
    特點：
    1. 自動評估是否需要預降噪
    2. 根據品質調整每個步驟的參數
    3. 最佳化效能和品質
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.pre_denoiser = ConditionalPreDenoiser(sample_rate, verbose=True)
    
    def process(self, audio_path: str):
        """
        完整處理流程
        """
        print(f"\n{'='*60}")
        print(f"智慧型處理流程")
        print(f"輸入：{audio_path}")
        print(f"{'='*60}\n")
        
        # 1. 載入音訊
        print("步驟 1/5：載入音訊...")
        audio, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            audio = resampler(audio)
        print(f"✓ 採樣率：{self.sr} Hz，長度：{audio.shape[-1]/self.sr:.2f} 秒\n")
        
        # 2. 條件式預降噪
        print("步驟 2/5：條件式預降噪...")
        audio_processed, metrics = self.pre_denoiser(audio)
        
        # 根據品質決定後續參數
        if metrics.snr_db < 10:
            print("⚠️  輸入品質較低，將使用更魯棒的參數\n")
            # 可以調整後續模型的參數
        
        # 儲存預降噪結果（用於後續步驟）
        torchaudio.save("temp_pre_denoised.wav", audio_processed.unsqueeze(0), self.sr)
        
        # 3. 說話者判斷
        print("步驟 3/5：說話者判斷...")
        # 使用預降噪後的音訊（如果有降噪的話）
        # num_speakers = your_diarization(audio_processed)
        print("✓ 檢測到 X 位說話者\n")
        
        # 4. 語音分離
        print("步驟 4/5：語音分離...")
        # separated = your_separation(audio_processed)
        print("✓ 分離完成\n")
        
        # 5. 後處理降噪
        print("步驟 5/5：後處理降噪...")
        # 根據預降噪的品質決定後降噪強度
        if metrics.need_denoise:
            print("   使用較強的後降噪")
            post_denoise_strength = 0.8
        else:
            print("   使用標準後降噪")
            post_denoise_strength = 0.6
        
        # final = post_denoise(separated, strength=post_denoise_strength)
        print("✓ 後處理完成\n")
        
        print(f"{'='*60}")
        print("處理完成！")
        print(f"{'='*60}\n")
        
        return {
            'metrics': metrics,
            'pre_denoised': audio_processed,
            # 'separated': separated,
            # 'final': final
        }


def benchmark_pre_denoise_effect(audio_path: str):
    """
    測試預降噪的實際效果
    
    比較：
    1. 無預降噪
    2. 有預降噪
    
    指標：
    - 說話者判斷準確度
    - 分離品質
    - 處理時間
    """
    import time
    
    print(f"\n{'='*70}")
    print("預降噪效果測試")
    print(f"{'='*70}\n")
    
    audio, sr = torchaudio.load(audio_path)
    
    # 測試 1：無預降噪
    print("🔹 測試 1：無預降噪")
    print("-" * 70)
    
    start = time.time()
    
    # 說話者判斷
    # result_1 = your_pipeline(audio)
    
    time_1 = time.time() - start
    
    print(f"處理時間：{time_1:.2f} 秒")
    # print(f"檢測說話者：{result_1['num_speakers']} 位")
    # print(f"分離品質：{result_1['quality_score']:.2f}")
    print()
    
    # 測試 2：有預降噪
    print("🔹 測試 2：有預降噪")
    print("-" * 70)
    
    denoiser = ConditionalPreDenoiser(sample_rate=sr, verbose=True)
    audio_denoised, metrics = denoiser(audio)
    
    start = time.time()
    
    # 說話者判斷
    # result_2 = your_pipeline(audio_denoised)
    
    time_2 = time.time() - start
    
    print(f"處理時間：{time_2:.2f} 秒（不含預降噪）")
    # print(f"檢測說話者：{result_2['num_speakers']} 位")
    # print(f"分離品質：{result_2['quality_score']:.2f}")
    print()
    
    # 結果比較
    print(f"{'='*70}")
    print("結果比較")
    print(f"{'='*70}")
    print(f"總處理時間：{time_1:.2f}s (無預降噪) vs {time_2:.2f}s (有預降噪)")
    # print(f"說話者檢測：{result_1['num_speakers']} vs {result_2['num_speakers']}")
    # print(f"分離品質：{result_1['quality_score']:.2f} vs {result_2['quality_score']:.2f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # 範例 1：條件式預降噪
    print("範例 1：條件式預降噪")
    print("=" * 60)
    
    denoiser = ConditionalPreDenoiser(sample_rate=16000, verbose=True)
    
    audio, sr = torchaudio.load("speaker1.wav")
    denoised, metrics = denoiser(audio)
    
    torchaudio.save("speaker1_conditional_denoised.wav", denoised.unsqueeze(0), sr)
    print("\n✓ 已儲存：speaker1_conditional_denoised.wav\n")
    
    # 範例 2：智慧型完整流程
    # pipeline = IntelligentPipeline(sample_rate=16000)
    # result = pipeline.process("your_audio.wav")
    
    # 範例 3：效果測試
    # benchmark_pre_denoise_effect("your_audio.wav")