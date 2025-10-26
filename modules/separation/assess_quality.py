"""
===============================================================================
音訊品質評估工具 (Audio Quality Assessment Tool)
===============================================================================

版本：v1.0.0
最後更新：2025-10-25

功能：
-----
1. 對比新舊方法分離出的音訊品質
2. 提供多維度的客觀評估指標
3. 生成詳細的對比報告
4. 支援批次評估多個檔案

評估指標：
---------
📊 基礎指標：
 • 峰值 (Peak)：音訊最大振幅
 • RMS：均方根能量
 • 動態範圍 (Dynamic Range)：最大與最小能量的比值

📈 頻譜指標：
 • 頻譜平坦度 (Spectral Flatness)：頻譜平滑程度
 • 頻譜質心 (Spectral Centroid)：頻率重心位置
 • 高頻能量比 (High Frequency Ratio)：高頻保留程度

🎯 語音品質指標：
 • 清晰度分數 (Clarity Score)：語音頻段能量
 • SNR 估計：信噪比
 • 零交叉率 (Zero Crossing Rate)：音訊清晰度

🔊 感知指標：
 • 響度 (Loudness)：感知音量
 • 尖銳度 (Sharpness)：高頻感知

使用方式：
---------
python audio_quality_assessment.py \
    --old_dir ./old_outputs \
    --new_dir ./new_outputs \
    --output_report ./quality_report.html

或使用 Python API：
from audio_quality_assessment import AudioQualityAssessment

evaluator = AudioQualityAssessment()
results = evaluator.compare_directories(
    old_dir="./old_outputs",
    new_dir="./new_outputs"
)
evaluator.generate_html_report(results, "report.html")

===============================================================================
"""

import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# 設定中文字型（如果需要）
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


@dataclass
class AudioMetrics:
    """音訊評估指標"""
    # 基礎指標
    peak: float
    rms: float
    dynamic_range: float
    
    # 頻譜指標
    spectral_flatness: float
    spectral_centroid: float
    high_freq_ratio: float
    
    # 語音品質
    clarity_score: float
    snr_estimate: float
    zero_crossing_rate: float
    
    # 感知指標
    loudness: float
    sharpness: float
    
    def to_dict(self) -> dict:
        """轉換為字典"""
        return {
            'basic': {
                'peak': float(self.peak),
                'rms': float(self.rms),
                'dynamic_range': float(self.dynamic_range)
            },
            'spectral': {
                'spectral_flatness': float(self.spectral_flatness),
                'spectral_centroid': float(self.spectral_centroid),
                'high_freq_ratio': float(self.high_freq_ratio)
            },
            'speech_quality': {
                'clarity_score': float(self.clarity_score),
                'snr_estimate': float(self.snr_estimate),
                'zero_crossing_rate': float(self.zero_crossing_rate)
            },
            'perceptual': {
                'loudness': float(self.loudness),
                'sharpness': float(self.sharpness)
            }
        }


class AudioQualityAssessment:
    """
    音訊品質評估工具
    
    用於對比新舊方法分離出的音訊品質差異
    """
    
    def __init__(self, sample_rate: int = 16000, verbose: bool = True):
        """
        初始化評估工具
        
        Args:
            sample_rate: 目標採樣率
            verbose: 是否顯示詳細資訊
        """
        self.sr = sample_rate
        self.verbose = verbose
    
    def load_audio(self, filepath: str) -> Tuple[torch.Tensor, int]:
        """
        載入音訊檔案
        
        Args:
            filepath: 音訊檔案路徑
        
        Returns:
            (audio, sample_rate): 音訊張量和採樣率
        """
        audio, sr = torchaudio.load(filepath)
        
        # 轉為單聲道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 重採樣
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            audio = resampler(audio)
        
        return audio.squeeze(0), self.sr
    
    def compute_basic_metrics(self, audio: torch.Tensor) -> dict:
        """
        計算基礎音訊指標
        
        Args:
            audio: 1D 音訊張量
        
        Returns:
            metrics: 基礎指標字典
        """
        # 峰值
        peak = audio.abs().max().item()
        
        # RMS
        rms = torch.sqrt(audio.pow(2).mean()).item()
        
        # 動態範圍
        frame_length = int(0.025 * self.sr)
        hop_length = int(0.010 * self.sr)
        
        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            energy = frame.pow(2).mean().item()
            if energy > 1e-10:
                energies.append(energy)
        
        if len(energies) > 0:
            dynamic_range = 10 * np.log10(max(energies) / (min(energies) + 1e-10))
        else:
            dynamic_range = 0.0
        
        return {
            'peak': peak,
            'rms': rms,
            'dynamic_range': dynamic_range
        }
    
    def compute_spectral_metrics(self, audio: torch.Tensor) -> dict:
        """
        計算頻譜指標
        
        Args:
            audio: 1D 音訊張量
        
        Returns:
            metrics: 頻譜指標字典
        """
        # STFT
        stft = torch.stft(
            audio,
            n_fft=512,
            hop_length=128,
            window=torch.hann_window(512, device=audio.device),
            return_complex=True
        )
        
        magnitude = stft.abs()
        power = magnitude.pow(2)
        
        # 頻譜平坦度
        geometric_mean = torch.exp(torch.log(magnitude + 1e-10).mean())
        arithmetic_mean = magnitude.mean()
        spectral_flatness = (geometric_mean / (arithmetic_mean + 1e-10)).item()
        
        # 頻譜質心
        freqs = torch.fft.rfftfreq(512, 1/self.sr, device=audio.device)
        weighted_sum = (freqs.unsqueeze(1) * power).sum()
        total_power = power.sum()
        spectral_centroid = (weighted_sum / (total_power + 1e-10)).item()
        
        # 高頻能量比 (2kHz 以上)
        high_freq_mask = freqs >= 2000
        high_freq_power = power[high_freq_mask].sum()
        total_power = power.sum()
        high_freq_ratio = (high_freq_power / (total_power + 1e-10)).item()
        
        return {
            'spectral_flatness': spectral_flatness,
            'spectral_centroid': spectral_centroid,
            'high_freq_ratio': high_freq_ratio
        }
    
    def compute_speech_quality_metrics(self, audio: torch.Tensor) -> dict:
        """
        計算語音品質指標
        
        Args:
            audio: 1D 音訊張量
        
        Returns:
            metrics: 語音品質指標字典
        """
        # STFT
        stft = torch.stft(
            audio,
            n_fft=512,
            hop_length=128,
            window=torch.hann_window(512, device=audio.device),
            return_complex=True
        )
        
        magnitude = stft.abs()
        power = magnitude.pow(2).mean(dim=-1)
        
        # 清晰度分數（語音頻段 300-3400 Hz 的能量佔比）
        freqs = torch.fft.rfftfreq(512, 1/self.sr, device=audio.device)
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        speech_energy = power[speech_mask].sum()
        total_energy = power.sum()
        clarity_score = (speech_energy / (total_energy + 1e-10)).item()
        
        # SNR 估計
        frame_length = int(0.025 * self.sr)
        hop_length = int(0.010 * self.sr)
        
        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            energy = frame.pow(2).mean().item()
            energies.append(energy)
        
        if len(energies) > 0:
            energies = np.array(energies)
            energy_threshold = np.percentile(energies, 40)
            speech_frames = energies > energy_threshold
            noise_frames = ~speech_frames
            
            if speech_frames.sum() > 0 and noise_frames.sum() > 0:
                speech_energy = energies[speech_frames].mean()
                noise_energy = energies[noise_frames].mean()
                snr_estimate = 10 * np.log10(speech_energy / (noise_energy + 1e-10))
            else:
                snr_estimate = 20.0
        else:
            snr_estimate = 20.0
        
        # 零交叉率
        zero_crossings = torch.sum(torch.diff(torch.sign(audio)) != 0).item()
        zero_crossing_rate = zero_crossings / len(audio)
        
        return {
            'clarity_score': clarity_score,
            'snr_estimate': snr_estimate,
            'zero_crossing_rate': zero_crossing_rate
        }
    
    def compute_perceptual_metrics(self, audio: torch.Tensor) -> dict:
        """
        計算感知指標
        
        Args:
            audio: 1D 音訊張量
        
        Returns:
            metrics: 感知指標字典
        """
        # STFT
        stft = torch.stft(
            audio,
            n_fft=512,
            hop_length=128,
            window=torch.hann_window(512, device=audio.device),
            return_complex=True
        )
        
        magnitude = stft.abs()
        power = magnitude.pow(2).mean(dim=-1)
        
        # 響度（近似）
        loudness = 10 * torch.log10(power.mean() + 1e-10).item()
        
        # 尖銳度（高頻權重）
        freqs = torch.fft.rfftfreq(512, 1/self.sr, device=audio.device)
        weights = torch.where(freqs > 1000, (freqs / 1000).pow(2), torch.ones_like(freqs))
        weighted_power = (power * weights).sum()
        sharpness = (weighted_power / (power.sum() + 1e-10)).item()
        
        return {
            'loudness': loudness,
            'sharpness': sharpness
        }
    
    def evaluate_audio(self, filepath: str) -> AudioMetrics:
        """
        完整評估一個音訊檔案
        
        Args:
            filepath: 音訊檔案路徑
        
        Returns:
            metrics: 完整的音訊指標
        """
        if self.verbose:
            print(f"評估: {os.path.basename(filepath)}")
        
        # 載入音訊
        audio, _ = self.load_audio(filepath)
        
        # 計算各類指標
        basic = self.compute_basic_metrics(audio)
        spectral = self.compute_spectral_metrics(audio)
        speech = self.compute_speech_quality_metrics(audio)
        perceptual = self.compute_perceptual_metrics(audio)
        
        return AudioMetrics(
            peak=basic['peak'],
            rms=basic['rms'],
            dynamic_range=basic['dynamic_range'],
            spectral_flatness=spectral['spectral_flatness'],
            spectral_centroid=spectral['spectral_centroid'],
            high_freq_ratio=spectral['high_freq_ratio'],
            clarity_score=speech['clarity_score'],
            snr_estimate=speech['snr_estimate'],
            zero_crossing_rate=speech['zero_crossing_rate'],
            loudness=perceptual['loudness'],
            sharpness=perceptual['sharpness']
        )
    
    def compare_files(
        self,
        old_file: str,
        new_file: str
    ) -> Dict[str, any]:
        """
        對比兩個音訊檔案
        
        Args:
            old_file: 舊方法的音訊檔案
            new_file: 新方法的音訊檔案
        
        Returns:
            comparison: 對比結果
        """
        old_metrics = self.evaluate_audio(old_file)
        new_metrics = self.evaluate_audio(new_file)
        
        # 計算改善百分比
        improvements = {}
        
        # 基礎指標（越高越好：peak 接近 1.0, rms 適中, dynamic_range 高）
        improvements['peak_improvement'] = (new_metrics.peak - old_metrics.peak) / old_metrics.peak * 100
        improvements['rms_improvement'] = (new_metrics.rms - old_metrics.rms) / old_metrics.rms * 100
        improvements['dynamic_range_improvement'] = (new_metrics.dynamic_range - old_metrics.dynamic_range) / (old_metrics.dynamic_range + 1e-10) * 100
        
        # 頻譜指標（越高越好）
        improvements['spectral_flatness_improvement'] = (new_metrics.spectral_flatness - old_metrics.spectral_flatness) / old_metrics.spectral_flatness * 100
        improvements['high_freq_ratio_improvement'] = (new_metrics.high_freq_ratio - old_metrics.high_freq_ratio) / old_metrics.high_freq_ratio * 100
        
        # 語音品質（越高越好）
        improvements['clarity_improvement'] = (new_metrics.clarity_score - old_metrics.clarity_score) / old_metrics.clarity_score * 100
        improvements['snr_improvement'] = (new_metrics.snr_estimate - old_metrics.snr_estimate) / (old_metrics.snr_estimate + 1e-10) * 100
        
        # 整體品質分數（0-100）
        old_score = self._calculate_overall_score(old_metrics)
        new_score = self._calculate_overall_score(new_metrics)
        improvements['overall_improvement'] = new_score - old_score
        
        return {
            'old_metrics': old_metrics,
            'new_metrics': new_metrics,
            'improvements': improvements,
            'old_score': old_score,
            'new_score': new_score
        }
    
    def _calculate_overall_score(self, metrics: AudioMetrics) -> float:
        """
        計算整體品質分數 (0-100)
        
        權重分配：
        - 峰值 (10%)：接近 0.98 最好
        - RMS (10%)：0.1-0.2 最好
        - 動態範圍 (15%)：越高越好
        - 清晰度 (20%)：越高越好
        - SNR (25%)：越高越好
        - 高頻保留 (10%)：0.15-0.25 最好
        - 頻譜平坦度 (10%)：0.2-0.4 最好
        """
        score = 0.0
        
        # 峰值分數（接近 0.98 最好）
        peak_score = max(0, 100 - abs(metrics.peak - 0.98) * 500)
        score += peak_score * 0.10
        
        # RMS 分數（0.1-0.2 最好）
        if 0.1 <= metrics.rms <= 0.2:
            rms_score = 100
        else:
            rms_score = max(0, 100 - abs(metrics.rms - 0.15) * 300)
        score += rms_score * 0.10
        
        # 動態範圍分數（20-60 dB 正常，越高越好）
        dr_score = min(100, (metrics.dynamic_range / 60) * 100)
        score += dr_score * 0.15
        
        # 清晰度分數（0.6-0.8 最好）
        if 0.6 <= metrics.clarity_score <= 0.8:
            clarity_score = 100
        else:
            clarity_score = max(0, 100 - abs(metrics.clarity_score - 0.7) * 200)
        score += clarity_score * 0.20
        
        # SNR 分數（15-30 dB 正常，越高越好）
        snr_score = min(100, (metrics.snr_estimate / 30) * 100)
        score += snr_score * 0.25
        
        # 高頻保留分數（0.15-0.25 最好）
        if 0.15 <= metrics.high_freq_ratio <= 0.25:
            hf_score = 100
        else:
            hf_score = max(0, 100 - abs(metrics.high_freq_ratio - 0.20) * 300)
        score += hf_score * 0.10
        
        # 頻譜平坦度分數（0.2-0.4 最好）
        if 0.2 <= metrics.spectral_flatness <= 0.4:
            sf_score = 100
        else:
            sf_score = max(0, 100 - abs(metrics.spectral_flatness - 0.3) * 200)
        score += sf_score * 0.10
        
        return score
    
    def compare_directories(
        self,
        old_dir: str,
        new_dir: str,
        pattern: str = "*.wav"
    ) -> Dict[str, any]:
        """
        對比兩個目錄中的所有音訊檔案
        
        Args:
            old_dir: 舊方法輸出目錄
            new_dir: 新方法輸出目錄
            pattern: 檔案匹配模式
        
        Returns:
            results: 完整對比結果
        """
        old_path = Path(old_dir)
        new_path = Path(new_dir)
        
        # 找到所有音訊檔案
        old_files = sorted(old_path.glob(pattern))
        new_files = sorted(new_path.glob(pattern))
        
        if self.verbose:
            print(f"\n找到 {len(old_files)} 個舊檔案，{len(new_files)} 個新檔案")
        
        # 配對檔案（基於檔名）
        comparisons = []
        for old_file in old_files:
            new_file = new_path / old_file.name
            if new_file.exists():
                if self.verbose:
                    print(f"\n對比: {old_file.name}")
                
                comparison = self.compare_files(str(old_file), str(new_file))
                comparison['filename'] = old_file.name
                comparisons.append(comparison)
            else:
                if self.verbose:
                    print(f"⚠️ 找不到對應的新檔案: {old_file.name}")
        
        # 計算平均改善
        if comparisons:
            avg_improvements = {}
            for key in comparisons[0]['improvements'].keys():
                values = [c['improvements'][key] for c in comparisons]
                avg_improvements[key] = np.mean(values)
            
            avg_old_score = np.mean([c['old_score'] for c in comparisons])
            avg_new_score = np.mean([c['new_score'] for c in comparisons])
        else:
            avg_improvements = {}
            avg_old_score = 0
            avg_new_score = 0
        
        return {
            'comparisons': comparisons,
            'avg_improvements': avg_improvements,
            'avg_old_score': avg_old_score,
            'avg_new_score': avg_new_score,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_report(
        self,
        results: Dict[str, any],
        output_file: str = "quality_report.txt"
    ):
        """
        生成文字格式的評估報告
        
        Args:
            results: compare_directories 的結果
            output_file: 輸出檔案路徑
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("音訊品質評估報告\n")
            f.write("="*80 + "\n\n")
            f.write(f"評估時間: {results['timestamp']}\n")
            f.write(f"評估檔案數: {len(results['comparisons'])}\n\n")
            
            # 整體改善摘要
            f.write("-"*80 + "\n")
            f.write("整體改善摘要\n")
            f.write("-"*80 + "\n")
            f.write(f"舊方法平均分數: {results['avg_old_score']:.2f}/100\n")
            f.write(f"新方法平均分數: {results['avg_new_score']:.2f}/100\n")
            f.write(f"整體改善: {results['avg_new_score'] - results['avg_old_score']:+.2f} 分\n\n")
            
            # 各項指標平均改善
            f.write("各項指標平均改善:\n")
            f.write(f"  峰值: {results['avg_improvements']['peak_improvement']:+.2f}%\n")
            f.write(f"  RMS: {results['avg_improvements']['rms_improvement']:+.2f}%\n")
            f.write(f"  動態範圍: {results['avg_improvements']['dynamic_range_improvement']:+.2f}%\n")
            f.write(f"  清晰度: {results['avg_improvements']['clarity_improvement']:+.2f}%\n")
            f.write(f"  SNR: {results['avg_improvements']['snr_improvement']:+.2f}%\n")
            f.write(f"  高頻保留: {results['avg_improvements']['high_freq_ratio_improvement']:+.2f}%\n\n")
            
            # 個別檔案詳細結果
            f.write("-"*80 + "\n")
            f.write("個別檔案詳細結果\n")
            f.write("-"*80 + "\n\n")
            
            for comp in results['comparisons']:
                f.write(f"檔案: {comp['filename']}\n")
                f.write(f"  舊方法分數: {comp['old_score']:.2f}/100\n")
                f.write(f"  新方法分數: {comp['new_score']:.2f}/100\n")
                f.write(f"  改善: {comp['improvements']['overall_improvement']:+.2f} 分\n")
                f.write("\n  詳細指標:\n")
                
                old = comp['old_metrics']
                new = comp['new_metrics']
                
                f.write(f"    峰值: {old.peak:.4f} → {new.peak:.4f} ({comp['improvements']['peak_improvement']:+.2f}%)\n")
                f.write(f"    RMS: {old.rms:.4f} → {new.rms:.4f} ({comp['improvements']['rms_improvement']:+.2f}%)\n")
                f.write(f"    動態範圍: {old.dynamic_range:.2f} → {new.dynamic_range:.2f} dB ({comp['improvements']['dynamic_range_improvement']:+.2f}%)\n")
                f.write(f"    清晰度: {old.clarity_score:.4f} → {new.clarity_score:.4f} ({comp['improvements']['clarity_improvement']:+.2f}%)\n")
                f.write(f"    SNR: {old.snr_estimate:.2f} → {new.snr_estimate:.2f} dB ({comp['improvements']['snr_improvement']:+.2f}%)\n")
                f.write(f"    高頻比: {old.high_freq_ratio:.4f} → {new.high_freq_ratio:.4f} ({comp['improvements']['high_freq_ratio_improvement']:+.2f}%)\n")
                f.write("\n")
        
        if self.verbose:
            print(f"\n✓ 報告已儲存至: {output_file}")
    
    def generate_html_report(
        self,
        results: Dict[str, any],
        output_file: str = "quality_report.html"
    ):
        """
        生成 HTML 格式的評估報告（含圖表）
        
        Args:
            results: compare_directories 的結果
            output_file: 輸出檔案路徑
        """
        # 準備數據
        filenames = [c['filename'] for c in results['comparisons']]
        old_scores = [c['old_score'] for c in results['comparisons']]
        new_scores = [c['new_score'] for c in results['comparisons']]
        
        # 創建圖表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 圖1: 整體分數對比
        ax = axes[0, 0]
        x = np.arange(len(filenames))
        width = 0.35
        ax.bar(x - width/2, old_scores, width, label='舊方法', alpha=0.8)
        ax.bar(x + width/2, new_scores, width, label='新方法', alpha=0.8)
        ax.set_xlabel('檔案')
        ax.set_ylabel('分數')
        ax.set_title('整體品質分數對比')
        ax.set_xticks(x)
        ax.set_xticklabels([f[:10] for f in filenames], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 圖2: 改善百分比
        ax = axes[0, 1]
        improvements = [c['improvements']['overall_improvement'] for c in results['comparisons']]
        colors = ['green' if i > 0 else 'red' for i in improvements]
        ax.bar(range(len(filenames)), improvements, color=colors, alpha=0.7)
        ax.set_xlabel('檔案')
        ax.set_ylabel('改善分數')
        ax.set_title('品質改善程度')
        ax.set_xticks(range(len(filenames)))
        ax.set_xticklabels([f[:10] for f in filenames], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # 圖3: 各項指標平均改善
        ax = axes[1, 0]
        metrics = ['峰值', 'RMS', '動態範圍', '清晰度', 'SNR', '高頻']
        improvements_values = [
            results['avg_improvements']['peak_improvement'],
            results['avg_improvements']['rms_improvement'],
            results['avg_improvements']['dynamic_range_improvement'],
            results['avg_improvements']['clarity_improvement'],
            results['avg_improvements']['snr_improvement'],
            results['avg_improvements']['high_freq_ratio_improvement']
        ]
        colors = ['green' if i > 0 else 'red' for i in improvements_values]
        ax.barh(metrics, improvements_values, color=colors, alpha=0.7)
        ax.set_xlabel('改善 (%)')
        ax.set_title('各項指標平均改善')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # 圖4: 分數分布
        ax = axes[1, 1]
        ax.hist([old_scores, new_scores], bins=10, label=['舊方法', '新方法'], alpha=0.7)
        ax.set_xlabel('分數')
        ax.set_ylabel('檔案數')
        ax.set_title('分數分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 儲存圖表
        chart_file = output_file.replace('.html', '_chart.png')
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 生成 HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>音訊品質評估報告</title>
    <style>
        body {{
            font-family: 'Microsoft JhengHei', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .summary-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #bdc3c7;
        }}
        .summary-item:last-child {{
            border-bottom: none;
        }}
        .score {{
            font-size: 24px;
            font-weight: bold;
        }}
        .score.old {{
            color: #e74c3c;
        }}
        .score.new {{
            color: #27ae60;
        }}
        .improvement {{
            color: #27ae60;
            font-weight: bold;
        }}
        .improvement.negative {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .chart {{
            text-align: center;
            margin: 30px 0;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metric-card {{
            display: inline-block;
            width: 30%;
            margin: 10px 1%;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            vertical-align: top;
        }}
        .metric-name {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        .arrow {{
            font-size: 18px;
            margin: 0 5px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 音訊品質評估報告</h1>
        
        <div class="summary">
            <div class="summary-item">
                <span>評估時間:</span>
                <span>{results['timestamp']}</span>
            </div>
            <div class="summary-item">
                <span>評估檔案數:</span>
                <span>{len(results['comparisons'])} 個</span>
            </div>
        </div>
        
        <h2>📊 整體改善摘要</h2>
        <div class="summary">
            <div class="summary-item">
                <span>舊方法平均分數:</span>
                <span class="score old">{results['avg_old_score']:.2f}/100</span>
            </div>
            <div class="summary-item">
                <span>新方法平均分數:</span>
                <span class="score new">{results['avg_new_score']:.2f}/100</span>
            </div>
            <div class="summary-item">
                <span>整體改善:</span>
                <span class="improvement {'negative' if results['avg_new_score'] - results['avg_old_score'] < 0 else ''}">{results['avg_new_score'] - results['avg_old_score']:+.2f} 分</span>
            </div>
        </div>
        
        <h2>📈 各項指標平均改善</h2>
        <div>
            <div class="metric-card">
                <div class="metric-name">峰值 (Peak)</div>
                <div class="metric-value">{results['avg_improvements']['peak_improvement']:+.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">RMS 能量</div>
                <div class="metric-value">{results['avg_improvements']['rms_improvement']:+.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">動態範圍</div>
                <div class="metric-value">{results['avg_improvements']['dynamic_range_improvement']:+.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">清晰度分數</div>
                <div class="metric-value">{results['avg_improvements']['clarity_improvement']:+.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">SNR 估計</div>
                <div class="metric-value">{results['avg_improvements']['snr_improvement']:+.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">高頻保留</div>
                <div class="metric-value">{results['avg_improvements']['high_freq_ratio_improvement']:+.2f}%</div>
            </div>
        </div>
        
        <h2>📊 視覺化圖表</h2>
        <div class="chart">
            <img src="{os.path.basename(chart_file)}" alt="評估圖表">
        </div>
        
        <h2>📋 個別檔案詳細結果</h2>
        <table>
            <thead>
                <tr>
                    <th>檔案名稱</th>
                    <th>舊方法分數</th>
                    <th>新方法分數</th>
                    <th>改善</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for comp in results['comparisons']:
            improvement_class = '' if comp['improvements']['overall_improvement'] >= 0 else 'negative'
            html_content += f"""
                <tr>
                    <td>{comp['filename']}</td>
                    <td>{comp['old_score']:.2f}</td>
                    <td>{comp['new_score']:.2f}</td>
                    <td class="improvement {improvement_class}">{comp['improvements']['overall_improvement']:+.2f}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated by Audio Quality Assessment Tool</p>
            <p>版本 v1.0.0 | 2025-10-25</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        if self.verbose:
            print(f"\n✓ HTML 報告已儲存至: {output_file}")
            print(f"✓ 圖表已儲存至: {chart_file}")


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='音訊品質評估工具')
    parser.add_argument('--old_dir', type=str, required=True, help='舊方法輸出目錄')
    parser.add_argument('--new_dir', type=str, required=True, help='新方法輸出目錄')
    parser.add_argument('--output_txt', type=str, default='quality_report.txt', help='文字報告輸出路徑')
    parser.add_argument('--output_html', type=str, default='quality_report.html', help='HTML 報告輸出路徑')
    parser.add_argument('--pattern', type=str, default='*.wav', help='檔案匹配模式')
    parser.add_argument('--sample_rate', type=int, default=16000, help='採樣率')
    
    args = parser.parse_args()
    
    # 創建評估工具
    print("🎵 音訊品質評估工具")
    print("="*80)
    
    evaluator = AudioQualityAssessment(sample_rate=args.sample_rate, verbose=True)
    
    # 執行評估
    print(f"\n開始對比...")
    print(f"舊方法目錄: {args.old_dir}")
    print(f"新方法目錄: {args.new_dir}")
    
    results = evaluator.compare_directories(
        old_dir=args.old_dir,
        new_dir=args.new_dir,
        pattern=args.pattern
    )
    
    # 生成報告
    print(f"\n生成報告...")
    evaluator.generate_report(results, args.output_txt)
    evaluator.generate_html_report(results, args.output_html)
    
    # 顯示摘要
    print("\n" + "="*80)
    print("📊 評估完成！")
    print("="*80)
    print(f"舊方法平均分數: {results['avg_old_score']:.2f}/100")
    print(f"新方法平均分數: {results['avg_new_score']:.2f}/100")
    print(f"整體改善: {results['avg_new_score'] - results['avg_old_score']:+.2f} 分")
    print("\n主要改善:")
    print(f"  • 清晰度: {results['avg_improvements']['clarity_improvement']:+.2f}%")
    print(f"  • SNR: {results['avg_improvements']['snr_improvement']:+.2f}%")
    print(f"  • 動態範圍: {results['avg_improvements']['dynamic_range_improvement']:+.2f}%")
    print(f"\n請查看詳細報告:")
    print(f"  • 文字報告: {args.output_txt}")
    print(f"  • HTML 報告: {args.output_html}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()