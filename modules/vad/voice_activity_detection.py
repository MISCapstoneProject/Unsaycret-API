"""
語音活動偵測與靜音去除工具
Voice Activity Detection (VAD) and Silence Removal Tool

功能：
- 偵測音檔中的語音段落
- 移除靜音和背景雜音
- 保留語音活動區間
- 支援多種偵測方法
"""

import os
import sys
import math
import numpy as np
import torch
import torchaudio
import librosa
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any
from dataclasses import dataclass

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from modules.separation.speaker_counter import SpeakerCounter

logger = get_logger(__name__)

@dataclass
class VoiceSegment:
    """語音段落資訊"""
    start: float  # 開始時間（秒）
    end: float    # 結束時間（秒）
    confidence: float  # 語音置信度
    energy: float      # 能量值

class VoiceActivityDetector:
    """語音活動偵測器"""
    
    def __init__(self, method: str = "speechbrain"):
        """
        初始化VAD
        
        Args:
            method: 偵測方法 ("speechbrain", "energy", "hybrid")
        """
        self.method = method
        
        if method in ["speechbrain", "hybrid"]:
            self.speaker_counter = SpeakerCounter()
        
        # 能量法參數
        self.energy_threshold = -40  # dBFS
        self.min_voice_duration = 0.3  # 最小語音段落長度（秒）
        self.min_silence_duration = 0.5  # 最小靜音段落長度（秒）
        self.frame_duration = 0.025  # 幀長度（秒）
        self.hop_duration = 0.010     # 跳躍長度（秒）
        
    def detect_voice_segments(self, audio: Union[str, np.ndarray, torch.Tensor], 
                            sr: int = 16000) -> List[VoiceSegment]:
        """
        偵測語音段落
        
        Args:
            audio: 音訊資料（檔案路徑、numpy陣列或torch張量）
            sr: 採樣率
            
        Returns:
            語音段落列表
        """
        # 載入音訊
        if isinstance(audio, str):
            waveform, sr = torchaudio.load(audio)
            # 確保單聲道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            waveform = audio
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        
        # 重採樣到16kHz（如果需要）
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
        
        if self.method == "energy":
            return self._detect_by_energy(waveform, sr)
        elif self.method == "speechbrain":
            return self._detect_by_speechbrain(waveform, sr)
        elif self.method == "hybrid":
            return self._detect_by_hybrid(waveform, sr)
        else:
            raise ValueError(f"不支援的偵測方法: {self.method}")
    
    def _detect_by_energy(self, waveform: torch.Tensor, sr: int) -> List[VoiceSegment]:
        """基於能量的語音偵測"""
        audio = waveform.squeeze().numpy()
        
        # 計算短時能量
        frame_length = int(self.frame_duration * sr)
        hop_length = int(self.hop_duration * sr)
        
        # 計算每幀的能量
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames ** 2, axis=0)
        
        # 轉換為dB
        energy_db = 10 * np.log10(energy + 1e-10)
        
        # 二值化
        voice_mask = energy_db > self.energy_threshold
        
        # 轉換為時間軸
        time_frames = librosa.frames_to_time(
            np.arange(len(voice_mask)), 
            sr=sr, 
            hop_length=hop_length
        )
        
        # 找到語音段落
        segments = self._mask_to_segments(voice_mask, time_frames, energy_db)
        
        return segments
    
    def _detect_by_speechbrain(self, waveform: torch.Tensor, sr: int) -> List[VoiceSegment]:
        """使用SpeechBrain的語音偵測"""
        # 使用現有的SpeakerCounter的VAD功能
        has_voice, metrics = self.speaker_counter._has_voice(
            waveform, sr, debug=False, return_metrics=True
        )
        
        if not has_voice:
            logger.info("🔇 未偵測到語音活動")
            return []
        
        # 獲取逐幀VAD結果
        frame_mask = self.speaker_counter._frame_vad_mask(waveform.squeeze(), sr)
        
        if len(frame_mask) == 0:
            return []
        
        # 轉換為時間軸
        hop_length = int(0.015 * sr)  # 15ms
        time_frames = np.arange(len(frame_mask)) * hop_length / sr
        
        # 計算置信度（使用能量資訊）
        confidences = np.ones(len(frame_mask)) * metrics['voiced_ratio']
        
        segments = self._mask_to_segments(
            frame_mask.cpu().numpy(), 
            time_frames, 
            confidences
        )
        
        return segments
    
    def _detect_by_hybrid(self, waveform: torch.Tensor, sr: int) -> List[VoiceSegment]:
        """混合方法：結合能量和SpeechBrain"""
        energy_segments = self._detect_by_energy(waveform, sr)
        sb_segments = self._detect_by_speechbrain(waveform, sr)
        
        # 如果SpeechBrain沒偵測到語音，返回能量法結果
        if not sb_segments:
            return energy_segments
        
        # 如果能量法沒偵測到，返回SpeechBrain結果
        if not energy_segments:
            return sb_segments
        
        # 取交集，提高精確度
        merged_segments = []
        for sb_seg in sb_segments:
            for energy_seg in energy_segments:
                # 計算重疊
                start = max(sb_seg.start, energy_seg.start)
                end = min(sb_seg.end, energy_seg.end)
                
                if start < end:  # 有重疊
                    merged_segments.append(VoiceSegment(
                        start=start,
                        end=end,
                        confidence=(sb_seg.confidence + energy_seg.confidence) / 2,
                        energy=max(sb_seg.energy, energy_seg.energy)
                    ))
        
        return merged_segments
    
    def _mask_to_segments(self, mask: np.ndarray, time_frames: np.ndarray, 
                         confidences: np.ndarray) -> List[VoiceSegment]:
        """將二進位遮罩轉換為語音段落"""
        segments = []
        
        # 找到語音區間的開始和結束
        voice_starts = []
        voice_ends = []
        
        in_voice = False
        for i, is_voice in enumerate(mask):
            if is_voice and not in_voice:
                voice_starts.append(i)
                in_voice = True
            elif not is_voice and in_voice:
                voice_ends.append(i)
                in_voice = False
        
        # 如果最後一個段落沒有結束
        if in_voice:
            voice_ends.append(len(mask))
        
        # 創建語音段落
        for start_idx, end_idx in zip(voice_starts, voice_ends):
            if start_idx >= len(time_frames) or end_idx > len(time_frames):
                continue
                
            start_time = time_frames[start_idx]
            end_time = time_frames[min(end_idx, len(time_frames)-1)]
            duration = end_time - start_time
            
            # 過濾太短的段落
            if duration >= self.min_voice_duration:
                # 計算該段落的平均置信度
                segment_confidences = confidences[start_idx:end_idx]
                avg_confidence = np.mean(segment_confidences)
                max_energy = np.max(segment_confidences)
                
                segments.append(VoiceSegment(
                    start=start_time,
                    end=end_time,
                    confidence=float(avg_confidence),
                    energy=float(max_energy)
                ))
        
        return segments
    
    def remove_silence(self, audio_path: str, output_path: str = None, 
                      padding: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """
        移除靜音並保存新音檔
        
        Args:
            audio_path: 輸入音檔路徑
            output_path: 輸出音檔路徑（None為自動生成）
            padding: 語音段落前後保留的時間（秒）
            
        Returns:
            (輸出檔案路徑, 處理統計資訊)
        """
        logger.info(f"🎵 處理音檔: {audio_path}")
        
        # 載入音檔
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        original_duration = waveform.shape[1] / sr
        
        # 偵測語音段落
        segments = self.detect_voice_segments(waveform, sr)
        
        if not segments:
            logger.warning("⚠️ 未偵測到語音段落，保留原音檔")
            segments = [VoiceSegment(0, original_duration, 0.5, 0.5)]
        
        logger.info(f"🔍 偵測到 {len(segments)} 個語音段落:")
        for i, seg in enumerate(segments):
            logger.info(f"   {i+1}. {seg.start:.2f}s - {seg.end:.2f}s "
                       f"(置信度: {seg.confidence:.2f})")
        
        # 提取語音段落並合併
        voice_segments = []
        for seg in segments:
            start_sample = max(0, int((seg.start - padding) * sr))
            end_sample = min(waveform.shape[1], int((seg.end + padding) * sr))
            
            segment_audio = waveform[:, start_sample:end_sample]
            voice_segments.append(segment_audio)
        
        # 合併所有語音段落
        if voice_segments:
            final_audio = torch.cat(voice_segments, dim=1)
        else:
            final_audio = waveform
        
        # 生成輸出路徑
        if output_path is None:
            input_path = Path(audio_path)
            output_path = input_path.parent / f"{input_path.stem}_voice_only{input_path.suffix}"
        
        # 保存結果
        torchaudio.save(str(output_path), final_audio, sr)
        
        # 統計資訊
        new_duration = final_audio.shape[1] / sr
        removed_duration = original_duration - new_duration
        compression_ratio = new_duration / original_duration
        
        stats = {
            "original_duration": original_duration,
            "voice_duration": new_duration,
            "removed_duration": removed_duration,
            "compression_ratio": compression_ratio,
            "voice_segments": len(segments),
            "method": self.method
        }
        
        logger.info(f"✅ 處理完成:")
        logger.info(f"   原始長度: {original_duration:.2f}s")
        logger.info(f"   語音長度: {new_duration:.2f}s")
        logger.info(f"   移除時間: {removed_duration:.2f}s ({(1-compression_ratio)*100:.1f}%)")
        logger.info(f"   輸出檔案: {output_path}")
        
        return str(output_path), stats


def main():
    """命令列介面"""
    import argparse
    
    parser = argparse.ArgumentParser(description="語音活動偵測與靜音移除工具")
    parser.add_argument("input", help="輸入音檔路徑")
    parser.add_argument("-o", "--output", help="輸出音檔路徑")
    parser.add_argument("-m", "--method", choices=["energy", "speechbrain", "hybrid"], 
                       default="hybrid", help="偵測方法")
    parser.add_argument("-p", "--padding", type=float, default=0.1, 
                       help="語音段落前後保留時間（秒）")
    parser.add_argument("--threshold", type=float, default=-40, 
                       help="能量閾值（dBFS）")
    
    args = parser.parse_args()
    
    # 創建VAD
    vad = VoiceActivityDetector(method=args.method)
    vad.energy_threshold = args.threshold
    
    # 處理音檔
    output_path, stats = vad.remove_silence(
        args.input, 
        args.output, 
        padding=args.padding
    )
    
    print(f"\n📊 處理結果:")
    print(f"壓縮比: {stats['compression_ratio']:.2f}")
    print(f"移除時間: {stats['removed_duration']:.2f}s")
    print(f"語音段落: {stats['voice_segments']} 個")


if __name__ == "__main__":
    main()
