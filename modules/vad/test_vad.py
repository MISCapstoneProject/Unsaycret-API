#!/usr/bin/env python3
"""
語音活動偵測測試腳本
測試6秒音檔的靜音移除功能
"""

import sys
import os
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vad.voice_activity_detection import VoiceActivityDetector, VoiceSegment

def test_vad_on_audio_files(audio_dir: str = "test_audio_me"):
    """測試指定目錄中的音檔"""
    
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        print(f"❌ 音檔目錄不存在: {audio_dir}")
        return
    
    # 支援的音檔格式
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(audio_path.glob(f"*{ext}"))
        audio_files.extend(audio_path.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        print(f"❌ 在 {audio_dir} 中未找到音檔")
        return
    
    audio_files = sorted(audio_files)
    print(f"🎵 找到 {len(audio_files)} 個音檔:")
    for i, file in enumerate(audio_files):
        print(f"   {i+1}. {file.name}")
    
    # 創建輸出目錄
    output_dir = Path("voice_processed")
    output_dir.mkdir(exist_ok=True)
    
    # 測試不同的VAD方法
    methods = ["energy", "speechbrain", "hybrid"]
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"🔧 測試方法: {method.upper()}")
        print(f"{'='*50}")
        
        # 創建VAD實例
        vad = VoiceActivityDetector(method=method)
        
        # 處理每個音檔
        for audio_file in audio_files:
            print(f"\n📁 處理: {audio_file.name}")
            
            try:
                # 生成輸出檔名
                output_file = output_dir / f"{audio_file.stem}_{method}_voice{audio_file.suffix}"
                
                # 移除靜音
                result_path, stats = vad.remove_silence(
                    str(audio_file), 
                    str(output_file),
                    padding=0.05  # 50ms padding
                )
                
                print(f"✅ 成功處理:")
                print(f"   壓縮比: {stats['compression_ratio']:.2f} "
                      f"({stats['removed_duration']:.2f}s 被移除)")
                print(f"   語音段落: {stats['voice_segments']} 個")
                print(f"   輸出: {Path(result_path).name}")
                
            except Exception as e:
                print(f"❌ 處理失敗: {e}")
                continue
    
    print(f"\n🎯 所有處理結果保存在: {output_dir}")
    
    # 顯示比較結果
    print(f"\n📊 處理結果比較:")
    print("音檔名稱\t\t原始長度\tEnergy\tSpeechBrain\tHybrid")
    print("-" * 70)
    
    for audio_file in audio_files:
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(audio_file))
            original_duration = waveform.shape[1] / sr
            
            results = [f"{audio_file.stem[:15]:15s}", f"{original_duration:.2f}s"]
            
            for method in methods:
                output_file = output_dir / f"{audio_file.stem}_{method}_voice{audio_file.suffix}"
                if output_file.exists():
                    processed_waveform, _ = torchaudio.load(str(output_file))
                    processed_duration = processed_waveform.shape[1] / sr
                    compression = processed_duration / original_duration
                    results.append(f"{compression:.2f}")
                else:
                    results.append("N/A")
            
            print("\t".join(results))
            
        except Exception as e:
            print(f"{audio_file.stem}\t錯誤: {e}")


def demonstrate_vad_segments(audio_file: str):
    """展示語音段落偵測結果"""
    
    print(f"🔍 分析音檔: {audio_file}")
    
    methods = ["energy", "speechbrain", "hybrid"]
    
    for method in methods:
        print(f"\n--- {method.upper()} 方法 ---")
        
        vad = VoiceActivityDetector(method=method)
        segments = vad.detect_voice_segments(audio_file)
        
        if not segments:
            print("   未偵測到語音活動")
            continue
        
        total_voice_time = sum(seg.end - seg.start for seg in segments)
        
        print(f"   偵測到 {len(segments)} 個語音段落:")
        print(f"   總語音時間: {total_voice_time:.2f}s")
        
        for i, seg in enumerate(segments):
            duration = seg.end - seg.start
            print(f"     {i+1}. {seg.start:.2f}s - {seg.end:.2f}s "
                  f"({duration:.2f}s, 置信度: {seg.confidence:.2f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="語音活動偵測測試")
    parser.add_argument("--audio_dir", default="test_audio_me", 
                       help="音檔目錄路徑")
    parser.add_argument("--demo_file", help="展示單一音檔的分析結果")
    
    args = parser.parse_args()
    
    if args.demo_file:
        demonstrate_vad_segments(args.demo_file)
    else:
        test_vad_on_audio_files(args.audio_dir)
