#!/usr/bin/env python3
"""
VAD模組 (Voice Activity Detection Module)
統一的語音活動偵測系統

主要功能:
- 🎯 預設配置選擇 (strict/normal/sensitive/ultra_sensitive/music/phone)
- 🔧 自訂參數調整  
- 📁 批次處理目錄
- 📊 詳細統計報告
- 💾 處理記錄保存

使用範例:
    # 直接使用
    from vad_module import VADProcessor
    vad = VADProcessor(preset="sensitive")
    vad.process_file("input.wav", "output.wav")
    
    # 命令列使用
    python vad_module.py input_folder output_folder --preset sensitive
    
    # 批次處理
    python vad_module.py test_audio/ processed_audio/ --preset sensitive --report
"""

import sys
import os
from pathlib import Path
import torch
import torchaudio
import argparse
import json
import copy
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import csv

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vad.voice_activity_detection import VoiceActivityDetector
from modules.separation.speaker_counter import SpeakerCounter

# ================== VAD配置管理 ==================

class VADConfig:
    """VAD配置管理類"""
    
    # 預設配置集
    PRESETS = {
        "strict": {
            "name": "嚴格模式",
            "description": "只保留清晰語音，適合清理高品質錄音",
            "energy_threshold": -40.0,
            "loud_threshold": -35.0,
            "zcr_min": 0.03,
            "zcr_max": 0.20,
            "min_voice_duration": 0.4,
            "padding": 0.03
        },
        "normal": {
            "name": "標準模式", 
            "description": "平衡的語音偵測，適合一般使用",
            "energy_threshold": -46.0,
            "loud_threshold": -40.0,
            "zcr_min": 0.02,
            "zcr_max": 0.25,
            "min_voice_duration": 0.3,
            "padding": 0.05
        },
        "sensitive": {
            "name": "敏感模式",
            "description": "保留更多語音，適合重要內容處理",
            "energy_threshold": -50.0,
            "loud_threshold": -45.0,
            "zcr_min": 0.01,
            "zcr_max": 0.30,
            "min_voice_duration": 0.2,
            "padding": 0.05
        },
        "ultra_sensitive": {
            "name": "超敏感模式",
            "description": "保留微弱語音，適合低音量或遠距錄音",
            "energy_threshold": -55.0,
            "loud_threshold": -50.0,
            "zcr_min": 0.005,
            "zcr_max": 0.35,
            "min_voice_duration": 0.15,
            "padding": 0.08
        },
        "music": {
            "name": "音樂模式",
            "description": "適合歌聲和音樂內容偵測",
            "energy_threshold": -48.0,
            "loud_threshold": -42.0,
            "zcr_min": 0.008,
            "zcr_max": 0.40,
            "min_voice_duration": 0.5,
            "padding": 0.1
        },
        "phone": {
            "name": "電話模式",
            "description": "適合電話錄音或低品質音訊",
            "energy_threshold": -52.0,
            "loud_threshold": -47.0,
            "zcr_min": 0.015,
            "zcr_max": 0.28,
            "min_voice_duration": 0.25,
            "padding": 0.08
        }
    }
    
    @classmethod
    def get_config(cls, preset_name: str) -> Dict:
        """獲取預設配置"""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"未知預設配置: {preset_name}. 可用配置: {list(cls.PRESETS.keys())}")
        return copy.deepcopy(cls.PRESETS[preset_name])
    
    @classmethod
    def list_presets(cls) -> None:
        """列出所有可用預設配置"""
        print("📋 可用的VAD預設配置:")
        print("=" * 80)
        
        for preset_name, config in cls.PRESETS.items():
            print(f"\n🔧 {preset_name.upper()}")
            print(f"📝 {config['name']} - {config['description']}")
            print("-" * 50)
            
            # 顯示關鍵參數
            key_params = {k: v for k, v in config.items() 
                         if k not in ['name', 'description']}
            
            for param, value in key_params.items():
                print(f"   {param:<20}: {value}")

# ================== 主要VAD處理器 ==================

class VADProcessor(VoiceActivityDetector):
    """主要VAD處理器"""
    
    def __init__(self, preset: str = "normal", method: str = "speechbrain", **custom_params):
        """
        初始化VAD處理器
        
        Args:
            preset: 預設配置名稱
            method: VAD方法 ("energy", "speechbrain", "hybrid")
            **custom_params: 自訂參數覆蓋
        """
        super().__init__(method)
        
        # 載入配置
        self.config = VADConfig.get_config(preset)
        self.config.update(custom_params)  # 自訂參數覆蓋預設值
        
        # 應用配置到VAD
        if method in ["speechbrain", "hybrid"] and hasattr(self, 'speaker_counter'):
            self._apply_config()
    
    def _apply_config(self):
        """應用配置參數到SpeakerCounter"""
        config = self.config
        
        # 能量閾值參數
        if 'energy_threshold' in config:
            self.speaker_counter.VAD_ABS_DBFS_MIN = config['energy_threshold']
            self.speaker_counter.VAD_MIN_DBFS = config['energy_threshold']
        
        if 'loud_threshold' in config:
            self.speaker_counter.VAD_LOUD_DBFS = config['loud_threshold']
        
        # ZCR參數
        if 'zcr_min' in config:
            self.speaker_counter.VAD_MIN_ZCR = config['zcr_min']
        
        if 'zcr_max' in config:
            self.speaker_counter.VAD_MAX_ZCR = config['zcr_max']
        
        # 時間參數
        if 'min_voice_duration' in config:
            self.min_voice_duration = config['min_voice_duration']
    
    def get_config_summary(self) -> str:
        """獲取當前配置摘要"""
        lines = ["🔧 當前VAD配置:"]
        for param, value in self.config.items():
            if param not in ['name', 'description']:
                lines.append(f"   {param:<20}: {value}")
        return "\n".join(lines)
    
    def process_file(self, input_file: str, output_file: str = None, 
                     padding: float = None) -> Tuple[str, Dict]:
        """
        處理單一音檔
        
        Args:
            input_file: 輸入音檔路徑
            output_file: 輸出音檔路徑 (None為自動生成)
            padding: 語音段落前後保留時間 (None使用配置值)
        
        Returns:
            Tuple[str, Dict]: (輸出檔案路徑, 統計數據)
        """
        if padding is None:
            padding = self.config.get('padding', 0.05)
        
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_vad{input_path.suffix}"
        
        return self.remove_silence(input_file, output_file, padding=padding)

# ================== 批次處理器 ==================

class BatchVADProcessor:
    """批次VAD處理器"""
    
    def __init__(self, vad: VADProcessor):
        self.vad = vad
        self.stats = []
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         audio_extensions: List[str] = None,
                         generate_report: bool = True) -> Dict:
        """
        批次處理目錄中的音檔
        
        Args:
            input_dir: 輸入目錄路徑
            output_dir: 輸出目錄路徑
            audio_extensions: 支援的音檔副檔名
            generate_report: 是否生成處理報告
        
        Returns:
            Dict: 處理統計結果
        """
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"輸入目錄不存在: {input_dir}")
        
        # 創建輸出目錄
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 尋找音檔
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f"*{ext}"))
        
        if not audio_files:
            print(f"❌ 在 {input_dir} 中未找到音檔")
            return {"success": False, "files_processed": 0}
        
        print(f"🎵 找到 {len(audio_files)} 個音檔")
        print(f"📁 輸出目錄: {output_dir}")
        print(f"🔧 使用配置: {self.vad.config.get('name', 'custom')}")
        print("-" * 60)
        
        # 處理每個音檔
        success_count = 0
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] 🎵 處理: {audio_file.name}")
            
            try:
                # 設定輸出檔案路徑
                output_file = output_path / f"{audio_file.stem}_vad{audio_file.suffix}"
                
                # 處理音檔
                result_path, file_stats = self.vad.remove_silence(
                    str(audio_file), 
                    str(output_file),
                    padding=self.vad.config.get('padding', 0.05)
                )
                
                # 記錄統計
                file_stats['input_file'] = str(audio_file)
                file_stats['output_file'] = str(result_path)
                file_stats['timestamp'] = datetime.now().isoformat()
                self.stats.append(file_stats)
                
                # 顯示結果
                print(f"   ✅ 成功! 原始: {file_stats['original_duration']:.1f}s → "
                      f"語音: {file_stats['voice_duration']:.1f}s "
                      f"(保留 {file_stats['compression_ratio']*100:.1f}%)")
                
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ 處理失敗: {str(e)}")
        
        # 生成處理報告
        if generate_report and self.stats:
            self._generate_report(output_path)
        
        # 返回統計結果
        return {
            "success": True,
            "files_processed": success_count,
            "total_files": len(audio_files),
            "output_directory": str(output_path),
            "stats": self.stats
        }
    
    def _generate_report(self, output_dir: Path):
        """生成處理報告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON報告
        json_report = output_dir / f"vad_report_{timestamp}.json"
        with open(json_report, 'w', encoding='utf-8') as f:
            json.dump({
                "config": self.vad.config,
                "processing_stats": self.stats,
                "summary": self._calculate_summary()
            }, f, ensure_ascii=False, indent=2)
        
        # CSV報告
        csv_report = output_dir / f"vad_report_{timestamp}.csv"
        if self.stats:
            fieldnames = ['input_file', 'output_file', 'original_duration', 
                         'voice_duration', 'removed_duration', 'compression_ratio',
                         'voice_segments', 'timestamp']
            
            with open(csv_report, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for stat in self.stats:
                    writer.writerow({k: stat.get(k, '') for k in fieldnames})
        
        print(f"\n📊 處理報告已保存:")
        print(f"   📄 JSON報告: {json_report}")
        print(f"   📈 CSV報告: {csv_report}")
    
    def _calculate_summary(self) -> Dict:
        """計算處理摘要"""
        if not self.stats:
            return {}
        
        total_original = sum(s['original_duration'] for s in self.stats)
        total_voice = sum(s['voice_duration'] for s in self.stats)
        total_removed = sum(s['removed_duration'] for s in self.stats)
        total_segments = sum(s['voice_segments'] for s in self.stats)
        
        return {
            "total_files": len(self.stats),
            "total_original_duration": round(total_original, 2),
            "total_voice_duration": round(total_voice, 2),
            "total_removed_duration": round(total_removed, 2),
            "average_compression_ratio": round(total_voice / total_original if total_original > 0 else 0, 3),
            "total_voice_segments": total_segments,
            "average_segments_per_file": round(total_segments / len(self.stats), 1)
        }

# ================== 命令列介面 ==================

def main():
    parser = argparse.ArgumentParser(
        description="統一語音活動偵測系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 列出所有預設配置
  python %(prog)s --list-presets
  
  # 使用敏感模式處理單一檔案
  python %(prog)s input.wav output.wav --preset sensitive
  
  # 批次處理目錄
  python %(prog)s test_audio/ processed_audio/ --preset sensitive --report
  
  # 自訂參數
  python %(prog)s input.wav output.wav --energy-threshold -50 --min-duration 0.2
  
  # 組合預設和自訂參數
  python %(prog)s input.wav output.wav --preset sensitive --energy-threshold -52
        """
    )
    
    # 基本參數
    parser.add_argument('input', help='輸入音檔或目錄路徑')
    parser.add_argument('output', nargs='?', help='輸出音檔或目錄路徑')
    
    # 預設配置
    parser.add_argument('--preset', choices=list(VADConfig.PRESETS.keys()),
                       default='normal', help='預設配置選擇 (default: normal)')
    parser.add_argument('--list-presets', action='store_true',
                       help='列出所有可用預設配置')
    
    # VAD方法
    parser.add_argument('--method', choices=['energy', 'speechbrain', 'hybrid'],
                       default='speechbrain', help='VAD方法選擇 (default: speechbrain)')
    
    # 自訂參數
    parser.add_argument('--energy-threshold', type=float,
                       help='能量閾值 (dBFS, 如: -50.0)')
    parser.add_argument('--loud-threshold', type=float,
                       help='響亮閾值 (dBFS, 如: -40.0)')
    parser.add_argument('--zcr-min', type=float,
                       help='最小過零率 (如: 0.01)')
    parser.add_argument('--zcr-max', type=float,
                       help='最大過零率 (如: 0.30)')
    parser.add_argument('--min-duration', type=float,
                       help='最小語音段落長度 (秒, 如: 0.2)')
    parser.add_argument('--padding', type=float,
                       help='語音段落前後保留時間 (秒, 如: 0.05)')
    
    # 處理選項
    parser.add_argument('--report', action='store_true',
                       help='生成詳細處理報告')
    parser.add_argument('--extensions', nargs='+',
                       default=['.wav', '.mp3', '.flac', '.m4a', '.aac'],
                       help='支援的音檔副檔名')
    
    args = parser.parse_args()
    
    # 列出預設配置
    if args.list_presets:
        VADConfig.list_presets()
        return
    
    # 檢查必要參數
    if not args.output:
        parser.error("需要指定輸出路徑")
    
    # 建立自訂參數字典
    custom_params = {}
    if args.energy_threshold is not None:
        custom_params['energy_threshold'] = args.energy_threshold
    if args.loud_threshold is not None:
        custom_params['loud_threshold'] = args.loud_threshold
    if args.zcr_min is not None:
        custom_params['zcr_min'] = args.zcr_min
    if args.zcr_max is not None:
        custom_params['zcr_max'] = args.zcr_max
    if args.min_duration is not None:
        custom_params['min_voice_duration'] = args.min_duration
    if args.padding is not None:
        custom_params['padding'] = args.padding
    
    try:
        # 初始化VAD
        print("🚀 初始化VAD系統...")
        vad = VADProcessor(preset=args.preset, method=args.method, **custom_params)
        
        # 顯示配置
        print(vad.get_config_summary())
        print("-" * 60)
        
        # 判斷輸入類型
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 單一檔案處理
            print(f"🎵 處理單一音檔: {input_path.name}")
            
            result_path, stats = vad.remove_silence(
                str(input_path), 
                args.output,
                padding=vad.config.get('padding', 0.05)
            )
            
            print(f"✅ 處理完成!")
            print(f"   原始長度: {stats['original_duration']:.2f}s")
            print(f"   語音長度: {stats['voice_duration']:.2f}s")
            print(f"   移除時間: {stats['removed_duration']:.2f}s ({(1-stats['compression_ratio'])*100:.1f}%)")
            print(f"   語音段落: {stats['voice_segments']} 個")
            print(f"   輸出檔案: {result_path}")
            
        elif input_path.is_dir():
            # 目錄批次處理
            processor = BatchVADProcessor(vad)
            result = processor.process_directory(
                args.input, 
                args.output,
                audio_extensions=args.extensions,
                generate_report=args.report
            )
            
            if result['success']:
                print(f"\n🎉 批次處理完成!")
                print(f"   成功處理: {result['files_processed']}/{result['total_files']} 個檔案")
                print(f"   輸出目錄: {result['output_directory']}")
                
                # 顯示總體統計
                if processor.stats:
                    summary = processor._calculate_summary()
                    print(f"\n📊 總體統計:")
                    print(f"   總原始時長: {summary['total_original_duration']:.1f}s")
                    print(f"   總語音時長: {summary['total_voice_duration']:.1f}s")
                    print(f"   平均壓縮率: {summary['average_compression_ratio']*100:.1f}%")
                    print(f"   平均語音段落: {summary['average_segments_per_file']:.1f} 個/檔案")
            else:
                print("❌ 批次處理失敗")
        else:
            print(f"❌ 輸入路徑無效: {args.input}")
            
    except Exception as e:
        print(f"❌ 處理過程發生錯誤: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
