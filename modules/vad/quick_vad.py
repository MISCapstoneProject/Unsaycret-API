#!/usr/bin/env python3
"""
快速VAD工具 (Quick VAD Tool)
最簡單的語音活動偵測介面

用法：
    python quick_vad.py input.wav                    # 使用敏感模式，自動命名輸出
    python quick_vad.py input.wav output.wav         # 指定輸出檔案
    python quick_vad.py audio_folder/                # 批次處理整個目錄
    python quick_vad.py audio_folder/ --strict       # 使用嚴格模式
"""

import sys
import os
from pathlib import Path
import argparse

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vad.vad_module import VADProcessor, BatchVADProcessor

def quick_process(input_path: str, output_path: str = None, mode: str = "sensitive"):
    """
    快速處理音檔或目錄
    
    Args:
        input_path: 輸入音檔或目錄
        output_path: 輸出音檔或目錄 (None為自動生成)
        mode: 處理模式 (strict/normal/sensitive/ultra_sensitive)
    """
    
    input_p = Path(input_path)
    
    if not input_p.exists():
        print(f"❌ 路徑不存在: {input_path}")
        return False
    
    # 初始化VAD
    print(f"🚀 使用 {mode} 模式初始化VAD...")
    vad = VADProcessor(preset=mode)
    
    if input_p.is_file():
        # 單一檔案處理
        if output_path is None:
            output_path = input_p.parent / f"{input_p.stem}_vad{input_p.suffix}"
        
        print(f"🎵 處理音檔: {input_p.name}")
        print(f"📁 輸出至: {output_path}")
        
        try:
            result_path, stats = vad.process_file(str(input_p), str(output_path))
            
            print(f"✅ 完成! {stats['original_duration']:.1f}s → {stats['voice_duration']:.1f}s "
                  f"(保留 {stats['compression_ratio']*100:.1f}%, {stats['voice_segments']} 段落)")
            return True
            
        except Exception as e:
            print(f"❌ 處理失敗: {e}")
            return False
    
    elif input_p.is_dir():
        # 目錄批次處理
        if output_path is None:
            output_path = input_p.parent / f"{input_p.name}_vad"
        
        print(f"📁 批次處理目錄: {input_p}")
        print(f"📁 輸出目錄: {output_path}")
        
        try:
            processor = BatchVADProcessor(vad)
            result = processor.process_directory(str(input_p), str(output_path), generate_report=True)
            
            if result['success']:
                print(f"\n🎉 批次處理完成! 成功處理 {result['files_processed']} 個檔案")
                return True
            else:
                print("❌ 批次處理失敗")
                return False
                
        except Exception as e:
            print(f"❌ 批次處理失敗: {e}")
            return False
    
    else:
        print(f"❌ 不支援的路徑類型: {input_path}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="快速VAD工具 - 最簡單的語音活動偵測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python %(prog)s audio.wav                    # 使用敏感模式，自動命名
  python %(prog)s audio.wav clean.wav         # 指定輸出檔案
  python %(prog)s audio_folder/               # 批次處理目錄
  python %(prog)s audio.wav --strict          # 使用嚴格模式
  python %(prog)s audio.wav --ultra           # 使用超敏感模式

模式說明:
  --strict        嚴格模式 - 只保留清晰語音
  --normal        標準模式 - 平衡處理
  --sensitive     敏感模式 - 保留更多語音 (預設)
  --ultra         超敏感模式 - 保留微弱語音
        """
    )
    
    parser.add_argument('input', help='輸入音檔或目錄路徑')
    parser.add_argument('output', nargs='?', help='輸出音檔或目錄路徑 (可選，自動生成)')
    
    # 模式選擇 (互斥)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--strict', action='store_const', const='strict', dest='mode',
                           help='嚴格模式 - 只保留清晰語音')
    mode_group.add_argument('--normal', action='store_const', const='normal', dest='mode',
                           help='標準模式 - 平衡處理')
    mode_group.add_argument('--sensitive', action='store_const', const='sensitive', dest='mode',
                           help='敏感模式 - 保留更多語音 (預設)')
    mode_group.add_argument('--ultra', action='store_const', const='ultra_sensitive', dest='mode',
                           help='超敏感模式 - 保留微弱語音')
    
    args = parser.parse_args()
    
    # 預設模式為敏感
    mode = args.mode or 'sensitive'
    
    print("🎯 快速VAD工具")
    print("=" * 50)
    
    success = quick_process(args.input, args.output, mode)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
