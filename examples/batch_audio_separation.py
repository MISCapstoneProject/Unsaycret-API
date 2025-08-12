#!/usr/bin/env python3
"""
批次音頻分離處理工具
================

批次處理指定資料夾內所有 .wav 檔案的語者分離功能，支援遞歸搜索子資料夾。
將分離後的音檔輸出到指定資料夾，保持原有的目錄結構。

使用方法：
1. 修改下方的設定變數
2. 執行: python batch_audio_separation.py

功能特色：
- 遞歸處理整個資料夾及其子資料夾的 .wav 檔案
- 保持原有目錄結構輸出分離後的音檔
- 支援多種分離模型（ConvTasNet、SepFormer等）
- 自動檢測語者數量
- 生成詳細的處理報告與統計資訊
- 支援中斷恢復（可選）
"""

import os
import json
import time
import traceback
import soundfile as sf
import numpy as np
import torch
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.separation.separator import AudioSeparator

# ==================== 設定區域 ====================
# 👇 請修改這裡的資料夾路徑
INPUT_DIRECTORY = "data/clean/speaker4"       # 輸入音檔資料夾路徑
OUTPUT_DIRECTORY = "separated_output" # 輸出分離音檔資料夾路徑

# 分離設定
ENABLE_NOISE_REDUCTION = True       # 是否啟用降噪
SNR_THRESHOLD = 5.0                 # 信噪比閾值

# 處理設定
ENABLE_VERBOSE = True               # 是否顯示詳細輸出
SAVE_DETAILED_RESULTS = True        # 是否保存詳細結果到 JSON 檔案
SAVE_SUMMARY_RESULTS = True         # 是否保存摘要結果
ENABLE_PROGRESS_SAVE = True         # 是否啟用進度保存（可中斷恢復）
PROGRESS_SAVE_INTERVAL = 5          # 每處理幾個檔案保存一次進度

# 檔案篩選設定
SUPPORTED_EXTENSIONS = ['.wav', '.WAV']  # 支援的音檔副檔名
SKIP_HIDDEN_FILES = True            # 是否跳過隱藏檔案（以 . 開頭）
MIN_FILE_SIZE_KB = 1                # 最小檔案大小（KB），過小的檔案會跳過

# 分離輸出設定
OUTPUT_FORMAT = "wav"               # 輸出格式
OUTPUT_SAMPLE_RATE = 16000          # 輸出取樣率
SPEAKER_PREFIX = "speaker"          # 分離後音檔的前綴名稱
# ==================================================

def find_all_audio_files(directory: str) -> List[str]:
    """
    遞歸搜索指定資料夾內的所有音檔
    
    Args:
        directory: 目標資料夾路徑
        
    Returns:
        List[str]: 按路徑排序的音檔路徑列表
    """
    audio_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"❌ 錯誤：資料夾不存在 - {directory}")
        return audio_files
    
    print(f"🔍 搜索資料夾: {directory}")
    
    # 遞歸搜索所有音檔
    for root, dirs, files in os.walk(directory):
        # 排序資料夾和檔案以確保一致的處理順序
        dirs.sort()
        files.sort()
        
        for file in files:
            file_path = Path(root) / file
            
            # 跳過隱藏檔案（如果啟用）
            if SKIP_HIDDEN_FILES and file.startswith('.'):
                continue
            
            # 檢查副檔名
            if file_path.suffix in SUPPORTED_EXTENSIONS:
                # 檢查檔案大小
                file_size_kb = file_path.stat().st_size / 1024
                if file_size_kb >= MIN_FILE_SIZE_KB:
                    audio_files.append(str(file_path))
                else:
                    print(f"⚠️  跳過過小檔案 ({file_size_kb:.1f}KB): {file}")
    
    # 按路徑排序
    audio_files.sort()
    
    print(f"📁 找到 {len(audio_files)} 個有效音檔")
    return audio_files

def get_output_path(input_path: str, input_directory: str, output_directory: str) -> Path:
    """
    根據輸入路徑生成對應的輸出路徑，保持目錄結構
    
    Args:
        input_path: 輸入音檔的完整路徑
        input_directory: 輸入根目錄
        output_directory: 輸出根目錄
        
    Returns:
        Path: 輸出資料夾路徑
    """
    input_path = Path(input_path)
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    
    # 獲取相對於輸入根目錄的路徑
    try:
        relative_path = input_path.relative_to(input_directory)
    except ValueError:
        # 如果檔案不在輸入根目錄內，使用檔案名
        relative_path = input_path.name
    
    # 構建輸出路徑（去掉副檔名，因為會有多個分離檔案）
    output_base = output_directory / relative_path.parent / relative_path.stem
    
    return output_base

def separate_single_audio_file(separator: AudioSeparator, 
                             input_path: str, 
                             output_base: Path) -> Dict:
    """
    分離單個音檔
    
    Args:
        separator: 音頻分離器實例
        input_path: 輸入音檔路徑
        output_base: 輸出基礎路徑（不含副檔名）
        
    Returns:
        Dict: 處理結果字典
    """
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"🎯 分離音檔: {Path(input_path).name}")
    print(f"📂 輸入路徑: {input_path}")
    print(f"📤 輸出基礎路徑: {output_base}")
    print(f"{'='*80}")
    
    result_dict = {
        "input_file": input_path,
        "output_base": str(output_base),
        "processing_time": 0,
        "timestamp": datetime.now().isoformat(),
        "status": "processing"
    }
    
    try:
        # 檢查輸入檔案
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"輸入音檔不存在: {input_path}")
        
        # 讀取音檔
        print("🔄 讀取音檔...")
        audio_data, sample_rate = sf.read(input_path)
        
        # 音檔信息
        duration = len(audio_data) / sample_rate
        print(f"📊 音檔信息:")
        print(f"   - 取樣率: {sample_rate} Hz")
        print(f"   - 音檔長度: {duration:.2f} 秒")
        print(f"   - 聲道數: {audio_data.shape[1] if len(audio_data.shape) > 1 else 1}")
        
        # 轉換為適當格式
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # 轉為單聲道
        
        # 轉換為 tensor
        audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)  # 添加 batch 維度
        
        # 重採樣至目標取樣率（如果需要）
        if sample_rate != OUTPUT_SAMPLE_RATE:
            print(f"🔄 重採樣至 {OUTPUT_SAMPLE_RATE} Hz...")
            import torchaudio
            resampler = torchaudio.transforms.Resample(sample_rate, OUTPUT_SAMPLE_RATE)
            audio_tensor = resampler(audio_tensor)
        
        # 確保輸出目錄存在
        output_base.parent.mkdir(parents=True, exist_ok=True)
        
        # 執行分離
        print("🔄 執行音頻分離...")
        separated_results = separator.separate_and_save(
            audio_tensor, 
            str(output_base.parent), 
            segment_index=0
        )
        
        # 處理分離結果
        separated_files = []
        if separated_results:
            print(f"✅ 分離成功，產生 {len(separated_results)} 個檔案:")
            
            for i, (file_path, start_time_seg, end_time_seg, timestamp) in enumerate(separated_results):
                # 重新命名檔案以符合要求
                old_path = Path(file_path)
                new_filename = f"{output_base.name}_{SPEAKER_PREFIX}{i+1}.{OUTPUT_FORMAT}"
                new_path = output_base.parent / new_filename
                
                # 移動並重命名檔案
                if old_path.exists():
                    old_path.rename(new_path)
                    separated_files.append({
                        "speaker": i + 1,
                        "file_path": str(new_path),
                        "start_time": start_time_seg,
                        "end_time": end_time_seg,
                        "duration": end_time_seg - start_time_seg
                    })
                    print(f"   - {SPEAKER_PREFIX}{i+1}: {new_path.name}")
        else:
            print("⚠️  分離結果為空，可能沒有檢測到多個語者")
        
        # 更新結果
        result_dict.update({
            "status": "success",
            "separated_files": separated_files,
            "num_speakers_detected": len(separated_files),
            "original_duration": duration,
            "original_sample_rate": sample_rate,
            "output_sample_rate": OUTPUT_SAMPLE_RATE
        })
        
    except Exception as e:
        error_msg = f"分離失敗: {str(e)}"
        print(f"❌ {error_msg}")
        if ENABLE_VERBOSE:
            print("🔍 錯誤詳情:")
            traceback.print_exc()
        
        result_dict.update({
            "status": "error",
            "error": error_msg,
            "separated_files": [],
            "num_speakers_detected": 0
        })
    
    # 計算處理時間
    processing_time = time.time() - start_time
    result_dict["processing_time"] = round(processing_time, 3)
    print(f"⏱️  處理時間: {processing_time:.3f} 秒")
    
    return result_dict

def save_progress(results: List[Dict], progress_file: str):
    """保存處理進度"""
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 進度已保存至: {progress_file}")
    except Exception as e:
        print(f"⚠️  保存進度失敗: {e}")

def load_progress(progress_file: str) -> Tuple[List[Dict], int]:
    """
    載入處理進度
    
    Returns:
        Tuple[List[Dict], int]: (已處理結果列表, 已處理檔案數量)
    """
    if not os.path.exists(progress_file):
        return [], 0
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        processed_count = len(results)
        print(f"📥 載入進度: 已處理 {processed_count} 個檔案")
        return results, processed_count
    except Exception as e:
        print(f"⚠️  載入進度失敗: {e}")
        return [], 0

def generate_summary_report(results: List[Dict], total_time: float) -> Dict:
    """生成摘要報告"""
    total_files = len(results)
    successful_files = len([r for r in results if r.get("status") == "success"])
    failed_files = total_files - successful_files
    
    # 統計各種情況
    total_speakers_detected = sum(r.get("num_speakers_detected", 0) for r in results)
    files_with_speakers = len([r for r in results if r.get("num_speakers_detected", 0) > 0])
    processing_times = []
    total_audio_duration = 0
    
    for result in results:
        if "processing_time" in result:
            processing_times.append(result["processing_time"])
        if "original_duration" in result:
            total_audio_duration += result["original_duration"]
    
    # 計算平均處理時間
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    summary = {
        "總處理時間": f"{total_time:.2f} 秒",
        "總檔案數": total_files,
        "成功處理": successful_files,
        "失敗檔案": failed_files,
        "成功率": f"{(successful_files/total_files*100):.1f}%" if total_files > 0 else "0%",
        "平均單檔處理時間": f"{avg_processing_time:.3f} 秒",
        "總音頻時長": f"{total_audio_duration:.2f} 秒",
        "檢測到語者的檔案數": files_with_speakers,
        "總檢測語者數": total_speakers_detected,
        "平均每檔語者數": f"{(total_speakers_detected/files_with_speakers):.1f}" if files_with_speakers > 0 else "0",
        "處理時間範圍": {
            "最快": f"{min(processing_times):.3f} 秒" if processing_times else "N/A",
            "最慢": f"{max(processing_times):.3f} 秒" if processing_times else "N/A"
        }
    }
    
    return summary

def batch_audio_separation(input_directory: str, output_directory: str):
    """
    批次音頻分離功能
    
    Args:
        input_directory: 輸入資料夾路徑
        output_directory: 輸出資料夾路徑
    """
    print("🚀 批次音頻分離處理工具")
    print(f"輸入資料夾: {input_directory}")
    print(f"輸出資料夾: {output_directory}")
    print(f"降噪功能: {'啟用' if ENABLE_NOISE_REDUCTION else '停用'}")
    print("=" * 80)
    
    # 檢查輸入資料夾
    if not os.path.exists(input_directory):
        print(f"❌ 錯誤：輸入資料夾不存在 - {input_directory}")
        return
    
    # 創建輸出資料夾
    os.makedirs(output_directory, exist_ok=True)
    
    # 搜索所有音檔
    audio_files = find_all_audio_files(input_directory)
    if not audio_files:
        print("❌ 錯誤：找不到任何支援的音檔")
        return
    
    # 設定輸出檔案名稱
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"batch_separation_results_{timestamp}"
    
    detailed_results_file = f"{base_name}_detailed.json"
    summary_results_file = f"{base_name}_summary.json"
    progress_file = f"{base_name}_progress.json"
    
    # 載入進度（如果啟用）
    results = []
    start_index = 0
    if ENABLE_PROGRESS_SAVE:
        results, start_index = load_progress(progress_file)
        if start_index > 0:
            print(f"🔄 繼續從第 {start_index + 1} 個檔案開始處理...")
    
    try:
        print("🔄 初始化音頻分離器...")
        separator = AudioSeparator(
            enable_noise_reduction=ENABLE_NOISE_REDUCTION,
            snr_threshold=SNR_THRESHOLD
        )
        print("✅ 音頻分離器初始化完成")
        
        # 開始批次處理
        total_start_time = time.time()
        
        for i, audio_file in enumerate(audio_files[start_index:], start_index):
            print(f"\n進度: {i + 1}/{len(audio_files)}")
            
            # 生成輸出路徑
            output_base = get_output_path(audio_file, input_directory, output_directory)
            
            # 處理單個檔案
            result = separate_single_audio_file(separator, audio_file, output_base)
            results.append(result)
            
            # 定期保存進度
            if ENABLE_PROGRESS_SAVE and (i + 1) % PROGRESS_SAVE_INTERVAL == 0:
                save_progress(results, progress_file)
        
        # 計算總處理時間
        total_processing_time = time.time() - total_start_time
        
        print(f"\n🎉 批次處理完成！")
        print(f"⏱️  總處理時間: {total_processing_time:.2f} 秒")
        
        # 生成摘要報告
        summary = generate_summary_report(results, total_processing_time)
        
        # 顯示摘要
        print("\n" + "=" * 80)
        print("📋 處理摘要")
        print("=" * 80)
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        
        # 保存結果
        if SAVE_DETAILED_RESULTS:
            with open(detailed_results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 詳細結果已保存至: {detailed_results_file}")
        
        if SAVE_SUMMARY_RESULTS:
            with open(summary_results_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"💾 摘要結果已保存至: {summary_results_file}")
        
        # 清理進度檔案
        if ENABLE_PROGRESS_SAVE and os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"🗑️  進度檔案已清理")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  使用者中斷處理")
        if ENABLE_PROGRESS_SAVE:
            save_progress(results, progress_file)
            print(f"💾 進度已保存，下次可繼續處理")
    except Exception as e:
        print(f"\n❌ 批次處理發生錯誤: {e}")
        if ENABLE_VERBOSE:
            traceback.print_exc()
        if ENABLE_PROGRESS_SAVE:
            save_progress(results, progress_file)

def main():
    """主函數"""
    print("🎵 批次音頻分離處理工具")
    print("=" * 80)
    
    # 提示用戶可以修改設定
    print(f"📂 輸入資料夾: {INPUT_DIRECTORY}")
    print(f"📤 輸出資料夾: {OUTPUT_DIRECTORY}")
    print(f"🔧 降噪功能: {'啟用' if ENABLE_NOISE_REDUCTION else '停用'}")
    print(f"🔧 詳細輸出: {'開啟' if ENABLE_VERBOSE else '關閉'}")
    print(f"💾 保存結果: {'開啟' if SAVE_DETAILED_RESULTS else '關閉'}")
    print(f"🔄 進度保存: {'開啟' if ENABLE_PROGRESS_SAVE else '關閉'}")
    print()
    
    if not os.path.exists(INPUT_DIRECTORY):
        print("💡 提示：找不到預設輸入資料夾")
        print("請編輯此檔案的 INPUT_DIRECTORY 變數，指向您想批次處理的資料夾")
        print()
        return
    
    batch_audio_separation(INPUT_DIRECTORY, OUTPUT_DIRECTORY)

if __name__ == "__main__":
    main()
