#!/usr/bin/env python3
"""
批次單語者音頻分離處理工具 (含 SI-SDR 品質評估)
================================================

專門針對單語者音檔的批次分離功能，支援遞歸搜索子資料夾，
每個語者資料夾可限制處理的音檔數量，並簡化輸出目錄結構。

🔥 新增功能：SI-SDR 自動品質評估
- 自動計算分離音檔與原音檔的 SI-SDR (Scale-Invariant SDR)
- 設定品質門檻，自動識別低品質分離結果  
- 生成低品質檔案清單，避免3000個檔案中遺漏問題檔案
- 詳細品質統計報告

使用方法：
1. 修改下方的設定變數
2. 執行: python batch_single_speaker_separation.py

功能特色：
- 🎯 專門針對單語者音檔優化的分離邏輯
- 📁 支援複雜的資料夾結構 (母資料夾/speakerX/utts/xxx.wav)
- 🔢 可限制每個語者資料夾的處理檔案數量
- 📤 簡化輸出結構 (新母資料夾/speakerX/xxx.wav)
- 🎛️ 使用固定2人模型 + SingleSpeakerSelector 確保最佳品質
- 📊 SI-SDR 自動品質評估，預防低品質檔案
- 🚨 自動警告系統，即時發現問題檔案
- 📈 詳細的品質統計與進度報告
- 💾 支援中斷恢復功能
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

from modules.separation.separator import AudioSeparator, SeparationModel
import torchaudio

# ==================== SI-SDR 計算函數 ====================

def to_mono(wav: torch.Tensor) -> torch.Tensor:
    """轉換為單聲道"""
    if wav.ndim == 2:
        if wav.size(0) > 1:
            wav = wav.mean(0)
        else:
            wav = wav.squeeze(0)
    elif wav.ndim == 1:
        pass
    else:
        raise ValueError(f"Unexpected wav shape: {tuple(wav.shape)}")
    return wav

def resample_if_needed(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    """如需要則重新取樣"""
    if sr_in == sr_out:
        return wav
    return torchaudio.functional.resample(wav, sr_in, sr_out)

def si_sdr_db(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    """
    計算 Scale-Invariant SDR (SI-SDR) in dB
    
    Args:
        ref: 參考音檔 (原始音檔)
        est: 估計音檔 (分離後音檔)
        eps: 數值穩定性小量
        
    Returns:
        float: SI-SDR in dB
    """
    ref = ref - ref.mean()
    est = est - est.mean()
    dot = torch.dot(est, ref)
    s_ref = torch.dot(ref, ref) + eps
    proj = (dot / s_ref) * ref
    noise = est - proj
    num = proj.pow(2).sum() + eps
    den = noise.pow(2).sum() + eps
    return 10.0 * torch.log10(num / den).item()

def calculate_sisdr_for_separated_files(original_file: Path, separated_files: List[Dict]) -> List[Dict]:
    """
    計算分離檔案與原始檔案的 SI-SDR
    
    Args:
        original_file: 原始音檔路徑
        separated_files: 分離檔案資訊列表
        
    Returns:
        List[Dict]: 包含 SI-SDR 分數的分離檔案資訊
    """
    try:
        # 載入原始音檔
        orig_wav, orig_sr = torchaudio.load(str(original_file))
        orig_wav = to_mono(orig_wav)
        orig_wav = resample_if_needed(orig_wav, orig_sr, OUTPUT_SAMPLE_RATE)
        
        for sep_file_info in separated_files:
            sep_file_path = Path(sep_file_info["file_path"])
            
            if sep_file_path.exists():
                # 載入分離音檔
                sep_wav, sep_sr = torchaudio.load(str(sep_file_path))
                sep_wav = to_mono(sep_wav)
                sep_wav = resample_if_needed(sep_wav, sep_sr, OUTPUT_SAMPLE_RATE)
                
                # 長度對齊（以較短者為準）
                min_len = min(orig_wav.numel(), sep_wav.numel())
                orig_aligned = orig_wav[:min_len]
                sep_aligned = sep_wav[:min_len]
                
                # 計算 SI-SDR
                sisdr = si_sdr_db(ref=orig_aligned, est=sep_aligned)
                sep_file_info["si_sdr_db"] = round(sisdr, 3)
                
                # 品質判斷
                if ENABLE_SISDR_EVALUATION:
                    sep_file_info["quality_good"] = sisdr >= SISDR_THRESHOLD
                    if sisdr < SISDR_THRESHOLD:
                        sep_file_info["quality_warning"] = f"低品質檔案 (SI-SDR: {sisdr:.2f} dB < {SISDR_THRESHOLD} dB)"
                
            else:
                sep_file_info["si_sdr_db"] = None
                sep_file_info["quality_good"] = False
                sep_file_info["quality_warning"] = "分離檔案不存在"
                
    except Exception as e:
        for sep_file_info in separated_files:
            sep_file_info["si_sdr_db"] = None
            sep_file_info["quality_good"] = False
            sep_file_info["quality_warning"] = f"SI-SDR 計算失敗: {str(e)}"
    
    return separated_files

# ==================== 設定區域 ====================
# 👇 請修改這裡的資料夾路徑和參數
INPUT_ROOT_DIRECTORY = "TCC-300_dynamic_2"     # 母資料夾路徑
OUTPUT_ROOT_DIRECTORY = "dataset_separated_single_tcc300"   # 輸出母資料夾路徑

# 處理限制設定
MAX_FILES_PER_SPEAKER = 10          # 每個語者資料夾最多處理幾個音檔
SPEAKER_FOLDER_PATTERN = "speaker*" # 語者資料夾的命名模式

# 分離設定
ENABLE_NOISE_REDUCTION = True       # 是否啟用降噪
SNR_THRESHOLD = 5.0                 # 信噪比閾值
USE_FIXED_2SPK_MODEL = False        # 改為 False：啟用智能選擇但避免3人模型

# 🔥 SI-SDR 品質檢測設定
ENABLE_SISDR_EVALUATION = True      # 是否啟用 SI-SDR 品質評估
SISDR_THRESHOLD = 5.0               # SI-SDR 品質門檻 (dB)，低於此值視為品質不佳
WARN_LOW_QUALITY = True             # 是否對低品質檔案發出警告
SAVE_LOW_QUALITY_LIST = True        # 是否保存低品質檔案清單

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

def natural_sort_key(text: str) -> List:
    """
    自然排序鍵函數，正確處理數字排序
    例如：speaker1, speaker2, speaker10, speaker11 而不是 speaker1, speaker10, speaker11, speaker2
    """
    import re
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', str(text))]

def find_speaker_folders(root_directory: str) -> List[Path]:
    """
    搜尋母資料夾下所有符合模式的語者資料夾
    
    Args:
        root_directory: 母資料夾路徑
        
    Returns:
        List[Path]: 語者資料夾路徑列表（按自然數字順序排序）
    """
    root_path = Path(root_directory)
    if not root_path.exists():
        print(f"❌ 錯誤：母資料夾不存在 - {root_directory}")
        return []
    
    print(f"🔍 搜索語者資料夾: {root_directory}")
    
    speaker_folders = []
    for folder in root_path.glob(SPEAKER_FOLDER_PATTERN):
        if folder.is_dir():
            speaker_folders.append(folder)
    
    # 使用自然排序而不是字符串排序
    speaker_folders.sort(key=lambda x: natural_sort_key(x.name))
    
    print(f"📁 找到 {len(speaker_folders)} 個語者資料夾")
    return speaker_folders

def find_audio_files_in_speaker_folder(speaker_folder: Path, max_files: int = None) -> List[Path]:
    """
    在語者資料夾中搜尋音檔，支援遞歸搜索
    
    Args:
        speaker_folder: 語者資料夾路徑
        max_files: 最大檔案數量限制
        
    Returns:
        List[Path]: 音檔路徑列表（已排序並限制數量）
    """
    audio_files = []
    
    # 遞歸搜索所有音檔
    for audio_file in speaker_folder.rglob("*"):
        if audio_file.is_file():
            # 跳過隱藏檔案（如果啟用）
            if SKIP_HIDDEN_FILES and audio_file.name.startswith('.'):
                continue
            
            # 檢查副檔名
            if audio_file.suffix in SUPPORTED_EXTENSIONS:
                # 檢查檔案大小
                file_size_kb = audio_file.stat().st_size / 1024
                if file_size_kb >= MIN_FILE_SIZE_KB:
                    audio_files.append(audio_file)
                else:
                    if ENABLE_VERBOSE:
                        print(f"  ⚠️  跳過過小檔案 ({file_size_kb:.1f}KB): {audio_file.name}")
    
    # 使用自然排序而不是字符串排序
    audio_files.sort(key=lambda x: natural_sort_key(x.name))
    
    # 限制檔案數量
    if max_files and len(audio_files) > max_files:
        selected_files = audio_files[:max_files]
        print(f"  📄 {speaker_folder.name}: 找到 {len(audio_files)} 個音檔，選取前 {max_files} 個")
        return selected_files
    else:
        print(f"  📄 {speaker_folder.name}: 找到 {len(audio_files)} 個音檔")
        return audio_files

def get_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    """
    根據輸入檔案路徑生成簡化的輸出路径
    
    原始結構: 母資料夾/speakerX/utts/xxx.wav
    輸出結構: 新母資料夾/speakerX/xxx.wav
    
    Args:
        input_file: 輸入音檔路徑
        input_root: 輸入根目錄
        output_root: 輸出根目錄
        
    Returns:
        Path: 輸出基礎路徑（不含副檔名）
    """
    try:
        # 取得相對於輸入根目錄的路徑
        relative_path = input_file.relative_to(input_root)
        
        # 提取語者資料夾名稱（第一層）
        speaker_name = relative_path.parts[0]
        
        # 構建簡化的輸出路徑：output_root/speakerX/filename
        output_base = output_root / speaker_name / input_file.stem
        
        return output_base
        
    except ValueError:
        # 容錯處理：如果檔案不在輸入根目錄內
        output_base = output_root / "unknown_speaker" / input_file.stem
        return output_base

def separate_single_speaker_audio(separator: AudioSeparator, 
                                input_path: Path, 
                                output_base: Path) -> Dict:
    """
    分離單個單語者音檔 - 使用固定2人模型 + SingleSpeakerSelector
    
    Args:
        separator: 音頻分離器實例
        input_path: 輸入音檔路徑
        output_base: 輸出基礎路徑（不含副檔名）
        
    Returns:
        Dict: 處理結果字典
    """
    start_time = time.time()
    
    if ENABLE_VERBOSE:
        print(f"\n{'='*80}")
        print(f"🎯 分離單語者音檔: {input_path.name}")
        print(f"📂 輸入路徑: {input_path}")
        print(f"📤 輸出基礎路徑: {output_base}")
        print(f"{'='*80}")
    
    result_dict = {
        "input_file": str(input_path),
        "output_base": str(output_base),
        "processing_time": 0,
        "timestamp": datetime.now().isoformat(),
        "status": "processing",
        "speaker_folder": input_path.parent.parent.name if len(input_path.parts) >= 3 else input_path.parent.name
    }
    
    try:
        # 檢查輸入檔案
        if not input_path.exists():
            raise FileNotFoundError(f"輸入音檔不存在: {input_path}")
        
        # 讀取音檔
        if ENABLE_VERBOSE:
            print("🔄 讀取音檔...")
        audio_data, sample_rate = sf.read(str(input_path))
        
        # 音檔信息
        duration = len(audio_data) / sample_rate
        if ENABLE_VERBOSE:
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
            if ENABLE_VERBOSE:
                print(f"🔄 重採樣至 {OUTPUT_SAMPLE_RATE} Hz...")
            import torchaudio
            resampler = torchaudio.transforms.Resample(sample_rate, OUTPUT_SAMPLE_RATE)
            audio_tensor = resampler(audio_tensor)
        
        # 確保輸出目錄存在
        output_base.parent.mkdir(parents=True, exist_ok=True)
        
        # 執行單語者分離 - 使用 separate_and_save 方法
        if ENABLE_VERBOSE:
            print("🔄 執行單語者音頻分離...")
        
        separated_results = separator.separate_and_save(
            audio_tensor, 
            str(output_base.parent), 
            segment_index=0,
            absolute_start_time=None
        )
        
        # 處理分離結果
        separated_files = []
        if separated_results:
            if ENABLE_VERBOSE:
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
                    if ENABLE_VERBOSE:
                        print(f"   - {SPEAKER_PREFIX}{i+1}: {new_path.name}")
        else:
            print("⚠️  分離結果為空，可能是單語者被正確識別但沒有產生多個輸出")
            # 對於單語者，這可能是正常情況
        
        # 🔥 計算 SI-SDR 品質評估
        if ENABLE_SISDR_EVALUATION and separated_files:
            if ENABLE_VERBOSE:
                print("🔍 計算 SI-SDR 品質評估...")
            
            separated_files = calculate_sisdr_for_separated_files(input_path, separated_files)
            
            # 顯示品質評估結果
            for sep_file in separated_files:
                sisdr = sep_file.get("si_sdr_db")
                if sisdr is not None:
                    quality_status = "✅ 良好" if sep_file.get("quality_good", False) else "⚠️  品質不佳"
                    if ENABLE_VERBOSE:
                        print(f"   - {Path(sep_file['file_path']).name}: SI-SDR = {sisdr:.2f} dB ({quality_status})")
                    
                    # 低品質警告
                    if WARN_LOW_QUALITY and not sep_file.get("quality_good", False):
                        warning_msg = sep_file.get("quality_warning", "品質不佳")
                        print(f"   ⚠️  警告: {warning_msg}")
                else:
                    if ENABLE_VERBOSE:
                        print(f"   - {Path(sep_file['file_path']).name}: SI-SDR 計算失敗")
        
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
    if ENABLE_VERBOSE:
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
    
    # 🔥 SI-SDR 品質統計
    sisdr_stats = {
        "total_separated_files": 0,
        "good_quality_files": 0,
        "poor_quality_files": 0,
        "failed_sisdr_calc": 0,
        "sisdr_scores": [],
        "poor_quality_list": []
    }
    
    # 按語者資料夾統計
    speaker_stats = {}
    for result in results:
        speaker_folder = result.get("speaker_folder", "unknown")
        if speaker_folder not in speaker_stats:
            speaker_stats[speaker_folder] = {
                "total": 0, "success": 0, "failed": 0,
                "good_quality": 0, "poor_quality": 0
            }
        
        speaker_stats[speaker_folder]["total"] += 1
        if result.get("status") == "success":
            speaker_stats[speaker_folder]["success"] += 1
        else:
            speaker_stats[speaker_folder]["failed"] += 1
            
        # 統計分離檔案的品質
        for sep_file in result.get("separated_files", []):
            sisdr_stats["total_separated_files"] += 1
            sisdr = sep_file.get("si_sdr_db")
            
            if sisdr is not None:
                sisdr_stats["sisdr_scores"].append(sisdr)
                if sep_file.get("quality_good", False):
                    sisdr_stats["good_quality_files"] += 1
                    speaker_stats[speaker_folder]["good_quality"] += 1
                else:
                    sisdr_stats["poor_quality_files"] += 1
                    speaker_stats[speaker_folder]["poor_quality"] += 1
                    # 記錄低品質檔案
                    sisdr_stats["poor_quality_list"].append({
                        "file": sep_file.get("file_path", "unknown"),
                        "si_sdr_db": sisdr,
                        "speaker_folder": speaker_folder,
                        "warning": sep_file.get("quality_warning", "")
                    })
            else:
                sisdr_stats["failed_sisdr_calc"] += 1
    
    processing_times = []
    total_audio_duration = 0
    
    for result in results:
        if "processing_time" in result:
            processing_times.append(result["processing_time"])
        if "original_duration" in result:
            total_audio_duration += result["original_duration"]
    
    # 計算平均處理時間
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    # 計算 SI-SDR 統計
    sisdr_statistics = {}
    if sisdr_stats["sisdr_scores"]:
        scores = sisdr_stats["sisdr_scores"]
        sisdr_statistics = {
            "平均 SI-SDR": f"{np.mean(scores):.2f} dB",
            "最高 SI-SDR": f"{max(scores):.2f} dB",
            "最低 SI-SDR": f"{min(scores):.2f} dB",
            "標準差": f"{np.std(scores):.2f} dB"
        }
    
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
        "語者資料夾統計": speaker_stats,
        "處理時間範圍": {
            "最快": f"{min(processing_times):.3f} 秒" if processing_times else "N/A",
            "最慢": f"{max(processing_times):.3f} 秒" if processing_times else "N/A"
        },
        "SI-SDR 品質統計": {
            "總分離檔案數": sisdr_stats["total_separated_files"],
            "高品質檔案數": sisdr_stats["good_quality_files"],
            "低品質檔案數": sisdr_stats["poor_quality_files"],
            "SI-SDR計算失敗": sisdr_stats["failed_sisdr_calc"],
            "品質良好率": f"{(sisdr_stats['good_quality_files']/max(1,sisdr_stats['total_separated_files'])*100):.1f}%",
            "SI-SDR門檻": f"{SISDR_THRESHOLD} dB",
            **sisdr_statistics
        },
        "低品質檔案清單": sisdr_stats["poor_quality_list"] if SAVE_LOW_QUALITY_LIST else "已停用"
    }
    
    return summary

def batch_single_speaker_separation(input_root: str, output_root: str):
    """
    批次單語者音頻分離功能
    
    Args:
        input_root: 輸入根資料夾路徑
        output_root: 輸出根資料夾路徑
    """
    print("🚀 批次單語者音頻分離處理工具")
    print(f"輸入根資料夾: {input_root}")
    print(f"輸出根資料夾: {output_root}")
    print(f"每個語者最大檔案數: {MAX_FILES_PER_SPEAKER}")
    print(f"降噪功能: {'啟用' if ENABLE_NOISE_REDUCTION else '停用'}")
    print(f"使用固定2人模型: {'是' if USE_FIXED_2SPK_MODEL else '否'}")
    print("=" * 80)
    
    # 檢查輸入資料夾
    if not os.path.exists(input_root):
        print(f"❌ 錯誤：輸入根資料夾不存在 - {input_root}")
        return
    
    # 創建輸出資料夾
    os.makedirs(output_root, exist_ok=True)
    
    # 搜索語者資料夾
    speaker_folders = find_speaker_folders(input_root)
    if not speaker_folders:
        print("❌ 錯誤：找不到任何語者資料夾")
        return
    
    # 收集所有要處理的音檔
    all_audio_files = []
    input_root_path = Path(input_root)
    output_root_path = Path(output_root)
    
    for speaker_folder in speaker_folders:
        audio_files = find_audio_files_in_speaker_folder(speaker_folder, MAX_FILES_PER_SPEAKER)
        for audio_file in audio_files:
            output_base = get_output_path(audio_file, input_root_path, output_root_path)
            all_audio_files.append((audio_file, output_base))
    
    if not all_audio_files:
        print("❌ 錯誤：找不到任何支援的音檔")
        return
    
    print(f"📊 總計要處理 {len(all_audio_files)} 個音檔")
    
    # 設定輸出檔案名稱
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"batch_single_speaker_separation_{timestamp}"
    
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
        
        # 根據設定選擇模型
        if USE_FIXED_2SPK_MODEL:
            separator = AudioSeparator(
                model_type=SeparationModel.SEPFORMER_2SPEAKER,
                enable_dynamic_model=False,  # 固定模型
                enable_noise_reduction=ENABLE_NOISE_REDUCTION,
                snr_threshold=SNR_THRESHOLD
            )
            print("✅ 音頻分離器初始化完成 (固定2人模型)")
        else:
            # 🔥 使用動態模型但限制最大語者數
            separator = AudioSeparator(
                model_type=SeparationModel.SEPFORMER_2SPEAKER,  # 預設從2人模型開始
                enable_dynamic_model=True,   # 啟用動態選擇
                enable_noise_reduction=ENABLE_NOISE_REDUCTION,
                snr_threshold=SNR_THRESHOLD
            )
            print(f"✅ 音頻分離器初始化完成")
        
        # 開始批次處理
        total_start_time = time.time()
        
        for i, (audio_file, output_base) in enumerate(all_audio_files[start_index:], start_index):
            print(f"\n進度: {i + 1}/{len(all_audio_files)}")
            
            # 處理單個檔案
            result = separate_single_speaker_audio(separator, audio_file, output_base)
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
                    if isinstance(sub_value, dict):
                        print(f"  {sub_key}:")
                        for subsub_key, subsub_value in sub_value.items():
                            print(f"    {subsub_key}: {subsub_value}")
                    else:
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
    print("🎵 批次單語者音頻分離處理工具")
    print("=" * 80)
    
    # 提示用戶可以修改設定
    print(f"📂 輸入根資料夾: {INPUT_ROOT_DIRECTORY}")
    print(f"📤 輸出根資料夾: {OUTPUT_ROOT_DIRECTORY}")
    print(f"🔢 每個語者最大檔案數: {MAX_FILES_PER_SPEAKER}")
    print(f"🎯 語者資料夾模式: {SPEAKER_FOLDER_PATTERN}")
    print(f"🔧 降噪功能: {'啟用' if ENABLE_NOISE_REDUCTION else '停用'}")
    print(f"🎛️  固定2人模型: {'是' if USE_FIXED_2SPK_MODEL else '否'}")
    print(f"� SI-SDR品質評估: {'啟用' if ENABLE_SISDR_EVALUATION else '停用'}")
    print(f"🚨 SI-SDR門檻: {SISDR_THRESHOLD} dB")
    print(f"⚠️  低品質警告: {'啟用' if WARN_LOW_QUALITY else '停用'}")
    print(f"�🔧 詳細輸出: {'開啟' if ENABLE_VERBOSE else '關閉'}")
    print(f"💾 保存結果: {'開啟' if SAVE_DETAILED_RESULTS else '關閉'}")
    print(f"🔄 進度保存: {'開啟' if ENABLE_PROGRESS_SAVE else '關閉'}")
    print()
    
    if not os.path.exists(INPUT_ROOT_DIRECTORY):
        print("💡 提示：找不到預設輸入根資料夾")
        print("請編輯此檔案的 INPUT_ROOT_DIRECTORY 變數，指向您的語者資料集根資料夾")
        print("預期結構: 母資料夾/speakerX/utts/xxx.wav")
        print()
        return
    
    batch_single_speaker_separation(INPUT_ROOT_DIRECTORY, OUTPUT_ROOT_DIRECTORY)

if __name__ == "__main__":
    main()
