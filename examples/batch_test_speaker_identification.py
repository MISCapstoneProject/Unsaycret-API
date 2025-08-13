#!/usr/bin/env python3
"""
批次語者識別測試檔（完整識別流程版）
================

批次處理指定資料夾內所有 .wav 檔案的語者識別功能，支援遞歸搜索子資料夾。
使用完整的識別流程，模擬真實 API 使用情況，會自動處理新語者註冊和現有語者更新。

使用方法：
1. 修改下方的 TARGET_DIRECTORY 變數
2. 調整 MAX_FILES_PER_FOLDER 來限制每個資料夾處理的檔案數量（測試用）
3. 執行: python batch_test_speaker_identification.py

功能特色：
- 遞歸處理整個資料夾及其子資料夾的 .wav 檔案
- 使用完整的識別流程（process_audio_file），模擬真實使用場景
- 自動註冊新語者、更新現有語者聲紋
- 顯示詳細的處理結果和動作描述
- 每個資料夾檔案數量限制（可設定前n個檔案進行快速測試）
- 生成詳細的處理報告與統計資訊，包含新創建語者列表
- 支援中斷恢復（可選）

⚠️ 注意：此版本會實際修改資料庫，新增語者和聲紋資料！
"""

import os
import json
import time
import traceback
import soundfile as sf
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.identification.VID_identify_v5 import SpeakerIdentifier

# ==================== 設定區域 ====================
# 👇 請修改這裡的目標資料夾路徑
TARGET_DIRECTORY = "separated_output"  # 修改為您要批次處理的資料夾路徑

# 其他設定
ENABLE_VERBOSE = True          # 是否顯示詳細輸出
SAVE_DETAILED_RESULTS = True   # 是否保存詳細結果到 JSON 檔案
SAVE_SUMMARY_RESULTS = True    # 是否保存摘要結果
ENABLE_PROGRESS_SAVE = True    # 是否啟用進度保存（可中斷恢復）
PROGRESS_SAVE_INTERVAL = 10    # 每處理幾個檔案保存一次進度
MAX_FILES_PER_FOLDER = 6       # 每個資料夾最多處理的檔案數量（0 = 無限制）
                              # 例如：設定為 5，則每個子資料夾只處理前 5 個音檔
                              # 用於快速測試，避免處理太多檔案

# 檔案篩選設定
SUPPORTED_EXTENSIONS = ['.wav', '.WAV']  # 支援的音檔副檔名
SKIP_HIDDEN_FILES = True       # 是否跳過隱藏檔案（以 . 開頭）
# ==================================================

def generate_voiceprint_name(file_path: str, base_directory: str) -> str:
    """
    根據檔案路徑生成聲紋名稱
    
    規則：只使用音檔名稱（無副檔名）作為聲紋名稱
    
    Args:
        file_path: 完整檔案路徑
        base_directory: 基準資料夾路徑（此參數保留以維持兼容性）
        
    Returns:
        str: 生成的聲紋名稱（僅音檔名稱）
    """
    file_path = Path(file_path)
    # 只返回檔案名稱（無副檔名）
    return file_path.stem

def find_all_audio_files(directory: str) -> List[str]:
    """
    遞歸搜索指定資料夾內的所有音檔，支援每個資料夾檔案數量限制
    
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
    if MAX_FILES_PER_FOLDER > 0:
        print(f"📝 每個資料夾最多處理 {MAX_FILES_PER_FOLDER} 個檔案")
    
    # 遞歸搜索所有音檔
    for root, dirs, files in os.walk(directory):
        # 排序資料夾和檔案以確保一致的處理順序
        dirs.sort()
        files.sort()
        
        # 當前資料夾的音檔
        folder_audio_files = []
        
        for file in files:
            file_path = Path(root) / file
            
            # 跳過隱藏檔案（如果啟用）
            if SKIP_HIDDEN_FILES and file.startswith('.'):
                continue
            
            # 檢查副檔名
            if file_path.suffix in SUPPORTED_EXTENSIONS:
                folder_audio_files.append(str(file_path))
        
        # 排序當前資料夾的音檔
        folder_audio_files.sort()
        
        # 限制每個資料夾的檔案數量
        if MAX_FILES_PER_FOLDER > 0 and len(folder_audio_files) > MAX_FILES_PER_FOLDER:
            original_count = len(folder_audio_files)
            folder_audio_files = folder_audio_files[:MAX_FILES_PER_FOLDER]
            print(f"📁 {root}: 限制檔案數量 {original_count} → {len(folder_audio_files)}")
        elif len(folder_audio_files) > 0:
            print(f"📁 {root}: {len(folder_audio_files)} 個音檔")
        
        # 添加到總列表
        audio_files.extend(folder_audio_files)
    
    # 按路徑排序
    audio_files.sort()
    
    print(f"📁 總共找到 {len(audio_files)} 個音檔")
    return audio_files

def get_action_description(status: str) -> str:
    """
    根據狀態返回動作描述
    
    Args:
        status: 處理狀態
        
    Returns:
        str: 動作描述
    """
    action_map = {
        "new_speaker": "🆕 創建新語者",
        "exact_match": "✅ 完全匹配，無需更新",
        "updated": "🔄 更新現有語者的聲紋向量",
        "added_voiceprint": "📁 為現有語者新增聲紋",
    }
    return action_map.get(status, "❓ 未知動作")

def test_single_audio_file(identifier: SpeakerIdentifier, 
                          audio_path: str, 
                          base_directory: str) -> Dict:
    """
    測試單個音檔的語者識別（使用完整的識別流程）
    
    Args:
        identifier: 語者識別器實例
        audio_path: 音檔路徑
        base_directory: 基準資料夾路徑
        
    Returns:
        Dict: 測試結果字典
    """
    import soundfile as sf
    
    start_time = time.time()
    voiceprint_name = generate_voiceprint_name(audio_path, base_directory)
    
    print(f"\n{'='*80}")
    print(f"🎯 處理音檔: {Path(audio_path).name}")
    print(f"📂 完整路徑: {audio_path}")
    print(f"🏷️  聲紋名稱: {voiceprint_name}")
    print(f"{'='*80}")
    
    result_dict = {
        "audio_file": audio_path,
        "voiceprint_name": voiceprint_name,
        "relative_path": str(Path(audio_path).relative_to(base_directory)),
        "processing_time": 0,
        "timestamp": datetime.now().isoformat(),
        "comparison_details": []  # 新增比對詳情
    }
    
    try:
        # 先獲取比對距離詳情（不進行實際註冊）
        signal, sr = sf.read(audio_path)
        new_embedding = identifier.audio_processor.extract_embedding_from_stream(signal, sr)
        
        # 與 Weaviate 中的嵌入向量比對，獲取所有距離資訊
        best_id, best_name, best_distance, all_distances = identifier.database.compare_embedding(new_embedding)
        
        # 記錄比對詳情
        comparison_details = []
        if all_distances:
            for match_id, match_name, distance, update_count in all_distances:
                comparison_details.append({
                    "speaker_id": match_id,
                    "speaker_name": match_name,
                    "distance": round(distance, 4),
                    "update_count": update_count
                })
        
        result_dict["comparison_details"] = comparison_details
        result_dict["best_comparison"] = {
            "speaker_id": best_id,
            "speaker_name": best_name,
            "distance": best_distance
        } if best_id is not None else None
        
        # 然後使用完整的識別流程（這會自動處理新語者註冊、現有語者更新等）
        result = identifier.process_audio_file(audio_path)
        
        if result is not None:
            speaker_id, speaker_name, distance = result
            
            # 判斷處理狀態
            if distance == -1:
                status = "new_speaker"
                status_desc = "新語者"
            elif distance < 0.26:  # THRESHOLD_LOW
                status = "exact_match"
                status_desc = "完全匹配（無需更新）"
            elif distance < 0.34:  # THRESHOLD_UPDATE
                status = "updated"
                status_desc = "已更新現有語者聲紋"
            elif distance < 0.385:  # THRESHOLD_NEW
                status = "added_voiceprint"
                status_desc = "已為現有語者新增聲紋"
            else:
                status = "new_speaker"
                status_desc = "識別為新語者"
            
            # 準備成功結果
            result_dict.update({
                "status": "success",
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "distance": round(distance, 4),
                "action_status": status,
                "action_description": status_desc
            })
            
            # 顯示結果摘要
            print(f"\n🎵 音檔名稱: {Path(audio_path).name}")
            print(f"✅ 識別成功！")
            
            # 顯示比對詳情（前3個）
            if comparison_details:
                print(f"🔍 比對結果:")
                for i, detail in enumerate(comparison_details[:3], 1):
                    speaker_id_short = detail['speaker_id'][:8] + "..." if len(detail['speaker_id']) > 8 else detail['speaker_id']
                    print(f"   {i}. {detail['speaker_name']} (距離: {detail['distance']}, ID: {speaker_id_short})")
            
            print(f"📋 處理結果: {status_desc}")
            print(f"   ├─ 語者 ID: {speaker_id}")
            print(f"   ├─ 語者名稱: {speaker_name}")
            print(f"   ├─ 處理距離: {distance}")
            print(f"   └─ 執行動作: {get_action_description(status)}")
            
        else:
            # 處理失敗
            result_dict.update({
                "status": "error",
                "error": "識別處理失敗",
                "speaker_id": None,
                "speaker_name": None,
                "distance": None,
                "action_status": "failed",
                "action_description": "處理失敗"
            })
            print(f"\n🎵 音檔名稱: {Path(audio_path).name}")
            print(f"❌ 處理失敗：識別處理失敗")
            
    except Exception as e:
        error_msg = f"處理失敗: {str(e)}"
        print(f"❌ {error_msg}")
        if ENABLE_VERBOSE:
            print("🔍 錯誤詳情:")
            traceback.print_exc()
        
        result_dict.update({
            "status": "error",
            "error": error_msg,
            "speaker_id": None,
            "speaker_name": None,
            "distance": None,
            "action_status": "error",
            "action_description": "發生錯誤"
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
    
    # 統計各種狀態
    status_counts = {}
    processing_times = []
    speakers_created = []
    
    for result in results:
        if result.get("status") == "success":
            action_status = result.get("action_status", "unknown")
            status_counts[action_status] = status_counts.get(action_status, 0) + 1
            
            # 收集新創建的語者
            if action_status == "new_speaker" and result.get("speaker_name"):
                speakers_created.append(result["speaker_name"])
        
        if "processing_time" in result:
            processing_times.append(result["processing_time"])
    
    # 計算平均處理時間
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    summary = {
        "總處理時間": f"{total_time:.2f} 秒",
        "總檔案數": total_files,
        "成功處理": successful_files,
        "失敗檔案": failed_files,
        "成功率": f"{(successful_files/total_files*100):.1f}%" if total_files > 0 else "0%",
        "平均單檔處理時間": f"{avg_processing_time:.3f} 秒",
        "動作統計": status_counts,
        "新創建語者數": len(set(speakers_created)),  # 去重計算
        "新創建語者列表": list(set(speakers_created)) if speakers_created else [],
        "處理時間範圍": {
            "最快": f"{min(processing_times):.3f} 秒" if processing_times else "N/A",
            "最慢": f"{max(processing_times):.3f} 秒" if processing_times else "N/A"
        }
    }
    
    return summary

def batch_test_speaker_identification(target_directory: str):
    """
    批次測試語者識別功能
    
    Args:
        target_directory: 目標資料夾路徑
    """
    print("🚀 批次語者識別測試工具")
    print(f"目標資料夾: {target_directory}")
    print("=" * 80)
    
    # 檢查目標資料夾
    if not os.path.exists(target_directory):
        print(f"❌ 錯誤：目標資料夾不存在 - {target_directory}")
        return
    
    # 搜索所有音檔
    audio_files = find_all_audio_files(target_directory)
    if not audio_files:
        print("❌ 錯誤：找不到任何支援的音檔")
        return
    
    # 設定輸出檔案名稱
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"batch_test_results_{timestamp}"
    
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
        print("🔄 初始化語者識別器...")
        identifier = SpeakerIdentifier()
        identifier.set_verbose(ENABLE_VERBOSE)
        print("✅ 語者識別器初始化完成")
        
        # 開始批次處理
        total_start_time = time.time()
        
        for i, audio_file in enumerate(audio_files[start_index:], start_index):
            print(f"\n進度: {i + 1}/{len(audio_files)}")
            
            # 處理單個檔案
            result = test_single_audio_file(identifier, audio_file, target_directory)
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
    print("🎵 批次語者識別測試工具")
    print("=" * 80)
    
    # 提示用戶可以修改設定
    print(f"📂 目標資料夾: {TARGET_DIRECTORY}")
    print(f" 詳細輸出: {'開啟' if ENABLE_VERBOSE else '關閉'}")
    print(f"💾 保存結果: {'開啟' if SAVE_DETAILED_RESULTS else '關閉'}")
    print(f"🔄 進度保存: {'開啟' if ENABLE_PROGRESS_SAVE else '關閉'}")
    print()
    
    if not os.path.exists(TARGET_DIRECTORY):
        print("💡 提示：找不到預設目標資料夾")
        print("請編輯此檔案的 TARGET_DIRECTORY 變數，指向您想批次處理的資料夾")
        print()
        return
    
    batch_test_speaker_identification(TARGET_DIRECTORY)

if __name__ == "__main__":
    main()
