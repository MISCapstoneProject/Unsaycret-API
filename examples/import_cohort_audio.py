#!/usr/bin/env python3
"""
導入音檔到 AS-Norm Cohort 資料庫
==================================

將分離後的音檔批量導入到 AS-Norm 專用的 cohort 資料庫中，
用於語者識別系統的分數正規化。

功能特色：
- 🎯 批量處理音檔資料夾
- 📊 即時進度顯示
- 🔍 重複檔案檢測
- 📈 詳細統計報告
- 💾 匯出處理記錄

使用方法：
1. 確保 Weaviate 資料庫已            import_record = {
                "file": str(audio_file),
                "source_dataset": source_dataset,
                "success": result_count > 0,
                "embeddings_count": result_count,
                "timestamp": get_taipei_time().isoformat()
            }改下方的設定參數
3. 執行: python examples/import_cohort_audio.py
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.database.cohort_manager import CohortDatabaseManager, get_taipei_time

# ==================== 設定區域 ====================
# 👇 請修改這裡的設定

# 音檔資料夾設定
AUDIO_FOLDER = "dataset_separated_single_tcc300"  # 音檔根目錄
SOURCE_DATASET_PREFIX = "TCC-300"  # 來源資料集前綴

# 元數據設定
METADATA = {
    "gender": "mixed",           # 性別：mixed, male, female
    "language": "zh-TW",         # 語言：zh-TW, zh-CN, en, etc.
    "description": "TCC-300 分離後的單語者音檔用於 AS-Norm cohort"
}

# 處理設定
AUDIO_EXTENSIONS = ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC']
ENABLE_PROGRESS_DISPLAY = True   # 是否顯示詳細進度
SAVE_IMPORT_LOG = True          # 是否保存導入記錄
BATCH_SIZE = 10                 # 批次處理大小（每N個檔案顯示一次進度）

# 安全設定
CHECK_EXISTING = True           # 是否檢查重複檔案
RESET_BEFORE_IMPORT = False     # 是否在導入前重置 cohort 資料庫
# ==================================================

class CohortImportProgress:
    """進度顯示器"""
    
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.processed_files = 0
        self.success_files = 0
        self.failed_files = 0
        self.start_time = time.time()
        self.last_update = time.time()
    
    def update(self, success: bool = True):
        """更新進度"""
        self.processed_files += 1
        if success:
            self.success_files += 1
        else:
            self.failed_files += 1
        
        # 每批次或最後一個檔案顯示進度
        if (self.processed_files % BATCH_SIZE == 0 or 
            self.processed_files == self.total_files):
            self.display_progress()
    
    def display_progress(self):
        """顯示進度資訊"""
        if not ENABLE_PROGRESS_DISPLAY:
            return
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # 計算進度百分比
        progress_pct = (self.processed_files / self.total_files) * 100
        
        # 計算預估剩餘時間
        if self.processed_files > 0:
            avg_time_per_file = elapsed / self.processed_files
            remaining_files = self.total_files - self.processed_files
            eta_seconds = avg_time_per_file * remaining_files
            eta_minutes = eta_seconds / 60
        else:
            eta_minutes = 0
        
        print(f"\n📊 進度報告 ({self.processed_files}/{self.total_files}, {progress_pct:.1f}%)")
        print(f"   ✅ 成功: {self.success_files}")
        print(f"   ❌ 失敗: {self.failed_files}")
        print(f"   ⏱️  已耗時: {elapsed/60:.1f} 分鐘")
        print(f"   🕐 預估剩餘: {eta_minutes:.1f} 分鐘")
        
        self.last_update = current_time

def check_audio_folder(folder_path: str) -> Dict[str, Any]:
    """
    檢查音檔資料夾並收集統計資訊
    
    Args:
        folder_path: 音檔資料夾路徑
        
    Returns:
        Dict[str, Any]: 資料夾統計資訊
    """
    folder = Path(folder_path)
    if not folder.exists():
        return {
            "exists": False,
            "error": f"資料夾不存在: {folder_path}"
        }
    
    print(f"🔍 掃描音檔資料夾: {folder_path}")
    
    # 統計音檔
    audio_files = []
    speaker_folders = {}
    total_size = 0
    
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(folder.rglob(f"*{ext}"))
    
    # 分析檔案
    for file_path in audio_files:
        # 計算檔案大小
        try:
            size = file_path.stat().st_size
            total_size += size
        except:
            size = 0
        
        # 分析語者資料夾
        relative_path = file_path.relative_to(folder)
        speaker_folder = relative_path.parts[0] if relative_path.parts else "unknown"
        
        if speaker_folder not in speaker_folders:
            speaker_folders[speaker_folder] = 0
        speaker_folders[speaker_folder] += 1
    
    return {
        "exists": True,
        "total_files": len(audio_files),
        "total_size_mb": total_size / (1024 * 1024),
        "speaker_folders": len(speaker_folders),
        "speaker_distribution": speaker_folders,
        "audio_files": audio_files
    }

def check_existing_cohort(manager: CohortDatabaseManager) -> Dict[str, Any]:
    """
    檢查現有的 cohort 資料庫狀態
    
    Args:
        manager: Cohort 資料庫管理器
        
    Returns:
        Dict[str, Any]: 現有資料庫狀態
    """
    print("🔍 檢查現有 cohort 資料庫...")
    
    stats = manager.get_cohort_statistics()
    if stats.get("exists", False):
        print(f"📊 現有 cohort 統計:")
        print(f"   - 總聲紋數: {stats.get('total_count', 0)}")
        print(f"   - 資料集數: {len(stats.get('source_datasets', {}))}")
        
        if stats.get('source_datasets'):
            print(f"   - 來源分布:")
            for dataset, count in stats.get('source_datasets', {}).items():
                print(f"     • {dataset}: {count}")
    else:
        print("📝 cohort 資料庫為空或不存在")
    
    return stats

def import_audio_files(manager: CohortDatabaseManager, audio_files: List[Path], 
                      folder_path: str) -> Dict[str, Any]:
    """
    批量導入音檔
    
    Args:
        manager: Cohort 資料庫管理器
        audio_files: 音檔路徑列表
        folder_path: 音檔根目錄
        
    Returns:
        Dict[str, Any]: 導入結果
    """
    print(f"\n🚀 開始批量導入 {len(audio_files)} 個音檔...")
    
    # 初始化進度追蹤
    progress = CohortImportProgress(len(audio_files))
    import_log = []
    
    success_count = 0
    failed_files = []
    
    for i, audio_file in enumerate(audio_files, 1):
        try:
            # 準備檔案專屬元數據
            file_metadata = METADATA.copy()
            file_metadata.update({
                "original_file": audio_file.name,
                "file_path": str(audio_file.relative_to(folder_path)),
                "import_timestamp": get_taipei_time().isoformat()
            })
            
            # 決定 source_dataset 名稱
            if SOURCE_DATASET_PREFIX:
                source_dataset = f"{SOURCE_DATASET_PREFIX}_{audio_file.stem}"
            else:
                source_dataset = audio_file.stem
            
            # 導入單個檔案
            result_count = manager.import_audio_file(
                str(audio_file), 
                source_dataset, 
                file_metadata
            )
            
            # 記錄結果
            log_entry = {
                "file": str(audio_file),
                "source_dataset": source_dataset,
                "success": result_count > 0,
                "embeddings_count": result_count,
                "timestamp": datetime.now().isoformat()
            }
            
            if result_count > 0:
                success_count += 1
                progress.update(success=True)
                if ENABLE_PROGRESS_DISPLAY and i <= 5:  # 只顯示前5個詳細訊息
                    print(f"   ✅ {audio_file.name} -> {source_dataset}")
            else:
                failed_files.append(str(audio_file))
                log_entry["error"] = "無法提取聲紋"
                progress.update(success=False)
                print(f"   ❌ {audio_file.name} - 無法提取聲紋")
            
            import_log.append(log_entry)
            
        except Exception as e:
            failed_files.append(str(audio_file))
            log_entry = {
                "file": str(audio_file),
                "success": False,
                "error": str(e),
                "timestamp": get_taipei_time().isoformat()
            }
            import_log.append(log_entry)
            progress.update(success=False)
            print(f"   ❌ {audio_file.name} - {e}")
    
    # 最終進度顯示
    progress.display_progress()
    
    return {
        "total_files": len(audio_files),
        "success_files": success_count,
        "failed_files": len(failed_files),
        "success_rate": (success_count / len(audio_files)) * 100 if audio_files else 0,
        "failed_file_list": failed_files,
        "import_log": import_log,
        "total_time": time.time() - progress.start_time
    }

def save_import_log(results: Dict[str, Any]) -> str:
    """保存導入記錄"""
    if not SAVE_IMPORT_LOG:
        return ""
    
    timestamp = get_taipei_time().strftime("%Y%m%d_%H%M%S")
    log_file = f"cohort_import_log_{timestamp}.json"
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 導入記錄已保存: {log_file}")
        return log_file
        
    except Exception as e:
        print(f"⚠️  保存導入記錄失敗: {e}")
        return ""

def main():
    """主函數"""
    print("🎵 AS-Norm Cohort 資料庫音檔導入工具")
    print("=" * 80)
    print(f"📂 音檔資料夾: {AUDIO_FOLDER}")
    print(f"🏷️  資料集前綴: {SOURCE_DATASET_PREFIX}")
    print(f"🔧 批次大小: {BATCH_SIZE}")
    print(f"🛡️  重置資料庫: {'是' if RESET_BEFORE_IMPORT else '否'}")
    print(f"📊 進度顯示: {'開啟' if ENABLE_PROGRESS_DISPLAY else '關閉'}")
    print(f"💾 保存記錄: {'開啟' if SAVE_IMPORT_LOG else '關閉'}")
    print()
    
    try:
        # 步驟1: 檢查音檔資料夾
        folder_info = check_audio_folder(AUDIO_FOLDER)
        if not folder_info["exists"]:
            print(f"❌ {folder_info['error']}")
            return
        
        print(f"📊 資料夾統計:")
        print(f"   - 音檔數量: {folder_info['total_files']}")
        print(f"   - 總大小: {folder_info['total_size_mb']:.2f} MB")
        print(f"   - 語者資料夾: {folder_info['speaker_folders']}")
        
        if folder_info['total_files'] == 0:
            print("❌ 沒有找到任何音檔")
            return
        
        # 步驟2: 初始化 cohort 管理器（使用 pyannote 模型）
        print(f"\n🔧 初始化 cohort 資料庫管理器...")
        manager = CohortDatabaseManager(model_type="pyannote")
        
        # 步驟3: 檢查現有資料庫
        existing_stats = check_existing_cohort(manager)
        
        # 步驟4: 重置資料庫（如果需要）
        if RESET_BEFORE_IMPORT:
            print(f"\n🗑️  重置 cohort 資料庫...")
            if manager.reset_cohort_collection():
                print("✅ 資料庫重置成功")
            else:
                print("❌ 資料庫重置失敗")
                return
        
        # 步驟5: 確認導入
        print(f"\n⚠️  即將導入 {folder_info['total_files']} 個音檔到 cohort 資料庫")
        print(f"來源: {AUDIO_FOLDER}")
        print(f"前綴: {SOURCE_DATASET_PREFIX}")
        
        # 在實際環境中可以加入確認提示
        # confirm = input("確定要繼續嗎？(y/N): ")
        # if confirm.lower() != 'y':
        #     print("取消導入")
        #     return
        
        # 步驟6: 執行導入
        start_time = time.time()
        results = import_audio_files(manager, folder_info['audio_files'], AUDIO_FOLDER)
        
        # 步驟7: 顯示結果
        print(f"\n" + "=" * 80)
        print("📈 導入完成！")
        print("=" * 80)
        print(f"總檔案數: {results['total_files']}")
        print(f"成功導入: {results['success_files']}")
        print(f"失敗檔案: {results['failed_files']}")
        print(f"成功率: {results['success_rate']:.1f}%")
        print(f"總耗時: {results['total_time']:.2f} 秒")
        
        if results['failed_files'] > 0:
            print(f"\n❌ 失敗檔案列表 (前10個):")
            for failed_file in results['failed_file_list'][:10]:
                print(f"   - {failed_file}")
            if len(results['failed_file_list']) > 10:
                print(f"   ... 還有 {len(results['failed_file_list']) - 10} 個")
        
        # 步驟8: 檢查最終統計
        final_stats = manager.get_cohort_statistics()
        print(f"\n📊 更新後的 cohort 統計:")
        print(f"   - 總聲紋數: {final_stats.get('total_count', 0)}")
        print(f"   - 資料集數: {len(final_stats.get('source_datasets', {}))}")
        
        # 步驟9: 保存記錄
        log_file = save_import_log(results)
        
        # 步驟10: 清理
        manager.close()
        print(f"\n✅ 導入程序完成")
        
    except Exception as e:
        print(f"\n❌ 導入過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
