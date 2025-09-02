#!/usr/bin/env python3
"""
批次音檔重新命名工具
===================

移除分離後音檔的語者後綴（_speaker1, _speaker2 等），
將檔案名稱還原為原始格式。

例如：
speaker5_utt1_speaker1.wav → speaker5_utt1.wav
speaker10_utt5_speaker2.wav → speaker10_utt5.wav

使用方法：
1. 修改下方的 ROOT_DIRECTORY 變數，指向您的音檔根目錄
2. 執行: python batch_rename_audio_files.py

功能特色：
- 🔍 遞歸搜索所有子資料夾中的音檔
- 🎯 自動識別 _speaker1, _speaker2 等後綴
- 🛡️ 安全模式：先預覽重新命名計畫，確認後才執行
- 📊 詳細統計報告
- 💾 支援復原功能（產生重新命名記錄）
- ⚠️ 檔名衝突檢測
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import re

# ==================== 設定區域 ====================
# 👇 請修改這裡的路徑
ROOT_DIRECTORY = "dataset_separated_single_tcc300"  # 音檔根目錄

# 處理設定
SUPPORTED_EXTENSIONS = ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC']  # 支援的音檔格式
SPEAKER_SUFFIX_PATTERN = r'_speaker\d+$'  # 要移除的後綴模式 (_speaker1, _speaker2, etc.)
DRY_RUN = False  # 預覽模式（True=只顯示計畫，False=實際執行）
SAVE_RENAME_LOG = True  # 是否保存重新命名記錄（用於復原）
ENABLE_BACKUP = False  # 是否建立備份（謹慎使用，會佔用大量空間）

# 顯示設定
ENABLE_VERBOSE = True  # 詳細輸出
SHOW_PREVIEW_LIMIT = 20  # 預覽模式最多顯示的檔案數量
# ==================================================

def find_audio_files_with_speaker_suffix(root_dir: str) -> List[Path]:
    """
    尋找所有包含 speaker 後綴的音檔
    
    Args:
        root_dir: 根目錄路徑
        
    Returns:
        List[Path]: 符合條件的音檔路徑列表
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"❌ 錯誤：根目錄不存在 - {root_dir}")
        return []
    
    print(f"🔍 搜索包含 speaker 後綴的音檔: {root_dir}")
    
    audio_files = []
    pattern = re.compile(SPEAKER_SUFFIX_PATTERN)
    
    # 遞歸搜索所有音檔
    for file_path in root_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
            # 檢查檔名（不含副檔名）是否包含 speaker 後綴
            stem = file_path.stem
            if pattern.search(stem):
                audio_files.append(file_path)
    
    print(f"📁 找到 {len(audio_files)} 個需要重新命名的音檔")
    return audio_files

def generate_new_filename(file_path: Path) -> Tuple[Path, str]:
    """
    產生新的檔案名稱（移除 speaker 後綴）
    
    Args:
        file_path: 原始檔案路徑
        
    Returns:
        Tuple[Path, str]: (新檔案路徑, 移除的後綴)
    """
    pattern = re.compile(SPEAKER_SUFFIX_PATTERN)
    original_stem = file_path.stem
    
    # 尋找並移除 speaker 後綴
    match = pattern.search(original_stem)
    if match:
        removed_suffix = match.group()
        new_stem = original_stem[:match.start()]
        new_filename = new_stem + file_path.suffix
        new_path = file_path.parent / new_filename
        return new_path, removed_suffix
    else:
        # 如果沒找到匹配（理論上不應該發生）
        return file_path, ""

def check_conflicts(rename_plan: List[Dict]) -> List[Dict]:
    """
    檢查檔名衝突
    
    Args:
        rename_plan: 重新命名計畫列表
        
    Returns:
        List[Dict]: 衝突檔案列表
    """
    conflicts = []
    target_paths = {}
    
    for plan in rename_plan:
        new_path = plan["new_path"]
        
        # 檢查目標路徑是否已存在於檔案系統中
        if Path(new_path).exists():
            conflicts.append({
                **plan,
                "conflict_type": "file_exists",
                "conflict_message": f"目標檔案已存在: {new_path}"
            })
        
        # 檢查是否有多個檔案要重新命名為同一個名稱
        if new_path in target_paths:
            conflicts.append({
                **plan,
                "conflict_type": "duplicate_target",
                "conflict_message": f"多個檔案要重新命名為: {new_path}",
                "conflicting_with": target_paths[new_path]
            })
        else:
            target_paths[new_path] = plan["original_path"]
    
    return conflicts

def create_rename_plan(audio_files: List[Path]) -> Tuple[List[Dict], List[Dict]]:
    """
    建立重新命名計畫
    
    Args:
        audio_files: 音檔路徑列表
        
    Returns:
        Tuple[List[Dict], List[Dict]]: (重新命名計畫, 衝突列表)
    """
    print("📋 建立重新命名計畫...")
    
    rename_plan = []
    
    for file_path in audio_files:
        new_path, removed_suffix = generate_new_filename(file_path)
        
        # 只有當檔名確實會改變時才加入計畫
        if new_path != file_path:
            plan_item = {
                "original_path": str(file_path),
                "new_path": str(new_path),
                "original_name": file_path.name,
                "new_name": new_path.name,
                "removed_suffix": removed_suffix,
                "folder": str(file_path.parent),
                "size_bytes": file_path.stat().st_size if file_path.exists() else 0
            }
            rename_plan.append(plan_item)
    
    # 檢查衝突
    conflicts = check_conflicts(rename_plan)
    
    print(f"📊 重新命名計畫統計:")
    print(f"   - 需要重新命名的檔案: {len(rename_plan)}")
    print(f"   - 檔名衝突: {len(conflicts)}")
    
    return rename_plan, conflicts

def preview_rename_plan(rename_plan: List[Dict], conflicts: List[Dict]):
    """顯示重新命名計畫預覽"""
    print("\n" + "=" * 80)
    print("📋 重新命名計畫預覽")
    print("=" * 80)
    
    if conflicts:
        print("⚠️  檔名衝突警告:")
        for conflict in conflicts[:10]:  # 最多顯示10個衝突
            print(f"   ❌ {conflict['original_name']} → {conflict['new_name']}")
            print(f"      原因: {conflict['conflict_message']}")
        if len(conflicts) > 10:
            print(f"   ... 還有 {len(conflicts) - 10} 個衝突未顯示")
        print()
    
    if rename_plan:
        print("✅ 正常重新命名:")
        # 過濾掉衝突的檔案
        conflict_originals = {c["original_path"] for c in conflicts}
        normal_renames = [p for p in rename_plan if p["original_path"] not in conflict_originals]
        
        display_count = min(len(normal_renames), SHOW_PREVIEW_LIMIT)
        for i, plan in enumerate(normal_renames[:display_count]):
            print(f"   {i+1:3d}. {plan['original_name']} → {plan['new_name']}")
            if ENABLE_VERBOSE:
                print(f"        路徑: {plan['folder']}")
                print(f"        移除: {plan['removed_suffix']}")
        
        if len(normal_renames) > SHOW_PREVIEW_LIMIT:
            print(f"   ... 還有 {len(normal_renames) - SHOW_PREVIEW_LIMIT} 個檔案未顯示")
        
        print(f"\n📊 統計:")
        print(f"   - 成功計畫: {len(normal_renames)}")
        print(f"   - 衝突檔案: {len(conflicts)}")
        total_size = sum(plan["size_bytes"] for plan in normal_renames) / 1024 / 1024
        print(f"   - 總檔案大小: {total_size:.2f} MB")

def execute_rename_plan(rename_plan: List[Dict], conflicts: List[Dict]) -> Dict:
    """
    執行重新命名計畫
    
    Args:
        rename_plan: 重新命名計畫
        conflicts: 衝突列表
        
    Returns:
        Dict: 執行結果統計
    """
    if DRY_RUN:
        print("\n🔍 這是預覽模式，未實際執行重新命名")
        return {
            "total_planned": len(rename_plan),
            "conflicts": len(conflicts),
            "executed": 0,
            "success": 0,
            "failed": 0,
            "mode": "preview"
        }
    
    print("\n🚀 開始執行重新命名...")
    
    # 過濾掉衝突的檔案
    conflict_originals = {c["original_path"] for c in conflicts}
    executable_plans = [p for p in rename_plan if p["original_path"] not in conflict_originals]
    
    success_count = 0
    failed_count = 0
    rename_log = []
    
    for i, plan in enumerate(executable_plans, 1):
        try:
            original_path = Path(plan["original_path"])
            new_path = Path(plan["new_path"])
            
            if ENABLE_VERBOSE:
                print(f"   {i:3d}/{len(executable_plans)} {original_path.name} → {new_path.name}")
            
            # 建立備份（如果啟用）
            if ENABLE_BACKUP:
                backup_dir = original_path.parent / "backup"
                backup_dir.mkdir(exist_ok=True)
                backup_path = backup_dir / original_path.name
                shutil.copy2(original_path, backup_path)
                plan["backup_path"] = str(backup_path)
            
            # 執行重新命名
            original_path.rename(new_path)
            
            # 記錄成功
            success_count += 1
            plan["status"] = "success"
            plan["timestamp"] = datetime.now().isoformat()
            rename_log.append(plan)
            
        except Exception as e:
            failed_count += 1
            plan["status"] = "failed"
            plan["error"] = str(e)
            plan["timestamp"] = datetime.now().isoformat()
            rename_log.append(plan)
            print(f"   ❌ 失敗: {plan['original_name']} - {e}")
    
    # 保存重新命名記錄
    if SAVE_RENAME_LOG and rename_log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"rename_log_{timestamp}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(rename_log, f, ensure_ascii=False, indent=2)
        print(f"💾 重新命名記錄已保存: {log_file}")
    
    results = {
        "total_planned": len(rename_plan),
        "conflicts": len(conflicts),
        "executed": len(executable_plans),
        "success": success_count,
        "failed": failed_count,
        "mode": "execute"
    }
    
    return results

def generate_revert_script(log_file: str):
    """根據重新命名記錄產生復原腳本"""
    if not os.path.exists(log_file):
        print(f"❌ 找不到記錄檔案: {log_file}")
        return
    
    with open(log_file, 'r', encoding='utf-8') as f:
        rename_log = json.load(f)
    
    successful_renames = [log for log in rename_log if log.get("status") == "success"]
    
    if not successful_renames:
        print("📝 沒有成功的重新命名記錄需要復原")
        return
    
    revert_script = f"revert_rename_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    script_content = f'''#!/usr/bin/env python3
"""
自動產生的重新命名復原腳本
產生時間: {datetime.now().isoformat()}
原始記錄: {log_file}
"""

import os
from pathlib import Path

def revert_renames():
    """復原重新命名操作"""
    renames_to_revert = {successful_renames}
    
    success_count = 0
    failed_count = 0
    
    for rename_record in renames_to_revert:
        try:
            # 注意：復原時 new_path 變成來源，original_path 變成目標
            current_path = Path(rename_record["new_path"])
            original_path = Path(rename_record["original_path"])
            
            if current_path.exists():
                current_path.rename(original_path)
                print(f"✅ 復原: {{current_path.name}} → {{original_path.name}}")
                success_count += 1
            else:
                print(f"⚠️  檔案不存在: {{current_path}}")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ 復原失敗: {{rename_record['new_path']}} - {{e}}")
            failed_count += 1
    
    print(f"\\n📊 復原結果: 成功 {{success_count}}, 失敗 {{failed_count}}")

if __name__ == "__main__":
    print("🔄 開始復原重新命名操作...")
    revert_renames()
    print("✅ 復原操作完成")
'''
    
    with open(revert_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 設定執行權限
    os.chmod(revert_script, 0o755)
    
    print(f"📝 復原腳本已產生: {revert_script}")
    print(f"   包含 {len(successful_renames)} 個復原操作")

def main():
    """主函數"""
    print("🎵 批次音檔重新命名工具")
    print("=" * 80)
    print(f"📂 根目錄: {ROOT_DIRECTORY}")
    print(f"🔍 搜尋模式: {SPEAKER_SUFFIX_PATTERN}")
    print(f"🎯 支援格式: {', '.join(SUPPORTED_EXTENSIONS)}")
    print(f"🔧 運行模式: {'預覽模式 (安全)' if DRY_RUN else '執行模式 (實際重新命名)'}")
    print(f"💾 重新命名記錄: {'啟用' if SAVE_RENAME_LOG else '停用'}")
    print(f"🛡️ 檔案備份: {'啟用' if ENABLE_BACKUP else '停用'}")
    print()
    
    # 檢查根目錄
    if not os.path.exists(ROOT_DIRECTORY):
        print(f"❌ 錯誤：根目錄不存在 - {ROOT_DIRECTORY}")
        print("💡 請修改腳本中的 ROOT_DIRECTORY 變數")
        return
    
    # 搜尋需要重新命名的檔案
    audio_files = find_audio_files_with_speaker_suffix(ROOT_DIRECTORY)
    if not audio_files:
        print("✅ 沒有找到需要重新命名的檔案")
        return
    
    # 建立重新命名計畫
    rename_plan, conflicts = create_rename_plan(audio_files)
    
    # 顯示預覽
    preview_rename_plan(rename_plan, conflicts)
    
    # 如果有衝突，警告用戶
    if conflicts:
        print(f"\n⚠️  警告：發現 {len(conflicts)} 個檔名衝突")
        print("請檢查並解決衝突後再執行")
        if not DRY_RUN:
            print("建議先使用預覽模式 (DRY_RUN=True) 檢查")
            return
    
    # 執行重新命名
    results = execute_rename_plan(rename_plan, conflicts)
    
    # 顯示結果
    print("\n" + "=" * 80)
    print("📊 執行結果")
    print("=" * 80)
    print(f"總計畫檔案: {results['total_planned']}")
    print(f"衝突檔案: {results['conflicts']}")
    print(f"實際執行: {results['executed']}")
    print(f"成功: {results['success']}")
    print(f"失敗: {results['failed']}")
    print(f"模式: {results['mode']}")
    
    if results['mode'] == 'preview':
        print(f"\n💡 提示：這是預覽模式")
        print(f"如要實際執行，請將腳本中的 DRY_RUN 設為 False")
    elif results['success'] > 0:
        print(f"\n✅ 重新命名完成！")
        if SAVE_RENAME_LOG:
            print(f"💾 重新命名記錄已保存，可用於復原操作")

if __name__ == "__main__":
    main()