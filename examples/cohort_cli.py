#!/usr/bin/env python3
"""
AS-Norm Cohort 資料庫快速管理工具

快速執行常用的 cohort 資料庫管理任務
"""

import os
import sys

# 添加模組路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from modules.database.cohort_manager import CohortDatabaseManager


def quick_init():
    """快速初始化 cohort 資料庫"""
    print("🚀 AS-Norm Cohort 資料庫快速初始化")
    print("=" * 50)
    
    manager = CohortDatabaseManager()
    
    try:
        # 初始化
        print("🔧 正在初始化 cohort collection...")
        success = manager.initialize_cohort_collection()
        
        if success:
            print("✅ Cohort collection 初始化成功！")
            
            # 顯示統計信息
            print("\n📊 當前統計信息:")
            stats = manager.get_cohort_statistics()
            print(f"   📁 Collection: {stats.get('collection_name', 'N/A')}")
            print(f"   📊 資料數量: {stats.get('total_count', 0)}")
            print(f"   ✅ 狀態: {'已就緒' if stats.get('exists', False) else '未就緒'}")
            
        else:
            print("❌ Cohort collection 初始化失敗！")
            
    finally:
        manager.close()


def quick_import(folder_path: str, dataset_name: str = "quick_import"):
    """快速導入音頻資料夾"""
    print(f"🚀 AS-Norm Cohort 資料快速導入")
    print("=" * 50)
    print(f"📁 來源資料夾: {folder_path}")
    print(f"🏷️  資料集名稱: {dataset_name}")
    
    manager = CohortDatabaseManager()
    
    try:
        # 檢查資料夾
        if not os.path.exists(folder_path):
            print(f"❌ 資料夾不存在: {folder_path}")
            return
        
        # 導入
        print(f"\n🎵 開始導入音頻檔案...")
        results = manager.import_audio_folder(
            folder_path=folder_path,
            source_dataset=dataset_name,
            chunk_length=3.0,  # 3秒切片
            overlap=0.5,       # 50%重疊
            metadata={"language": "zh-TW"}  # 預設中文
        )
        
        # 顯示結果
        print(f"\n📈 導入完成:")
        print(f"   📁 總檔案數: {results['total_files']}")
        print(f"   ✅ 成功檔案: {results['success_files']}")
        print(f"   ❌ 失敗檔案: {results['failed_files']}")
        print(f"   🎯 總聲紋數: {results['total_embeddings']}")
        
        if results['total_embeddings'] > 0:
            print(f"\n✅ 成功導入 {results['total_embeddings']} 個聲紋到 cohort 資料庫！")
        else:
            print(f"\n❌ 沒有成功導入任何聲紋，請檢查音頻檔案格式。")
            
    finally:
        manager.close()


def quick_stats():
    """快速查看統計信息"""
    print("🚀 AS-Norm Cohort 資料庫統計信息")
    print("=" * 50)
    
    manager = CohortDatabaseManager()
    
    try:
        stats = manager.get_cohort_statistics()
        
        if not stats.get('exists', False):
            print("❌ Cohort collection 不存在！")
            print("💡 請先執行初始化: python cohort_cli.py init")
            return
        
        print(f"📊 Collection: {stats.get('collection_name', 'N/A')}")
        print(f"📈 總資料數量: {stats.get('total_count', 0)}")
        
        # 來源資料集分佈
        datasets = stats.get('source_datasets', {})
        if datasets:
            print(f"\n📁 來源資料集分佈:")
            for dataset, count in datasets.items():
                print(f"   {dataset}: {count}")
        
        # 性別分佈
        genders = stats.get('genders', {})
        if genders:
            print(f"\n👤 性別分佈:")
            for gender, count in genders.items():
                print(f"   {gender}: {count}")
        
        # 語言分佈
        languages = stats.get('languages', {})
        if languages:
            print(f"\n🌍 語言分佈:")
            for language, count in languages.items():
                print(f"   {language}: {count}")
        
        # 時間範圍
        time_range = stats.get('time_range', {})
        if time_range:
            print(f"\n📅 資料時間範圍:")
            print(f"   最早: {time_range.get('earliest', 'N/A')}")
            print(f"   最新: {time_range.get('latest', 'N/A')}")
            
    finally:
        manager.close()


def quick_reset():
    """快速重置 cohort 資料庫"""
    print("🚀 AS-Norm Cohort 資料庫重置")
    print("=" * 50)
    
    # 確認操作
    confirm = input("⚠️  警告：這將刪除所有 cohort 資料！確定要繼續嗎？ (y/N): ")
    if confirm.lower() != 'y':
        print("❌ 操作已取消")
        return
    
    manager = CohortDatabaseManager()
    
    try:
        print("🗑️  正在重置 cohort collection...")
        success = manager.reset_cohort_collection()
        
        if success:
            print("✅ Cohort collection 重置成功！")
        else:
            print("❌ Cohort collection 重置失敗！")
            
    finally:
        manager.close()


def show_help():
    """顯示使用說明"""
    print("🚀 AS-Norm Cohort 資料庫管理工具")
    print("=" * 50)
    print("\n📋 可用命令:")
    print("  init                    - 初始化 cohort 資料庫")
    print("  import <folder_path>    - 導入音頻資料夾")
    print("  stats                   - 查看統計信息")
    print("  reset                   - 重置資料庫（刪除所有資料）")
    print("  help                    - 顯示此說明")
    print("\n📖 使用範例:")
    print("  python cohort_cli.py init")
    print("  python cohort_cli.py import /path/to/audio/folder")
    print("  python cohort_cli.py stats")
    print("\n💡 提示:")
    print("  - 音頻檔案會自動切片為 3 秒片段，重疊度 50%")
    print("  - 支援格式: .wav, .mp3, .flac, .m4a, .aac, .ogg")
    print("  - 建議使用乾淨的背景語音資料，避免包含實際會出現的語者")


def main():
    """主程式"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "init":
        quick_init()
        
    elif command == "import":
        if len(sys.argv) < 3:
            print("❌ 請指定音頻資料夾路徑")
            print("💡 使用方式: python cohort_cli.py import /path/to/audio/folder")
            return
        
        folder_path = sys.argv[2]
        dataset_name = sys.argv[3] if len(sys.argv) > 3 else "quick_import"
        quick_import(folder_path, dataset_name)
        
    elif command == "stats":
        quick_stats()
        
    elif command == "reset":
        quick_reset()
        
    elif command == "help":
        show_help()
        
    else:
        print(f"❌ 未知命令: {command}")
        show_help()


if __name__ == "__main__":
    main()
