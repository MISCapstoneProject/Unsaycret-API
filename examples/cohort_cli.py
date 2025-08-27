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


def quick_import_folder(folder_path: str, dataset_prefix: str = None):
    """快速導入音頻資料夾（每個檔案使用檔名作為 source_dataset）"""
    print(f"🚀 AS-Norm Cohort 資料夾批量導入")
    print("=" * 50)
    print(f"📁 來源資料夾: {folder_path}")
    if dataset_prefix:
        print(f"🏷️  資料集前綴: {dataset_prefix}")
    else:
        print(f"🏷️  使用檔名作為 source_dataset")
    
    manager = CohortDatabaseManager()
    
    try:
        # 檢查資料夾
        if not os.path.exists(folder_path):
            print(f"❌ 資料夾不存在: {folder_path}")
            return
        
        # 導入（不再需要切片參數）
        print(f"\n🎵 開始導入音頻檔案（處理完整 6 秒音檔）...")
        results = manager.import_audio_folder(
            folder_path=folder_path,
            source_dataset_prefix=dataset_prefix,
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


def quick_import_file(file_path: str, source_dataset: str = None):
    """快速導入單個音頻檔案"""
    print(f"🚀 AS-Norm Cohort 單檔導入")
    print("=" * 50)
    print(f"📄 音頻檔案: {file_path}")
    if source_dataset:
        print(f"🏷️  source_dataset: {source_dataset}")
    else:
        print(f"🏷️  使用檔名作為 source_dataset")
    
    manager = CohortDatabaseManager()
    
    try:
        # 檢查檔案
        if not os.path.exists(file_path):
            print(f"❌ 檔案不存在: {file_path}")
            return
        
        # 導入單檔
        print(f"\n🎵 開始導入音頻檔案...")
        success_count = manager.import_audio_file(
            audio_path=file_path,
            source_dataset=source_dataset,
            metadata={"language": "zh-TW"}
        )
        
        if success_count > 0:
            print(f"✅ 成功導入 1 個聲紋到 cohort 資料庫！")
        else:
            print(f"❌ 導入失敗，請檢查音頻檔案格式。")
            
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
            for dataset, count in sorted(datasets.items()):
                print(f"   📂 {dataset}: {count} 筆")
        else:
            print(f"\n📁 無來源資料集資訊")
        
        # 性別分佈
        genders = stats.get('genders', {})
        if genders:
            print(f"\n👤 性別分佈:")
            for gender, count in genders.items():
                print(f"   🚻 {gender}: {count} 筆")
        else:
            print(f"\n👤 無性別分佈資訊")
        
        # 語言分佈
        languages = stats.get('languages', {})
        if languages:
            print(f"\n🌍 語言分佈:")
            for language, count in languages.items():
                print(f"   🗣️  {language}: {count} 筆")
        else:
            print(f"\n🌍 無語言分佈資訊")
        
        # 時間範圍
        time_range = stats.get('time_range', {})
        if time_range:
            print(f"\n📅 資料時間範圍:")
            print(f"   ⏰ 最早: {time_range.get('earliest', 'N/A')}")
            print(f"   🕐 最新: {time_range.get('latest', 'N/A')}")
        else:
            print(f"\n📅 無時間範圍資訊")
            
        # 顯示快速摘要
        total = stats.get('total_count', 0)
        if total > 0:
            print(f"\n📋 快速摘要:")
            print(f"   🎯 可用於 AS-Norm 的聲紋數量: {total}")
            print(f"   📊 資料集數量: {len(datasets)} 個")
            print(f"   🆔 唯一 cohort_id 數量: ~{total}")
            
    finally:
        manager.close()


def detailed_stats():
    """詳細統計信息"""
    print("🚀 AS-Norm Cohort 詳細統計資訊")
    print("=" * 60)
    
    manager = CohortDatabaseManager()
    
    try:
        stats = manager.get_cohort_statistics()
        
        if not stats.get('exists', False):
            print("❌ Cohort collection 不存在！")
            print("💡 請先執行初始化: python cohort_cli.py init")
            return
        
        total = stats.get('total_count', 0)
        print(f"🗄️  Collection 名稱: {stats.get('collection_name', 'N/A')}")
        print(f"📊 總聲紋數量: {total}")
        print(f"💾 向量維度: 192 (ECAPA-TDNN)")
        print(f"📏 距離度量: COSINE")
        
        # 資料分佈分析
        datasets = stats.get('source_datasets', {})
        if datasets:
            print(f"\n📂 資料集詳細分析:")
            print(f"   📈 資料集總數: {len(datasets)}")
            print(f"   📊 分佈詳情:")
            for i, (dataset, count) in enumerate(sorted(datasets.items(), key=lambda x: x[1], reverse=True), 1):
                percentage = (count / total * 100) if total > 0 else 0
                print(f"      {i:2d}. {dataset}: {count} 筆 ({percentage:.1f}%)")
        
        # 統計摘要
        if total > 0:
            avg_per_dataset = total / len(datasets) if datasets else 0
            print(f"\n📈 統計摘要:")
            print(f"   📊 平均每資料集聲紋數: {avg_per_dataset:.1f}")
            print(f"   🎯 AS-Norm Cohort 覆蓋度: {'良好' if total >= 100 else '需增加' if total >= 50 else '不足'}")
            print(f"   💡 建議 Cohort 大小: 100-500 筆")
            
        # 匯出選項
        print(f"\n💾 匯出選項:")
        export_file = manager.export_cohort_info()
        if export_file:
            print(f"   📄 詳細資訊已匯出至: {export_file}")
        
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
    print("  init                         - 初始化 cohort 資料庫")
    print("  import-folder <folder_path>  - 導入音頻資料夾")
    print("  import-file <file_path>      - 導入單個音頻檔案")
    print("  stats                        - 查看基本統計信息")
    print("  detailed-stats               - 查看詳細統計信息與分析")
    print("  reset                        - 重置資料庫（刪除所有資料）")
    print("  help                         - 顯示此說明")
    print("\n📖 使用範例:")
    print("  python cohort_cli.py init")
    print("  python cohort_cli.py import-folder /path/to/audio/folder")
    print("  python cohort_cli.py import-folder /path/to/audio/folder prefix_name")
    print("  python cohort_cli.py import-file /path/to/audio.wav")
    print("  python cohort_cli.py import-file /path/to/audio.wav custom_dataset_name")
    print("  python cohort_cli.py stats")
    print("  python cohort_cli.py detailed-stats")
    print("\n💡 新版本重要變更:")
    print("  - ✅ 直接處理完整 6 秒音檔，不再進行切片")
    print("  - ✅ 自動使用檔名作為 source_dataset（提升可追蹤性）")
    print("  - ✅ 與 VID_identify_v5.py 完全一致的聲紋提取流程")
    print("  - ✅ 支援單檔和資料夾批量導入")
    print("\n📁 支援格式:")
    print("  .wav, .mp3, .flac, .m4a, .aac, .ogg")
    print("\n🎯 使用建議:")
    print("  - 使用乾淨的背景語音資料，避免包含實際會出現的語者")
    print("  - 建議 cohort 大小: 100-500 筆聲紋")
    print("  - 每個音檔應為 6 秒長度，格式為 16kHz 單聲道")


def main():
    """主程式"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "init":
        quick_init()
        
    elif command == "import-folder":
        if len(sys.argv) < 3:
            print("❌ 請指定音頻資料夾路徑")
            print("💡 使用方式: python cohort_cli.py import-folder /path/to/audio/folder [prefix]")
            return
        
        folder_path = sys.argv[2]
        dataset_prefix = sys.argv[3] if len(sys.argv) > 3 else None
        quick_import_folder(folder_path, dataset_prefix)
        
    elif command == "import-file":
        if len(sys.argv) < 3:
            print("❌ 請指定音頻檔案路徑")
            print("💡 使用方式: python cohort_cli.py import-file /path/to/audio.wav [source_dataset]")
            return
        
        file_path = sys.argv[2]
        source_dataset = sys.argv[3] if len(sys.argv) > 3 else None
        quick_import_file(file_path, source_dataset)
        
    elif command == "stats":
        quick_stats()
        
    elif command == "detailed-stats":
        detailed_stats()
        
    elif command == "reset":
        quick_reset()
        
    elif command == "help":
        show_help()
        
    # 向後兼容的舊命令
    elif command == "import":
        print("⚠️  注意: 'import' 命令已棄用，請使用 'import-folder'")
        if len(sys.argv) < 3:
            print("❌ 請指定音頻資料夾路徑")
            print("💡 使用方式: python cohort_cli.py import-folder /path/to/audio/folder")
            return
        
        folder_path = sys.argv[2]
        dataset_prefix = sys.argv[3] if len(sys.argv) > 3 else None
        quick_import_folder(folder_path, dataset_prefix)
        
    else:
        print(f"❌ 未知命令: {command}")
        show_help()


if __name__ == "__main__":
    main()
