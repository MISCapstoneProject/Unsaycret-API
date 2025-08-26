#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AS-Norm 語者識別簡單使用示例
==========================

這個腳本展示了如何在語者識別中使用 AS-Norm 功能
"""

import sys
from pathlib import Path

# 將專案根目錄加入 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.identification.VID_identify_v5 import SpeakerIdentifier, set_output_enabled


def main():
    """AS-Norm 使用示例"""
    
    print("🎯 AS-Norm 語者識別示例")
    print("=" * 50)
    
    # 啟用詳細輸出
    set_output_enabled(True)
    
    # 創建語者識別器
    identifier = SpeakerIdentifier()
    identifier.set_verbose(True)
    
    print("\n1️⃣ 查看初始 AS-Norm 狀態")
    status = identifier.get_as_norm_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n2️⃣ 啟用 AS-Norm 功能")
    identifier.set_as_norm_enabled(True)
    
    print("\n3️⃣ 配置 AS-Norm 參數")
    identifier.configure_as_norm(
        t_norm=True,      # 啟用 T-Norm
        z_norm=True,      # 啟用 Z-Norm
        s_norm=True,      # 啟用 S-Norm  
        cohort_size=50,   # 使用 50 個語者作為 cohort
        top_k=10,         # 使用前 10 個 impostor
        alpha=0.8         # S-Norm 權重參數
    )
    
    print("\n4️⃣ 查看更新後的狀態")
    status = identifier.get_as_norm_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n5️⃣ 測試不同的 AS-Norm 配置")
    
    # 測試音檔路徑（請替換為實際存在的音檔）
    test_audio = "test_audio/vad_test.wav"
    
    if (project_root / test_audio).exists():
        print(f"\n🎵 使用測試音檔: {test_audio}")
        
        # 測試 1: 無正規化
        print("\n📊 測試 1: 無正規化")
        identifier.set_as_norm_enabled(False)
        result1 = identifier.process_audio_file(str(project_root / test_audio))
        
        # 測試 2: 僅 T-Norm
        print("\n📊 測試 2: 僅 T-Norm")
        identifier.set_as_norm_enabled(True)
        identifier.configure_as_norm(t_norm=True, z_norm=False, s_norm=False)
        result2 = identifier.process_audio_file(str(project_root / test_audio))
        
        # 測試 3: 完整 S-Norm
        print("\n📊 測試 3: 完整 S-Norm")
        identifier.configure_as_norm(t_norm=True, z_norm=True, s_norm=True, alpha=0.8)
        result3 = identifier.process_audio_file(str(project_root / test_audio))
        
        # 比較結果
        print("\n📈 結果比較:")
        results = [
            ("無正規化", result1),
            ("僅 T-Norm", result2), 
            ("完整 S-Norm", result3)
        ]
        
        for name, result in results:
            if result:
                speaker_id, speaker_name, distance = result
                print(f"   {name:<12}: {speaker_name} (距離: {distance:.4f})")
            else:
                print(f"   {name:<12}: 識別失敗")
                
    else:
        print(f"\n⚠️ 測試音檔不存在: {test_audio}")
        print("請將有效的音檔放在該路徑，或修改 test_audio 變數")
    
    print("\n6️⃣ 進階配置示例")
    print("\n🔧 高精度配置 (適用於重要場景):")
    identifier.configure_as_norm(
        t_norm=True,
        z_norm=True, 
        s_norm=True,
        cohort_size=100,
        top_k=20,
        alpha=0.85
    )
    print("   ✅ 已配置高精度模式")
    
    print("\n⚡ 快速配置 (適用於即時場景):")
    identifier.configure_as_norm(
        t_norm=True,
        z_norm=False,
        s_norm=False,
        cohort_size=30,
        top_k=5
    )
    print("   ✅ 已配置快速模式")
    
    print("\n🎯 平衡配置 (適用於一般場景):")
    identifier.configure_as_norm(
        t_norm=True,
        z_norm=True,
        s_norm=False,  # 不使用 S-Norm，但結合 T 和 Z
        cohort_size=50,
        top_k=10
    )
    print("   ✅ 已配置平衡模式")
    
    print("\n" + "=" * 50)
    print("🎉 AS-Norm 示例完成！")
    print("\n💡 提示:")
    print("- 在 constants.py 中修改 ENABLE_AS_NORM 來全域控制")
    print("- 根據您的資料集大小調整 cohort_size 參數")
    print("- 在嘈雜環境中 AS-Norm 效果更顯著")
    print("- 使用 examples/test_as_norm.py 進行深入測試")


if __name__ == "__main__":
    main()
