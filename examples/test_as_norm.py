#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
AS-Norm (Adaptive Score Normalization) 測試腳本
===============================================================================

版本：v1.0.0
作者：CYouuu
最後更新：2025-08-25

功能摘要：
-----------
本腳本用於測試語者識別模組中的 AS-Norm 功能，包括：

1. T-Norm (Test Normalization): 使用 impostor 模型分數進行正規化
2. Z-Norm (Zero Normalization): 使用統計 Z-score 正規化  
3. S-Norm (Symmetric Normalization): 結合 T-Norm 和 Z-Norm

測試項目：
-----------
- 不同 AS-Norm 設定的比較
- 語者識別準確度的影響
- 正規化前後的分數差異
- 不同參數對識別效果的影響

使用方式：
-----------
python examples/test_as_norm.py

===============================================================================
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional

# 將專案根目錄加入 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.identification.VID_identify_v5 import SpeakerIdentifier, set_output_enabled
from utils.logger import get_logger

# 創建模組專屬日誌器
logger = get_logger(__name__)

# 測試配置
TEST_AUDIO_PATHS = [
    "test_audio/vad_test.wav",  # 替換為實際存在的測試音檔
    # "path/to/test2.wav",
    # "path/to/test3.wav",
]

# AS-Norm 測試配置組合
AS_NORM_CONFIGS = [
    {
        "name": "無正規化",
        "enabled": False,
        "t_norm": False,
        "z_norm": False, 
        "s_norm": False
    },
    {
        "name": "僅 T-Norm",
        "enabled": True,
        "t_norm": True,
        "z_norm": False,
        "s_norm": False,
        "cohort_size": 50,
        "top_k": 10
    },
    {
        "name": "僅 Z-Norm", 
        "enabled": True,
        "t_norm": False,
        "z_norm": True,
        "s_norm": False,
        "cohort_size": 50
    },
    {
        "name": "T-Norm + Z-Norm",
        "enabled": True,
        "t_norm": True,
        "z_norm": True,
        "s_norm": False,
        "cohort_size": 50,
        "top_k": 10
    },
    {
        "name": "完整 S-Norm",
        "enabled": True,
        "t_norm": True,
        "z_norm": True, 
        "s_norm": True,
        "cohort_size": 50,
        "top_k": 10,
        "alpha": 0.8
    },
    {
        "name": "S-Norm (高 Alpha)",
        "enabled": True,
        "t_norm": True,
        "z_norm": True,
        "s_norm": True, 
        "cohort_size": 50,
        "top_k": 10,
        "alpha": 0.9
    },
    {
        "name": "S-Norm (低 Alpha)",
        "enabled": True,
        "t_norm": True,
        "z_norm": True,
        "s_norm": True,
        "cohort_size": 50,
        "top_k": 10,
        "alpha": 0.3
    }
]


def print_separator(title: str) -> None:
    """列印分隔線和標題"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_config_info(config: Dict[str, Any]) -> None:
    """列印配置資訊"""
    print(f"\n🔧 配置: {config['name']}")
    print(f"   啟用狀態: {config['enabled']}")
    if config['enabled']:
        print(f"   T-Norm: {config.get('t_norm', False)}")
        print(f"   Z-Norm: {config.get('z_norm', False)}")
        print(f"   S-Norm: {config.get('s_norm', False)}")
        if 'cohort_size' in config:
            print(f"   Cohort Size: {config['cohort_size']}")
        if 'top_k' in config:
            print(f"   Top-K: {config['top_k']}")
        if 'alpha' in config:
            print(f"   Alpha: {config['alpha']}")


def test_single_audio_with_config(identifier: SpeakerIdentifier, 
                                 audio_path: str, 
                                 config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    使用指定配置測試單個音檔
    
    Args:
        identifier: 語者識別器實例
        audio_path: 音檔路徑
        config: AS-Norm 配置
        
    Returns:
        Dict[str, Any]: 測試結果，包含語者ID、名稱、距離等資訊
    """
    try:
        # 應用配置
        identifier.set_as_norm_enabled(config['enabled'])
        
        if config['enabled']:
            configure_params = {
                't_norm': config.get('t_norm', True),
                'z_norm': config.get('z_norm', True), 
                's_norm': config.get('s_norm', True),
                'cohort_size': config.get('cohort_size', 100),
                'top_k': config.get('top_k', 10),
                'alpha': config.get('alpha', 0.9)
            }
            identifier.configure_as_norm(**configure_params)
        
        # 處理音檔
        result = identifier.process_audio_file(audio_path)
        
        if result is not None:
            speaker_id, speaker_name, distance = result
            return {
                'success': True,
                'speaker_id': speaker_id,
                'speaker_name': speaker_name,
                'distance': distance,
                'config_name': config['name']
            }
        else:
            return {
                'success': False,
                'error': '識別失敗',
                'config_name': config['name']
            }
            
    except Exception as e:
        logger.error(f"測試配置 {config['name']} 時發生錯誤: {e}")
        return {
            'success': False,
            'error': str(e),
            'config_name': config['name']
        }


def compare_results(results: List[Dict[str, Any]]) -> None:
    """
    比較不同配置的結果
    
    Args:
        results: 不同配置的測試結果列表
    """
    print_separator("結果比較")
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ 沒有成功的測試結果可以比較")
        return
    
    print(f"\n📊 成功測試數量: {len(successful_results)}/{len(results)}")
    print("\n距離比較:")
    print("-" * 60)
    print(f"{'配置名稱':<20} {'語者名稱':<15} {'距離':<10} {'相對差異'}")
    print("-" * 60)
    
    # 以第一個結果作為基準
    baseline_distance = successful_results[0]['distance']
    
    for result in successful_results:
        distance = result['distance']
        relative_diff = ((distance - baseline_distance) / baseline_distance * 100) if baseline_distance != 0 else 0
        
        print(f"{result['config_name']:<20} {result['speaker_name']:<15} {distance:<10.4f} {relative_diff:+6.2f}%")
    
    # 統計分析
    distances = [r['distance'] for r in successful_results]
    print(f"\n📈 統計資訊:")
    print(f"   平均距離: {np.mean(distances):.4f}")
    print(f"   距離範圍: {np.min(distances):.4f} - {np.max(distances):.4f}")
    print(f"   標準差: {np.std(distances):.4f}")
    print(f"   變異係數: {np.std(distances)/np.mean(distances)*100:.2f}%")


def test_as_norm_performance() -> None:
    """執行 AS-Norm 效能測試"""
    
    print_separator("AS-Norm 語者識別測試")
    print("🚀 開始測試 AS-Norm 功能...")
    
    # 初始化語者識別器
    try:
        identifier = SpeakerIdentifier()
        identifier.set_verbose(True)
        print("✅ 語者識別器初始化成功")
    except Exception as e:
        logger.error(f"語者識別器初始化失敗: {e}")
        return
    
    # 檢查測試音檔
    valid_audio_paths = []
    for audio_path in TEST_AUDIO_PATHS:
        full_path = project_root / audio_path
        if full_path.exists():
            valid_audio_paths.append(str(full_path))
            print(f"✅ 找到測試音檔: {audio_path}")
        else:
            print(f"⚠️  測試音檔不存在: {audio_path}")
    
    if not valid_audio_paths:
        print("❌ 沒有找到有效的測試音檔，請檢查 TEST_AUDIO_PATHS 配置")
        return
    
    # 對每個音檔測試所有配置
    for i, audio_path in enumerate(valid_audio_paths):
        print_separator(f"測試音檔 {i+1}: {Path(audio_path).name}")
        
        results = []
        
        # 測試每個 AS-Norm 配置
        for config in AS_NORM_CONFIGS:
            print_config_info(config)
            
            result = test_single_audio_with_config(identifier, audio_path, config)
            if result:
                results.append(result)
                
                if result['success']:
                    print(f"✅ 識別成功: {result['speaker_name']}, 距離: {result['distance']:.4f}")
                else:
                    print(f"❌ 識別失敗: {result['error']}")
            
            print("-" * 40)
        
        # 比較結果
        if len(results) > 1:
            compare_results(results)
    
    print_separator("測試完成")
    print("🎉 AS-Norm 測試完成！")


def demonstrate_as_norm_features() -> None:
    """演示 AS-Norm 功能特性"""
    
    print_separator("AS-Norm 功能演示")
    
    try:
        identifier = SpeakerIdentifier()
        
        # 演示狀態查詢
        print("\n🔍 初始 AS-Norm 狀態:")
        status = identifier.get_as_norm_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # 演示配置修改
        print("\n🔧 啟用 AS-Norm 並配置參數...")
        identifier.set_as_norm_enabled(True)
        identifier.configure_as_norm(
            t_norm=True,
            z_norm=True,
            s_norm=True,
            cohort_size=30,
            top_k=5,
            alpha=0.7
        )
        
        print("\n🔍 更新後的 AS-Norm 狀態:")
        status = identifier.get_as_norm_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # 演示關閉功能
        print("\n❌ 關閉 AS-Norm...")
        identifier.set_as_norm_enabled(False)
        
        print("\n🔍 關閉後的 AS-Norm 狀態:")
        status = identifier.get_as_norm_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        logger.error(f"AS-Norm 功能演示時發生錯誤: {e}")


def main() -> None:
    """主函數"""
    
    # 啟用輸出
    set_output_enabled(True)
    
    print("🎯 AS-Norm 測試腳本")
    print("作者：CYouuu")
    print("版本：v1.0.0")
    
    # 功能演示
    demonstrate_as_norm_features()
    
    # 效能測試
    test_as_norm_performance()
    
    print("\n" + "=" * 80)
    print("  測試說明")
    print("=" * 80)
    print("1. T-Norm: 使用 impostor 語者的分數統計進行正規化")
    print("2. Z-Norm: 使用所有語者的分數統計進行正規化")
    print("3. S-Norm: 結合 T-Norm 和 Z-Norm，透過 alpha 參數控制權重")
    print("4. Cohort Size: 用於統計的語者數量")
    print("5. Top-K: 使用前 K 個最相似的 impostor")
    print("6. Alpha: S-Norm 中 T-Norm 的權重 (1-alpha 為 Z-Norm 權重)")
    print("\n💡 提示：在實際使用中，建議根據您的資料集調整這些參數")


if __name__ == "__main__":
    main()
