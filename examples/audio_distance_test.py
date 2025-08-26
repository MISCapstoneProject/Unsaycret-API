#!/usr/bin/env python3
"""
音檔距離比較測試工具
================================================================================

功能：
- 手動設定兩個音檔位置
- 提取聲紋特徵
- 計算餘弦距離
- 顯示詳細比較結果

使用方式：
1. 修改下方的 AUDIO_FILE_1 和 AUDIO_FILE_2 變數
2. 執行腳本：python audio_distance_test.py

作者：CYouuu
版本：v1.0.0
================================================================================
"""

import os
import sys
import numpy as np
from scipy.spatial.distance import cosine
from pathlib import Path

# 添加專案根目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.identification.VID_identify_v5 import AudioProcessor

# ========== 手動設定音檔位置 ==========
# 請修改以下兩個變數為你要比較的音檔路徑
AUDIO_FILE_1 = "data_separated/speaker1/speaker1_01.wav"
AUDIO_FILE_2 = "speaker1_01.wav"

# =====================================

def extract_embedding(audio_processor: AudioProcessor, audio_path: str) -> np.ndarray:
    """
    從音檔提取聲紋嵌入向量
    
    Args:
        audio_processor: 音頻處理器
        audio_path: 音檔路徑
    
    Returns:
        np.ndarray: 聲紋嵌入向量
    """
    try:
        print(f"   處理音檔: {audio_path}")
        
        # 檢查檔案是否存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音檔不存在: {audio_path}")
        
        # 提取嵌入向量
        embedding = audio_processor.extract_embedding(audio_path)
        
        print(f"   ✅ 成功提取 {len(embedding)} 維向量")
        return embedding
        
    except Exception as e:
        print(f"   ❌ 提取失敗: {e}")
        raise

def calculate_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    計算兩個嵌入向量之間的餘弦距離
    
    Args:
        embedding1: 第一個嵌入向量
        embedding2: 第二個嵌入向量
    
    Returns:
        float: 餘弦距離 (0=完全相同, 1=完全不同)
    """
    return cosine(embedding1, embedding2)

def interpret_distance(distance: float) -> str:
    """
    解釋距離值的含義
    
    Args:
        distance: 餘弦距離值
    
    Returns:
        str: 距離解釋
    """
    if distance < 0.1:
        return "非常相似 (可能是同一人或相同音檔)"
    elif distance < 0.3:
        return "相當相似 (很可能是同一人)"
    elif distance < 0.5:
        return "中等相似 (可能是同一人)"
    elif distance < 0.7:
        return "較不相似 (可能不是同一人)"
    else:
        return "非常不相似 (很可能是不同人)"

def main():
    """主要測試函數"""
    print("🎯 音檔距離比較測試")
    print("=" * 80)
    
    # 載入環境配置
    print("✅ 環境配置會在導入模組時自動載入")
    
    # 顯示要比較的音檔
    print(f"\n📁 比較音檔:")
    print(f"   音檔 1: {AUDIO_FILE_1}")
    print(f"   音檔 2: {AUDIO_FILE_2}")
    
    # 檢查音檔是否存在
    if not os.path.exists(AUDIO_FILE_1):
        print(f"❌ 音檔 1 不存在: {AUDIO_FILE_1}")
        return
    
    if not os.path.exists(AUDIO_FILE_2):
        print(f"❌ 音檔 2 不存在: {AUDIO_FILE_2}")
        return
    
    print("✅ 音檔檢查通過")
    
    try:
        # 初始化音頻處理器
        print(f"\n🔧 初始化音頻處理器...")
        audio_processor = AudioProcessor()
        print("✅ 音頻處理器初始化成功")
        
        # 提取第一個音檔的嵌入向量
        print(f"\n🎵 提取音檔 1 特徵...")
        embedding1 = extract_embedding(audio_processor, AUDIO_FILE_1)
        
        # 提取第二個音檔的嵌入向量
        print(f"\n🎵 提取音檔 2 特徵...")
        embedding2 = extract_embedding(audio_processor, AUDIO_FILE_2)
        
        # 計算距離
        print(f"\n📏 計算距離...")
        distance = calculate_distance(embedding1, embedding2)
        similarity_percentage = (1 - distance) * 100
        
        # 顯示結果
        print("\n" + "=" * 80)
        print("📊 比較結果")
        print("=" * 80)
        print(f"餘弦距離: {distance:.6f}")
        print(f"相似度:   {similarity_percentage:.2f}%")
        print(f"解釋:     {interpret_distance(distance)}")
        
        # 顯示向量資訊
        print(f"\n📋 向量資訊:")
        print(f"   音檔 1 向量維度: {embedding1.shape}")
        print(f"   音檔 2 向量維度: {embedding2.shape}")
        print(f"   音檔 1 向量範圍: [{embedding1.min():.4f}, {embedding1.max():.4f}]")
        print(f"   音檔 2 向量範圍: [{embedding2.min():.4f}, {embedding2.max():.4f}]")
        
        # 額外分析
        print(f"\n🔍 詳細分析:")
        if AUDIO_FILE_1 == AUDIO_FILE_2:
            print("   注意: 兩個音檔路徑相同，應該距離接近 0")
        
        if distance < 0.01:
            print("   ⚡ 距離極小，可能是相同音檔或相同錄音")
        elif distance > 0.8:
            print("   ⚠️  距離很大，很可能是不同的語者")
        
        print("=" * 80)
        print("🎉 測試完成！")
        
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

def show_usage():
    """顯示使用說明"""
    print("使用說明:")
    print("1. 編輯此檔案，修改 AUDIO_FILE_1 和 AUDIO_FILE_2 變數")
    print("2. 設定為你要比較的兩個音檔路徑")
    print("3. 執行: python audio_distance_test.py")
    print("\n範例:")
    print('AUDIO_FILE_1 = "/path/to/speaker1.wav"')
    print('AUDIO_FILE_2 = "/path/to/speaker2.wav"')

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_usage()
    else:
        main()
