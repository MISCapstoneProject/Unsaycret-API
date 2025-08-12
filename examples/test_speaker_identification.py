#!/usr/bin/env python3
"""
語者識別模組測試檔
==============

簡單測試腳本，用於測試 VID_identify_v5.py 模組的語者識別功能
可以手動修改音檔路徑進行測試

使用方法：
1. 修改下方的 AUDIO_FILE_PATH 變數
2. 執行: python test_speaker_identification.py
"""

import os
import json
import traceback
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.identification.VID_identify_v5 import SpeakerIdentifier

# ==================== 設定區域 ====================
# 👇 請修改這裡的音檔路徑
AUDIO_FILE_PATH = "stream_output/20250809_233353/segment_005/speaker1.wav"  # 修改為您要測試的音檔路徑

# 其他設定
ENABLE_VERBOSE = True  # 是否顯示詳細輸出
SAVE_RESULT = True     # 是否將結果保存到 JSON 檔案
# ==================================================

def test_speaker_identification(audio_path: str):
    """測試語者識別功能"""
    
    print("=" * 60)
    print(f"🎯 語者識別測試")
    print(f"音檔路徑: {audio_path}")
    print("=" * 60)
    
    # 檢查音檔是否存在
    if not os.path.exists(audio_path):
        print(f"❌ 錯誤：音檔不存在 - {audio_path}")
        print("請確認：")
        print("1. 音檔路徑是否正確")
        print("2. 音檔是否存在於指定位置")
        return None
    
    # 檢查音檔格式
    audio_ext = Path(audio_path).suffix.lower()
    supported_formats = ['.wav']
    if audio_ext not in supported_formats:
        print(f"⚠️  警告：音檔格式 {audio_ext} 可能不被支援")
        print(f"建議使用格式：{', '.join(supported_formats)}")
    
    try:
        print("🔄 初始化語者識別器...")
        
        # 創建語者識別器實例
        identifier = SpeakerIdentifier()
        
        # 設定是否顯示詳細輸出
        identifier.set_verbose(ENABLE_VERBOSE)
        
        print("✅ 語者識別器初始化完成")
        print("\n🔍 開始處理音檔...")
        
        # 處理音檔
        result = identifier.process_audio_file(audio_path)
        
        if result is None:
            print("❌ 語者識別失敗：返回結果為 None")
            return None
        
        # 解析結果
        speaker_id, speaker_name, similarity = result
        
        print("\n" + "=" * 60)
        print("🎉 語者識別完成！")
        print("=" * 60)
        print(f"📋 結果摘要:")
        print(f"   語者 ID: {speaker_id}")
        print(f"   語者名稱: {speaker_name}")
        print(f"   相似度距離: {similarity:.4f}")
        print(f"   數據類型檢查:")
        print(f"     - speaker_id type: {type(speaker_id)}")
        print(f"     - speaker_name type: {type(speaker_name)}")
        print(f"     - similarity type: {type(similarity)}")
        
        # 判斷識別結果類型
        if similarity == -1:
            print(f"   🆕 狀態: 新語者")
        elif similarity < identifier.threshold_low:
            print(f"   ✅ 狀態: 完全匹配（無需更新）")
        elif similarity < identifier.threshold_update:
            print(f"   🔄 狀態: 已更新現有語者聲紋")
        elif similarity < identifier.threshold_new:
            print(f"   📁 狀態: 已為現有語者新增聲紋")
        else:
            print(f"   🆕 狀態: 識別為新語者")
        
        # 測試 JSON 序列化
        print(f"\n🧪 JSON 序列化測試...")
        test_data = {
            "audio_file": audio_path,
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "similarity": similarity,
            "timestamp": "2025-01-01T00:00:00+08:00"
        }
        
        try:
            json_str = json.dumps(test_data, ensure_ascii=False, indent=2)
            print("✅ JSON 序列化成功")
            
            if SAVE_RESULT:
                result_file = f"test_result_{Path(audio_path).stem}.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                print(f"💾 結果已保存至: {result_file}")
                
        except Exception as e:
            print(f"❌ JSON 序列化失敗: {e}")
            print("這可能表示返回的數據包含不可序列化的對象")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 發生錯誤:")
        print(f"錯誤類型: {type(e).__name__}")
        print(f"錯誤訊息: {str(e)}")
        print(f"\n🔍 詳細錯誤追蹤:")
        traceback.print_exc()
        return None
    
    finally:
        print(f"\n" + "=" * 60)
        print("🏁 測試結束")
        print("=" * 60)

def main():
    """主函數"""
    print("🚀 語者識別測試工具")
    print(f"測試音檔: {AUDIO_FILE_PATH}")
    print()
    
    # 提示用戶可以修改路徑
    if not os.path.exists(AUDIO_FILE_PATH):
        print("💡 提示：找不到預設測試音檔")
        print("請編輯此檔案的 AUDIO_FILE_PATH 變數，指向您想測試的音檔")
        print()
    
    result = test_speaker_identification(AUDIO_FILE_PATH)
    
    if result:
        print("\n✨ 測試完成！語者識別功能正常運作")
    else:
        print("\n⚠️  測試未成功完成，請檢查上述錯誤訊息")

if __name__ == "__main__":
    main()
