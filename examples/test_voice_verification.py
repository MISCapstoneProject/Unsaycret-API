# examples/test_voice_verification.py
"""
語音驗證API測試範例

此腳本展示如何使用 /speaker/verify 端點進行純讀取的語音身份驗證
"""
import requests
import json
from pathlib import Path

# API基礎URL
BASE_URL = "http://localhost:8000"

def test_voice_verification(audio_file_path: str, threshold: float = 0.4, max_results: int = 3):
    """
    測試語音驗證API
    
    Args:
        audio_file_path: 要驗證的音檔路徑
        threshold: 比對閾值，距離小於此值才認為是匹配
        max_results: 最大結果數量
    """
    url = f"{BASE_URL}/speaker/verify"
    
    # 檢查檔案是否存在
    if not Path(audio_file_path).exists():
        print(f"錯誤：音檔 {audio_file_path} 不存在")
        return None
    
    try:
        # 準備檔案和參數
        with open(audio_file_path, "rb") as audio_file:
            files = {"file": audio_file}
            data = {
                "threshold": threshold,
                "max_results": max_results
            }
            
            # 發送請求
            response = requests.post(url, files=files, data=data)
        
        print(f"POST /speaker/verify")
        print(f"音檔: {audio_file_path}")
        print(f"比對閾值: {threshold}")
        print(f"最大結果數: {max_results}")
        print(f"狀態碼: {response.status_code}")
        print(f"回應:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        print("-" * 60)
        
        return response.json() if response.status_code == 200 else None
        
    except Exception as e:
        print(f"錯誤: {e}")
        return None

def main():
    """主測試函數"""
    print("🎵 語音驗證API測試")
    print("=" * 60)
    
    # 測試音檔路徑（請根據實際情況修改）
    test_audio_files = [
        "16K-model/Audios-16K-IDTF/speaker1_20250629-22_00_14_1.wav",
        "16K-model/Audios-16K-IDTF/speaker2_20250629-22_00_14_1.wav",
        "16K-model/Audios-16K-IDTF/mixed_audio_20250629-22_00_22.wav"
    ]
    
    for audio_file in test_audio_files:
        print(f"\n📁 測試音檔: {audio_file}")
        
        # 基本測試
        result = test_voice_verification(audio_file)
        
        if result:
            print(f"✅ 驗證結果:")
            print(f"   - 是否為已知語者: {result.get('is_known_speaker')}")
            print(f"   - 比對閾值: {result.get('threshold'):.3f}")
            print(f"   - 候選數量: {result.get('total_candidates')}")
            
            if result.get('best_match'):
                best = result['best_match']
                print(f"   - 最佳匹配: {best['speaker_name']} (距離: {best['distance']:.4f})")
        else:
            print("❌ 測試失敗")
    
    print("\n🔍 高閾值測試 (0.8 - 更嚴格)")
    if test_audio_files:
        test_voice_verification(test_audio_files[0], threshold=0.8, max_results=5)
    
    print("\n📊 低閾值測試 (0.2 - 更寬鬆)")
    if test_audio_files:
        test_voice_verification(test_audio_files[0], threshold=0.2, max_results=1)

if __name__ == "__main__":
    main()
