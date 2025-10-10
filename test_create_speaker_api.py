#!/usr/bin/env python3
"""
測試手動建立語者 API 的腳本

使用方式:
    python test_create_speaker_api.py [音檔路徑]
    
範例:
    python test_create_speaker_api.py test_audio_me/output_001.wav
"""

import requests
import sys
import os
from pathlib import Path

def test_create_speaker_api(audio_file_path: str, api_base_url: str = "http://localhost:8000"):
    """
    測試手動建立語者 API
    
    Args:
        audio_file_path: 音檔路徑
        api_base_url: API 基礎 URL
    """
    # 檢查音檔是否存在
    if not os.path.exists(audio_file_path):
        print(f"❌ 音檔不存在: {audio_file_path}")
        return False
    
    audio_path = Path(audio_file_path)
    print(f"🎤 測試音檔: {audio_path.name}")
    
    # 準備請求
    url = f"{api_base_url}/speakers/create"
    
    # 準備表單資料
    form_data = {
        'full_name': '測試語者_' + audio_path.stem,  # 必填
        'nickname': '小測',  # 選填
        'gender': '測試用性別'  # 選填，現在不限制選項
    }
    
    # 準備檔案
    files = {
        'file': (audio_path.name, open(audio_file_path, 'rb'), 'audio/wav')
    }
    
    try:
        print(f"📤 發送請求到: {url}")
        print(f"📋 表單資料: {form_data}")
        
        # 發送請求
        response = requests.post(url, data=form_data, files=files, timeout=30)
        
        # 關閉檔案
        files['file'][1].close()
        
        print(f"📥 回應狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 建立語者成功!")
            print("📊 回應資料:")
            print(f"   - 成功: {result.get('success')}")
            print(f"   - 訊息: {result.get('message')}")
            
            if 'data' in result:
                data = result['data']
                print(f"   - 語者 UUID: {data.get('speaker_uuid')}")
                print(f"   - 語者 ID: {data.get('speaker_id')}")
                print(f"   - 全名: {data.get('full_name')}")
                print(f"   - 暱稱: {data.get('nickname')}")
                print(f"   - 性別: {data.get('gender')}")
                print(f"   - 聲紋 UUID: {data.get('voiceprint_uuid')}")
                print(f"   - 聲紋數量: {data.get('voiceprint_count')}")
            
            return True
        else:
            print(f"❌ 建立語者失敗!")
            print(f"📄 錯誤內容: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 無法連接到 API 服務，請確認服務是否運行在 http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        return False

def main():
    """主函數"""
    if len(sys.argv) < 2:
        print("使用方式: python test_create_speaker_api.py [音檔路徑]")
        print("範例: python test_create_speaker_api.py test_audio_me/output_001.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    success = test_create_speaker_api(audio_file)
    
    if success:
        print("\n🎉 測試完成 - 成功!")
    else:
        print("\n💥 測試完成 - 失敗!")
        sys.exit(1)

if __name__ == "__main__":
    main()