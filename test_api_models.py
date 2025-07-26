#!/usr/bin/env python3
"""
測試 API 模型和資料庫整合
"""

import sys
sys.path.append('.')

from services.data_facade import SpeakerHandler
from modules.database.database import DatabaseService

def test_api_models():
    """測試 API 模型和資料回傳格式"""
    print("🧪 開始測試 API 模型...")
    
    try:
        # 1. 測試資料庫連接
        db = DatabaseService()
        if not db.check_database_connection():
            print("❌ 資料庫連接失敗")
            return
        print("✅ 資料庫連接成功")
        
        # 2. 測試 list_all_speakers 
        print("\n📋 測試 list_all_speakers...")
        handler = SpeakerHandler()
        speakers = handler.list_all_speakers()
        
        if not speakers:
            print("⚠️  資料庫中沒有語者資料")
            return
            
        print(f"✅ 找到 {len(speakers)} 位語者")
        
        # 檢查第一個語者的資料結構
        first_speaker = speakers[0]
        print(f"\n🔍 第一位語者的資料結構:")
        for key, value in first_speaker.items():
            print(f"  {key}: {value} ({type(value).__name__})")
        
        # 3. 測試 get_speaker_info (使用UUID)
        if 'uuid' in first_speaker:
            uuid_test = first_speaker['uuid']
            print(f"\n🔍 測試使用 UUID 查詢語者: {uuid_test}")
            try:
                speaker_by_uuid = handler.get_speaker_info(uuid_test)
                print("✅ UUID 查詢成功")
                print(f"  返回的資料鍵: {list(speaker_by_uuid.keys())}")
            except Exception as e:
                print(f"❌ UUID 查詢失敗: {e}")
        
        # 4. 測試 get_speaker_info (使用序號ID)
        if 'speaker_id' in first_speaker and first_speaker['speaker_id'] != -1:
            numeric_id = str(first_speaker['speaker_id'])
            print(f"\n🔍 測試使用序號ID 查詢語者: {numeric_id}")
            try:
                speaker_by_id = handler.get_speaker_info(numeric_id)
                print("✅ 序號ID 查詢成功")
                print(f"  返回的資料鍵: {list(speaker_by_id.keys())}")
            except Exception as e:
                print(f"❌ 序號ID 查詢失敗: {e}")
        
        print("\n✅ 測試完成！")
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_models()
