"""
測試語者管理API的示例腳本
"""
import requests
import json

# API基礎URL（請根據您的實際部署調整）
BASE_URL = "http://localhost:8000"

def test_get_speaker_info(speaker_id: str):
    """測試獲取說話者資訊"""
    url = f"{BASE_URL}/speaker/{speaker_id}"
    try:
        response = requests.get(url)
        print(f"GET /speaker/{speaker_id}")
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print("-" * 50)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"錯誤: {e}")
        return None

def test_rename_speaker(speaker_id: str, current_name: str, new_name: str):
    """測試說話者改名功能"""
    url = f"{BASE_URL}/speaker/rename"
    data = {
        "speaker_id": speaker_id,
        "current_name": current_name,
        "new_name": new_name
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"POST /speaker/rename")
        print(f"請求資料: {json.dumps(data, indent=2, ensure_ascii=False)}")
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print("-" * 50)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"錯誤: {e}")
        return None

def test_transfer_voiceprints(source_speaker_id: str, source_name: str, 
                             target_speaker_id: str, target_name: str):
    """測試聲紋轉移功能"""
    url = f"{BASE_URL}/speaker/transfer"
    data = {
        "source_speaker_id": source_speaker_id,
        "source_speaker_name": source_name,
        "target_speaker_id": target_speaker_id,
        "target_speaker_name": target_name
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"POST /speaker/transfer")
        print(f"請求資料: {json.dumps(data, indent=2, ensure_ascii=False)}")
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print("-" * 50)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"錯誤: {e}")
        return None

if __name__ == "__main__":
    print("語者管理API測試腳本")
    print("=" * 50)
    
    # 示例測試（請替換為實際的speaker ID）
    # 1. 測試獲取說話者資訊
    print("1. 測試獲取說話者資訊")
    speaker_info = test_get_speaker_info("81d60ed8-3c8b-43b8-808d-2dd4409ca814")
    
    # 2. 測試改名功能
    print("2. 測試說話者改名功能")
    test_rename_speaker(
        speaker_id="81d60ed8-3c8b-43b8-808d-2dd4409ca814",
        current_name="n1",
        new_name="noise"
    )
    
    # 3. 測試聲紋轉移功能
    print("3. 測試聲紋轉移功能")
    test_transfer_voiceprints(
        source_speaker_id="a372ca19-8531-4f3d-bee2-a7580989acf6",
        source_name="n4",
        target_speaker_id="81d60ed8-3c8b-43b8-808d-2dd4409ca814",
        target_name="noise"
    )
    
    print("測試完成！")
