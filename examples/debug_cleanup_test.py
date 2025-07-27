#!/usr/bin/env python3
"""
Debug cleanup test
"""

import requests
import sys


def debug_cleanup_test():
    """調試清理測試"""
    base_url = "http://localhost:8000"
    
    # 1. 獲取所有sessions
    print("1. 獲取所有sessions...")
    response = requests.get(f"{base_url}/sessions")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        sessions = response.json()
        print(f"   Sessions count: {len(sessions)}")
        
        # 2. 逐個刪除
        print("2. 開始刪除sessions...")
        deleted_count = 0
        for i, session in enumerate(sessions):
            uuid = session['uuid']
            print(f"   刪除 {i+1}/{len(sessions)}: {uuid}")
            delete_response = requests.delete(f"{base_url}/sessions/{uuid}")
            print(f"      狀態: {delete_response.status_code}")
            if delete_response.status_code == 200:
                deleted_count += 1
                print(f"      成功")
            else:
                print(f"      失敗: {delete_response.text}")
        
        print(f"   總計刪除: {deleted_count}/{len(sessions)}")
        
        # 3. 驗證清理結果
        print("3. 驗證清理結果...")
        response = requests.get(f"{base_url}/sessions")
        if response.status_code == 200:
            remaining_sessions = response.json()
            print(f"   剩餘Sessions: {len(remaining_sessions)}")
        else:
            print(f"   檢查Sessions失敗: {response.status_code}")
        
        response = requests.get(f"{base_url}/speechlogs")
        if response.status_code == 200:
            remaining_speechlogs = response.json()
            print(f"   剩餘SpeechLogs: {len(remaining_speechlogs)}")
        else:
            print(f"   檢查SpeechLogs失敗: {response.status_code}")
    
    else:
        print(f"   無法獲取sessions: {response.text}")


if __name__ == "__main__":
    debug_cleanup_test()
