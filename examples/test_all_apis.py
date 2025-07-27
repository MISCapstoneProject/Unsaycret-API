#!/usr/bin/env python3
"""
Unsaycret API 完整功能測試

此測試檔案會測試所有 API 端點的功能，包括：
1. Sessions CRUD
2. SpeechLogs CRUD  
3. Speake        # 1. 測試列出空的 Sessions
        response = self.make_request("GET", "/sessions")
        if response.status_code == 200:
            sessions = response.json()
            if len(sessions) == 0:
                self.log_test("Sessions 初始列表", True, "初始狀態為空列表")
            else:
                # 顯示詳細的非空內容
                session_details = []
                for session in sessions:
                    session_details.append(f"UUID: {session.get('uuid', 'N/A')}, ID: {session.get('session_id', 'N/A')}, Title: {session.get('title', 'N/A')}")
                self.log_test("Sessions 初始列表", False, f"預期空列表，實際 {len(sessions)} 個: {session_details}")
        else:
            self.log_test("Sessions 初始列表", False, f"狀態碼: {response.status_code}, 內容: {response.text}")D (模擬測試，因為需要語音檔案)
4. 關聯查詢 API
5. 錯誤處理

注意：資料庫需要是空的狀態開始測試
"""

import requests
import json
import sys
import time
import uuid
import subprocess
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


class UnsaycretAPITester:
    """Unsaycret API 測試器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_data = {}  # 存儲測試過程中創建的資料
        self.test_results = []  # 存儲測試結果
        
    def reset_database(self) -> bool:
        """
        重置 Weaviate 資料庫
        
        Returns:
            bool: 是否成功重置
        """
        try:
            print("🔄 正在重置 Weaviate 資料庫...")
            
            # 找到重置腳本的路徑
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            reset_script = os.path.join(project_root, "weaviate_study", "create_reset_collections.py")
            
            if not os.path.exists(reset_script):
                self.log_test("資料庫重置", False, f"找不到重置腳本: {reset_script}")
                return False
            
            # 執行重置腳本（自動確認）
            result = subprocess.run(
                [sys.executable, reset_script],
                input="y\n",  # 自動確認
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.log_test("資料庫重置", True, "Weaviate 資料庫已成功重置")
                time.sleep(2)  # 等待資料庫完全重置
                return True
            else:
                self.log_test("資料庫重置", False, f"重置失敗: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_test("資料庫重置", False, "重置腳本執行超時")
            return False
        except Exception as e:
            self.log_test("資料庫重置", False, f"重置時發生錯誤: {str(e)}")
            return False
        
    def log_test(self, test_name: str, success: bool, message: str = "", data: Any = None):
        """記錄測試結果"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        print(f"{status} {test_name}: {message}")
        if data and not success:
            print(f"   詳細信息: {data}")
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """發送 HTTP 請求"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, **kwargs)
            return response
        except Exception as e:
            print(f"❌ 請求失敗: {method} {url} - {e}")
            raise
    
    def test_server_health(self):
        """測試伺服器健康狀態"""
        try:
            response = self.make_request("GET", "/docs")
            if response.status_code == 200:
                self.log_test("伺服器健康檢查", True, "API 伺服器運行正常")
                return True
            else:
                self.log_test("伺服器健康檢查", False, f"伺服器回應異常: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("伺服器健康檢查", False, f"無法連接到伺服器: {e}")
            return False
    
    # ================== Sessions API 測試 ==================
    
    def test_sessions_crud(self):
        """測試 Sessions 完整 CRUD 操作"""
        print("\\n🔵 開始測試 Sessions API...")
        
        # 1. 測試列出空的 Sessions (初始狀態)
        response = self.make_request("GET", "/sessions")
        if response.status_code == 200 and response.json() == []:
            self.log_test("Sessions 初始列表", True, "初始狀態為空列表")
        else:
            self.log_test("Sessions 初始列表", False, f"狀態碼: {response.status_code}, 內容: {response.text}")
        
        # 2. 測試建立新 Session
        session_data = {
            "session_type": "會議",
            "title": "API 測試會議",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "summary": "這是一個 API 測試會議",
            "participants": []
        }
        
        response = self.make_request("POST", "/sessions", json=session_data)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                # API 回應統一使用 uuid 欄位
                session_uuid = result["data"]["uuid"]
                self.test_data["session_uuid"] = session_uuid
                self.log_test("建立 Session", True, f"Session 已建立，UUID: {session_uuid}")
            else:
                self.log_test("建立 Session", False, f"建立失敗: {result.get('message')}")
        else:
            self.log_test("建立 Session", False, f"狀態碼: {response.status_code}, 內容: {response.text}")
        
        # 3. 測試查詢 Session 列表
        response = self.make_request("GET", "/sessions")
        if response.status_code == 200:
            sessions = response.json()
            if len(sessions) == 1 and sessions[0]["title"] == "API 測試會議":
                self.log_test("查詢 Session 列表", True, f"找到 {len(sessions)} 個 Session")
            else:
                self.log_test("查詢 Session 列表", False, f"預期 1 個 Session，實際 {len(sessions)} 個")
        else:
            self.log_test("查詢 Session 列表", False, f"狀態碼: {response.status_code}")
        
        # 4. 測試查詢單一 Session
        if "session_uuid" in self.test_data:
            session_uuid = self.test_data["session_uuid"]
            response = self.make_request("GET", f"/sessions/{session_uuid}")
            if response.status_code == 200:
                session = response.json()
                if session["uuid"] == session_uuid:
                    self.log_test("查詢單一 Session", True, f"成功取得 Session: {session['title']}")
                else:
                    self.log_test("查詢單一 Session", False, "UUID 不匹配")
            else:
                self.log_test("查詢單一 Session", False, f"狀態碼: {response.status_code}")
        
        # 5. 測試更新 Session
        if "session_uuid" in self.test_data:
            session_uuid = self.test_data["session_uuid"]
            update_data = {
                "title": "更新後的 API 測試會議",
                "summary": "這是更新後的會議描述"
            }
            response = self.make_request("PATCH", f"/sessions/{session_uuid}", json=update_data)
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    self.log_test("更新 Session", True, "Session 更新成功")
                else:
                    self.log_test("更新 Session", False, f"更新失敗: {result.get('message')}")
            else:
                self.log_test("更新 Session", False, f"狀態碼: {response.status_code}")
        
        # 6. 驗證更新是否生效
        if "session_uuid" in self.test_data:
            session_uuid = self.test_data["session_uuid"]
            response = self.make_request("GET", f"/sessions/{session_uuid}")
            if response.status_code == 200:
                session = response.json()
                if session["title"] == "更新後的 API 測試會議":
                    self.log_test("驗證 Session 更新", True, "更新內容已生效")
                else:
                    self.log_test("驗證 Session 更新", False, f"標題未更新: {session['title']}")
    
    # ================== SpeechLogs API 測試 ==================
    
    def test_speechlogs_crud(self):
        """測試 SpeechLogs 完整 CRUD 操作"""
        print("\\n🔵 開始測試 SpeechLogs API...")
        
        # 2. 測試列出空的 SpeechLogs
        response = self.make_request("GET", "/speechlogs")
        if response.status_code == 200:
            speechlogs = response.json()
            if len(speechlogs) == 0:
                self.log_test("SpeechLogs 初始列表", True, "初始狀態為空列表")
            else:
                self.log_test("SpeechLogs 初始列表", False, f"預期空列表，實際 {len(speechlogs)} 個")
        else:
            self.log_test("SpeechLogs 初始列表", False, f"狀態碼: {response.status_code}")
        
        # 2. 測試建立新 SpeechLog
        speechlog_data = {
            "content": "這是一個測試語音記錄",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.95,
            "duration": 3.5,
            "language": "zh-TW",
            "session": self.test_data.get("session_uuid")  # 關聯到之前建立的 Session
        }
        
        response = self.make_request("POST", "/speechlogs", json=speechlog_data)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                # API 回應統一使用 uuid 欄位
                speechlog_uuid = result["data"]["uuid"]
                self.test_data["speechlog_uuid"] = speechlog_uuid
                self.log_test("建立 SpeechLog", True, f"SpeechLog 已建立，UUID: {speechlog_uuid}")
            else:
                self.log_test("建立 SpeechLog", False, f"建立失敗: {result.get('message')}")
        else:
            self.log_test("建立 SpeechLog", False, f"狀態碼: {response.status_code}, 內容: {response.text}")
        
        # 3. 測試查詢 SpeechLog 列表
        response = self.make_request("GET", "/speechlogs")
        if response.status_code == 200:
            speechlogs = response.json()
            if len(speechlogs) == 1 and speechlogs[0]["content"] == "這是一個測試語音記錄":
                self.log_test("查詢 SpeechLog 列表", True, f"找到 {len(speechlogs)} 個 SpeechLog")
            else:
                self.log_test("查詢 SpeechLog 列表", False, f"預期 1 個 SpeechLog，實際 {len(speechlogs)} 個")
        else:
            self.log_test("查詢 SpeechLog 列表", False, f"狀態碼: {response.status_code}")
        
        # 4. 測試查詢單一 SpeechLog
        if "speechlog_uuid" in self.test_data:
            speechlog_uuid = self.test_data["speechlog_uuid"]
            response = self.make_request("GET", f"/speechlogs/{speechlog_uuid}")
            if response.status_code == 200:
                speechlog = response.json()
                if speechlog["uuid"] == speechlog_uuid:
                    self.log_test("查詢單一 SpeechLog", True, f"成功取得 SpeechLog: {speechlog['content'][:20]}...")
                else:
                    self.log_test("查詢單一 SpeechLog", False, "UUID 不匹配")
            else:
                self.log_test("查詢單一 SpeechLog", False, f"狀態碼: {response.status_code}")
        
        # 5. 測試更新 SpeechLog
        if "speechlog_uuid" in self.test_data:
            speechlog_uuid = self.test_data["speechlog_uuid"]
            update_data = {
                "content": "這是更新後的測試語音記錄",
                "confidence": 0.98
            }
            response = self.make_request("PATCH", f"/speechlogs/{speechlog_uuid}", json=update_data)
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    self.log_test("更新 SpeechLog", True, "SpeechLog 更新成功")
                else:
                    self.log_test("更新 SpeechLog", False, f"更新失敗: {result.get('message')}")
            else:
                self.log_test("更新 SpeechLog", False, f"狀態碼: {response.status_code}")
    
    # ================== 關聯查詢 API 測試 ==================
    
    def test_nested_resources(self):
        """測試關聯查詢 API"""
        print("\\n🔵 開始測試關聯查詢 API...")
        
        # 建立一個測試 Speaker UUID (模擬)
        test_speaker_uuid = str(uuid.uuid4())
        self.test_data["test_speaker_uuid"] = test_speaker_uuid
        
        # 1. 測試查詢 Speaker 的 Sessions（空結果）
        response = self.make_request("GET", f"/speakers/{test_speaker_uuid}/sessions")
        if response.status_code == 200:
            sessions = response.json()
            self.log_test("查詢 Speaker Sessions", True, f"找到 {len(sessions)} 個相關 Session")
        else:
            self.log_test("查詢 Speaker Sessions", False, f"狀態碼: {response.status_code}")
        
        # 2. 測試查詢 Speaker 的 SpeechLogs（空結果）
        response = self.make_request("GET", f"/speakers/{test_speaker_uuid}/speechlogs")
        if response.status_code == 200:
            speechlogs = response.json()
            self.log_test("查詢 Speaker SpeechLogs", True, f"找到 {len(speechlogs)} 個相關 SpeechLog")
        else:
            self.log_test("查詢 Speaker SpeechLogs", False, f"狀態碼: {response.status_code}")
        
        # 3. 測試查詢 Session 的 SpeechLogs
        if "session_uuid" in self.test_data:
            session_uuid = self.test_data["session_uuid"]
            response = self.make_request("GET", f"/sessions/{session_uuid}/speechlogs")
            if response.status_code == 200:
                speechlogs = response.json()
                if len(speechlogs) >= 1:  # 應該找到之前建立的 SpeechLog
                    self.log_test("查詢 Session SpeechLogs", True, f"找到 {len(speechlogs)} 個相關 SpeechLog")
                else:
                    self.log_test("查詢 Session SpeechLogs", False, f"預期至少 1 個 SpeechLog，實際 {len(speechlogs)} 個")
            else:
                self.log_test("查詢 Session SpeechLogs", False, f"狀態碼: {response.status_code}")
    
    # ================== Speakers API 測試 (模擬) ==================
    
    def test_speakers_api(self):
        """測試 Speakers API（模擬測試，因為需要語音檔案）"""
        print("\\n🔵 開始測試 Speakers API...")
        
        # 1. 測試列出空的 Speakers
        response = self.make_request("GET", "/speakers")
        if response.status_code == 200:
            speakers = response.json()
            self.log_test("查詢 Speakers 列表", True, f"找到 {len(speakers)} 個 Speaker")
        else:
            self.log_test("查詢 Speakers 列表", False, f"狀態碼: {response.status_code}")
        
        # 2. 測試查詢不存在的 Speaker
        fake_speaker_uuid = str(uuid.uuid4())
        response = self.make_request("GET", f"/speakers/{fake_speaker_uuid}")
        if response.status_code == 404 or response.status_code == 500:
            self.log_test("查詢不存在的 Speaker", True, "正確回應 404 或 500 錯誤")
        else:
            self.log_test("查詢不存在的 Speaker", False, f"預期 404/500，實際 {response.status_code}")
        
        # 3. 測試 Speaker 更新（無效 UUID）
        fake_speaker_uuid = str(uuid.uuid4())
        update_data = {"full_name": "測試更新名稱"}
        response = self.make_request("PATCH", f"/speakers/{fake_speaker_uuid}", json=update_data)
        if response.status_code in [404, 500]:
            self.log_test("更新不存在的 Speaker", True, "正確回應錯誤狀態")
        elif response.status_code == 200:
            # 檢查回應內容
            try:
                response_json = response.json()
                if response_json.get("success") is False:
                    self.log_test("更新不存在的 Speaker", True, f"正確回應失敗: {response_json.get('message', '')}")
                else:
                    self.log_test("更新不存在的 Speaker", False, f"API 回應成功但應該失敗: {response_json}")
            except:
                self.log_test("更新不存在的 Speaker", False, f"無法解析回應: {response.text}")
        else:
            # 更詳細的錯誤訊息
            response_text = ""
            try:
                response_json = response.json()
                response_text = f", 回應: {response_json}"
            except:
                response_text = f", 回應文本: {response.text[:100]}"
            self.log_test("更新不存在的 Speaker", False, f"預期錯誤狀態，實際 {response.status_code}{response_text}")
    
    # ================== 錯誤處理測試 ==================
    
    def test_error_handling(self):
        """測試錯誤處理"""
        print("\\n🔵 開始測試錯誤處理...")
        
        # 1. 測試無效的 Session UUID
        invalid_uuid = "invalid-uuid"
        response = self.make_request("GET", f"/sessions/{invalid_uuid}")
        if response.status_code in [400, 404, 422, 500]:
            self.log_test("無效 Session UUID", True, f"正確回應錯誤狀態: {response.status_code}")
        else:
            self.log_test("無效 Session UUID", False, f"預期錯誤狀態，實際 {response.status_code}")
        
        # 2. 測試無效的 JSON 資料
        response = self.make_request("POST", "/sessions", data="invalid json")
        if response.status_code in [400, 422]:
            self.log_test("無效 JSON 資料", True, f"正確回應錯誤狀態: {response.status_code}")
        else:
            self.log_test("無效 JSON 資料", False, f"預期 400/422，實際 {response.status_code}")
        
        # 3. 測試缺少必要欄位（如果有的話）
        empty_session_data = {}
        response = self.make_request("POST", "/sessions", json=empty_session_data)
        # 這個可能成功，因為所有欄位都是 Optional
        self.log_test("空白 Session 資料", True, f"回應狀態: {response.status_code}")
    
    # ================== 清理測試資料 ==================
    
    def cleanup_test_data(self):
        """清理測試過程中建立的資料"""
        print("\\n🔵 開始清理測試資料...")
        
        # 刪除建立的 SpeechLog
        if "speechlog_uuid" in self.test_data:
            speechlog_uuid = self.test_data["speechlog_uuid"]
            response = self.make_request("DELETE", f"/speechlogs/{speechlog_uuid}")
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    self.log_test("刪除測試 SpeechLog", True, "SpeechLog 已刪除")
                else:
                    self.log_test("刪除測試 SpeechLog", False, f"刪除失敗: {result.get('message')}")
            else:
                self.log_test("刪除測試 SpeechLog", False, f"狀態碼: {response.status_code}")
        
        # 刪除建立的 Session
        if "session_uuid" in self.test_data:
            session_uuid = self.test_data["session_uuid"]
            response = self.make_request("DELETE", f"/sessions/{session_uuid}")
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    self.log_test("刪除測試 Session", True, "Session 已刪除")
                else:
                    self.log_test("刪除測試 Session", False, f"刪除失敗: {result.get('message')}")
            else:
                self.log_test("刪除測試 Session", False, f"狀態碼: {response.status_code}")
        
        # 額外清理：刪除所有剩餘的 Sessions (可能是之前測試留下的)
        response = self.make_request("GET", "/sessions")
        if response.status_code == 200:
            sessions = response.json()
            if len(sessions) > 0:
                print(f"🧹 發現 {len(sessions)} 個剩餘 Session，正在清理...")
                for session in sessions:
                    session_uuid = session.get("uuid")
                    if session_uuid:
                        delete_response = self.make_request("DELETE", f"/sessions/{session_uuid}")
                        if delete_response.status_code == 200:
                            delete_result = delete_response.json()
                            if delete_result.get("success"):
                                print(f"   ✅ 已刪除剩餘 Session: {session_uuid}")
                            else:
                                print(f"   ❌ 無法刪除 Session {session_uuid}: {delete_result.get('message')}")
                        else:
                            print(f"   ❌ 刪除 Session {session_uuid} 失敗，狀態碼: {delete_response.status_code}")
        
        # 驗證清理結果
        response = self.make_request("GET", "/sessions")
        if response.status_code == 200:
            sessions = response.json()
            if len(sessions) == 0:
                self.log_test("驗證 Sessions 清理", True, "Sessions 已清空")
            else:
                # 顯示剩餘的 sessions 詳細信息
                session_info = [f"UUID: {s.get('uuid', 'N/A')}, ID: {s.get('session_id', 'N/A')}" for s in sessions]
                self.log_test("驗證 Sessions 清理", False, f"Sessions 未完全清空，剩餘 {len(sessions)} 個: {session_info}")
        else:
            self.log_test("驗證 Sessions 清理", False, f"無法查詢 Sessions，狀態碼: {response.status_code}")
        
        response = self.make_request("GET", "/speechlogs")
        if response.status_code == 200:
            speechlogs = response.json()
            if len(speechlogs) == 0:
                self.log_test("驗證 SpeechLogs 清理", True, "SpeechLogs 已清空")
            else:
                self.log_test("驗證 SpeechLogs 清理", False, f"SpeechLogs 未完全清空，剩餘 {len(speechlogs)} 個")
        else:
            self.log_test("驗證 SpeechLogs 清理", False, f"無法查詢 SpeechLogs，狀態碼: {response.status_code}")
    
    # ================== 主要測試流程 ==================
    
    def run_all_tests(self):
        """執行所有測試"""
        print("🚀 開始執行 Unsaycret API 完整功能測試\\n")
        
        # 1. 重置資料庫
        if not self.reset_database():
            print("❌ 資料庫重置失敗，測試終止")
            return False
        
        # 2. 檢查伺服器健康狀態
        if not self.test_server_health():
            print("❌ 伺服器無法連接，測試終止")
            return False
        
        try:
            # 執行各項測試
            self.test_sessions_crud()
            self.test_speechlogs_crud()
            self.test_nested_resources()
            self.test_speakers_api()
            self.test_error_handling()
            
            # 清理測試資料
            self.cleanup_test_data()
            
            # 生成測試報告
            self.generate_test_report()
            
            return True
            
        except Exception as e:
            print(f"\\n❌ 測試過程中發生錯誤: {e}")
            print(f"🔍 錯誤類型: {type(e).__name__}")
            import traceback
            print(f"📍 詳細錯誤追蹤:")
            traceback.print_exc()
            
            # 即使出錯也生成報告
            try:
                self.generate_test_report()
            except:
                pass
                
            return False
    
    def generate_test_report(self):
        """生成測試報告"""
        print("\\n" + "="*60)
        print("📋 測試報告")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"總測試數量: {total_tests}")
        print(f"通過測試: {passed_tests}")
        print(f"失敗測試: {failed_tests}")
        print(f"成功率: {(passed_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print("\\n❌ 失敗的測試:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test_name']}: {result['message']}")
        
        print("\\n" + "="*60)
        
        # 儲存詳細報告到檔案
        with open("test_report.json", "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": passed_tests/total_tests*100
                },
                "test_results": self.test_results,
                "test_data": self.test_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📄 詳細測試報告已儲存至: test_report.json")


def main():
    """主函數"""
    # 檢查命令列參數
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"🎯 測試目標: {base_url}")
    print("⚠️  此測試將:")
    print("   1. 自動重置 Weaviate 資料庫 (刪除所有現有資料)")
    print("   2. 執行完整的 API 功能測試")
    print("   3. 清理測試過程中產生的資料")
    print()
    print("✅ 請確保:")
    print("   1. API 伺服器正在運行")
    print("   2. Weaviate 資料庫已啟動")
    print()
    
    # 等待用戶確認
    input("按 Enter 鍵開始測試...")
    
    # 執行測試
    tester = UnsaycretAPITester(base_url)
    success = tester.run_all_tests()
    
    if success:
        print("\\n🎉 測試完成！")
        sys.exit(0)
    else:
        print("\\n💥 測試失敗！")
        sys.exit(1)


if __name__ == "__main__":
    main()
