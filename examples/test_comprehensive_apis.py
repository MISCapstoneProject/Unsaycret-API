#!/usr/bin/env python3
"""
Unsaycret API 全面測試套件

這個測試套件涵蓋更多複雜的測試場景，包括：
1. 有資料狀態下的所有API功能
2. 關聯資料的查詢測試
3. Speaker完整功能測試（包含實際聲紋）
4. 邊界條件和異常情況測試
5. 併發操作測試
"""

import requests
import json
import sys
import time
import uuid
import subprocess
import os
import threading
import tempfile
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class ComprehensiveAPITester:
    """全面的API測試器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_data = {}
        self.test_results = []
        
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
            response = requests.request(method, url, timeout=30, **kwargs)
            return response
        except requests.RequestException as e:
            print(f"❌ 請求失敗: {method} {url} - {e}")
            raise
    
    def reset_database(self) -> bool:
        """重置資料庫"""
        try:
            print("🔄 正在重置 Weaviate 資料庫...")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            reset_script = os.path.join(project_root, "weaviate_study", "create_reset_collections.py")
            
            if not os.path.exists(reset_script):
                self.log_test("資料庫重置", False, f"找不到重置腳本: {reset_script}")
                return False
            
            result = subprocess.run(
                [sys.executable, reset_script],
                input="y\n",
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.log_test("資料庫重置", True, "Weaviate 資料庫已成功重置")
                time.sleep(2)
                return True
            else:
                self.log_test("資料庫重置", False, f"重置失敗: {result.stderr}")
                return False
                
        except Exception as e:
            self.log_test("資料庫重置", False, f"重置時發生錯誤: {str(e)}")
            return False
    
    def create_test_audio_file(self) -> str:
        """創建測試用的音檔（模擬）"""
        # 創建一個小的WAV檔案（靜音）
        sample_rate = 16000
        duration = 1  # 1秒
        silence = np.zeros(sample_rate * duration, dtype=np.int16)
        
        # 創建臨時WAV檔案
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        
        # 寫入WAV header（簡化版本）
        import wave
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 單聲道
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(silence.tobytes())
        
        return temp_file.name
    
    def test_with_populated_data(self):
        """測試有資料狀態下的API功能"""
        print("\\n🔵 開始測試有資料狀態下的API功能...")
        
        # 1. 建立多個 Sessions
        print("📝 建立測試資料...")
        session_uuids = []
        for i in range(3):
            session_data = {
                "session_type": f"type_{i}",
                "title": f"測試會議 {i+1}",
                "summary": f"這是第 {i+1} 個測試會議"
            }
            response = self.make_request("POST", "/sessions", json=session_data)
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    session_uuid = result["data"]["uuid"]
                    session_uuids.append(session_uuid)
                    self.test_data[f"session_{i}"] = session_uuid
        
        self.log_test("建立多個Sessions", len(session_uuids) == 3, f"成功建立 {len(session_uuids)} 個Sessions")
        
        # 2. 建立多個 SpeechLogs，關聯到不同Sessions
        speechlog_uuids = []
        for i, session_uuid in enumerate(session_uuids):
            for j in range(2):  # 每個Session建立2個SpeechLog
                speechlog_data = {
                    "content": f"這是會議 {i+1} 的第 {j+1} 段語音記錄",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "confidence": 0.8 + (j * 0.1),
                    "duration": 2.5 + j,
                    "language": "zh-TW",
                    "session": session_uuid
                }
                response = self.make_request("POST", "/speechlogs", json=speechlog_data)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        speechlog_uuid = result["data"]["uuid"]
                        speechlog_uuids.append(speechlog_uuid)
                        self.test_data[f"speechlog_{i}_{j}"] = speechlog_uuid
        
        self.log_test("建立多個SpeechLogs", len(speechlog_uuids) == 6, f"成功建立 {len(speechlog_uuids)} 個SpeechLogs")
        
        # 3. 測試有資料時的列表查詢
        response = self.make_request("GET", "/sessions")
        if response.status_code == 200:
            sessions = response.json()
            if len(sessions) == 3:
                self.log_test("有資料時的Sessions查詢", True, f"正確返回 {len(sessions)} 個Sessions")
                # 驗證排序
                session_ids = [int(s["session_id"]) for s in sessions if s["session_id"].isdigit()]
                is_sorted = session_ids == sorted(session_ids, reverse=True)
                self.log_test("Sessions排序驗證", is_sorted, f"Sessions按ID排序: {session_ids}")
            else:
                self.log_test("有資料時的Sessions查詢", False, f"預期3個，實際 {len(sessions)} 個")
        else:
            self.log_test("有資料時的Sessions查詢", False, f"狀態碼: {response.status_code}")
        
        response = self.make_request("GET", "/speechlogs")
        if response.status_code == 200:
            speechlogs = response.json()
            self.log_test("有資料時的SpeechLogs查詢", len(speechlogs) == 6, f"正確返回 {len(speechlogs)} 個SpeechLogs")
        else:
            self.log_test("有資料時的SpeechLogs查詢", False, f"狀態碼: {response.status_code}")
        
        # 4. 測試關聯查詢（有實際關聯資料）
        if session_uuids:
            first_session = session_uuids[0]
            response = self.make_request("GET", f"/sessions/{first_session}/speechlogs")
            if response.status_code == 200:
                related_speechlogs = response.json()
                self.log_test("Session-SpeechLog關聯查詢", len(related_speechlogs) == 2, 
                            f"第一個Session找到 {len(related_speechlogs)} 個關聯SpeechLog")
            else:
                self.log_test("Session-SpeechLog關聯查詢", False, f"狀態碼: {response.status_code}")
    
    def test_speaker_voice_verification(self):
        """測試語音驗證功能"""
        print("\\n🔵 開始測試語音驗證功能...")
        
        # 創建測試音檔
        audio_file_path = None
        try:
            audio_file_path = self.create_test_audio_file()
            
            # 測試語音驗證（空資料庫）
            with open(audio_file_path, 'rb') as f:
                files = {'file': ('test_audio.wav', f, 'audio/wav')}
                data = {'threshold': 0.5, 'max_results': 3}
                response = self.make_request("POST", "/speakers/verify", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                expected_no_speaker = not result.get("is_known_speaker", True)
                self.log_test("語音驗證（空資料庫）", expected_no_speaker, 
                            f"正確識別為未知語者: {result.get('message', '')}")
            else:
                self.log_test("語音驗證（空資料庫）", False, f"狀態碼: {response.status_code}")
        
        except Exception as e:
            self.log_test("語音驗證測試", False, f"測試過程發生錯誤: {str(e)}")
        
        finally:
            # 清理測試音檔
            if audio_file_path and os.path.exists(audio_file_path):
                try:
                    os.remove(audio_file_path)
                except:
                    pass
    
    def test_concurrent_operations(self):
        """測試併發操作"""
        print("\\n🔵 開始測試併發操作...")
        
        def create_session(session_id):
            """併發建立Session"""
            try:
                session_data = {
                    "title": f"併發測試Session {session_id}",
                    "session_type": "concurrent_test"
                }
                response = self.make_request("POST", "/sessions", json=session_data)
                return response.status_code == 200 and response.json().get("success", False)
            except:
                return False
        
        # 使用ThreadPoolExecutor進行併發測試
        concurrent_sessions = 5
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=concurrent_sessions) as executor:
            futures = [executor.submit(create_session, i) for i in range(concurrent_sessions)]
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
        
        self.log_test("併發建立Sessions", success_count == concurrent_sessions, 
                    f"成功建立 {success_count}/{concurrent_sessions} 個Sessions")
        
        # 測試併發查詢
        def query_sessions():
            try:
                response = self.make_request("GET", "/sessions")
                return response.status_code == 200
            except:
                return False
        
        query_success = 0
        concurrent_queries = 10
        
        with ThreadPoolExecutor(max_workers=concurrent_queries) as executor:
            futures = [executor.submit(query_sessions) for _ in range(concurrent_queries)]
            for future in as_completed(futures):
                if future.result():
                    query_success += 1
        
        self.log_test("併發查詢Sessions", query_success == concurrent_queries, 
                    f"成功執行 {query_success}/{concurrent_queries} 個併發查詢")
    
    def test_boundary_conditions(self):
        """測試邊界條件"""
        print("\\n🔵 開始測試邊界條件...")
        
        # 1. 測試超長字串
        long_title = "A" * 1000  # 1000字符的標題
        session_data = {
            "title": long_title,
            "summary": "B" * 2000  # 2000字符的摘要
        }
        response = self.make_request("POST", "/sessions", json=session_data)
        self.log_test("超長字串處理", response.status_code in [200, 400], 
                    f"處理超長字串回應: {response.status_code}")
        
        # 2. 測試特殊字符
        special_chars_data = {
            "title": "測試🎤語音📝記錄💻系統",
            "summary": "包含特殊字符: @#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        }
        response = self.make_request("POST", "/sessions", json=special_chars_data)
        self.log_test("特殊字符處理", response.status_code == 200, 
                    f"處理特殊字符回應: {response.status_code}")
        
        # 3. 測試空值和None值
        empty_data = {
            "title": "",
            "summary": None,
            "session_type": ""
        }
        response = self.make_request("POST", "/sessions", json=empty_data)
        self.log_test("空值處理", response.status_code == 200, 
                    f"處理空值回應: {response.status_code}")
    
    def test_error_recovery(self):
        """測試錯誤恢復"""
        print("\\n🔵 開始測試錯誤恢復...")
        
        # 1. 測試無效UUID格式
        invalid_uuids = [
            "invalid-uuid",
            "12345",
            "not-a-uuid-at-all",
            "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        ]
        
        for invalid_uuid in invalid_uuids:
            response = self.make_request("GET", f"/sessions/{invalid_uuid}")
            expected_error = response.status_code in [400, 404, 422, 500]
            self.log_test(f"無效UUID處理 ({invalid_uuid[:10]}...)", expected_error, 
                        f"回應狀態: {response.status_code}")
        
        # 1.1 特殊情況：空字串（會重定向到/sessions）
        response = self.make_request("GET", "/sessions/")
        # 空字串會導致重定向到/sessions端點，應該返回200和sessions列表
        expected_redirect = response.status_code == 200
        self.log_test("無效UUID處理 (空字串)", expected_redirect, 
                    f"空字串重定向回應狀態: {response.status_code}")
        
        # 2. 測試不存在的資源
        fake_uuid = str(uuid.uuid4())
        response = self.make_request("GET", f"/sessions/{fake_uuid}")
        self.log_test("不存在資源處理", response.status_code in [404, 500], 
                    f"查詢不存在Session回應: {response.status_code}")
        
        # 3. 測試惡意輸入
        malicious_data = {
            "title": "<script>alert('xss')</script>",
            "summary": "'; DROP TABLE sessions; --"
        }
        response = self.make_request("POST", "/sessions", json=malicious_data)
        # 系統應該能處理這些輸入而不崩潰
        self.log_test("惡意輸入處理", response.status_code in [200, 400], 
                    f"處理惡意輸入回應: {response.status_code}")
    
    def cleanup_test_data(self):
        """清理所有測試資料"""
        print("\\n🔵 開始清理測試資料...")
        
        # 清理所有Sessions（這會級聯清理相關的SpeechLogs）
        response = self.make_request("GET", "/sessions")
        if response.status_code == 200:
            sessions = response.json()
            deleted_count = 0
            for session in sessions:
                delete_response = self.make_request("DELETE", f"/sessions/{session['uuid']}")
                if delete_response.status_code == 200:
                    deleted_count += 1
            
            self.log_test("清理測試Sessions", deleted_count == len(sessions), 
                        f"成功清理 {deleted_count}/{len(sessions)} 個Sessions")
        
        # 驗證清理結果
        response = self.make_request("GET", "/sessions")
        sessions_cleaned = response.status_code == 200 and len(response.json()) == 0
        self.log_test("驗證Sessions清理", sessions_cleaned, 
                    "Sessions已完全清空" if sessions_cleaned else "Sessions未完全清空")
        
        response = self.make_request("GET", "/speechlogs")
        speechlogs_cleaned = response.status_code == 200 and len(response.json()) == 0
        self.log_test("驗證SpeechLogs清理", speechlogs_cleaned, 
                    "SpeechLogs已完全清空" if speechlogs_cleaned else "SpeechLogs未完全清空")
    
    def run_comprehensive_tests(self):
        """執行全面測試"""
        print("🚀 開始執行 Unsaycret API 全面測試\\n")
        
        # 1. 重置資料庫
        if not self.reset_database():
            print("❌ 資料庫重置失敗，測試終止")
            return False
        
        try:
            # 2. 測試基本功能
            self.test_with_populated_data()
            
            # 3. 測試語音驗證
            self.test_speaker_voice_verification()
            
            # 4. 測試併發操作
            self.test_concurrent_operations()
            
            # 5. 測試邊界條件
            self.test_boundary_conditions()
            
            # 6. 測試錯誤恢復
            self.test_error_recovery()
            
            # 7. 清理測試資料
            self.cleanup_test_data()
            
            # 8. 生成測試報告
            self.generate_test_report()
            
            return True
            
        except Exception as e:
            print(f"\\n❌ 測試過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_test_report(self):
        """生成測試報告"""
        print("\\n" + "="*60)
        print("📋 全面測試報告")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"總測試數量: {total_tests}")
        print(f"通過測試: {passed_tests}")
        print(f"失敗測試: {failed_tests}")
        print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\\n❌ 失敗的測試:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test_name']}: {result['message']}")
        
        # 儲存詳細報告
        with open("comprehensive_test_report.json", "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": (passed_tests/total_tests)*100
                },
                "test_results": self.test_results
            }, f, indent=2, ensure_ascii=False)
        
        print("\\n📄 詳細測試報告已儲存至: comprehensive_test_report.json")


def main():
    """主函數"""
    import sys
    
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"🎯 測試目標: {base_url}")
    print("⚠️  此全面測試將:")
    print("   1. 自動重置 Weaviate 資料庫")
    print("   2. 建立測試資料並驗證所有功能")
    print("   3. 測試併發操作和邊界條件")
    print("   4. 測試錯誤恢復機制")
    print("   5. 清理所有測試資料")
    print()
    print("✅ 請確保:")
    print("   1. API 伺服器正在運行")
    print("   2. Weaviate 資料庫已啟動")
    print()
    
    input("按 Enter 鍵開始全面測試...")
    
    tester = ComprehensiveAPITester(base_url)
    success = tester.run_comprehensive_tests()
    
    if success:
        print("\\n🎉 全面測試完成！")
        sys.exit(0)
    else:
        print("\\n💥 全面測試失敗！")
        sys.exit(1)


if __name__ == "__main__":
    main()
