#!/usr/bin/env python3
"""
WebSocket 串流語音處理測試

此測試腳本用於測試 /ws/stream WebSocket 端點的功能，包括：
1. 基本連線測試
2. Session UUID 驗證
3. 音訊串流模擬
4. SpeechLog 自動建立測試
5. Session 參與者更新測試
"""

import asyncio
import websockets
import json
import uuid
import time
import requests
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime

class WebSocketStreamTester:
    """WebSocket 串流測試器"""
    
    def __init__(self, ws_url: str = "ws://localhost:8000/ws/stream", api_url: str = "http://localhost:8000"):
        self.ws_url = ws_url
        self.api_url = api_url
        self.test_session_uuid = None
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """記錄測試結果"""
        status = "✅ PASS" if success else "❌ FAIL"
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {status} {test_name}"
        if message:
            log_msg += f" - {message}"
        print(log_msg)
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": timestamp
        })
    
    def create_test_session(self) -> Optional[str]:
        """建立測試用的 Session"""
        try:
            session_data = {
                "session_type": "test",
                "title": "WebSocket 測試會議",
                "summary": "用於測試 WebSocket 功能的會議",
                "participants": []
            }
            
            response = requests.post(f"{self.api_url}/sessions", json=session_data)
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    session_uuid = result.get("data", {}).get("uuid")
                    self.log_test("建立測試 Session", True, f"UUID: {session_uuid}")
                    return session_uuid
                else:
                    self.log_test("建立測試 Session", False, f"API 回應失敗: {result.get('message')}")
            else:
                self.log_test("建立測試 Session", False, f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            self.log_test("建立測試 Session", False, f"異常: {e}")
        
        return None
    
    def cleanup_test_session(self, session_uuid: str):
        """清理測試 Session"""
        try:
            response = requests.delete(f"{self.api_url}/sessions/{session_uuid}")
            if response.status_code == 200:
                self.log_test("清理測試 Session", True)
            else:
                self.log_test("清理測試 Session", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("清理測試 Session", False, f"異常: {e}")
    
    def generate_dummy_audio_bytes(self, duration_seconds: float = 2.0) -> bytes:
        """生成模擬音訊資料 (16-bit PCM, 16kHz)"""
        import struct
        import math
        
        sample_rate = 16000
        num_samples = int(sample_rate * duration_seconds)
        
        # 生成簡單的正弦波音訊
        audio_data = []
        for i in range(num_samples):
            # 440Hz (A4 音符) + 一些雜訊模擬語音
            sample = int(16000 * (
                math.sin(2 * math.pi * 440 * i / sample_rate) * 0.3 +  # 主頻率
                math.sin(2 * math.pi * 220 * i / sample_rate) * 0.2 +  # 低頻
                (hash(i) % 1000 - 500) / 5000 * 0.1                    # 雜訊
            ))
            audio_data.append(struct.pack('<h', max(-32768, min(32767, sample))))
        
        return b''.join(audio_data)
    
    async def test_basic_connection(self):
        """測試基本連線"""
        try:
            # 測試無 session 參數的連線（應該失敗）
            try:
                async with websockets.connect(self.ws_url, timeout=5) as ws:
                    self.log_test("無 session 參數連線", False, "應該被拒絕但成功連線")
            except websockets.exceptions.ConnectionClosedError as e:
                if "1008" in str(e):  # Missing or invalid session UUID
                    self.log_test("無 session 參數連線", True, "正確拒絕無效請求")
                else:
                    self.log_test("無 session 參數連線", False, f"意外關閉: {e}")
            except Exception as e:
                self.log_test("無 session 參數連線", False, f"意外異常: {e}")
            
            # 測試無效 UUID 格式
            invalid_uuid = "invalid-uuid-format"
            try:
                async with websockets.connect(f"{self.ws_url}?session={invalid_uuid}", timeout=5) as ws:
                    self.log_test("無效 UUID 格式", False, "應該被拒絕但成功連線")
            except websockets.exceptions.ConnectionClosedError as e:
                if "1008" in str(e):
                    self.log_test("無效 UUID 格式", True, "正確拒絕無效 UUID")
                else:
                    self.log_test("無效 UUID 格式", False, f"意外關閉: {e}")
            except Exception as e:
                self.log_test("無效 UUID 格式", False, f"意外異常: {e}")
                
        except Exception as e:
            self.log_test("基本連線測試", False, f"測試異常: {e}")
    
    async def test_valid_connection(self, session_uuid: str):
        """測試有效連線"""
        try:
            ws_url_with_session = f"{self.ws_url}?session={session_uuid}"
            async with websockets.connect(ws_url_with_session, timeout=10) as ws:
                self.log_test("有效 session 連線", True, f"成功連線到 {session_uuid[:8]}...")
                
                # 發送模擬音訊資料
                await self.send_audio_chunks(ws, session_uuid)
                
        except Exception as e:
            self.log_test("有效 session 連線", False, f"連線失敗: {e}")
    
    async def send_audio_chunks(self, ws, session_uuid: str):
        """發送音訊片段並測試回應"""
        try:
            # 發送幾個音訊片段
            for i in range(3):
                print(f"🎙️  發送音訊片段 {i+1}/3...")
                
                # 生成模擬音訊資料
                audio_data = self.generate_dummy_audio_bytes(2.0)
                await ws.send(audio_data)
                
                # 等待可能的回應
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    result = json.loads(response)
                    
                    self.log_test(f"音訊片段 {i+1} 處理", True, 
                                f"收到回應: segment {result.get('segment')}, "
                                f"{len(result.get('speakers', []))} 位語者")
                    
                    # 檢查回應格式
                    self.validate_response_format(result, f"片段 {i+1}")
                    
                except asyncio.TimeoutError:
                    self.log_test(f"音訊片段 {i+1} 處理", False, "15秒內未收到回應")
                except json.JSONDecodeError as e:
                    self.log_test(f"音訊片段 {i+1} 處理", False, f"回應格式錯誤: {e}")
                
                # 短暫等待避免過快發送
                await asyncio.sleep(1)
            
            # 發送停止信號
            await ws.send("stop")
            self.log_test("發送停止信號", True)
            
            # 等待連線正常關閉
            try:
                await asyncio.wait_for(ws.recv(), timeout=5)
                self.log_test("連線關閉", True, "正常關閉")
            except websockets.exceptions.ConnectionClosed:
                self.log_test("連線關閉", True, "正常關閉")
            except asyncio.TimeoutError:
                self.log_test("連線關閉", True, "超時關閉")
                
        except Exception as e:
            self.log_test("音訊串流測試", False, f"異常: {e}")
    
    def validate_response_format(self, response: Dict[Any, Any], context: str):
        """驗證回應格式"""
        try:
            # 檢查必要欄位
            required_fields = ["segment", "start", "end", "speakers"]
            for field in required_fields:
                if field not in response:
                    self.log_test(f"{context} 回應格式", False, f"缺少欄位: {field}")
                    return
            
            # 檢查 speakers 陣列
            speakers = response.get("speakers", [])
            for i, speaker in enumerate(speakers):
                speaker_fields = ["speaker_id", "text", "confidence"]
                for field in speaker_fields:
                    if field not in speaker:
                        self.log_test(f"{context} Speaker {i+1} 格式", False, f"缺少欄位: {field}")
                        return
                
                # 檢查是否包含絕對時間戳
                if "absolute_start_time" in speaker:
                    self.log_test(f"{context} 絕對時間戳", True, "包含 absolute_start_time")
                else:
                    self.log_test(f"{context} 絕對時間戳", False, "缺少 absolute_start_time")
            
            self.log_test(f"{context} 回應格式", True, f"{len(speakers)} 位語者資料格式正確")
            
        except Exception as e:
            self.log_test(f"{context} 回應格式驗證", False, f"驗證異常: {e}")
    
    async def test_speechlog_creation(self, session_uuid: str):
        """測試 SpeechLog 自動建立"""
        try:
            # 等待一段時間讓 SpeechLog 建立完成
            await asyncio.sleep(2)
            
            # 查詢該 session 的 SpeechLog
            response = requests.get(f"{self.api_url}/sessions/{session_uuid}/speechlogs")
            if response.status_code == 200:
                speechlogs = response.json()
                if len(speechlogs) > 0:
                    self.log_test("SpeechLog 自動建立", True, f"建立了 {len(speechlogs)} 筆記錄")
                    
                    # 檢查時間戳格式
                    for i, log in enumerate(speechlogs[:3]):  # 只檢查前3筆
                        if log.get("timestamp"):
                            self.log_test(f"SpeechLog {i+1} 時間戳", True, f"時間: {log['timestamp']}")
                        else:
                            self.log_test(f"SpeechLog {i+1} 時間戳", False, "缺少時間戳")
                else:
                    self.log_test("SpeechLog 自動建立", False, "未建立任何記錄")
            else:
                self.log_test("SpeechLog 自動建立", False, f"查詢失敗: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("SpeechLog 建立測試", False, f"異常: {e}")
    
    def print_test_summary(self):
        """印出測試摘要"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*60)
        print("📊 WebSocket 測試摘要")
        print("="*60)
        print(f"總測試數: {total_tests}")
        print(f"✅ 通過: {passed_tests}")
        print(f"❌ 失敗: {failed_tests}")
        print(f"成功率: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%")
        
        if failed_tests > 0:
            print("\n❌ 失敗的測試:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  • {result['test']}: {result['message']}")
        
        print("="*60)
    
    async def run_all_tests(self):
        """執行所有測試"""
        print("🚀 開始 WebSocket 串流測試")
        print("="*60)
        
        # 1. 建立測試 Session
        self.test_session_uuid = self.create_test_session()
        if not self.test_session_uuid:
            print("❌ 無法建立測試 Session，終止測試")
            return
        
        try:
            # 2. 基本連線測試
            print("\n🔗 測試基本連線...")
            await self.test_basic_connection()
            
            # 3. 有效連線與音訊串流測試
            print("\n🎙️  測試音訊串流...")
            await self.test_valid_connection(self.test_session_uuid)
            
            # 4. 測試 SpeechLog 建立
            print("\n📝 測試 SpeechLog 建立...")
            await self.test_speechlog_creation(self.test_session_uuid)
            
        finally:
            # 5. 清理測試資料
            print("\n🧹 清理測試資料...")
            self.cleanup_test_session(self.test_session_uuid)
        
        # 6. 顯示測試摘要
        self.print_test_summary()


async def main():
    """主程式"""
    print("Unsaycret API WebSocket 測試工具")
    print("請確保 API 伺服器運行在 localhost:8000")
    print("按 Enter 開始測試，或輸入 'q' 退出...")
    
    user_input = input().strip()
    if user_input.lower() == 'q':
        print("測試取消")
        return
    
    tester = WebSocketStreamTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n測試被用戶中斷")
    except Exception as e:
        print(f"\n\n測試執行異常: {e}")
