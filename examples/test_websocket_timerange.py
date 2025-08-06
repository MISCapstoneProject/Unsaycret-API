#!/usr/bin/env python3
"""
WebSocket Session 時間範圍自動更新測試

測試 WebSocket 連線結束後，Session 的 start_time 和 end_time 
是否正確更新為第一筆和最後一筆 SpeechLog 的實際時間範圍。
"""

import asyncio
import websockets
import json
import requests
from datetime import datetime
from typing import Dict, Any
import numpy as np
import time

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/stream"

def log_test(message: str, level: str = "INFO") -> None:
    """測試日誌輸出"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = "✅" if level == "SUCCESS" else "❌" if level == "ERROR" else "⚠️" if level == "WARNING" else "ℹ️"
    print(f"[{timestamp}] {prefix} {message}")

def make_request(method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
    """統一的 HTTP 請求處理"""
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.request(method, url, **kwargs)
        log_test(f"{method} {endpoint} -> {response.status_code}")
        
        if response.status_code >= 400:
            log_test(f"請求失敗: {response.text}", "ERROR")
            return {"success": False, "error": response.text}
        
        result = response.json()
        if "success" not in result:
            result["success"] = True
        
        return result
    except Exception as e:
        log_test(f"請求異常: {e}", "ERROR")
        return {"success": False, "error": str(e)}

def generate_audio_chunk(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """生成模擬音訊資料"""
    num_samples = int(sample_rate * duration_seconds)
    
    # 生成混合音調的音訊信號
    t = np.linspace(0, duration_seconds, num_samples)
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 音調
        0.2 * np.sin(2 * np.pi * 220 * t) +  # A3 音調
        0.1 * np.random.normal(0, 0.1, num_samples)  # 背景雜音
    )
    
    # 轉換為 16-bit PCM
    signal = np.clip(signal, -1.0, 1.0)
    pcm_data = (signal * 32767).astype(np.int16)
    
    return pcm_data.tobytes()

async def test_websocket_session_timerange():
    """測試 WebSocket Session 時間範圍自動更新"""
    log_test("🧪 開始 WebSocket Session 時間範圍測試")
    
    # 1. 建立測試 Session
    session_data = {
        "session_type": "websocket_test",
        "title": "WebSocket 時間範圍測試"
    }
    
    response = make_request("POST", "/sessions", json=session_data)
    if not response.get("success"):
        log_test("Session 建立失敗", "ERROR")
        return False
    
    session_uuid = response["data"]["uuid"]
    log_test(f"測試 Session 建立成功: {session_uuid}", "SUCCESS")
    
    # 2. 檢查初始 Session 狀態
    session_info = make_request("GET", f"/sessions/{session_uuid}")
    log_test(f"初始狀態 - start_time: {session_info.get('start_time')}")
    log_test(f"初始狀態 - end_time: {session_info.get('end_time')}")
    
    # 3. 建立 WebSocket 連線並發送音訊
    ws_url = f"{WS_URL}?session={session_uuid}"
    log_test(f"連接到 WebSocket: {ws_url}")
    
    connection_start_time = datetime.now()
    
    try:
        async with websockets.connect(ws_url) as websocket:
            log_test("WebSocket 連線建立成功", "SUCCESS")
            
            # 發送多個音訊片段（模擬 6 秒的音訊，分 3 次發送）
            audio_chunks = [
                generate_audio_chunk(2.0),  # 2秒音訊
                generate_audio_chunk(2.0),  # 2秒音訊
                generate_audio_chunk(2.0),  # 2秒音訊
            ]
            
            response_count = 0
            
            for i, chunk in enumerate(audio_chunks):
                log_test(f"發送音訊片段 {i+1}/3 ({len(chunk)} bytes)")
                await websocket.send(chunk)
                
                # 等待並接收回應
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    response_data = json.loads(response)
                    response_count += 1
                    
                    log_test(f"收到回應 {response_count}: Segment {response_data.get('segment', 'N/A')}")
                    log_test(f"  時間範圍: {response_data.get('start', 'N/A')}s - {response_data.get('end', 'N/A')}s")
                    log_test(f"  語者數量: {len(response_data.get('speakers', []))}")
                    
                    # 顯示語者資訊
                    for j, speaker in enumerate(response_data.get('speakers', [])):
                        log_test(f"    語者 {j+1}: {speaker.get('speaker_id', 'N/A')} - \"{speaker.get('text', 'N/A')}\"")
                        if 'absolute_start_time' in speaker:
                            log_test(f"      絕對時間: {speaker['absolute_start_time']}")
                
                except asyncio.TimeoutError:
                    log_test(f"等待回應 {i+1} 超時", "WARNING")
                except json.JSONDecodeError as e:
                    log_test(f"解析回應 {i+1} 失敗: {e}", "ERROR")
                
                # 片段間等待
                await asyncio.sleep(1.0)
            
            # 發送停止信號
            log_test("發送停止信號")
            await websocket.send("stop")
            
            # 等待最後的回應
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    response_count += 1
                    log_test(f"收到最終回應 {response_count}: Segment {response_data.get('segment', 'N/A')}")
            except asyncio.TimeoutError:
                log_test("沒有更多回應", "INFO")
            except websockets.exceptions.ConnectionClosed:
                log_test("WebSocket 連線已關閉", "INFO")
    
    except Exception as e:
        log_test(f"WebSocket 連線失敗: {e}", "ERROR")
        return False
    
    connection_end_time = datetime.now()
    log_test(f"WebSocket 連線持續時間: {(connection_end_time - connection_start_time).total_seconds():.2f} 秒")
    
    # 4. 等待一段時間讓系統處理完成
    log_test("等待系統處理完成...")
    await asyncio.sleep(3.0)
    
    # 5. 檢查連線結束後的 Session 狀態
    log_test("檢查 Session 時間範圍是否正確更新")
    session_info_after = make_request("GET", f"/sessions/{session_uuid}")
    
    log_test(f"更新後狀態 - start_time: {session_info_after.get('start_time')}")
    log_test(f"更新後狀態 - end_time: {session_info_after.get('end_time')}")
    log_test(f"參與者數量: {len(session_info_after.get('participants', []))}")
    
    # 6. 檢查 SpeechLog 記錄
    speechlogs = make_request("GET", f"/sessions/{session_uuid}/speechlogs")
    if isinstance(speechlogs, list):
        log_test(f"產生的 SpeechLog 數量: {len(speechlogs)}")
        
        if speechlogs:
            # 分析時間範圍
            timestamps = []
            end_times = []
            
            for log in speechlogs:
                if log.get('timestamp'):
                    timestamps.append(log['timestamp'])
                    
                    # 計算結束時間
                    if log.get('duration'):
                        try:
                            from datetime import datetime, timedelta
                            start_time = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
                            end_time = start_time + timedelta(seconds=float(log['duration']))
                            end_times.append(end_time.isoformat() + 'Z')
                        except:
                            end_times.append(log['timestamp'])
            
            if timestamps:
                earliest_time = min(timestamps)
                latest_time = max(end_times) if end_times else max(timestamps)
                
                log_test(f"SpeechLog 分析:")
                log_test(f"  最早時間: {earliest_time}")
                log_test(f"  最晚結束時間: {latest_time}")
                
                # 驗證 Session 時間是否正確
                session_start = session_info_after.get('start_time')
                session_end = session_info_after.get('end_time')
                
                if session_start == earliest_time:
                    log_test("✅ Session start_time 正確對應最早的 SpeechLog", "SUCCESS")
                else:
                    log_test(f"❌ Session start_time 不匹配! Session: {session_start}, SpeechLog: {earliest_time}", "ERROR")
                
                if session_end == latest_time:
                    log_test("✅ Session end_time 正確對應最晚的 SpeechLog 結束時間", "SUCCESS")
                else:
                    log_test(f"❌ Session end_time 不匹配! Session: {session_end}, SpeechLog: {latest_time}", "ERROR")
    
    log_test("WebSocket Session 時間範圍測試完成", "SUCCESS")
    return True

async def main():
    """主測試流程"""
    print("🚀 WebSocket Session 時間範圍自動更新測試開始")
    print("=" * 70)
    
    try:
        # 檢查 API 服務器
        health = make_request("GET", "/health")
        if not health.get("status"):
            log_test("API 服務器未運行，請先啟動服務器", "ERROR")
            return
        
        log_test("API 服務器連接正常", "SUCCESS")
        
        # 執行測試
        success = await test_websocket_session_timerange()
        
        print("=" * 70)
        if success:
            log_test("🎉 所有測試通過！", "SUCCESS")
        else:
            log_test("💥 測試失敗", "ERROR")
        
    except KeyboardInterrupt:
        log_test("測試被用戶中斷", "ERROR")
    except Exception as e:
        log_test(f"測試過程中發生異常: {e}", "ERROR")

if __name__ == "__main__":
    asyncio.run(main())
