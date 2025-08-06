#!/usr/bin/env python3
"""
WebSocket 真實音訊檔案測試

此腳本用於測試真實音訊檔案的 WebSocket 串流處理
"""

import asyncio
import websockets
import json
import requests
import os
import sys
import wave
import time
from pathlib import Path
from typing import Optional

class WebSocketAudioTester:
    """真實音訊 WebSocket 測試器"""
    
    def __init__(self, ws_url: str = "ws://localhost:8000/ws/stream", api_url: str = "http://localhost:8000"):
        self.ws_url = ws_url
        self.api_url = api_url
    
    def create_test_session(self) -> Optional[str]:
        """建立測試 Session"""
        try:
            session_data = {
                "session_type": "test",
                "title": "真實音訊測試會議",
                "summary": "使用真實音訊檔案測試 WebSocket"
            }
            
            response = requests.post(f"{self.api_url}/sessions", json=session_data)
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    return result.get("data", {}).get("uuid")
            return None
        except:
            return None
    
    def load_audio_file(self, file_path: str) -> Optional[bytes]:
        """載入音訊檔案"""
        try:
            if file_path.endswith('.wav'):
                with wave.open(file_path, 'rb') as wav_file:
                    # 檢查音訊格式
                    print(f"音訊格式: {wav_file.getnchannels()} 聲道, {wav_file.getframerate()} Hz, {wav_file.getsampwidth()*8} bit")
                    
                    # 如果不是 16kHz mono，需要轉換
                    if wav_file.getframerate() != 16000 or wav_file.getnchannels() != 1:
                        print("⚠️  音訊格式不符合要求 (需要 16kHz mono)，但仍嘗試發送...")
                    
                    return wav_file.readframes(wav_file.getnframes())
            else:
                # 讀取二進位檔案
                with open(file_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            print(f"❌ 載入音訊檔案失敗: {e}")
            return None
    
    def chunk_audio_data(self, audio_data: bytes, chunk_size: int = 16000 * 2) -> list:
        """將音訊資料分割成片段"""
        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i + chunk_size])
        return chunks
    
    async def stream_audio_file(self, session_uuid: str, audio_file: str):
        """串流音訊檔案"""
        try:
            ws_url_with_session = f"{self.ws_url}?session={session_uuid}"
            
            # 載入音訊檔案
            audio_data = self.load_audio_file(audio_file)
            if not audio_data:
                return
            
            print(f"🎵 載入音訊檔案: {audio_file}")
            print(f"📊 音訊大小: {len(audio_data):,} bytes")
            
            # 將音訊分割成片段
            chunks = self.chunk_audio_data(audio_data)
            print(f"📦 分割成 {len(chunks)} 個片段")
            
            async with websockets.connect(ws_url_with_session, timeout=30) as ws:
                print(f"🔗 已連線到 WebSocket")
                
                # 發送音訊片段
                responses = []
                for i, chunk in enumerate(chunks):
                    print(f"📤 發送片段 {i+1}/{len(chunks)} ({len(chunk)} bytes)")
                    await ws.send(chunk)
                    
                    # 嘗試接收回應
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        result = json.loads(response)
                        responses.append(result)
                        
                        print(f"📥 收到回應 {i+1}: segment {result.get('segment')}")
                        speakers = result.get('speakers', [])
                        for j, sp in enumerate(speakers):
                            print(f"   🗣️  Speaker {j+1}: {sp.get('speaker_id')} - \"{sp.get('text', '')[:50]}...\"")
                            if sp.get('absolute_start_time'):
                                print(f"       ⏰ 絕對時間: {sp.get('absolute_start_time')}")
                        
                    except asyncio.TimeoutError:
                        print(f"⏰ 片段 {i+1} 回應超時")
                    except json.JSONDecodeError as e:
                        print(f"❌ 片段 {i+1} 回應格式錯誤: {e}")
                    
                    # 短暫延遲模擬即時串流
                    await asyncio.sleep(0.5)
                
                # 發送停止信號
                print("🛑 發送停止信號...")
                await ws.send("stop")
                
                # 等待最後的回應
                try:
                    final_response = await asyncio.wait_for(ws.recv(), timeout=5)
                    print("📥 收到最終回應")
                except:
                    print("✅ 連線正常關閉")
                
                return responses
                
        except Exception as e:
            print(f"❌ 串流測試失敗: {e}")
            return []
    
    async def check_speechlogs(self, session_uuid: str):
        """檢查生成的 SpeechLog"""
        try:
            # 等待資料處理完成
            await asyncio.sleep(3)
            
            response = requests.get(f"{self.api_url}/sessions/{session_uuid}/speechlogs")
            if response.status_code == 200:
                speechlogs = response.json()
                print(f"\n📝 生成的 SpeechLog 數量: {len(speechlogs)}")
                
                for i, log in enumerate(speechlogs):
                    print(f"   {i+1}. Speaker: {log.get('speaker', 'N/A')}")
                    print(f"      內容: \"{log.get('content', '')[:100]}...\"")
                    print(f"      時間: {log.get('timestamp', 'N/A')}")
                    print(f"      信心值: {log.get('confidence', 'N/A')}")
                    print(f"      持續時間: {log.get('duration', 'N/A')}s")
                    print()
                
                return speechlogs
            else:
                print(f"❌ 查詢 SpeechLog 失敗: HTTP {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ 檢查 SpeechLog 失敗: {e}")
            return []


def find_audio_files():
    """尋找可用的音訊檔案"""
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    
    # 檢查常見位置
    search_paths = [
        ".",
        "examples",
        "data",
        "test_data",
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Downloads")
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for file in Path(search_path).rglob("*"):
                if file.suffix.lower() in audio_extensions:
                    audio_files.append(str(file))
    
    return audio_files[:10]  # 限制顯示前10個


async def main():
    """主程式"""
    print("🎵 WebSocket 真實音訊測試工具")
    print("="*50)
    
    # 尋找音訊檔案
    audio_files = find_audio_files()
    
    if not audio_files:
        print("❌ 未找到任何音訊檔案")
        print("請將音訊檔案放在以下位置之一:")
        print("  • 當前目錄")
        print("  • examples/ 目錄")
        print("  • data/ 目錄")
        print("支援格式: .wav, .mp3, .flac, .m4a")
        return
    
    print("📁 找到以下音訊檔案:")
    for i, file in enumerate(audio_files):
        file_size = os.path.getsize(file) / 1024 / 1024  # MB
        print(f"  {i+1}. {file} ({file_size:.1f} MB)")
    
    print(f"\n請選擇音訊檔案 (1-{len(audio_files)})，或輸入檔案路徑:")
    user_input = input(">>> ").strip()
    
    # 選擇音訊檔案
    audio_file = None
    if user_input.isdigit():
        index = int(user_input) - 1
        if 0 <= index < len(audio_files):
            audio_file = audio_files[index]
    elif os.path.exists(user_input):
        audio_file = user_input
    
    if not audio_file:
        print("❌ 無效的選擇")
        return
    
    print(f"✅ 選擇音訊檔案: {audio_file}")
    
    # 開始測試
    tester = WebSocketAudioTester()
    
    # 建立測試 Session
    print("\n🔧 建立測試 Session...")
    session_uuid = tester.create_test_session()
    if not session_uuid:
        print("❌ 無法建立測試 Session")
        return
    
    print(f"✅ Session UUID: {session_uuid}")
    
    try:
        # 串流音訊檔案
        print(f"\n🚀 開始串流音訊檔案...")
        responses = await tester.stream_audio_file(session_uuid, audio_file)
        
        # 檢查 SpeechLog
        print(f"\n🔍 檢查生成的 SpeechLog...")
        speechlogs = await tester.check_speechlogs(session_uuid)
        
        # 顯示摘要
        print(f"\n📊 測試摘要:")
        print(f"  • 處理的音訊片段: {len(responses)}")
        print(f"  • 生成的 SpeechLog: {len(speechlogs)}")
        print(f"  • Session UUID: {session_uuid}")
        
    finally:
        # 清理 Session (可選)
        cleanup = input("\n🗑️  是否刪除測試 Session? (y/N): ").strip().lower()
        if cleanup == 'y':
            try:
                requests.delete(f"http://localhost:8000/sessions/{session_uuid}")
                print("✅ 測試 Session 已清理")
            except:
                print("⚠️  清理 Session 失敗")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n測試被用戶中斷")
    except Exception as e:
        print(f"\n\n測試執行異常: {e}")
