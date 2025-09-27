#!/usr/bin/env python3
"""
語音處理模型比較測試工具

此工具可以：
1. 使用預錄音檔測試不同的語音處理配置
2. 模擬即時處理流程
3. 比較不同模型的效果
4. 生成詳細的測試報告

使用方式:
python model_comparison_test.py --audio_file test_audio/vad_test.wav
"""

import asyncio
import websockets
import json
import requests
import argparse
import os
import time
import wave
import struct
import glob
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from pathlib import Path
from typing import List, Dict, Any, Tuple
import threading
import queue
import math


class VoiceProcessingTester:
    """語音處理測試器"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", resample_method: str = "scipy"):
        self.api_base_url = api_base_url
        self.ws_base_url = api_base_url.replace("http", "ws")
        self.test_results = []
        self.resample_method = resample_method
        
        if resample_method not in ["librosa", "scipy"]:
            raise ValueError("resample_method 必須是 'librosa' 或 'scipy'")
            
        print(f"🔧 使用重採樣方法: {resample_method}")
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        高品質音訊重採樣，避免混疊和量化誤差
        
        Args:
            audio: 輸入音訊
            orig_sr: 原始採樣率
            target_sr: 目標採樣率
            
        Returns:
            重採樣後的音訊
        """
        if orig_sr == target_sr:
            return audio
            
        if self.resample_method == "librosa":
            # 使用librosa的高品質重採樣，採用kaiser_fast模式平衡品質與速度
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type='kaiser_fast')
        
        elif self.resample_method == "scipy":
            # 使用scipy.signal.resample_poly的高品質實現
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a
            
            # 簡化採樣率比例
            common_divisor = gcd(target_sr, orig_sr)
            up = target_sr // common_divisor
            down = orig_sr // common_divisor
            
            print(f"   🔢 重採樣參數: up={up}, down={down} ({orig_sr}Hz → {target_sr}Hz)")
            
            # 使用kaiser窗口的高品質濾波器，beta=8.0提供良好的旁瓣抑制
            # window=('kaiser', 8.0) 可減少混疊失真
            resampled = resample_poly(audio, up, down, window=('kaiser', 8.0))
            return resampled
        
    def check_api_connection(self) -> bool:
        """檢查API服務是否可用"""
        try:
            response = requests.get(f"{self.api_base_url}/docs", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            return False
        except Exception:
            return False
    
    def create_test_session(self) -> str:
        """建立測試用的Session"""
        try:
            response = requests.post(f"{self.api_base_url}/sessions", json={
                "session_type": "test",
                "title": "模型比較測試",
                "summary": "自動化語音處理模型效果比較"
            }, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    session_uuid = result["data"]["uuid"]
                    print(f"✅ 建立測試Session: {session_uuid}")
                    return session_uuid
            
            raise Exception(f"建立Session失敗: {response.text}")
            
        except requests.exceptions.ConnectionError:
            raise Exception(f"❌ 無法連接到API服務 ({self.api_base_url})。請確保API服務正在運行！")
        except requests.exceptions.Timeout:
            raise Exception(f"❌ API請求超時。請檢查API服務狀態。")
        except Exception as e:
            if "Connection refused" in str(e):
                raise Exception(f"❌ API服務未啟動！請先執行: python main.py")
            raise
    
    def load_audio_from_directory(self, directory_path: str, target_sr: int = 16000) -> Tuple[bytes, Dict[str, Any]]:
        """載入資料夾中的多個音檔，合併並重採樣為16kHz單聲道PCM"""
        print(f"📁 載入資料夾音檔: {directory_path}")
        
        # 支援的音檔格式
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        
        # 收集所有音檔
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))
            audio_files.extend(glob.glob(os.path.join(directory_path, f"*{ext.upper()}")))
        
        if not audio_files:
            raise Exception(f"資料夾中沒有找到音檔: {directory_path}")
        
        # 按檔名排序確保正確順序
        audio_files.sort()
        print(f"🎵 找到 {len(audio_files)} 個音檔:")
        for i, file in enumerate(audio_files):
            print(f"   {i+1}. {os.path.basename(file)}")
        
        # 載入並合併所有音檔
        combined_audio = []
        audio_info = {
            "total_files": len(audio_files),
            "files": [],
            "original_duration": 0,
            "resampled_duration": 0,
            "original_sample_rates": set(),
            "target_sample_rate": target_sr
        }
        
        for i, file_path in enumerate(audio_files):
            try:
                # 使用librosa載入音檔，自動轉為單聲道
                audio, sr = librosa.load(file_path, sr=None, mono=True)
                original_duration = len(audio) / sr
                
                # 記錄原始資訊
                audio_info["files"].append({
                    "filename": os.path.basename(file_path),
                    "original_sr": sr,
                    "original_duration": original_duration,
                    "samples": len(audio)
                })
                audio_info["original_sample_rates"].add(sr)
                audio_info["original_duration"] += original_duration
                
                # 重採樣到目標採樣率
                if sr != target_sr:
                    audio = self._resample_audio(audio, sr, target_sr)
                    print(f"   📊 {os.path.basename(file_path)}: {sr}Hz → {target_sr}Hz ({self.resample_method})")
                else:
                    print(f"   ✅ {os.path.basename(file_path)}: 已是 {target_sr}Hz")
                
                combined_audio.append(audio)
                
            except Exception as e:
                print(f"   ❌ 載入失敗 {os.path.basename(file_path)}: {e}")
                continue
        
        if not combined_audio:
            raise Exception("沒有成功載入任何音檔")
        
        # 合併所有音檔
        final_audio = np.concatenate(combined_audio)
        audio_info["resampled_duration"] = len(final_audio) / target_sr
        
        # 轉換為16bit PCM bytes - 使用更精確的方法
        # 正規化到 [-1, 1] 範圍，避免clipping
        max_val = np.max(np.abs(final_audio))
        if max_val > 1.0:
            final_audio = final_audio / max_val  # 動態正規化
            print(f"   ⚠️ 音訊正規化: 最大值 {max_val:.3f} → 1.0")
        else:
            final_audio = np.clip(final_audio, -1.0, 1.0)
        
        # 使用更精確的轉換，減少量化誤差
        # 使用32767而不是32768，避免溢出
        audio_int16 = np.round(final_audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # 驗證轉換品質
        reconstructed = audio_int16.astype(np.float32) / 32767.0
        snr = 20 * np.log10(np.std(final_audio) / np.std(final_audio - reconstructed))
        print(f"   📊 量化品質: SNR = {snr:.1f} dB")
        
        # 轉換原始採樣率集合為列表以便JSON序列化
        audio_info["original_sample_rates"] = list(audio_info["original_sample_rates"])
        audio_info["quantization_snr_db"] = float(snr)
        
        print(f"✅ 音檔處理完成:")
        print(f"   合併檔案: {len(audio_files)} 個")
        print(f"   原始總長度: {audio_info['original_duration']:.2f} 秒")
        print(f"   重採樣後長度: {audio_info['resampled_duration']:.2f} 秒")
        print(f"   原始採樣率: {audio_info['original_sample_rates']}")
        print(f"   目標採樣率: {target_sr} Hz")
        print(f"   輸出格式: 16-bit PCM, {len(audio_bytes)} bytes")
        
        return audio_bytes, audio_info
    
    def load_audio_file(self, file_path: str, target_sr: int = 16000) -> Tuple[bytes, Dict[str, Any]]:
        """載入單個音檔並重採樣為16kHz單聲道PCM"""
        print(f"🎵 載入音檔: {file_path}")
        
        try:
            # 使用librosa載入音檔
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            original_duration = len(audio) / sr
            
            audio_info = {
                "filename": os.path.basename(file_path),
                "original_sr": sr,
                "target_sr": target_sr,
                "original_duration": original_duration,
                "original_samples": len(audio)
            }
            
            # 重採樣到目標採樣率
            if sr != target_sr:
                audio = self._resample_audio(audio, sr, target_sr)
                print(f"   📊 重採樣: {sr}Hz → {target_sr}Hz ({self.resample_method})")
            else:
                print(f"   ✅ 已是目標採樣率: {target_sr}Hz")
            
            # 轉換為16bit PCM bytes
            final_duration = len(audio) / target_sr
            audio_info["resampled_duration"] = final_duration
            audio_info["resampled_samples"] = len(audio)
            
            # 正規化並轉換
            audio = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            print(f"✅ 音檔處理完成:")
            print(f"   原始長度: {original_duration:.2f} 秒 ({sr} Hz)")
            print(f"   重採樣後: {final_duration:.2f} 秒 ({target_sr} Hz)")
            print(f"   輸出: 16-bit PCM, {len(audio_bytes)} bytes")
            
            return audio_bytes, audio_info
            
        except Exception as e:
            print(f"❌ 音檔載入失敗: {e}")
            raise
    
    def chunk_audio_data(self, audio_data: bytes, chunk_size_ms: int = 6000, sample_rate: int = 16000) -> List[bytes]:
        """將音檔分割成指定大小的chunks"""
        bytes_per_sample = 2  # 16-bit
        bytes_per_second = sample_rate * bytes_per_sample
        chunk_size_bytes = int(bytes_per_second * chunk_size_ms / 1000)
        
        chunks = []
        for i in range(0, len(audio_data), chunk_size_bytes):
            chunk = audio_data[i:i + chunk_size_bytes]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        total_duration = len(audio_data) / bytes_per_second
        chunk_duration = chunk_size_ms / 1000
        
        print(f"📊 音檔分割資訊:")
        print(f"   總長度: {total_duration:.2f} 秒")
        print(f"   分割為: {len(chunks)} 個chunks")
        print(f"   每個chunk: {chunk_duration:.2f} 秒 ({chunk_size_bytes} bytes)")
        
        return chunks
    
    async def test_websocket_streaming(self, session_uuid: str, audio_chunks: List[bytes]) -> List[Dict]:
        """測試WebSocket即時處理"""
        print(f"🌐 開始WebSocket串流測試...")
        
        ws_url = f"{self.ws_base_url}/ws/stream?session={session_uuid}"
        results = []
        
        try:
            async with websockets.connect(ws_url) as websocket:
                print(f"✅ WebSocket連線成功: {ws_url}")
                
                # 建立接收結果的任務
                result_queue = asyncio.Queue()
                
                async def receive_results():
                    try:
                        while True:
                            message = await websocket.recv()
                            data = json.loads(message)
                            await result_queue.put(data)
                            print(f"📥 收到結果: Segment {data.get('segment', 'N/A')}")
                    except websockets.exceptions.ConnectionClosed:
                        await result_queue.put(None)  # 結束信號
                
                # 啟動接收任務
                receive_task = asyncio.create_task(receive_results())
                
                # 發送音檔chunks
                for i, chunk in enumerate(audio_chunks):
                    await websocket.send(chunk)
                    print(f"📤 發送chunk {i+1}/{len(audio_chunks)}")
                    
                    # 模擬即時發送間隔
                    await asyncio.sleep(0.1)
                
                # 發送停止信號
                await websocket.send("stop")
                print("🛑 發送停止信號")
                
                # 收集所有結果
                while True:
                    result = await result_queue.get()
                    if result is None:
                        break
                    results.append(result)
                
                receive_task.cancel()
                
        except Exception as e:
            print(f"❌ WebSocket測試失敗: {e}")
            raise
        
        print(f"✅ WebSocket測試完成，收到 {len(results)} 個結果")
        return results
    
    def test_file_api(self, audio_file_path: str) -> Dict:
        """測試檔案上傳API - 使用現有的transcribe_dir接口"""
        print(f"📁 開始檔案API測試...")
        
        try:
            if os.path.isdir(audio_file_path):
                # 直接使用 transcribe_dir 接口處理資料夾
                print(f"📁 使用 transcribe_dir 接口處理資料夾: {audio_file_path}")
                
                response = requests.post(
                    f"{self.api_base_url}/transcribe_dir",
                    data={"path": audio_file_path},
                    timeout=300  # 增加超時時間，因為處理可能較久
                )
                
            else:
                # 單一檔案使用 transcribe 接口
                print(f"📄 使用 transcribe 接口處理單一檔案: {audio_file_path}")
                
                with open(audio_file_path, 'rb') as f:
                    files = {'file': (os.path.basename(audio_file_path), f, 'audio/wav')}
                    response = requests.post(
                        f"{self.api_base_url}/transcribe",
                        files=files,
                        timeout=300
                    )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 檔案API測試完成")
                
                # 如果是 transcribe_dir，結果格式可能不同，需要處理
                if os.path.isdir(audio_file_path):
                    print(f"📊 transcribe_dir 結果: {result}")
                    # transcribe_dir 返回摘要檔案路徑，我們需要讀取實際結果
                    return self._process_transcribe_dir_result(result)
                else:
                    return result
            else:
                raise Exception(f"API請求失敗: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ 檔案API測試失敗: {e}")
            raise
    
    def _process_transcribe_dir_result(self, transcribe_dir_result: Dict) -> Dict:
        """處理 transcribe_dir 的結果格式"""
        # transcribe_dir 通常返回 {"summary_tsv": "path/to/summary.tsv"}
        # 我們需要轉換為與 transcribe 相同的格式以便比較
        
        summary_path = transcribe_dir_result.get("summary_tsv")
        if summary_path and os.path.exists(summary_path):
            print(f"📄 讀取摘要檔案: {summary_path}")
            
            # 讀取TSV摘要檔案
            try:
                import pandas as pd
                df = pd.read_csv(summary_path, sep='\t')
                
                # 轉換為標準格式
                segments = []
                for _, row in df.iterrows():
                    segment = {
                        "start": row.get("start", 0),
                        "end": row.get("end", 0),
                        "speakers": [{
                            "speaker": row.get("speaker", "unknown"),
                            "text": row.get("text", ""),
                            "confidence": row.get("confidence", 0)
                        }]
                    }
                    segments.append(segment)
                
                return {
                    "segments": segments,
                    "pretty": [],  # transcribe_dir 可能沒有pretty格式
                    "stats": {"total_segments": len(segments)}
                }
                
            except Exception as e:
                print(f"⚠️ 無法讀取摘要檔案: {e}")
                return {
                    "segments": [],
                    "pretty": [],
                    "stats": {"error": str(e)}
                }
        else:
            print(f"⚠️ 摘要檔案不存在: {summary_path}")
            return {
                "segments": [],
                "pretty": [],
                "stats": {"error": "No summary file"}
            }
    
    def get_session_results(self, session_uuid: str) -> Dict:
        """獲取Session的詳細結果"""
        try:
            # 獲取Session資訊
            session_response = requests.get(f"{self.api_base_url}/sessions/{session_uuid}")
            session_data = session_response.json()
            
            # 獲取SpeechLogs
            speechlogs_response = requests.get(f"{self.api_base_url}/sessions/{session_uuid}/speechlogs")
            speechlogs_data = speechlogs_response.json()
            
            return {
                "session": session_data,
                "speechlogs": speechlogs_data
            }
        except Exception as e:
            print(f"⚠️ 獲取Session結果失敗: {e}")
            return {}
    
    def compare_results(self, websocket_results: List[Dict], file_api_result: Dict) -> Dict:
        """比較WebSocket和檔案API的結果"""
        print(f"📊 開始結果比較分析...")
        
        comparison = {
            "websocket_segments": len(websocket_results),
            "file_api_segments": len(file_api_result.get("segments", [])),
            "websocket_speakers": [],
            "file_api_speakers": [],
            "processing_differences": []
        }
        
        # 分析WebSocket結果
        for segment in websocket_results:
            for speaker in segment.get("speakers", []):
                comparison["websocket_speakers"].append({
                    "speaker_id": speaker.get("speaker_id"),
                    "text": speaker.get("text", ""),
                    "confidence": speaker.get("confidence", 0)
                })
        
        # 分析檔案API結果
        for segment in file_api_result.get("segments", []):
            for speaker in segment.get("speakers", []):
                comparison["file_api_speakers"].append({
                    "speaker_id": speaker.get("speaker"),
                    "text": speaker.get("text", ""),
                    "confidence": speaker.get("confidence", 0)
                })
        
        print(f"📈 比較完成:")
        print(f"   WebSocket: {len(comparison['websocket_speakers'])} 個語者片段")
        print(f"   檔案API: {len(comparison['file_api_speakers'])} 個語者片段")
        
        return comparison
    
    def save_test_report(self, results: Dict, output_file: str = None):
        """儲存測試報告"""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"voice_processing_test_report_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"📄 測試報告已儲存: {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="語音處理模型比較測試")
    parser.add_argument("--audio_path", default="test_audio_me_ordered_test", 
                        help="測試音檔路徑或包含音檔的資料夾路徑")
    parser.add_argument("--chunk_size_ms", type=int, default=6000,
                        help="WebSocket chunk大小(毫秒)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="目標採樣率(Hz)")
    parser.add_argument("--resample_method", choices=["librosa", "scipy"], default="scipy",
                        help="重採樣方法: librosa 或 scipy (預設)")
    parser.add_argument("--api_url", default="http://localhost:8000",
                        help="API伺服器URL")
    parser.add_argument("--output", help="輸出報告檔案名稱")
    parser.add_argument("--skip_websocket", action="store_true",
                        help="跳過WebSocket測試，只執行檔案API測試")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"❌ 路徑不存在: {args.audio_path}")
        return
    
    is_directory = os.path.isdir(args.audio_path)
    path_type = "資料夾" if is_directory else "檔案"
    
    print("🎯 語音處理模型比較測試開始")
    print(f"   音檔{path_type}: {args.audio_path}")
    print(f"   API: {args.api_url}")
    if not args.skip_websocket:
        print(f"   Chunk大小: {args.chunk_size_ms}ms")
        print(f"   目標採樣率: {args.sample_rate}Hz")
        print(f"   重採樣方法: {args.resample_method}")
    print(f"   跳過WebSocket: {args.skip_websocket}")
    print("-" * 50)
    
    tester = VoiceProcessingTester(args.api_url, args.resample_method)
    
    # 檢查API服務連線
    print("🔍 檢查API服務狀態...")
    if not tester.check_api_connection():
        print(f"❌ 無法連接到API服務 ({args.api_url})")
        print("💡 請確保以下服務正在運行：")
        print("   1. API服務: python main.py")
        print("   2. Weaviate資料庫: docker-compose up -d")
        print("   3. 確保端口8000未被占用")
        return
    else:
        print("✅ API服務連線正常")
    
    try:
        websocket_results = []
        session_results = {}
        session_uuid = None
        
        # 1. WebSocket測試 (可選)
        if not args.skip_websocket:
            print("\n📡 執行WebSocket即時處理測試...")
            session_uuid = tester.create_test_session()
            
            # 載入並處理音檔
            if is_directory:
                audio_data, audio_info = tester.load_audio_from_directory(args.audio_path, args.sample_rate)
            else:
                audio_data, audio_info = tester.load_audio_file(args.audio_path, args.sample_rate)
            
            # 分割音檔
            audio_chunks = tester.chunk_audio_data(audio_data, args.chunk_size_ms, args.sample_rate)
            
            # 執行WebSocket測試
            websocket_results = await tester.test_websocket_streaming(session_uuid, audio_chunks)
            
            # 獲取Session詳細結果
            session_results = tester.get_session_results(session_uuid)
        else:
            print("\n⏭️ 跳過WebSocket測試")
            audio_info = {"skipped": True}
        
        # 2. 檔案API測試 (直接使用現有接口)
        print("\n📁 執行檔案API測試...")
        file_api_result = tester.test_file_api(args.audio_path)
        
        # 3. 比較結果 (如果有WebSocket結果的話)
        if websocket_results:
            comparison = tester.compare_results(websocket_results, file_api_result)
        else:
            comparison = {
                "websocket_segments": 0,
                "file_api_segments": len(file_api_result.get("segments", [])),
                "websocket_speakers": [],
                "file_api_speakers": [
                    {
                        "speaker_id": speaker.get("speaker"),
                        "text": speaker.get("text", ""),
                        "confidence": speaker.get("confidence", 0)
                    }
                    for segment in file_api_result.get("segments", [])
                    for speaker in segment.get("speakers", [])
                ],
                "processing_differences": ["WebSocket測試已跳過"]
            }
        
        # 4. 準備完整報告
        final_report = {
            "test_info": {
                "audio_path": args.audio_path,
                "is_directory": is_directory,
                "chunk_size_ms": args.chunk_size_ms if not args.skip_websocket else None,
                "target_sample_rate": args.sample_rate if not args.skip_websocket else None,
                "resample_method": args.resample_method if not args.skip_websocket else None,
                "websocket_skipped": args.skip_websocket,
                "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "session_uuid": session_uuid
            },
            "audio_info": audio_info,
            "websocket_results": websocket_results,
            "file_api_result": file_api_result,
            "session_results": session_results,
            "comparison": comparison
        }
        
        # 5. 儲存報告
        tester.save_test_report(final_report, args.output)
        
        print("\n🎉 測試完成！")
        print(f"📊 主要結果:")
        if not args.skip_websocket:
            print(f"   音檔資訊: {audio_info}")
            print(f"   WebSocket處理: {len(websocket_results)} 個segments")
        print(f"   檔案API處理: {len(file_api_result.get('segments', []))} 個segments")
        if session_uuid:
            print(f"   Session UUID: {session_uuid}")
        
    except Exception as e:
        print(f"💥 測試過程發生錯誤: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
