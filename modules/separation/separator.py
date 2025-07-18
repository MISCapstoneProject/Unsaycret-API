"""
===============================================================================
Voice_ID
即時語者分離與識別系統 (Real-time Speech Separation and Speaker Identification System)
===============================================================================

版本：v2.2.0
作者：EvanLo62, CYouuu
最後更新：2025-07-01

功能摘要：
-----------
本系統實現了先進的即時語音處理功能，能夠在混合語音環境中實時將不同語者的聲音分離，
並利用深度學習模型對每位語者進行辨識與追蹤。主要特點包括：

 1. 即時處理：邊錄音邊處理，不需等待完整錄音
 2. 語者分離：能夠將多位語者的混合語音分離成獨立的音檔（支援最多3人）
 3. 即時識別：分離後立即進行語者識別，顯示實時識別結果
 4. 聲紋更新：自動更新語者聲紋向量，提高識別準確率
 5. 語者管理：獨立模組化的語者與聲紋管理功能

** 重要說明 **：目前使用的語者分離模型是 ConvTasNet 3人預訓練模型，
因此本系統使用時可以分離最多三個說話者的混合語音。
 
系統模組架構：
-----------
 - speaker_system_v2.py：主系統，負責語者分離與識別
 - main_identify_v5.py：語者識別引擎，負責聲紋比對
 - speaker_manager.py：語者與聲紋管理模組

技術架構：
-----------
 - 語者分離模型: ConvTasNet (16kHz 三聲道分離)
 - 語者識別模型: SpeechBrain ECAPA-TDNN 模型 (192維特徵向量)
 - 向量資料庫: Weaviate，用於儲存和檢索語者嵌入向量
 - 即時處理: 多執行緒並行處理，邊錄音邊識別
 - 音訊增強: 頻譜閘控降噪、維納濾波、動態範圍壓縮，提高分離品質

Weaviate 資料庫設定：
-----------
 - 安裝並啟動 Weaviate 向量資料庫，使用docker-compose.yml配置：
   ```
   docker-compose up -d
   ```
 - 執行 `create_collections.py` 建立必要的2個集合：
   ```
   python create_collections.py
   ```
 - 若要匯入現有語者嵌入向量，可執行：
   ```
   python weaviate_studY/npy_to_weaviate.py
   ```

處理流程：
-----------
 1. 錄音：連續從麥克風接收音訊流
 2. 分塊處理：每6秒音訊(可自訂)為一個處理單元，重疊率50%
 3. 分離處理：將每段混合音訊分離為獨立的聲音流
 4. 即時識別：對每位分離後的語者立即進行識別
 5. 顯示結果：即時顯示每段識別結果及識別型態

使用方式：
-----------
 1. 直接運行主程式:
    ```
    python speaker_system_v2.py
    ```

 2. 按下 Ctrl+C 停止錄音和識別

前置需求：
-----------
 - Python 3.9+
 - PyTorch with torchaudio
 - SpeechBrain
 - PyAudio (錄音功能)
 - Weaviate 向量資料庫 (需通過 Docker 啟動)
 - 其他依賴套件 (見 requirements.txt)

系統參數：
-----------
 - THRESHOLD_LOW = 0.26: 過於相似，不更新向量
 - THRESHOLD_UPDATE = 0.34: 相似度足夠，更新向量
 - THRESHOLD_NEW = 0.385: 超過此值視為新語者
 - WINDOW_SIZE = 6: 處理窗口大小（秒）
 - OVERLAP = 0.5: 窗口重疊率

輸出結果：
-----------
 - 分離後的音檔: 16K-model/Audios-16K-IDTF/ 目錄下
 - 混合音檔: 同目錄下，前綴為 mixed_audio_
 - 日誌檔案: system_output.log

詳細資訊：
-----------
請參考專案文件: https://github.com/LCY000/ProjectStudy_SpeechRecognition

===============================================================================
"""

import os
import numpy as np
import torch
import torchaudio
import pyaudio # type: ignore
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Any, Union
from speechbrain.inference import SepformerSeparation as separator
from speechbrain.inference import SpeakerRecognition
import noisereduce as nr # type: ignore
import threading
import time
from scipy import signal
from scipy.ndimage import uniform_filter1d

# 修改模型載入方式
try:
    from asteroid.models import ConvTasNet
    USE_ASTEROID = True
except ImportError:
    from transformers import AutoModel
    USE_ASTEROID = False

# 導入日誌模組
from utils.logger import get_logger

# 導入 main_identify_v5 模組
from modules.identification import VID_identify_v5 as speaker_id

# 基本錄音參數
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
TARGET_RATE = 16000
WINDOW_SIZE = 6
OVERLAP = 0.5
DEVICE_INDEX = None

# 音訊處理參數
MIN_ENERGY_THRESHOLD = 0.001
NOISE_REDUCE_STRENGTH = 0.05  # 降低降噪強度以保持音質
MAX_BUFFER_MINUTES = 5
SNR_THRESHOLD = 8  # 降低 SNR 閾值

# 音訊品質改善參數
WIENER_FILTER_STRENGTH = 0.01  # 更溫和的維納濾波
HIGH_FREQ_CUTOFF = 7500  # 提高高頻截止點
DYNAMIC_RANGE_COMPRESSION = 0.7  # 動態範圍壓縮

# ConvTasNet 模型參數
MODEL_NAME = "JorisCos/ConvTasNet_Libri3Mix_sepnoisy_16k"
NUM_SPEAKERS = 3

# 全域參數設定，使用 v5 版本的閾值
EMBEDDING_DIR = "embeddingFiles"  # 所有語者嵌入資料的根目錄
THRESHOLD_LOW = speaker_id.THRESHOLD_LOW     # 過於相似，不更新
THRESHOLD_UPDATE = speaker_id.THRESHOLD_UPDATE # 更新嵌入向量
THRESHOLD_NEW = speaker_id.THRESHOLD_NEW    # 判定為新語者

# 輸出目錄
OUTPUT_DIR = "R3SI/Audio-storage"  # 儲存分離後音訊的目錄
IDENTIFIED_DIR = "R3SI/Identified-Speakers"

# 初始化日誌系統
logger = get_logger(__name__)


# ================== 語者分離部分 ======================

class AudioSeparator:
    def __init__(self, enable_noise_reduction=True, snr_threshold=SNR_THRESHOLD):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_noise_reduction = enable_noise_reduction
        self.snr_threshold = snr_threshold
        
        logger.info(f"使用設備: {self.device}")
        logger.info(f"模型: {MODEL_NAME}")
        
        # 設計更溫和的低通濾波器
        nyquist = TARGET_RATE // 2
        cutoff = min(HIGH_FREQ_CUTOFF, nyquist - 100)
        self.lowpass_filter = signal.butter(2, cutoff / nyquist, btype='low', output='sos')
        
        try:
            logger.info("載入模型中...")
            self.model = self._load_model()
            logger.info("模型載入完成")
            self._test_model()
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            raise
        
        try:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=RATE,
                new_freq=TARGET_RATE
            ).to(self.device)
        except Exception as e:
            logger.error(f"重新取樣器初始化失敗: {e}")
            raise
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []
        self.is_recording = False
        self.output_files = []  # 儲存分離後的音檔路徑
        self.save_audio_files = True  # 設定: 是否將分離後的音訊儲存為wav檔案
        
        # 處理統計
        self.processing_stats = {
            'segments_processed': 0,
            'segments_skipped': 0,
            'errors': 0
        }
        
        self.max_buffer_size = int(RATE * MAX_BUFFER_MINUTES * 60 / CHUNK)
        logger.info("AudioSeparator 初始化完成")

    def _load_model(self):
        """載入 ConvTasNet 模型"""
        if USE_ASTEROID:
            try:
                model = ConvTasNet.from_pretrained(MODEL_NAME)
                model = model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                logger.warning(f"Asteroid 載入失敗，嘗試其他方法...")
        
        try:
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.info("嘗試 torch.hub 載入...")
            return self._manual_load_model()

    def _manual_load_model(self):
        """手動載入模型"""
        try:
            model_dir = "models/ConvTasNet"
            os.makedirs(model_dir, exist_ok=True)
            model = torch.hub.load('JorisCos/ConvTasNet', 'ConvTasNet_Libri3Mix_sepnoisy_16k', pretrained=True)
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"所有載入方法都失敗: {e}")
            raise

    def _test_model(self):
        """測試模型"""
        try:
            test_audio = torch.randn(1, 1, 16000).to(self.device)
            with torch.no_grad():
                output = self.model(test_audio)
                logger.info("模型測試通過")
        except Exception as e:
            logger.error(f"模型測試失敗: {e}")
            raise

    def estimate_snr(self, signal):
        """估算信號雜訊比"""
        try:
            signal_power = np.mean(signal ** 2)
            if len(signal) > 1000:
                noise_estimate = np.std(signal[-1000:]) ** 2
            else:
                noise_estimate = np.std(signal) ** 2 * 0.1
            noise_estimate = max(noise_estimate, 1e-10)
            snr = 10 * np.log10(signal_power / noise_estimate)
            return snr
        except:
            return 0

    def wiener_filter(self, audio_signal):
        """維納濾波器 - 更溫和的處理"""
        try:
            f, t, stft = signal.stft(audio_signal, fs=TARGET_RATE, nperseg=512, noverlap=256)
            
            # 使用更溫和的雜訊估計
            quiet_samples = min(int(TARGET_RATE * 0.05), len(audio_signal) // 8)
            noise_sample = audio_signal[:quiet_samples]
            _, _, noise_stft = signal.stft(noise_sample, fs=TARGET_RATE, nperseg=512, noverlap=256)
            noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
            
            signal_power = np.abs(stft) ** 2
            wiener_gain = signal_power / (signal_power + WIENER_FILTER_STRENGTH * noise_power)
            
            # 限制增益範圍以避免過度處理
            wiener_gain = np.clip(wiener_gain, 0.1, 1.0)
            
            filtered_stft = stft * wiener_gain
            _, filtered_audio = signal.istft(filtered_stft, fs=TARGET_RATE)
            
            return filtered_audio[:len(audio_signal)]
        except:
            return audio_signal

    def smooth_audio(self, audio_signal):
        """音訊平滑處理"""
        try:
            # 移除突然的跳躍
            diff = np.diff(audio_signal)
            threshold = np.std(diff) * 3  # 更寬鬆的閾值
            artifact_indices = np.where(np.abs(diff) > threshold)[0]
            
            for idx in artifact_indices[:20]:  # 限制處理數量
                if 0 < idx < len(audio_signal) - 1:
                    audio_signal[idx] = (audio_signal[idx-1] + audio_signal[idx+1]) / 2
            
            # 輕微平滑
            audio_signal = uniform_filter1d(audio_signal, size=3)
            
            # 輕微低通濾波
            audio_signal = signal.sosfilt(self.lowpass_filter, audio_signal)
            
            return audio_signal
        except:
            return audio_signal

    def dynamic_range_compression(self, audio_signal):
        """動態範圍壓縮"""
        try:
            # 軟限制器
            threshold = 0.8
            ratio = DYNAMIC_RANGE_COMPRESSION
            
            # 計算絕對值
            abs_signal = np.abs(audio_signal)
            
            # 對超過閾值的部分進行壓縮
            mask = abs_signal > threshold
            compressed = np.copy(audio_signal)
            
            if np.any(mask):
                over_threshold = abs_signal[mask]
                compressed_magnitude = threshold + (over_threshold - threshold) * ratio
                compressed[mask] = np.sign(audio_signal[mask]) * compressed_magnitude
            
            return compressed
        except:
            return audio_signal

    def spectral_gating(self, audio):
        """改良的頻譜閘控降噪"""
        try:
            noise_sample_length = max(int(TARGET_RATE * 0.05), 1)
            noise_sample = audio[:noise_sample_length]
            
            return nr.reduce_noise(
                y=audio,
                y_noise=noise_sample,
                sr=TARGET_RATE,
                prop_decrease=NOISE_REDUCE_STRENGTH,
                stationary=False,  # 非穩態雜訊處理
                n_jobs=1
            )
        except:
            return audio

    def enhance_separation(self, separated_signals):
        """增強分離效果 - 改善音質"""
        if not self.enable_noise_reduction:
            return separated_signals
        
        # 處理形狀
        if len(separated_signals.shape) == 3:
            if separated_signals.shape[1] == NUM_SPEAKERS:
                enhanced_signals = torch.zeros_like(separated_signals)
                speaker_dim = 1
                time_dim = 2
            elif separated_signals.shape[2] == NUM_SPEAKERS:
                separated_signals = separated_signals.transpose(1, 2)
                enhanced_signals = torch.zeros_like(separated_signals)
                speaker_dim = 1
                time_dim = 2
            else:
                enhanced_signals = torch.zeros_like(separated_signals)
                speaker_dim = 2
                time_dim = 1
        else:
            if separated_signals.shape[0] == NUM_SPEAKERS:
                separated_signals = separated_signals.unsqueeze(0)
                enhanced_signals = torch.zeros_like(separated_signals)
                speaker_dim = 1
                time_dim = 2
            else:
                separated_signals = separated_signals.unsqueeze(0).transpose(1, 2)
                enhanced_signals = torch.zeros_like(separated_signals)
                speaker_dim = 1
                time_dim = 2
        
        num_speakers = separated_signals.shape[speaker_dim]
        
        for i in range(num_speakers):
            if speaker_dim == 1:
                current_signal = separated_signals[0, i, :].cpu().numpy()
            else:
                current_signal = separated_signals[0, :, i].cpu().numpy()
            
            # 多階段音質改善
            processed_signal = current_signal
            
            # 1. 維納濾波
            signal_snr = self.estimate_snr(current_signal)
            if signal_snr < self.snr_threshold + 3:
                processed_signal = self.wiener_filter(processed_signal)
            
            # 2. 傳統降噪（僅在必要時）
            if signal_snr < self.snr_threshold:
                processed_signal = self.spectral_gating(processed_signal)
            
            # 3. 音訊平滑和修復
            processed_signal = self.smooth_audio(processed_signal)
            
            # 4. 動態範圍壓縮
            processed_signal = self.dynamic_range_compression(processed_signal)
            
            # 5. 最終正規化
            max_val = np.max(np.abs(processed_signal))
            if max_val > 0:
                processed_signal = processed_signal / max_val * 0.95
            
            length = min(len(processed_signal), separated_signals.shape[time_dim])
            
            if speaker_dim == 1:
                enhanced_signals[0, i, :length] = torch.from_numpy(processed_signal[:length]).to(self.device)
            else:
                enhanced_signals[0, :length, i] = torch.from_numpy(processed_signal[:length]).to(self.device)
        
        return enhanced_signals
        
    def set_save_audio_files(self, save: bool) -> None:
        """
        設定是否儲存分離後的音訊檔案
        
        Args:
            save: True 表示儲存音訊檔案，False 表示不儲存
        """
        self.save_audio_files = save
        logger.info(f"音訊檔案儲存設定：{'已啟用' if save else '已停用'}")

    def process_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """處理音訊格式：將原始錄音資料轉換為模型可用的格式"""
        try:
            # 轉換為 float32
            if FORMAT == pyaudio.paInt16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # 能量檢測：過低則略過
            energy = np.mean(np.abs(audio_float))
            if energy < MIN_ENERGY_THRESHOLD:
                logger.debug(f"音訊能量 ({energy:.6f}) 低於閾值 ({MIN_ENERGY_THRESHOLD})")
                return None
            
            # 重塑為正確形狀
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, CHANNELS)

            # 調整形狀以符合模型輸入：[channels, time]
            audio_tensor = torch.from_numpy(audio_float).T.float()

            # 如果是雙聲道而模型只支援單聲道則取平均
            if audio_tensor.shape[0] == 2:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            # 移至 GPU 並重新取樣至 16kHz
            audio_tensor = audio_tensor.to(self.device)
            resampled = self.resampler(audio_tensor)
            
            # 確保形狀正確
            if len(resampled.shape) == 1:
                resampled = resampled.unsqueeze(0)
            
            return resampled
            
        except Exception as e:
            logger.error(f"音訊處理錯誤：{e}")
            self.processing_stats['errors'] += 1
            return None

    def cleanup_futures(self):
        """清理已完成的任務"""
        completed_futures = []
        for future in self.futures:
            if future.done():
                try:
                    future.result()  # 獲取結果以捕獲任何異常
                except Exception as e:
                    logger.error(f"處理任務發生錯誤：{e}")
                    self.processing_stats['errors'] += 1
                completed_futures.append(future)
        
        # 移除已完成的任務
        for future in completed_futures:
            self.futures.remove(future)

    def record_and_process(self, output_dir):
        """錄音並處理音訊的主要方法"""
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        # 儲存原始混合音訊的緩衝區
        mixed_audio_buffer = []
        
        try:
            # 步驟0: 初始化語者識別器
            identifier = SpeakerIdentifier()

            # 步驟1: 初始化錄音裝置
            p = pyaudio.PyAudio()
            
            # 檢查設備可用性
            if DEVICE_INDEX is not None:
                device_info = p.get_device_info_by_index(DEVICE_INDEX)
                logger.info(f"使用音訊設備: {device_info['name']}")
            
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=DEVICE_INDEX
            )
            
            logger.info("開始錄音...")
            
            # 步驟2: 計算緩衝區參數
            samples_per_window = int(WINDOW_SIZE * RATE)
            window_frames = int(samples_per_window / CHUNK)
            overlap_frames = int((OVERLAP * RATE) / CHUNK)
            slide_frames = window_frames - overlap_frames
            
            # 初始化處理變數
            buffer = []
            segment_index = 0
            self.is_recording = True
            last_stats_time = time.time()
            
            # 步驟3: 錄音主循環
            while self.is_recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frame = np.frombuffer(data, dtype=np.float32 if FORMAT == pyaudio.paFloat32 else np.int16)
                    buffer.append(frame)
                    
                    # 限制 mixed_audio_buffer 大小以防止記憶體耗盡
                    mixed_audio_buffer.append(frame.copy())
                    if len(mixed_audio_buffer) > self.max_buffer_size:
                        mixed_audio_buffer.pop(0)
                        
                except IOError as e:
                    logger.warning(f"錄音時發生IO錯誤：{e}")
                    continue
                
                # 步驟4: 當累積足夠資料時進行處理
                if len(buffer) >= window_frames:
                    segment_index += 1
                    audio_data = np.concatenate(buffer[:window_frames])
                    audio_tensor = self.process_audio(audio_data)
                    
                    # 步驟5: 如果音訊有效，啟動語者分離處理
                    if audio_tensor is not None:
                        logger.info(f"處理片段 {segment_index}")
                        future = self.executor.submit(
                            self.separate_and_identify,
                            audio_tensor,
                            output_dir,
                            segment_index
                        )
                        self.futures.append(future)
                        self.processing_stats['segments_processed'] += 1
                    else:
                        self.processing_stats['segments_skipped'] += 1
                    
                    # 步驟6: 移動視窗
                    buffer = buffer[slide_frames:]
                    
                    # 定期清理已完成的任務
                    if segment_index % 10 == 0:
                        self.cleanup_futures()
                    
                    # 每30秒報告一次統計資訊
                    current_time = time.time()
                    if current_time - last_stats_time > 30:
                        self._log_statistics()
                        last_stats_time = current_time
                    
        except Exception as e:
            logger.error(f"錄音過程中發生錯誤：{e}")
        finally:
            # 步驟7: 清理資源
            self._cleanup_resources(p, stream, mixed_audio_buffer, output_dir)

    def _cleanup_resources(self, p, stream, mixed_audio_buffer, output_dir):
        """清理資源"""
        # 停止並關閉音訊流
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
                logger.info("音訊流已關閉")
            except Exception as e:
                logger.error(f"關閉音訊流時發生錯誤：{e}")
        
        if p is not None:
            try:
                p.terminate()
                logger.info("PyAudio 已終止")
            except Exception as e:
                logger.error(f"終止 PyAudio 時發生錯誤：{e}")
        
        # 等待所有處理任務完成
        logger.info("等待處理任務完成...")
        for future in self.futures:
            try:
                future.result(timeout=15.0)
            except Exception as e:
                logger.error(f"處理任務發生錯誤：{e}")
        
        self.executor.shutdown(wait=True)
        logger.info("線程池已關閉")
        
        # 儲存原始混合音訊
        self._save_mixed_audio(mixed_audio_buffer, output_dir)
        
        # 記錄最終統計
        self._log_final_statistics()
        
        # 清理GPU記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("錄音結束，所有資源已清理")

    def _save_mixed_audio(self, mixed_audio_buffer, output_dir):
        """儲存混合音訊"""
        if not mixed_audio_buffer:
            return ""
            
        try:
            mixed_audio = np.concatenate(mixed_audio_buffer)
            mixed_audio = mixed_audio.reshape(-1, CHANNELS)
            
            timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            mixed_output_file = os.path.join(
                output_dir,
                f"mixed_audio_{timestamp}.wav"
            )
            
            mixed_tensor = torch.from_numpy(mixed_audio).T.float()
            torchaudio.save(
                mixed_output_file,
                mixed_tensor,
                RATE
            )
            logger.info(f"已儲存原始混合音訊：{mixed_output_file}")
            return mixed_output_file
            
        except Exception as e:
            logger.error(f"儲存混合音訊時發生錯誤：{e}")
            return ""

    def _log_statistics(self):
        """記錄統計資訊"""
        stats = self.processing_stats
        logger.info(f"統計 - 已處理: {stats['segments_processed']}, "
                   f"已跳過: {stats['segments_skipped']}, "
                   f"錯誤: {stats['errors']}, "
                   f"進行中任務: {len(self.futures)}")

    def _log_final_statistics(self):
        """記錄最終統計資訊"""
        stats = self.processing_stats
        total = stats['segments_processed'] + stats['segments_skipped']
        if total > 0:
            success_rate = (stats['segments_processed'] / total) * 100
            logger.info(f"最終統計 - 總片段: {total}, "
                       f"成功處理: {stats['segments_processed']} ({success_rate:.1f}%), "
                       f"跳過: {stats['segments_skipped']}, "
                       f"錯誤: {stats['errors']}")

    def separate_and_save(self, audio_tensor, output_dir, segment_index):
        """分離並儲存音訊，並回傳 (path, start, end) 列表。"""
        try:
            # 初始化累計時間戳
            current_t0 = getattr(self, "_current_t0", 0.0)
            results = []   # 用來收 (path, start, end)
            seg_duration = audio_tensor.shape[-1] / TARGET_RATE
            
            with torch.no_grad():
                if len(audio_tensor.shape) == 2:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                separated = self.model(audio_tensor)
                
                if self.enable_noise_reduction:
                    enhanced_separated = self.enhance_separation(separated)
                else:
                    enhanced_separated = separated
                
                del separated
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
                
                # 處理輸出格式
                if len(enhanced_separated.shape) == 3:
                    if enhanced_separated.shape[1] == NUM_SPEAKERS:
                        num_speakers = enhanced_separated.shape[1]
                        speaker_dim = 1
                    else:
                        num_speakers = enhanced_separated.shape[2]
                        speaker_dim = 2
                else:
                    num_speakers = 1
                    speaker_dim = 0
                
                saved_count = 0
                start_time = current_t0
                for i in range(min(num_speakers, NUM_SPEAKERS)):
                    try:
                        if speaker_dim == 1:
                            speaker_audio = enhanced_separated[0, i, :].cpu()
                        elif speaker_dim == 2:
                            speaker_audio = enhanced_separated[0, :, i].cpu()
                        else:
                            speaker_audio = enhanced_separated.cpu().squeeze()
                        
                        # 改善的正規化處理
                        if len(speaker_audio.shape) > 1:
                            speaker_audio = speaker_audio.squeeze()
                        
                        # 檢查音訊品質
                        rms = torch.sqrt(torch.mean(speaker_audio ** 2))
                        if rms > 0.01:  # 只保存有意義的音訊
                            # 溫和的正規化
                            max_val = torch.max(torch.abs(speaker_audio))
                            if max_val > 0:
                                # 使用軟限制器
                                normalized = speaker_audio / max_val
                                speaker_audio = torch.tanh(normalized * 0.9) * 0.85
                        
                            final_tensor = speaker_audio.unsqueeze(0)
                            
                            output_file = os.path.join(
                                output_dir,
                                f"speaker{i+1}.wav"
                                # f"speaker{i+1}_{timestamp}_{segment_index}.wav"
                            )
                            
                            torchaudio.save(
                                output_file,
                                final_tensor,
                                TARGET_RATE
                            )

                            results.append((output_file,
                                    start_time,
                                    start_time + seg_duration))
                            self.output_files.append(output_file)

                            saved_count += 1
                    except Exception as e:
                        logger.warning(f"儲存說話者 {i+1} 失敗: {e}")
                
                if saved_count > 0:
                    logger.info(f"片段 {segment_index} 完成，儲存 {saved_count} 個檔案")
                
            # 更新累計時間到下一段
            current_t0 += seg_duration

            # —— 新增：存回屬性，下次呼叫會接續
            self._current_t0 = current_t0

            if not results:
                raise RuntimeError("Speaker separation produced no valid tracks")

            # —— 新增：把結果回傳
            return results

        except Exception as e:
            logger.error(f"處理片段 {segment_index} 失敗: {e}")
            self.processing_stats['errors'] += 1
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def separate_and_identify(self, audio_tensor: torch.Tensor, output_dir: str, segment_index: int) -> None:
        """分離音訊並直接進行語音識別，可選擇是否儲存音訊檔案"""
        try:
            audio_files = []
            audio_streams = []
            
            timestamp_obj = datetime.now()
            timestamp = timestamp_obj.strftime('%Y%m%d-%H_%M_%S')
            
            with torch.no_grad():
                # 確保輸入形狀正確
                if len(audio_tensor.shape) == 2:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                # 步驟1: 語者分離
                separated = self.model(audio_tensor)
                
                if self.enable_noise_reduction:
                    enhanced_separated = self.enhance_separation(separated)
                else:
                    enhanced_separated = separated
                
                del separated
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 處理輸出格式
                if len(enhanced_separated.shape) == 3:
                    if enhanced_separated.shape[1] == NUM_SPEAKERS:
                        num_speakers = enhanced_separated.shape[1]
                        speaker_dim = 1
                    else:
                        num_speakers = enhanced_separated.shape[2]
                        speaker_dim = 2
                else:
                    num_speakers = 1
                    speaker_dim = 0
                
                saved_count = 0
                for i in range(min(num_speakers, NUM_SPEAKERS)):
                    try:
                        if speaker_dim == 1:
                            speaker_audio = enhanced_separated[0, i, :].cpu()
                        elif speaker_dim == 2:
                            speaker_audio = enhanced_separated[0, :, i].cpu()
                        else:
                            speaker_audio = enhanced_separated.cpu().squeeze()
                        
                        # 改善的正規化處理
                        if len(speaker_audio.shape) > 1:
                            speaker_audio = speaker_audio.squeeze()
                        
                        # 檢查音訊品質
                        rms = torch.sqrt(torch.mean(speaker_audio ** 2))
                        if rms > 0.01:  # 只保存有意義的音訊
                            # 溫和的正規化
                            max_val = torch.max(torch.abs(speaker_audio))
                            if max_val > 0:
                                # 使用軟限制器
                                normalized = speaker_audio / max_val
                                speaker_audio = torch.tanh(normalized * 0.9) * 0.85
                        
                            final_audio = speaker_audio.numpy()
                            final_tensor = speaker_audio.unsqueeze(0)
                            
                            # 儲存音訊流資料供直接辨識
                            audio_streams.append({
                                'audio_data': final_audio,
                                'sample_rate': TARGET_RATE,
                                'name': f"speaker{i+1}_{timestamp}_{segment_index}"
                            })
                            
                            # 如果設定要儲存音訊檔案，則額外儲存分離檔案
                            if self.save_audio_files:
                                output_file = os.path.join(
                                    output_dir,
                                    f"speaker{i+1}_{timestamp}_{segment_index}.wav"
                                )
                                
                                torchaudio.save(
                                    output_file,
                                    final_tensor,
                                    TARGET_RATE
                                )
                                
                                audio_files.append(output_file)
                                self.output_files.append(output_file)
                            
                            saved_count += 1
                            
                    except Exception as e:
                        logger.warning(f"處理說話者 {i+1} 失敗: {e}")
                
                if saved_count > 0:
                    logger.info(f"片段 {segment_index} 分離完成，共 {saved_count} 個語者")
            
            # 步驟2: 即時進行語者識別
            logger.info(
                f"片段 {segment_index} 分離完成，開始進行即時語者識別...",
                extra={"simple": True}
            )
            
            try:
                identifier = SpeakerIdentifier()
                
                results = {}
                if audio_streams:
                    results = identifier.process_audio_streams(audio_streams, timestamp_obj)
                
                # 使用簡化格式輸出識別結果
                result_message = []
                for audio_name, (speaker, distance, result) in results.items():
                    result_message.append(f"【{audio_name} → {result}】")
                
                if result_message:
                    message = f"片段 {segment_index} 識別結果:  " + "  ".join(result_message)
                    logger.info(
                        message,
                        extra={"simple": True}
                    )
                    
            except Exception as e:
                logger.error(f"識別片段 {segment_index} 時發生錯誤：{e}")
            
            logger.info(f"片段 {segment_index} 處理完成")
            
        except Exception as e:
            logger.error(f"處理片段 {segment_index} 失敗: {e}")
            self.processing_stats['errors'] += 1
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
        logger.info("準備停止錄音...")

    def get_output_files(self):
        """獲取所有分離後的音檔路徑"""
        return self.output_files


# ================== 語者識別部分 ======================

class SpeakerIdentifier:
    """語者識別類，負責呼叫 v5 版本的語者識別功能，使用單例模式"""
    
    _instance = None
    
    def __new__(cls) -> 'SpeakerIdentifier':
        """實現單例模式，確保全局只有一個實例"""
        if cls._instance is None:
            cls._instance = super(SpeakerIdentifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """初始化語者識別器，使用 v5 版本的 SpeakerIdentifier"""
        # 若已初始化，則跳過
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        try:
            # 初始化 v5 模組 (它也會使用單例模式，避免重複加載模型)
            self.identifier = speaker_id.SpeakerIdentifier()
            
            # 設置詳細度，減少非必要輸出
            self.identifier.set_verbose(False)
            
            logger.info("語者識別器初始化完成")
            self._initialized = True
        except Exception as e:
            logger.error(f"初始化語者識別器時發生錯誤：{e}")
            raise
    
    def process_audio_streams(self, audio_streams: list, timestamp: datetime) -> dict:
        """
        處理多個音訊流並進行語者識別
        
        Args:
            audio_streams: 音訊流資料列表，每個元素包含 'audio_data', 'sample_rate', 'name'
            timestamp: 音訊流的時間戳記物件
            
        Returns:
            dict: 音訊流名稱 -> (語者名稱, 相似度, 識別結果描述)
        """
        results = {}
        
        try:
            for stream in audio_streams:
                audio_data = stream['audio_data']
                sample_rate = stream['sample_rate']
                name = stream['name']
                
                logger.info(f"識別音訊流: {name}")
                
                # 呼叫 v5 版本的語者識別功能，傳入時間戳記
                result = self.identifier.process_audio_stream(
                    audio_data, 
                    sample_rate, 
                    audio_source=name,
                    timestamp=timestamp
                )
                
                if result:
                    speaker_id_, speaker_name, distance = result
                    
                    # 根據距離判斷識別結果
                    if distance == -1:
                        # 距離為 -1 表示新建立的語者
                        result_desc = f"新語者 {speaker_name} \t(已建立新聲紋:{distance:.4f})"
                    elif distance < THRESHOLD_LOW:
                        result_desc = f"語者 {speaker_name} \t(聲音非常相似:{distance:.4f})"
                    elif distance < THRESHOLD_UPDATE:
                        result_desc = f"語者 {speaker_name} \t(已更新聲紋:{distance:.4f})"
                    elif distance < THRESHOLD_NEW:
                        result_desc = f"語者 {speaker_name} \t(新增新的聲紋:{distance:.4f})"
                    else:
                        # 此處不應該執行到，因為距離大於 THRESHOLD_NEW 時應該創建新語者
                        result_desc = f"語者 {speaker_name} \t(判斷不明確):{distance:.4f}"
                    
                    results[name] = (speaker_name, distance, result_desc)
                    # logger.info(f"結果: {result_desc}")
                else:
                    results[name] = (None, -1, "識別失敗")
                    logger.warning("識別失敗")
        except Exception as e:
            logger.error(f"處理音訊流時發生錯誤：{e}")
        
        return results
    
def check_weaviate_connection() -> bool:
    """
    檢查 Weaviate 資料庫連線狀態。

    Returns:
        bool: 若連線成功回傳 True，否則回傳 False。
    """
    try:
        import weaviate  # type: ignore
        client = weaviate.connect_to_local()
        # 檢查是否能存取必要集合
        if not client.is_live():
            logger.error("Weaviate 服務未啟動或無法存取。")
            return False
        if not (client.collections.exists("Speaker") and client.collections.exists("VoicePrint")):
            logger.error("Weaviate 缺少必要集合 (Speaker 或 VoicePrint)。請先執行 create_collections.py。")
            return False
        return True
    except Exception as e:
        logger.error(f"Weaviate 連線失敗：{e}")
        return False
    
def run_realtime(output_dir: str = OUTPUT_DIR) -> str:
    """方便外部呼叫的錄音處理函式"""
    separator = AudioSeparator()
    return separator.record_and_process(output_dir)


def run_offline(file_path: str, output_dir: str = OUTPUT_DIR, save_files: bool = True) -> None:
    """方便外部呼叫的離線音檔處理函式"""
    separator = AudioSeparator()
    separator.set_save_audio_files(save_files)
    separator.process_audio_file(file_path, output_dir)

