import os
import pyaudio
import logging
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# 從 utils 模組載入配置
from utils.constants import (
    AUDIO_CHUNK_SIZE, AUDIO_PYAUDIO_FORMAT_STR, AUDIO_CHANNELS, 
    AUDIO_RECORDING_RATE, AUDIO_TARGET_RATE, AUDIO_WINDOW_SIZE, 
    AUDIO_OVERLAP, AUDIO_MIN_ENERGY_THRESHOLD, NOISE_REDUCE_STRENGTH, 
    AUDIO_MAX_BUFFER_MINUTES, SNR_THRESHOLD, WIENER_FILTER_STRENGTH,
    HIGH_FREQ_CUTOFF, DYNAMIC_RANGE_COMPRESSION, CONVTASNET_MODEL_NAME,
    NUM_SPEAKERS_SEPARATION
)
from utils.env_config import FORCE_CPU, CUDA_DEVICE_INDEX

# 設定日誌 - 改為 INFO 級別
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 修改模型載入方式
try:
    from asteroid.models import ConvTasNet
    USE_ASTEROID = True
    logger.info("使用 Asteroid 套件載入 ConvTasNet")
except ImportError:
    from transformers import AutoModel
    USE_ASTEROID = False
    logger.info("使用 Transformers 套件載入模型")
import torch
import torchaudio
import noisereduce as nr
import threading
import time
from scipy import signal
from scipy.ndimage import uniform_filter1d

# 設定音訊裝置參數
DEVICE_INDEX = None


class AudioSeparator:
    def __init__(self, enable_noise_reduction=True, snr_threshold=SNR_THRESHOLD):
        # 設備選擇邏輯：優先考慮 FORCE_CPU 設定
        if FORCE_CPU:
            self.device = "cpu"
            logger.info("🔧 FORCE_CPU=true，強制使用 CPU")
        else:
            if torch.cuda.is_available():
                # 檢查指定的CUDA設備是否存在
                if CUDA_DEVICE_INDEX < torch.cuda.device_count():
                    self.device = f"cuda:{CUDA_DEVICE_INDEX}"
                    # 確保設定正確的設備
                    torch.cuda.set_device(CUDA_DEVICE_INDEX)
                    if CUDA_DEVICE_INDEX != 0:
                        logger.info(f"🎯 使用 CUDA 設備索引: {CUDA_DEVICE_INDEX}")
                else:
                    logger.warning(f"⚠️  CUDA 設備索引 {CUDA_DEVICE_INDEX} 不存在，使用 cuda:0")
                    self.device = "cuda:0"
                    torch.cuda.set_device(0)
            else:
                self.device = "cpu"
                logger.info("🖥️  未偵測到 GPU，使用 CPU")
        
        self.enable_noise_reduction = enable_noise_reduction
        self.snr_threshold = snr_threshold
        
        logger.info(f"使用設備: {self.device}")
        logger.info(f"模型: {CONVTASNET_MODEL_NAME}")
        
        # 設計更溫和的低通濾波器
        nyquist = AUDIO_TARGET_RATE // 2
        cutoff = min(HIGH_FREQ_CUTOFF, nyquist - 100)
        self.lowpass_filter = signal.butter(2, cutoff / nyquist, btype='low', output='sos')  # 降低濾波器階數
        
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
                orig_freq=AUDIO_RECORDING_RATE,
                new_freq=AUDIO_TARGET_RATE
            ).to(self.device)
        except Exception as e:
            logger.error(f"重新取樣器初始化失敗: {e}")
            raise
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []
        self.is_recording = False
        self.processing_stats = {
            'segments_processed': 0,
            'segments_skipped': 0,
            'errors': 0
        }
        
        self.max_buffer_size = int(AUDIO_RECORDING_RATE * AUDIO_MAX_BUFFER_MINUTES * 60 / AUDIO_CHUNK_SIZE)
        logger.info("AudioSeparator 初始化完成")

    def _load_model(self):
        """載入 ConvTasNet 模型"""
        if USE_ASTEROID:
            try:
                model = ConvTasNet.from_pretrained(CONVTASNET_MODEL_NAME)
                model = model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                logger.warning(f"Asteroid 載入失敗，嘗試其他方法...")
        
        try:
            model = AutoModel.from_pretrained(
                CONVTASNET_MODEL_NAME,
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
            f, t, stft = signal.stft(audio_signal, fs=AUDIO_TARGET_RATE, nperseg=512, noverlap=256)
            
            # 使用更溫和的雜訊估計
            quiet_samples = min(int(AUDIO_TARGET_RATE * 0.05), len(audio_signal) // 8)
            noise_sample = audio_signal[:quiet_samples]
            _, _, noise_stft = signal.stft(noise_sample, fs=AUDIO_TARGET_RATE, nperseg=512, noverlap=256)
            noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
            
            signal_power = np.abs(stft) ** 2
            wiener_gain = signal_power / (signal_power + WIENER_FILTER_STRENGTH * noise_power)
            
            # 限制增益範圍以避免過度處理
            wiener_gain = np.clip(wiener_gain, 0.1, 1.0)
            
            filtered_stft = stft * wiener_gain
            _, filtered_audio = signal.istft(filtered_stft, fs=AUDIO_TARGET_RATE)
            
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
            noise_sample_length = max(int(AUDIO_TARGET_RATE * 0.05), 1)
            noise_sample = audio[:noise_sample_length]
            
            return nr.reduce_noise(
                y=audio,
                y_noise=noise_sample,
                sr=AUDIO_TARGET_RATE,
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
            if separated_signals.shape[1] == NUM_SPEAKERS_SEPARATION:
                enhanced_signals = torch.zeros_like(separated_signals)
                speaker_dim = 1
                time_dim = 2
            elif separated_signals.shape[2] == NUM_SPEAKERS_SEPARATION:
                separated_signals = separated_signals.transpose(1, 2)
                enhanced_signals = torch.zeros_like(separated_signals)
                speaker_dim = 1
                time_dim = 2
            else:
                enhanced_signals = torch.zeros_like(separated_signals)
                speaker_dim = 2
                time_dim = 1
        else:
            if separated_signals.shape[0] == NUM_SPEAKERS_SEPARATION:
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

    def process_audio(self, audio_data):
        """處理音訊格式"""
        try:
            # 轉換為 float32
            if AUDIO_PYAUDIO_FORMAT_STR == "paInt16":
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # 能量檢測：過低則略過
            energy = np.mean(np.abs(audio_float))
            if energy < AUDIO_MIN_ENERGY_THRESHOLD:
                logger.debug(f"音訊能量 ({energy:.6f}) 低於閾值 ({AUDIO_MIN_ENERGY_THRESHOLD})")
                return None
            
            # 重塑為正確形狀
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, AUDIO_CHANNELS)

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
        """錄音並處理"""
        mixed_audio_buffer = []
        p = None
        stream = None
        
        try:
            p = pyaudio.PyAudio()
            
            # 檢查設備可用性
            if DEVICE_INDEX is not None:
                device_info = p.get_device_info_by_index(DEVICE_INDEX)
                logger.info(f"使用音訊設備: {device_info['name']}")
            
            stream = p.open(
                format=pyaudio.paFloat32,  # 使用實際的 pyaudio 常數
                channels=AUDIO_CHANNELS,
                rate=AUDIO_RECORDING_RATE,
                input=True,
                frames_per_buffer=AUDIO_CHUNK_SIZE,
                input_device_index=DEVICE_INDEX
            )
            
            logger.info("開始錄音")
            
            # 計算緩衝區大小與重疊數據
            samples_per_window = int(AUDIO_WINDOW_SIZE * AUDIO_RECORDING_RATE)
            window_frames = int(samples_per_window / AUDIO_CHUNK_SIZE)
            overlap_frames = int((AUDIO_OVERLAP * AUDIO_RECORDING_RATE) / AUDIO_CHUNK_SIZE)
            slide_frames = window_frames - overlap_frames
            
            buffer = []
            segment_index = 0
            self.is_recording = True
            last_stats_time = time.time()
            
            while self.is_recording:
                try:
                    data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                    frame = np.frombuffer(data, dtype=np.float32 if AUDIO_PYAUDIO_FORMAT_STR == "paFloat32" else np.int16)
                    
                    buffer.append(frame)
                    
                    # 限制 mixed_audio_buffer 大小以防止記憶體耗盡
                    mixed_audio_buffer.append(frame.copy())
                    if len(mixed_audio_buffer) > self.max_buffer_size:
                        mixed_audio_buffer.pop(0)
                        
                except IOError as e:
                    logger.warning(f"錄音時發生IO錯誤：{e}")
                    continue
                
                if len(buffer) >= window_frames:
                    segment_index += 1
                    audio_data = np.concatenate(buffer[:window_frames])
                    audio_tensor = self.process_audio(audio_data)
                    
                    if audio_tensor is not None:
                        logger.info(f"處理片段 {segment_index}")
                        future = self.executor.submit(
                            self.separate_and_save,
                            audio_tensor,
                            output_dir,
                            segment_index
                        )
                        self.futures.append(future)
                        self.processing_stats['segments_processed'] += 1
                    else:
                        self.processing_stats['segments_skipped'] += 1
                    
                    # 保留重疊部分
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
            self._cleanup_resources(p, stream, mixed_audio_buffer)

    def _cleanup_resources(self, p, stream, mixed_audio_buffer):
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
        self._save_mixed_audio(mixed_audio_buffer)
        
        # 記錄最終統計
        self._log_final_statistics()
        
        # 清理GPU記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("錄音結束，所有資源已清理")

    def _save_mixed_audio(self, mixed_audio_buffer):
        """儲存混合音訊"""
        if not mixed_audio_buffer:
            return
            
        try:
            mixed_audio = np.concatenate(mixed_audio_buffer)
            mixed_audio = mixed_audio.reshape(-1, AUDIO_CHANNELS)
            
            timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            mixed_output_file = os.path.join(
                "Audios_Storage",
                f"mixed_audio_{timestamp}.wav"
            )
            
            mixed_tensor = torch.from_numpy(mixed_audio).T.float()
            torchaudio.save(
                mixed_output_file,
                mixed_tensor,
                AUDIO_RECORDING_RATE
            )
            logger.info(f"已儲存原始混合音訊：{mixed_output_file}")
            
        except Exception as e:
            logger.error(f"儲存混合音訊時發生錯誤：{e}")

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
        """分離並儲存音訊"""
        try:
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
                    if enhanced_separated.shape[1] == NUM_SPEAKERS_SEPARATION:
                        num_speakers = enhanced_separated.shape[1]
                        speaker_dim = 1
                    else:
                        num_speakers = enhanced_separated.shape[2]
                        speaker_dim = 2
                else:
                    num_speakers = 1
                    speaker_dim = 0
                
                saved_count = 0
                for i in range(min(num_speakers, NUM_SPEAKERS_SEPARATION)):
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
                                f"speaker{i+1}_{timestamp}_{segment_index}.wav"
                            )
                            
                            torchaudio.save(
                                output_file,
                                final_tensor,
                                AUDIO_TARGET_RATE
                            )
                            saved_count += 1
                    except Exception as e:
                        logger.warning(f"儲存說話者 {i+1} 失敗: {e}")
                
                if saved_count > 0:
                    logger.info(f"片段 {segment_index} 完成，儲存 {saved_count} 個檔案")
                
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


def main():
    """主程式"""
    try:
        logger.info("=== ConvTasNet 3人語音分離系統 ===")
        
        output_dir = "Audios_Storage"
        os.makedirs(output_dir, exist_ok=True)
        
        # 檢查 PyAudio
        try:
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            p.terminate()
            logger.info(f"偵測到 {device_count} 個音訊設備")
        except Exception as e:
            logger.error(f"PyAudio 檢查失敗: {e}")
            return
        
        enable_noise_reduction = True
        snr_threshold = 8  # 降低閾值以提供更多處理
        
        separator_instance = AudioSeparator(
            enable_noise_reduction=enable_noise_reduction,
            snr_threshold=snr_threshold
        )
        
        logger.info("開始錄音...")
        separator_instance.record_and_process(output_dir)
        
    except KeyboardInterrupt:
        logger.info("停止錄音")
        if 'separator_instance' in locals():
            separator_instance.stop_recording()
    except Exception as e:
        logger.error(f"程式錯誤: {e}")
    finally:
        logger.info("程式結束")

if __name__ == "__main__":
    main()