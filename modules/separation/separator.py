"""
# 語者分離與識別系統 (Speech Separation and Speaker Identification System)

## 安裝前置條件

1. **Python 環境**：
   - 需要 Python 3.9+ 環境
   - Windows 建議使用虛擬環境 (venv)

2. **安裝依賴套件**：
   ```
   pip install -r requirements.txt
   ```

3. **Weaviate 資料庫設定**：
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
     python weaviate_study/npy_to_weaviate.py
     ```

4. **模型下載**：
   - 首次運行時會自動下載必要的 Speechbrain 預訓練模型
   - 需要網際網路連接

5. **啟動程式**：
   ```
   python test_identify.py
   ```


   
更多詳細資訊請參考：https://github.com/LCY000/ProjectStudy_SpeechRecognition

"""

import os
import numpy as np
import torch
import torchaudio
import pyaudio
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference import SepformerSeparation as separator
from speechbrain.inference import SpeakerRecognition
import noisereduce as nr
# 導入 scipy.signal.resample_poly 進行重採樣
from scipy.signal import resample_poly
# 導入 main_identify_v4_weaviate 模組
from modules.speaker_id import VID_identify_v5 as speaker_id_v4

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
MIN_ENERGY_THRESHOLD = 0.005
NOISE_REDUCE_STRENGTH = 0.1

# 全域參數設定
EMBEDDING_DIR = "embeddingFiles"  # 所有說話者嵌入資料的根目錄
THRESHOLD_LOW = speaker_id_v4.THRESHOLD_LOW     # 過於相似，不更新
THRESHOLD_UPDATE = speaker_id_v4.THRESHOLD_UPDATE # 更新嵌入向量
THRESHOLD_NEW = speaker_id_v4.THRESHOLD_NEW    # 判定為新說話者
# 目前測試可能同一個人0.37以下都有可能，似乎因為分離效果很飄，很吃分離效果

# 輸出目錄
OUTPUT_DIR = "16K-model/Audios-16K-IDTF"
IDENTIFIED_DIR = "16K-model/Identified-Speakers"

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== 語者分離部分 ======================

class AudioSeparator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用設備: {self.device}")
        # 使用16KHz分離模型，分離兩語者
        self.model = separator.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            savedir='pretrained_models/sepformer-whamr16k',
            run_opts={"device": self.device}
        )
        
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=RATE,
            new_freq=TARGET_RATE
        ).to(self.device)
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []
        self.is_recording = False
        self.output_files = []  # 儲存分離後的音檔路徑
        logger.info("AudioSeparator 初始化完成")

    def spectral_gating(self, audio):
        """應用頻譜閘控降噪"""
        noise_sample = audio[:max(int(TARGET_RATE * 0.1), 1)]
        return nr.reduce_noise(
            y=audio,
            y_noise=noise_sample,
            sr=TARGET_RATE,
            prop_decrease=NOISE_REDUCE_STRENGTH,
            n_jobs=-1
        )

    def enhance_separation(self, separated_signals):
        """增強分離效果，僅應用一次降噪以避免過度處理"""
        enhanced_signals = torch.zeros_like(separated_signals)
        
        for i in range(separated_signals.shape[2]):
            current_signal = separated_signals[0, :, i].cpu().numpy()
            denoised_signal = self.spectral_gating(current_signal)
            length = min(len(denoised_signal), separated_signals.shape[1])
            enhanced_signals[0, :length, i] = torch.from_numpy(denoised_signal).to(self.device)
        
        return enhanced_signals

    def process_audio(self, audio_data):
        """處理音訊格式"""
        try:
            # 轉換為 float32
            if FORMAT == pyaudio.paInt16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # 能量檢測：過低則略過
            energy = np.mean(np.abs(audio_float))
            if energy < MIN_ENERGY_THRESHOLD:
                logger.debug(f"音訊能量 ({energy}) 低於閾值 ({MIN_ENERGY_THRESHOLD})")
                return None
            
            # 重塑為正確形狀
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, CHANNELS)

            # 調整形狀以符合模型輸入：[channels, time]
            audio_tensor = torch.from_numpy(audio_float).T.float()

            # 如果是雙聲道而模型只支援單聲道則取平均
            if audio_tensor.shape[0] == 2:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            # 移至 GPU 並重新取樣
            audio_tensor = audio_tensor.to(self.device)
            resampled = self.resampler(audio_tensor)
            
            # 確保形狀正確
            if len(resampled.shape) == 1:
                resampled = resampled.unsqueeze(0)
            
            return resampled
            
        except Exception as e:
            logger.error(f"音訊處理錯誤：{e}")
            return None

    def record_and_process(self, output_dir):
        """錄音並處理"""
        os.makedirs(output_dir, exist_ok=True)
        mixed_audio_buffer = []
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=DEVICE_INDEX
            )
            
            logger.info("開始錄音...")
            
            # 計算緩衝區大小與重疊數據
            samples_per_window = int(WINDOW_SIZE * RATE)
            window_frames = int(samples_per_window / CHUNK)
            overlap_frames = int((OVERLAP * RATE) / CHUNK)
            slide_frames = window_frames - overlap_frames
            
            buffer = []
            segment_index = 0
            self.is_recording = True
            
            while self.is_recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frame = np.frombuffer(data, dtype=np.float32 if FORMAT == pyaudio.paFloat32 else np.int16)
                    buffer.append(frame)
                    mixed_audio_buffer.append(frame.copy())
                except IOError as e:
                    logger.warning(f"錄音時發生IO錯誤：{e}")
                    continue
                
                if len(buffer) >= window_frames:
                    segment_index += 1
                    audio_data = np.concatenate(buffer[:window_frames])
                    audio_tensor = self.process_audio(audio_data)
                    
                    if (audio_tensor is not None):
                        logger.info(f"處理片段 {segment_index}")
                        future = self.executor.submit(
                            self.separate_and_save,
                            audio_tensor,
                            output_dir,
                            segment_index
                        )
                        self.futures.append(future)
                    
                    # 保留重疊部分
                    buffer = buffer[slide_frames:]
                    self.futures = [f for f in self.futures if not f.done()]
                    
        except Exception as e:
            logger.error(f"錄音過程中發生錯誤：{e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            for future in self.futures:
                try:
                    future.result(timeout=10.0)
                except Exception as e:
                    logger.error(f"處理任務發生錯誤：{e}")
            
            self.executor.shutdown(wait=True)
            
            # 儲存原始混合音訊為單獨檔案
            timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            mixed_output_file = ""
            
            if mixed_audio_buffer:
                try:
                    mixed_audio = np.concatenate(mixed_audio_buffer)
                    mixed_audio = mixed_audio.reshape(-1, CHANNELS)
                    
                    mixed_output_file = os.path.join(
                        output_dir,
                        f"mixed_audio_{timestamp}.wav"
                    )
                    
                    mixed_tensor = torch.from_numpy(mixed_audio).T.float()
                    torchaudio.save(
                        mixed_output_file,
                        mixed_tensor,
                        RATE  # 使用原始採樣率 44100Hz 儲存
                    )
                    logger.info(f"已儲存原始混合音訊：{mixed_output_file}")
                except Exception as e:
                    logger.error(f"儲存混合音訊時發生錯誤：{e}")
            
            logger.info("錄音結束，資源已清理")
            return mixed_output_file

    def separate_and_save(self, audio_tensor, output_dir, segment_index):
        """分離並儲存音訊"""
        try:
            # —— 新增：初始化累計時間戳
            current_t0 = getattr(self, "_current_t0", 0.0)
            results = []   # 用來收 (path, start, end)

            with torch.no_grad():
                separated = self.model.separate_batch(audio_tensor)
                enhanced_separated = self.enhance_separation(separated)
                
                timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
                for i in range(enhanced_separated.shape[2]):
                    speaker_audio = enhanced_separated[:, :, i].cpu()
                    
                    # 正規化音量
                    max_val = torch.max(torch.abs(speaker_audio))
                    if max_val > 0:
                        speaker_audio = speaker_audio / max_val * 0.9
                    
                    final_audio = speaker_audio[0].numpy()
                    final_tensor = torch.from_numpy(final_audio).unsqueeze(0)
                    
                    output_file = os.path.join(
                        output_dir,
                        f"speaker{i+1}_{timestamp}_{segment_index}.wav"
                    )
                    
                    torchaudio.save(
                        output_file,
                        final_tensor,
                        TARGET_RATE
                    )
                    
                    # —— 新增：計算這段的長度（秒）
                    duration = final_audio.shape[-1] / TARGET_RATE

                    # —— 新增：把 (路徑, start, end) 收到 results
                    results.append((output_file,
                                    current_t0,
                                    current_t0 + duration))

                    # —— 新增：更新累計時間
                    current_t0 += duration

                    # 如果你還想保留舊有的 output_files
                    self.output_files.append(output_file)
                
            logger.info(f"片段 {segment_index} 處理完成")

            # —— 新增：存回屬性，下次呼叫會接續
            self._current_t0 = current_t0

            # —— 新增：把結果回傳
            return results

        except Exception as e:
            logger.error(f"處理片段 {segment_index} 時發生錯誤：{e}")

    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
        logger.info("準備停止錄音...")

    def get_output_files(self):
        """獲取所有分離後的音檔路徑"""
        return self.output_files


# ================== 語者識別部分 ======================

class SpeakerIdentifier:
    """說話者識別類，負責呼叫 v4 版本的語者識別功能"""
    
    def __init__(self) -> None:
        """初始化說話者識別器，直接使用 v4 版本的 SpeakerIdentifier"""
        try:
            # 直接使用 v4 版本的 SpeakerIdentifier
            self.identifier = speaker_id_v4.SpeakerIdentifier()
            logger.info("語者識別器初始化完成")
        except Exception as e:
            logger.error(f"初始化語者識別器時發生錯誤：{e}")
            raise
    
    def process_audio_files(self, audio_files: list) -> dict:
        """
        處理多個音檔並進行說話者識別
        
        Args:
            audio_files: 音檔路徑列表
            
        Returns:
            dict: 音檔路徑 -> (說話者名稱, 相似度, 識別結果描述)
        """
        results = {}
        
        try:
            for audio_file in audio_files:
                if not os.path.exists(audio_file):
                    logger.warning(f"音檔不存在: {audio_file}")
                    results[audio_file] = (None, -1, "檔案不存在")
                    continue
                
                logger.info(f"識別音檔: {audio_file}")
                
                # 呼叫 v4 版本的語者識別功能
                result = self.identifier.process_audio_file(audio_file)
                
                if result:
                    speaker_id, speaker_name, distance = result
                    
                    # 根據距離判斷識別結果
                    if distance < speaker_id_v4.THRESHOLD_LOW:
                        result_desc = f"已識別為說話者 {speaker_name} (非常相似)"
                    elif distance < speaker_id_v4.THRESHOLD_UPDATE:
                        result_desc = f"已識別為說話者 {speaker_name} (並已更新聲紋)"
                    elif distance < speaker_id_v4.THRESHOLD_NEW:
                        result_desc = f"已識別為說話者 {speaker_name} (相似但未更新)"
                    else:
                        result_desc = f"已識別為新說話者 {speaker_name}"
                    
                    results[audio_file] = (speaker_name, distance, result_desc)
                else:
                    results[audio_file] = (None, -1, "識別失敗")
        except Exception as e:
            logger.error(f"處理音檔時發生錯誤：{e}")
        
        return results