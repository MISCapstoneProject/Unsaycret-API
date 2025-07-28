"""
===============================================================================
Voice_ID
即時語者分離與識別系統 (Real-time Speech Separation and Speaker Identification System)
===============================================================================

版本：v2.2.0
作者：EvanLo62, CYouuu
最後更新：2025-07-09

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
因此本系統使用時可以分離最多三個語者的混合語音。
 
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

# 修復 SVML 錯誤：在導入 PyTorch 之前設定環境變數
os.environ["MKL_DISABLE_FAST_MM"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
from enum import Enum
from sklearn.cluster import DBSCAN # type: ignore
import librosa # type: ignore

# 修改模型載入方式
try:
    from asteroid.models import ConvTasNet
    USE_ASTEROID = True
except ImportError:
    from transformers import AutoModel
    USE_ASTEROID = False

# 導入日誌模組
from utils.logger import get_logger

# 導入配置 (環境變數)
from utils.env_config import (
    AUDIO_RATE, MODELS_BASE_DIR, FORCE_CPU, CUDA_DEVICE_INDEX
)

# 導入常數 (應用程式參數)  
from utils.constants import (
    THRESHOLD_LOW, THRESHOLD_UPDATE, THRESHOLD_NEW,
    DEFAULT_SEPARATION_MODEL, SPEECHBRAIN_SEPARATOR_MODEL,
    AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE, AUDIO_CHANNELS, 
    AUDIO_WINDOW_SIZE, AUDIO_OVERLAP, AUDIO_MIN_ENERGY_THRESHOLD, 
    AUDIO_MAX_BUFFER_MINUTES, API_MAX_WORKERS, AUDIO_TARGET_RATE
)

# 導入 main_identify_v5 模組
from modules.identification import VID_identify_v5 as speaker_id

# 新增模型類型枚舉
class SeparationModel(Enum):
    CONVTASNET_3SPEAKER = "convtasnet_3speaker"  # ConvTasNet 3人模型
    SEPFORMER_2SPEAKER = "sepformer_2speaker"    # SepFormer 2人模型

# 模型配置
MODEL_CONFIGS = {
    SeparationModel.CONVTASNET_3SPEAKER: {
        "model_name": "JorisCos/ConvTasNet_Libri3Mix_sepnoisy_16k",
        "num_speakers": 3,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "use_speechbrain": False
    },
    SeparationModel.SEPFORMER_2SPEAKER: {
        "model_name": SPEECHBRAIN_SEPARATOR_MODEL,
        "num_speakers": 2,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "use_speechbrain": True
    }
}

# 基本錄音參數（從配置讀取）
CHUNK = AUDIO_CHUNK_SIZE
FORMAT = pyaudio.paFloat32
CHANNELS = AUDIO_CHANNELS
RATE = AUDIO_RATE
TARGET_RATE = AUDIO_TARGET_RATE
WINDOW_SIZE = AUDIO_WINDOW_SIZE
OVERLAP = AUDIO_OVERLAP
DEVICE_INDEX = 2

# 處理參數（從配置讀取）
MIN_ENERGY_THRESHOLD = AUDIO_MIN_ENERGY_THRESHOLD
MAX_BUFFER_MINUTES = AUDIO_MAX_BUFFER_MINUTES

# 音訊處理參數
MIN_ENERGY_THRESHOLD = 0.001
NOISE_REDUCE_STRENGTH = 0.05  # 降低降噪強度以保持音質
MAX_BUFFER_MINUTES = 5
SNR_THRESHOLD = 8  # 降低 SNR 閾值

# 音訊品質改善參數
WIENER_FILTER_STRENGTH = 0.01  # 更溫和的維納濾波
HIGH_FREQ_CUTOFF = 7500  # 提高高頻截止點
DYNAMIC_RANGE_COMPRESSION = 0.7  # 動態範圍壓縮

# ConvTasNet 模型參數 (使用常數配置)
DEFAULT_MODEL = DEFAULT_SEPARATION_MODEL
# 修正 DEFAULT_MODEL 的賦值
if DEFAULT_SEPARATION_MODEL == "sepformer_2speaker":
    DEFAULT_MODEL = SeparationModel.SEPFORMER_2SPEAKER
elif DEFAULT_SEPARATION_MODEL == "convtasnet_3speaker":
    DEFAULT_MODEL = SeparationModel.CONVTASNET_3SPEAKER
else:
    DEFAULT_MODEL = SeparationModel.SEPFORMER_2SPEAKER  # 預設值

MODEL_NAME = MODEL_CONFIGS[DEFAULT_MODEL]["model_name"]
NUM_SPEAKERS = MODEL_CONFIGS[DEFAULT_MODEL]["num_speakers"]

# 新增: 便利的模型選擇函式
def set_default_model(model_type: SeparationModel):
    """設定預設模型類型"""
    global DEFAULT_MODEL, MODEL_NAME, NUM_SPEAKERS
    DEFAULT_MODEL = model_type
    MODEL_NAME = MODEL_CONFIGS[model_type]["model_name"]
    NUM_SPEAKERS = MODEL_CONFIGS[model_type]["num_speakers"]
    logger.info(f"預設模型已設定為: {model_type.value}")

def get_available_models():
    """取得可用的模型列表"""
    return {
        "convtasnet_3speaker": "ConvTasNet 3人語者分離模型",
        "sepformer_2speaker": "SepFormer 2人語者分離模型"
    }

def create_separator(model_name: str = None, **kwargs):
    """
    建立 AudioSeparator 實例的便利函式
    
    Args:
        model_name: 模型名稱 ("convtasnet_3speaker" 或 "sepformer_2speaker")
        **kwargs: 其他參數傳遞給 AudioSeparator
    
    Returns:
        AudioSeparator 實例
    """
    if model_name:
        if model_name == "convtasnet_3speaker":
            model_type = SeparationModel.CONVTASNET_3SPEAKER
        elif model_name == "sepformer_2speaker":
            model_type = SeparationModel.SEPFORMER_2SPEAKER
        else:
            raise ValueError(f"不支援的模型: {model_name}。可用模型: {list(get_available_models().keys())}")
    else:
        model_type = DEFAULT_MODEL
    
    return AudioSeparator(model_type=model_type, **kwargs)

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
    def __init__(self, model_type: SeparationModel = DEFAULT_MODEL, enable_noise_reduction=True, snr_threshold=SNR_THRESHOLD):
        # 設備選擇邏輯：優先考慮 FORCE_CPU 設定
        if FORCE_CPU:
            self.device = "cpu"
            logger.info("🔧 FORCE_CPU=true，強制使用 CPU 運算")
        else:
            if torch.cuda.is_available():
                # 檢查指定的 CUDA 設備是否存在
                if CUDA_DEVICE_INDEX < torch.cuda.device_count():
                    self.device = f"cuda:{CUDA_DEVICE_INDEX}"
                    # 確保設定正確的設備
                    torch.cuda.set_device(CUDA_DEVICE_INDEX)
                    if CUDA_DEVICE_INDEX != 0:
                        logger.info(f"🎯 使用指定的 CUDA 設備: {CUDA_DEVICE_INDEX}")
                else:
                    logger.warning(f"⚠️  指定的 CUDA 設備索引 {CUDA_DEVICE_INDEX} 不存在，改用 cuda:0")
                    self.device = "cuda:0"
                    torch.cuda.set_device(0)
            else:
                self.device = "cpu"
                logger.info("🖥️  未偵測到 GPU 設備，使用 CPU 運算")
                
        self.model_type = model_type
        self.model_config = MODEL_CONFIGS[model_type]
        self.num_speakers = self.model_config["num_speakers"]
        self.enable_noise_reduction = enable_noise_reduction
        self.snr_threshold = snr_threshold
        
        # 語者偵測相關參數 - 進一步降低為更靈敏的設定
        self.vad_threshold = 0.1  # 進一步降低語音活動檢測閾值（原0.08）
        self.speaker_energy_threshold = 0.1  # 進一步降低說話者能量閾值（原0.15）
        self.silence_threshold = 0.002  # 進一步降低靜音檢測閾值（原0.002）
        self.min_speech_duration = 0.8  # 進一步降低最小語音持續時間（原0.3秒）

        logger.info(f"使用設備: {self.device}")
        logger.info(f"模型類型: {model_type.value}")
        logger.info(f"載入模型: {self.model_config['model_name']}")
        logger.info(f"支援語者數量: {self.num_speakers}")
        
        # 設計更溫和的低通濾波器
        nyquist = TARGET_RATE // 2
        cutoff = min(HIGH_FREQ_CUTOFF, nyquist - 100)
        self.lowpass_filter = signal.butter(2, cutoff / nyquist, btype='low', output='sos')
        
        try:
            logger.info("正在載入模型...")
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
        
        self.executor = ThreadPoolExecutor(max_workers=API_MAX_WORKERS)
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
        """載入語者分離模型"""
        model_name = self.model_config["model_name"]
        
        if self.model_config["use_speechbrain"]:
            # 使用 SpeechBrain SepFormer 模型
            try:
                logger.info(f"載入 SpeechBrain 模型: {model_name}")
                
                # 檢查本地模型目錄是否包含無效的符號連結
                local_model_path = os.path.abspath(f"models/{self.model_type.value}")
                if os.path.exists(local_model_path):
                    logger.info(f"檢查本地模型路徑: {local_model_path}")
                    
                    # 檢查是否有無效的符號連結 (Windows JUNCTION 指向 Linux 路徑)
                    hyperparams_file = os.path.join(local_model_path, "hyperparams.yaml")
                    if os.path.exists(hyperparams_file):
                        try:
                            # 測試檔案讀取權限
                            with open(hyperparams_file, 'r', encoding='utf-8') as f:
                                content = f.read(100)  # 讀取前100個字符來測試
                            logger.info("本地模型檔案讀取正常")
                        except (PermissionError, OSError, UnicodeDecodeError) as e:
                            logger.warning(f"本地模型檔案無法讀取: {e}")
                            logger.info("偵測到無效的符號連結，準備重新下載模型...")
                            
                            # 刪除包含無效符號連結的目錄
                            try:
                                import shutil
                                shutil.rmtree(local_model_path, ignore_errors=True)
                                logger.info(f"已刪除無效的模型目錄: {local_model_path}")
                            except Exception as rm_error:
                                logger.warning(f"刪除模型目錄時發生錯誤: {rm_error}")
                
                # 嘗試載入模型 (如果本地檔案無效，SpeechBrain 會自動重新下載)
                model = separator.from_hparams(
                    source=model_name,
                    savedir=os.path.abspath(f"models/{self.model_type.value}"),
                    run_opts={"device": self.device}
                )
                logger.info("SpeechBrain 模型載入成功")
                return model
                
            except Exception as e:
                logger.error(f"SpeechBrain 模型載入失敗: {e}")
                
                # 最後嘗試：強制重新下載
                try:
                    logger.info("嘗試強制重新下載模型...")
                    import shutil
                    local_model_path = os.path.abspath(f"models/{self.model_type.value}")
                    if os.path.exists(local_model_path):
                        shutil.rmtree(local_model_path, ignore_errors=True)
                        logger.info("已清除本地模型快取")
                    
                    model = separator.from_hparams(
                        source=model_name,
                        savedir=os.path.abspath(f"models/{self.model_type.value}"),
                        run_opts={"device": self.device}
                    )
                    logger.info("強制重新下載後，模型載入成功")
                    return model
                    
                except Exception as final_error:
                    logger.error(f"所有載入嘗試均失敗: {final_error}")
                    raise Exception(f"模型載入完全失敗。請檢查網路連線和模型可用性。原始錯誤: {e}")
        else:
            # 使用 ConvTasNet 模型（原有邏輯）
            if USE_ASTEROID:
                try:
                    model = ConvTasNet.from_pretrained(model_name)
                    model = model.to(self.device)
                    model.eval()
                    return model
                except Exception as e:
                    logger.warning(f"Asteroid 載入失敗，嘗試其他載入方式...")
            
            try:
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                model = model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                logger.info("嘗試使用 torch.hub 載入...")
                return self._manual_load_model()

    def _manual_load_model(self):
        """手動載入 ConvTasNet 模型"""
        try:
            model_dir = f"models/{self.model_type.value}"
            os.makedirs(model_dir, exist_ok=True)
            model = torch.hub.load('JorisCos/ConvTasNet', 'ConvTasNet_Libri3Mix_sepnoisy_16k', pretrained=True)
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"所有載入方式均失敗: {e}")
            raise

    def _test_model(self):
        """測試模型"""
        try:
            with torch.no_grad():
                if self.model_config["use_speechbrain"]:
                    # SpeechBrain SepFormer 模型測試
                    # SepFormer 期望輸入格式為 [batch, samples]
                    test_audio = torch.randn(1, AUDIO_SAMPLE_RATE).to(self.device)
                    logger.debug(f"SpeechBrain 測試音訊形狀: {test_audio.shape}")
                    output = self.model.separate_batch(test_audio)
                else:
                    # ConvTasNet 模型測試
                    # ConvTasNet 期望輸入格式為 [batch, channels, samples]
                    test_audio = torch.randn(1, 1, AUDIO_SAMPLE_RATE).to(self.device)
                    logger.debug(f"ConvTasNet 測試音訊形狀: {test_audio.shape}")
                    output = self.model(test_audio)
                    
            logger.info("模型測試通過")
            logger.debug(f"輸出形狀: {output.shape if hasattr(output, 'shape') else type(output)}")
            
        except Exception as e:
            logger.error(f"模型測試失敗: {e}")
            logger.error(f"測試音訊形狀: {test_audio.shape if 'test_audio' in locals() else 'N/A'}")
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
            # 移除突然的跳疊
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
        
        # 處理形狀 - 根據不同模型調整
        if self.model_config["use_speechbrain"]:
            # SpeechBrain 模型輸出格式處理
            if len(separated_signals.shape) == 3:
                # 格式通常為 [batch, time, speakers]
                enhanced_signals = torch.zeros_like(separated_signals)
                speaker_dim = 2
                time_dim = 1
            else:
                separated_signals = separated_signals.unsqueeze(0)
                enhanced_signals = torch.zeros_like(separated_signals)
                speaker_dim = 2
                time_dim = 1
        else:
            # ConvTasNet 模型輸出格式處理（原有邏輯）
            if len(separated_signals.shape) == 3:
                if separated_signals.shape[1] == self.num_speakers:
                    enhanced_signals = torch.zeros_like(separated_signals)
                    speaker_dim = 1
                    time_dim = 2
                elif separated_signals.shape[2] == self.num_speakers:
                    separated_signals = separated_signals.transpose(1, 2)
                    enhanced_signals = torch.zeros_like(separated_signals)
                    speaker_dim = 1
                    time_dim = 2
                else:
                    enhanced_signals = torch.zeros_like(separated_signals)
                    speaker_dim = 2
                    time_dim = 1
            else:
                if separated_signals.shape[0] == self.num_speakers:
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
        
        for i in range(min(num_speakers, self.num_speakers)):
            if speaker_dim == 1:
                current_signal = separated_signals[0, i, :].cpu().numpy()
            elif speaker_dim == 2:
                current_signal = separated_signals[0, :, i].cpu().numpy()
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
            elif speaker_dim == 2:
                enhanced_signals[0, :length, i] = torch.from_numpy(processed_signal[:length]).to(self.device)
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
                            self.separate_and_save,
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

    def detect_speaker_count(self, audio_tensor: torch.Tensor) -> int:
        """
        偵測音訊中的說話者數量
        
        Args:
            audio_tensor: 輸入音訊張量 [channels, samples] 或 [batch, channels, samples]
            
        Returns:
            int: 偵測到的說話者數量
        """
        try:
            # 確保音訊格式正確
            if len(audio_tensor.shape) == 3:
                audio_data = audio_tensor[0, 0, :].cpu().numpy()
            elif len(audio_tensor.shape) == 2:
                audio_data = audio_tensor[0, :].cpu().numpy()
            else:
                audio_data = audio_tensor.cpu().numpy()
            
            # 0. 基本靜音檢測 - 進一步放寬條件
            audio_rms = np.sqrt(np.mean(audio_data ** 2))
            audio_max = np.max(np.abs(audio_data))
            audio_std = np.std(audio_data)
            
            logger.info(f"音訊統計 - RMS: {audio_rms:.6f}, Max: {audio_max:.6f}, STD: {audio_std:.6f}, 靜音閾值: {self.silence_threshold:.6f}")
            
            # 更寬鬆的靜音檢測：只要有一個指標超過閾值就認為可能有語音
            is_silent = (audio_rms < self.silence_threshold and 
                        audio_max < self.silence_threshold * 4 and 
                        audio_std < self.silence_threshold * 1.5)
            
            if is_silent:
                logger.info(f"判定為靜音 - 所有指標都低於閾值")
                return 0
            
            logger.info(f"通過靜音檢測，進行語音活動偵測...")
            
            # 1. 簡化的語音活動檢測 (VAD)
            frame_length = int(TARGET_RATE * 0.025)  # 25ms 幀
            hop_length = int(TARGET_RATE * 0.010)   # 10ms 跳躍
            
            # 計算短時能量
            frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames ** 2, axis=0)
            
            # 正規化能量
            if np.max(energy) > 0:
                energy = energy / np.max(energy)
            
            # 簡化的VAD：主要基於能量檢測
            energy_active = energy > self.vad_threshold
            
            # 額外的過零率檢測作為輔助（但不是必需的）
            try:
                zero_crossings = []
                for i in range(min(frames.shape[1], 100)):  # 限制處理數量以提高速度
                    frame = frames[:, i]
                    zcr = np.sum(np.diff(np.sign(frame)) != 0) / len(frame)
                    zero_crossings.append(zcr)
                zero_crossings = np.array(zero_crossings)
                
                # 如果有合理的過零率，結合使用；否則只用能量
                if len(zero_crossings) > 0:
                    zcr_active = (zero_crossings > 0.005) & (zero_crossings < 0.8)  # 非常寬鬆的範圍
                    if np.sum(zcr_active) > len(zcr_active) * 0.05:  # 如果至少5%的幀有合理過零率
                        combined_active = energy_active[:len(zcr_active)] & zcr_active
                        if np.sum(combined_active) > np.sum(energy_active) * 0.3:  # 如果結合檢測結果不會過度降低
                            vad_frames = np.zeros_like(energy_active, dtype=bool)
                            vad_frames[:len(combined_active)] = combined_active
                        else:
                            vad_frames = energy_active
                    else:
                        vad_frames = energy_active
                else:
                    vad_frames = energy_active
            except Exception as e:
                logger.debug(f"過零率檢測失敗，使用純能量檢測: {e}")
                vad_frames = energy_active
            
            logger.info(f"VAD結果 - 總幀數: {len(vad_frames)}, 活動幀數: {np.sum(vad_frames)}, 比例: {np.sum(vad_frames)/len(vad_frames):.3f}")
            
            # 2. 更寬鬆的語音區段檢查
            if np.any(vad_frames):
                total_speech_ratio = np.sum(vad_frames) / len(vad_frames)
                
                # 大幅降低語音活動比例要求
                if total_speech_ratio < 0.02:  # 只要2%的幀有活動就可能是語音
                    logger.info(f"語音活動比例太低: {total_speech_ratio:.3f}")
                    return 0
                
                # 檢查連續區段（但要求更寬鬆）
                diff = np.diff(np.concatenate(([False], vad_frames, [False])).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                if len(starts) > 0 and len(ends) > 0:
                    speech_durations = (ends - starts) * hop_length / TARGET_RATE
                    max_duration = np.max(speech_durations) if len(speech_durations) > 0 else 0
                    
                    logger.info(f"語音區段分析 - 區段數: {len(starts)}, 最長持續時間: {max_duration:.2f}秒, 最小要求: {self.min_speech_duration:.2f}秒")
                    
                    # 如果有任何區段超過最小要求，或者總活動比例足夠
                    if max_duration >= self.min_speech_duration or total_speech_ratio > 0.05:
                        logger.info("通過語音區段檢查")
                    else:
                        logger.info(f"語音區段太短且活動比例不足")
                        return 0
                else:
                    # 如果無法計算區段，檢查總體活動
                    if total_speech_ratio < 0.03:
                        logger.info(f"無法計算區段且活動比例不足: {total_speech_ratio:.3f}")
                        return 0
            else:
                logger.info("未檢測到任何語音活動")
                return 0
            
            # 3. 如果通過了所有檢查，嘗試進行說話者數量估計
            try:
                # 簡化的MFCC分析
                mfccs = librosa.feature.mfcc(
                    y=audio_data, 
                    sr=TARGET_RATE,
                    n_mfcc=13,
                    hop_length=hop_length,
                    n_fft=frame_length*2
                )
                
                # 確保維度匹配
                min_frames = min(mfccs.shape[1], len(vad_frames))
                mfccs = mfccs[:, :min_frames]
                vad_frames = vad_frames[:min_frames]
                
                if np.any(vad_frames):
                    active_mfccs = mfccs[:, vad_frames]
                    
                    if active_mfccs.shape[1] < 5:  # 進一步降低要求
                        logger.info(f"有效語音幀數很少 ({active_mfccs.shape[1]})，直接判定為單一語者")
                        return 1
                    
                    # 簡化的說話者數量估計
                    features = active_mfccs.T
                    audio_duration = len(audio_data) / TARGET_RATE
                    
                    # 對於短音訊，直接判定為單一語者
                    if audio_duration < 3.0 or features.shape[0] < 20:
                        logger.info(f"短音訊 ({audio_duration:.2f}秒) 或特徵少 ({features.shape[0]})，判定為單一語者")
                        return 1
                    
                    # 嘗試聚類分析（但如果失敗就回傳1）
                    try:
                        clustering = DBSCAN(
                            eps=1.2,  # 更大的聚類半徑
                            min_samples=max(3, int(features.shape[0] * 0.1))
                        ).fit(features)
                        
                        unique_labels = set(clustering.labels_)
                        if -1 in unique_labels:
                            unique_labels.remove(-1)
                        
                        detected_speakers = len(unique_labels)
                        
                        if detected_speakers == 0:
                            detected_speakers = 1  # 後備方案
                        
                        logger.info(f"聚類分析結果: {detected_speakers} 位說話者")
                        
                    except Exception as e:
                        logger.info(f"聚類分析失敗，預設為單一語者: {e}")
                        detected_speakers = 1
                    
                else:
                    detected_speakers = 1
                    
            except Exception as e:
                logger.info(f"MFCC分析失敗，預設為單一語者: {e}")
                detected_speakers = 1
            
            # 4. 最終限制和驗證
            detected_speakers = min(detected_speakers, self.num_speakers)
            detected_speakers = max(detected_speakers, 1)  # 既然通過了語音檢測，至少是1個說話者
            
            logger.info(f"最終偵測結果: {detected_speakers} 位說話者")
            return detected_speakers
            
        except Exception as e:
            logger.warning(f"語者數量偵測失敗，使用後備方案: {e}")
            return self._fallback_speaker_detection(audio_data if 'audio_data' in locals() else audio_tensor.cpu().numpy())

    def _fallback_speaker_detection(self, audio_data: np.ndarray) -> int:
        """
        後備的語者數量偵測方法，基於簡化的能量分析
        
        Args:
            audio_data: 音訊數據
            
        Returns:
            int: 估計的說話者數量
        """
        try:
            # 非常寬鬆的靜音檢測
            audio_rms = np.sqrt(np.mean(audio_data ** 2))
            audio_max = np.max(np.abs(audio_data))
            
            logger.info(f"後備檢測 - RMS: {audio_rms:.6f}, Max: {audio_max:.6f}")
            
            # 只要有一個指標超過很低的閾值就認為有語音
            if audio_rms < self.silence_threshold * 0.3 and audio_max < self.silence_threshold:
                logger.info("後備檢測：判定為靜音")
                return 0
            
            # 如果通過靜音檢測，至少回傳1個說話者
            logger.info("後備檢測：判定為有語音活動")
            return 1
            
        except Exception as e:
            logger.warning(f"後備語者偵測也失敗: {e}")
            # 終極後備方案：只要音訊不是完全靜音就認為有1個說話者
            try:
                if np.any(np.abs(audio_data) > 0.0001):  # 極低的閾值
                    logger.info("終極後備：偵測到非零音訊")
                    return 1
                else:
                    logger.info("終極後備：音訊完全靜音")
                    return 0
            except:
                logger.info("所有檢測都失敗，預設回傳1")
                return 1  # 最保險的選擇

    def separate_and_save(self, audio_tensor, output_dir, segment_index):
        """分離並儲存音訊，並回傳 (path, start, end) 列表。"""
        try:
            # 新增：在分離前先偵測說話者數量
            detected_speakers = self.detect_speaker_count(audio_tensor)
            logger.info(f"片段 {segment_index} - 偵測到 {detected_speakers} 位說話者")
            
            # 如果沒有偵測到說話者，跳過處理
            if detected_speakers == 0:
                logger.info(f"片段 {segment_index} - 未偵測到說話者，跳過處理")
                return []
            
            # 初始化累計時間戳
            current_t0 = getattr(self, "_current_t0", 0.0)
            results = []   # 用來收 (path, start, end)
            seg_duration = audio_tensor.shape[-1] / TARGET_RATE
            
            with torch.no_grad():
                
                # 根據模型類型選擇不同的分離方法
                # 在分離方法中，確保 SpeechBrain 模型得到正確格式
                if self.model_config["use_speechbrain"]:
                    # 確保輸入是 [batch, samples] 格式
                    if len(audio_tensor.shape) == 3:
                        # 如果是 [batch, channels, samples]，需要去掉 channels 維度
                        if audio_tensor.shape[1] == 1:
                            audio_tensor = audio_tensor.squeeze(1)  # 變成 [batch, samples]
                    separated = self.model.separate_batch(audio_tensor)
                else:
                    # ConvTasNet 需要 [batch, channels, samples] 格式
                    if len(audio_tensor.shape) == 2:
                        audio_tensor = audio_tensor.unsqueeze(0)  # 加上 batch 維度
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
                if self.model_config["use_speechbrain"]:
                    # SpeechBrain 模型輸出處理
                    if len(enhanced_separated.shape) == 3:
                        num_speakers = enhanced_separated.shape[2]
                        speaker_dim = 2
                    else:
                        num_speakers = 1
                        speaker_dim = 0
                else:
                    # ConvTasNet 模型輸出處理（原有邏輯）
                    if len(enhanced_separated.shape) == 3:
                        if enhanced_separated.shape[1] == self.num_speakers:
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
                
                # 根據偵測到的說話者數量限制輸出
                effective_speakers = min(detected_speakers, num_speakers, self.num_speakers)
                
                for i in range(effective_speakers):
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
                        logger.warning(f"儲存語者 {i+1} 失敗: {e}")
                
                if saved_count > 0:
                    logger.info(f"片段 {segment_index} 完成，實際儲存 {saved_count}/{effective_speakers} 個檔案")
                
            # 更新累計時間到下一段
            current_t0 += seg_duration
            self._current_t0 = current_t0

            if not results:
                raise RuntimeError("Speaker separation produced no valid tracks")

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
            # 新增：在分離前先偵測說話者數量
            detected_speakers = self.detect_speaker_count(audio_tensor)
            logger.info(f"片段 {segment_index} - 偵測到 {detected_speakers} 位說話者")
            
            # 如果沒有偵測到說話者，跳過處理
            if detected_speakers == 0:
                logger.info(f"片段 {segment_index} - 未偵測到說話者，跳過處理")
                return []
            
            audio_files = []
            audio_streams = []
            
            timestamp_obj = datetime.now()
            timestamp = timestamp_obj.strftime('%Y%m%d-%H_%M_%S')
            
            with torch.no_grad():
                # 確保輸入形狀正確
                if len(audio_tensor.shape) == 2:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                # 步驟1: 語者分離 - 根據模型類型選擇不同方法
                if self.model_config["use_speechbrain"]:
                    separated = self.model.separate_batch(audio_tensor)
                else:
                    separated = self.model(audio_tensor)
                
                if self.enable_noise_reduction:
                    enhanced_separated = self.enhance_separation(separated)
                else:
                    enhanced_separated = separated
                
                del separated
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 處理輸出格式
                if self.model_config["use_speechbrain"]:
                    # SpeechBrain 模型輸出處理
                    if len(enhanced_separated.shape) == 3:
                        num_speakers = enhanced_separated.shape[2]
                        speaker_dim = 2
                    else:
                        num_speakers = 1
                        speaker_dim = 0
                else:
                    # ConvTasNet 模型輸出處理（原有邏輯）
                    if len(enhanced_separated.shape) == 3:
                        if enhanced_separated.shape[1] == self.num_speakers:
                            num_speakers = enhanced_separated.shape[1]
                            speaker_dim = 1
                        else:
                            num_speakers = enhanced_separated.shape[2]
                            speaker_dim = 2
                    else:
                        num_speakers = 1
                        speaker_dim = 0
                
                saved_count = 0
                for i in range(min(num_speakers, self.num_speakers)):
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
                        if rms > 0.01:  # 只儲存有意義的音訊
                            # 溫和的正規化
                            max_val = torch.max(torch.abs(speaker_audio))
                            if max_val > 0:
                                # 使用軟限制器
                                normalized = speaker_audio / max_val
                                speaker_audio = torch.tanh(normalized * 0.9) * 0.85
                        
                            final_audio = speaker_audio.numpy()
                            final_tensor = speaker_audio.unsqueeze(0)
                            
                            # 儲存音訊串流資料供直接辨識使用
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
                        logger.warning(f"處理語者 {i+1} 時失敗: {e}")
                
                if saved_count > 0:
                    logger.info(f"片段 {segment_index} 分離完成，共處理 {saved_count} 個語者")
            
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
    
def run_realtime(output_dir: str = OUTPUT_DIR, model_type: SeparationModel = None, model_name: str = None) -> str:
    """方便外部呼叫的錄音處理函式，支援模型選擇"""
    if model_name:
        separator = create_separator(model_name)
    elif model_type:
        separator = AudioSeparator(model_type=model_type)
    else:
        separator = AudioSeparator(model_type=DEFAULT_MODEL)
    return separator.record_and_process(output_dir)


def run_offline(file_path: str, output_dir: str = OUTPUT_DIR, save_files: bool = True, 
                model_type: SeparationModel = None, model_name: str = None) -> None:
    """方便外部呼叫的離線音檔處理函式，支援模型選擇"""
    if model_name:
        separator = create_separator(model_name)
    elif model_type:
        separator = AudioSeparator(model_type=model_type)
    else:
        separator = AudioSeparator(model_type=DEFAULT_MODEL)
    separator.set_save_audio_files(save_files)
    separator.process_audio_file(file_path, output_dir)

# if __name__ == '__main__':
#     run_realtime(output_dir=OUTPUT_DIR, model_type=SeparationModel.SEPFORMER_2SPEAKER)