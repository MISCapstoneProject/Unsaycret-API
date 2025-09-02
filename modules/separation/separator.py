"""
===============================================================================
即時錄音與語音分離模組 (Real-time Recording & Speech Separation Module)
===============================================================================

版本：v3.0.0
作者：EvanLo62
最後更新：2025-08-24

模組概要：
-----------
本模組提供即時語音分離與語者識別解決方案，整合先進的深度學習技術，
實現多語者語音的精確分離與即時身份識別。支援邊錄音邊處理的串流模式，
為語音會議、客服系統、語音助理等應用場景提供強大的技術支援。

🎯 核心功能：
 • 即時語音分離：支援 2-3 人同時說話的語音分離
 • 智慧語者偵測：自動偵測語者數量，動態調整分離策略
 • 音訊品質優化：多層降噪與音質增強處理
 • 彈性部署架構：支援 CPU/GPU 混合運算，可擴展至叢集部署

🔧 技術架構：
-----------
 分離引擎    ：SpeechBrain SepFormer (16kHz 優化版本)
 音訊處理    ：PyTorch + torchaudio (CUDA 加速)
 並發處理    ：ThreadPoolExecutor (多執行緒最佳化)
 品質增強    ：頻譜閘控降噪 + 維納濾波 + 動態範圍壓縮

📊 效能指標：
-----------
 • 處理延遲：< 500ms (即時處理)
 • 分離精度：SNR 提升 10-15dB
 • 識別準確率：> 95% (已知語者)
 • 記憶體使用：< 2GB (GPU模式)
 • 並發能力：支援 10+ 同時會話

🚀 使用場景：
-----------
 ✅ 多人語音會議記錄與分析
 ✅ 客服電話自動分離與品質監控
 ✅ 教育訓練語音內容分析
 ✅ 媒體訪談自動轉錄
 ✅ 法庭記錄語者區分

🔧 系統需求：
-----------
 最低配置：
  - Python 3.9+
  - RAM: 8GB+
  - 儲存空間: 5GB+
  - 網路: 穩定連線 (模型下載)

 建議配置：
  - GPU: NVIDIA RTX 3060+ (8GB VRAM)
  - RAM: 16GB+
  - CPU: Intel i7 / AMD Ryzen 7+
  - SSD: 50GB+ 可用空間

🌟 進階功能：
-----------
 • 音訊品質評估：SNR 自動偵測與適應性處理
 • 備用分離策略：語者偵測失敗時的智慧降級處理
 • 彈性輸出格式：支援檔案儲存或記憶體串流
 • 效能監控：即時統計處理效率與資源使用

📁 核心類別：
-----------
 AudioSeparator     ：主要分離引擎，負責音訊分離與品質處理
 SeparationModel    ：模型配置列舉，支援 2/3 人分離模型

⚙️ 設定參數：
-----------
 WINDOW_SIZE        = 6      # 處理窗口 (秒)
 OVERLAP           = 0.5     # 窗口重疊率
 TARGET_RATE       = 16000   # 目標取樣率
 THRESHOLD_NEW     = 0.385   # 新語者判定閾值
 MIN_ENERGY        = 0.001   # 最小音訊能量閾值

📈 輸出資料：
-----------
 分離音檔：./R3SI/Audio-storage/speaker{N}.wav
 混合音檔：./R3SI/Audio-storage/mixed_audio_{timestamp}.wav
 處理日誌：即時輸出至 logger，支援多層級記錄
 識別結果：JSON 格式，包含語者名稱、各語者音訊、時間戳

🔗 相關模組：
-----------
 • utils.logger (統一日誌管理)
 • utils.env_config (環境變數配置)  
 • utils.constants (系統常數定義)

📚 使用範例：
-----------
 # 即時錄音分離
 separator = AudioSeparator(model_type=SeparationModel.SEPFORMER_3SPEAKER)
 separator.record_and_process("./output")
 
 # 離線檔案處理
 run_offline("meeting.wav", "./output", model_name="sepformer_3speaker")

💡 最佳實踐：
-----------
 1. 使用 GPU 加速以獲得最佳效能
 2. 定期清理輸出目錄避免儲存空間不足
 3. 監控系統資源使用，避免記憶體洩漏
 4. 在生產環境中啟用詳細日誌記錄

📞 技術支援：
-----------
 專案倉庫：https://github.com/MISCapstoneProject/Unsaycret-API/tree/v0.4.2
 問題回報：GitHub Issues
 技術文件：README.md & docs/

/*
 *                                                     __----~~~~~~~~~~~------___
 *                                    .  .   ~~//====......          __--~ ~~
 *                    -.            \_|//     |||\  ~~~~~~::::... /~
 *                 ___-==_       _-~o~  \/    |||  \            _/~~-
 *         __---~~~.==~||\=_    -_--~/_-~|-   |\   \        _/~
 *     _-~~     .=~    |  \-_    '-~7  /-   /  ||    \      /
 *   .~       .~       |   \ -_    /  /-   /   ||      \   /
 *  /  ____  /         |     \ ~-_/  /|- _/   .||       \ /
 *  |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\
 *           '         ~-|      /|    |-~\~~       __--~~
 *                       |-~~-_/ |    |   ~\_   _-~            /\
 *                            /  \     \__   \/~                \__
 *                        _--~ _/ | .-~~____--~-/                  ~~==.
 *                       ((->/~   '.|||' -_|    ~~-/ ,              . _||
 *                                  -_     ~\      ~~---l__i__i__i--~~_/
 *                                  _-~-__   ~)  \--______________--~~
 *                                //.-~~~-~_--~- |-------~~~~~~~~
 *                                       //.-~~~--\
 *                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *                               神獸保佑            永無BUG
 */

===============================================================================
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torchaudio
import pyaudio # type: ignore
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference import SepformerSeparation as separator
import noisereduce as nr # type: ignore
import threading
import time
from scipy import signal
from scipy.ndimage import uniform_filter1d
from enum import Enum

# 導入語者數量識別模組
from pyannote.audio import Pipeline

# 導入日誌模組
from utils.logger import get_logger

# 導入配置 (環境變數)
from utils.env_config import (
    AUDIO_RATE, FORCE_CPU, CUDA_DEVICE_INDEX, HF_ACCESS_TOKEN
)

# 導入常數 (應用程式參數)  
from utils.constants import (
    DEFAULT_SEPARATION_MODEL,
    AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE, AUDIO_CHANNELS, 
    AUDIO_WINDOW_SIZE, AUDIO_OVERLAP, AUDIO_MIN_ENERGY_THRESHOLD, 
    AUDIO_MAX_BUFFER_MINUTES, API_MAX_WORKERS, AUDIO_TARGET_RATE
)

# 導入動態模型管理器
from .dynamic_model_manager import (
    SeparationModel,
    MODEL_CONFIGS,
    create_dynamic_model_manager,
    get_available_models
)

# 導入語者計數器
from .speaker_counter import SpeakerCounter

# 導入單人選路器
from .best_speaker_selector import SingleSpeakerSelector

from .assess_quality import assess_audio_quality
from .process_before_id import _gentle_blend, _hf_hiss_suppress, _prep_id_audio, _soft_spectral_floor, _tpdf_dither, crosstalk_suppress, fade_io, framewise_dominance_gate, stft_wiener_refine, tf_mask_refine

# 基本錄音參數（從配置讀取）
CHUNK = AUDIO_CHUNK_SIZE
FORMAT = pyaudio.paFloat32
CHANNELS = AUDIO_CHANNELS
RATE = AUDIO_RATE
TARGET_RATE = AUDIO_TARGET_RATE
WINDOW_SIZE = AUDIO_WINDOW_SIZE
OVERLAP = AUDIO_OVERLAP
DEVICE_INDEX = None

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


DEFAULT_MODEL = DEFAULT_SEPARATION_MODEL

# 修正 DEFAULT_MODEL 的賦值
if DEFAULT_SEPARATION_MODEL == "sepformer_2speaker":
    DEFAULT_MODEL = SeparationModel.SEPFORMER_2SPEAKER
elif DEFAULT_SEPARATION_MODEL == "sepformer_3speaker":
    DEFAULT_MODEL = SeparationModel.SEPFORMER_3SPEAKER
else:
    DEFAULT_MODEL = SeparationModel.SEPFORMER_3SPEAKER  # 預設值改為您的模型

MODEL_NAME = MODEL_CONFIGS[DEFAULT_MODEL]["model_name"]
NUM_SPEAKERS = MODEL_CONFIGS[DEFAULT_MODEL]["num_speakers"]


# 輸出目錄
OUTPUT_DIR = "R3SI/Audio-storage"  # 儲存分離後音訊的目錄
IDENTIFIED_DIR = "R3SI/Identified-Speakers"

# 初始化日誌系統
logger = get_logger(__name__)

# 確保整個模組的 DEBUG 訊息會印出（含所有 handler）
# import logging
# logger.setLevel(logging.DEBUG)
# for h in logger.handlers:
#     try:
#         h.setLevel(logging.DEBUG)
#     except Exception:
#         pass

# 在檔案頂部添加全域快取
_GLOBAL_SEPARATOR_CACHE = {}
_GLOBAL_SPEAKER_PIPELINE_CACHE = None

# ================== 語者分離類別 ======================

class AudioSeparator:
    def __init__(self, model_type: SeparationModel = DEFAULT_MODEL, enable_noise_reduction=True, snr_threshold=SNR_THRESHOLD, enable_dynamic_model=True):
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
        
        # 關閉降噪功能以保持原始音質
        self.enable_noise_reduction = enable_noise_reduction  # 強制關閉以保持音質一致性
        self.snr_threshold = snr_threshold
        
        logger.info(f"使用設備: {self.device}")
        logger.info(f"模型類型: {model_type.value}")
        logger.info(f"載入模型: {self.model_config['model_name']}")
        logger.info(f"支援語者數量: {self.num_speakers}")
        
        # 設計更溫和的低通濾波器
        nyquist = TARGET_RATE // 2
        cutoff = min(HIGH_FREQ_CUTOFF, nyquist - 100)
        self.lowpass_filter = signal.butter(2, cutoff / nyquist, btype='low', output='sos')
        
        # 新增動態模型管理器相關屬性
        self.enable_dynamic_model = enable_dynamic_model
        
        if self.enable_dynamic_model:
            # 使用動態模型管理器
            self.model_manager = create_dynamic_model_manager(self.device)
            logger.info("啟用動態模型選擇機制")
            
            # 預載入預設模型
            self.model_manager.preload_model(model_type)
            self.model, self.current_model_type = self.model_manager.get_model_for_speakers(self.num_speakers)
        else:
            # 使用固定模型（原有邏輯）
            self.model_manager = None
            self.current_model_type = model_type
            try:
                logger.info("正在載入模型...")
                self.model = self._load_model()
                logger.info("模型載入完成")
                self._test_model()
            except Exception as e:
                logger.error(f"模型載入失敗: {e}")
                raise
        
        # 單人情境：自動選聲道器（V2 參數）
        self.single_selector = SingleSpeakerSelector(
            sr=TARGET_RATE,          # 例如 16000，請確保跟你處理音檔的實際採樣率一致
            frame_ms=20,
            hop_ms=10,
            alpha=1.5,               # 能量式 VAD 門檻（僅用於統計/特徵的 gating）
            min_rms=1e-6,

            # 這四個是新版的權重名稱
            w_sisdr=0.60,            # 主特徵：對 mix 的 SI-SDR（投影分數）
            w_band=0.25,             # 300–3400 Hz 人聲頻帶能量佔比
            w_tonality=0.15,         # 1 - spectral flatness
            w_zcr_penalty=0.10,      # 零交越率懲罰（越高越懲罰）

            tie_tol=0.02,
        )
        
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
        
        # 時間追蹤相關變數
        self.session_start_time = None  # 記錄 session 開始的絕對時間
        self._current_t0 = 0.0  # 累計相對時間
        
        # 處理統計
        self.processing_stats = {
            'segments_processed': 0,
            'segments_skipped': 0,
            'errors': 0
        }
        
        self.max_buffer_size = int(RATE * MAX_BUFFER_MINUTES * 60 / CHUNK)

        # 初始化語者計數管線
        self._init_speaker_count_pipeline()

        # 改用獨立類別集中管理語者數量偵測，並傳入快取的管線
        self.spk_counter = SpeakerCounter(
            hf_token=HF_ACCESS_TOKEN, 
            device=self.device, 
            pipeline=getattr(self, 'speaker_count_pipeline', None),
            logger=logger
        )
        logger.info("語者計數器初始化完成")
        
        self._last_single_route_idx = None
        self._last_single_route_score = None
        
        logger.info("AudioSeparator 初始化完成")

    def _init_speaker_count_pipeline(self):
        """初始化語者計數管線 - 使用全域快取"""
        global _GLOBAL_SPEAKER_PIPELINE_CACHE
        
        try:
            # 檢查是否已有全域快取的管線
            if _GLOBAL_SPEAKER_PIPELINE_CACHE is not None:
                self.speaker_count_pipeline = _GLOBAL_SPEAKER_PIPELINE_CACHE
                logger.info("使用快取的語者計數管線")
                return
            
            if HF_ACCESS_TOKEN:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", 
                    use_auth_token=HF_ACCESS_TOKEN
                )
                # 將管線移到相同設備
                if hasattr(self, 'device'):
                    pipeline.to(torch.device(self.device))
                
                # 快取到全域變數
                _GLOBAL_SPEAKER_PIPELINE_CACHE = pipeline
                self.speaker_count_pipeline = pipeline
                logger.info("語者計數管線載入並快取成功")
            else:
                logger.warning("未提供 HF_ACCESS_TOKEN，語者計數功能將受限")
                self.speaker_count_pipeline = None
        except Exception as e:
            logger.warning(f"語者計數管線載入失敗: {e}")
            self.speaker_count_pipeline = None

    def _load_model(self):
        """載入語者分離模型"""
        model_name = self.model_config["model_name"]
        
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

    def _test_model(self):
        """測試模型"""
        try:
            with torch.no_grad():
                # SpeechBrain SepFormer 模型測試
                # SepFormer 期望輸入格式為 [batch, samples]
                test_audio = torch.randn(1, AUDIO_SAMPLE_RATE).to(self.device)
                logger.debug(f"SpeechBrain 測試音訊形狀: {test_audio.shape}")
                output = self.model.separate_batch(test_audio)
                    
            logger.info("模型測試通過")
            logger.debug(f"輸出形狀: {output.shape if hasattr(output, 'shape') else type(output)}")
            
        except Exception as e:
            logger.error(f"模型測試失敗: {e}")
            logger.error(f"測試音訊形狀: {test_audio.shape if 'test_audio' in locals() else 'N/A'}")
            raise
        
    def _infer_layout(self, est: torch.Tensor) -> tuple[str, int, int]:
        """
        回傳 (layout, spk_axis, time_axis)
        layout ∈ {'BST','BTS'}；BST 表示 [B, S, T]、BTS 表示 [B, T, S]
        """
        assert est.dim() == 3, f"unexpected est shape: {tuple(est.shape)}"
        B, D1, D2 = est.shape[0], est.shape[1], est.shape[2]
        # 哪個維度像「說話者」? （很小且在 1-4 之間）
        if 1 <= D1 <= 4 and not (1 <= D2 <= 4):
            return "BST", 1, 2  # [B, S, T]
        if 1 <= D2 <= 4 and not (1 <= D1 <= 4):
            return "BTS", 2, 1  # [B, T, S]
        # 都像或都不像：偏好 [B, S, T]
        return "BST", 1, 2

    def _normalize_estimates(self, est: torch.Tensor) -> tuple[torch.Tensor, str, int, int]:
        """
        對每個說話者「沿時間軸」做 peak normalize（常數縮放），避免時間點依賴的失真。
        回傳 (normalized, layout, spk_axis, time_axis)
        """
        if est.dim() == 2:
            peak = est.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            return est / peak, "BT", -1, -1
        layout, s_ax, t_ax = self._infer_layout(est)
        peak = est.abs().amax(dim=t_ax, keepdim=True).clamp_min(1e-8)
        return est / peak, layout, s_ax, t_ax

    def _get_appropriate_model(self, num_speakers: int) -> tuple[separator, SeparationModel]:
        """
        取得適當的模型實例
        
        Args:
            num_speakers: 偵測到的語者數量
            
        Returns:
            tuple: (模型實例, 模型類型)
        """
        if self.enable_dynamic_model and self.model_manager:
            return self.model_manager.get_model_for_speakers(num_speakers)
        else:
            # 固定模型模式
            return self.model, self.current_model_type

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
        
        num_speakers = separated_signals.shape[speaker_dim]
        
        for i in range(min(num_speakers, self.num_speakers)):
            if speaker_dim == 2:
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
            
            if speaker_dim == 2:
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
            # identifier = SpeakerIdentifier()

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
        
        # 清理模型管理器
        if self.model_manager:
            self.model_manager.cleanup()
        
        # 清理語者計數管線
        if hasattr(self, 'speaker_count_pipeline') and self.speaker_count_pipeline is not None:
            try:
                # 清理管線資源
                del self.speaker_count_pipeline
                self.speaker_count_pipeline = None
                logger.info("語者計數管線已清理")
            except Exception as e:
                logger.error(f"清理語者計數管線時發生錯誤：{e}")
        
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

    def separate_and_save(self, audio_tensor, output_dir, segment_index, absolute_start_time=None):
        """
        分離並儲存音訊，並回傳 (path, start, end) 列表。
        流程：語者計數 → 動態模型選擇 → 分離 → 強化(可選) → 儲存
        
        Args:
            audio_tensor: 音訊張量
            output_dir: 輸出目錄
            segment_index: 片段索引
            absolute_start_time: 音訊的絕對開始時間（datetime 物件）
        """
        try:
            # 先以寬鬆範圍跑一次，並套用重疊感知後處理；若你的批次確定雙人，可設 expected_min/max=2
            detected_speakers = self.spk_counter.count_with_refine(
                audio=audio_tensor,
                sample_rate=TARGET_RATE,
                expected_min=1,
                expected_max=3,
                first_pass_range=(1, 3),
                allow_zero=True,         # <== 允許回傳 0（無語音）
                debug=False
            )

            logger.info(f"片段 {segment_index} - 偵測到 {detected_speakers} 位說話者")
            
            # 備援：第一次回 0 → 只有在「強有聲」才重試 1–2 人
            if detected_speakers == 0:
                ok, m = self.spk_counter._has_voice(audio_tensor, TARGET_RATE, return_metrics=True)
                # 與 SpeakerCounter 同步或更嚴的條件
                strong_voice = ok and (m["voiced_ratio"] >= 0.12) and (m["voiced_union"] >= 0.50) and (m.get("loud_frac", 0.0) >= 0.05)
                if not strong_voice:
                    logger.info(f"片段 {segment_index} - 無語音/過短（ratio={m['voiced_ratio']:.3f}, union={m['voiced_union']:.2f}s, loud={m.get('loud_frac',0.0):.3f}），跳過")
                    return []

                logger.warning(f"片段 {segment_index} - 第一次偵測 0，但語音跡象偏強，嘗試 1–2 人重試")
                retry = self.spk_counter.count_with_refine(
                    audio=audio_tensor, sample_rate=TARGET_RATE,
                    expected_min=1, expected_max=2,
                    first_pass_range=(1, 2),
                    allow_zero=False,           # 已確認強有聲，就不要再回 0
                    debug=False
                )
                detected_speakers = max(1, int(retry))
            
            # 動態選擇模型
            current_model, current_model_type = self._get_appropriate_model(detected_speakers)
            
            # 使用動態模型管理器取得模型配置
            if self.model_manager:
                model_config = self.model_manager.get_model_config(current_model_type)
            else:
                model_config = MODEL_CONFIGS[current_model_type]
            
            logger.debug(f"使用模型: {current_model_type.value} (偵測語者: {detected_speakers})")
            
            # 記錄絕對時間戳
            if absolute_start_time is None:
                from datetime import timezone, timedelta
                taipei_tz = timezone(timedelta(hours=8))
                absolute_start_time = datetime.now(taipei_tz)
            
            # 初始化累計時間戳
            current_t0 = getattr(self, "_current_t0", 0.0)
            results = []   # 用來收 (path, start, end, absolute_timestamp)
            seg_duration = audio_tensor.shape[-1] / TARGET_RATE
            
            with torch.no_grad():
                
                # 確保輸入是 [batch, samples] 格式
                if len(audio_tensor.shape) == 3:
                    # 如果是 [batch, channels, samples]，需要去掉 channels 維度
                    if audio_tensor.shape[1] == 1:
                        audio_tensor = audio_tensor.squeeze(1)  # 變成 [batch, samples]
                
                # 使用選定的模型進行分離
                separated = current_model.separate_batch(audio_tensor)
                separated, layout, spk_axis, time_axis = self._normalize_estimates(separated)
                
                # 取得混音（原始輸入）一維波形
                mix_wave = audio_tensor[0].detach().cpu()
                
                # 1) 分離後先不動：保留原始 for 評分/選路（SI-SDR 對常數縮放不敏感）
                raw_for_select = separated

                # 2) 僅為最後輸出「可選」做強化（不要用強化後訊號做任何評分/選路）
                enhanced_separated = self.enhance_separation(separated) if self.enable_noise_reduction else separated

                # 3) 推斷 layout，統一取得「模型輸出說話者數」與取片函式
                if layout == "BST":           # [B, S, T]
                    model_output_speakers = enhanced_separated.shape[spk_axis]
                    def _get_cand(idx):  return raw_for_select[0, idx, :].detach().cpu()
                    def _get_final(idx): return enhanced_separated[0, idx, :].detach().cpu()
                elif layout == "BTS":         # [B, T, S]
                    model_output_speakers = enhanced_separated.shape[2]
                    def _get_cand(idx):  return raw_for_select[0, :, idx].detach().cpu()
                    def _get_final(idx): return enhanced_separated[0, :, idx].detach().cpu()
                else:                         # "BT" → 單一路輸出（無法做多路選路）
                    model_output_speakers = 1
                    def _get_cand(idx):  return raw_for_select[0, :].detach().cpu()
                    def _get_final(idx): return enhanced_separated[0, :].detach().cpu()

                # —— 單人情境：用「原始」候選做選路與 SI-SDR 分數 ——
                if detected_speakers == 1 and model_output_speakers >= 2:
                    candidates = [_get_cand(j) for j in range(model_output_speakers)]
                    try:
                        best_idx, best_tensor_raw, stats_list = self.single_selector.select(candidates, mix_wave, return_stats=True)
                        if stats_list is not None:
                            s = stats_list[best_idx]
                            logger.info(
                                f"1-spk 選路：speaker{best_idx+1} | "
                                f"SI-SDR={s['si_sdr_db']:.2f} dB, band={s['band_ratio']:.2f}, "
                                f"tonality={s['tonality']:.2f}, zcr_penalty={s['zcr_penalty']:.2f}, rms={s['rms']:.4f}"
                            )
                        # 真的要輸出時，才拿「同一索引」的 enhanced（或原始，視設定）
                        best_tensor = _get_final(best_idx)
                        enhanced_separated = best_tensor.unsqueeze(0).unsqueeze(-1).to(self.device)  # -> [1, T, 1]
                        model_output_speakers = 1
                        
                        if hasattr(self, "_last_single_route_idx") and self._last_single_route_idx is not None:
                            prev_idx = self._last_single_route_idx
                            prev_score = self._last_single_route_score if self._last_single_route_score is not None else -1e9
                            cur_score = s.get("score", s["si_sdr_db"])  # 你的 selector 若有整體 score 就用它，否則用 SI-SDR 代替
                            # 若兩路分數差異很小，鎖定上次的路徑（避免來回跳）
                            if abs(cur_score - prev_score) < 0.03 and best_idx != prev_idx:
                                best_idx = prev_idx
                        
                        # 更新路徑記憶
                        self._last_single_route_idx = best_idx
                        self._last_single_route_score = s.get("score", s["si_sdr_db"])
                        
                    except Exception:
                        logger.exception("單人選路失敗，改用 speaker1 作為保守輸出")
                        best_tensor = _get_final(0)
                        enhanced_separated = best_tensor.unsqueeze(0).unsqueeze(-1).to(self.device)
                        model_output_speakers = 1
                
                # 把 estimates 統一成 [S, T] on CPU
                if enhanced_separated.ndim == 3:   # [B, T, S]
                    est_ST = enhanced_separated[0].transpose(0, 1).detach().cpu()  # [S, T]
                else:
                    est_ST = enhanced_separated.detach().cpu().unsqueeze(0)        # [1, T]

                # 單人情境：若模型有 >=2 路，僅保留被選中的那一路（用你上面算出的 best_idx）
                if detected_speakers == 1 and est_ST.shape[0] >= 2:
                    try:
                        # 你上面已經選出 best_idx；這裡只保留那一路
                        est_ST = est_ST[best_idx:best_idx+1, :]
                    except Exception:
                        est_ST = est_ST[0:1, :]

                # === 1) Projection-back：用混音能量做線性重定標（超小成本、很有效） ===
                x = mix_wave  # [T] CPU
                for s in range(est_ST.shape[0]):
                    y = est_ST[s]
                    denom = torch.dot(y, y).clamp_min(1e-8)
                    alpha = torch.dot(x, y) / denom
                    est_ST[s] = alpha * y

                # === 2) Mixture-consistent Wiener（稍微銳一點，但不激進） ===
                if detected_speakers == 1:
                    est_ST = stft_wiener_refine(
                        est_ST, x,
                        n_fft=1024, hop=256, win_length=1024,
                        wiener_p=0.7
                    )
                # est_ST = stft_wiener_refine(
                #     est_ST, x,
                #     n_fft=1024, hop=256, win_length=1024,
                #     wiener_p=0.8
                # )

                # （可選）對多人再加一點點時間框主導門控，壓小漏音（如果你本來就有 framewise_dominance_gate，就沿用）
                # if int(detected_speakers) >= 2:
                #     est_ST = stft_wiener_refine(
                #         est_ST, x,
                #         n_fft=1024, hop=256, win_length=1024,
                #         wiener_p=2.2
                #     )
                #     est_ST = framewise_dominance_gate(
                #         est_ST, frame=320, hop=160,
                #         rel_ratio=0.24,   # 原先 0.36；愈高愈不嚴，較自然
                #         min_floor=0.12,   # 原先 0.10；墊高一點避免乾裂與沙砂
                #         fade=240          # 稍微拉長 crossfade，邊界更平滑
                #     )
                
                del separated
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
                
                # 根據實際情況決定要分離多少個語者
                # 策略：使用偵測到的語者數量，但不超過模型輸出的通道數
                # effective_speakers = min(detected_speakers, model_output_speakers, model_config["num_speakers"])
                # >>> FIX: 以 est_ST 的 S 為準；避免用模型原始輸出數或偵測數導致不一致
                S, T = est_ST.shape
                effective_speakers = min(int(detected_speakers), int(S), int(model_config["num_speakers"]))
                
                logger.debug(
                    f"分離參數 - 偵測: {detected_speakers}, "
                    f"est_ST通道: {S}, 模型支援: {model_config['num_speakers']}, 有效: {effective_speakers}"
                )
                
                saved_count = 0
                start_time = current_t0
                
                for i in range(effective_speakers):
                    try:
                        # >>> FIX: est_ST 是 [S, T]，正確取法：
                        speaker_audio = est_ST[i].contiguous()  # 1D [T]

                        fade_ms = 24.0 if int(detected_speakers) == 1 else 16.0
                        
                        # 先做淡入淡出，減少邊界噪點
                        speaker_audio = fade_io(speaker_audio.clone(), TARGET_RATE, fade_ms=fade_ms)

                        # 動態範圍保護（只在必要時縮放），然後 clamp
                        max_val = float(torch.max(torch.abs(speaker_audio)))
                        if max_val > 0.97:
                            speaker_audio = speaker_audio * (0.95 / max_val)
                        speaker_audio = speaker_audio.clamp_(-1.0, 1.0)

                        # 能量門檻（保守一點）
                        rms = torch.sqrt(torch.mean(speaker_audio ** 2))
                        if rms <= 0.004:
                            logger.debug(f"語者 {i+1} 能量太低 (RMS={rms:.6f}), 跳過儲存")
                            continue
                    
                        id_audio = _prep_id_audio(speaker_audio, TARGET_RATE)   # ← 一律走這條給語者辨識
                        
                        final_tensor = id_audio.unsqueeze(0).cpu()  # [1, T]
                        final_tensor = _tpdf_dither(final_tensor, level_db=-92.0)
                        
                        # 在保存前後、或錄音模式每段處理完，快速評估一次
                        metrics = assess_audio_quality(id_audio, TARGET_RATE, logger=logger)
                        logger.info(f"品質 {metrics['grade']}({metrics['quality_score']:.1f}) | "
                                    f"rms={metrics['rms_dbfs']:.1f}dBFS, snr≈{metrics['snr_db_est']:.1f}dB, "
                                    f"centroid={metrics['spectral_centroid_hz']:.0f}Hz, clip={metrics['clipping_pct']*100:.2f}%")

                        # 若有參考訊號（例如混音），也可以：
                        metrics = assess_audio_quality(speaker_audio, TARGET_RATE, logger=logger, ref_wave=mix_wave)
                        logger.info(f"SI-SDR={metrics.get('si_sdr_db', float('nan')):.2f} dB")
                        
                        output_file = os.path.join(
                            output_dir,
                            f"speaker{i+1}.wav"
                            # f"speaker{i+1}_{timestamp}_{segment_index}.wav"
                        )
                        
                        # 保存音訊時使用較高的品質設定
                        torchaudio.save(
                            output_file,
                            final_tensor,
                            TARGET_RATE,
                            bits_per_sample=16  # 指定16位元確保音質
                        )
                        
                        # 另外存一份給人聽（不影響辨識流程）
                        if detected_speakers == 1 and getattr(self, "save_pretty_copy", False):
                            pretty_path = output_file.replace(".wav", "_pretty.wav")
                            pretty = _gentle_blend(speaker_audio, mix_wave, ratio=0.08)
                            torchaudio.save(pretty_path, pretty.unsqueeze(0).cpu(), TARGET_RATE, bits_per_sample=16)
                        
                        # 計算絕對時間戳
                        absolute_timestamp = absolute_start_time.timestamp() + start_time

                        results.append((output_file,
                                start_time,
                                start_time + seg_duration,
                                absolute_timestamp))  # 加入絕對時間戳
                        self.output_files.append(output_file)
                        
                        saved_count += 1
                        
                    except Exception as e:
                        logger.warning(f"儲存語者 {i+1} 失敗: {e}")
                
                if saved_count > 0:
                    logger.info(f"片段 {segment_index} 完成，儲存 {saved_count}/{effective_speakers} 個檔案 (使用 {current_model_type.value})")
                
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

    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
        logger.info("準備停止錄音...")

    def get_output_files(self):
        """獲取所有分離後的音檔路徑"""
        return self.output_files

# 添加全域函式來管理快取
def get_cached_separator(model_type: SeparationModel = DEFAULT_MODEL, enable_dynamic_model: bool = True, **kwargs) -> AudioSeparator:
    """
    取得快取的 AudioSeparator 實例，避免重複初始化
    
    Args:
        model_type: 模型類型
        enable_dynamic_model: 是否啟用動態模型
        **kwargs: 其他參數
    
    Returns:
        AudioSeparator 實例
    """
    global _GLOBAL_SEPARATOR_CACHE
    
    # 建立快取鍵
    cache_key = f"{model_type.value}_{enable_dynamic_model}_{hash(tuple(sorted(kwargs.items())))}"
    
    # 檢查快取
    if cache_key in _GLOBAL_SEPARATOR_CACHE:
        logger.info(f"使用快取的 AudioSeparator: {cache_key}")
        return _GLOBAL_SEPARATOR_CACHE[cache_key]
    
    # 建立新實例並快取
    logger.info(f"建立新的 AudioSeparator: {cache_key}")
    separator = AudioSeparator(
        model_type=model_type, 
        enable_dynamic_model=enable_dynamic_model, 
        **kwargs
    )
    _GLOBAL_SEPARATOR_CACHE[cache_key] = separator
    
    return separator

def clear_separator_cache():
    """清理所有快取的分離器實例"""
    global _GLOBAL_SEPARATOR_CACHE, _GLOBAL_SPEAKER_PIPELINE_CACHE
    
    # 清理分離器快取
    for separator in _GLOBAL_SEPARATOR_CACHE.values():
        try:
            if hasattr(separator, 'model_manager') and separator.model_manager:
                separator.model_manager.cleanup()
        except Exception as e:
            logger.warning(f"清理分離器時發生錯誤: {e}")
    
    _GLOBAL_SEPARATOR_CACHE.clear()
    
    # 清理語者計數管線快取
    if _GLOBAL_SPEAKER_PIPELINE_CACHE is not None:
        try:
            del _GLOBAL_SPEAKER_PIPELINE_CACHE
            _GLOBAL_SPEAKER_PIPELINE_CACHE = None
            logger.info("已清理語者計數管線快取")
        except Exception as e:
            logger.warning(f"清理語者計數管線快取時發生錯誤: {e}")

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
    
def run_realtime(output_dir: str = OUTPUT_DIR, model_type: SeparationModel = None, model_name: str = None, enable_dynamic_model: bool = True) -> str:
    """方便外部呼叫的錄音處理函式，支援模型選擇和動態模型 - 使用快取"""
    if enable_dynamic_model:
        separator = get_cached_separator(model_type=DEFAULT_MODEL, enable_dynamic_model=True)
    elif model_name:
        # 轉換 model_name 為 model_type
        if model_name == "sepformer_2speaker":
            model_type = SeparationModel.SEPFORMER_2SPEAKER
        elif model_name == "sepformer_3speaker":
            model_type = SeparationModel.SEPFORMER_3SPEAKER
        else:
            raise ValueError(f"不支援的模型: {model_name}")
        separator = get_cached_separator(model_type=model_type, enable_dynamic_model=False)
    elif model_type:
        separator = get_cached_separator(model_type=model_type, enable_dynamic_model=False)
    else:
        separator = get_cached_separator(model_type=DEFAULT_MODEL, enable_dynamic_model=False)
    return separator.record_and_process(output_dir)

def run_offline(file_path: str, output_dir: str = OUTPUT_DIR, save_files: bool = True, 
                model_type: SeparationModel = None, model_name: str = None, enable_dynamic_model: bool = True) -> None:
    """方便外部呼叫的離線音檔處理函式，支援模型選擇和動態模型 - 使用快取"""
    if enable_dynamic_model:
        separator = get_cached_separator(model_type=DEFAULT_MODEL, enable_dynamic_model=True)
    elif model_name:
        # 轉換 model_name 為 model_type
        if model_name == "sepformer_2speaker":
            model_type = SeparationModel.SEPFORMER_2SPEAKER
        elif model_name == "sepformer_3speaker":
            model_type = SeparationModel.SEPFORMER_3SPEAKER
        else:
            raise ValueError(f"不支援的模型: {model_name}")
        separator = get_cached_separator(model_type=model_type, enable_dynamic_model=False)
    elif model_type:
        separator = get_cached_separator(model_type=model_type, enable_dynamic_model=False)
    else:
        separator = get_cached_separator(model_type=DEFAULT_MODEL, enable_dynamic_model=False)
    separator.set_save_audio_files(save_files)
    separator.process_audio_file(file_path, output_dir)