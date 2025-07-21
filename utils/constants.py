"""
應用程式常數定義
這些值是演算法的核心參數，不應該隨意修改
"""

# 語者識別閾值 (演算法核心參數，經過實驗調校)
THRESHOLD_LOW = 0.26      # 過於相似，不更新向量
THRESHOLD_UPDATE = 0.34   # 相似度足夠，更新向量
THRESHOLD_NEW = 0.385     # 超過此值視為新語者

# 音訊處理固定參數 (技術規格要求)
AUDIO_SAMPLE_RATE = 16000    # SpeechBrain 模型要求的取樣率
AUDIO_TARGET_RATE = 16000    # 處理時統一的目標取樣率
AUDIO_CHANNELS = 1           # 單聲道
WHISPER_MODEL_CACHE_DIR = "models/faster-whisper"

# 預設模型配置 (經過測試的穩定版本)
DEFAULT_WHISPER_MODEL = "medium"
DEFAULT_WHISPER_BEAM_SIZE = 5
DEFAULT_SEPARATION_MODEL = "sepformer_2speaker"

# 模型名稱常數 (固定的 HuggingFace 模型ID)
SPEECHBRAIN_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SPEECHBRAIN_SEPARATOR_MODEL = "speechbrain/sepformer-whamr16k"

# Whisper 模型對照表
WHISPER_MODEL_MAP = {
    "tiny": "guillaumekln/faster-whisper-tiny",
    "base": "guillaumekln/faster-whisper-base",
    "small": "guillaumekln/faster-whisper-small",
    "medium": "guillaumekln/faster-whisper-medium",
    "large-v2": "guillaumekln/faster-whisper-large-v2",
    "large-v3": "guillaumekln/faster-whisper-large-v3",
}

# WebSocket 處理參數
WEBSOCKET_CHUNK_SECS = 6
WEBSOCKET_TIMEOUT = 0.05
WEBSOCKET_MAX_WORKERS = 2

# 音訊品質參數
AUDIO_MIN_ENERGY_THRESHOLD = 0.001
AUDIO_MAX_BUFFER_MINUTES = 5
AUDIO_CHUNK_SIZE = 1024
AUDIO_WINDOW_SIZE = 6
AUDIO_OVERLAP = 0.5

# 音訊分離處理參數 (RSS_3_v1)
# 注意: 這些是 pyaudio 枚舉值，需要在使用時轉換
AUDIO_PYAUDIO_FORMAT_STR = "paFloat32"  # 用於字符串比較
AUDIO_RECORDING_RATE = 44100
NOISE_REDUCE_STRENGTH = 0.05
SNR_THRESHOLD = 8
WIENER_FILTER_STRENGTH = 0.01
HIGH_FREQ_CUTOFF = 7500
DYNAMIC_RANGE_COMPRESSION = 0.7

# ConvTasNet 模型參數
CONVTASNET_MODEL_NAME = "JorisCos/ConvTasNet_Libri3Mix_sepnoisy_16k"
NUM_SPEAKERS_SEPARATION = 3

# API 預設值
API_DEFAULT_VERIFICATION_THRESHOLD = 0.4
API_DEFAULT_MAX_RESULTS = 3
API_MAX_WORKERS = 2
