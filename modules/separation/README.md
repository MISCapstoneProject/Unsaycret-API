# Separation (語音分離) 模組

**版本**: v0.4.0
**作者**: Gino  
**最後更新者**: CYouuu  
**最後更新**: 2025-07-21

本程式是一支「即時語音多工處理」腳本，能夠在錄音進行中：

分離同一段音訊裡的兩位說話者（SpeechBrain Sepformer，16 kHz，雙語者預訓練）。
立刻呼叫語者識別模組（modules/identification/VID_identify_v5.py）進行聲紋比對。
將過程、結果與輸出檔全部列入 system_output.log 方便後續分析。
⚠️ 模型限制：當前 Sepformer 僅支援「最多兩位」同講，三人以上會被壓縮成兩軌。

## 目錄結構
modules/voice_id/
├── speaker_system_v2.py   # 這支腳本（主流程）
└── README.md              # 你現在看的檔案
modules/identification/
└── VID_identify_v5.py     # 語者辨識工具箱（被呼叫）
16K-model/Audios-16K-IDTF/ # 預設音檔輸出目錄

## 快速安裝

# 建議 Python 3.9+，GPU 可選
pip install -r requirements.txt  # 或 requirements-gpu.txt

# 啟動 Weaviate  (必要)
docker-compose -f weaviate_study/docker-compose.yml up -d
python weaviate_study/create_collections.py  # 建立 Speaker / VoicePrint schema

## 如何運行
錄音（即時）模式
python modules/voice_id/separator.py  # 預設開啟麥克風錄音

檔案（離線）模式
from modules.voice_id.separator import AudioSeparator
sep = AudioSeparator()
audio = torchaudio.load("mixed.wav")[0]  # tensor shape [1, time]
sep.separate_and_identify(audio, "output", segment_index=1)

## 重要參數
    名稱            預設           意義
WINDOW_SIZE          6      每次取幾秒音訊做分離

OVERLAP              0.5    視窗重疊率 (50 %)

MIN_ENERGY_THRESHOLD 0.005  靜音門檻，低於此值不處理

TARGET_RATE          16000  Sepformer 要求的取樣率

## 輸出

    內容                儲存                              路徑說明

已分離單人音檔   16K-model/Audios-16K-IDTF/     檔名格式 speaker{N}_<timestamp>_<seg>.wav

原始混合音檔    同目錄，檔名前綴 mixed_audio_     便於重播與除錯

識別結果 LOG    system_output.log              含時間戳、片段索引、語者名稱與距離

## 已知限制

目前僅支援 雙語者分離。

ASR（Whisper）尚未整合在本腳本，需在上層 orchestrator 處理。

如果要真・即時字幕，需改用 streaming‑friendly ASR（Vosk、Google STT）。