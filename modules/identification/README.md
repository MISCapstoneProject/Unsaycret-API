Speaker Identification Engine (v5.1.2)

作者：Gino
最後更新：2025‑06‑18

## 模組定位

這個模組提供「輸入任何音檔 / 音訊流 → 輸出說話者身分」的一站式能力，並可根據相似度自動：
    更新既有聲紋向量（加權平均）
    為同一語者新增額外聲紋
    建立全新語者與其第一筆聲紋
所有結果會即時同步到 Weaviate 向量資料庫，方便後續查詢與擴充。
一句話：給我一段聲音，我告訴你「他是誰」或「他是新人」。

## 特色亮點
類別                 功能                        說明

AudioProcessor      取樣率自適應、最長 10 秒裁切    支援 8/16/44.1 kHz，自動重採樣 → 16 kHz │ ECAPA‑TDNN 嵌入模型（SpeechBrain）

WeaviateRepository  向量 CRUD + 去向管理          包辦比對、平均更新、新增聲紋、建新語者

SpeakerIdentifier   單例入口、策略判斷             依距離閾值自動選擇「跳過 / 更新 / 新增 / 建新語者」

## 快速開始

# 1. 安裝依賴（CPU 版）
pip install speechbrain weaviate-client numpy scipy soundfile torch

# 2. 啟動 Weaviate（Docker）
docker-compose -f weaviate_study/docker-compose.yml up -d

from modules.identification.VID_identify_v5 import SpeakerIdentifier

idtf = SpeakerIdentifier()

# 單檔識別
idtf.process_audio_file("samples/voice.wav")

# 批次處理資料夾
stats = idtf.process_audio_directory("samples/")

# 已知語者新增聲紋
idtf.add_voiceprint_to_speaker("voice.wav", speaker_uuid)

## 閾值策略
距離範圍         動作
< 0.26          距離極小 → 跳過（認定同一檔）
0.26 – 0.34     更新 現有聲紋（加權平均）
0.34 – 0.385    新增聲紋 至同語者
> 0.385         新語者，建立 Speaker & VoicePrint