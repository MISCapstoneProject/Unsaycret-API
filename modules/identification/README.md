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

自訂：
idtf.threshold_low = 0.25
idtf.threshold_update = 0.33
idtf.threshold_new = 0.40

## SpeakerIdentifier 工具箱

| 功能               | 對應函式                                                | 白話說明                           |
| ---------------- | --------------------------------------------------- | ------------------------------ |
| 處理單個音檔來辨識是誰講的    | `process_audio_file(audio_path)`                   | 給它一個音檔路徑，它會幫你找出最像的語者|
| 批次處理資料夾中所有音檔     | `process_audio_directory(dir_path)`                | 把整個資料夾丟進去，會幫你一個一個辨識語者。|
| 處理即時音訊流（像直播或麥克風） | `process_audio_stream(stream)`                  | 如果你拿到的是一段音訊流資料，它也可以辨識。|
| 把音檔新增到某位語者身上     | `add_voiceprint_to_speaker(audio_path, speaker_id)` | 如果你知道這是誰的聲音，可以把這個聲音加入那個語者的聲紋中。|
| 刪除整個語者的聲紋資料 | `delete_speaker(speaker_id)`                             | 完整清空某位語者的所有資料。
| 手動設定/更新語者的聲紋向量   | `update_speaker_embedding(speaker_id, new_vector)`| 比較進階，直接塞新的特徵向量進語者資料裡。
| 查看目前資料庫中有哪些語者    | `list_all_speakers()`                             | 列出所有已註冊過的語者 ID。          |
| 將音檔轉換為語者向量（不比對）  | `extract_embedding(audio_path)`                 | 把聲音轉成向量，但不做比對，適合分析或可視化。|



## 前置需求

Python 3.9+

SpeechBrain（自動下載 ECAPA‑TDNN 權重）

Weaviate 向量資料庫（Docker 一鍵啟動）

NumPy / PyTorch / SoundFile / SciPy

##  注意事項

確保 Weaviate 已啟動，且 Speaker / VoicePrint 兩個 Schema 已建立（weaviate_study/create_collections.py）。

最佳效果建議輸入 16 kHz 單聲道 音檔。

如需無人為監督批次匯入，大量音檔可先調寬 threshold_new，避免產生過多新語者。