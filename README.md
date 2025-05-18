SPEECHPROJECT_CLEAN 專案說明
專案簡介
本專案為語音處理系統整合範例，包含語音分離、語音辨識、說話人辨識、API 封裝，以及資料庫串接（Weaviate）等模組，目標是提供一套可拓展的語音處理流程框架。

## 📂 專案架構

SPEECHPROJECT_CLEAN/
├── ct2_models/ # Faster-Whisper 模型資料（.bin、tokenizer等）
├── examples/ # 測試範例與使用範例腳本
├── models/ # SpeechBrain 語音模型與配置
├── modules/ # 自定義模組（ASR, Separation 等邏輯）
├── pipelines/ # 整合流程邏輯（分離 + ASR 流程）
├── pretrained_models/ # 預訓練模型檔案（如 Sepformer）
├── services/ # FastAPI 伺服器與 API 定義
├── utils/ # 輔助工具模組
├── weaviate_study/ # 與 Weaviate 向量資料庫的整合實驗
├── work_output/ # 分離後的音檔與辨識結果（不納入 Git）
├── logs/ # 執行過程的日誌輸出（不納入 Git）
├── venv/ # 虛擬環境（建議自行建立）
│
├── .env # 環境變數設定（建議保留為 .gitignore）
├── .gitignore # Git 忽略項目設定
├── pyproject.toml # Python 專案描述（如使用 poetry）
├── requirements.txt # 依賴套件清單
├── README.md # 本說明文件


## 🚀 功能特色

- 🎙 **語者分離（Speaker Separation）**
  - 使用 SpeechBrain Sepformer 模型
  - 可分離雙人以上交談

- 🧠 **語音辨識（ASR）**
  - 使用 Faster-Whisper GPU/CPU 動態選擇
  - 回傳逐詞 timestamp 以及信心值 fallback 機制

- 🛜 **API 服務**
  - 使用 FastAPI 提供 RESTful 接口
  - 支援：
    - 語音上傳轉文字
    - 即時 WebSocket 傳輸（v0.2.0 功能）
    - PATCH 手動校正字幕（v0.2.0 功能）

- 🧠 **Weaviate 整合**  
  - 可儲存語音向量與辨識結果，支援語者比對與搜尋（測試中）

# 📘 更新紀錄（CHANGELOG）

## 2024-05-16
- ✅ 將 logging 功能從 speaker_id 模組抽出，改為 `utils/log_utils.py`
- ✅ logging 支援自動建立 logs 資料夾與自動命名 log 檔
- ✅ 主程式引用方式統一改為 `from utils.log_utils import ...`
- ✅ 刪除舊的 `output_log.txt` 檔案，並新增 `.gitignore` 排除 logs
- ✅ 新增專案 README 結構與說明


