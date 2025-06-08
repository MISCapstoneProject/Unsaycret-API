# Unsaycret-API

## 專案簡介
Unsaycret-API 是一套模組化的語音處理系統，整合語音分離、說者辨識、語音辨識、API 服務，並支援向量資料庫串接。
這專案執行後會架起 API 伺服器，以供呼叫使用。

---

## 🚀 主要功能

- 🎙 **語者分離**：採用 SpeechBrain Sepformer，支援雙人語音分離。
- 🧠 **語音辨識（ASR）**：Faster-Whisper，支援 GPU/CPU 動態切換，逐詞時間戳與信心值。
- 🗣 **說話人辨識**：ECAPA-TDNN 語者聲紋比對，支援聲紋自動更新。
- 🛜 **API 服務**：FastAPI 提供 RESTful 與 WebSocket 介面。
- 🧠 **Weaviate 整合**：語音向量與辨識結果可存入 Weaviate，支援語者搜尋與比對。

---

## 📂 目錄結構與模組說明

```
sonic-slicer-api/
├── examples/                # 測試與範例腳本
├── models/                  # 語音模型（含分離與辨識）
│   ├── faster-whisper/      # Faster-Whisper ASR 模型快取
│   ├── speechbrain_recognition/   # 語者辨識模型檔
│   └── speechbrain_Separation_16k/ # 語音分離模型檔
├── modules/                 # 主要自定義功能模組
│   ├── asr/                 # 語音辨識（ASR）模組
│   │   ├── asr_model.py         # Faster-Whisper 載入與封裝
│   │   ├── text_utils.py        # 文字處理輔助
│   │   └── whisper_asr.py       # ASR 主流程
│   ├── database/            # 資料庫操作（如 Weaviate）
│   │   └── database.py
│   ├── identification/      # 說話人辨識
│   │   └── VID_identify_v5.py   # 語者辨識主程式
│   ├── management/          # 語者/資料管理
│   │   └── manager.py
│   ├── separation/          # 語者分離
│   │   └── separator.py         # 分離主流程
│   └── utils/               # 工具模組
│       └── logger.py            # 集中式日誌系統
├── pipelines/               # 處理流程（分離+辨識）
│   └── orchestrator.py          # 分離+辨識+ASR 整合流程
├── services/                # FastAPI 伺服器與 API 定義
│   └── api.py                   # API 入口
├── utils/                   # 其他輔助腳本
│   ├── logger.py
│   └── sync_npy_username.py
├── weaviate_study/          # Weaviate 向量資料庫整合與測試
│   ├── create_collections.py
│   ├── docker-compose.yml
│   ├── npy_to_weaviate.py
│   ├── test_create_collection_1.py
│   ├── test_create_collection_2.py
│   ├── tool_delete_all.py
│   └── tool_search.py
├── work_output/             # 處理結果輸出（分離音檔、辨識結果）
│   └── <日期時間>/              # 每次處理自動建立資料夾
│       ├── output.json          # 分離+辨識結果
│       ├── speaker1_xxx.wav     # 分離後音檔
│       └── speaker2_xxx.wav
├── requirements.txt         # 依賴套件清單
├── pyproject.toml           # Python 專案描述
├── README.md                # 本說明文件
└── ...
```

---

### 各資料夾/模組詳細說明

- **examples/**  
  測試與範例腳本，快速驗證各模組功能。

- **models/**  
  - `faster-whisper/`：ASR 模型快取（自動下載）。
  - `speechbrain_recognition/`：語者辨識模型（ECAPA-TDNN）。
  - `speechbrain_Separation_16k/`：語音分離模型（Sepformer）。

- **modules/**  
  - `asr/`：語音辨識（ASR）相關模組。
    - `asr_model.py`：Faster-Whisper 載入與快取管理。
    - `text_utils.py`：ASR 文字處理輔助。
    - `whisper_asr.py`：ASR 主流程與 API。
  - `database/`：資料庫操作（如 Weaviate）。
    - `database.py`：資料庫連線與查詢。
  - `identification/`：說話人辨識。
    - `VID_identify_v5.py`：語者辨識主程式，聲紋比對與管理。
  - `management/`：語者與資料管理。
    - `manager.py`：語者資料管理輔助。
  - `separation/`：語者分離。
    - `separator.py`：語音分離主流程，呼叫 SpeechBrain。
  - `utils/`：工具模組。
    - `logger.py`：集中式日誌系統，支援多模組共用。

- **pipelines/**  
  - `orchestrator.py`：分離、辨識、ASR 整合主流程，供 API 或 CLI 呼叫。

- **services/**  
  - `api.py`：FastAPI 伺服器主程式，定義 RESTful 與 WebSocket 介面。

- **utils/**  
  其他輔助腳本（如日誌、同步工具）。

- **weaviate_study/**  
  Weaviate 向量資料庫整合、測試與工具腳本（如集合建立、資料匯入、查詢、刪除等）。

- **work_output/**  
  每次語音處理的結果資料夾，包含分離音檔與辨識結果（output.json）。

---

## ⚡️ 快速啟動

1. 安裝依賴
   ```cmd
   pip install -r requirements.txt
   ```

2. 啟動 FastAPI 伺服器
   ```cmd
   uvicorn services.api:app --reload
   ```
   - API 文件（Swagger UI）：http://localhost:8000/docs

3. 測試腳本
   ```cmd
   python examples/test_modules.py
   ```

---

## Docker 和 Weaviate 設定 (必要步驟)

1. 確保 Docker Desktop 已安裝並運行
2. 啟動 Weaviate 容器：
   ```cmd
   cd weaviate_study
   docker-compose up -d
   ```
3. 等待容器啟動完成（通常需要 30-60 秒）
4. 初始化 Weaviate 結構：
   ```cmd
   python weaviate_study\create_collections.py
   ```

> **注意**: 首次啟動後，必須執行 `create_collections.py` 來建立必要的資料集合。此步驟只需執行一次，除非您刪除了 Weaviate 容器或需要重置資料庫的所有資料。

---

## 其他說明

- **模型快取**：ASR 與分離模型會自動下載至 `models/` 相關資料夾。
- **日誌系統**：所有模組統一使用 `utils/logger.py`，支援多檔案與顏色輸出。
- **Weaviate**：如需向量資料庫功能，請先啟動 docker-compose 並建立集合。
