# Unsaycret-API

## 專案簡介
Unsaycret-API 是一套模組化的語音處理系統，整合語音分離、說者辨識、語音辨識、API 服務，並支援向量資料庫串接。
這專案執行後會架起 API 伺服器，以供呼叫使用。


## 🚀 主要功能

- 🎙 **語者分離**：採用 SpeechBrain Sepformer，支援雙人語音分離。
- 🧠 **語音辨識（ASR）**：Faster-Whisper，支援 GPU/CPU 動態切換，逐詞時間戳與信心值。
- 🗣 **說話人辨識**：ECAPA-TDNN 語者聲紋比對，支援聲紋自動更新。
- 🛜 **API 服務**：FastAPI 提供 RESTful 與 WebSocket 介面。
- 🧠 **Weaviate 整合**：語音向量與辨識結果可存入 Weaviate，支援語者搜尋與比對。


## 📂 目錄結構與模組說明

```
Unsaycret-API/
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
    - `VID_identify_v5.py`：語者辨識主程式，聲紋比對與管理。  - `management/`：語者與資料管理。
    - `VID_manager.py`：語者資料管理輔助。
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


## ⚡️ 快速啟動

### 1. 安裝依賴
```cmd
pip install -r requirements.txt
```

### 2. 啟動 Weaviate 資料庫
```cmd
docker-compose up -d
```

### 3. 初始化資料結構
```cmd
python weaviate_study/create_collections.py
```

### 4. 啟動 FastAPI 伺服器

您可以選擇以下任一方式啟動API服務：

```cmd
# 方法一：使用 main.py 啟動（推薦）
python main.py

# 方法二：使用 uvicorn 命令啟動
uvicorn services.api:app --reload

# 方法三：指定主機和端口
uvicorn services.api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 訪問API服務

啟動成功後，您可以通過以下方式訪問API：

- **API 互動式文檔（Swagger UI）**：http://localhost:8000/docs
- **API 文檔（ReDoc）**：http://localhost:8000/redoc
- **API 基礎URL**：http://localhost:8000


## 🐳 Docker & Weaviate 部署指南 （2025‑06-16 更新）(必要步驟)

### 1 啟動 Weaviate + Console

```bash
# 於專案根目錄
docker compose up -d
```

> 第一次啟動需 30–60 秒；執行 `docker compose ps`，待 `STATUS` 顯示 **Up (healthy)** 即完成。


### 2 初始化資料結構（**只需一次**）

```bash
python weaviate_study/create_collections.py
```

將為語者資訊、向量索引等建立必要 Class；除非想重置資料庫全部，否則不必重跑。


### 📦 資料庫備份、還原與轉移功能

已實現資料庫的資料進行備份、還原與轉移功能，方便進行資料保護與環境遷移。詳細操作步驟與指令將於後續文件補充。

## 系統需求

- **Python 版本**：Python 3.8+ (建議使用 Python 3.10)
- **作業系統**：Windows 10/11, Linux, macOS
- **硬體需求**：
  - 最低：4GB RAM，4核心 CPU
  - 建議：8GB+ RAM，GPU 加速(CUDA 支援)以提高處理速度
- **依賴套件**：請參閱 `requirements.txt`

## 🔌 API 使用指南

### 完整API文檔

詳細的API使用說明、請求格式、回應範例請參考：
**📖 [API_DOCUMENTATION.md](API_DOCUMENTATION.md)**

### 主要API端點

1. **語音轉錄**: `POST /transcribe` - 上傳音訊檔案進行語音分離、說話者識別與轉錄
2. **說話者改名**: `POST /speaker/rename` - 更改說話者名稱
3. **聲紋轉移**: `POST /speaker/transfer` - 合併錯誤識別的說話者
4. **說話者查詢**: `GET /speaker/{speaker_id}` - 獲取說話者詳細資訊

## 其他說明

- **模型快取**：ASR 與分離模型會自動下載至 `models/` 相關資料夾。
- **日誌系統**：所有模組統一使用 `utils/logger.py`，支援多檔案與顏色輸出。
- **Weaviate**：如需向量資料庫功能，請先啟動 docker-compose 並建立集合。
- **聲紋管理**：系統會自動儲存語者聲紋，可用於後續辨識與比對。
- **輸出目錄**：處理結果會儲存在 `work_output/<日期時間>/` 目錄下。

## 系統工作流程

```
           ┌───────────────┐
           │ 輸入音訊檔案  │
           └───────┬───────┘
                   ▼
        ┌──────────────────────┐
        │ 語者分離 (Separator) │
        └──────────┬───────────┘
                   │
┌──────────────────┴────────────────────┐
▼                                       ▼
┌────────────────────────┐    ┌──────────────────────┐
│ 聲紋辨識 (Identifier)  │    │ 語音辨識 (ASR)       │
└───────────┬────────────┘    └──────────┬───────────┘
            │                            │
            └────────────┬───────────────┘
                         ▼
              ┌────────────────────┐
              │ 整合結果 (Output)  │
              └────────────────────┘
```
