# syntax=docker/dockerfile:1.7   # 告訴 Docker 使用 BuildKit 前端，才能用 --mount=type=cache

########################  Builder stage  ########################
# ✅ 編譯階段仍用 slim，體積小
FROM python:3.11-slim AS builder          

# 只放安裝出的 site-packages
WORKDIR /install                          

# 關閉互動提示，避免 build 卡住
ENV DEBIAN_FRONTEND=noninteractive        

# 只安裝「編譯期」才需要的工具與開發版頭檔
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential python3-dev \
      libsndfile1-dev portaudio19-dev \  
   && apt-get clean && rm -rf /var/lib/apt/lists/*

# 複製 requirements 清單
COPY requirements-base.txt requirements-cpu.txt ./

# 使用 BuildKit 快取 pip wheel，加速二次建置
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir \
        -r requirements-base.txt -r requirements-cpu.txt

########################  Runtime stage  ########################
FROM python:3.11-slim                       

# 只安裝「執行期」需要的共享函式庫（無 compiler）
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 libasound2 libportaudio2 \  
   && apt-get clean && rm -rf /var/lib/apt/lists/*

# 常用環境變數
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Taipei \
    PYTHONPATH=/app

WORKDIR /app

# 從 builder 複製安裝好的 site-packages（/install/usr/local/** → /usr/local/**）
COPY --from=builder /install /usr/local

# 複製專案程式碼（會受 .dockerignore 過濾）
COPY . .

EXPOSE 18000                             

# 健康檢查：打輕量 /health (或你的現有端點)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:18000/docs || exit 1

# 啟動命令：若 main.py 已內含 uvicorn 啟動邏輯，可以保留
CMD ["python", "main.py"]                  # 🔧 若要更佳效能，可改用 uvicorn CLI
