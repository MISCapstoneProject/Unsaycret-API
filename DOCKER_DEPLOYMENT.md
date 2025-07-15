# Unsaycret API Docker 部署指南

## 📋 概述

本指南說明如何使用 Docker 部署 Unsaycret API 語音處理服務。

## 🔧 系統需求

- **Docker**: 20.10 或更高版本
- **Docker Compose**: 2.0 或更高版本
- **記憶體**: 至少 4GB RAM
- **硬碟空間**: 15GB 可用空間
- **作業系統**: Windows 10/11, macOS, Linux

## 🚀 快速開始

### 方案一：使用部署腳本

```bash
# Linux/macOS
./deploy.sh build
./deploy.sh start

# Windows
deploy.bat build
deploy.bat start
```

### 方案二：使用原生 Docker 命令

```bash
# 建置映像
docker build -t unsaycret-api:latest .

# 啟動服務
docker-compose up -d

# 檢查狀態
docker-compose ps
```

> 💡 **提示**: 兩種方案功能相同，部署腳本提供額外的系統檢查和健康測試。

### 驗證服務狀態

```bash
# 使用部署腳本
# Linux/macOS
./deploy.sh status

# Windows
deploy.bat status

# 或使用原生 Docker 命令
docker-compose ps
curl http://localhost:8000/docs
curl http://localhost:8200/v1/.well-known/ready
```

### 存取服務

- **API 服務**: http://localhost:18000
- **API 文檔**: http://localhost:18000/docs
- **Weaviate 資料庫**: http://localhost:8200
- **Weaviate 控制台**: http://localhost:8081

> 📂 **目錄自動建立**: Docker 會自動建立所需的目錄（`models/`, `work_output/`, `logs/` 等），無需手動建立。

## 📦 服務架構

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Unsaycret API     │    │     Weaviate DB     │    │  Weaviate Console   │
│   (語音處理服務)      │◄──►│   (向量資料庫)       │◄──►│   (管理界面)        │
│   Port: 18000       │    │   Port: 8200        │    │   Port: 8081        │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 🔧 環境變數配置

您可以透過環境變數自定義服務配置：

```yaml
environment:
  - WEAVIATE_HOST=weaviate     # Weaviate 主機
  - WEAVIATE_PORT=8080         # Weaviate 端口
  - WEAVIATE_SCHEME=http       # Weaviate 協議
  - FASTAPI_HOST=0.0.0.0       # API 服務主機
  - FASTAPI_PORT=8000          # API 服務端口
  - LOG_LEVEL=INFO             # 日誌級別
  - MAX_WORKERS=3              # 最大工作執行緒數
  - GPU_ENABLED=false          # 是否啟用 GPU
```

## 📂 目錄結構

```
unsaycret-api/
├── models/                  # 模型檔案 (自動建立)
├── work_output/             # 工作輸出 (自動建立)
├── stream_output/           # 串流輸出 (自動建立)
├── embeddingFiles/          # 嵌入向量檔案 (自動建立)
├── logs/                    # 日誌檔案 (自動建立)
├── 16K-model/               # 16K 模型資料 (自動建立)
├── Dockerfile               # Docker 映像定義
├── docker-compose.yml       # 服務編排配置
├── .dockerignore            # Docker 忽略檔案
├── deploy.sh                # Linux/macOS 部署腳本
└── deploy.bat               # Windows 部署腳本
```

> 📝 **注意**: 標記為「自動建立」的目錄會在 Docker 啟動時自動建立，無需手動建立。

## 🤖 Docker 自動化機制

### 目錄自動建立原理

Docker 會在以下階段自動建立目錄：

1. **Dockerfile 階段**: 在映像建置時建立容器內目錄
   ```dockerfile
   RUN mkdir -p /app/models /app/work_output /app/stream_output
   ```

2. **Volume 掛載階段**: 在服務啟動時建立主機目錄
   ```yaml
   volumes:
     - ./models:/app/models        # 自動建立 ./models
     - ./work_output:/app/work_output  # 自動建立 ./work_output
   ```

3. **應用程式階段**: 在程式執行時建立缺少的目錄
   ```python
   # 在 docker_config.py 中
   def ensure_directories():
       directory.mkdir(parents=True, exist_ok=True)
   ```

### 簡化的部署流程

由於 Docker 的自動化機制，您的部署流程可以簡化為：

```bash
# 最簡單的部署方式
docker build -t unsaycret-api:latest .
docker-compose up -d

# 使用部署腳本（額外提供系統檢查和健康測試）
./deploy.sh build
./deploy.sh start
```

兩種方式都會自動處理目錄建立，您無需手動建立任何目錄。

## 🛠️ 常用命令

### 部署腳本命令

```bash
# 建置映像
./deploy.sh build    # 等同於: docker build -t unsaycret-api:latest .

# 啟動服務
./deploy.sh start    # 等同於: docker-compose up -d

# 停止服務
./deploy.sh stop     # 等同於: docker-compose down

# 重新啟動服務
./deploy.sh restart  # 等同於: docker-compose restart

# 查看日誌
./deploy.sh logs     # 等同於: docker-compose logs -f

# 檢查狀態
./deploy.sh status   # 包含健康檢查 + docker-compose ps

# 清理資源
./deploy.sh clean    # 包含完整的清理流程
```

### 原生 Docker 命令

```bash
# 基本操作
docker build -t unsaycret-api:latest .
docker-compose up -d
docker-compose down
docker-compose restart
docker-compose logs -f
docker-compose ps

# 進入容器
docker-compose exec unsaycret-api bash

# 重新建置特定服務
docker-compose build unsaycret-api
```

### Docker Compose 命令

```bash
# 檢視服務狀態
docker-compose ps

# 檢視日誌
docker-compose logs -f [service_name]

# 重新建置特定服務
docker-compose build unsaycret-api

# 重新啟動特定服務
docker-compose restart unsaycret-api

# 進入容器
docker-compose exec unsaycret-api bash
```

## 🐛 故障排除

### 0. 部署腳本 vs 原生命令

如果部署腳本出現問題，您可以使用原生 Docker 命令：

```bash
# 部署腳本失敗時的替代方案
# 替代 ./deploy.sh build
docker build -t unsaycret-api:latest .

# 替代 ./deploy.sh start
docker-compose up -d

# 替代 ./deploy.sh status
docker-compose ps
curl http://localhost:8000/docs
curl http://localhost:8200/v1/.well-known/ready

# 替代 ./deploy.sh stop
docker-compose down
```

### 1. 服務無法啟動

```bash
# 檢查日誌
docker-compose logs unsaycret-api

# 檢查 Weaviate 連線
curl http://localhost:8200/v1/.well-known/ready
```

### 2. 記憶體不足

```bash
# 檢查 Docker 資源使用
docker stats

# 清理未使用的映像
docker image prune -f
```

### 3. 端口衝突

修改 `docker-compose.yml` 中的端口映射：

```yaml
services:
  unsaycret-api:
    ports:
      - "8001:8000"  # 將本機端口改為 8001
```

### 4. 權限問題

```bash
# Linux/macOS 給予腳本執行權限
chmod +x deploy.sh

# 檢查目錄權限
ls -la models/ work_output/ stream_output/
```

## 📊 監控與日誌

### 日誌位置

- **容器日誌**: `docker-compose logs -f`
- **應用日誌**: `./logs/system_output.log`
- **Weaviate 日誌**: `docker-compose logs -f weaviate`

### 健康檢查

```bash
# API 服務健康檢查
curl http://localhost:8000/docs

# Weaviate 健康檢查
curl http://localhost:8200/v1/.well-known/ready
```

## 🔐 安全性考量

### 生產環境建議

1. **修改預設端口**
2. **啟用 HTTPS**
3. **設定防火牆規則**
4. **定期備份數據**
5. **監控資源使用**

### 網路安全

```yaml
# 僅允許內部網路存取
services:
  weaviate:
    networks:
      - internal
networks:
  internal:
    driver: bridge
```

## 🚀 效能調優

### 資源配置

```yaml
services:
  unsaycret-api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'
```

### GPU 支援

```yaml
services:
  unsaycret-api:
    environment:
      - GPU_ENABLED=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 📝 API 使用範例

### 語音轉錄

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav"
```

### 語者驗證

```bash
curl -X POST "http://localhost:8000/speaker/verify" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@speaker.wav"
```

## 📚 相關文檔

- [API 文檔](http://localhost:8000/docs)
- [Weaviate 文檔](https://weaviate.io/developers/weaviate)
- [Docker 文檔](https://docs.docker.com/)

## 🆘 支援

如果遇到問題，請檢查：

1. **系統需求**是否滿足
2. **端口**是否被佔用
3. **日誌檔案**中的錯誤訊息
4. **Docker 服務**是否正常運行

---

**注意**: 首次啟動可能需要下載模型檔案，請耐心等待。
