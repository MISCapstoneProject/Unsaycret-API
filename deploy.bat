@echo off
REM Unsaycret API Docker 部署腳本 (Windows 版本)
REM 使用方法: deploy.bat [build|start|stop|restart|logs|clean]

setlocal enabledelayedexpansion

set PROJECT_NAME=unsaycret-api
set IMAGE_NAME=unsaycret-api:latest

if "%1"=="" (
    call :print_usage
    exit /b 1
)

if "%1"=="build" (
    call :check_requirements
    call :build_image
) else if "%1"=="start" (
    call :check_requirements
    call :start_services
) else if "%1"=="stop" (
    call :stop_services
) else if "%1"=="restart" (
    call :restart_services
) else if "%1"=="logs" (
    call :show_logs
) else if "%1"=="clean" (
    call :clean_up
) else if "%1"=="status" (
    call :check_status
) else (
    call :print_usage
    exit /b 1
)

goto :eof

:print_usage
echo 使用方法: %0 [command]
echo.
echo 可用命令:
echo   build     - 建置 Docker 映像
echo   start     - 啟動所有服務
echo   stop      - 停止所有服務
echo   restart   - 重新啟動所有服務
echo   logs      - 查看服務日誌
echo   clean     - 清理映像和容器
echo   status    - 檢查服務狀態
echo.
goto :eof

:check_requirements
echo 🔍 檢查系統需求...

docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker 未安裝，請先安裝 Docker
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose 未安裝，請先安裝 Docker Compose
    exit /b 1
)

echo ✅ 系統需求檢查通過
goto :eof

:build_image
echo 🏗️  建置 Docker 映像...
docker build -t %IMAGE_NAME% .
echo ✅ 映像建置完成
goto :eof

:start_services
echo 🚀 啟動所有服務...

REM 啟動服務
docker-compose up -d

echo ✅ 服務啟動完成
echo.
echo 📋 服務端點:
echo   🔹 API 服務: http://localhost:8000
echo   🔹 API 文檔: http://localhost:8000/docs
echo   🔹 Weaviate: http://localhost:8010
echo   🔹 Weaviate 控制台: http://localhost:8081
echo.
echo 使用 '%0 logs' 查看服務日誌
goto :eof

:stop_services
echo 🛑 停止所有服務...
docker-compose down
echo ✅ 服務已停止
goto :eof

:restart_services
echo 🔄 重新啟動服務...
docker-compose restart
echo ✅ 服務重新啟動完成
goto :eof

:show_logs
echo 📋 顯示服務日誌...
docker-compose logs -f
goto :eof

:clean_up
echo 🧹 清理 Docker 資源...

REM 停止並移除容器
docker-compose down -v

REM 移除映像
docker rmi %IMAGE_NAME% 2>nul

REM 移除未使用的映像
docker image prune -f

echo ✅ 清理完成
goto :eof

:check_status
echo 📊 檢查服務狀態...
echo.

REM 檢查 Docker Compose 服務
docker-compose ps

echo.
echo 🔍 健康檢查:

REM 檢查 API 服務
curl -f -s http://localhost:8000/docs >nul 2>&1
if errorlevel 1 (
    echo ❌ API 服務無法訪問
) else (
    echo ✅ API 服務運行正常
)

REM 檢查 Weaviate
curl -f -s http://localhost:8010/v1/.well-known/ready >nul 2>&1
if errorlevel 1 (
    echo ❌ Weaviate 無法訪問
) else (
    echo ✅ Weaviate 運行正常
)

REM 檢查 Weaviate 控制台
curl -f -s http://localhost:8081 >nul 2>&1
if errorlevel 1 (
    echo ❌ Weaviate 控制台無法訪問
) else (
    echo ✅ Weaviate 控制台運行正常
)

goto :eof
