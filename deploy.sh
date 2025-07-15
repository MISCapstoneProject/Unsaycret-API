#!/bin/bash

# Unsaycret API Docker 部署腳本
# 使用方法: ./deploy.sh [build|start|stop|restart|logs|clean]

set -e

PROJECT_NAME="unsaycret-api"
IMAGE_NAME="unsaycret-api:latest"

print_usage() {
    echo "使用方法: $0 [command]"
    echo ""
    echo "可用命令:"
    echo "  build     - 建置 Docker 映像"
    echo "  start     - 啟動所有服務"
    echo "  stop      - 停止所有服務"
    echo "  restart   - 重新啟動所有服務"
    echo "  logs      - 查看服務日誌"
    echo "  clean     - 清理映像和容器"
    echo "  status    - 檢查服務狀態"
    echo ""
}

check_requirements() {
    echo "🔍 檢查系統需求..."
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker 未安裝，請先安裝 Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose 未安裝，請先安裝 Docker Compose"
        exit 1
    fi
    
    echo "✅ 系統需求檢查通過"
}

build_image() {
    echo "🏗️  建置 Docker 映像..."
    docker build -t $IMAGE_NAME .
    echo "✅ 映像建置完成"
}

start_services() {
    echo "🚀 啟動所有服務..."
    
    # 啟動服務
    docker-compose up -d
    
    echo "✅ 服務啟動完成"
    echo ""
    echo "📋 服務端點:"
    echo "  🔹 API 服務: http://localhost:8000"
    echo "  🔹 API 文檔: http://localhost:8000/docs"
    echo "  🔹 Weaviate: http://localhost:8010"
    echo "  🔹 Weaviate 控制台: http://localhost:8081"
    echo ""
    echo "使用 '$0 logs' 查看服務日誌"
}

stop_services() {
    echo "🛑 停止所有服務..."
    docker-compose down
    echo "✅ 服務已停止"
}

restart_services() {
    echo "🔄 重新啟動服務..."
    docker-compose restart
    echo "✅ 服務重新啟動完成"
}

show_logs() {
    echo "📋 顯示服務日誌..."
    docker-compose logs -f
}

clean_up() {
    echo "🧹 清理 Docker 資源..."
    
    # 停止並移除容器
    docker-compose down -v
    
    # 移除映像
    docker rmi $IMAGE_NAME 2>/dev/null || true
    
    # 移除未使用的映像
    docker image prune -f
    
    echo "✅ 清理完成"
}

check_status() {
    echo "📊 檢查服務狀態..."
    echo ""
    
    # 檢查 Docker Compose 服務
    docker-compose ps
    
    echo ""
    echo "🔍 健康檢查:"
    
    # 檢查 API 服務
    if curl -f -s http://localhost:8000/docs > /dev/null 2>&1; then
        echo "✅ API 服務運行正常"
    else
        echo "❌ API 服務無法訪問"
    fi
    
    # 檢查 Weaviate
    if curl -f -s http://localhost:8010/v1/.well-known/ready > /dev/null 2>&1; then
        echo "✅ Weaviate 運行正常"
    else
        echo "❌ Weaviate 無法訪問"
    fi
    
    # 檢查 Weaviate 控制台
    if curl -f -s http://localhost:8081 > /dev/null 2>&1; then
        echo "✅ Weaviate 控制台運行正常"
    else
        echo "❌ Weaviate 控制台無法訪問"
    fi
}

# 主程序
main() {
    case "${1:-}" in
        build)
            check_requirements
            build_image
            ;;
        start)
            check_requirements
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs
            ;;
        clean)
            clean_up
            ;;
        status)
            check_status
            ;;
        *)
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
