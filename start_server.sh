#!/bin/bash
# FastAPI 服务启动脚本（后台运行，支持2路并发）

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "已激活虚拟环境: $(which python)"
else
    echo "警告: 未找到虚拟环境 .venv，使用系统 Python"
fi

# 设置日志文件
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/api_server.log"
PID_FILE="$SCRIPT_DIR/api_server.pid"

# 检查服务是否已在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "服务已在运行中 (PID: $OLD_PID)"
        echo "如需重启，请先运行: ./stop_server.sh"
        exit 1
    else
        echo "清理旧的 PID 文件"
        rm -f "$PID_FILE"
    fi
fi

# 启动服务（生产模式，2个worker）
echo "正在启动 FastAPI 服务（2个worker进程）..."
nohup python api_server.py prod > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# 保存 PID
echo $SERVER_PID > "$PID_FILE"
echo "服务已启动，PID: $SERVER_PID"
echo "日志文件: $LOG_FILE"
echo "PID 文件: $PID_FILE"
echo ""
echo "查看日志: tail -f $LOG_FILE"
echo "停止服务: ./stop_server.sh"

