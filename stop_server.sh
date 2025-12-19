#!/bin/bash
# FastAPI 服务停止脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/api_server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "未找到 PID 文件，服务可能未运行"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "进程 $PID 不存在，清理 PID 文件"
    rm -f "$PID_FILE"
    exit 0
fi

echo "正在停止服务 (PID: $PID)..."

# 获取所有相关进程（包括主进程和worker进程）
# uvicorn 会创建多个进程，需要停止整个进程组
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
if [ -n "$PGID" ]; then
    # 停止整个进程组
    kill -TERM -"$PGID" 2>/dev/null
    sleep 2
    
    # 如果还在运行，强制杀死
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "强制停止进程..."
        kill -KILL -"$PGID" 2>/dev/null
    fi
else
    # 备用方案：直接停止主进程
    kill -TERM "$PID" 2>/dev/null
    sleep 2
    if ps -p "$PID" > /dev/null 2>&1; then
        kill -KILL "$PID" 2>/dev/null
    fi
fi

# 清理 PID 文件
rm -f "$PID_FILE"

# 检查是否还有相关进程
REMAINING=$(ps aux | grep "[a]pi_server.py" | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "警告: 可能还有相关进程在运行，请手动检查"
    ps aux | grep "[a]pi_server.py"
else
    echo "服务已停止"
fi

