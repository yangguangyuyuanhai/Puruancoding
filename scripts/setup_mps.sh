#!/bin/bash
# ============================================================
# NVIDIA MPS (Multi-Process Service) 配置脚本
# MPS 允许同一张 GPU 上的多个 CUDA 进程共享 GPU 上下文，
# 实现真正的并行执行（而非默认的时间片轮转）
# 适用于多个 Worker 进程共享一张 GPU 的场景
# 用法: sudo bash setup_mps.sh start|stop
# ============================================================
set -e  # 任何命令失败立即退出

# 读取命令行参数，默认为 start
ACTION="${1:-start}"

if [ "$ACTION" = "start" ]; then
    echo "[MPS] Starting NVIDIA MPS daemon..."

    # 设置 MPS 通信管道和日志的存储目录
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

    # 检查 nvidia-cuda-mps-control 命令是否可用
    if ! command -v nvidia-cuda-mps-control &> /dev/null; then
        echo "[MPS] nvidia-cuda-mps-control not found, skipping MPS setup"
        exit 0
    fi

    # 停止已有的 MPS 守护进程（如果存在），避免冲突
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 1

    # 将所有 GPU 设置为默认计算模式（MPS 要求非独占模式）
    for gpu_id in $(seq 0 7); do
        nvidia-smi -i $gpu_id -c DEFAULT 2>/dev/null || true
    done

    # 以守护进程模式启动 MPS 控制器
    nvidia-cuda-mps-control -d
    echo "[MPS] MPS daemon started"

    # 等待 1 秒后验证 MPS 是否正常运行
    sleep 1
    if nvidia-cuda-mps-control -h &> /dev/null; then
        echo "[MPS] MPS is running"
    else
        echo "[MPS] Warning: MPS may not be running correctly"
    fi

elif [ "$ACTION" = "stop" ]; then
    # 向 MPS 控制器发送 quit 命令，停止 MPS 守护进程
    echo "[MPS] Stopping NVIDIA MPS daemon..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    echo "[MPS] MPS daemon stopped"

else
    # 无效参数提示
    echo "Usage: $0 {start|stop}"
    exit 1
fi
