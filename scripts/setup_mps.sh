#!/bin/bash
# ============================================================
# NVIDIA MPS (Multi-Process Service) 配置脚本
# 启用 MPS 后，同一张 GPU 上的多个 Worker 进程可以真正并行执行
# ============================================================
set -e

ACTION="${1:-start}"

if [ "$ACTION" = "start" ]; then
    echo "[MPS] Starting NVIDIA MPS daemon..."

    # 设置 MPS 管道目录
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

    # 确保 nvidia-cuda-mps-control 存在
    if ! command -v nvidia-cuda-mps-control &> /dev/null; then
        echo "[MPS] nvidia-cuda-mps-control not found, skipping MPS setup"
        exit 0
    fi

    # 停止现有 MPS (如果有)
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 1

    # 设置所有 GPU 为默认计算模式
    for gpu_id in $(seq 0 7); do
        nvidia-smi -i $gpu_id -c DEFAULT 2>/dev/null || true
    done

    # 启动 MPS control daemon
    nvidia-cuda-mps-control -d
    echo "[MPS] MPS daemon started"

    # 验证
    sleep 1
    if nvidia-cuda-mps-control -h &> /dev/null; then
        echo "[MPS] MPS is running"
    else
        echo "[MPS] Warning: MPS may not be running correctly"
    fi

elif [ "$ACTION" = "stop" ]; then
    echo "[MPS] Stopping NVIDIA MPS daemon..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    echo "[MPS] MPS daemon stopped"

else
    echo "Usage: $0 {start|stop}"
    exit 1
fi
