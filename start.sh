#!/bin/bash
# ============================================================
# 违章建筑检测系统 - 批量推理平台 一键启动脚本
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_ROOT="${BATCH_SHARED_ROOT:-/data/batch_shared}"

echo "========================================"
echo " 违章建筑检测 - 批量推理平台"
echo "========================================"

# ============================================================
# 1. 创建共享目录
# ============================================================
echo "[1/5] Creating shared directories..."
sudo mkdir -p "$SHARED_ROOT"/{heartbeat,jobs,scaling,spillover,pgdata}
sudo chmod -R 777 "$SHARED_ROOT"

# ============================================================
# 2. 检查 conda 环境
# ============================================================
echo "[2/5] Checking Gateway conda environment..."
CONDA_ENV="batch_gateway"

if ! conda info --envs | grep -q "$CONDA_ENV"; then
    echo "  Creating conda environment: $CONDA_ENV"
    conda create -n "$CONDA_ENV" python=3.10 -y
fi

# 激活环境并安装依赖
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
pip install -r "$SCRIPT_DIR/requirements_gateway.txt" -q

echo "  Gateway environment ready"

# ============================================================
# 3. 构建 Worker 镜像
# ============================================================
echo "[3/5] Building Worker Docker images..."
cd "$SCRIPT_DIR"

echo "  Building SAM3 Worker image..."
docker build -t batch-sam3-worker -f worker/Dockerfile.sam3 .

echo "  Building DreamSim Worker image..."
docker build -t batch-ds-worker -f worker/Dockerfile.dreamsim .

# ============================================================
# 4. 启动 Docker 容器 (PostgreSQL + Workers)
# ============================================================
echo "[4/5] Starting Docker containers..."
docker compose up -d

# 等待 PostgreSQL 就绪
echo "  Waiting for PostgreSQL to be ready..."
for i in $(seq 1 30); do
    if docker exec batch-postgres pg_isready -U batch -d batch_detection > /dev/null 2>&1; then
        echo "  PostgreSQL is ready"
        break
    fi
    sleep 1
done

# ============================================================
# 5. 启动 Gateway
# ============================================================
echo "[5/5] Starting Gateway..."
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo ""
echo "========================================"
echo " System Ready!"
echo " Gateway: http://0.0.0.0:8090"
echo " API Docs: http://0.0.0.0:8090/docs"
echo "========================================"
echo ""

python -m gateway.app
