#!/bin/bash
# ============================================================
# 违章建筑检测系统 - 批量推理平台 一键启动脚本
# 执行流程：
#   1. 创建共享卷目录结构
#   2. 创建/检查 Gateway conda 环境并安装依赖
#   3. 构建 SAM3 和 DreamSim Worker Docker 镜像
#   4. 启动 PostgreSQL + 8 个 Worker 容器
#   5. 启动 Gateway FastAPI 服务
# ============================================================
set -e  # 任何命令失败立即退出

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 共享卷根目录（可通过环境变量覆盖）
SHARED_ROOT="${BATCH_SHARED_ROOT:-/data/batch_shared}"

echo "========================================"
echo " 违章建筑检测 - 批量推理平台"
echo "========================================"

# ============================================================
# 1. 创建共享目录
# 这些目录是 Gateway 和 Worker 容器之间的 IPC 通道
# ============================================================
echo "[1/5] Creating shared directories..."
# heartbeat: Worker 心跳文件
# jobs: 每个 Job 的任务描述符和结果文件
# scaling: AutoScaler 伸缩指令文件
# spillover: 结果文件溢出落盘区
# pgdata: PostgreSQL 数据持久化
sudo mkdir -p "$SHARED_ROOT"/{heartbeat,jobs,scaling,spillover,pgdata}
# 设置 777 权限，确保容器内进程可读写
sudo chmod -R 777 "$SHARED_ROOT"

# ============================================================
# 2. 检查 conda 环境
# Gateway 运行在宿主机的 conda 环境中，只需轻量 Web 依赖
# ============================================================
echo "[2/5] Checking Gateway conda environment..."
CONDA_ENV="batch_gateway"

# 如果 conda 环境不存在，创建 Python 3.10 环境
if ! conda info --envs | grep -q "$CONDA_ENV"; then
    echo "  Creating conda environment: $CONDA_ENV"
    conda create -n "$CONDA_ENV" python=3.10 -y
fi

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
# 安装 Gateway 依赖（fastapi, uvicorn, asyncpg 等）
pip install -r "$SCRIPT_DIR/requirements_gateway.txt" -q

echo "  Gateway environment ready"

# ============================================================
# 3. 构建 Worker 镜像
# 基于现有镜像 FROM 扩展，仅添加 worker/ 和 shared/ 代码 + psycopg2
# ============================================================
echo "[3/5] Building Worker Docker images..."
cd "$SCRIPT_DIR"

# 构建 SAM3 Worker 镜像（基于 sam3-sam3:latest）
echo "  Building SAM3 Worker image..."
docker build -t batch-sam3-worker -f worker/Dockerfile.sam3 .

# 构建 DreamSim Worker 镜像（基于 diffsim-diffsim-runner:latest）
echo "  Building DreamSim Worker image..."
docker build -t batch-ds-worker -f worker/Dockerfile.dreamsim .

# ============================================================
# 4. 启动 Docker 容器 (PostgreSQL + Workers)
# ============================================================
echo "[4/5] Starting Docker containers..."
# 以守护模式启动所有容器（PG + 4 SAM3 Worker + 4 DreamSim Worker）
docker compose up -d

# 等待 PostgreSQL 数据库就绪
echo "  Waiting for PostgreSQL to be ready..."
for i in $(seq 1 30); do
    if docker exec batch-postgres pg_isready -U batch -d batch_detection > /dev/null 2>&1; then
        echo "  PostgreSQL is ready"
        break
    fi
    sleep 1  # 每秒检查一次，最多等 30 秒
done

# ============================================================
# 5. 启动 Gateway
# Gateway 直接在宿主机运行，通过 TCP:5433 连接容器中的 PostgreSQL
# ============================================================
echo "[5/5] Starting Gateway..."
cd "$SCRIPT_DIR"
# 将项目根目录加入 PYTHONPATH，确保 import shared/gateway 能正常工作
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo ""
echo "========================================"
echo " System Ready!"
echo " Gateway: http://0.0.0.0:8090"
echo " API Docs: http://0.0.0.0:8090/docs"
echo "========================================"
echo ""

# 前台启动 Gateway（日志直接输出到终端）
# 后台运行请改用: nohup python -m gateway.app > /var/log/batch_gateway.log 2>&1 &
python -m gateway.app
