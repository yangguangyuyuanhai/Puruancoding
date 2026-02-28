# 环境构建与版本说明

---

## 一、宿主机环境

### 1.1 系统与驱动

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 22.04 LTS |
| Python (系统) | 3.13.9 |
| conda | 25.9.1 |
| Docker | 29.2.0 |
| Docker Compose | v5.0.2 |
| NVIDIA Driver | 580.95.05 |
| CUDA (驱动层) | 13.0 |

### 1.2 Gateway conda 环境

**环境名**：`batch_gateway`

**创建步骤**：

```bash
conda create -n batch_gateway python=3.10 -y
conda activate batch_gateway
pip install -r requirements_gateway.txt
```

**依赖清单** (`requirements_gateway.txt`)：

| 包 | 最低版本 | 说明 |
|----|----------|------|
| fastapi | >= 0.109.0 | HTTP API 框架 |
| uvicorn[standard] | >= 0.27.0 | ASGI 服务器 |
| asyncpg | >= 0.29.0 | PostgreSQL 异步驱动 (Gateway 用) |
| psycopg2-binary | >= 2.9.9 | PostgreSQL 同步驱动 (备用/脚本) |
| pydantic | >= 2.5.0 | 数据校验 |
| python-multipart | >= 0.0.6 | 文件上传支持 |

> Gateway 不依赖 GPU、PyTorch 或任何 AI 库，仅需轻量 Web 栈。

---

## 二、Docker 镜像

### 2.1 镜像清单

| 镜像 | Tag | 大小 | 用途 |
|------|-----|------|------|
| `batch-sam3-worker` | `verified` | 19 GB | SAM3 批量推理 Worker (已验证) |
| `sam3-sam3` | `latest` | 18.7 GB | SAM3 基础镜像 (只读，不可修改) |
| `diffsim-diffsim-runner` | `latest` | 18.3 GB | DreamSim 基础镜像 (只读，不可修改) |
| `postgres` | `15-alpine` | 274 MB | 任务数据库 |

### 2.2 SAM3 Worker 镜像

**基础镜像**：`sam3-sam3:latest`

**容器内环境**：

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 22.04.4 LTS |
| Python | 3.10.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA Toolkit (容器内) | 12.8 |
| sam3 | 0.1.0 (源码安装于 `/app/sam3_code`) |
| opencv-python-headless | 4.13.0.90 |
| numpy | 1.26.4 |
| psycopg2-binary | 2.9.11 |

**模型文件**（容器内 `/app/checkpoints/`）：

| 文件 | 大小 | 说明 |
|------|------|------|
| `sam3.pt` | ~3.3 GB | SAM3 分割模型权重 |
| `Zero-DCE.pth` | 313 KB | Zero-DCE 暗光增强权重 |
| `bpe_simple_vocab_16e6.txt.gz` | - | CLIP 文本编码词表 |

**构建方式**：

```bash
# 方式 A：从 Dockerfile 构建
docker build -t batch-sam3-worker -f worker/Dockerfile.sam3 .

# 方式 B：加载已验证的 tar 包
docker load -i batch-sam3-worker-verified.tar
# 加载后镜像名: batch-sam3-worker:verified
```

**Dockerfile.sam3 内容**：

```dockerfile
FROM sam3-sam3:latest
COPY worker/ /app/worker/
COPY shared/ /app/shared/
RUN pip install --no-cache-dir psycopg2-binary
WORKDIR /app
ENTRYPOINT ["python3", "-m", "worker.launcher"]
```

**附加操作**（已包含在 verified tar 中）：

```bash
# 复制批量推理代码
docker cp worker/ <容器>:/app/worker/
docker cp shared/ <容器>:/app/shared/

# 安装数据库驱动
docker exec <容器> pip install psycopg2-binary

# 复制 Zero-DCE 权重 (如未包含)
docker cp Zero-DCE.pth <容器>:/app/checkpoints/Zero-DCE.pth
```

### 2.3 DreamSim Worker 镜像

**基础镜像**：`diffsim-diffsim-runner:latest`

**容器内环境**：

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 22.04.4 LTS |
| Python | 3.10.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA Toolkit (容器内) | 12.8 |
| dreamsim | 0.2.1 |
| torchvision | 0.25.0+cu128 |
| timm | 1.0.24 |
| opencv-python | 4.13.0.92 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| matplotlib | 3.10.8 |

**模型文件**（容器内 `/app/models/`）：

| 文件 | 说明 |
|------|------|
| `ensemble_lora/` | DreamSim ensemble LoRA 权重 |
| `clip_vitb16_pretrain.pth.tar` | CLIP ViT-B/16 预训练 |
| `dino_vitb16_pretrain.pth` | DINO ViT-B/16 预训练 |
| `open_clip_vitb16_pretrain.pth.tar` | OpenCLIP ViT-B/16 预训练 |

**构建方式**：

```bash
# 从 Dockerfile 构建
docker build -t batch-ds-worker -f worker/Dockerfile.dreamsim .
```

**Dockerfile.dreamsim 内容**：

```dockerfile
FROM diffsim-diffsim-runner:latest
COPY worker/ /app/worker/
COPY shared/ /app/shared/
RUN pip install --no-cache-dir psycopg2-binary
WORKDIR /app
ENTRYPOINT ["python3", "-m", "worker.launcher"]
```

### 2.4 PostgreSQL

**镜像**：`postgres:15-alpine`

**配置**（通过 docker-compose.yml 环境变量设定）：

| 参数 | 值 |
|------|-----|
| 端口 | 5433 (宿主机映射，避开现有 5432) |
| 数据库名 | `batch_detection` |
| 用户名 | `batch` |
| 密码 | `batch_secret` |
| 数据持久化 | `/data/batch_shared/pgdata` |
| 初始化脚本 | `sql/init.sql` (自动建表) |

---

## 三、完整环境构建流程

### 3.1 前置检查

```bash
# 1. 确认 GPU 驱动
nvidia-smi

# 2. 确认 Docker + GPU 支持
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# 3. 确认基础镜像存在
docker images | grep -E "sam3-sam3|diffsim-diffsim-runner"
```

### 3.2 构建 Gateway 环境

```bash
cd ~/yolo919/puruan_2_28

# 创建 conda 环境
conda create -n batch_gateway python=3.10 -y
conda activate batch_gateway
pip install -r requirements_gateway.txt

# 验证
python -c "import fastapi, uvicorn, asyncpg; print('Gateway deps OK')"
```

### 3.3 构建 Worker 镜像

```bash
cd ~/yolo919/puruan_2_28

# SAM3 Worker
docker build -t batch-sam3-worker -f worker/Dockerfile.sam3 .

# DreamSim Worker
docker build -t batch-ds-worker -f worker/Dockerfile.dreamsim .

# 验证
docker images | grep batch
```

如有已验证的 tar 包，直接加载：

```bash
docker load -i batch-sam3-worker-verified.tar
# docker load -i batch-ds-worker-verified.tar  # DreamSim 同理
```

### 3.4 创建共享目录与启动

```bash
# 创建共享目录
sudo mkdir -p /data/batch_shared/{heartbeat,jobs,scaling,spillover,pgdata}
sudo chmod -R 777 /data/batch_shared

# 启动 PostgreSQL + Worker 容器
docker compose up -d

# 启动 Gateway
export PYTHONPATH="$(pwd):$PYTHONPATH"
conda activate batch_gateway
python -m gateway.app
```

---

## 四、数据库迁移（从 v1.0 升级）

v1.1 新增了 SAM3→DreamSim 流水线 (Pipeline) 功能，需要在 `jobs` 表中添加两个字段。

**全新部署**无需额外操作，`sql/init.sql` 已包含所有字段。

**从 v1.0 升级**需手动执行以下 SQL：

```bash
docker exec -i batch-postgres psql -U batch -d batch_detection <<'EOF'
-- 添加流水线父任务 ID（子任务才有值，父任务删除时级联删除子任务）
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS parent_job_id UUID REFERENCES jobs(id) ON DELETE CASCADE;
-- 添加流水线阶段标识：sam3_before / sam3_after / dreamsim
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS pipeline_phase VARCHAR(20);
-- 添加流水线子任务查询索引
CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_job_id);
EOF
```

**验证：**

```bash
docker exec batch-postgres psql -U batch -d batch_detection \
  -c "\d jobs" | grep -E "parent_job_id|pipeline_phase"
# 预期输出：
#  parent_job_id  | uuid                     |           |          |
#  pipeline_phase | character varying(20)    |           |          |
```

---

## 五、版本兼容性说明

### CUDA 版本链路

```
宿主机驱动 (580.95 / CUDA 13.0)
  └─ 容器运行时 (NVIDIA Container Toolkit)
       └─ 容器内 CUDA Toolkit 12.8
            └─ PyTorch 2.10.0+cu128
```

> 宿主机驱动版本向下兼容容器内的 CUDA Toolkit 版本，只要驱动 CUDA >= 容器 CUDA 即可。

### Python 版本

| 环境 | Python 版本 | 说明 |
|------|-------------|------|
| 宿主机系统 | 3.13.9 | 不直接使用 |
| Gateway conda | 3.10.x | FastAPI + asyncpg |
| SAM3 容器 | 3.10.12 | 与 sam3 包兼容 |
| DreamSim 容器 | 3.10.12 | 与 dreamsim 包兼容 |

### PyTorch 版本

两个 Worker 容器统一使用 **PyTorch 2.10.0+cu128**，确保推理行为一致。

### 关键约束

- 基础镜像 `sam3-sam3:latest` 和 `diffsim-diffsim-runner:latest` 为生产参考环境，**严禁修改**
- Worker 镜像通过 `FROM` 扩展构建，仅添加批量推理代码和 psycopg2 驱动
- Gateway 与 Worker 通过 PostgreSQL (TCP:5433) + 共享卷 (`/data/batch_shared/`) 通信，无直接依赖
