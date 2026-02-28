# 违章建筑检测系统 — 批量推理平台

# 使用与部署文档 v1.1

---

## 目录

- [一、产品概述（产品经理视角）](#一产品概述)
  - [1.1 系统定位](#11-系统定位)
  - [1.2 核心能力](#12-核心能力)
  - [1.3 使用场景](#13-使用场景)
  - [1.4 操作流程](#14-操作流程)
  - [1.5 性能指标](#15-性能指标)
- [二、系统架构（技术经理视角）](#二系统架构)
  - [2.1 架构总览](#21-架构总览)
  - [2.2 项目结构](#22-项目结构)
  - [2.3 分层设计](#23-分层设计)
  - [2.4 数据库设计](#24-数据库设计)
  - [2.5 数据流](#25-数据流)
  - [2.6 弹性伸缩策略](#26-弹性伸缩策略)
  - [2.7 容错机制](#27-容错机制)
  - [2.8 扩展新模型指南](#28-扩展新模型指南)
- [三、部署指南（运维视角）](#三部署指南)
  - [3.1 环境要求](#31-环境要求)
  - [3.2 一键部署](#32-一键部署)
  - [3.3 手动分步部署](#33-手动分步部署)
  - [3.4 环境变量参考](#34-环境变量参考)
  - [3.5 GPU 分配策略](#35-gpu-分配策略)
  - [3.6 Docker 容器清单](#36-docker-容器清单)
- [四、API 使用手册](#四api-使用手册)
  - [4.1 提交 SAM3 批量任务](#41-提交-sam3-批量任务)
  - [4.2 提交 DreamSim 批量任务](#42-提交-dreamsim-批量任务)
  - [4.3 提交 Pipeline 流水线任务](#43-提交-pipeline-流水线任务)
  - [4.4 查询任务进度](#44-查询任务进度)
  - [4.5 列出所有任务](#45-列出所有任务)
  - [4.6 取消任务](#46-取消任务)
  - [4.7 重试失败任务](#47-重试失败任务)
  - [4.8 查看失败详情](#48-查看失败详情)
  - [4.9 导出结果](#49-导出结果)
  - [4.10 系统健康检查](#410-系统健康检查)
- [五、运维手册](#五运维手册)
  - [5.1 日常巡检](#51-日常巡检)
  - [5.2 日志查看](#52-日志查看)
  - [5.3 常见故障排查](#53-常见故障排查)
  - [5.4 数据库维护](#54-数据库维护)
  - [5.5 磁盘空间管理](#55-磁盘空间管理)
  - [5.6 性能调优](#56-性能调优)
  - [5.7 备份与恢复](#57-备份与恢复)
  - [5.8 停机与重启](#58-停机与重启)
- [六、附录](#六附录)
  - [6.1 共享卷目录结构](#61-共享卷目录结构)
  - [6.2 全量环境变量速查表](#62-全量环境变量速查表)
  - [6.3 端口占用清单](#63-端口占用清单)

---

# 一、产品概述

## 1.1 系统定位

本系统是面向城市管理/规划部门的**无人机航拍图片批量违建检测平台**。操作员将无人机采集的大批量航拍图片提交给系统，系统自动调度 8 张 NVIDIA L20 GPU 并行处理，实现从「人工逐张操作」到「一键批量出结果」的升级。

**解决的核心痛点：**

| 痛点 | 旧方案 | 新方案 |
|------|--------|--------|
| 无批量能力 | Streamlit UI 每次 1-2 张 | 一次提交数千张，自动处理 |
| GPU 浪费 | 8 张 L20 只用 1 张 (12.5%) | 8 卡全部参与，利用率 > 80% |
| 容器通信慢 | 每张图 docker exec 启停 | 常驻 Worker 进程，零启停开销 |
| 无任务管理 | 中途失败从头来过 | 失败自动标记，支持断点重试 |

## 1.2 核心能力

### 场景一：SAM3 分割检测

- **用途**：从航拍图中分割出指定目标（如建筑物、违章搭建）
- **输入**：航拍图片目录或 ZIP + 文本提示词（如 "building"）
- **输出**：黑底抠像结果图（只保留检测到的目标区域）
- **增强**：内置 Zero-DCE 暗光增强，暗光场景也能准确检测

### 场景二：DreamSim 变化检测

- **用途**：对比前后两个时期的航拍图，检测建筑变化
- **输入**：包含 `before/` 和 `after/` 子目录的图片目录，同名文件自动配对
- **输出**：热力图版（叠加热力图+红框标注）+ 纯净版（原图+红框标注）

### 场景三：SAM3→DreamSim 流水线 (Pipeline)

- **用途**：一键完成「分割抠像 + 变化检测」全流程
- **输入**：包含 `before/` 和 `after/` 子目录的图片目录 + 文本提示词
- **流程**：系统自动先对两批图片分别跑 SAM3 分割，再将 SAM3 结果按文件名配对后跑 DreamSim 变化检测
- **输出**：DreamSim 热力图（基于 SAM3 抠像结果的变化对比）
- **优势**：用户无需手动衔接两个阶段，全程一键触发，Janitor 自动推进流水线

## 1.3 使用场景

| 场景 | 模型 | 典型规模 | 预计耗时(8卡) |
|------|------|----------|---------------|
| 园区违建普查 | SAM3 | 500-2000 张 | 5-20 分钟 |
| 季度变化比对 | DreamSim | 200-1000 对 | 10-30 分钟 |
| 紧急拆违取证 | SAM3 | 50-100 张 | < 2 分钟 |
| 全流程违建检测 | Pipeline | 200-500 对 | 15-40 分钟 (SAM3+DreamSim) |

## 1.4 操作流程

```
操作员                          系统
  │                              │
  │  1. 将航拍图片上传到服务器目录  │
  │─────────────────────────────>│
  │                              │
  │  2. 调用 API 提交批量任务      │
  │     POST /api/jobs           │
  │─────────────────────────────>│
  │                              │  自动索引 → 分桶 → 分发到 8 卡
  │  3. 返回 job_id              │
  │<─────────────────────────────│
  │                              │
  │  4. 轮询进度                  │
  │     GET /api/jobs/{id}       │
  │─────────────────────────────>│
  │                              │  返回 progress: 45.5%
  │<─────────────────────────────│
  │                              │
  │  5. 任务完成后导出结果         │
  │     GET /api/jobs/{id}/export│
  │─────────────────────────────>│
  │                              │  流式 ZIP 下载
  │<═════════════════════════════│
```

## 1.5 性能指标

| 指标 | 目标值 |
|------|--------|
| SAM3 吞吐量 | 1000 张 < 30 分钟 (8卡 24 Worker) |
| DreamSim 吞吐量 | 500 对 < 20 分钟 (8卡 24 Worker) |
| GPU 利用率 | 峰值 > 80% |
| 单图失败恢复 | 不影响整个批次，自动标记可重试 |
| Worker 崩溃恢复 | 自动重启，任务自动回收 |

---

# 二、系统架构

## 2.1 架构总览

```
┌────────────────────────────────────────────────────────────────┐
│ 宿主机                                                         │
│                                                                │
│  ┌──────────────────────────┐                                  │
│  │  Gateway (FastAPI)        │ ← conda env / 宿主机 Python      │
│  │  :8090                    │                                  │
│  │  ┌──────────────────────┐ │                                  │
│  │  │ /api/jobs   任务管理  │ │                                  │
│  │  │ /api/health 健康监控  │ │                                  │
│  │  │ Janitor  后台清理守护  │ │                                  │
│  │  │ Scaler   弹性伸缩控制 │ │                                  │
│  │  └──────────────────────┘ │                                  │
│  └────────────┬──────────────┘                                  │
│               │ asyncpg (TCP:5433)                              │
│  ┌────────────▼──────────────┐   共享卷                         │
│  │  PostgreSQL 15             │   /data/batch_shared/           │
│  │  (Docker: batch-postgres)  │◄──────────────────────────┐    │
│  │  :5433                     │                            │    │
│  └────────────────────────────┘                            │    │
│                                                            │    │
│  ┌─────────────────────────────────────────────────────────┤    │
│  │  GPU Worker Containers (Docker)                         │    │
│  │                                                         │    │
│  │  SAM3 Workers                DreamSim Workers           │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐ │    │
│  │  │ GPU 0     │ │ GPU 1     │ │ GPU 4     │ │ GPU 5   │ │    │
│  │  │ 3 Worker  │ │ 3 Worker  │ │ 3 Worker  │ │ 3 Worker│ │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └─────────┘ │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐ │    │
│  │  │ GPU 2     │ │ GPU 3     │ │ GPU 6     │ │ GPU 7   │ │    │
│  │  │ 3 Worker  │ │ 3 Worker  │ │ 3 Worker  │ │ 3 Worker│ │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └─────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

**核心组件：**

| 组件 | 运行环境 | 职责 |
|------|----------|------|
| **Gateway** | 宿主机 Python | HTTP API、任务索引/分发、进度监控、弹性伸缩 |
| **PostgreSQL** | Docker 容器 | 任务状态持久化、并发任务分配（行锁） |
| **SAM3 Worker** | Docker 容器 (GPU) | SAM3 模型推理（含 Zero-DCE 增强） |
| **DreamSim Worker** | Docker 容器 (GPU) | DreamSim 滑窗变化检测 |
| **共享卷** | `/data/batch_shared/` | 心跳文件、任务描述符、结果文件、伸缩指令 |

## 2.2 项目结构

```
puruan_2_28/
├── gateway/                    # Gateway 服务 (宿主机运行)
│   ├── app.py                  #   FastAPI 入口，7 个 API 路由
│   ├── db.py                   #   PostgreSQL asyncpg 封装
│   ├── indexer.py              #   ZIP/目录索引器 + DreamSim 配对 + Pipeline 索引
│   ├── scheduler.py            #   Bucket 划分与任务分发
│   ├── janitor.py              #   后台守护：心跳巡检/spillover/任务回收/流水线推进
│   └── scaler.py               #   Auto-Scaler 弹性伸缩控制器
├── worker/                     # Worker 服务 (Docker 容器内运行)
│   ├── launcher.py             #   容器入口：N 进程启动器 + 弹性伸缩
│   ├── base_worker.py          #   Worker 主循环：抢占→推理→写结果→心跳
│   ├── engine_sam3.py          #   SAM3 推理引擎
│   ├── engine_dreamsim.py      #   DreamSim 推理引擎
│   ├── Dockerfile.sam3         #   FROM sam3-sam3:latest
│   └── Dockerfile.dreamsim     #   FROM diffsim-diffsim-runner:latest
├── shared/                     # 共享模块 (Gateway 和 Worker 均使用)
│   ├── config.py               #   全局配置（环境变量覆盖）
│   └── protocol.py             #   数据结构 + EngineInterface 抽象
├── sql/
│   └── init.sql                # PostgreSQL 建表脚本
├── scripts/
│   └── setup_mps.sh            # NVIDIA MPS 启停脚本
├── docker-compose.yml          # 容器编排（PG + 8 Worker）
├── start.sh                    # 一键启动脚本
└── requirements_gateway.txt    # Gateway Python 依赖
```

## 2.3 分层设计

```
┌─────────────────────────────────────────────────┐
│ API 层 (gateway/app.py)                          │
│  只负责 HTTP 路由、参数校验、响应格式化             │
│  不包含任何业务逻辑                               │
└────────────────────┬────────────────────────────┘
                     │ 调用
┌────────────────────▼────────────────────────────┐
│ Service 层 (indexer / scheduler / janitor / scaler)│
│  纯业务逻辑，不依赖 FastAPI                       │
│  通过 db.py 访问数据库                            │
└────────────────────┬────────────────────────────┘
                     │ 共享卷 + DB
┌────────────────────▼────────────────────────────┐
│ Worker 层 (base_worker.py)                       │
│  任务调度循环，不知道具体推理细节                   │
│  通过 EngineInterface 抽象接口调用引擎             │
└────────────────────┬────────────────────────────┘
                     │ 多态调用
┌────────────────────▼────────────────────────────┐
│ Engine 层 (engine_sam3 / engine_dreamsim)         │
│  每个引擎独立封装，只暴露统一接口                   │
│  新增模型只需实现接口 + 装饰器注册                  │
└─────────────────────────────────────────────────┘
```

**模块依赖规则：**

| 模块 | 允许依赖 | 禁止依赖 |
|------|----------|----------|
| `shared/` | 无外部依赖 | gateway/, worker/ |
| `gateway/app.py` | shared/, gateway/* | worker/ |
| `worker/base_worker.py` | shared/, engine_* (通过注册表) | gateway/ |
| `worker/engine_*.py` | shared/protocol.py | base_worker.py, gateway/ |

## 2.4 数据库设计

使用独立 PostgreSQL 15 实例（端口 5433，不与现有 5432 实例冲突）。

### jobs 表 — 批量任务

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | UUID (PK) | 任务唯一标识 |
| `model_type` | VARCHAR(20) | `sam3`、`dreamsim` 或 `pipeline` |
| `status` | VARCHAR(20) | `pending` → `running` → `completed` / `failed` / `cancelled` |
| `total_tasks` | INTEGER | 子任务总数（pipeline parent 为 0） |
| `completed_tasks` | INTEGER | 已完成数 |
| `failed_tasks` | INTEGER | 失败数 |
| `input_path` | TEXT | 输入目录/ZIP 路径 |
| `params` | JSONB | 模型参数（pipeline 存储 `{sam3:{...}, dreamsim:{...}, num_workers:N}`） |
| `created_at` | TIMESTAMPTZ | 创建时间 |
| `updated_at` | TIMESTAMPTZ | 最后更新时间 |
| `parent_job_id` | UUID (FK → jobs) | 流水线父任务 ID（仅子任务有值，级联删除） |
| `pipeline_phase` | VARCHAR(20) | 流水线阶段标识：`sam3_before` / `sam3_after` / `dreamsim` |

### tasks 表 — 单个子任务

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | UUID (PK) | 子任务唯一标识 |
| `job_id` | UUID (FK → jobs) | 所属批量任务 |
| `bucket_id` | INTEGER | 分桶编号 |
| `status` | VARCHAR(20) | `pending` → `running` → `done` / `failed` |
| `image_path` | TEXT | SAM3 单图路径 |
| `image_pair` | JSONB | DreamSim 配对 `{"before":"...","after":"..."}` |
| `result_path` | TEXT | 结果文件路径 |
| `error_msg` | TEXT | 失败错误信息 |
| `worker_id` | VARCHAR(50) | 执行该任务的 Worker ID |
| `started_at` | TIMESTAMPTZ | 开始时间 |
| `finished_at` | TIMESTAMPTZ | 完成时间 |

### 索引

```sql
idx_tasks_job_status  ON tasks(job_id, status)       -- 任务状态查询
idx_tasks_bucket      ON tasks(job_id, bucket_id, status) -- 分桶抢占
idx_jobs_status       ON jobs(status)                 -- 活跃任务筛选
idx_jobs_parent       ON jobs(parent_job_id)          -- 流水线子任务查询
idx_tasks_worker      ON tasks(worker_id, status)     -- Worker 任务查询
```

## 2.5 数据流

### SAM3 批量处理流程

```
1. 用户 POST /api/jobs
   {input_path: "/data/images/", model_type: "sam3", text: "building"}

2. Gateway:
   ├── indexer.py 扫描目录 → 发现 1000 张图片
   ├── scheduler.py 按路径排序 → 划分 24 个 Bucket
   ├── 写入 DB: 1 条 jobs + 1000 条 tasks
   └── 写入文件系统: pending/ 下 1000 个 task JSON

3. 24 个 Worker 并行:
   ├── base_worker.py 通过 SQL SKIP LOCKED 抢占 pending 任务
   ├── engine_sam3.py: 读图 → Zero-DCE 增强 → SAM3 推理 → 形态学后处理
   ├── 结果写入 results/ 目录
   └── 更新 DB: task.status = done, result_path = ...

4. Janitor 每 10 秒巡检:
   ├── 同步 completed_tasks / failed_tasks 到 jobs 表
   ├── 回收超时 300 秒未完成的 running 任务
   └── 检测全部完成后更新 job.status = completed

5. 用户 GET /api/jobs/{id}/export → 流式 ZIP 下载
```

### DreamSim 批量处理流程

```
1. 用户提供目录结构:
   /data/pairs/
   ├── before/         # 基准时期图片
   │   ├── area_001.jpg
   │   └── area_002.jpg
   └── after/          # 现状时期图片
       ├── area_001.jpg  ← 与 before/area_001.jpg 自动配对
       └── area_002.jpg

2. Gateway indexer 自动匹配同名文件 → 生成 image_pair

3. Worker engine_dreamsim: 滑窗切片 → DreamSim 比对 → 热力图 + 标注图
   每对输出两个文件:
   ├── heatmap_area_001.jpg       (热力图版：叠加热力色 + 红框)
   └── heatmap_area_001_clean.jpg (纯净版：原图 + 红框)
```

### SAM3→DreamSim 流水线处理流程

```
1. 用户 POST /api/jobs
   {input_path: "/data/pairs/", model_type: "pipeline", text: "building", thresh: 0.3}

2. Gateway:
   ├── indexer.py 扫描 before/ 和 after/ → 各发现 200 张图片
   ├── 创建 Pipeline parent job (DB, total_tasks=0, 仅作聚合点)
   ├── 创建 SAM3 Before child job → 分发 200 个 SAM3 任务
   └── 创建 SAM3 After child job  → 分发 200 个 SAM3 任务

3. SAM3 Workers 并行处理:
   ├── 24 个 SAM3 Worker 同时抢占 before 和 after 两个 child job 的任务
   ├── 结果分别写入各自 child job 的 results/ 目录
   └── Janitor 巡检自动同步 child job 进度

4. 阶段推进 (Janitor _check_pipeline_progress):
   ├── 两个 SAM3 child 都 completed → 触发 _create_dreamsim_phase
   ├── 收集两个 SAM3 child 的 results/ 目录
   ├── 按文件名配对 (result_area_001.jpg ↔ result_area_001.jpg)
   └── 自动创建 DreamSim child job → 分发 200 个 DreamSim 任务

5. DreamSim Workers 处理:
   ├── 抢占 DreamSim child job 的任务
   └── 输出热力图到 DreamSim child 的 results/ 目录

6. 完成:
   ├── DreamSim child completed → Janitor 标记 pipeline parent completed
   └── 用户 GET /api/jobs/{id}/export → 只导出 DreamSim 阶段结果
```

**Pipeline 数据流图：**

```
before/img001.jpg ──→ [SAM3 Before Job] ──→ results/result_img001.jpg ─┐
before/img002.jpg ──→                   ──→ results/result_img002.jpg ─┤
...                                                                     ├→ 配对 → [DreamSim Job]
after/img001.jpg  ──→ [SAM3 After Job]  ──→ results/result_img001.jpg ─┤      → heatmap_result_img001.jpg
after/img002.jpg  ──→                   ──→ results/result_img002.jpg ─┘      → heatmap_result_img002.jpg
```

## 2.6 弹性伸缩策略

### 参数

| 参数 | 默认值 | 环境变量 | 说明 |
|------|--------|----------|------|
| 最小 Worker/卡 | 1 | `MIN_WORKERS_PER_GPU` | 空闲时保活数量 |
| 最大 Worker/卡 | 4 | `MAX_WORKERS_PER_GPU` | 48GB / 12GB = 4 |
| 扩容阈值 | pending > active × 2 | `SCALE_UP_MULTIPLIER` | 触发扩容条件 |
| 扩容持续时间 | 10 秒 | `SCALE_UP_SUSTAIN_SEC` | 条件持续多久才扩 |
| 缩容持续时间 | 30 秒 | `SCALE_DOWN_SUSTAIN_SEC` | 队列空多久才缩 |
| 冷却间隔 | 15 秒 | `SCALE_COOLDOWN_SEC` | 两次伸缩最小间隔 |

### 伸缩流程示例

```
场景：4 张 GPU，每卡初始 1 Worker (共 4)，用户提交 2000 张图

t=0s    Gateway 分发 2000 个 task，4 Worker 开始处理
t=10s   每 Worker 积压 ~490 张 → 持续 10s → 触发 SCALE_UP
        AutoScaler 写入指令: 每卡 → 2 Worker (共 8)
t=25s   仍积压 → 每卡 → 3 Worker (共 12)
t=40s   仍积压 → 每卡 → 4 Worker (共 16，到达 MAX)
t=180s  全部完成，队列清空
t=210s  空闲 30s → SCALE_DOWN → 逐步收缩回每卡 1 Worker
```

### 实现机制

- **Gateway AutoScaler** (`scaler.py`): 每 5 秒检查队列深度，写入 JSON 指令到 `/data/batch_shared/scaling/gpu_{id}.json`
- **容器 Launcher** (`launcher.py`): 轮询指令文件，动态 `fork` / `terminate` 子进程
- 缩容时等当前任务完成后再终止进程，不中断推理

## 2.7 容错机制

| 故障场景 | 自动应对 | 恢复方式 |
|----------|----------|----------|
| 单张图片推理失败 | task 标记为 `failed`，不影响其他任务 | `POST /api/jobs/{id}/retry` |
| Worker 进程崩溃 | Launcher 自动检测并重启子进程 | 自动 |
| Worker 容器崩溃 | Docker `restart: unless-stopped` 自动重启 | 自动 |
| 任务超时 (300s) | Janitor 回收 running → pending，其他 Worker 重新抢占 | 自动 |
| 结果目录过大 | Janitor 自动 spillover 到溢出区 | 自动 |
| 数据库连接断开 | Worker 自动重连 | 自动 |
| Gateway 重启 | 任务状态在 DB 中持久化，不丢失 | 重启后自动恢复 |
| Pipeline SAM3 阶段失败 | Janitor 检测 SAM3 child failed → 标记 pipeline 整体 failed | 重试对应 child job |
| Pipeline DreamSim 阶段失败 | Janitor 检测 DreamSim child failed → 标记 pipeline 整体 failed | 重试 DreamSim child |
| Pipeline 创建 DreamSim 失败 | 异常捕获 → 标记 pipeline failed，记录错误日志 | 排查后重新提交 |

## 2.8 扩展新模型指南

只需 3 步，无需修改任何已有代码：

**步骤 1：创建引擎文件** `worker/engine_xxx.py`

```python
from shared.protocol import EngineInterface, TaskDescriptor, register_engine

@register_engine("xxx")  # 注册名称，对应 API 中的 model_type
class XXXEngine(EngineInterface):

    def load(self, device: str, **kwargs) -> None:
        # 加载模型到 device (如 "cuda:0")
        ...

    def infer(self, task: TaskDescriptor, result_dir: str) -> dict:
        # task.image_path 或 task.image_pair 获取输入
        # task.params 获取参数
        # 结果写入 result_dir
        return {"result_path": "...", "metadata": {...}}

    def unload(self) -> None:
        # 释放显存
        ...
```

**步骤 2：创建 Dockerfile** `worker/Dockerfile.xxx`

```dockerfile
FROM your-base-image:latest
COPY worker/ /app/worker/
COPY shared/ /app/shared/
WORKDIR /app
ENTRYPOINT ["python3", "-m", "worker.launcher"]
```

**步骤 3：在 docker-compose.yml 中添加 Worker 服务**

无需修改 `base_worker.py`、`app.py`、`scheduler.py` 中的任何一行代码。

---

# 三、部署指南

## 3.1 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | 1 × NVIDIA GPU (16GB+) | 8 × NVIDIA L20 (48GB) |
| CPU | 8 核 | 32 核+ |
| 内存 | 32 GB | 128 GB |
| 磁盘 | 100 GB 可用空间 | 500 GB+ SSD |

### 软件要求

| 组件 | 版本要求 |
|------|----------|
| OS | Ubuntu 20.04 / 22.04 |
| NVIDIA Driver | >= 535.x (CUDA 12.2+) |
| Docker | >= 24.0 |
| NVIDIA Container Toolkit | >= 1.14 |
| Docker Compose | v2 (docker compose) |
| Python | 3.10+ |
| conda (可选) | Miniconda3 |

### 前置条件

```bash
# 验证 GPU 驱动
nvidia-smi

# 验证 Docker + GPU 支持
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# 验证现有基础镜像存在
docker images | grep -E "sam3-sam3|diffsim-diffsim-runner"
```

所需基础镜像：
- `sam3-sam3:latest` (约 18.7GB) — SAM3 推理环境
- `diffsim-diffsim-runner:latest` (约 18.3GB) — DreamSim 推理环境

## 3.2 一键部署

```bash
cd puruan_2_28/
chmod +x start.sh
./start.sh
```

`start.sh` 会自动执行：
1. 创建共享目录 `/data/batch_shared/`
2. 配置 conda 环境 + 安装 Gateway 依赖
3. 构建 Worker Docker 镜像
4. 启动 PostgreSQL + 8 个 Worker 容器
5. 启动 Gateway 服务

启动完成后访问：
- **API 入口**: `http://<服务器IP>:8090`
- **Swagger 文档**: `http://<服务器IP>:8090/docs`

## 3.3 手动分步部署

### 步骤 1：创建共享目录

```bash
sudo mkdir -p /data/batch_shared/{heartbeat,jobs,scaling,spillover,pgdata}
sudo chmod -R 777 /data/batch_shared
```

### 步骤 2：安装 Gateway 依赖

```bash
# 方式 A: 直接安装
pip install -r requirements_gateway.txt

# 方式 B: 使用 conda 隔离环境
conda create -n batch_gateway python=3.10 -y
conda activate batch_gateway
pip install -r requirements_gateway.txt
```

### 步骤 3：构建 Worker 镜像

```bash
cd puruan_2_28/

# SAM3 Worker 镜像
docker build -t batch-sam3-worker -f worker/Dockerfile.sam3 .

# DreamSim Worker 镜像
docker build -t batch-ds-worker -f worker/Dockerfile.dreamsim .
```

### 步骤 4：启动 Docker 容器

```bash
cd puruan_2_28/
docker compose up -d

# 验证
docker compose ps
docker exec batch-postgres pg_isready -U batch -d batch_detection
```

### 步骤 5：（可选）启用 NVIDIA MPS

MPS 可让同一 GPU 上的多个 Worker 进程真正并行执行：

```bash
sudo bash scripts/setup_mps.sh start
```

### 步骤 6：启动 Gateway

```bash
cd puruan_2_28/
export PYTHONPATH="$(pwd):$PYTHONPATH"
python -m gateway.app

# 或后台运行
nohup python -m gateway.app > /var/log/batch_gateway.log 2>&1 &
```

### 步骤 7：验证

```bash
# 健康检查
curl http://localhost:8090/api/health

# 预期返回
# {"status":"healthy","workers":[...],"active_jobs":0}
```

## 3.4 环境变量参考

所有配置均可通过环境变量覆盖，无需修改代码。

### 路径配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SHARED_ROOT` | `/data/batch_shared` | 共享卷根目录 |
| `BATCH_SPILLOVER_DIR` | `{SHARED_ROOT}/spillover` | 溢出落盘目录 |

### 数据库配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_PG_HOST` | `localhost` | PG 主机 (容器内用 `batch-postgres`) |
| `BATCH_PG_PORT` | `5433` | PG 端口 (宿主机映射) |
| `BATCH_PG_DB` | `batch_detection` | 数据库名 |
| `BATCH_PG_USER` | `batch` | 数据库用户 |
| `BATCH_PG_PASS` | `batch_secret` | 数据库密码 |

### Gateway 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GATEWAY_HOST` | `0.0.0.0` | 监听地址 |
| `GATEWAY_PORT` | `8090` | 监听端口 |

### Worker 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WORKER_POLL_INTERVAL` | `0.5` | 无任务时轮询间隔 (秒) |
| `HEARTBEAT_INTERVAL` | `5` | 心跳写入间隔 (秒) |
| `HEARTBEAT_TIMEOUT` | `30` | 心跳超时判定 (秒) |
| `TASK_TIMEOUT` | `300` | 任务超时回收 (秒) |

### 弹性伸缩配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MIN_WORKERS_PER_GPU` | `1` | 每卡最少 Worker |
| `MAX_WORKERS_PER_GPU` | `4` | 每卡最多 Worker |
| `SCALE_UP_MULTIPLIER` | `2` | 扩容阈值倍数 |
| `SCALE_UP_SUSTAIN_SEC` | `10` | 扩容条件持续时间 |
| `SCALE_DOWN_SUSTAIN_SEC` | `30` | 缩容条件持续时间 |
| `SCALE_COOLDOWN_SEC` | `15` | 伸缩冷却间隔 |

### Janitor 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `JANITOR_INTERVAL` | `10` | 巡检间隔 (秒) |
| `SPILLOVER_THRESHOLD_MB` | `10240` | 触发 spillover 的目录大小 (MB) |

### 模型配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SAM3_MODEL_PATH` | `/app/checkpoints/sam3.pt` | SAM3 权重路径 |
| `SAM3_DCE_WEIGHTS` | `/app/checkpoints/Zero-DCE.pth` | Zero-DCE 权重路径 |
| `SAM3_DCE_DEVICE` | `cuda` | Zero-DCE 运行设备 (`cuda` 或 `cpu`) |
| `DREAMSIM_MODEL_TYPE` | `ensemble` | DreamSim 模型类型 |

## 3.5 GPU 分配策略

默认分配方案（8 × L20）：

| GPU ID | 容器名 | 模型 | 初始 Worker 数 |
|--------|--------|------|----------------|
| 0 | sam3-worker-0 | SAM3 | 3 |
| 1 | sam3-worker-1 | SAM3 | 3 |
| 2 | sam3-worker-2 | SAM3 | 3 |
| 3 | sam3-worker-3 | SAM3 | 3 |
| 4 | ds-worker-0 | DreamSim | 3 |
| 5 | ds-worker-1 | DreamSim | 3 |
| 6 | ds-worker-2 | DreamSim | 3 |
| 7 | ds-worker-3 | DreamSim | 3 |

**自定义分配：** 编辑 `docker-compose.yml`，修改各 Worker 的 `device_ids` 和 `--gpu-id` 参数。

**全部给 SAM3：** 注释掉 `ds-worker-*` 部分，增加 `sam3-worker-4` ~ `sam3-worker-7` 使用 GPU 4-7。

## 3.6 Docker 容器清单

| 容器名 | 镜像 | 端口 | 持久化卷 | 说明 |
|--------|------|------|----------|------|
| `batch-postgres` | postgres:15-alpine | 5433:5432 | `/data/batch_shared/pgdata` | 任务数据库 |
| `sam3-worker-0` | batch-sam3-worker | - | `/data/batch_shared` | SAM3 GPU 0 |
| `sam3-worker-1` | batch-sam3-worker | - | `/data/batch_shared` | SAM3 GPU 1 |
| `sam3-worker-2` | batch-sam3-worker | - | `/data/batch_shared` | SAM3 GPU 2 |
| `sam3-worker-3` | batch-sam3-worker | - | `/data/batch_shared` | SAM3 GPU 3 |
| `ds-worker-0` | batch-ds-worker | - | `/data/batch_shared` | DreamSim GPU 4 |
| `ds-worker-1` | batch-ds-worker | - | `/data/batch_shared` | DreamSim GPU 5 |
| `ds-worker-2` | batch-ds-worker | - | `/data/batch_shared` | DreamSim GPU 6 |
| `ds-worker-3` | batch-ds-worker | - | `/data/batch_shared` | DreamSim GPU 7 |

---

# 四、API 使用手册

**Base URL**: `http://<服务器IP>:8090`
**Swagger 文档**: `http://<服务器IP>:8090/docs`

## 4.1 提交 SAM3 批量任务

```
POST /api/jobs
```

**请求体：**

```json
{
    "input_path": "/data/drone_images/zone_a/",
    "model_type": "sam3",
    "text": "building",
    "conf": 0.25,
    "skip_dce": false,
    "dilate": 2,
    "fill_holes": true,
    "rect_fill": false,
    "bbox_pad": 0,
    "num_workers": 24
}
```

**参数说明：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `input_path` | string | 是 | - | 服务器上图片目录或 ZIP 文件的绝对路径 |
| `model_type` | string | 是 | - | 固定 `"sam3"` |
| `text` | string | 是(SAM3) | - | 检测提示词，如 `"building"`, `"house"` |
| `conf` | float | 否 | 0.2 | 置信度阈值 (0.0 ~ 1.0)，越高越严格 |
| `skip_dce` | bool | 否 | false | 是否跳过 Zero-DCE 暗光增强 |
| `dilate` | int | 否 | 0 | 边缘膨胀核大小 (0 = 不膨胀) |
| `fill_holes` | bool | 否 | false | 填充 mask 孔洞 |
| `rect_fill` | bool | 否 | false | 矩形修复 |
| `bbox_pad` | int | 否 | 0 | 检测框外扩像素 |
| `num_workers` | int | 否 | 24 | 影响分桶数量（通常等于 Worker 总数） |

**响应：**

```json
{
    "job_id": "4ea406bd-2ae6-461c-971b-3431141b4239",
    "total_tasks": 1000,
    "status": "running"
}
```

**支持的输入格式：**
- 目录：`/data/images/`，递归扫描所有 jpg/jpeg/png/bmp/tif/tiff/webp 文件
- ZIP：`/data/images.zip`，自动解压后索引

## 4.2 提交 DreamSim 批量任务

```
POST /api/jobs
```

**请求体：**

```json
{
    "input_path": "/data/comparison/zone_b/",
    "model_type": "dreamsim",
    "thresh": 0.30,
    "crop": 224,
    "step": 0,
    "batch": 16,
    "num_workers": 24
}
```

**参数说明：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `input_path` | string | 是 | - | 包含 `before/` 和 `after/` 子目录的路径 |
| `model_type` | string | 是 | - | 固定 `"dreamsim"` |
| `thresh` | float | 否 | 0.3 | 变化检测阈值 (0.0 ~ 1.0)，越低越敏感 |
| `crop` | int | 否 | 224 | 滑窗切片大小 (像素) |
| `step` | int | 否 | 0 | 滑窗步长 (0 = 自动设为 crop/2) |
| `batch` | int | 否 | 16 | GPU 批推理大小 |

**输入目录结构要求：**

```
/data/comparison/zone_b/
├── before/                 # 基准时期
│   ├── area_001.jpg
│   ├── area_002.jpg
│   └── area_003.jpg
└── after/                  # 现状时期 (同名自动配对)
    ├── area_001.jpg        ← 与 before/area_001.jpg 配对
    ├── area_002.jpg
    └── area_003.jpg
```

- 文件名必须一一对应（不含扩展名部分必须相同）
- 未配对的文件会被跳过并记录警告日志

## 4.3 提交 Pipeline 流水线任务

```
POST /api/jobs
```

**请求体：**

```json
{
    "input_path": "/data/comparison/zone_c/",
    "model_type": "pipeline",
    "text": "building",
    "conf": 0.25,
    "skip_dce": false,
    "dilate": 2,
    "fill_holes": true,
    "thresh": 0.30,
    "crop": 224,
    "step": 0,
    "batch": 16,
    "num_workers": 24
}
```

**参数说明：**

Pipeline 同时接受 SAM3 和 DreamSim 的参数，SAM3 参数用于第一阶段分割，DreamSim 参数用于第二阶段变化检测。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `input_path` | string | 是 | 包含 `before/` 和 `after/` 子目录的路径 |
| `model_type` | string | 是 | 固定 `"pipeline"` |
| `text` | string | 是 | SAM3 检测提示词 |
| `conf` | float | 否 | SAM3 置信度阈值 (默认 0.2) |
| `skip_dce` | bool | 否 | 跳过暗光增强 (默认 false) |
| `dilate` | int | 否 | 边缘膨胀核大小 (默认 0) |
| `fill_holes` | bool | 否 | 填充 mask 孔洞 (默认 false) |
| `rect_fill` | bool | 否 | 矩形修复 (默认 false) |
| `bbox_pad` | int | 否 | 检测框外扩 (默认 0) |
| `thresh` | float | 否 | DreamSim 变化阈值 (默认 0.3) |
| `crop` | int | 否 | DreamSim 滑窗大小 (默认 224) |
| `step` | int | 否 | DreamSim 滑窗步长 (默认 0=自动) |
| `batch` | int | 否 | DreamSim 批大小 (默认 16) |

**输入目录结构：** 与 DreamSim 相同，需包含 `before/` 和 `after/` 子目录。

**响应：**

```json
{
    "job_id": "a1b2c3d4-...",
    "total_tasks": 0,
    "status": "running",
    "children": [
        {"job_id": "e5f6g7h8-...", "phase": "sam3_before", "total_tasks": 200},
        {"job_id": "i9j0k1l2-...", "phase": "sam3_after", "total_tasks": 200}
    ]
}
```

**流程说明：**

1. 系统立即创建 1 个 parent pipeline job + 2 个 SAM3 child job
2. SAM3 Worker 自动抢占 child job 的任务并行处理
3. 两个 SAM3 child 全部 completed 后，Janitor 自动创建 DreamSim child job
4. DreamSim Worker 抢占处理完成后，pipeline parent 标记为 completed
5. 全程无需人工干预，通过 `GET /api/jobs/{parent_id}` 监控进度

## 4.4 查询任务进度

```
GET /api/jobs/{job_id}
```

**响应（普通任务）：**

```json
{
    "job_id": "4ea406bd-2ae6-461c-971b-3431141b4239",
    "model_type": "sam3",
    "status": "running",
    "total_tasks": 1000,
    "completed_tasks": 456,
    "failed_tasks": 2,
    "progress_pct": 45.6,
    "input_path": "/data/drone_images/zone_a/",
    "params": {
        "text": "building",
        "conf": 0.25
    },
    "created_at": "2026-02-28T04:50:31.156855+00:00",
    "updated_at": "2026-02-28T05:02:15.789000+00:00"
}
```

**响应（Pipeline 任务）：**

```json
{
    "job_id": "a1b2c3d4-...",
    "model_type": "pipeline",
    "status": "running",
    "total_tasks": 600,
    "completed_tasks": 350,
    "failed_tasks": 0,
    "progress_pct": 58.33,
    "input_path": "/data/comparison/zone_c/",
    "phase": "sam3",
    "children": [
        {"job_id": "e5f6-...", "phase": "sam3_before", "status": "completed", "total_tasks": 200, "completed_tasks": 200, "failed_tasks": 0, "progress_pct": 100.0},
        {"job_id": "i9j0-...", "phase": "sam3_after", "status": "running", "total_tasks": 200, "completed_tasks": 150, "failed_tasks": 0, "progress_pct": 75.0},
        {"job_id": "m3n4-...", "phase": "dreamsim", "status": "pending", "total_tasks": 200, "completed_tasks": 0, "failed_tasks": 0, "progress_pct": 0}
    ]
}
```

Pipeline 特有字段：
- `phase`: 当前阶段 — `sam3`（SAM3 处理中）/ `sam3_done`（SAM3 完成等待创建 DreamSim）/ `dreamsim`（DreamSim 处理中）
- `children`: 各子任务的独立进度
- `total_tasks` / `completed_tasks`: 聚合所有 children 的数据
```

**status 状态机：**

```
pending ──► running ──► completed  (全部成功)
                    └─► completed  (部分失败，可查看 failed_tasks)
                    └─► cancelled  (被用户取消)
```

**轮询建议：** 每 2-5 秒轮询一次 `progress_pct`，到达 100% 即可导出。

## 4.5 列出所有任务

```
GET /api/jobs?limit=50&offset=0
```

**查询参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `limit` | int | 50 | 返回数量 (1-200) |
| `offset` | int | 0 | 偏移量 |

**响应：** `JobResponse` 数组，按创建时间倒序排列。

## 4.6 取消任务

```
DELETE /api/jobs/{job_id}
```

**响应：**

```json
{
    "job_id": "4ea406bd-...",
    "status": "cancelled"
}
```

取消后正在运行的子任务会在当前图片处理完后停止（不会暴力中断）。

## 4.7 重试失败任务

```
POST /api/jobs/{job_id}/retry
```

将该 Job 下所有 `failed` 状态的子任务重置为 `pending`，Worker 会自动重新抢占处理。

**响应：**

```json
{
    "job_id": "4ea406bd-...",
    "retried_tasks": 12
}
```

## 4.8 查看失败详情

```
GET /api/jobs/{job_id}/failed
```

**响应：**

```json
[
    {
        "task_id": "abc123...",
        "image_path": "/data/images/broken_file.jpg",
        "image_pair": null,
        "error_msg": "ValueError: Failed to read image..."
    }
]
```

用于定位失败原因：图片损坏、路径不存在、推理异常等。

## 4.9 导出结果

```
GET /api/jobs/{job_id}/export
```

**响应：** 流式 ZIP 文件下载

```bash
# curl 下载示例
curl -o results.zip http://localhost:8090/api/jobs/{job_id}/export

# wget 下载示例
wget -O results.zip "http://localhost:8090/api/jobs/{job_id}/export"
```

ZIP 内容：
- SAM3: `result_{原文件名}.jpg` — 黑底抠像
- DreamSim: `heatmap_{原文件名}.jpg` + `heatmap_{原文件名}_clean.jpg`
- Pipeline: 只导出 DreamSim 阶段的最终结果（热力图），不包含中间 SAM3 抠像

## 4.10 系统健康检查

```
GET /api/health
```

**响应：**

```json
{
    "status": "healthy",
    "workers": [
        {
            "worker_id": "gpu0_w0",
            "last_heartbeat": "2026-02-28T05:00:12",
            "age_seconds": 3.2,
            "healthy": true
        },
        {
            "worker_id": "gpu0_w1",
            "last_heartbeat": "2026-02-28T05:00:10",
            "age_seconds": 5.2,
            "healthy": true
        }
    ],
    "active_jobs": 2
}
```

**status 含义：**

| 值 | 含义 |
|----|------|
| `healthy` | 有活跃且健康的 Worker，或无活跃任务 |
| `degraded` | 有活跃任务但无健康 Worker（需要排查） |

---

# 五、运维手册

## 5.1 日常巡检

```bash
# 1. 系统状态总览
curl -s http://localhost:8090/api/health | python3 -m json.tool

# 2. GPU 使用情况
nvidia-smi

# 3. 容器状态
docker compose -f puruan_2_28/docker-compose.yml ps

# 4. 磁盘空间
df -h /data/batch_shared/
du -sh /data/batch_shared/jobs/*/results/ | sort -rh | head -10

# 5. 查看活跃任务
curl -s http://localhost:8090/api/jobs?limit=10 | python3 -m json.tool
```

## 5.2 日志查看

```bash
# Gateway 日志
tail -f /var/log/batch_gateway.log            # 如果用 nohup 启动
# 或直接在前台看标准输出

# Worker 容器日志
docker logs -f sam3-worker-0 --tail 100       # SAM3 GPU 0
docker logs -f ds-worker-0 --tail 100         # DreamSim GPU 4

# PostgreSQL 日志
docker logs batch-postgres --tail 50

# 所有 Worker 日志汇总
docker compose -f puruan_2_28/docker-compose.yml logs -f --tail 20
```

**关键日志关键字：**

| 关键字 | 含义 |
|--------|------|
| `completed task` | 正常完成一个子任务 |
| `failed task` | 子任务推理失败 |
| `Reclaimed ... stale tasks` | Janitor 回收了超时任务 |
| `Stale workers detected` | 发现心跳超时的 Worker |
| `Scale UP/DOWN` | AutoScaler 触发伸缩 |
| `Spillover: moved` | 结果目录过大，触发溢出落盘 |

## 5.3 常见故障排查

### 问题：Worker 容器启动失败

```bash
# 查看具体错误
docker logs sam3-worker-0

# 常见原因:
# 1. GPU 驱动不兼容 → 检查 nvidia-smi 与容器内 CUDA 版本
# 2. 基础镜像不存在 → docker images | grep sam3-sam3
# 3. PostgreSQL 未就绪 → docker logs batch-postgres
```

### 问题：任务一直 pending，不被处理

```bash
# 1. 检查 Worker 是否存活
curl -s http://localhost:8090/api/health | python3 -m json.tool

# 2. 检查 Worker 是否在处理对应 model_type
docker logs sam3-worker-0 --tail 20  # SAM3 任务要看 sam3-worker
docker logs ds-worker-0 --tail 20    # DreamSim 任务要看 ds-worker

# 3. 检查 DB 中任务状态
docker exec batch-postgres psql -U batch -d batch_detection \
  -c "SELECT status, COUNT(*) FROM tasks WHERE job_id='<JOB_ID>' GROUP BY status;"
```

### 问题：任务失败率高

```bash
# 查看失败原因
curl -s http://localhost:8090/api/jobs/<JOB_ID>/failed | python3 -m json.tool

# 常见原因:
# 1. 图片文件损坏 → error_msg 含 "Failed to read image"
# 2. 图片路径不存在 → Worker 容器内看不到宿主机路径
#    解决: 确保图片目录挂载到了容器的共享卷中
# 3. GPU OOM → 降低 MAX_WORKERS_PER_GPU 或减小 batch size
```

### 问题：Gateway 无法连接 PostgreSQL

```bash
# 检查 PG 容器
docker exec batch-postgres pg_isready -U batch -d batch_detection

# 检查端口映射
ss -tlnp | grep 5433

# 检查连通性
psql -h localhost -p 5433 -U batch -d batch_detection -c "SELECT 1;"
```

### 问题：Pipeline 任务卡在 SAM3 阶段不推进

```bash
# 1. 检查 SAM3 child job 是否真正 completed
docker exec batch-postgres psql -U batch -d batch_detection \
  -c "SELECT id, pipeline_phase, status, total_tasks, completed_tasks, failed_tasks
      FROM jobs WHERE parent_job_id = '<PIPELINE_JOB_ID>';"

# 2. 如果 SAM3 child 的 completed_tasks + failed_tasks < total_tasks
#    说明还有任务未完成，检查 Worker 是否存活
curl -s http://localhost:8090/api/health | python3 -m json.tool

# 3. 如果 SAM3 child 已 completed 但 DreamSim child 未创建
#    检查 Gateway 日志中 Janitor 错误信息
grep "Pipeline\|pipeline\|dreamsim" /var/log/batch_gateway.log | tail -20

# 4. 确认 SAM3 结果文件确实存在
ls /data/batch_shared/jobs/<SAM3_CHILD_ID>/results/ | head -5
```

### 问题：Pipeline DreamSim 阶段没有配对成功

```bash
# Janitor 会在日志中记录配对情况，检查是否降级到位置配对
grep "no exact name matches\|paired.*image pairs" /var/log/batch_gateway.log

# 常见原因:
# 1. before/ 和 after/ 的原始图片文件名不同 → SAM3 结果文件名也不同 → 无法按名称配对
# 2. 结果被 spillover 移走 → Janitor 同时检查 results/ 和 spillover/ 目录
```

## 5.4 数据库维护

```bash
# 连接数据库
docker exec -it batch-postgres psql -U batch -d batch_detection

# 查看任务统计
SELECT model_type, status, COUNT(*) FROM jobs GROUP BY model_type, status;

# 清理已完成的旧任务 (保留最近 7 天)
DELETE FROM jobs WHERE status IN ('completed','cancelled')
  AND created_at < now() - interval '7 days';

# 查看表大小
SELECT pg_size_pretty(pg_total_relation_size('tasks')) AS tasks_size,
       pg_size_pretty(pg_total_relation_size('jobs')) AS jobs_size;

# 手动回收超时任务
UPDATE tasks SET status='pending', worker_id=NULL, started_at=NULL
WHERE status='running' AND started_at < now() - interval '10 minutes';
```

## 5.5 磁盘空间管理

**关键目录：**

| 路径 | 内容 | 增长速度 | 清理策略 |
|------|------|----------|----------|
| `/data/batch_shared/jobs/{id}/results/` | 推理结果图 | 随任务增长 | 导出后删除 |
| `/data/batch_shared/spillover/` | 溢出文件 | 自动 spillover | 同上 |
| `/data/batch_shared/pgdata/` | PostgreSQL 数据 | 缓慢增长 | 定期清理旧任务 |
| `/data/batch_shared/heartbeat/` | 心跳文件 | 极小 | 自动清理 |

```bash
# 查看各 Job 结果目录大小
du -sh /data/batch_shared/jobs/*/results/ 2>/dev/null | sort -rh

# 清理已导出的 Job 结果
rm -rf /data/batch_shared/jobs/<JOB_ID>/

# 清理所有已完成 Job 的结果 (谨慎操作)
for jid in $(docker exec batch-postgres psql -U batch -d batch_detection \
  -t -c "SELECT id FROM jobs WHERE status IN ('completed','cancelled');"); do
  rm -rf /data/batch_shared/jobs/$jid/
done
```

## 5.6 性能调优

### 调优每卡 Worker 数

```bash
# 查看单 Worker 显存占用
nvidia-smi  # 观察 Memory-Usage

# SAM3: ~3.5GB/Worker → 48GB 卡最多 12 Worker (安全起见建议 4)
# DreamSim: ~3GB/Worker → 48GB 卡最多 14 Worker (安全起见建议 4)

# 调整 (通过环境变量)
export MAX_WORKERS_PER_GPU=3
export MIN_WORKERS_PER_GPU=1
```

### 调优 Zero-DCE 运行设备

```bash
# 默认 DCE 在 GPU 运行，大显存卡 (48GB L20) 推荐保持默认
export SAM3_DCE_DEVICE=cuda

# 小显存卡 (<24GB) 可能出现 OOM，改为 CPU 运行
# DCE 模型仅 313KB，CPU 推理每张约 5-6 秒，不影响整体吞吐
export SAM3_DCE_DEVICE=cpu
```

### 调优任务超时

```bash
# 大图片 (>10000px) 可能需要更长推理时间
export TASK_TIMEOUT=600  # 10 分钟
```

### 调优 Spillover 阈值

```bash
# 磁盘充足时调高，减少文件移动开销
export SPILLOVER_THRESHOLD_MB=51200  # 50GB
```

## 5.7 备份与恢复

### 备份

```bash
# 1. 数据库备份
docker exec batch-postgres pg_dump -U batch batch_detection > backup_$(date +%Y%m%d).sql

# 2. 结果文件备份 (可选)
tar -czf results_backup.tar.gz /data/batch_shared/jobs/
```

### 恢复

```bash
# 1. 恢复数据库
docker exec -i batch-postgres psql -U batch -d batch_detection < backup_20260228.sql

# 2. 恢复结果文件
tar -xzf results_backup.tar.gz -C /
```

## 5.8 停机与重启

### 优雅停机

```bash
# 1. 停止 Gateway (不再接受新任务)
kill $(pgrep -f "gateway.app")

# 2. 等待 Worker 处理完当前任务 (观察日志)
docker logs -f sam3-worker-0 --tail 5
# 等待看到 Worker 进入空闲轮询状态

# 3. 停止 Worker 容器
cd puruan_2_28/
docker compose stop

# 4. (可选) 停止 MPS
sudo bash scripts/setup_mps.sh stop
```

### 重启

```bash
# 1. 启动容器
cd puruan_2_28/
docker compose up -d

# 2. 等待 PG 就绪
docker exec batch-postgres pg_isready -U batch -d batch_detection

# 3. 启动 Gateway
export PYTHONPATH="$(pwd):$PYTHONPATH"
nohup python -m gateway.app > /var/log/batch_gateway.log 2>&1 &

# 4. 验证
curl http://localhost:8090/api/health
```

**注意：** 重启后，之前 running 状态的任务会在 300 秒 (TASK_TIMEOUT) 后被 Janitor 自动回收为 pending，由 Worker 重新抢占处理。无需人工干预。

---

# 六、附录

## 6.1 共享卷目录结构

```
/data/batch_shared/
├── heartbeat/                      # Worker 心跳文件
│   ├── gpu0_w0.ts                  #   内容: unix timestamp
│   ├── gpu0_w1.ts
│   ├── gpu1_w0.ts
│   └── ...
├── scaling/                        # AutoScaler 伸缩指令
│   ├── gpu_0.json                  #   {"target_workers": 3, "timestamp": ...}
│   └── ...
├── jobs/
│   └── {job_id}/
│       ├── meta.json               # 任务元数据
│       ├── pending/                # 待处理任务描述符
│       │   ├── task_000000.json
│       │   └── ...
│       ├── running/                # (预留) 正在处理的任务
│       ├── done/                   # (预留) 已完成的任务描述符
│       ├── failed/                 # (预留) 失败的任务描述符
│       └── results/                # 推理结果文件
│           ├── result_xxx.jpg      #   SAM3 黑底抠像
│           ├── heatmap_xxx.jpg     #   DreamSim 热力图版
│           └── heatmap_xxx_clean.jpg # DreamSim 纯净版
├── spillover/                      # 背压溢出区
│   └── {job_id}/
│       └── ...                     # Janitor 自动迁移过来的结果文件
└── pgdata/                         # PostgreSQL 数据持久化
```

## 6.2 全量环境变量速查表

| 变量 | 默认值 | 使用方 | 说明 |
|------|--------|--------|------|
| `BATCH_SHARED_ROOT` | `/data/batch_shared` | Gateway + Worker | 共享卷根目录 |
| `BATCH_SPILLOVER_DIR` | `{ROOT}/spillover` | Gateway | 溢出落盘目录 |
| `BATCH_PG_HOST` | `localhost` | Gateway + Worker | PostgreSQL 主机名 |
| `BATCH_PG_PORT` | `5433` | Gateway + Worker | PostgreSQL 端口 |
| `BATCH_PG_DB` | `batch_detection` | Gateway + Worker | 数据库名 |
| `BATCH_PG_USER` | `batch` | Gateway + Worker | 数据库用户 |
| `BATCH_PG_PASS` | `batch_secret` | Gateway + Worker | 数据库密码 |
| `GATEWAY_HOST` | `0.0.0.0` | Gateway | 监听地址 |
| `GATEWAY_PORT` | `8090` | Gateway | 监听端口 |
| `MIN_WORKERS_PER_GPU` | `1` | Gateway + Worker | 每卡最少 Worker |
| `MAX_WORKERS_PER_GPU` | `4` | Gateway + Worker | 每卡最多 Worker |
| `SCALE_UP_MULTIPLIER` | `2` | Gateway | 扩容阈值倍数 |
| `SCALE_UP_SUSTAIN_SEC` | `10` | Gateway | 扩容持续条件 (秒) |
| `SCALE_DOWN_SUSTAIN_SEC` | `30` | Gateway | 缩容持续条件 (秒) |
| `SCALE_COOLDOWN_SEC` | `15` | Gateway | 伸缩冷却间隔 (秒) |
| `WORKER_POLL_INTERVAL` | `0.5` | Worker | 无任务轮询间隔 (秒) |
| `HEARTBEAT_INTERVAL` | `5` | Worker | 心跳写入间隔 (秒) |
| `HEARTBEAT_TIMEOUT` | `30` | Gateway | 心跳超时判定 (秒) |
| `TASK_TIMEOUT` | `300` | Gateway | 任务超时回收 (秒) |
| `JANITOR_INTERVAL` | `10` | Gateway | Janitor 巡检间隔 (秒) |
| `SPILLOVER_THRESHOLD_MB` | `10240` | Gateway | Spillover 阈值 (MB) |
| `SAM3_MODEL_PATH` | `/app/checkpoints/sam3.pt` | Worker (SAM3) | SAM3 权重路径 |
| `SAM3_DCE_WEIGHTS` | `/app/checkpoints/Zero-DCE.pth` | Worker (SAM3) | DCE 权重路径 |
| `SAM3_DCE_DEVICE` | `cuda` | Worker (SAM3) | DCE 运行设备 (`cuda`/`cpu`) |
| `DREAMSIM_MODEL_TYPE` | `ensemble` | Worker (DreamSim) | DreamSim 模型类型 |

## 6.3 端口占用清单

| 端口 | 服务 | 协议 | 说明 |
|------|------|------|------|
| 8090 | Gateway (FastAPI) | HTTP | API 入口 (可通过 GATEWAY_PORT 修改) |
| 5433 | PostgreSQL | TCP | 任务数据库 (避开现有 5432) |

**不占用的端口：** Worker 容器不暴露端口，全部通过共享卷 + 数据库通信。
