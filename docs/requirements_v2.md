# 违章建筑检测系统 — 批量推理平台 v2.0 需求文档

## 1. 项目概述

### 1.1 系统简介

批量对比卫星/航拍照片，自动检测违章建筑的推理平台。用户提交一批图片，系统自动处理，处理完下载结果。

### 1.2 现有架构

```
用户 → Gateway (FastAPI, 宿主机) → PostgreSQL (Docker 容器)
                                  → 共享卷 /data/batch_shared/
                                  → SAM3 Worker ×4 (GPU 0~3)
                                  → DreamSim Worker ×4 (GPU 4~7)
```

- **Gateway**：宿主机 conda 环境 `batch_gateway` (Python 3.10)，`0.0.0.0:8090`
- **PostgreSQL**：容器 `batch-postgres`，端口 5433:5432，数据持久化到 `/data/batch_shared/pgdata`
- **SAM3 Worker**：4 个容器 (sam3-worker-0~3)，各绑 GPU 0~3，每容器 3 个进程
- **DreamSim Worker**：4 个容器 (ds-worker-0~3)，各绑 GPU 4~7，每容器 3 个进程
- **共享卷**：`/data/batch_shared/`，子目录 heartbeat/、jobs/、scaling/、spillover/、pgdata/
- **总计**：8 GPU，默认 24 Worker 进程，弹性 1~4 进程/GPU

### 1.3 现有模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 前台接待 | gateway/app.py | 8 个 API 路由：创建任务、查进度、取消、下载、重试、健康检查 |
| 任务调度 | gateway/scheduler.py + indexer.py | 扫描图片、分桶（24 份）、入库 |
| 弹性伸缩 | gateway/scaler.py | 每 5s 监控队列深度，写 JSON 伸缩指令到共享卷 |
| 巡检守卫 | gateway/janitor.py | 每 10s 巡检：进度同步、完成检测、流水线推进、超时回收、毒丸隔离、心跳检查、磁盘管理 |
| 数据库 | gateway/db.py + sql/init.sql | jobs 表 + tasks 表，asyncpg 封装 |
| 进程管理 | worker/launcher.py | 容器内工头：拉起/重启/增减 Worker 子进程 |
| Worker 主循环 | worker/base_worker.py | 心跳→抢任务→推理→写结果 |
| SAM3 引擎 | worker/engine_sam3.py | Zero-DCE 暗光增强 + SAM3 分割 + 后处理 |
| DreamSim 引擎 | worker/engine_dreamsim.py | 滑窗 patch 对比 + 热力图 + 标注框 |
| 配置 | shared/config.py | 所有配置项（环境变量覆盖） |
| 协议 | shared/protocol.py | 枚举、数据结构、EngineInterface 抽象接口、引擎注册表 |

### 1.4 现有两种算法

| | SAM3 分割检测 | DreamSim 变化检测 |
|---|---|---|
| 输入 | 单张图片 | 一对图片（before + after） |
| 输出 | 黑底抠像图（只留建筑部分） | 热力图 + 标注框图 |
| 用途 | 识别建筑物轮廓 | 发现新增违建 |

现有流水线：SAM3 → DreamSim（先抠建筑再对比）

### 1.5 v2.0 变更总览

| # | 功能 | 类型 |
|---|------|------|
| F1 | 新增 DINOv3 变化检测引擎 | 新增 |
| F2 | 新增 LoFTR 配准预处理 | 新增 |
| F3 | 结果浏览翻页 API | 新增 |
| F4 | 单任务重推 + WebSocket 推送 | 新增 |
| F5 | 流水线更新（支持 DINOv3） | 变更 |
| F6 | 历史自动清理 | 新增 |
| F7 | DINOv3 + LoFTR 容器镜像构建 | 新增 |

---

## 2. F1 — 新增 DINOv3 变化检测引擎

### 2.1 功能描述

新增基于 DINOv3 模型的变化检测算法，替代 DreamSim 占用的 4 张 GPU。核心优势：抗阴影干扰（树影、云影不误判为变化）。

### 2.2 GPU 分配变更

| GPU | v1.0 | v2.0 |
|-----|------|------|
| 0~3 | SAM3 | SAM3（不变） |
| 4~7 | DreamSim | **DINOv3**（替代） |

DreamSim 代码保留，但不再独占 GPU。

### 2.3 算法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| peak_ratio | float | 0.4 | 敏感度，越小越敏感 |
| res | int | 2048 | 推理分辨率 |

### 2.4 输出

每张图产出 2 个文件：
1. **热力图版**：变化区域用颜色渐变标注
2. **干净框线版**：只画变化区域的矩形框，无热力图底色

### 2.5 引擎接口

实现 `shared/protocol.py` 中的 `EngineInterface` 抽象接口，新增文件 `worker/engine_dinov3.py`。

### 2.6 模型依赖

- DINOv3 模型权重：~1.2GB，从现有生产容器拷贝
- DINOv3 源码包：从现有生产容器拷贝
- 基础镜像：从生产容器 `2803c0f9fb6b` 导出

---

## 3. F2 — 新增 LoFTR 配准预处理

### 3.1 功能描述

在执行变化检测（DreamSim / DINOv3）之前，自动对 before 和 after 两张图进行几何配准（对齐到同一视角）。对用户不可见。

### 3.2 算法说明

**LoFTR**（Detector-Free Local Feature Matching with Transformers）：
1. 将 before 和 after 图片 resize 到最大 1024px 边长
2. 提取特征点并匹配
3. 用 RANSAC 算单应性矩阵
4. 对 after 图做透视变换，对齐到 before 视角
5. 输出配准后的 before 和 after 图片

### 3.3 失败处理

配准失败条件（满足任一）：
- 匹配特征点 < 30 个
- RANSAC 求解失败（H 矩阵为 None）
- 图片读取失败

**配准失败 → 直接标记该任务为失败**，不继续做变化检测。

### 3.4 执行规则

- 每对图都必须执行配准，不跳过
- 配准结果为临时文件，用完即删（不持久化）

### 3.5 调用位置

```
变化检测任务开始
  → 读取 before + after 图片
  → 调用 LoFTR 配准
  → 配准成功 → 将配准后的图片送入 DINOv3 / DreamSim
  → 配准失败 → 标记任务 failed，写失败原因
```

配准作为引擎内部预处理步骤，不作为独立的流水线阶段。

### 3.6 模型依赖

| 文件 | 来源（生产容器路径） | 大小 |
|------|----------------------|------|
| outdoor_ds.ckpt | /app/images/weights/outdoor_ds.ckpt | 45MB |
| LoFTR 源码 | /app/images/src/ | 240KB |
| LoFTR 配置 | /app/images/configs/ | 152KB |
| 配准主逻辑 | /app/images/register_all.py | ~5KB |

---

## 4. F3 — 结果浏览翻页 API

### 4.1 功能描述

前端可像翻相册一样逐条浏览推理结果，支持跨任务、筛选、搜索。

### 4.2 API 设计

**`GET /api/results/browse`**

请求参数：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| page | int | 否 | 页码，默认 1 |
| engine | string | 否 | 筛选模型类型：sam3 / dreamsim / dinov3 |
| status | string | 否 | 筛选状态：done / failed |
| keyword | string | 否 | 搜索图片路径关键词 |

### 4.3 响应结构

```json
{
  "total": 12345,
  "page": 1,
  "page_size": 1,
  "task": {
    "task_id": "uuid",
    "job_id": "uuid",
    "engine": "dinov3",
    "status": "done",
    "input_path": "/data/batch_shared/jobs/.../input.jpg",
    "created_at": "2026-03-06T10:00:00Z",
    "finished_at": "2026-03-06T10:01:30Z",
    "params": {"peak_ratio": 0.4, "res": 2048},
    "images": [
      {
        "label": "热力图",
        "base64": "data:image/jpeg;base64,/9j/4AAQ..."
      },
      {
        "label": "框线图",
        "base64": "data:image/jpeg;base64,/9j/4AAQ..."
      }
    ]
  }
}
```

### 4.4 业务规则

- 每页 1 条任务结果
- 图片以 Base64 编码返回
- 变化检测类（DreamSim / DINOv3）返回 2 张图（热力图 + 干净框线图），SAM3 返回 1 张图（抠像图）
- 只返回最近 1 个月内的数据
- 按完成时间倒序排列（最新的在前）
- 跨任务浏览（不局限于某个 Job）

---

## 5. F4 — 单任务重推 + WebSocket 推送

### 5.1 功能描述

用户在浏览结果时，对某张不满意，可调整参数后对该张重新推理。完成后通过 WebSocket 实时推送结果。

### 5.2 API 设计

**`POST /api/tasks/{task_id}/rerun`**

请求体：

```json
{
  "params": {
    "peak_ratio": 0.3,
    "res": 2048
  }
}
```

### 5.3 可调参数

**DINOv3：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| peak_ratio | float | 0.4 | 敏感度 |
| res | int | 2048 | 推理分辨率 |

**SAM3：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| text | string | - | 检测文本提示 |
| conf | float | - | 置信度阈值 |
| skip_dce | bool | false | 跳过暗光增强 |
| dilate | int | - | 膨胀像素 |
| fill_holes | bool | - | 填充孔洞 |
| rect_fill | bool | - | 矩形填充 |
| bbox_pad | int | - | 边框内缩像素 |

### 5.4 业务规则

- 重推任务进入**优先队列**，插队到普通批量任务之前
- 结果**覆盖原文件**（不新建）
- 重推期间，该任务状态变为 `running`
- 重推前也要执行配准（如果是变化检测类引擎）

### 5.5 WebSocket 推送

**`WS /ws/tasks/{task_id}`**

客户端连接后，当该任务重推完成时收到推送：

```json
{
  "event": "rerun_complete",
  "task_id": "uuid",
  "status": "done",
  "images": [
    {
      "label": "热力图",
      "base64": "data:image/jpeg;base64,..."
    },
    {
      "label": "框线图",
      "base64": "data:image/jpeg;base64,..."
    }
  ]
}
```

失败时推送：

```json
{
  "event": "rerun_complete",
  "task_id": "uuid",
  "status": "failed",
  "error": "配准失败：匹配特征点不足"
}
```

---

## 6. F5 — 流水线更新

### 6.1 功能描述

现有流水线 SAM3→DreamSim 扩展为支持选择第二阶段算法。

### 6.2 流水线类型

| 流水线 | 阶段 1 | 预处理 | 阶段 2 | 说明 |
|--------|--------|--------|--------|------|
| sam3_dinov3（默认） | SAM3 分割 | LoFTR 配准 | DINOv3 变化检测 | 推荐，抗阴影 |
| sam3_dreamsim | SAM3 分割 | LoFTR 配准 | DreamSim 变化检测 | 保留兼容 |

### 6.3 创建任务时指定

```json
POST /api/jobs
{
  "input_dir": "/path/to/images",
  "mode": "pipeline",
  "pipeline_type": "sam3_dinov3"
}
```

`pipeline_type` 不传时默认 `sam3_dinov3`。

### 6.4 流水线执行流程

```
阶段 1：SAM3
  → 对 before 和 after 各自做分割，抠出建筑部分
  → 全部完成后进入阶段 2

预处理：LoFTR 配准
  → 对 SAM3 输出的 before/after 抠像图做几何对齐

阶段 2：DINOv3 / DreamSim
  → 对配准后的图片做变化检测
  → 输出热力图 + 框线图
```

### 6.5 Janitor 变更

`janitor.py` 的流水线推进逻辑需适配：
- 检测阶段 1 完成后，根据 `pipeline_type` 字段决定创建 DINOv3 还是 DreamSim 阶段的任务

---

## 7. F6 — 历史自动清理

### 7.1 功能描述

Janitor 后台自动清理超过 1 个月的历史数据，避免磁盘和数据库无限增长。

### 7.2 清理规则

| 条件 | 说明 |
|------|------|
| 时间 | 任务完成/失败/取消时间 > 1 个月 |
| 状态 | 仅清理 `done`、`failed`、`cancelled` 状态的 Job |
| 排除 | `pending`、`running` 状态的 Job 不清理 |

### 7.3 清理内容

1. **数据库**：删除 jobs 表记录 + 关联的 tasks 表记录
2. **文件**：删除 `/data/batch_shared/jobs/{job_id}/` 整个目录（输入文件 + 结果文件）

### 7.4 执行频率

在 Janitor 巡检循环中执行，建议每轮巡检（10s）检查一次，或降频为每 1 小时执行一次清理扫描。

### 7.5 日志

清理时记录：清理了哪些 Job ID、释放了多少磁盘空间。

---

## 8. F7 — DINOv3 + LoFTR 容器镜像构建

### 8.1 功能描述

构建新的 Docker 镜像用于运行 DINOv3 引擎（内含 LoFTR 配准能力）。

### 8.2 从生产容器拷贝的文件

**DINOv3 相关：**
- 模型权重 (~1.2GB)
- DINOv3 源码包
- 来源容器：`2803c0f9fb6b`

**LoFTR 相关：**

| 容器路径 | 大小 |
|----------|------|
| /app/images/weights/outdoor_ds.ckpt | 45MB |
| /app/images/src/ | 240KB |
| /app/images/configs/ | 152KB |
| /app/images/register_all.py | ~5KB |

### 8.3 原则

- 不改动生产容器
- 只从中拷贝所需文件到构建上下文中

### 8.4 Dockerfile

新增 `worker/Dockerfile.dinov3`，需要：
- 基础镜像：从生产容器导出的镜像
- 安装 worker/ + shared/ + psycopg2
- 包含 LoFTR 权重 + 源码 + 配置
- 包含 DINOv3 权重 + 源码

### 8.5 docker-compose.yml 变更

替换 ds-worker-0~3 为 dinov3-worker-0~3：

```yaml
dinov3-worker-0:
  image: batch-dinov3-worker:latest
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=4
    - WORKER_ENGINE=dinov3
  volumes:
    - /data/batch_shared:/data/batch_shared
  depends_on:
    - batch-postgres
```

GPU 4~7 分别分配给 dinov3-worker-0~3。

---

## 9. 数据库变更

### 9.1 jobs 表变更

| 字段 | 类型 | 说明 |
|------|------|------|
| pipeline_type | varchar(32) | 流水线类型：sam3_dinov3（默认）/ sam3_dreamsim |

### 9.2 tasks 表变更

| 字段 | 类型 | 说明 |
|------|------|------|
| params | jsonb | 推理参数（支持重推时覆盖） |
| priority | int | 优先级，默认 0，重推任务设为 1（越大越优先） |
| engine | varchar(32) | 新增值 `dinov3` |

### 9.3 Worker 抢任务 SQL 变更

抢任务时按 `priority DESC, created_at ASC` 排序，保证重推任务优先被领取。

---

## 10. 配置变更（shared/config.py）

新增配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| DINOV3_PEAK_RATIO | 0.4 | DINOv3 默认敏感度 |
| DINOV3_RES | 2048 | DINOv3 默认分辨率 |
| LOFTR_WEIGHT_PATH | /app/weights/outdoor_ds.ckpt | LoFTR 模型权重路径 |
| LOFTR_CONFIG_PATH | /app/configs/loftr/outdoor/loftr_ds.py | LoFTR 配置文件路径 |
| LOFTR_MIN_MATCHES | 30 | 配准最少匹配点数 |
| CLEANUP_RETENTION_DAYS | 30 | 历史清理保留天数 |
| RERUN_PRIORITY | 1 | 重推任务优先级 |

---

## 11. 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新增 | worker/engine_dinov3.py | DINOv3 引擎实现 |
| 新增 | worker/engine_loftr.py | LoFTR 配准模块封装 |
| 新增 | worker/Dockerfile.dinov3 | DINOv3 容器镜像 |
| 修改 | gateway/app.py | 新增浏览翻页 API + 重推 API + WebSocket 端点 |
| 修改 | gateway/janitor.py | 流水线推进适配 DINOv3 + 历史自动清理 |
| 修改 | gateway/scheduler.py | 支持 pipeline_type 参数 |
| 修改 | gateway/db.py | 新增浏览查询方法 + 重推方法 + 清理方法 + priority 排序 |
| 修改 | shared/protocol.py | 新增 dinov3 引擎枚举 + 注册 |
| 修改 | shared/config.py | 新增配置项 |
| 修改 | sql/init.sql | 表结构变更 |
| 修改 | docker-compose.yml | DreamSim Worker → DINOv3 Worker |
| 修改 | worker/base_worker.py | 抢任务支持 priority 排序 |
| 修改 | worker/engine_dreamsim.py | 推理前调用 LoFTR 配准 |
