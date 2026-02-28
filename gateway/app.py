"""
FastAPI 入口 - HTTP 路由、参数校验、响应格式化
本模块是系统的 API 层，只负责：
  1. 接收 HTTP 请求并校验参数
  2. 委托给 Service 层处理业务逻辑
  3. 格式化并返回 HTTP 响应
不包含任何业务逻辑实现
"""
from __future__ import annotations  # 支持类型注解的前向引用

import asyncio    # 异步任务调度
import json       # JSON 解析
import logging    # 日志
import os         # 文件系统操作
import shutil     # 文件移动
import tempfile   # 临时文件
import uuid       # UUID 生成
from contextlib import asynccontextmanager  # 异步上下文管理器
from datetime import datetime    # 时间格式化
from io import BytesIO           # 内存字节流
from typing import Optional      # 可选类型注解
from zipfile import ZipFile, ZIP_DEFLATED  # ZIP 打包

from fastapi import FastAPI, HTTPException, Query  # Web 框架核心组件
from fastapi.responses import StreamingResponse     # 流式响应（用于 ZIP 下载）
from pydantic import BaseModel, Field               # 请求/响应数据模型

# 导入共享配置（路径、端口等）
from shared.config import (
    SHARED_ROOT, JOBS_DIR, SPILLOVER_DIR, HEARTBEAT_DIR,
    GATEWAY_HOST, GATEWAY_PORT,
)
# 导入数据协议（枚举、数据结构）
from shared.protocol import JobMeta, ModelType, JobStatus
# 导入数据库操作层
from gateway.db import init_pool, close_pool, JobRepository, TaskRepository
# 导入文件索引器（扫描目录/解析 ZIP）
from gateway.indexer import index_sam3_input, index_dreamsim_input
# 导入任务调度器（分桶、分发任务）
from gateway.scheduler import (
    compute_bucket_count, create_job_dirs, write_job_meta,
    distribute_sam3_tasks, distribute_dreamsim_tasks,
)
# 导入后台守护线程
from gateway.janitor import Janitor   # 心跳巡检、任务回收、spillover
from gateway.scaler import AutoScaler  # 弹性伸缩控制器

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 创建后台守护实例（随 Gateway 生命周期启停）
janitor = Janitor()    # 负责心跳巡检、任务超时回收、spillover 落盘
scaler = AutoScaler()  # 负责根据队列深度动态调整 Worker 数量


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器 (FastAPI lifespan)
    startup: 初始化数据库连接池、创建共享目录、启动后台任务
    shutdown: 停止后台任务、关闭数据库连接池
    """
    # === Startup ===
    await init_pool()  # 初始化 PostgreSQL asyncpg 连接池
    # 创建共享卷所需的目录结构
    for d in [SHARED_ROOT, JOBS_DIR, SPILLOVER_DIR, HEARTBEAT_DIR]:
        os.makedirs(d, exist_ok=True)
    # 启动后台异步任务（Janitor 和 AutoScaler 各自独立循环运行）
    janitor_task = asyncio.create_task(janitor.start())
    scaler_task = asyncio.create_task(scaler.start())
    logger.info("Gateway started")
    yield  # 应用运行期间挂起
    # === Shutdown ===
    janitor.stop()       # 通知 Janitor 停止循环
    scaler.stop()        # 通知 AutoScaler 停止循环
    janitor_task.cancel()  # 取消异步任务
    scaler_task.cancel()
    await close_pool()   # 关闭数据库连接池
    logger.info("Gateway stopped")


# 创建 FastAPI 应用实例
app = FastAPI(
    title="违章建筑检测 - 批量推理平台",  # Swagger 文档标题
    version="1.0.0",                       # API 版本
    lifespan=lifespan,                     # 绑定生命周期管理器
)


# ============================================================
# 请求/响应数据模型 (Pydantic 自动校验)
# ============================================================
class CreateJobRequest(BaseModel):
    """创建批量任务的请求体"""
    input_path: str = Field(..., description="输入目录或 ZIP 文件路径 (服务器本地路径)")
    model_type: str = Field(..., description="模型类型: sam3 | dreamsim")
    # SAM3 专用参数
    text: Optional[str] = Field(None, description="SAM3 检测提示词，描述要检测的目标物体（如 'building'）")
    conf: float = Field(0.2, description="SAM3 置信度阈值，低于此值的检测结果被过滤（范围 0~1）")
    skip_dce: bool = Field(False, description="跳过 Zero-DCE 暗光增强（光照条件好时可跳过以加速）")
    dilate: int = Field(0, description="形态学膨胀核大小（像素），用于扩大掩码边缘，弥补分割精度不足")
    fill_holes: bool = Field(False, description="填充掩码内部孔洞（使用轮廓检测填充封闭区域）")
    rect_fill: bool = Field(False, description="在检测框区域内做闭运算修复（填补掩码中的小间隙）")
    bbox_pad: int = Field(0, description="检测框向外扩展的像素数（扩大处理区域，防止边缘被截断）")
    # DreamSim 专用参数
    thresh: float = Field(0.3, description="DreamSim 变化检测阈值，高于此值的区域判定为变化（范围 0~1）")
    crop: int = Field(224, description="DreamSim 滑窗 patch 大小（像素），224 匹配 DreamSim 默认输入尺寸")
    step: int = Field(0, description="DreamSim 滑窗步长（0 = 自动设为 patch_size//2，即 50%% 重叠）")
    batch: int = Field(16, description="DreamSim GPU 批推理大小，增大可加速但消耗更多显存")
    # 调度参数
    num_workers: int = Field(24, description="Worker 数量，用于 Bucket 划分（默认 24 = 8 GPU × 3 进程/卡）")


class JobResponse(BaseModel):
    """任务进度查询的响应体"""
    job_id: str                          # 任务 ID
    model_type: str                      # 模型类型
    status: str                          # 当前状态
    total_tasks: int                     # 子任务总数
    completed_tasks: int                 # 已完成数
    failed_tasks: int                    # 失败数
    progress_pct: float                  # 进度百分比
    input_path: str                      # 输入路径
    params: Optional[dict] = None        # 推理参数
    created_at: Optional[str] = None     # 创建时间
    updated_at: Optional[str] = None     # 最后更新时间


class HealthResponse(BaseModel):
    """系统健康检查的响应体"""
    status: str           # healthy | degraded
    workers: list[dict]   # 各 Worker 的心跳状态列表
    active_jobs: int      # 当前活跃的 Job 数量


# ============================================================
# API 路由
# ============================================================
@app.post("/api/jobs", response_model=dict)
async def create_job(req: CreateJobRequest):
    """
    注册批量检测任务
    流程: 验证参数 → 索引图片 → 分桶 → 写入 DB 和文件系统 → 返回 job_id
    """
    # 验证模型类型是否合法
    if req.model_type not in (ModelType.SAM3.value, ModelType.DREAMSIM.value):
        raise HTTPException(400, f"Invalid model_type: {req.model_type}")

    # 验证输入路径是否存在
    if not os.path.exists(req.input_path):
        raise HTTPException(400, f"Input path does not exist: {req.input_path}")

    # 生成全局唯一的 Job ID
    job_id = str(uuid.uuid4())
    # ZIP 解压基础目录（如果输入是 ZIP 文件）
    extract_base = os.path.join(JOBS_DIR, job_id)

    try:
        if req.model_type == ModelType.SAM3.value:
            # --- SAM3 分割检测任务 ---
            # SAM3 必须提供文本提示词
            if not req.text:
                raise HTTPException(400, "SAM3 model requires 'text' parameter")

            # 索引图片：扫描目录或解压 ZIP，得到图片路径列表
            image_paths = index_sam3_input(req.input_path, extract_base)
            if not image_paths:
                raise HTTPException(400, "No images found in input path")

            total_tasks = len(image_paths)  # 子任务总数 = 图片数
            # 组装推理参数字典
            params = {
                "text": req.text,            # 检测提示词
                "conf": req.conf,            # 置信度阈值
                "skip_dce": req.skip_dce,    # 是否跳过暗光增强
                "dilate": req.dilate,        # 膨胀核大小
                "fill_holes": req.fill_holes,  # 填充孔洞
                "rect_fill": req.rect_fill,    # 矩形修复
                "bbox_pad": req.bbox_pad,      # 框外扩像素
            }

            # 计算 Bucket 数量（按 Worker 数量划分，保证空间局部性）
            num_buckets = compute_bucket_count(total_tasks, req.num_workers)

            # 在数据库中创建 Job 记录（传入 job_id 确保 DB 和文件系统使用同一 ID）
            await JobRepository.create_job(req.model_type, req.input_path, params, total_tasks, job_id=job_id)

            # 分发任务：写入文件系统 pending/ 目录 + 返回任务描述符列表
            task_descriptors = distribute_sam3_tasks(job_id, image_paths, params, num_buckets)
            # 批量写入数据库 tasks 表
            await TaskRepository.bulk_create(job_id, task_descriptors)

        else:  # dreamsim
            # --- DreamSim 变化检测任务 ---
            # 索引图片对：扫描 before/ 和 after/ 子目录，同名文件自动配对
            image_pairs = index_dreamsim_input(req.input_path, extract_base)
            if not image_pairs:
                raise HTTPException(400, "No image pairs found")

            total_tasks = len(image_pairs)  # 子任务总数 = 图片对数
            # 组装推理参数字典
            params = {
                "thresh": req.thresh,  # 变化检测阈值
                "crop": req.crop,      # 滑窗切片大小
                "step": req.step,      # 滑窗步长
                "batch": req.batch,    # GPU 批推理大小
            }

            num_buckets = compute_bucket_count(total_tasks, req.num_workers)

            # 在数据库中创建 Job 记录
            await JobRepository.create_job(req.model_type, req.input_path, params, total_tasks, job_id=job_id)

            # 分发任务到文件系统 + 数据库
            task_descriptors = distribute_dreamsim_tasks(job_id, image_pairs, params, num_buckets)
            await TaskRepository.bulk_create(job_id, task_descriptors)

        # 写入 meta.json 到共享卷（供调试和审计）
        meta = JobMeta(
            job_id=job_id,
            model_type=req.model_type,
            input_path=req.input_path,
            params=params,
            total_tasks=total_tasks,
        )
        write_job_meta(job_id, meta)

        # 更新 Job 状态为 running（Worker 可以开始抢占任务了）
        await JobRepository.update_status(job_id, JobStatus.RUNNING.value)

        logger.info(f"Job {job_id} created: model={req.model_type}, tasks={total_tasks}")
        return {"job_id": job_id, "total_tasks": total_tasks, "status": "running"}

    except HTTPException:
        raise  # 业务异常直接抛出
    except Exception as e:
        logger.error(f"Failed to create job: {e}", exc_info=True)
        raise HTTPException(500, f"Internal error: {str(e)}")


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """查询单个任务的进度和状态"""
    # 从数据库查询 Job 记录
    job = await JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    total = job["total_tasks"]
    completed = job["completed_tasks"]
    # 计算进度百分比
    progress = (completed / total * 100) if total > 0 else 0

    return JobResponse(
        job_id=str(job["id"]),
        model_type=job["model_type"],
        status=job["status"],
        total_tasks=total,
        completed_tasks=completed,
        failed_tasks=job["failed_tasks"],
        progress_pct=round(progress, 2),
        input_path=job["input_path"],
        params=json.loads(job["params"]) if job["params"] else None,
        created_at=job["created_at"].isoformat() if job.get("created_at") else None,
        updated_at=job["updated_at"].isoformat() if job.get("updated_at") else None,
    )


@app.get("/api/jobs", response_model=list[JobResponse])
async def list_jobs(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    """列出所有任务，按创建时间倒序，支持分页"""
    jobs = await JobRepository.list_jobs(limit=limit, offset=offset)
    results = []
    for job in jobs:
        total = job["total_tasks"]
        completed = job["completed_tasks"]
        progress = (completed / total * 100) if total > 0 else 0
        results.append(JobResponse(
            job_id=str(job["id"]),
            model_type=job["model_type"],
            status=job["status"],
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=job["failed_tasks"],
            progress_pct=round(progress, 2),
            input_path=job["input_path"],
            params=json.loads(job["params"]) if job["params"] else None,
            created_at=job["created_at"].isoformat() if job.get("created_at") else None,
            updated_at=job["updated_at"].isoformat() if job.get("updated_at") else None,
        ))
    return results


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """取消任务，将状态置为 cancelled，正在运行的子任务处理完当前图后停止"""
    job = await JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    await JobRepository.delete_job(job_id)  # 软删除：状态改为 cancelled
    logger.info(f"Job {job_id} cancelled")
    return {"job_id": job_id, "status": "cancelled"}


@app.post("/api/jobs/{job_id}/retry")
async def retry_failed_tasks(job_id: str):
    """将该 Job 下所有 failed 状态的子任务重置为 pending，Worker 会自动重新抢占"""
    job = await JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    # 重置 failed → pending
    count = await TaskRepository.retry_failed_tasks(job_id)
    # 如果有重试的任务，将 Job 状态改回 running
    if count > 0:
        await JobRepository.update_status(job_id, JobStatus.RUNNING.value)

    return {"job_id": job_id, "retried_tasks": count}


@app.get("/api/jobs/{job_id}/failed")
async def get_failed_tasks(job_id: str):
    """获取失败任务列表，包含失败原因，用于排查问题"""
    job = await JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    tasks = await TaskRepository.get_failed_tasks(job_id)
    return [
        {
            "task_id": str(t["id"]),
            "image_path": t.get("image_path"),           # SAM3 图片路径
            "image_pair": json.loads(t["image_pair"]) if t.get("image_pair") else None,  # DreamSim 图片对
            "error_msg": t.get("error_msg"),              # 错误信息
        }
        for t in tasks
    ]


@app.get("/api/jobs/{job_id}/export")
async def export_results(job_id: str):
    """
    导出结果为流式 ZIP 下载
    收集 results 目录和 spillover 目录下的所有结果文件，打包为 ZIP 流式返回
    """
    job = await JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    # 结果文件可能分布在两个目录：正常结果目录 + spillover 溢出目录
    results_dir = os.path.join(JOBS_DIR, job_id, "results")
    spillover_dir = os.path.join(SPILLOVER_DIR, job_id)

    # 收集所有结果文件路径
    result_files = []
    for d in [results_dir, spillover_dir]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                result_files.append(os.path.join(d, f))

    if not result_files:
        raise HTTPException(404, "No result files found")

    def generate_zip():
        """
        生成器：在内存中创建 ZIP 并 yield 数据块
        注意：当前实现是将所有结果文件打包到内存中的 BytesIO 缓冲区，
        适用于结果文件总量在几百 MB 以内的场景。
        如果结果文件总量超过服务器可用内存，应改用流式写入临时文件的方式。
        """
        buf = BytesIO()
        with ZipFile(buf, "w", ZIP_DEFLATED) as zf:
            for fpath in result_files:
                arcname = os.path.basename(fpath)  # ZIP 内只保留文件名，不含路径
                zf.write(fpath, arcname)
        buf.seek(0)
        yield buf.read()

    # 生成下载文件名（取 job_id 前 8 位）
    filename = f"batch_results_{job_id[:8]}.zip"
    return StreamingResponse(
        generate_zip(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    系统健康检查
    返回各 Worker 的心跳状态和活跃 Job 数量
    """
    workers = []
    if os.path.isdir(HEARTBEAT_DIR):
        import time
        now = time.time()
        # 遍历心跳目录，读取每个 Worker 的最后心跳时间
        for fname in sorted(os.listdir(HEARTBEAT_DIR)):
            if not fname.endswith(".ts"):  # 只处理 .ts 后缀的心跳文件
                continue
            fpath = os.path.join(HEARTBEAT_DIR, fname)
            try:
                with open(fpath, "r") as f:
                    ts = float(f.read().strip())  # 读取 unix timestamp
                worker_id = fname.replace(".ts", "")
                age = now - ts  # 距离上次心跳的秒数
                workers.append({
                    "worker_id": worker_id,
                    "last_heartbeat": datetime.fromtimestamp(ts).isoformat(),
                    "age_seconds": round(age, 1),
                    "healthy": age < 30,  # 30 秒内有心跳视为健康
                })
            except (ValueError, OSError):
                pass  # 忽略无法读取的心跳文件

    # 统计活跃 Job 数量
    jobs = await JobRepository.list_jobs(limit=100)
    active_jobs = sum(1 for j in jobs if j["status"] in ("pending", "running"))

    # 判断系统整体健康状态
    healthy_workers = sum(1 for w in workers if w["healthy"])
    status = "healthy" if healthy_workers > 0 or active_jobs == 0 else "degraded"

    return HealthResponse(
        status=status,
        workers=workers,
        active_jobs=active_jobs,
    )


# ============================================================
# 启动入口
# ============================================================
def main():
    """使用 uvicorn 启动 FastAPI 应用"""
    import uvicorn
    uvicorn.run(
        "gateway.app:app",          # 应用模块路径
        host=GATEWAY_HOST,          # 监听地址 (默认 0.0.0.0)
        port=GATEWAY_PORT,          # 监听端口 (默认 8090)
        reload=False,               # 生产环境关闭自动重载
        log_level="info",           # 日志级别
    )


if __name__ == "__main__":
    main()
