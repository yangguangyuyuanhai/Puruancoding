"""
FastAPI 入口 - HTTP 路由、参数校验、响应格式化
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from typing import Optional
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from shared.config import (
    SHARED_ROOT, JOBS_DIR, SPILLOVER_DIR, HEARTBEAT_DIR,
    GATEWAY_HOST, GATEWAY_PORT,
)
from shared.protocol import JobMeta, ModelType, JobStatus
from gateway.db import init_pool, close_pool, JobRepository, TaskRepository
from gateway.indexer import index_sam3_input, index_dreamsim_input
from gateway.scheduler import (
    compute_bucket_count, create_job_dirs, write_job_meta,
    distribute_sam3_tasks, distribute_dreamsim_tasks,
)
from gateway.janitor import Janitor
from gateway.scaler import AutoScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

janitor = Janitor()
scaler = AutoScaler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup & shutdown"""
    # Startup
    await init_pool()
    # 创建共享目录
    for d in [SHARED_ROOT, JOBS_DIR, SPILLOVER_DIR, HEARTBEAT_DIR]:
        os.makedirs(d, exist_ok=True)
    # 启动后台任务
    janitor_task = asyncio.create_task(janitor.start())
    scaler_task = asyncio.create_task(scaler.start())
    logger.info("Gateway started")
    yield
    # Shutdown
    janitor.stop()
    scaler.stop()
    janitor_task.cancel()
    scaler_task.cancel()
    await close_pool()
    logger.info("Gateway stopped")


app = FastAPI(
    title="违章建筑检测 - 批量推理平台",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Request/Response Models
# ============================================================
class CreateJobRequest(BaseModel):
    input_path: str = Field(..., description="输入目录或 ZIP 文件路径 (服务器本地路径)")
    model_type: str = Field(..., description="模型类型: sam3 | dreamsim")
    # SAM3 参数
    text: Optional[str] = Field(None, description="SAM3 检测提示词")
    conf: float = Field(0.2, description="SAM3 置信度阈值")
    skip_dce: bool = Field(False, description="跳过 Zero-DCE 暗光增强")
    dilate: int = Field(0, description="边缘膨胀核大小")
    fill_holes: bool = Field(False, description="填充孔洞")
    rect_fill: bool = Field(False, description="矩形修复")
    bbox_pad: int = Field(0, description="框外扩像素")
    # DreamSim 参数
    thresh: float = Field(0.3, description="DreamSim 检测阈值")
    crop: int = Field(224, description="DreamSim 切片大小")
    step: int = Field(0, description="DreamSim 步长 (0=自动)")
    batch: int = Field(16, description="DreamSim 批次大小")
    # 调度参数
    num_workers: int = Field(24, description="Worker 数量 (用于 Bucket 划分)")


class JobResponse(BaseModel):
    job_id: str
    model_type: str
    status: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    progress_pct: float
    input_path: str
    params: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    workers: list[dict]
    active_jobs: int


# ============================================================
# API Routes
# ============================================================
@app.post("/api/jobs", response_model=dict)
async def create_job(req: CreateJobRequest):
    """注册批量检测任务"""
    # 验证模型类型
    if req.model_type not in (ModelType.SAM3.value, ModelType.DREAMSIM.value):
        raise HTTPException(400, f"Invalid model_type: {req.model_type}")

    # 验证输入路径
    if not os.path.exists(req.input_path):
        raise HTTPException(400, f"Input path does not exist: {req.input_path}")

    job_id = str(uuid.uuid4())
    extract_base = os.path.join(JOBS_DIR, job_id)

    try:
        if req.model_type == ModelType.SAM3.value:
            if not req.text:
                raise HTTPException(400, "SAM3 model requires 'text' parameter")

            # 索引图片
            image_paths = index_sam3_input(req.input_path, extract_base)
            if not image_paths:
                raise HTTPException(400, "No images found in input path")

            total_tasks = len(image_paths)
            params = {
                "text": req.text,
                "conf": req.conf,
                "skip_dce": req.skip_dce,
                "dilate": req.dilate,
                "fill_holes": req.fill_holes,
                "rect_fill": req.rect_fill,
                "bbox_pad": req.bbox_pad,
            }

            # 计算 Bucket
            num_buckets = compute_bucket_count(total_tasks, req.num_workers)

            # 在数据库中创建 Job
            await JobRepository.create_job(req.model_type, req.input_path, params, total_tasks, job_id=job_id)

            # 分发任务到文件系统 + 数据库
            task_descriptors = distribute_sam3_tasks(job_id, image_paths, params, num_buckets)
            await TaskRepository.bulk_create(job_id, task_descriptors)

        else:  # dreamsim
            # 索引图片对
            image_pairs = index_dreamsim_input(req.input_path, extract_base)
            if not image_pairs:
                raise HTTPException(400, "No image pairs found")

            total_tasks = len(image_pairs)
            params = {
                "thresh": req.thresh,
                "crop": req.crop,
                "step": req.step,
                "batch": req.batch,
            }

            num_buckets = compute_bucket_count(total_tasks, req.num_workers)

            await JobRepository.create_job(req.model_type, req.input_path, params, total_tasks, job_id=job_id)

            task_descriptors = distribute_dreamsim_tasks(job_id, image_pairs, params, num_buckets)
            await TaskRepository.bulk_create(job_id, task_descriptors)

        # 写入 meta.json
        meta = JobMeta(
            job_id=job_id,
            model_type=req.model_type,
            input_path=req.input_path,
            params=params,
            total_tasks=total_tasks,
        )
        write_job_meta(job_id, meta)

        # 更新状态为 running
        await JobRepository.update_status(job_id, JobStatus.RUNNING.value)

        logger.info(f"Job {job_id} created: model={req.model_type}, tasks={total_tasks}")
        return {"job_id": job_id, "total_tasks": total_tasks, "status": "running"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create job: {e}", exc_info=True)
        raise HTTPException(500, f"Internal error: {str(e)}")


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """查询任务进度"""
    job = await JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    total = job["total_tasks"]
    completed = job["completed_tasks"]
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
    """列出所有任务"""
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
    """取消任务"""
    job = await JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    await JobRepository.delete_job(job_id)
    logger.info(f"Job {job_id} cancelled")
    return {"job_id": job_id, "status": "cancelled"}


@app.post("/api/jobs/{job_id}/retry")
async def retry_failed_tasks(job_id: str):
    """重试失败的任务"""
    job = await JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    count = await TaskRepository.retry_failed_tasks(job_id)
    if count > 0:
        await JobRepository.update_status(job_id, JobStatus.RUNNING.value)

    return {"job_id": job_id, "retried_tasks": count}


@app.get("/api/jobs/{job_id}/failed")
async def get_failed_tasks(job_id: str):
    """获取失败任务列表"""
    job = await JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    tasks = await TaskRepository.get_failed_tasks(job_id)
    return [
        {
            "task_id": str(t["id"]),
            "image_path": t.get("image_path"),
            "image_pair": json.loads(t["image_pair"]) if t.get("image_pair") else None,
            "error_msg": t.get("error_msg"),
        }
        for t in tasks
    ]


@app.get("/api/jobs/{job_id}/export")
async def export_results(job_id: str):
    """导出结果为流式 ZIP 下载"""
    job = await JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    results_dir = os.path.join(JOBS_DIR, job_id, "results")
    spillover_dir = os.path.join(SPILLOVER_DIR, job_id)

    # 收集所有结果文件
    result_files = []
    for d in [results_dir, spillover_dir]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                result_files.append(os.path.join(d, f))

    if not result_files:
        raise HTTPException(404, "No result files found")

    def generate_zip():
        buf = BytesIO()
        with ZipFile(buf, "w", ZIP_DEFLATED) as zf:
            for fpath in result_files:
                arcname = os.path.basename(fpath)
                zf.write(fpath, arcname)
        buf.seek(0)
        yield buf.read()

    filename = f"batch_results_{job_id[:8]}.zip"
    return StreamingResponse(
        generate_zip(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """系统健康检查"""
    workers = []
    if os.path.isdir(HEARTBEAT_DIR):
        import time
        now = time.time()
        for fname in sorted(os.listdir(HEARTBEAT_DIR)):
            if not fname.endswith(".ts"):
                continue
            fpath = os.path.join(HEARTBEAT_DIR, fname)
            try:
                with open(fpath, "r") as f:
                    ts = float(f.read().strip())
                worker_id = fname.replace(".ts", "")
                age = now - ts
                workers.append({
                    "worker_id": worker_id,
                    "last_heartbeat": datetime.fromtimestamp(ts).isoformat(),
                    "age_seconds": round(age, 1),
                    "healthy": age < 30,
                })
            except (ValueError, OSError):
                pass

    # 统计活跃 Job
    jobs = await JobRepository.list_jobs(limit=100)
    active_jobs = sum(1 for j in jobs if j["status"] in ("pending", "running"))

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
    import uvicorn
    uvicorn.run(
        "gateway.app:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
