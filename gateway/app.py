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
import base64     # Base64 编码
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

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect  # Web 框架核心组件
from fastapi.responses import StreamingResponse     # 流式响应（用于 ZIP 下载）
from pydantic import BaseModel, Field               # 请求/响应数据模型

# 导入共享配置（路径、端口等）
from shared.config import (
    SHARED_ROOT, JOBS_DIR, SPILLOVER_DIR, HEARTBEAT_DIR,
    GATEWAY_HOST, GATEWAY_PORT, RERUN_PRIORITY,
)
# 导入数据协议（枚举、数据结构）
from shared.protocol import JobMeta, ModelType, JobStatus
# 导入数据库操作层
from gateway.db import init_pool, close_pool, JobRepository, TaskRepository
# 导入文件索引器（扫描目录/解析 ZIP）
from gateway.indexer import index_sam3_input, index_dreamsim_input, index_pipeline_input
# 导入任务调度器（分桶、分发任务）
from gateway.scheduler import (
    compute_bucket_count, create_job_dirs, write_job_meta,
    distribute_sam3_tasks, distribute_dreamsim_tasks, distribute_dinov3_tasks,
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
    version="2.0.0",                       # API 版本
    lifespan=lifespan,                     # 绑定生命周期管理器
)


# ============================================================
# 请求/响应数据模型 (Pydantic 自动校验)
# ============================================================
class CreateJobRequest(BaseModel):
    """创建批量任务的请求体"""
    input_path: str = Field(..., description="输入目录或 ZIP 文件路径 (服务器本地路径)")
    model_type: str = Field(..., description="模型类型: sam3 | dreamsim | dinov3 | pipeline")
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
    # DINOv3 专用参数
    peak_ratio: float = Field(0.4, description="DINOv3 敏感度，越小越敏感（范围 0~1）")
    res: int = Field(2048, description="DINOv3 推理分辨率")
    # 流水线参数
    pipeline_type: str = Field("sam3_dinov3", description="流水线类型: sam3_dinov3 (默认) | sam3_dreamsim")
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
    # 流水线扩展字段
    phase: Optional[str] = None          # 当前流水线阶段 (sam3/dreamsim)
    children: Optional[list[dict]] = None  # 子任务进度列表


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
    valid_types = (ModelType.SAM3.value, ModelType.DREAMSIM.value,
                   ModelType.DINOV3.value, ModelType.PIPELINE.value)
    if req.model_type not in valid_types:
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

        elif req.model_type == ModelType.PIPELINE.value:
            # --- SAM3→DINOv3/DreamSim 流水线任务 ---
            # 流水线必须提供 SAM3 的文本提示词
            if not req.text:
                raise HTTPException(400, "Pipeline model requires 'text' parameter for SAM3 phase")

            # 验证 pipeline_type
            if req.pipeline_type not in ("sam3_dinov3", "sam3_dreamsim"):
                raise HTTPException(400, f"Invalid pipeline_type: {req.pipeline_type}")

            # 索引 before/ 和 after/ 目录
            pipeline_input = index_pipeline_input(req.input_path, extract_base)
            before_images = pipeline_input["before_images"]
            after_images = pipeline_input["after_images"]

            # 组装 SAM3 推理参数
            sam3_params = {
                "text": req.text,
                "conf": req.conf,
                "skip_dce": req.skip_dce,
                "dilate": req.dilate,
                "fill_holes": req.fill_holes,
                "rect_fill": req.rect_fill,
                "bbox_pad": req.bbox_pad,
            }
            # 根据 pipeline_type 组装第二阶段参数
            if req.pipeline_type == "sam3_dreamsim":
                comparison_params = {
                    "thresh": req.thresh,
                    "crop": req.crop,
                    "step": req.step,
                    "batch": req.batch,
                }
            else:  # sam3_dinov3
                comparison_params = {
                    "peak_ratio": req.peak_ratio,
                    "res": req.res,
                }
            # parent job 保存所有参数
            pipeline_params = {
                "sam3": sam3_params,
                "dreamsim": comparison_params if req.pipeline_type == "sam3_dreamsim" else {},
                "dinov3": comparison_params if req.pipeline_type == "sam3_dinov3" else {},
                "num_workers": req.num_workers,
            }

            # 1. 创建 parent pipeline job（无自己的 tasks，total_tasks=0）
            total_tasks = 0
            await JobRepository.create_job(
                ModelType.PIPELINE.value, req.input_path, pipeline_params, total_tasks,
                job_id=job_id, pipeline_type=req.pipeline_type,
            )

            # 2. 创建 SAM3 Before child job
            before_job_id = str(uuid.uuid4())
            before_total = len(before_images)
            before_buckets = compute_bucket_count(before_total, req.num_workers)
            await JobRepository.create_job(
                ModelType.SAM3.value, req.input_path, sam3_params, before_total,
                job_id=before_job_id, parent_job_id=job_id, pipeline_phase="sam3_before",
            )
            before_tasks = distribute_sam3_tasks(before_job_id, before_images, sam3_params, before_buckets)
            await TaskRepository.bulk_create(before_job_id, before_tasks)
            await JobRepository.update_status(before_job_id, JobStatus.RUNNING.value)

            # 3. 创建 SAM3 After child job
            after_job_id = str(uuid.uuid4())
            after_total = len(after_images)
            after_buckets = compute_bucket_count(after_total, req.num_workers)
            await JobRepository.create_job(
                ModelType.SAM3.value, req.input_path, sam3_params, after_total,
                job_id=after_job_id, parent_job_id=job_id, pipeline_phase="sam3_after",
            )
            after_tasks = distribute_sam3_tasks(after_job_id, after_images, sam3_params, after_buckets)
            await TaskRepository.bulk_create(after_job_id, after_tasks)
            await JobRepository.update_status(after_job_id, JobStatus.RUNNING.value)

            # 写入 meta.json（pipeline parent）
            meta = JobMeta(
                job_id=job_id,
                model_type=ModelType.PIPELINE.value,
                input_path=req.input_path,
                params=pipeline_params,
                total_tasks=total_tasks,
            )
            write_job_meta(job_id, meta)

            # 更新 parent job 状态为 running
            await JobRepository.update_status(job_id, JobStatus.RUNNING.value)

            logger.info(
                f"Pipeline job {job_id} created: before={before_total}, after={after_total}, "
                f"children=[{before_job_id}, {after_job_id}]"
            )
            return {
                "job_id": job_id,
                "total_tasks": total_tasks,
                "status": "running",
                "children": [
                    {"job_id": before_job_id, "phase": "sam3_before", "total_tasks": before_total},
                    {"job_id": after_job_id, "phase": "sam3_after", "total_tasks": after_total},
                ],
            }

        elif req.model_type == ModelType.DINOV3.value:
            # --- DINOv3 变化检测任务 ---
            image_pairs = index_dreamsim_input(req.input_path, extract_base)
            if not image_pairs:
                raise HTTPException(400, "No image pairs found")

            total_tasks = len(image_pairs)
            params = {
                "peak_ratio": req.peak_ratio,
                "res": req.res,
            }

            num_buckets = compute_bucket_count(total_tasks, req.num_workers)
            await JobRepository.create_job(req.model_type, req.input_path, params, total_tasks, job_id=job_id)
            task_descriptors = distribute_dinov3_tasks(job_id, image_pairs, params, num_buckets)
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

    # 流水线扩展：如果是 pipeline job，返回子任务进度
    phase = None
    children_info = None
    if job["model_type"] == ModelType.PIPELINE.value:
        children = await JobRepository.get_children_jobs(job_id)
        children_info = []
        has_dreamsim = False
        sam3_all_done = True
        for child in children:
            c_total = child["total_tasks"]
            c_completed = child["completed_tasks"]
            c_progress = (c_completed / c_total * 100) if c_total > 0 else 0
            children_info.append({
                "job_id": str(child["id"]),
                "phase": child["pipeline_phase"],
                "status": child["status"],
                "total_tasks": c_total,
                "completed_tasks": c_completed,
                "failed_tasks": child["failed_tasks"],
                "progress_pct": round(c_progress, 2),
            })
            if child["pipeline_phase"] in ("dreamsim", "dinov3"):
                has_dreamsim = True
            if child["pipeline_phase"] in ("sam3_before", "sam3_after") and child["status"] != "completed":
                sam3_all_done = False
        # 确定当前流水线阶段
        if has_dreamsim:
            phase = "comparison"  # dreamsim or dinov3 phase
        elif sam3_all_done and children:
            phase = "sam3_done"
        else:
            phase = "sam3"
        # pipeline job 的总进度 = 所有 children 的聚合进度
        total = sum(c["total_tasks"] for c in children_info)
        completed = sum(c["completed_tasks"] for c in children_info)
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
        phase=phase,
        children=children_info,
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
    对 pipeline job：只导出 DreamSim child job 的结果（最终结果）
    """
    job = await JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    # 确定需要收集结果的 job_id 列表
    if job["model_type"] == ModelType.PIPELINE.value:
        # pipeline job：只收集 DreamSim child 的结果
        children = await JobRepository.get_children_jobs(job_id)
        export_job_ids = [
            str(c["id"]) for c in children if c["pipeline_phase"] in ("dreamsim", "dinov3")
        ]
        if not export_job_ids:
            raise HTTPException(404, "Comparison phase not yet created or completed")
    else:
        export_job_ids = [job_id]

    # 收集所有结果文件路径
    result_files = []
    for eid in export_job_ids:
        results_dir = os.path.join(JOBS_DIR, eid, "results")
        spillover_dir = os.path.join(SPILLOVER_DIR, eid)
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
# 结果浏览翻页 API
# ============================================================
@app.get("/api/results/browse")
async def browse_results(
    page: int = Query(1, ge=1),
    engine: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
):
    """
    翻页浏览推理结果，每页 1 条
    支持按引擎类型、状态、路径关键词筛选
    返回结果图片的 Base64 编码
    """
    total, task_row = await TaskRepository.browse_results(
        page=page, engine=engine, status=status, keyword=keyword,
    )

    if task_row is None:
        return {"total": total, "page": page, "page_size": 1, "task": None}

    # 构造返回数据
    task_engine = task_row.get("engine", "unknown")
    images = []

    if task_row.get("status") == "done" and task_row.get("result_path"):
        # 读取结果图片转 Base64
        result_path = task_row["result_path"]
        if os.path.exists(result_path):
            images.append({
                "label": "热力图",
                "base64": _file_to_base64(result_path),
            })

        # 变化检测类返回 2 张图（热力图 + clean）
        if task_engine in ("dreamsim", "dinov3"):
            metadata = task_row.get("metadata")
            if metadata:
                meta = json.loads(metadata) if isinstance(metadata, str) else metadata
                clean_path = meta.get("clean_result")
                if clean_path and os.path.exists(clean_path):
                    images.append({
                        "label": "框线图",
                        "base64": _file_to_base64(clean_path),
                    })

    # 获取 task 级别的 params（重推覆盖的），fallback 到 job params
    task_params = task_row.get("params")
    if task_params:
        display_params = json.loads(task_params) if isinstance(task_params, str) else task_params
    else:
        job_params = task_row.get("job_params")
        display_params = json.loads(job_params) if isinstance(job_params, str) else job_params

    return {
        "total": total,
        "page": page,
        "page_size": 1,
        "task": {
            "task_id": str(task_row["id"]),
            "job_id": str(task_row["job_id"]),
            "engine": task_engine,
            "status": task_row["status"],
            "input_path": task_row.get("image_path") or (
                json.loads(task_row["image_pair"]) if task_row.get("image_pair") else None
            ),
            "created_at": task_row.get("started_at", "").isoformat() if task_row.get("started_at") else None,
            "finished_at": task_row.get("finished_at", "").isoformat() if task_row.get("finished_at") else None,
            "params": display_params,
            "images": images,
        },
    }


def _file_to_base64(file_path: str) -> str:
    """读取文件并转为 data URI 格式的 Base64 字符串"""
    ext = os.path.splitext(file_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


# ============================================================
# 单任务重推 API
# ============================================================
class RerunRequest(BaseModel):
    """重推请求体"""
    params: dict = Field(..., description="覆盖的推理参数")


@app.post("/api/tasks/{task_id}/rerun")
async def rerun_task(task_id: str, req: RerunRequest):
    """
    对单个任务重新推理，使用新参数覆盖原参数
    重推任务设置 priority=1，优先被 Worker 抢占
    """
    # 查询 task 是否存在
    task = await TaskRepository.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task not found: {task_id}")

    # 确保 job 状态为 running（允许 Worker 抢占）
    job = await JobRepository.get_job(str(task["job_id"]))
    if not job:
        raise HTTPException(404, f"Job not found for task: {task_id}")

    # 重置任务状态
    success = await TaskRepository.rerun_task(task_id, req.params, RERUN_PRIORITY)
    if not success:
        raise HTTPException(500, "Failed to rerun task")

    # 确保 job 状态为 running
    if job["status"] not in ("pending", "running"):
        await JobRepository.update_status(str(job["id"]), "running")

    logger.info(f"Task {task_id} set to rerun with params: {req.params}")
    return {"task_id": task_id, "status": "pending", "priority": RERUN_PRIORITY}


# ============================================================
# WebSocket 重推结果推送
# ============================================================
@app.websocket("/ws/tasks/{task_id}")
async def ws_task_status(websocket: WebSocket, task_id: str):
    """
    WebSocket 端点：客户端连接后轮询 task 状态，
    状态变为 done/failed 时推送结果并关闭连接
    """
    await websocket.accept()

    try:
        while True:
            await asyncio.sleep(2)  # 每 2 秒轮询

            task = await TaskRepository.get_task(task_id)
            if not task:
                await websocket.send_json({
                    "event": "error",
                    "task_id": task_id,
                    "error": "Task not found",
                })
                break

            if task["status"] == "done":
                # 成功：推送结果图片
                images = []
                job = await JobRepository.get_job(str(task["job_id"]))
                engine = job["model_type"] if job else "unknown"

                if task.get("result_path") and os.path.exists(task["result_path"]):
                    images.append({
                        "label": "热力图",
                        "base64": _file_to_base64(task["result_path"]),
                    })

                if engine in ("dreamsim", "dinov3") and task.get("metadata"):
                    meta = json.loads(task["metadata"]) if isinstance(task["metadata"], str) else task["metadata"]
                    clean_path = meta.get("clean_result")
                    if clean_path and os.path.exists(clean_path):
                        images.append({
                            "label": "框线图",
                            "base64": _file_to_base64(clean_path),
                        })

                await websocket.send_json({
                    "event": "rerun_complete",
                    "task_id": task_id,
                    "status": "done",
                    "images": images,
                })
                break

            elif task["status"] == "failed":
                await websocket.send_json({
                    "event": "rerun_complete",
                    "task_id": task_id,
                    "status": "failed",
                    "error": task.get("error_msg", "Unknown error"),
                })
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


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
