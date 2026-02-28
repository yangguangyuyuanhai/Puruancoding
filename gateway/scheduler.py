"""
任务调度器 - Bucket 划分与任务分发
负责将索引后的图片列表按 Bucket 策略分配，并生成任务描述符写入文件系统和数据库
Bucket 机制保证同一 GPU Worker 处理的文件路径相近，提高 I/O 局部性
"""
from __future__ import annotations

import json       # JSON 序列化
import logging    # 日志
import os         # 文件系统操作
import uuid       # UUID 生成
from typing import Optional

# 导入共享配置和数据协议
from shared.config import SHARED_ROOT, JOBS_DIR
from shared.protocol import JobMeta, TaskDescriptor, ModelType

logger = logging.getLogger(__name__)


def compute_bucket_count(total_tasks: int, num_workers: int = 24) -> int:
    """
    计算 Bucket 数量
    策略：Bucket 数 = Worker 数，确保每个 Worker 至少分到一个 Bucket
    如果总任务数少于 Worker 数，则 Bucket 数 = 任务数（避免空桶）

    为什么 Bucket 数 = Worker 数？
    - 每个 Worker 优先从自己的 Bucket 中取任务（空间局部性，减少竞争）
    - 同一 Bucket 内的图片路径相近，可能在同一磁盘分区，I/O 更高效
    - 当 Worker 自己的 Bucket 消耗完后，可以跨桶抢占其他 Bucket 的任务

    Args:
        total_tasks: 子任务总数
        num_workers: Worker 总数（默认 24 = 8 卡 × 3 进程/卡）
    Returns:
        Bucket 数量
    """
    if total_tasks <= num_workers:
        return total_tasks  # 任务太少时，每个任务一个桶
    return num_workers


def create_job_dirs(job_id: str) -> dict[str, str]:
    """
    创建 Job 所需的文件系统目录结构
    Args:
        job_id: 任务 ID
    Returns:
        目录名到路径的映射字典
    """
    job_dir = os.path.join(JOBS_DIR, job_id)
    dirs = {
        "root": job_dir,                                # Job 根目录
        "pending": os.path.join(job_dir, "pending"),     # 待处理任务描述符（Worker 从此读取）
        "running": os.path.join(job_dir, "running"),     # 正在处理（预留：用于文件系统级别的任务跟踪，
                                                          #   当前使用 DB 跟踪状态，此目录作为兼容预留）
        "done": os.path.join(job_dir, "done"),           # 已完成任务描述符（预留：同上，DB 优先）
        "failed": os.path.join(job_dir, "failed"),       # 失败任务描述符（预留：同上，DB 优先）
        "results": os.path.join(job_dir, "results"),     # 推理结果文件（图片/热力图等实际输出）
    }
    # 确保所有目录存在
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def write_job_meta(job_id: str, meta: JobMeta):
    """
    将 Job 元数据写入 meta.json 文件
    供调试、审计和 Worker 读取参数使用
    """
    job_dir = os.path.join(JOBS_DIR, job_id)
    meta_path = os.path.join(job_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(meta.to_json())
    logger.info(f"Job meta written: {meta_path}")


def distribute_sam3_tasks(
    job_id: str,
    image_paths: list[str],
    params: dict,
    num_buckets: int,
) -> list[dict]:
    """
    将 SAM3 图片列表按轮转方式分配到 Bucket，生成任务描述符
    分配策略：第 i 张图 → Bucket (i % num_buckets)
    同时将任务描述符写入 pending/ 目录
    Args:
        job_id: 任务 ID
        image_paths: 图片路径列表
        params: SAM3 推理参数
        num_buckets: Bucket 数量
    Returns:
        任务描述符列表（dict 格式，用于 DB 批量写入）
    """
    # 创建 Job 目录结构
    dirs = create_job_dirs(job_id)
    tasks = []

    for idx, img_path in enumerate(image_paths):
        # 轮转分桶：按索引取模分配 Bucket
        bucket_id = idx % num_buckets
        # 为每个子任务生成唯一 ID
        task_id = str(uuid.uuid4())
        # 构造任务描述符
        td = TaskDescriptor(
            task_id=task_id,
            job_id=job_id,
            bucket_id=bucket_id,
            model_type=ModelType.SAM3.value,
            params=params,
            image_path=img_path,
        )
        # 将任务描述符写入 pending/ 目录（JSON 文件）
        task_file = os.path.join(dirs["pending"], f"task_{idx:06d}.json")
        with open(task_file, "w", encoding="utf-8") as f:
            f.write(td.to_json())
        # 收集用于 DB 批量插入的数据
        tasks.append({
            "task_id": task_id,
            "bucket_id": bucket_id,
            "image_path": img_path,
        })

    logger.info(f"Distributed {len(tasks)} SAM3 tasks into {num_buckets} buckets for job {job_id}")
    return tasks


def distribute_dreamsim_tasks(
    job_id: str,
    image_pairs: list[dict],
    params: dict,
    num_buckets: int,
) -> list[dict]:
    """
    将 DreamSim 图片对按轮转方式分配到 Bucket，生成任务描述符
    与 SAM3 类似，但每个任务包含 before/after 图片对
    Args:
        job_id: 任务 ID
        image_pairs: 图片对列表 [{"before": path, "after": path}, ...]
        params: DreamSim 推理参数
        num_buckets: Bucket 数量
    Returns:
        任务描述符列表（dict 格式，用于 DB 批量写入）
    """
    dirs = create_job_dirs(job_id)
    tasks = []

    for idx, pair in enumerate(image_pairs):
        bucket_id = idx % num_buckets
        task_id = str(uuid.uuid4())
        # 构造包含图片对的任务描述符
        td = TaskDescriptor(
            task_id=task_id,
            job_id=job_id,
            bucket_id=bucket_id,
            model_type=ModelType.DREAMSIM.value,
            params=params,
            image_pair=pair,  # {"before": ..., "after": ...}
        )
        # 写入 pending/ 目录
        task_file = os.path.join(dirs["pending"], f"task_{idx:06d}.json")
        with open(task_file, "w", encoding="utf-8") as f:
            f.write(td.to_json())
        tasks.append({
            "task_id": task_id,
            "bucket_id": bucket_id,
            "image_pair": pair,
        })

    logger.info(f"Distributed {len(tasks)} DreamSim tasks into {num_buckets} buckets for job {job_id}")
    return tasks
