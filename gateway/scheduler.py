"""
任务调度器 - Bucket 划分与任务分发
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Optional

from shared.config import SHARED_ROOT, JOBS_DIR
from shared.protocol import JobMeta, TaskDescriptor, ModelType

logger = logging.getLogger(__name__)


def compute_bucket_count(total_tasks: int, num_workers: int = 24) -> int:
    """计算 Bucket 数量，默认按 Worker 数量划分"""
    if total_tasks <= num_workers:
        return total_tasks
    return num_workers


def create_job_dirs(job_id: str) -> dict[str, str]:
    """创建 Job 所需的文件系统目录结构"""
    job_dir = os.path.join(JOBS_DIR, job_id)
    dirs = {
        "root": job_dir,
        "pending": os.path.join(job_dir, "pending"),
        "running": os.path.join(job_dir, "running"),
        "done": os.path.join(job_dir, "done"),
        "failed": os.path.join(job_dir, "failed"),
        "results": os.path.join(job_dir, "results"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def write_job_meta(job_id: str, meta: JobMeta):
    """将 Job 元数据写入 meta.json"""
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
    将 SAM3 图片列表分配到 Bucket，生成任务描述符
    Returns: 任务描述符列表 (dict)
    """
    dirs = create_job_dirs(job_id)
    tasks = []

    for idx, img_path in enumerate(image_paths):
        bucket_id = idx % num_buckets
        task_id = str(uuid.uuid4())
        td = TaskDescriptor(
            task_id=task_id,
            job_id=job_id,
            bucket_id=bucket_id,
            model_type=ModelType.SAM3.value,
            params=params,
            image_path=img_path,
        )
        # 写入 pending 目录
        task_file = os.path.join(dirs["pending"], f"task_{idx:06d}.json")
        with open(task_file, "w", encoding="utf-8") as f:
            f.write(td.to_json())
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
    将 DreamSim 图片对分配到 Bucket，生成任务描述符
    Returns: 任务描述符列表 (dict)
    """
    dirs = create_job_dirs(job_id)
    tasks = []

    for idx, pair in enumerate(image_pairs):
        bucket_id = idx % num_buckets
        task_id = str(uuid.uuid4())
        td = TaskDescriptor(
            task_id=task_id,
            job_id=job_id,
            bucket_id=bucket_id,
            model_type=ModelType.DREAMSIM.value,
            params=params,
            image_pair=pair,
        )
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
