"""
Janitor 守护线程 - 心跳巡检、spillover 落盘、完成检测、任务回收
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime

from shared.config import (
    SHARED_ROOT, HEARTBEAT_DIR, JOBS_DIR, SPILLOVER_DIR,
    HEARTBEAT_TIMEOUT, JANITOR_INTERVAL, SPILLOVER_THRESHOLD_MB,
    TASK_TIMEOUT,
)
from gateway.db import JobRepository, TaskRepository

logger = logging.getLogger(__name__)


class Janitor:
    """后台守护任务，负责系统健康维护"""

    def __init__(self):
        self._running = False

    async def start(self):
        """启动 Janitor 后台循环"""
        self._running = True
        logger.info("Janitor started")
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"Janitor tick error: {e}", exc_info=True)
            await asyncio.sleep(JANITOR_INTERVAL)

    def stop(self):
        self._running = False
        logger.info("Janitor stopped")

    async def _tick(self):
        """每次巡检执行的操作"""
        await self._check_job_completion()
        await self._reclaim_stale_tasks()
        self._check_heartbeats()
        self._check_spillover()

    async def _check_job_completion(self):
        """检查正在运行的 Job 是否已全部完成"""
        jobs = await JobRepository.list_jobs(limit=100)
        for job in jobs:
            if job["status"] not in ("pending", "running"):
                continue

            stats = await TaskRepository.get_task_stats(str(job["id"]))
            total = job["total_tasks"]
            done = stats.get("done", 0)
            failed = stats.get("failed", 0)
            completed_total = done + failed

            # 同步进度到 jobs 表
            if job["completed_tasks"] != done or job["failed_tasks"] != failed:
                from gateway.db import get_pool
                pool = await get_pool()
                await pool.execute(
                    "UPDATE jobs SET completed_tasks=$1, failed_tasks=$2, updated_at=now() WHERE id=$3",
                    done, failed, job["id"],
                )

            # 状态转移
            if job["status"] == "pending" and stats.get("running", 0) > 0:
                await JobRepository.update_status(str(job["id"]), "running")
            elif completed_total >= total:
                new_status = "completed" if failed == 0 else "completed"
                await JobRepository.update_status(str(job["id"]), new_status)
                logger.info(f"Job {job['id']} completed: done={done}, failed={failed}")

    async def _reclaim_stale_tasks(self):
        """回收超时的 running 任务"""
        jobs = await JobRepository.list_jobs(limit=100)
        for job in jobs:
            if job["status"] not in ("pending", "running"):
                continue
            await TaskRepository.reclaim_stale_tasks(str(job["id"]), TASK_TIMEOUT)

    def _check_heartbeats(self):
        """检查 Worker 心跳，标记不健康的 Worker"""
        if not os.path.isdir(HEARTBEAT_DIR):
            return

        now = time.time()
        stale_workers = []
        for fname in os.listdir(HEARTBEAT_DIR):
            if not fname.endswith(".ts"):
                continue
            fpath = os.path.join(HEARTBEAT_DIR, fname)
            try:
                with open(fpath, "r") as f:
                    ts = float(f.read().strip())
                if now - ts > HEARTBEAT_TIMEOUT:
                    stale_workers.append(fname.replace(".ts", ""))
            except (ValueError, OSError):
                pass

        if stale_workers:
            logger.warning(f"Stale workers detected: {stale_workers}")

    def _check_spillover(self):
        """检查结果目录大小，超过阈值时执行 spillover 落盘"""
        if not os.path.isdir(JOBS_DIR):
            return

        for job_id in os.listdir(JOBS_DIR):
            results_dir = os.path.join(JOBS_DIR, job_id, "results")
            if not os.path.isdir(results_dir):
                continue

            # 计算 results 目录大小
            total_size_mb = _dir_size_mb(results_dir)
            if total_size_mb > SPILLOVER_THRESHOLD_MB:
                spillover_dest = os.path.join(SPILLOVER_DIR, job_id)
                os.makedirs(spillover_dest, exist_ok=True)
                # 移动最老的文件到 spillover
                files = sorted(
                    os.listdir(results_dir),
                    key=lambda f: os.path.getmtime(os.path.join(results_dir, f)),
                )
                moved = 0
                for f in files[:len(files) // 2]:
                    src = os.path.join(results_dir, f)
                    dst = os.path.join(spillover_dest, f)
                    shutil.move(src, dst)
                    moved += 1
                logger.info(
                    f"Spillover: moved {moved} files from job {job_id} "
                    f"({total_size_mb:.0f}MB > {SPILLOVER_THRESHOLD_MB}MB threshold)"
                )


def _dir_size_mb(path: str) -> float:
    """计算目录大小 (MB)"""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)
