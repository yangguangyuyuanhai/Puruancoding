"""
Auto-Scaler - 监控队列深度，动态调整 Worker 数量
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

from shared.config import (
    SCALING_DIR, HEARTBEAT_DIR,
    MIN_WORKERS_PER_GPU, MAX_WORKERS_PER_GPU,
    SCALE_UP_THRESHOLD_MULTIPLIER, SCALE_UP_SUSTAIN_SEC,
    SCALE_DOWN_SUSTAIN_SEC, SCALE_COOLDOWN_SEC,
)
from gateway.db import JobRepository, TaskRepository

logger = logging.getLogger(__name__)


class AutoScaler:
    """Gateway 侧弹性伸缩控制器"""

    def __init__(self, gpu_ids: Optional[list[int]] = None):
        self.gpu_ids = gpu_ids or list(range(8))
        self._running = False
        self._last_scale_time: dict[int, float] = {}
        self._scale_up_start: dict[int, float] = {}   # 记录开始满足 scale-up 条件的时间
        self._scale_down_start: dict[int, float] = {}  # 记录开始满足 scale-down 条件的时间

    async def start(self):
        self._running = True
        os.makedirs(SCALING_DIR, exist_ok=True)
        logger.info(f"AutoScaler started, monitoring GPUs: {self.gpu_ids}")
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"AutoScaler tick error: {e}", exc_info=True)
            await asyncio.sleep(5)

    def stop(self):
        self._running = False
        logger.info("AutoScaler stopped")

    async def _tick(self):
        """每 5 秒执行一次伸缩检查"""
        now = time.time()

        # 统计各 GPU 的 pending 任务和活跃 Worker
        for gpu_id in self.gpu_ids:
            active = self._count_active_workers(gpu_id)
            pending = await self._count_pending_for_gpu(gpu_id)

            # 冷却检查
            last_scale = self._last_scale_time.get(gpu_id, 0)
            if now - last_scale < SCALE_COOLDOWN_SEC:
                continue

            # Scale-up 判断
            if pending > active * SCALE_UP_THRESHOLD_MULTIPLIER and active < MAX_WORKERS_PER_GPU:
                if gpu_id not in self._scale_up_start:
                    self._scale_up_start[gpu_id] = now
                elif now - self._scale_up_start[gpu_id] >= SCALE_UP_SUSTAIN_SEC:
                    target = min(active + 1, MAX_WORKERS_PER_GPU)
                    self._write_scaling_instruction(gpu_id, target)
                    self._last_scale_time[gpu_id] = now
                    self._scale_up_start.pop(gpu_id, None)
                    self._scale_down_start.pop(gpu_id, None)
                    logger.info(f"Scale UP gpu={gpu_id}: {active} -> {target} (pending={pending})")
            else:
                self._scale_up_start.pop(gpu_id, None)

            # Scale-down 判断
            if pending == 0 and active > MIN_WORKERS_PER_GPU:
                if gpu_id not in self._scale_down_start:
                    self._scale_down_start[gpu_id] = now
                elif now - self._scale_down_start[gpu_id] >= SCALE_DOWN_SUSTAIN_SEC:
                    target = max(active - 1, MIN_WORKERS_PER_GPU)
                    self._write_scaling_instruction(gpu_id, target)
                    self._last_scale_time[gpu_id] = now
                    self._scale_down_start.pop(gpu_id, None)
                    self._scale_up_start.pop(gpu_id, None)
                    logger.info(f"Scale DOWN gpu={gpu_id}: {active} -> {target}")
            else:
                self._scale_down_start.pop(gpu_id, None)

    def _count_active_workers(self, gpu_id: int) -> int:
        """通过心跳目录计算 GPU 上活跃的 Worker 数"""
        if not os.path.isdir(HEARTBEAT_DIR):
            return MIN_WORKERS_PER_GPU

        now = time.time()
        count = 0
        prefix = f"gpu{gpu_id}_"
        for fname in os.listdir(HEARTBEAT_DIR):
            if not fname.startswith(prefix):
                continue
            fpath = os.path.join(HEARTBEAT_DIR, fname)
            try:
                with open(fpath, "r") as f:
                    ts = float(f.read().strip())
                if now - ts < 30:
                    count += 1
            except (ValueError, OSError):
                pass
        return max(count, MIN_WORKERS_PER_GPU)

    async def _count_pending_for_gpu(self, gpu_id: int) -> int:
        """统计分配给指定 GPU (bucket) 的 pending 任务数"""
        # 简化实现：统计所有活跃 job 的总 pending 数
        # 实际应按 bucket 与 GPU 的映射关系来统计
        jobs = await JobRepository.list_jobs(limit=10)
        total_pending = 0
        for job in jobs:
            if job["status"] in ("pending", "running"):
                pending = await TaskRepository.get_pending_count(str(job["id"]))
                total_pending += pending
        # 平均分配到各 GPU
        return total_pending // max(len(self.gpu_ids), 1)

    def _write_scaling_instruction(self, gpu_id: int, target_workers: int):
        """写入伸缩指令文件到共享卷"""
        os.makedirs(SCALING_DIR, exist_ok=True)
        instruction = {
            "target_workers": target_workers,
            "timestamp": time.time(),
        }
        path = os.path.join(SCALING_DIR, f"gpu_{gpu_id}.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(instruction, f)
        os.rename(tmp_path, path)
