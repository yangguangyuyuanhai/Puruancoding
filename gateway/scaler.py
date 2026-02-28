"""
Auto-Scaler - 监控队列深度，动态调整 Worker 数量
Gateway 侧的弹性伸缩控制器，每 5 秒检查一次：
  - 如果 pending 任务多且持续：扩容（增加 Worker 进程）
  - 如果队列为空且持续：缩容（减少 Worker 进程）
通过向共享卷写入 JSON 指令文件实现，Launcher 读取后执行 fork/terminate
"""
from __future__ import annotations

import asyncio   # 异步循环
import json      # JSON 序列化
import logging   # 日志
import os        # 文件系统操作
import time      # 时间戳
from typing import Optional

# 导入伸缩相关配置
from shared.config import (
    SCALING_DIR, HEARTBEAT_DIR,
    MIN_WORKERS_PER_GPU, MAX_WORKERS_PER_GPU,
    SCALE_UP_THRESHOLD_MULTIPLIER, SCALE_UP_SUSTAIN_SEC,
    SCALE_DOWN_SUSTAIN_SEC, SCALE_COOLDOWN_SEC,
)
# 导入数据库操作
from gateway.db import JobRepository, TaskRepository

logger = logging.getLogger(__name__)


class AutoScaler:
    """Gateway 侧弹性伸缩控制器"""

    def __init__(self, gpu_ids: Optional[list[int]] = None):
        """
        Args:
            gpu_ids: 监控的 GPU ID 列表，默认 0-7（8 卡）
        """
        self.gpu_ids = gpu_ids or list(range(8))  # 默认监控 8 张 GPU
        self._running = False                      # 运行状态标志
        self._last_scale_time: dict[int, float] = {}    # 每个 GPU 最后一次伸缩的时间
        self._scale_up_start: dict[int, float] = {}     # 开始满足扩容条件的时间
        self._scale_down_start: dict[int, float] = {}   # 开始满足缩容条件的时间

    async def start(self):
        """启动 AutoScaler 后台循环"""
        self._running = True
        os.makedirs(SCALING_DIR, exist_ok=True)  # 确保伸缩指令目录存在
        logger.info(f"AutoScaler started, monitoring GPUs: {self.gpu_ids}")
        while self._running:
            try:
                await self._tick()  # 执行一次伸缩检查
            except Exception as e:
                logger.error(f"AutoScaler tick error: {e}", exc_info=True)
            await asyncio.sleep(5)  # 每 5 秒检查一次

    def stop(self):
        """通知 AutoScaler 停止循环"""
        self._running = False
        logger.info("AutoScaler stopped")

    async def _tick(self):
        """
        每 5 秒执行一次伸缩检查
        对每个 GPU：统计 pending 任务数和活跃 Worker 数，决定扩容或缩容
        """
        now = time.time()

        for gpu_id in self.gpu_ids:
            # 获取该 GPU 上当前活跃的 Worker 数（通过心跳目录统计）
            active = self._count_active_workers(gpu_id)
            # 获取分配给该 GPU 的 pending 任务数
            pending = await self._count_pending_for_gpu(gpu_id)

            # 冷却期检查：两次伸缩操作之间必须间隔 SCALE_COOLDOWN_SEC 秒
            last_scale = self._last_scale_time.get(gpu_id, 0)
            if now - last_scale < SCALE_COOLDOWN_SEC:
                continue  # 冷却期内不做任何操作

            # === Scale-up 判断 ===
            # 条件：pending 任务 > 当前 Worker 数 × 倍数，且未到达上限
            if pending > active * SCALE_UP_THRESHOLD_MULTIPLIER and active < MAX_WORKERS_PER_GPU:
                if gpu_id not in self._scale_up_start:
                    # 首次满足条件，记录开始时间
                    self._scale_up_start[gpu_id] = now
                elif now - self._scale_up_start[gpu_id] >= SCALE_UP_SUSTAIN_SEC:
                    # 条件持续满足 SCALE_UP_SUSTAIN_SEC 秒，执行扩容
                    target = min(active + 1, MAX_WORKERS_PER_GPU)  # 每次只扩 1 个
                    self._write_scaling_instruction(gpu_id, target)
                    self._last_scale_time[gpu_id] = now       # 记录伸缩时间
                    self._scale_up_start.pop(gpu_id, None)    # 重置计时器
                    self._scale_down_start.pop(gpu_id, None)
                    logger.info(f"Scale UP gpu={gpu_id}: {active} -> {target} (pending={pending})")
            else:
                # 条件不满足，重置扩容计时器
                self._scale_up_start.pop(gpu_id, None)

            # === Scale-down 判断 ===
            # 条件：pending 为 0 且当前 Worker 数 > 最小值
            if pending == 0 and active > MIN_WORKERS_PER_GPU:
                if gpu_id not in self._scale_down_start:
                    # 首次满足条件，记录开始时间
                    self._scale_down_start[gpu_id] = now
                elif now - self._scale_down_start[gpu_id] >= SCALE_DOWN_SUSTAIN_SEC:
                    # 条件持续满足 SCALE_DOWN_SUSTAIN_SEC 秒，执行缩容
                    target = max(active - 1, MIN_WORKERS_PER_GPU)  # 每次只缩 1 个
                    self._write_scaling_instruction(gpu_id, target)
                    self._last_scale_time[gpu_id] = now
                    self._scale_down_start.pop(gpu_id, None)
                    self._scale_up_start.pop(gpu_id, None)
                    logger.info(f"Scale DOWN gpu={gpu_id}: {active} -> {target}")
            else:
                # 条件不满足，重置缩容计时器
                self._scale_down_start.pop(gpu_id, None)

    def _count_active_workers(self, gpu_id: int) -> int:
        """
        通过心跳目录计算指定 GPU 上活跃的 Worker 数
        心跳文件命名格式: gpu{id}_w{index}.ts
        """
        if not os.path.isdir(HEARTBEAT_DIR):
            return MIN_WORKERS_PER_GPU  # 心跳目录不存在，返回最小值

        now = time.time()
        count = 0
        prefix = f"gpu{gpu_id}_"  # 按 GPU ID 前缀过滤
        for fname in os.listdir(HEARTBEAT_DIR):
            if not fname.startswith(prefix):
                continue  # 不属于该 GPU 的心跳文件
            fpath = os.path.join(HEARTBEAT_DIR, fname)
            try:
                with open(fpath, "r") as f:
                    ts = float(f.read().strip())
                if now - ts < 30:  # 30 秒内有心跳视为活跃
                    count += 1
            except (ValueError, OSError):
                pass
        # 至少返回 MIN_WORKERS_PER_GPU，防止全部心跳丢失时误缩到 0
        return max(count, MIN_WORKERS_PER_GPU)

    async def _count_pending_for_gpu(self, gpu_id: int) -> int:
        """
        统计分配给指定 GPU 的 pending 任务数
        简化实现：统计所有活跃 Job 的总 pending 数，平均分配到各 GPU

        简化说明：
        - 理想情况下应按 Bucket 与 GPU 的映射关系精确统计
          （即只统计分配到该 GPU 的 Bucket 内的 pending 任务）
        - 但由于 Bucket 到 GPU 的映射由 Worker 自行决定（竞争抢占），
          Gateway 侧并不维护精确的映射关系
        - 因此采用均匀分摊的近似策略：总 pending / GPU 数量
        - 这种近似对于伸缩决策足够准确，因为伸缩的判断有持续时间门槛
          （SCALE_UP_SUSTAIN_SEC / SCALE_DOWN_SUSTAIN_SEC），瞬间偏差不会触发误操作
        """
        jobs = await JobRepository.list_jobs(limit=10)
        total_pending = 0
        for job in jobs:
            if job["status"] in ("pending", "running"):
                pending = await TaskRepository.get_pending_count(str(job["id"]))
                total_pending += pending
        # 将总 pending 数平均分到各 GPU
        return total_pending // max(len(self.gpu_ids), 1)

    def _write_scaling_instruction(self, gpu_id: int, target_workers: int):
        """
        写入伸缩指令文件到共享卷
        Launcher (launcher.py) 轮询此文件并执行 fork/terminate 操作
        使用原子 rename 确保指令文件不会被部分读取
        """
        os.makedirs(SCALING_DIR, exist_ok=True)
        instruction = {
            "target_workers": target_workers,  # 目标 Worker 数量
            "timestamp": time.time(),          # 指令生成时间
        }
        path = os.path.join(SCALING_DIR, f"gpu_{gpu_id}.json")
        tmp_path = path + ".tmp"  # 先写临时文件
        with open(tmp_path, "w") as f:
            json.dump(instruction, f)
        os.rename(tmp_path, path)  # 原子 rename，确保文件完整性
