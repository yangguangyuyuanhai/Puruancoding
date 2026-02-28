"""
容器入口：N 进程启动器
- 启动时拉起 MIN_WORKERS_PER_GPU 个子进程
- 轮询伸缩指令文件，根据 target_workers 动态 fork/terminate 子进程
- 缩容时等当前任务完成后再 terminate
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from typing import Optional

from shared.config import (
    MIN_WORKERS_PER_GPU, MAX_WORKERS_PER_GPU,
    SCALING_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("launcher")


def _worker_entry(worker_id: str, gpu_id: int, model_type: str, process_index: int):
    """子进程入口点"""
    # 延迟导入以避免在主进程中加载 CUDA
    from worker.base_worker import BaseWorker
    worker = BaseWorker(
        worker_id=worker_id,
        gpu_id=gpu_id,
        model_type=model_type,
        process_index=process_index,
    )
    worker.start()


class ProcessManager:
    """管理 Worker 子进程的生命周期"""

    def __init__(self, gpu_id: int, model_type: str):
        self.gpu_id = gpu_id
        self.model_type = model_type
        self.processes: dict[int, multiprocessing.Process] = {}  # index -> Process
        self._shutdown = False

    def start_initial_workers(self, count: int):
        """启动初始 Worker"""
        for i in range(count):
            self._spawn_worker(i)

    def _spawn_worker(self, index: int):
        """启动一个 Worker 子进程"""
        worker_id = f"{self.model_type}_gpu{self.gpu_id}_w{index}"
        p = multiprocessing.Process(
            target=_worker_entry,
            args=(worker_id, self.gpu_id, self.model_type, index),
            name=worker_id,
            daemon=True,
        )
        p.start()
        self.processes[index] = p
        logger.info(f"Spawned worker: {worker_id} (pid={p.pid})")

    def scale_to(self, target: int):
        """调整 Worker 数量到目标值"""
        target = max(MIN_WORKERS_PER_GPU, min(target, MAX_WORKERS_PER_GPU))
        current = len(self.processes)

        if target > current:
            # 扩容
            for i in range(current, target):
                self._spawn_worker(i)
            logger.info(f"GPU {self.gpu_id}: scaled up {current} -> {target}")
        elif target < current:
            # 缩容：终止最高 index 的进程
            indices_to_remove = sorted(self.processes.keys(), reverse=True)[:current - target]
            for idx in indices_to_remove:
                p = self.processes.pop(idx)
                logger.info(f"Terminating worker gpu{self.gpu_id}_w{idx} (pid={p.pid})")
                p.terminate()
                p.join(timeout=30)
                if p.is_alive():
                    logger.warning(f"Force killing worker gpu{self.gpu_id}_w{idx}")
                    p.kill()
            logger.info(f"GPU {self.gpu_id}: scaled down {current} -> {target}")

    def check_and_restart_dead(self):
        """检查并重启崩溃的子进程"""
        for idx, p in list(self.processes.items()):
            if not p.is_alive():
                logger.warning(f"Worker gpu{self.gpu_id}_w{idx} died (exit={p.exitcode}), restarting")
                self.processes.pop(idx)
                self._spawn_worker(idx)

    def read_scaling_instruction(self) -> Optional[int]:
        """读取伸缩指令文件"""
        path = os.path.join(SCALING_DIR, f"gpu_{self.gpu_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            # 删除已读取的指令
            os.remove(path)
            target = data.get("target_workers")
            if target is not None and isinstance(target, int):
                return target
        except (json.JSONDecodeError, OSError, KeyError):
            pass
        return None

    def shutdown_all(self):
        """关闭所有子进程"""
        self._shutdown = True
        for idx, p in self.processes.items():
            logger.info(f"Stopping worker gpu{self.gpu_id}_w{idx}")
            p.terminate()

        for idx, p in self.processes.items():
            p.join(timeout=30)
            if p.is_alive():
                p.kill()

        self.processes.clear()
        logger.info(f"All workers on GPU {self.gpu_id} stopped")

    @property
    def worker_count(self) -> int:
        return len(self.processes)


def main():
    """Launcher 入口"""
    import argparse
    parser = argparse.ArgumentParser(description="Worker Launcher")
    parser.add_argument("--gpu-id", type=int, required=True, help="GPU device ID")
    parser.add_argument("--model-type", type=str, required=True, help="sam3 or dreamsim")
    parser.add_argument("--initial-workers", type=int, default=None, help="Initial worker count")
    args = parser.parse_args()

    gpu_id = args.gpu_id
    model_type = args.model_type
    initial = args.initial_workers or MIN_WORKERS_PER_GPU

    logger.info(f"Launcher starting: GPU={gpu_id}, model={model_type}, initial_workers={initial}")

    mgr = ProcessManager(gpu_id, model_type)

    # 注册信号处理
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down")
        mgr.shutdown_all()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # 启动初始 Worker
    mgr.start_initial_workers(initial)

    # 主循环：监控子进程 + 读取伸缩指令
    while True:
        try:
            # 检查并重启死掉的进程
            mgr.check_and_restart_dead()

            # 读取伸缩指令
            target = mgr.read_scaling_instruction()
            if target is not None:
                mgr.scale_to(target)

            time.sleep(3)
        except KeyboardInterrupt:
            break

    mgr.shutdown_all()


if __name__ == "__main__":
    main()
