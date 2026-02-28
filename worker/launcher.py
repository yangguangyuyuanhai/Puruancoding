"""
容器入口：N 进程启动器
本模块是 Worker 容器的入口点，负责：
  1. 启动时拉起 MIN_WORKERS_PER_GPU 个 BaseWorker 子进程
  2. 轮询伸缩指令文件，根据 target_workers 动态 fork/terminate 子进程
  3. 缩容时等当前任务完成后再 terminate
  4. 监控子进程健康状态，崩溃时自动重启
属于 worker 层，使用 multiprocessing 管理子进程的生命周期
"""
from __future__ import annotations  # 支持类型注解的前向引用

import json            # JSON 解析，用于读取伸缩指令文件
import logging         # 日志记录
import multiprocessing  # 多进程管理，每个 Worker 运行在独立子进程中
import os              # 文件系统操作，读取伸缩指令文件
import signal          # 信号处理，捕获 SIGTERM/SIGINT 实现优雅退出
import sys             # 系统退出
import time            # 主循环的休眠间隔
from typing import Optional  # 可选类型注解

# 导入共享配置：Worker 数量限制和伸缩指令目录
from shared.config import (
    MIN_WORKERS_PER_GPU,   # 每张 GPU 最少 Worker 进程数
    MAX_WORKERS_PER_GPU,   # 每张 GPU 最多 Worker 进程数
    SCALING_DIR,           # 伸缩指令文件目录
)

# 配置日志格式（Launcher 是容器入口，这里初始化全局日志配置）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("launcher")


# ============================================================
# 子进程入口点
# ============================================================
def _worker_entry(worker_id: str, gpu_id: int, model_type: str, process_index: int):
    """
    子进程入口点，每个 Worker 进程从这个函数开始执行
    延迟导入 BaseWorker 以避免在主进程（Launcher）中加载 CUDA 和模型
    这样可以确保 GPU 资源只在子进程中分配，避免主进程不必要的显存占用
    """
    # 延迟导入以避免在主进程中加载 CUDA
    from worker.base_worker import BaseWorker
    # 创建 Worker 实例
    worker = BaseWorker(
        worker_id=worker_id,
        gpu_id=gpu_id,
        model_type=model_type,
        process_index=process_index,
    )
    # 启动 Worker 主循环（阻塞直到退出）
    worker.start()


# ============================================================
# ProcessManager - 子进程生命周期管理器
# ============================================================
class ProcessManager:
    """
    管理 Worker 子进程的生命周期
    职责包括：启动初始 Worker、动态扩缩容、崩溃重启、优雅关闭
    使用 dict[int, Process] 按进程序号跟踪所有子进程
    """

    def __init__(self, gpu_id: int, model_type: str):
        self.gpu_id = gpu_id        # 绑定的 GPU 设备编号
        self.model_type = model_type  # 模型类型: "sam3" 或 "dreamsim"
        self.processes: dict[int, multiprocessing.Process] = {}  # 进程序号 -> Process 对象
        self._shutdown = False  # 全局关闭标志

    def start_initial_workers(self, count: int):
        """
        启动初始 Worker 进程
        在容器启动时调用，一次性拉起 count 个子进程
        """
        for i in range(count):
            self._spawn_worker(i)

    def _spawn_worker(self, index: int):
        """
        启动一个 Worker 子进程
        使用 multiprocessing.Process 创建子进程，设置 daemon=True 确保主进程退出时子进程也退出
        Args:
            index: 进程序号，用于 worker_id 命名和心跳文件命名
        """
        # 生成 Worker 唯一标识，格式: {模型类型}_gpu{GPU编号}_w{进程序号}
        worker_id = f"{self.model_type}_gpu{self.gpu_id}_w{index}"
        # 创建子进程，target 指向 _worker_entry 入口函数
        p = multiprocessing.Process(
            target=_worker_entry,                                     # 子进程入口函数
            args=(worker_id, self.gpu_id, self.model_type, index),   # 传递给入口函数的参数
            name=worker_id,                                           # 进程名（ps 命令和日志中可见）
            daemon=True,                                              # 守护进程标志：
            # daemon=True 的作用：
            # 1. 主进程（Launcher）退出时，所有 daemon 子进程自动终止
            # 2. 防止 Launcher 被 kill 后留下僵尸 Worker 进程持续占用 GPU 显存
            # 3. 正常关闭流程中 shutdown_all() 会先发 SIGTERM 让 Worker 优雅退出
        )
        p.start()  # 启动子进程
        self.processes[index] = p  # 记录到进程字典中
        logger.info(f"Spawned worker: {worker_id} (pid={p.pid})")

    # ============================================================
    # 弹性伸缩 - 根据目标值动态调整 Worker 数量
    # ============================================================
    def scale_to(self, target: int):
        """
        调整 Worker 数量到目标值
        目标值会被限制在 [MIN_WORKERS_PER_GPU, MAX_WORKERS_PER_GPU] 范围内
        扩容: 启动新的子进程
        缩容: 终止最高 index 的进程（后启动的先终止）
        """
        # 将目标值限制在合法范围内
        target = max(MIN_WORKERS_PER_GPU, min(target, MAX_WORKERS_PER_GPU))
        current = len(self.processes)  # 当前运行的 Worker 数量

        if target > current:
            # 扩容：按序号递增启动新的 Worker 子进程
            for i in range(current, target):
                self._spawn_worker(i)
            logger.info(f"GPU {self.gpu_id}: scaled up {current} -> {target}")
        elif target < current:
            # 缩容：终止最高 index 的进程（LIFO 策略）
            # 按序号倒序取出需要移除的进程
            indices_to_remove = sorted(self.processes.keys(), reverse=True)[:current - target]
            for idx in indices_to_remove:
                p = self.processes.pop(idx)  # 从字典中移除
                logger.info(f"Terminating worker gpu{self.gpu_id}_w{idx} (pid={p.pid})")
                p.terminate()  # 发送 SIGTERM 信号，Worker 会在当前任务完成后退出
                p.join(timeout=30)  # 等待子进程退出，最多等 30 秒
                if p.is_alive():
                    # 30 秒后仍未退出，强制杀死（发送 SIGKILL）
                    logger.warning(f"Force killing worker gpu{self.gpu_id}_w{idx}")
                    p.kill()
            logger.info(f"GPU {self.gpu_id}: scaled down {current} -> {target}")

    # ============================================================
    # 崩溃检测与自动重启
    # ============================================================
    def check_and_restart_dead(self):
        """
        检查并重启崩溃的子进程
        遍历所有子进程，发现已退出的进程后自动重新启动
        这保证了 Worker 即使因为 OOM、段错误等原因崩溃，也能自动恢复
        """
        for idx, p in list(self.processes.items()):  # list() 避免遍历时修改字典
            if not p.is_alive():
                # 子进程已退出，记录退出码并重启
                logger.warning(f"Worker gpu{self.gpu_id}_w{idx} died (exit={p.exitcode}), restarting")
                self.processes.pop(idx)    # 移除已死的进程
                self._spawn_worker(idx)    # 使用相同的序号重新启动

    # ============================================================
    # 伸缩指令读取 - 从文件系统读取 AutoScaler 写入的指令
    # ============================================================
    def read_scaling_instruction(self) -> Optional[int]:
        """
        读取伸缩指令文件
        AutoScaler 在 Gateway 端根据队列深度决定伸缩，将指令写入 JSON 文件
        Launcher 定期读取此文件，执行伸缩操作后删除文件
        指令文件格式: {"target_workers": int}
        文件路径: {SCALING_DIR}/gpu_{gpu_id}.json
        """
        path = os.path.join(SCALING_DIR, f"gpu_{self.gpu_id}.json")
        if not os.path.exists(path):
            return None  # 没有伸缩指令，保持当前状态
        try:
            with open(path, "r") as f:
                data = json.load(f)  # 解析 JSON 指令文件
            # 删除已读取的指令文件（一次性消费，防止重复执行）
            os.remove(path)
            target = data.get("target_workers")
            # 校验 target 值的合法性（必须是整数）
            if target is not None and isinstance(target, int):
                return target
        except (json.JSONDecodeError, OSError, KeyError):
            pass  # 文件损坏或读取失败时忽略
        return None

    # ============================================================
    # 优雅关闭 - 停止所有子进程
    # ============================================================
    def shutdown_all(self):
        """
        关闭所有子进程
        先发送 SIGTERM 通知所有 Worker 优雅退出，然后等待它们完成当前任务
        超时后强制杀死仍在运行的进程
        """
        self._shutdown = True
        # 第一轮：向所有子进程发送 SIGTERM 终止信号
        for idx, p in self.processes.items():
            logger.info(f"Stopping worker gpu{self.gpu_id}_w{idx}")
            p.terminate()

        # 第二轮：等待所有子进程退出，每个最多等 30 秒
        for idx, p in self.processes.items():
            p.join(timeout=30)
            if p.is_alive():
                p.kill()  # 超时未退出则强制杀死

        self.processes.clear()  # 清空进程字典
        logger.info(f"All workers on GPU {self.gpu_id} stopped")

    @property
    def worker_count(self) -> int:
        """返回当前运行的 Worker 进程数量"""
        return len(self.processes)


# ============================================================
# Launcher 入口 - 命令行解析与主循环
# ============================================================
def main():
    """
    Launcher 入口函数
    解析命令行参数 -> 创建 ProcessManager -> 启动初始 Worker -> 进入监控循环
    监控循环职责: 崩溃重启 + 读取伸缩指令
    """
    import argparse
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Worker Launcher")
    parser.add_argument("--gpu-id", type=int, required=True, help="GPU device ID")
    parser.add_argument("--model-type", type=str, required=True, help="sam3 or dreamsim")
    parser.add_argument("--initial-workers", type=int, default=None, help="Initial worker count")
    args = parser.parse_args()

    gpu_id = args.gpu_id            # 绑定的 GPU 编号
    model_type = args.model_type    # 模型类型
    # 初始 Worker 数量：命令行指定的值或配置文件中的最小值
    initial = args.initial_workers or MIN_WORKERS_PER_GPU

    logger.info(f"Launcher starting: GPU={gpu_id}, model={model_type}, initial_workers={initial}")

    # 创建进程管理器实例
    mgr = ProcessManager(gpu_id, model_type)

    # 注册信号处理器，捕获 SIGTERM 和 SIGINT 实现优雅退出
    # - SIGTERM: Docker stop / docker compose down 默认发送此信号
    # - SIGINT:  用户在终端按 Ctrl+C 时发送
    # 收到信号后会先停止所有 Worker（等待当前推理完成），再退出主进程
    def signal_handler(signum, frame):
        """信号处理函数：收到终止信号后优雅关闭所有 Worker 并退出"""
        logger.info(f"Received signal {signum}, shutting down")
        mgr.shutdown_all()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)  # Docker stop 默认发送 SIGTERM
    signal.signal(signal.SIGINT, signal_handler)   # 终端 Ctrl+C 发送 SIGINT

    # 启动初始 Worker 子进程
    mgr.start_initial_workers(initial)

    # 主监控循环：每 3 秒执行一次检查
    while True:
        try:
            # 检查并重启崩溃的子进程（确保 Worker 高可用）
            mgr.check_and_restart_dead()

            # 读取伸缩指令文件（AutoScaler 写入的动态伸缩命令）
            target = mgr.read_scaling_instruction()
            if target is not None:
                mgr.scale_to(target)  # 执行伸缩操作

            time.sleep(3)  # 3 秒轮询间隔
        except KeyboardInterrupt:
            break  # Ctrl+C 退出主循环

    # 主循环退出后，关闭所有子进程
    mgr.shutdown_all()


if __name__ == "__main__":
    main()
