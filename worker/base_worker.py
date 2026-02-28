"""
Worker 主循环 - 取任务 -> 加载图片 -> 推理 -> 写结果 -> 心跳
本模块是 Worker 端的核心调度器，负责：
  1. 从 PostgreSQL 数据库中抢占 pending 状态的任务
  2. 调用 EngineInterface 抽象接口执行推理（不知道具体推理细节）
  3. 将推理结果写入文件系统和数据库
  4. 定期写入心跳文件表示存活，供 Janitor 巡检
属于 worker 层，依赖 shared/ 中的配置和协议定义
"""
from __future__ import annotations  # 支持类型注解的前向引用

import json       # JSON 序列化/反序列化，用于解析任务参数
import logging    # 日志记录
import os         # 文件系统操作（创建目录、写心跳文件等）
import signal     # 信号处理：捕获 SIGTERM 实现优雅退出并释放 GPU 显存
import time       # 时间戳和计时
import traceback  # 异常堆栈格式化，用于记录失败任务的详细错误
from typing import Optional  # 可选类型注解

import psycopg2          # PostgreSQL 同步驱动（Worker 使用同步模式）
import psycopg2.extras   # psycopg2 扩展模块，提供 RealDictCursor 游标

# 导入共享配置：数据库连接参数、共享目录路径、轮询间隔等
from shared.config import (
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS,  # PostgreSQL 连接参数
    SHARED_ROOT, HEARTBEAT_DIR, JOBS_DIR,         # 共享卷目录路径
    WORKER_POLL_INTERVAL, HEARTBEAT_INTERVAL,     # Worker 轮询和心跳间隔
)
# 导入数据协议：引擎接口、任务描述符、结果结构体、状态枚举、引擎工厂
from shared.protocol import (
    EngineInterface, TaskDescriptor, TaskResult,
    TaskStatus, JobStatus, get_engine,
)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# ============================================================
# BaseWorker - Worker 主循环类
# ============================================================
class BaseWorker:
    """
    Worker 主循环
    每个 BaseWorker 实例运行在一个独立的子进程中，绑定到一张 GPU 上
    主循环流程: 心跳 -> 抢占任务 -> 推理 -> 写结果 -> 回到心跳
    """

    def __init__(
        self,
        worker_id: str,       # Worker 唯一标识，格式: {model_type}_gpu{gpu_id}_w{index}
        gpu_id: int,          # 绑定的 GPU 设备编号，-1 表示 CPU
        model_type: str,      # 模型类型: "sam3" 或 "dreamsim"
        process_index: int = 0,  # 同一 GPU 上的进程序号（用于心跳文件命名）
    ):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.model_type = model_type
        self.process_index = process_index
        self.engine: Optional[EngineInterface] = None  # 推理引擎实例，启动后通过工厂方法创建
        self.conn: Optional[psycopg2.extensions.connection] = None  # PostgreSQL 数据库连接
        self._running = False       # 主循环运行标志，stop() 设为 False 后循环退出
        self._last_heartbeat = 0    # 上次心跳写入的 unix 时间戳

    # ============================================================
    # 主循环启动与停止
    # ============================================================
    def start(self):
        """
        启动 Worker 主循环
        流程: 注册信号 -> 连接数据库 -> 加载推理引擎 -> 进入无限循环 (心跳+抢任务+推理)
        """
        self._running = True
        logger.info(f"Worker {self.worker_id} starting on GPU {self.gpu_id}")

        # 注册 SIGTERM 信号处理器（关键：防止僵尸 GPU 显存）
        # Python 默认收到 SIGTERM 时直接终止进程，不执行 finally 块
        # 这会导致 engine.unload() 不被调用，CUDA Context 泄漏
        # 注册自定义 handler 后，SIGTERM 仅设置 _running=False，
        # 让主循环在当前任务完成后正常退出，finally 块释放显存
        def _sigterm_handler(signum, frame):
            logger.info(f"Worker {self.worker_id} received SIGTERM, finishing current task...")
            self._running = False

        signal.signal(signal.SIGTERM, _sigterm_handler)

        try:
            # 第一步：建立 PostgreSQL 数据库连接
            self._connect_db()

            # 第二步：通过引擎注册表获取引擎类并实例化
            engine_cls = get_engine(self.model_type)  # 按 model_type 从注册表中查找引擎类
            self.engine = engine_cls()                 # 创建引擎实例
            # 确定推理设备：gpu_id >= 0 使用对应 GPU，否则使用 CPU
            device = f"cuda:{self.gpu_id}" if self.gpu_id >= 0 else "cpu"
            self.engine.load(device=device)  # 加载模型权重到指定设备
            logger.info(f"Worker {self.worker_id}: engine {self.model_type} loaded on {device}")

            # 第三步：进入主循环
            while self._running:
                self._heartbeat()            # 定期写入心跳文件
                task = self._claim_task()    # 从数据库抢占一个 pending 任务
                if task:
                    self._process_task(task)  # 有任务则执行推理
                else:
                    # 没有可用任务，休眠后再轮询（避免空转浪费 CPU）
                    time.sleep(WORKER_POLL_INTERVAL)

        except KeyboardInterrupt:
            # 用户按 Ctrl+C 中断，正常退出
            logger.info(f"Worker {self.worker_id} interrupted")
        except Exception as e:
            # 不可恢复的致命错误，记录日志后退出
            logger.error(f"Worker {self.worker_id} fatal error: {e}", exc_info=True)
        finally:
            # 无论正常退出还是异常退出，都执行资源清理
            self._cleanup()

    def stop(self):
        """优雅停止：设置标志位，主循环在下一次迭代时退出"""
        self._running = False

    # ============================================================
    # 数据库连接管理
    # ============================================================
    def _connect_db(self):
        """
        建立 PostgreSQL 数据库连接
        使用同步 psycopg2 驱动（Worker 运行在独立进程中，无需异步）
        """
        self.conn = psycopg2.connect(
            host=PG_HOST,         # 数据库主机地址
            port=PG_PORT,         # 数据库端口
            dbname=PG_DB,         # 数据库名称
            user=PG_USER,         # 用户名
            password=PG_PASS,     # 密码
        )
        self.conn.autocommit = True  # 开启自动提交，每条 SQL 立即生效，无需手动 commit
        logger.info(f"Worker {self.worker_id} connected to PostgreSQL")

    # ============================================================
    # 任务抢占 - 使用 PostgreSQL 行锁实现无冲突的分布式任务分配
    # ============================================================
    def _claim_task(self) -> Optional[dict]:
        """
        从数据库抢占一个 pending 任务
        使用 SELECT ... FOR UPDATE SKIP LOCKED 实现无锁竞争的任务抢占：
        - FOR UPDATE: 对选中的行加排他锁
        - SKIP LOCKED: 如果行已被其他 Worker 锁定，跳过而非等待
        这样多个 Worker 可以并行抢占任务而不会互相阻塞
        """
        if not self.conn:
            return None

        try:
            # 使用 RealDictCursor 将查询结果转为字典格式
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # 原子操作：在子查询中选择一行 pending 任务并加锁，外层 UPDATE 将其改为 running
                # 使用 CTE (Common Table Expression) 风格的子查询实现「抢占式任务分配」:
                # 1. 内层 SELECT 查找一条可用任务，加 FOR UPDATE 排他锁防止并发冲突
                # 2. SKIP LOCKED 跳过已被其他 Worker 锁定的行（无阻塞并发）
                # 3. 外层 UPDATE 将选中的任务状态改为 running，绑定 worker_id
                # 4. RETURNING * 返回完整行数据，省去额外查询
                cur.execute(
                    """
                    UPDATE tasks SET
                        status = 'running',
                        worker_id = %s,
                        started_at = now()
                    WHERE id = (
                        SELECT t.id FROM tasks t
                        JOIN jobs j ON t.job_id = j.id
                        WHERE t.status = 'pending'
                          AND j.status IN ('pending', 'running')
                          AND j.model_type = %s
                        ORDER BY t.id
                        LIMIT 1
                        FOR UPDATE OF t SKIP LOCKED
                    )
                    RETURNING *
                    """,
                    (self.worker_id, self.model_type),
                )
                # 子查询筛选条件详解:
                # - JOIN jobs j ON t.job_id = j.id: 关联 jobs 表，获取 Job 级别的过滤条件
                #   （单独查 tasks 表无法判断 Job 是否已取消或完成）
                # - t.status = 'pending': 只抢占等待中的任务
                # - j.status IN ('pending', 'running'): 只处理活跃 Job 的任务
                #   （cancelled/completed/failed 的 Job 下的 pending 任务不应再处理）
                # - j.model_type = %s: 只处理匹配当前引擎类型的任务
                #   （SAM3 Worker 不应抢占 DreamSim 任务，反之亦然）
                # - ORDER BY t.id: 按创建顺序 (UUID v4 按时间排序) 抢占，近似 FIFO
                # - LIMIT 1: 每次只抢占一个任务（避免一个 Worker 批量锁定太多任务）
                # - FOR UPDATE OF t: 对 tasks 表的行加排他锁（不锁 jobs 表）
                # - SKIP LOCKED: 跳过已被其他 Worker 锁定的行（并发安全，无阻塞）
                row = cur.fetchone()
                return dict(row) if row else None  # 有任务返回字典，无任务返回 None
        except Exception as e:
            logger.error(f"Worker {self.worker_id} claim error: {e}")
            self._reconnect_db()  # 数据库连接可能已断开，尝试重连
            return None

    # ============================================================
    # 任务处理 - 构造描述符 -> 调用引擎推理 -> 更新状态
    # ============================================================
    def _process_task(self, task_row: dict):
        """
        处理单个任务
        流程: 解析任务行 -> 构造 TaskDescriptor -> 调用引擎 infer -> 标记成功/失败
        """
        task_id = str(task_row["id"])       # 子任务 UUID
        job_id = str(task_row["job_id"])    # 所属 Job UUID
        logger.info(f"Worker {self.worker_id} processing task {task_id[:8]}")

        # 创建结果输出目录（在 Job 目录下的 results/ 子目录）
        results_dir = os.path.join(JOBS_DIR, job_id, "results")
        os.makedirs(results_dir, exist_ok=True)  # exist_ok=True 防止目录已存在时报错

        start_time = time.time()  # 记录推理开始时间，用于计算耗时

        try:
            # 解析图片对信息（DreamSim 模式使用，SAM3 模式为 None）
            image_pair = None
            if task_row.get("image_pair"):
                ip = task_row["image_pair"]
                # image_pair 可能是 JSON 字符串或已解析的字典
                image_pair = json.loads(ip) if isinstance(ip, str) else ip

            # 从 jobs 表获取该 Job 的推理参数（text, conf, thresh 等）
            params = self._get_job_params(job_id)

            # 构造统一的任务描述符 TaskDescriptor
            td = TaskDescriptor(
                task_id=task_id,
                job_id=job_id,
                bucket_id=task_row.get("bucket_id", 0),  # 分桶编号
                model_type=self.model_type,
                params=params,                              # 推理参数
                image_path=task_row.get("image_path"),      # SAM3: 单张图片路径
                image_pair=image_pair,                      # DreamSim: 图片对路径
            )

            # 调用推理引擎的 infer 方法执行推理，返回结果字典
            result = self.engine.infer(td, results_dir)

            duration = time.time() - start_time  # 计算推理耗时

            # 推理成功：更新数据库中 task 状态为 done，Job 完成计数 +1
            self._complete_task(task_id, job_id, result.get("result_path", ""))
            logger.info(
                f"Worker {self.worker_id} completed task {task_id[:8]} "
                f"in {duration:.1f}s"
            )

        except Exception as e:
            # 推理失败：记录完整的错误堆栈，更新数据库状态为 failed
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self._fail_task(task_id, job_id, error_msg)
            logger.error(
                f"Worker {self.worker_id} failed task {task_id[:8]} "
                f"after {duration:.1f}s: {e}"
            )

    def _get_job_params(self, job_id: str) -> dict:
        """
        从数据库获取 Job 的推理参数
        params 字段存储为 JSON 格式，需要反序列化为字典
        """
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # 查询 jobs 表中指定 Job 的 params 字段
                cur.execute("SELECT params FROM jobs WHERE id = %s", (job_id,))
                row = cur.fetchone()
                if row and row["params"]:
                    p = row["params"]
                    # params 可能是 JSON 字符串或已解析的字典
                    return json.loads(p) if isinstance(p, str) else p
        except Exception as e:
            logger.error(f"Failed to get job params: {e}")
        return {}  # 获取失败时返回空字典，推理引擎使用默认参数

    # ============================================================
    # 任务状态更新 - 标记成功或失败
    # ============================================================
    def _complete_task(self, task_id: str, job_id: str, result_path: str):
        """
        标记任务成功
        更新 tasks 表状态为 done，同时更新 jobs 表的 completed_tasks 计数
        """
        try:
            with self.conn.cursor() as cur:
                # 更新 tasks 表：状态改为 done，写入结果文件路径和完成时间
                cur.execute(
                    "UPDATE tasks SET status='done', result_path=%s, finished_at=now() WHERE id=%s",
                    (result_path, task_id),
                )
                # 更新 jobs 表：已完成任务计数 +1，刷新更新时间
                cur.execute(
                    "UPDATE jobs SET completed_tasks = completed_tasks + 1, updated_at=now() WHERE id=%s",
                    (job_id,),
                )
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            self._reconnect_db()  # 写入失败可能是连接断开，尝试重连

    def _fail_task(self, task_id: str, job_id: str, error_msg: str):
        """
        标记任务失败
        更新 tasks 表状态为 failed，记录错误信息，同时更新 jobs 表的 failed_tasks 计数
        """
        try:
            with self.conn.cursor() as cur:
                # 更新 tasks 表：状态改为 failed，写入错误信息
                # 截断到 2000 字符：防止超长堆栈信息撑爆数据库字段
                # （PostgreSQL TEXT 类型虽无长度限制，但过长的 error_msg
                #   会影响查询性能和 API 响应体积，2000 字符足以定位问题）
                cur.execute(
                    "UPDATE tasks SET status='failed', error_msg=%s, finished_at=now() WHERE id=%s",
                    (error_msg[:2000], task_id),
                )
                # 更新 jobs 表：失败任务计数 +1，刷新更新时间
                cur.execute(
                    "UPDATE jobs SET failed_tasks = failed_tasks + 1, updated_at=now() WHERE id=%s",
                    (job_id,),
                )
        except Exception as e:
            logger.error(f"Failed to mark task {task_id} as failed: {e}")
            self._reconnect_db()

    # ============================================================
    # 心跳机制 - 定期写入时间戳文件表示 Worker 存活
    # ============================================================
    def _heartbeat(self):
        """
        定期写入心跳文件
        心跳文件格式: {HEARTBEAT_DIR}/gpu{gpu_id}_w{process_index}.ts
        文件内容为 unix 时间戳，Janitor 通过读取此文件判断 Worker 是否存活
        使用 先写临时文件再 rename 的方式保证原子性（避免读到半写的文件）
        """
        now = time.time()
        # 距离上次心跳未超过心跳间隔则跳过（避免频繁写磁盘）
        if now - self._last_heartbeat < HEARTBEAT_INTERVAL:
            return

        # 确保心跳目录存在
        os.makedirs(HEARTBEAT_DIR, exist_ok=True)
        # 心跳文件路径
        hb_file = os.path.join(HEARTBEAT_DIR, f"gpu{self.gpu_id}_w{self.process_index}.ts")
        # 临时文件路径（先写临时文件再原子 rename）
        tmp_file = hb_file + ".tmp"
        try:
            # 将当前时间戳写入临时文件
            with open(tmp_file, "w") as f:
                f.write(str(now))
            # 原子重命名：确保读取方要么看到旧值要么看到新值，不会看到半写状态
            os.rename(tmp_file, hb_file)
            self._last_heartbeat = now  # 更新上次心跳时间
        except OSError as e:
            logger.warning(f"Heartbeat write failed: {e}")

    # ============================================================
    # 数据库重连
    # ============================================================
    def _reconnect_db(self):
        """
        重连数据库
        当数据库操作失败时调用，先关闭旧连接再建立新连接
        """
        try:
            if self.conn:
                self.conn.close()  # 关闭旧连接（忽略关闭时的异常）
        except Exception:
            pass
        try:
            self._connect_db()  # 建立新连接
        except Exception as e:
            logger.error(f"DB reconnect failed: {e}")
            self.conn = None  # 重连失败则置空，后续操作会跳过数据库写入

    # ============================================================
    # 资源清理 - Worker 退出时释放所有资源
    # ============================================================
    def _cleanup(self):
        """
        清理资源，在 Worker 退出时调用（无论正常还是异常退出）
        依次释放: 推理引擎 -> CUDA 显存缓存 -> 数据库连接 -> 心跳文件
        """
        # 释放推理引擎（卸载模型权重、释放 GPU 显存）
        if self.engine:
            try:
                self.engine.unload()
            except Exception as e:
                logger.error(f"Engine unload error: {e}")

        # 强制释放 CUDA 缓存（兜底：即使 unload 未完全释放，也清理残留显存）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 重置已累积的峰值显存统计，帮助后续 Worker 获得干净的显存状态
                torch.cuda.reset_peak_memory_stats()
                logger.info(f"Worker {self.worker_id}: CUDA cache cleared")
        except Exception as e:
            logger.warning(f"CUDA cache cleanup failed: {e}")

        # 关闭数据库连接
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass

        # 清除心跳文件（告诉 Janitor 此 Worker 已下线）
        hb_file = os.path.join(HEARTBEAT_DIR, f"gpu{self.gpu_id}_w{self.process_index}.ts")
        try:
            os.remove(hb_file)
        except OSError:
            pass  # 文件不存在时忽略

        logger.info(f"Worker {self.worker_id} cleaned up")
