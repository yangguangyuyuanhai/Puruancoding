"""
Worker 主循环 - 取任务 → 加载图片 → 推理 → 写结果 → 心跳
通过 EngineInterface 抽象接口调用推理引擎，不知道具体推理细节
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Optional

import psycopg2
import psycopg2.extras

from shared.config import (
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS,
    SHARED_ROOT, HEARTBEAT_DIR, JOBS_DIR,
    WORKER_POLL_INTERVAL, HEARTBEAT_INTERVAL,
)
from shared.protocol import (
    EngineInterface, TaskDescriptor, TaskResult,
    TaskStatus, JobStatus, get_engine,
)

logger = logging.getLogger(__name__)


class BaseWorker:
    """Worker 主循环"""

    def __init__(
        self,
        worker_id: str,
        gpu_id: int,
        model_type: str,
        process_index: int = 0,
    ):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.model_type = model_type
        self.process_index = process_index
        self.engine: Optional[EngineInterface] = None
        self.conn: Optional[psycopg2.extensions.connection] = None
        self._running = False
        self._last_heartbeat = 0

    def start(self):
        """启动 Worker 主循环"""
        self._running = True
        logger.info(f"Worker {self.worker_id} starting on GPU {self.gpu_id}")

        try:
            # 连接数据库
            self._connect_db()

            # 加载推理引擎
            engine_cls = get_engine(self.model_type)
            self.engine = engine_cls()
            device = f"cuda:{self.gpu_id}" if self.gpu_id >= 0 else "cpu"
            self.engine.load(device=device)
            logger.info(f"Worker {self.worker_id}: engine {self.model_type} loaded on {device}")

            # 主循环
            while self._running:
                self._heartbeat()
                task = self._claim_task()
                if task:
                    self._process_task(task)
                else:
                    time.sleep(WORKER_POLL_INTERVAL)

        except KeyboardInterrupt:
            logger.info(f"Worker {self.worker_id} interrupted")
        except Exception as e:
            logger.error(f"Worker {self.worker_id} fatal error: {e}", exc_info=True)
        finally:
            self._cleanup()

    def stop(self):
        """优雅停止"""
        self._running = False

    def _connect_db(self):
        """建立数据库连接"""
        self.conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASS,
        )
        self.conn.autocommit = True
        logger.info(f"Worker {self.worker_id} connected to PostgreSQL")

    def _claim_task(self) -> Optional[dict]:
        """从数据库抢占一个 pending 任务"""
        if not self.conn:
            return None

        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # 先尝试抢占活跃 Job 的 pending 任务
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
                row = cur.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Worker {self.worker_id} claim error: {e}")
            self._reconnect_db()
            return None

    def _process_task(self, task_row: dict):
        """处理单个任务"""
        task_id = str(task_row["id"])
        job_id = str(task_row["job_id"])
        logger.info(f"Worker {self.worker_id} processing task {task_id[:8]}")

        results_dir = os.path.join(JOBS_DIR, job_id, "results")
        os.makedirs(results_dir, exist_ok=True)

        start_time = time.time()

        try:
            # 构造 TaskDescriptor
            image_pair = None
            if task_row.get("image_pair"):
                ip = task_row["image_pair"]
                image_pair = json.loads(ip) if isinstance(ip, str) else ip

            # 从 job 表获取 params
            params = self._get_job_params(job_id)

            td = TaskDescriptor(
                task_id=task_id,
                job_id=job_id,
                bucket_id=task_row.get("bucket_id", 0),
                model_type=self.model_type,
                params=params,
                image_path=task_row.get("image_path"),
                image_pair=image_pair,
            )

            # 调用引擎推理
            result = self.engine.infer(td, results_dir)

            duration = time.time() - start_time

            # 标记成功
            self._complete_task(task_id, job_id, result.get("result_path", ""))
            logger.info(
                f"Worker {self.worker_id} completed task {task_id[:8]} "
                f"in {duration:.1f}s"
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self._fail_task(task_id, job_id, error_msg)
            logger.error(
                f"Worker {self.worker_id} failed task {task_id[:8]} "
                f"after {duration:.1f}s: {e}"
            )

    def _get_job_params(self, job_id: str) -> dict:
        """从数据库获取 Job 的参数"""
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT params FROM jobs WHERE id = %s", (job_id,))
                row = cur.fetchone()
                if row and row["params"]:
                    p = row["params"]
                    return json.loads(p) if isinstance(p, str) else p
        except Exception as e:
            logger.error(f"Failed to get job params: {e}")
        return {}

    def _complete_task(self, task_id: str, job_id: str, result_path: str):
        """标记任务成功"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE tasks SET status='done', result_path=%s, finished_at=now() WHERE id=%s",
                    (result_path, task_id),
                )
                cur.execute(
                    "UPDATE jobs SET completed_tasks = completed_tasks + 1, updated_at=now() WHERE id=%s",
                    (job_id,),
                )
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            self._reconnect_db()

    def _fail_task(self, task_id: str, job_id: str, error_msg: str):
        """标记任务失败"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE tasks SET status='failed', error_msg=%s, finished_at=now() WHERE id=%s",
                    (error_msg[:2000], task_id),
                )
                cur.execute(
                    "UPDATE jobs SET failed_tasks = failed_tasks + 1, updated_at=now() WHERE id=%s",
                    (job_id,),
                )
        except Exception as e:
            logger.error(f"Failed to mark task {task_id} as failed: {e}")
            self._reconnect_db()

    def _heartbeat(self):
        """定期写入心跳文件"""
        now = time.time()
        if now - self._last_heartbeat < HEARTBEAT_INTERVAL:
            return

        os.makedirs(HEARTBEAT_DIR, exist_ok=True)
        hb_file = os.path.join(HEARTBEAT_DIR, f"gpu{self.gpu_id}_w{self.process_index}.ts")
        tmp_file = hb_file + ".tmp"
        try:
            with open(tmp_file, "w") as f:
                f.write(str(now))
            os.rename(tmp_file, hb_file)
            self._last_heartbeat = now
        except OSError as e:
            logger.warning(f"Heartbeat write failed: {e}")

    def _reconnect_db(self):
        """重连数据库"""
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass
        try:
            self._connect_db()
        except Exception as e:
            logger.error(f"DB reconnect failed: {e}")
            self.conn = None

    def _cleanup(self):
        """清理资源"""
        if self.engine:
            try:
                self.engine.unload()
            except Exception as e:
                logger.error(f"Engine unload error: {e}")

        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass

        # 清除心跳文件
        hb_file = os.path.join(HEARTBEAT_DIR, f"gpu{self.gpu_id}_w{self.process_index}.ts")
        try:
            os.remove(hb_file)
        except OSError:
            pass

        logger.info(f"Worker {self.worker_id} cleaned up")
