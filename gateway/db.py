"""
PostgreSQL asyncpg 封装 - 仅暴露业务操作接口，隐藏 SQL 细节
"""
from __future__ import annotations

import uuid
import json
import logging
from datetime import datetime
from typing import Optional

import asyncpg

from shared.config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def init_pool() -> asyncpg.Pool:
    """初始化连接池"""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DB,
            user=PG_USER,
            password=PG_PASS,
            min_size=2,
            max_size=10,
        )
        logger.info("PostgreSQL connection pool created")
    return _pool


async def close_pool():
    """关闭连接池"""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        await init_pool()
    return _pool


# ============================================================
# JobRepository
# ============================================================
class JobRepository:

    @staticmethod
    async def create_job(
        model_type: str,
        input_path: str,
        params: dict,
        total_tasks: int,
        job_id: Optional[str] = None,
    ) -> str:
        pool = await get_pool()
        if job_id is None:
            job_id = str(uuid.uuid4())
        await pool.execute(
            """
            INSERT INTO jobs (id, model_type, status, total_tasks, input_path, params)
            VALUES ($1, $2, 'pending', $3, $4, $5)
            """,
            uuid.UUID(job_id),
            model_type,
            total_tasks,
            input_path,
            json.dumps(params),
        )
        logger.info(f"Job created: {job_id}, model={model_type}, total={total_tasks}")
        return job_id

    @staticmethod
    async def get_job(job_id: str) -> Optional[dict]:
        pool = await get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM jobs WHERE id = $1",
            uuid.UUID(job_id),
        )
        if row is None:
            return None
        return dict(row)

    @staticmethod
    async def update_status(job_id: str, status: str):
        pool = await get_pool()
        await pool.execute(
            "UPDATE jobs SET status = $1, updated_at = now() WHERE id = $2",
            status,
            uuid.UUID(job_id),
        )

    @staticmethod
    async def update_progress(job_id: str, completed_delta: int = 0, failed_delta: int = 0):
        pool = await get_pool()
        await pool.execute(
            """
            UPDATE jobs SET
                completed_tasks = completed_tasks + $1,
                failed_tasks = failed_tasks + $2,
                updated_at = now()
            WHERE id = $3
            """,
            completed_delta,
            failed_delta,
            uuid.UUID(job_id),
        )

    @staticmethod
    async def list_jobs(limit: int = 50, offset: int = 0) -> list[dict]:
        pool = await get_pool()
        rows = await pool.fetch(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            limit,
            offset,
        )
        return [dict(r) for r in rows]

    @staticmethod
    async def delete_job(job_id: str):
        pool = await get_pool()
        await pool.execute(
            "UPDATE jobs SET status = 'cancelled', updated_at = now() WHERE id = $1",
            uuid.UUID(job_id),
        )


# ============================================================
# TaskRepository
# ============================================================
class TaskRepository:

    @staticmethod
    async def bulk_create(job_id: str, task_descriptors: list[dict]) -> None:
        pool = await get_pool()
        job_uuid = uuid.UUID(job_id)
        records = []
        for td in task_descriptors:
            records.append((
                uuid.UUID(td["task_id"]),
                job_uuid,
                td["bucket_id"],
                "pending",
                td.get("image_path"),
                json.dumps(td.get("image_pair")) if td.get("image_pair") else None,
            ))
        await pool.executemany(
            """
            INSERT INTO tasks (id, job_id, bucket_id, status, image_path, image_pair)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            records,
        )
        logger.info(f"Bulk created {len(records)} tasks for job {job_id}")

    @staticmethod
    async def claim_task(worker_id: str, job_id: str, bucket_id: int) -> Optional[dict]:
        pool = await get_pool()
        row = await pool.fetchrow(
            """
            UPDATE tasks SET
                status = 'running',
                worker_id = $1,
                started_at = now()
            WHERE id = (
                SELECT id FROM tasks
                WHERE job_id = $2 AND bucket_id = $3 AND status = 'pending'
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
            """,
            worker_id,
            uuid.UUID(job_id),
            bucket_id,
        )
        return dict(row) if row else None

    @staticmethod
    async def claim_any_task(worker_id: str, job_id: str) -> Optional[dict]:
        """不限 bucket，抢占任意 pending 任务"""
        pool = await get_pool()
        row = await pool.fetchrow(
            """
            UPDATE tasks SET
                status = 'running',
                worker_id = $1,
                started_at = now()
            WHERE id = (
                SELECT id FROM tasks
                WHERE job_id = $2 AND status = 'pending'
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
            """,
            worker_id,
            uuid.UUID(job_id),
        )
        return dict(row) if row else None

    @staticmethod
    async def complete_task(task_id: str, result_path: str):
        pool = await get_pool()
        await pool.execute(
            """
            UPDATE tasks SET
                status = 'done',
                result_path = $1,
                finished_at = now()
            WHERE id = $2
            """,
            result_path,
            uuid.UUID(task_id),
        )

    @staticmethod
    async def fail_task(task_id: str, error_msg: str):
        pool = await get_pool()
        await pool.execute(
            """
            UPDATE tasks SET
                status = 'failed',
                error_msg = $1,
                finished_at = now()
            WHERE id = $2
            """,
            error_msg,
            uuid.UUID(task_id),
        )

    @staticmethod
    async def get_pending_count(job_id: str) -> int:
        pool = await get_pool()
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM tasks WHERE job_id = $1 AND status = 'pending'",
            uuid.UUID(job_id),
        )
        return count or 0

    @staticmethod
    async def get_pending_count_by_bucket(job_id: str, bucket_id: int) -> int:
        pool = await get_pool()
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM tasks WHERE job_id = $1 AND bucket_id = $2 AND status = 'pending'",
            uuid.UUID(job_id),
            bucket_id,
        )
        return count or 0

    @staticmethod
    async def get_task_stats(job_id: str) -> dict:
        pool = await get_pool()
        rows = await pool.fetch(
            "SELECT status, COUNT(*) as cnt FROM tasks WHERE job_id = $1 GROUP BY status",
            uuid.UUID(job_id),
        )
        stats = {r["status"]: r["cnt"] for r in rows}
        return stats

    @staticmethod
    async def reclaim_stale_tasks(job_id: str, timeout_seconds: int = 300) -> int:
        """回收超时的 running 任务"""
        pool = await get_pool()
        result = await pool.execute(
            """
            UPDATE tasks SET
                status = 'pending',
                worker_id = NULL,
                started_at = NULL
            WHERE job_id = $1
              AND status = 'running'
              AND started_at < now() - make_interval(secs => $2)
            """,
            uuid.UUID(job_id),
            float(timeout_seconds),
        )
        count = int(result.split()[-1])
        if count > 0:
            logger.warning(f"Reclaimed {count} stale tasks for job {job_id}")
        return count

    @staticmethod
    async def retry_failed_tasks(job_id: str) -> int:
        """将 failed 任务重置为 pending"""
        pool = await get_pool()
        result = await pool.execute(
            """
            UPDATE tasks SET
                status = 'pending',
                worker_id = NULL,
                error_msg = NULL,
                started_at = NULL,
                finished_at = NULL
            WHERE job_id = $1 AND status = 'failed'
            """,
            uuid.UUID(job_id),
        )
        count = int(result.split()[-1])
        logger.info(f"Retried {count} failed tasks for job {job_id}")
        return count

    @staticmethod
    async def get_failed_tasks(job_id: str) -> list[dict]:
        pool = await get_pool()
        rows = await pool.fetch(
            "SELECT * FROM tasks WHERE job_id = $1 AND status = 'failed'",
            uuid.UUID(job_id),
        )
        return [dict(r) for r in rows]
