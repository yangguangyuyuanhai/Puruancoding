"""
PostgreSQL asyncpg 封装 - 仅暴露业务操作接口，隐藏 SQL 细节
本模块将数据库操作抽象为 JobRepository 和 TaskRepository 两个类
上层代码（app.py、janitor.py）只调用业务方法，不直接写 SQL
如果未来需要更换数据库，只需重写这两个类的实现
"""
from __future__ import annotations

import uuid       # UUID 类型转换
import json       # JSON 序列化（params 字段）
import logging    # 日志
from datetime import datetime  # 时间类型
from typing import Optional    # 可选类型注解

import asyncpg  # PostgreSQL 异步驱动，支持连接池和事务

# 导入数据库连接配置
from shared.config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS

logger = logging.getLogger(__name__)

# 全局连接池实例（模块级单例）
_pool: Optional[asyncpg.Pool] = None


async def init_pool() -> asyncpg.Pool:
    """
    初始化 asyncpg 连接池
    Gateway 启动时调用一次，后续所有数据库操作复用连接池中的连接
    """
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=PG_HOST,       # 数据库主机（容器内为 "batch-postgres"，宿主机为 "localhost"）
            port=PG_PORT,       # 数据库端口（默认 5433，容器内部为 5432）
            database=PG_DB,     # 数据库名（batch_detection）
            user=PG_USER,       # 用户名（batch）
            password=PG_PASS,   # 密码（batch_secret）
            min_size=2,         # 连接池最小连接数（保证低延迟：至少 2 个连接随时可用）
            max_size=10,        # 连接池最大连接数（限制并发 DB 连接，防止 PG 过载；
                                # Gateway 的并发来源主要是 API 请求 + Janitor + AutoScaler，
                                # 10 个连接足以覆盖峰值场景）
        )
        logger.info("PostgreSQL connection pool created")
    return _pool


async def close_pool():
    """关闭连接池，Gateway 停机时调用"""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


async def get_pool() -> asyncpg.Pool:
    """获取连接池，如果未初始化则自动创建"""
    if _pool is None:
        await init_pool()
    return _pool


# ============================================================
# JobRepository - 批量任务 (Job) 的数据库操作
# ============================================================
class JobRepository:
    """封装 jobs 表的所有 CRUD 操作"""

    @staticmethod
    async def create_job(
        model_type: str,
        input_path: str,
        params: dict,
        total_tasks: int,
        job_id: Optional[str] = None,
        parent_job_id: Optional[str] = None,
        pipeline_phase: Optional[str] = None,
    ) -> str:
        """
        创建新的批量任务记录
        Args:
            model_type: 模型类型 (sam3/dreamsim/pipeline)
            input_path: 输入路径
            params: 推理参数字典
            total_tasks: 子任务总数
            job_id: 可选，指定 Job ID（确保与文件系统一致）
            parent_job_id: 可选，父任务 ID（流水线子任务才有值）
            pipeline_phase: 可选，流水线阶段标识 (sam3_before/sam3_after/dreamsim)
        Returns:
            job_id 字符串
        """
        pool = await get_pool()
        # 如果未指定 job_id，自动生成
        if job_id is None:
            job_id = str(uuid.uuid4())
        await pool.execute(
            """
            INSERT INTO jobs (id, model_type, status, total_tasks, input_path, params,
                              parent_job_id, pipeline_phase)
            VALUES ($1, $2, 'pending', $3, $4, $5, $6, $7)
            """,
            uuid.UUID(job_id),           # $1: UUID 类型
            model_type,                   # $2: 模型类型
            total_tasks,                  # $3: 子任务总数
            input_path,                   # $4: 输入路径
            json.dumps(params),           # $5: 参数序列化为 JSON 字符串
            uuid.UUID(parent_job_id) if parent_job_id else None,  # $6: 父任务 ID
            pipeline_phase,               # $7: 流水线阶段
        )
        logger.info(f"Job created: {job_id}, model={model_type}, total={total_tasks}, phase={pipeline_phase}")
        return job_id

    @staticmethod
    async def get_job(job_id: str) -> Optional[dict]:
        """按 ID 查询单个 Job，返回字典或 None"""
        pool = await get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM jobs WHERE id = $1",
            uuid.UUID(job_id),
        )
        if row is None:
            return None
        return dict(row)  # asyncpg.Record → dict

    @staticmethod
    async def update_status(job_id: str, status: str):
        """更新 Job 状态并刷新 updated_at 时间戳"""
        pool = await get_pool()
        await pool.execute(
            "UPDATE jobs SET status = $1, updated_at = now() WHERE id = $2",
            status,
            uuid.UUID(job_id),
        )

    @staticmethod
    async def update_progress(job_id: str, completed_delta: int = 0, failed_delta: int = 0):
        """
        增量更新 Job 的完成/失败计数
        使用 += 而非直接赋值，支持并发安全
        """
        pool = await get_pool()
        await pool.execute(
            """
            UPDATE jobs SET
                completed_tasks = completed_tasks + $1,
                failed_tasks = failed_tasks + $2,
                updated_at = now()
            WHERE id = $3
            """,
            completed_delta,      # 完成数增量
            failed_delta,         # 失败数增量
            uuid.UUID(job_id),
        )

    @staticmethod
    async def list_jobs(limit: int = 50, offset: int = 0) -> list[dict]:
        """列出所有 Job，按创建时间倒序，支持分页"""
        pool = await get_pool()
        rows = await pool.fetch(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            limit,
            offset,
        )
        return [dict(r) for r in rows]

    @staticmethod
    async def delete_job(job_id: str):
        """软删除 Job：将状态置为 cancelled（不物理删除数据）"""
        pool = await get_pool()
        await pool.execute(
            "UPDATE jobs SET status = 'cancelled', updated_at = now() WHERE id = $1",
            uuid.UUID(job_id),
        )

    @staticmethod
    async def get_children_jobs(parent_id: str) -> list[dict]:
        """查询某个 pipeline job 的所有子任务，按创建时间排序"""
        pool = await get_pool()
        rows = await pool.fetch(
            "SELECT * FROM jobs WHERE parent_job_id = $1 ORDER BY created_at",
            uuid.UUID(parent_id),
        )
        return [dict(r) for r in rows]

    @staticmethod
    async def get_pipeline_parent(child_id: str) -> Optional[dict]:
        """查询某个子任务的父任务"""
        pool = await get_pool()
        row = await pool.fetchrow(
            """
            SELECT p.* FROM jobs p
            JOIN jobs c ON c.parent_job_id = p.id
            WHERE c.id = $1
            """,
            uuid.UUID(child_id),
        )
        return dict(row) if row else None


# ============================================================
# TaskRepository - 子任务 (Task) 的数据库操作
# ============================================================
class TaskRepository:
    """封装 tasks 表的所有 CRUD 操作"""

    @staticmethod
    async def bulk_create(job_id: str, task_descriptors: list[dict]) -> None:
        """
        批量创建子任务记录
        使用 executemany 一次性插入所有任务，比逐条插入高效
        """
        pool = await get_pool()
        job_uuid = uuid.UUID(job_id)
        records = []
        for td in task_descriptors:
            records.append((
                uuid.UUID(td["task_id"]),   # 任务 ID
                job_uuid,                    # 所属 Job ID
                td["bucket_id"],             # 分桶编号
                "pending",                   # 初始状态
                td.get("image_path"),        # SAM3 图片路径（DreamSim 为 None）
                json.dumps(td.get("image_pair")) if td.get("image_pair") else None,  # DreamSim 图片对 JSON
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
        """
        抢占指定 Bucket 中的一个 pending 任务
        使用 FOR UPDATE SKIP LOCKED 实现无锁竞争：
        - FOR UPDATE: 锁定选中的行，防止其他 Worker 同时抢占
        - SKIP LOCKED: 如果行已被锁定，跳过它选下一个，避免阻塞
        """
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
        """不限 Bucket，抢占任意 pending 任务（用于 Bucket 内任务耗尽后的跨桶抢占）"""
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
        """标记任务为成功完成，记录结果文件路径"""
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
        """标记任务为失败，记录错误信息"""
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
        """获取指定 Job 的 pending 任务数量"""
        pool = await get_pool()
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM tasks WHERE job_id = $1 AND status = 'pending'",
            uuid.UUID(job_id),
        )
        return count or 0

    @staticmethod
    async def get_pending_count_by_bucket(job_id: str, bucket_id: int) -> int:
        """获取指定 Job 指定 Bucket 的 pending 任务数量"""
        pool = await get_pool()
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM tasks WHERE job_id = $1 AND bucket_id = $2 AND status = 'pending'",
            uuid.UUID(job_id),
            bucket_id,
        )
        return count or 0

    @staticmethod
    async def get_task_stats(job_id: str) -> dict:
        """
        获取指定 Job 的任务状态统计
        Returns: {"pending": N, "running": N, "done": N, "failed": N}
        """
        pool = await get_pool()
        rows = await pool.fetch(
            "SELECT status, COUNT(*) as cnt FROM tasks WHERE job_id = $1 GROUP BY status",
            uuid.UUID(job_id),
        )
        stats = {r["status"]: r["cnt"] for r in rows}
        return stats

    @staticmethod
    async def reclaim_stale_tasks(job_id: str, timeout_seconds: int = 300, max_retries: int = 3) -> int:
        """
        回收超时的 running 任务（含毒丸防护）
        如果一个任务 running 超过 timeout_seconds 秒仍未完成，
        说明 Worker 可能已崩溃。此时：
        - retry_count < max_retries: 重置为 pending 并 retry_count + 1
        - retry_count >= max_retries: 直接标记为 failed（毒丸隔离）
        """
        pool = await get_pool()

        # 第一步：回收 retry_count 未超限的超时任务 → pending
        result_reclaimed = await pool.execute(
            """
            UPDATE tasks SET
                status = 'pending',
                worker_id = NULL,
                started_at = NULL,
                retry_count = retry_count + 1
            WHERE job_id = $1
              AND status = 'running'
              AND started_at < now() - make_interval(secs => $2)
              AND retry_count < $3
            """,
            uuid.UUID(job_id),
            float(timeout_seconds),
            max_retries,
        )
        reclaimed_count = int(result_reclaimed.split()[-1])

        # 第二步：将 retry_count 已达上限的超时任务 → failed（毒丸隔离）
        result_poisoned = await pool.execute(
            """
            UPDATE tasks SET
                status = 'failed',
                error_msg = 'Poison task: exceeded max retries (' || $3 || '), likely causes OOM or crash',
                finished_at = now()
            WHERE job_id = $1
              AND status = 'running'
              AND started_at < now() - make_interval(secs => $2)
              AND retry_count >= $3
            """,
            uuid.UUID(job_id),
            float(timeout_seconds),
            max_retries,
        )
        poisoned_count = int(result_poisoned.split()[-1])

        if reclaimed_count > 0:
            logger.warning(f"Reclaimed {reclaimed_count} stale tasks for job {job_id}")
        if poisoned_count > 0:
            logger.error(
                f"Poison pill: {poisoned_count} tasks in job {job_id} exceeded "
                f"{max_retries} retries, marked as failed"
            )
        return reclaimed_count + poisoned_count

    @staticmethod
    async def retry_failed_tasks(job_id: str) -> int:
        """将 failed 任务重置为 pending，清除错误信息和重试计数，允许 Worker 重新处理"""
        pool = await get_pool()
        result = await pool.execute(
            """
            UPDATE tasks SET
                status = 'pending',
                worker_id = NULL,
                error_msg = NULL,
                started_at = NULL,
                finished_at = NULL,
                retry_count = 0
            WHERE job_id = $1 AND status = 'failed'
            """,
            uuid.UUID(job_id),
        )
        count = int(result.split()[-1])
        logger.info(f"Retried {count} failed tasks for job {job_id}")
        return count

    @staticmethod
    async def get_failed_tasks(job_id: str) -> list[dict]:
        """查询指定 Job 的所有失败任务，用于排查问题"""
        pool = await get_pool()
        rows = await pool.fetch(
            "SELECT * FROM tasks WHERE job_id = $1 AND status = 'failed'",
            uuid.UUID(job_id),
        )
        return [dict(r) for r in rows]
