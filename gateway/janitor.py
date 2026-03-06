"""
Janitor 守护线程 - 心跳巡检、spillover 落盘、完成检测、任务回收
Janitor 是 Gateway 内的后台异步循环，每 JANITOR_INTERVAL 秒执行一次巡检
主要职责：
  1. 同步 tasks 表的统计到 jobs 表（completed_tasks, failed_tasks）
  2. 检测 Job 是否全部完成，更新状态为 completed
  3. 回收超时的 running 任务（Worker 崩溃后的自愈机制）
  4. 检测 Worker 心跳，标记不健康的 Worker
  5. 当结果目录过大时自动 spillover（移动旧文件到溢出区）
"""
from __future__ import annotations

import asyncio   # 异步循环
import json      # JSON 解析（预留）
import logging   # 日志
import os        # 文件系统操作
import shutil    # 文件移动（spillover）
import time      # 时间戳
import uuid      # UUID 生成（流水线创建 DreamSim child）
from datetime import datetime  # 时间格式化（预留）

# 导入配置参数
from shared.config import (
    SHARED_ROOT, HEARTBEAT_DIR, JOBS_DIR, SPILLOVER_DIR,
    HEARTBEAT_TIMEOUT, JANITOR_INTERVAL, SPILLOVER_THRESHOLD_MB,
    TASK_TIMEOUT, TASK_MAX_RETRIES, IMAGE_EXTENSIONS,
    CLEANUP_RETENTION_DAYS,
)
# 导入数据库操作
from gateway.db import JobRepository, TaskRepository
# 导入调度器（流水线创建变化检测子任务时复用）
from gateway.scheduler import compute_bucket_count, distribute_dreamsim_tasks, distribute_dinov3_tasks
# 导入数据协议
from shared.protocol import ModelType, JobStatus, JobMeta

logger = logging.getLogger(__name__)


class Janitor:
    """后台守护任务，负责系统健康维护"""

    def __init__(self):
        self._running = False  # 运行状态标志
        self._tick_count = 0   # 巡检计数器（用于降频执行清理任务）

    async def start(self):
        """启动 Janitor 后台循环，每 JANITOR_INTERVAL 秒执行一次巡检"""
        self._running = True
        logger.info("Janitor started")
        while self._running:
            try:
                await self._tick()  # 执行一次巡检
            except Exception as e:
                # 巡检出错不中断循环，记录错误后继续
                logger.error(f"Janitor tick error: {e}", exc_info=True)
            await asyncio.sleep(JANITOR_INTERVAL)  # 等待下次巡检

    def stop(self):
        """通知 Janitor 停止循环"""
        self._running = False
        logger.info("Janitor stopped")

    async def _tick(self):
        """每次巡检执行的操作"""
        self._tick_count += 1
        await self._check_job_completion()     # 1. 同步进度 + 检测完成（普通 job）
        await self._check_pipeline_progress()  # 2. 流水线阶段推进（pipeline job）
        await self._reclaim_stale_tasks()      # 3. 回收超时任务
        self._check_heartbeats()               # 4. 检查 Worker 心跳
        self._check_spillover()                # 5. 检查结果目录大小
        # 6. 历史清理（每 360 轮 ≈ 1 小时执行一次）
        if self._tick_count % 360 == 0:
            await self._cleanup_old_jobs()

    async def _check_job_completion(self):
        """
        检查正在运行的 Job 是否已全部完成
        遍历所有活跃 Job，从 tasks 表统计实际 done/failed 数，同步到 jobs 表
        如果所有子任务都已完成（done + failed = total），则标记 Job 为 completed
        """
        # 查询所有 Job
        jobs = await JobRepository.list_jobs(limit=100)
        for job in jobs:
            # 只处理 pending 和 running 状态的 Job
            if job["status"] not in ("pending", "running"):
                continue
            # 跳过 pipeline parent job（由 _check_pipeline_progress 单独处理）
            if job["model_type"] == ModelType.PIPELINE.value:
                continue

            # 从 tasks 表获取实际的状态统计
            stats = await TaskRepository.get_task_stats(str(job["id"]))
            total = job["total_tasks"]
            done = stats.get("done", 0)        # 成功完成的数量
            failed = stats.get("failed", 0)    # 失败的数量
            completed_total = done + failed     # 已处理总数

            # 如果 jobs 表中的计数与实际不符，同步更新
            if job["completed_tasks"] != done or job["failed_tasks"] != failed:
                from gateway.db import get_pool
                pool = await get_pool()
                await pool.execute(
                    "UPDATE jobs SET completed_tasks=$1, failed_tasks=$2, updated_at=now() WHERE id=$3",
                    done, failed, job["id"],
                )

            # 状态转移：pending → running（有任务开始运行时）
            if job["status"] == "pending" and stats.get("running", 0) > 0:
                await JobRepository.update_status(str(job["id"]), "running")
            # 状态转移：running → completed
            # 注意：即使有部分 failed 任务，Job 整体仍标记为 completed
            # （用户可通过 /api/jobs/{id}/failed 查看失败列表，POST /api/jobs/{id}/retry 重试）
            elif completed_total >= total:
                await JobRepository.update_status(str(job["id"]), "completed")
                logger.info(f"Job {job['id']} completed: done={done}, failed={failed}")

    async def _check_pipeline_progress(self):
        """
        检查所有 pipeline 类型的 parent job，推进流水线阶段
        状态机：SAM3 children 都完成 → 创建 DreamSim child → DreamSim 完成 → pipeline 完成
        """
        jobs = await JobRepository.list_jobs(limit=100)
        for job in jobs:
            if job["model_type"] != ModelType.PIPELINE.value:
                continue
            if job["status"] not in ("pending", "running"):
                continue

            children = await JobRepository.get_children_jobs(str(job["id"]))
            sam3_children = [c for c in children if c["pipeline_phase"] in ("sam3_before", "sam3_after")]
            ds_children = [c for c in children if c["pipeline_phase"] in ("dreamsim", "dinov3")]

            # 检查是否有 SAM3 child 失败
            sam3_failed = any(c["status"] == "failed" for c in sam3_children)
            if sam3_failed:
                await JobRepository.update_status(str(job["id"]), "failed")
                logger.warning(f"Pipeline {job['id']} failed: SAM3 child failed")
                continue

            # 检查 SAM3 阶段是否全部完成
            sam3_all_done = len(sam3_children) == 2 and all(
                c["status"] == "completed" for c in sam3_children
            )

            if sam3_all_done and not ds_children:
                # SAM3 全部完成且变化检测还没创建 → 根据 pipeline_type 创建对应子任务
                try:
                    await self._create_comparison_phase(job, sam3_children)
                    logger.info(f"Pipeline {job['id']}: comparison phase created")
                except Exception as e:
                    logger.error(f"Pipeline {job['id']}: failed to create comparison phase: {e}", exc_info=True)
                    await JobRepository.update_status(str(job["id"]), "failed")

            elif ds_children:
                ds_child = ds_children[0]
                if ds_child["status"] == "completed":
                    # 变化检测也完成了 → 标记 pipeline job 完成
                    await JobRepository.update_status(str(job["id"]), "completed")
                    logger.info(f"Pipeline {job['id']} completed")
                elif ds_child["status"] == "failed":
                    await JobRepository.update_status(str(job["id"]), "failed")
                    logger.warning(f"Pipeline {job['id']} failed: comparison child failed")

    async def _create_comparison_phase(self, parent_job: dict, sam3_children: list[dict]):
        """
        创建变化检测子任务阶段（DINOv3 或 DreamSim）
        根据 parent_job.pipeline_type 决定创建哪种引擎的任务
        """
        parent_id = str(parent_job["id"])
        params = json.loads(parent_job["params"]) if isinstance(parent_job["params"], str) else parent_job["params"]
        num_workers = params.get("num_workers", 24)

        # 根据 pipeline_type 确定第二阶段引擎和参数
        pipeline_type = parent_job.get("pipeline_type", "sam3_dinov3")
        if pipeline_type == "sam3_dreamsim":
            comparison_model_type = ModelType.DREAMSIM.value
            comparison_phase = "dreamsim"
            comparison_params = params.get("dreamsim", {})
        else:  # sam3_dinov3 (default)
            comparison_model_type = ModelType.DINOV3.value
            comparison_phase = "dinov3"
            comparison_params = params.get("dinov3", {})

        # 区分 before 和 after 的 SAM3 child
        before_child = None
        after_child = None
        for c in sam3_children:
            if c["pipeline_phase"] == "sam3_before":
                before_child = c
            elif c["pipeline_phase"] == "sam3_after":
                after_child = c

        if not before_child or not after_child:
            raise ValueError("Missing SAM3 before or after child")

        # 收集两个 SAM3 child 的结果文件
        before_results_dir = os.path.join(JOBS_DIR, str(before_child["id"]), "results")
        after_results_dir = os.path.join(JOBS_DIR, str(after_child["id"]), "results")

        # 也检查 spillover 目录
        before_spillover_dir = os.path.join(SPILLOVER_DIR, str(before_child["id"]))
        after_spillover_dir = os.path.join(SPILLOVER_DIR, str(after_child["id"]))

        def collect_result_files(results_dir: str, spillover_dir: str) -> dict[str, str]:
            """收集结果文件，返回 {stem: full_path} 映射"""
            files = {}
            for d in [results_dir, spillover_dir]:
                if not os.path.isdir(d):
                    continue
                for fname in os.listdir(d):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        stem = os.path.splitext(fname)[0]
                        files[stem] = os.path.join(d, fname)
            return files

        before_files = collect_result_files(before_results_dir, before_spillover_dir)
        after_files = collect_result_files(after_results_dir, after_spillover_dir)

        if not before_files:
            raise ValueError(f"No result files in SAM3 before child {before_child['id']}")
        if not after_files:
            raise ValueError(f"No result files in SAM3 after child {after_child['id']}")

        # 按文件名配对（stem 匹配）
        image_pairs = []
        matched_keys = set(before_files.keys()) & set(after_files.keys())
        for key in sorted(matched_keys):
            image_pairs.append({
                "before": before_files[key],
                "after": after_files[key],
            })

        if not image_pairs:
            # 如果精确匹配失败，尝试按排序顺序一一配对
            before_sorted = sorted(before_files.values())
            after_sorted = sorted(after_files.values())
            pair_count = min(len(before_sorted), len(after_sorted))
            for i in range(pair_count):
                image_pairs.append({
                    "before": before_sorted[i],
                    "after": after_sorted[i],
                })
            logger.warning(
                f"Pipeline {parent_id}: no exact name matches, "
                f"falling back to positional pairing ({pair_count} pairs)"
            )

        if not image_pairs:
            raise ValueError("No image pairs could be formed from SAM3 results")

        logger.info(f"Pipeline {parent_id}: paired {len(image_pairs)} image pairs for {comparison_phase}")

        # 创建变化检测 child job
        ds_job_id = str(uuid.uuid4())
        ds_total = len(image_pairs)
        ds_buckets = compute_bucket_count(ds_total, num_workers)

        await JobRepository.create_job(
            comparison_model_type,
            parent_job["input_path"],
            comparison_params,
            ds_total,
            job_id=ds_job_id,
            parent_job_id=parent_id,
            pipeline_phase=comparison_phase,
        )

        # 分发任务
        if comparison_phase == "dreamsim":
            task_descriptors = distribute_dreamsim_tasks(ds_job_id, image_pairs, comparison_params, ds_buckets)
        else:
            task_descriptors = distribute_dinov3_tasks(ds_job_id, image_pairs, comparison_params, ds_buckets)

        await TaskRepository.bulk_create(ds_job_id, task_descriptors)
        await JobRepository.update_status(ds_job_id, JobStatus.RUNNING.value)

        logger.info(
            f"Pipeline {parent_id}: {comparison_phase} child {ds_job_id} created "
            f"with {ds_total} tasks in {ds_buckets} buckets"
        )

    async def _reclaim_stale_tasks(self):
        """
        回收超时的 running 任务
        如果 Worker 崩溃，其正在处理的任务会一直停留在 running 状态
        Janitor 将超时的任务重置为 pending，让其他 Worker 重新抢占
        """
        jobs = await JobRepository.list_jobs(limit=100)
        for job in jobs:
            if job["status"] not in ("pending", "running"):
                continue
            # 回收 running 超过 TASK_TIMEOUT 秒的任务（含毒丸防护）
            await TaskRepository.reclaim_stale_tasks(
                str(job["id"]), TASK_TIMEOUT, TASK_MAX_RETRIES
            )

    def _check_heartbeats(self):
        """
        检查 Worker 心跳，标记不健康的 Worker
        每个 Worker 定期在心跳目录写入 timestamp 文件
        如果某个 Worker 的心跳超过 HEARTBEAT_TIMEOUT 秒未更新，视为不健康
        """
        if not os.path.isdir(HEARTBEAT_DIR):
            return  # 心跳目录不存在，跳过

        now = time.time()
        stale_workers = []
        # 遍历心跳文件
        for fname in os.listdir(HEARTBEAT_DIR):
            if not fname.endswith(".ts"):
                continue  # 只处理 .ts 后缀文件
            fpath = os.path.join(HEARTBEAT_DIR, fname)
            try:
                with open(fpath, "r") as f:
                    ts = float(f.read().strip())  # 读取 unix timestamp
                # 判断是否超时
                if now - ts > HEARTBEAT_TIMEOUT:
                    stale_workers.append(fname.replace(".ts", ""))
            except (ValueError, OSError):
                pass  # 文件读取异常时跳过

        # 记录不健康的 Worker（仅告警，不自动处理）
        if stale_workers:
            logger.warning(f"Stale workers detected: {stale_workers}")

    def _check_spillover(self):
        """
        检查结果目录大小，超过阈值时执行 spillover 落盘
        当某个 Job 的 results/ 目录超过 SPILLOVER_THRESHOLD_MB 时，
        自动将最老的一半文件移动到 spillover/ 目录，防止磁盘爆满
        """
        if not os.path.isdir(JOBS_DIR):
            return

        for job_id in os.listdir(JOBS_DIR):
            results_dir = os.path.join(JOBS_DIR, job_id, "results")
            if not os.path.isdir(results_dir):
                continue

            # 计算 results 目录总大小 (MB)
            total_size_mb = _dir_size_mb(results_dir)
            if total_size_mb > SPILLOVER_THRESHOLD_MB:
                # 创建 spillover 目标目录
                spillover_dest = os.path.join(SPILLOVER_DIR, job_id)
                os.makedirs(spillover_dest, exist_ok=True)
                # 按修改时间排序，移动最老的一半文件
                # 策略说明：
                # - 只移动一半（而非全部）是为了避免 spillover 目录也因一次性大量写入导致 I/O 尖峰
                # - 移动最老的文件优先，因为它们更可能已经被用户下载或不再需要实时访问
                # - 移动后 Janitor 下次巡检时如果仍然超标，会再移动一半（渐进式释放）
                files = sorted(
                    os.listdir(results_dir),
                    key=lambda f: os.path.getmtime(os.path.join(results_dir, f)),
                )
                moved = 0
                for f in files[:len(files) // 2]:  # 移动前一半（最老的）
                    src = os.path.join(results_dir, f)
                    dst = os.path.join(spillover_dest, f)
                    shutil.move(src, dst)
                    moved += 1
                logger.info(
                    f"Spillover: moved {moved} files from job {job_id} "
                    f"({total_size_mb:.0f}MB > {SPILLOVER_THRESHOLD_MB}MB threshold)"
                )


    async def _cleanup_old_jobs(self):
        """
        清理超过 CLEANUP_RETENTION_DAYS 天的已终态 Job
        删除数据库记录（CASCADE 自动删 tasks）和文件系统目录
        """
        try:
            deleted_ids = await JobRepository.cleanup_old_jobs(CLEANUP_RETENTION_DAYS)
            if not deleted_ids:
                return

            # 删除文件系统中对应的 Job 目录和 spillover 目录
            for job_id in deleted_ids:
                for base_dir in [JOBS_DIR, SPILLOVER_DIR]:
                    job_dir = os.path.join(base_dir, job_id)
                    if os.path.isdir(job_dir):
                        shutil.rmtree(job_dir, ignore_errors=True)

            logger.info(f"Cleanup: removed {len(deleted_ids)} old jobs")
        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)


def _dir_size_mb(path: str) -> float:
    """
    计算目录总大小 (MB)
    递归遍历所有文件，累加文件大小
    """
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass  # 文件被删除时跳过
    return total / (1024 * 1024)  # bytes → MB
