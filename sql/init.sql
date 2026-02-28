-- ============================================================
-- 违章建筑检测系统 - 批量推理平台
-- PostgreSQL 15 初始化脚本
-- 本脚本通过 docker-compose 挂载到容器的
-- /docker-entrypoint-initdb.d/ 目录，在 PG 首次启动时自动执行
-- ============================================================

-- 如果数据库不存在则创建 (通过 docker-compose 环境变量 POSTGRES_DB 自动创建)
-- CREATE DATABASE batch_detection;

-- 启用 pgcrypto 扩展，提供 gen_random_uuid() 函数用于生成 UUID 主键
-- UUID v4 (随机) 作为主键的优势：
--   1. 全局唯一，多 Gateway 实例并行创建 Job 不会冲突
--   2. 无需数据库序列 (sequence)，降低写入竞争
--   3. 不可猜测，避免安全枚举攻击
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- Jobs 表：批量任务 (一次用户提交 = 一条 Job 记录)
-- ============================================================
CREATE TABLE IF NOT EXISTS jobs (
    -- 主键：使用 UUID 而非自增 ID，避免多 Gateway 实例 ID 冲突
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 模型类型：只允许 "sam3" (分割检测) 或 "dreamsim" (变化检测)
    model_type VARCHAR(20) NOT NULL,
    -- 状态机：pending → running → completed / failed / cancelled
    status VARCHAR(20) DEFAULT 'pending',
    -- 子任务总数 (图片数或图片对数)
    total_tasks INTEGER NOT NULL,
    -- 已成功完成的子任务数 (Worker 每完成一个 +1)
    completed_tasks INTEGER DEFAULT 0,
    -- 失败的子任务数 (Worker 每失败一个 +1)
    failed_tasks INTEGER DEFAULT 0,
    -- 用户提供的输入路径 (服务器本地目录或 ZIP 路径)
    input_path TEXT NOT NULL,
    -- 推理参数 (JSONB 格式, 如 {"text":"building","conf":0.2} 或 {"thresh":0.3,"crop":224})
    params JSONB,
    -- 任务创建时间 (带时区)
    created_at TIMESTAMPTZ DEFAULT now(),
    -- 最后更新时间 (每次状态变更或进度更新时刷新)
    updated_at TIMESTAMPTZ DEFAULT now(),
    -- 流水线支持：父任务 ID（子任务才有值，父任务删除时级联删除子任务）
    parent_job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    -- 流水线阶段标识: "sam3_before" | "sam3_after" | "dreamsim"（仅子任务有值）
    pipeline_phase VARCHAR(20)
);

-- ============================================================
-- Tasks 表：单个子任务 (一张图或一对图 = 一条 Task 记录)
-- ============================================================
CREATE TABLE IF NOT EXISTS tasks (
    -- 主键：子任务唯一标识
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 外键：所属的 Job，Job 被删除时级联删除其下所有 Task
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    -- 分桶编号：用于将任务按空间局部性分配给不同 Worker
    bucket_id INTEGER,
    -- 状态机：pending → running → done / failed
    status VARCHAR(20) DEFAULT 'pending',
    -- SAM3 模式：单张图片的绝对路径
    image_path TEXT,
    -- DreamSim 模式：图片对 JSON {"before": "/path/a.jpg", "after": "/path/b.jpg"}
    image_pair JSONB,
    -- 推理结果文件的绝对路径 (成功时填写)
    result_path TEXT,
    -- 错误信息 (失败时填写，用于排查)
    error_msg TEXT,
    -- 执行该任务的 Worker ID (格式: sam3_gpu0_w0)
    worker_id VARCHAR(50),
    -- 开始处理时间 (Worker 抢占任务时填写)
    started_at TIMESTAMPTZ,
    -- 完成时间 (成功或失败时填写)
    finished_at TIMESTAMPTZ
);

-- ============================================================
-- 索引 - 加速高频查询
-- 说明：索引设计基于实际查询模式，覆盖了最频繁的 WHERE 条件组合
-- PostgreSQL 使用 B-tree 索引，对等值查询和范围查询均有加速效果
-- ============================================================
-- 索引 1: 按 Job 和状态查询任务
-- 使用场景: Janitor 统计 Job 完成度 (SELECT ... WHERE job_id=? AND status=?)
--           Worker 抢占任务 (SELECT ... WHERE job_id=? AND status='pending')
CREATE INDEX IF NOT EXISTS idx_tasks_job_status ON tasks(job_id, status);
-- 索引 2: 按 Job、Bucket 和状态查询
-- 使用场景: Worker 按 Bucket 抢占任务 (WHERE job_id=? AND bucket_id=? AND status='pending')
--           AutoScaler 按 Bucket 统计 pending 数
CREATE INDEX IF NOT EXISTS idx_tasks_bucket ON tasks(job_id, bucket_id, status);
-- 索引 3: 按 Job 状态查询
-- 使用场景: Gateway 筛选活跃任务 (WHERE status IN ('pending', 'running'))
--           Dashboard 按状态分类展示
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
-- 索引 4: 按父任务 ID 查询子任务
-- 使用场景: Janitor 查询 pipeline job 的所有子任务 (WHERE parent_job_id=?)
CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_job_id);
-- 索引 4: 按 Worker 和状态查询
-- 使用场景: 诊断某个 Worker 处理了哪些任务 (WHERE worker_id=? AND status=?)
--           排查 Worker 崩溃时定位其未完成的任务
CREATE INDEX IF NOT EXISTS idx_tasks_worker ON tasks(worker_id, status);
