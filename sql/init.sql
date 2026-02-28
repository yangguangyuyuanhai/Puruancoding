-- ============================================================
-- 违章建筑检测系统 - 批量推理平台
-- PostgreSQL 15 初始化脚本
-- ============================================================

-- 如果数据库不存在则创建 (通过 docker-compose 环境变量自动创建)
-- CREATE DATABASE batch_detection;

-- 启用 UUID 生成扩展
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- Jobs 表：批量任务
-- ============================================================
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type VARCHAR(20) NOT NULL,        -- "sam3" | "dreamsim"
    status VARCHAR(20) DEFAULT 'pending',   -- pending | running | completed | failed | cancelled
    total_tasks INTEGER NOT NULL,
    completed_tasks INTEGER DEFAULT 0,
    failed_tasks INTEGER DEFAULT 0,
    input_path TEXT NOT NULL,
    params JSONB,                           -- 模型参数 (conf, text, thresh, etc.)
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Tasks 表：单个子任务
-- ============================================================
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    bucket_id INTEGER,
    status VARCHAR(20) DEFAULT 'pending',   -- pending | running | done | failed
    image_path TEXT,                         -- SAM3 单图路径
    image_pair JSONB,                       -- DreamSim {"before": "...", "after": "..."}
    result_path TEXT,                        -- 结果文件路径
    error_msg TEXT,
    worker_id VARCHAR(50),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

-- ============================================================
-- 索引
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_tasks_job_status ON tasks(job_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_bucket ON tasks(job_id, bucket_id, status);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_tasks_worker ON tasks(worker_id, status);
