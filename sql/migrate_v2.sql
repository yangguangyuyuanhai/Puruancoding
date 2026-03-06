-- ============================================================
-- 违章建筑检测系统 v2.0 增量迁移脚本
-- 在现有 v1.0 数据库上执行，添加 v2.0 所需的新字段和索引
-- 使用 IF NOT EXISTS / ADD COLUMN IF NOT EXISTS 保证幂等性
-- ============================================================

-- jobs 表新增 pipeline_type 字段
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS pipeline_type VARCHAR(32);

-- tasks 表新增 params、priority、metadata 字段
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS params JSONB;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS priority INTEGER DEFAULT 0;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS metadata JSONB;

-- 新增索引：优先级抢占
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(status, priority DESC, id);
-- 新增索引：清理查询
CREATE INDEX IF NOT EXISTS idx_jobs_cleanup ON jobs(status, updated_at);
