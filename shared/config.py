"""
全局配置 - 所有可变参数集中管理，通过环境变量覆盖
所有配置项均有默认值，部署时可通过环境变量灵活修改，无需改动代码
"""
import os  # 标准库：用于读取环境变量

# ============================================================
# 路径配置
# ============================================================
# 共享卷根目录，Gateway 和 Worker 容器均挂载此目录进行 IPC 通信
SHARED_ROOT = os.getenv("BATCH_SHARED_ROOT", "/data/batch_shared")
# 溢出落盘目录，当结果文件过多时 Janitor 自动将旧文件移到此目录
SPILLOVER_DIR = os.getenv("BATCH_SPILLOVER_DIR", f"{SHARED_ROOT}/spillover")
# Worker 心跳文件目录，每个 Worker 定期写入 timestamp 文件表示存活
HEARTBEAT_DIR = f"{SHARED_ROOT}/heartbeat"
# Job 数据目录，每个 Job 在此下创建子目录存放任务描述符和结果
JOBS_DIR = f"{SHARED_ROOT}/jobs"
# AutoScaler 伸缩指令目录，Gateway 写入 JSON 指令，Launcher 读取执行
SCALING_DIR = f"{SHARED_ROOT}/scaling"

# ============================================================
# 弹性伸缩配置
# ============================================================
# 每张 GPU 最少保活的 Worker 进程数，空闲时不会低于此值
MIN_WORKERS_PER_GPU = int(os.getenv("MIN_WORKERS_PER_GPU", "1"))
# 每张 GPU 最多启动的 Worker 进程数 (48GB L20 / 12GB ≈ 4)
MAX_WORKERS_PER_GPU = int(os.getenv("MAX_WORKERS_PER_GPU", "4"))
# 扩容阈值倍数：pending 任务 > 当前 Worker 数 × 此值时触发扩容
SCALE_UP_THRESHOLD_MULTIPLIER = int(os.getenv("SCALE_UP_MULTIPLIER", "2"))
# 扩容条件需持续满足的秒数，避免瞬间抖动导致频繁扩容
SCALE_UP_SUSTAIN_SEC = int(os.getenv("SCALE_UP_SUSTAIN_SEC", "10"))
# 缩容条件需持续满足的秒数（队列为空），避免任务间歇期误缩
SCALE_DOWN_SUSTAIN_SEC = int(os.getenv("SCALE_DOWN_SUSTAIN_SEC", "30"))
# 两次伸缩操作的最小间隔秒数，防止振荡
SCALE_COOLDOWN_SEC = int(os.getenv("SCALE_COOLDOWN_SEC", "15"))

# ============================================================
# 数据库配置 (独立 PostgreSQL 15 实例)
# ============================================================
# PostgreSQL 主机地址，容器内 Worker 使用容器名 "batch-postgres"
PG_HOST = os.getenv("BATCH_PG_HOST", "localhost")
# PostgreSQL 端口，使用 5433 避开现有 5432 实例
# 宿主机 Gateway 连接时使用 5433（docker-compose 映射 5433:5432）
# 容器内 Worker 连接时使用 5432（容器间通过 Docker 网络直连 PG 原始端口）
PG_PORT = int(os.getenv("BATCH_PG_PORT", "5433"))
# 数据库名称
PG_DB = os.getenv("BATCH_PG_DB", "batch_detection")
# 数据库用户名
PG_USER = os.getenv("BATCH_PG_USER", "batch")
# 数据库密码
PG_PASS = os.getenv("BATCH_PG_PASS", "batch_secret")
# 完整的 PostgreSQL DSN 连接字符串，由上述参数自动拼接
# 格式: postgresql://user:password@host:port/database
# 注意: 此 DSN 仅供参考或某些需要完整连接串的库使用，
# Gateway (asyncpg) 和 Worker (psycopg2) 实际使用上述独立参数连接
PG_DSN = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# ============================================================
# Worker 配置
# ============================================================
# 无任务时 Worker 轮询数据库的间隔秒数
WORKER_POLL_INTERVAL = float(os.getenv("WORKER_POLL_INTERVAL", "0.5"))
# Worker 写入心跳文件的间隔秒数
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "5"))
# 心跳超时判定秒数，超过此时间未更新心跳的 Worker 被视为不健康
HEARTBEAT_TIMEOUT = int(os.getenv("HEARTBEAT_TIMEOUT", "30"))
# 任务超时回收秒数，running 状态超过此时间的任务被 Janitor 重置为 pending
TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "300"))
# 毒丸防护：任务最大重试次数，超过此值的超时任务直接标记为 failed 而非重新排队
# 防止一张导致 OOM 的超大图片在所有 Worker 间无限循环导致集群雪崩
TASK_MAX_RETRIES = int(os.getenv("TASK_MAX_RETRIES", "3"))

# ============================================================
# Janitor 配置 (后台守护线程)
# ============================================================
# Janitor 巡检间隔秒数
JANITOR_INTERVAL = int(os.getenv("JANITOR_INTERVAL", "10"))
# 触发 spillover 落盘的结果目录大小阈值 (MB)
# 默认 10240MB = 10GB，超过后 Janitor 自动将最老的一半文件移到 spillover/ 目录
# 可根据服务器磁盘容量调整：磁盘紧张时调小，磁盘充裕时调大
SPILLOVER_THRESHOLD_MB = int(os.getenv("SPILLOVER_THRESHOLD_MB", "10240"))

# ============================================================
# SAM3 模型配置
# ============================================================
# SAM3 分割模型权重文件路径 (容器内路径)
SAM3_MODEL_PATH = os.getenv("SAM3_MODEL_PATH", "/app/checkpoints/sam3.pt")
# Zero-DCE 暗光增强模型权重路径 (容器内路径)
SAM3_DCE_WEIGHTS = os.getenv("SAM3_DCE_WEIGHTS", "/app/checkpoints/Zero-DCE.pth")
# Zero-DCE 运行设备：cuda (GPU，默认) 或 cpu (小显存卡防 OOM)
SAM3_DCE_DEVICE = os.getenv("SAM3_DCE_DEVICE", "cuda")  # cuda | cpu

# ============================================================
# DreamSim 模型配置
# ============================================================
# DreamSim 模型类型，ensemble 模式效果最好但稍慢
DREAMSIM_MODEL_TYPE = os.getenv("DREAMSIM_MODEL_TYPE", "ensemble")

# ============================================================
# DINOv3 模型配置
# ============================================================
# DINOv3 默认敏感度（peak_ratio），越小越敏感
DINOV3_PEAK_RATIO = float(os.getenv("DINOV3_PEAK_RATIO", "0.4"))
# DINOv3 默认推理分辨率
DINOV3_RES = int(os.getenv("DINOV3_RES", "2048"))
# DINOv3 最低置信度阈值
DINOV3_DINO_CONF = float(os.getenv("DINOV3_DINO_CONF", "0.15"))

# ============================================================
# LoFTR 配准配置
# ============================================================
# LoFTR 模型权重路径（容器内路径）
LOFTR_WEIGHT_PATH = os.getenv("LOFTR_WEIGHT_PATH", "/app/weights/outdoor_ds.ckpt")
# LoFTR 配置文件路径（容器内路径）
LOFTR_CONFIG_PATH = os.getenv("LOFTR_CONFIG_PATH", "/app/configs/loftr/outdoor/loftr_ds.py")
# 配准最少匹配特征点数，低于此值视为配准失败
LOFTR_MIN_MATCHES = int(os.getenv("LOFTR_MIN_MATCHES", "30"))

# ============================================================
# 历史清理配置
# ============================================================
# 历史任务保留天数，超过此天数的已终态 Job 将被 Janitor 自动清理
CLEANUP_RETENTION_DAYS = int(os.getenv("CLEANUP_RETENTION_DAYS", "30"))

# ============================================================
# 重推配置
# ============================================================
# 重推任务优先级（高于普通批量任务的 0）
RERUN_PRIORITY = int(os.getenv("RERUN_PRIORITY", "1"))

# ============================================================
# Gateway 配置
# ============================================================
# Gateway HTTP 服务监听地址
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
# Gateway HTTP 服务监听端口
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8090"))

# ============================================================
# 图片文件扩展名 (用于索引器扫描过滤)
# ============================================================
# 支持的图片格式集合，扫描目录时只匹配这些扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
