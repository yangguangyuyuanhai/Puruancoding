"""
全局配置 - 所有可变参数集中管理，通过环境变量覆盖
"""
import os

# ============================================================
# 路径配置
# ============================================================
SHARED_ROOT = os.getenv("BATCH_SHARED_ROOT", "/data/batch_shared")
SPILLOVER_DIR = os.getenv("BATCH_SPILLOVER_DIR", f"{SHARED_ROOT}/spillover")
HEARTBEAT_DIR = f"{SHARED_ROOT}/heartbeat"
JOBS_DIR = f"{SHARED_ROOT}/jobs"
SCALING_DIR = f"{SHARED_ROOT}/scaling"

# ============================================================
# 伸缩配置
# ============================================================
MIN_WORKERS_PER_GPU = int(os.getenv("MIN_WORKERS_PER_GPU", "1"))
MAX_WORKERS_PER_GPU = int(os.getenv("MAX_WORKERS_PER_GPU", "4"))
SCALE_UP_THRESHOLD_MULTIPLIER = int(os.getenv("SCALE_UP_MULTIPLIER", "2"))
SCALE_UP_SUSTAIN_SEC = int(os.getenv("SCALE_UP_SUSTAIN_SEC", "10"))
SCALE_DOWN_SUSTAIN_SEC = int(os.getenv("SCALE_DOWN_SUSTAIN_SEC", "30"))
SCALE_COOLDOWN_SEC = int(os.getenv("SCALE_COOLDOWN_SEC", "15"))

# ============================================================
# DB 配置
# ============================================================
PG_HOST = os.getenv("BATCH_PG_HOST", "localhost")
PG_PORT = int(os.getenv("BATCH_PG_PORT", "5433"))
PG_DB = os.getenv("BATCH_PG_DB", "batch_detection")
PG_USER = os.getenv("BATCH_PG_USER", "batch")
PG_PASS = os.getenv("BATCH_PG_PASS", "batch_secret")
PG_DSN = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# ============================================================
# Worker 配置
# ============================================================
WORKER_POLL_INTERVAL = float(os.getenv("WORKER_POLL_INTERVAL", "0.5"))
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "5"))
HEARTBEAT_TIMEOUT = int(os.getenv("HEARTBEAT_TIMEOUT", "30"))
TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "300"))

# ============================================================
# Janitor 配置
# ============================================================
JANITOR_INTERVAL = int(os.getenv("JANITOR_INTERVAL", "10"))
SPILLOVER_THRESHOLD_MB = int(os.getenv("SPILLOVER_THRESHOLD_MB", "10240"))

# ============================================================
# SAM3 模型配置
# ============================================================
SAM3_MODEL_PATH = os.getenv("SAM3_MODEL_PATH", "/app/checkpoints/sam3.pt")
SAM3_DCE_WEIGHTS = os.getenv("SAM3_DCE_WEIGHTS", "/app/checkpoints/Zero-DCE.pth")
SAM3_DCE_DEVICE = os.getenv("SAM3_DCE_DEVICE", "cuda")  # cuda | cpu

# ============================================================
# DreamSim 模型配置
# ============================================================
DREAMSIM_MODEL_TYPE = os.getenv("DREAMSIM_MODEL_TYPE", "ensemble")

# ============================================================
# Gateway 配置
# ============================================================
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8090"))

# ============================================================
# 图片文件扩展名
# ============================================================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
