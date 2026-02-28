"""
数据协议 - 任务/结果数据结构 + EngineInterface 抽象
"""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ============================================================
# 枚举定义
# ============================================================
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ModelType(str, Enum):
    SAM3 = "sam3"
    DREAMSIM = "dreamsim"


# ============================================================
# 数据结构
# ============================================================
@dataclass
class JobMeta:
    """任务元数据，写入 meta.json"""
    job_id: str
    model_type: str
    input_path: str
    params: dict
    total_tasks: int
    status: str = JobStatus.PENDING.value
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, data: str | dict) -> "JobMeta":
        if isinstance(data, str):
            data = json.loads(data)
        return cls(**data)


@dataclass
class TaskDescriptor:
    """单个任务描述符，写入 pending/ 目录"""
    task_id: str
    job_id: str
    bucket_id: int
    model_type: str
    params: dict
    image_path: Optional[str] = None          # SAM3 单图
    image_pair: Optional[dict] = None          # DreamSim {"before": ..., "after": ...}

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, data: str | dict) -> "TaskDescriptor":
        if isinstance(data, str):
            data = json.loads(data)
        return cls(**data)

    @property
    def filename(self) -> str:
        return f"task_{self.task_id[:8]}.json"


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    job_id: str
    status: str
    result_path: Optional[str] = None
    error_msg: Optional[str] = None
    duration_sec: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# ============================================================
# 引擎抽象接口
# ============================================================
class EngineInterface(ABC):
    """所有推理引擎必须实现此接口"""

    @abstractmethod
    def load(self, device: str, **kwargs) -> None:
        """加载模型到指定设备"""

    @abstractmethod
    def infer(self, task: TaskDescriptor, result_dir: str) -> dict:
        """
        执行推理
        Args:
            task: 任务描述符
            result_dir: 结果写入目录
        Returns:
            {"result_path": str, "metadata": dict}
        """

    @abstractmethod
    def unload(self) -> None:
        """释放模型资源"""


# ============================================================
# 引擎注册表 (工厂模式)
# ============================================================
ENGINE_REGISTRY: dict[str, type[EngineInterface]] = {}


def register_engine(name: str):
    """装饰器：注册推理引擎到全局注册表"""
    def decorator(cls):
        ENGINE_REGISTRY[name] = cls
        return cls
    return decorator


def get_engine(name: str) -> type[EngineInterface]:
    """按 model_type 获取引擎类"""
    if name not in ENGINE_REGISTRY:
        raise ValueError(f"Unknown engine: {name}. Available: {list(ENGINE_REGISTRY.keys())}")
    return ENGINE_REGISTRY[name]
