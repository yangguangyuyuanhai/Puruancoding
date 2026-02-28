"""
数据协议 - 任务/结果数据结构 + EngineInterface 抽象接口
本模块是 Gateway 和 Worker 之间的通信契约，定义了所有共享数据结构
属于 shared 层，不依赖 gateway/ 或 worker/ 中的任何模块
"""
from __future__ import annotations  # 允许在类型注解中使用字符串形式的前向引用

import json       # JSON 序列化/反序列化
import uuid       # UUID 生成
from abc import ABC, abstractmethod  # 抽象基类，定义引擎接口规范
from dataclasses import dataclass, field, asdict  # 数据类装饰器，简化数据结构定义
from datetime import datetime  # 时间戳处理
from enum import Enum          # 枚举类型
from typing import Any, Optional  # 类型注解


# ============================================================
# 枚举定义 - Job 和 Task 的状态机 + 模型类型
# ============================================================
class JobStatus(str, Enum):
    """批量任务 (Job) 的生命周期状态"""
    PENDING = "pending"        # 已创建，等待 Worker 开始处理
    RUNNING = "running"        # 至少一个子任务正在运行
    COMPLETED = "completed"    # 所有子任务处理完毕（可能有部分失败）
    FAILED = "failed"          # 整体失败
    CANCELLED = "cancelled"    # 被用户手动取消


class TaskStatus(str, Enum):
    """单个子任务 (Task) 的生命周期状态"""
    PENDING = "pending"    # 等待被 Worker 抢占
    RUNNING = "running"    # 已被某个 Worker 抢占，正在推理
    DONE = "done"          # 推理成功完成
    FAILED = "failed"      # 推理失败（图片损坏、OOM 等）


class ModelType(str, Enum):
    """支持的推理模型类型"""
    SAM3 = "sam3"          # SAM3 分割检测
    DREAMSIM = "dreamsim"  # DreamSim 变化检测


# ============================================================
# 数据结构 - 用 dataclass 定义，支持 JSON 序列化
# ============================================================
@dataclass
class JobMeta:
    """
    任务元数据，写入 meta.json 文件到共享卷
    每个 Job 在文件系统中都有一个 meta.json 记录基本信息
    """
    job_id: str          # 任务唯一标识 (UUID)
    model_type: str      # 模型类型: "sam3" 或 "dreamsim"
    input_path: str      # 输入目录或 ZIP 文件路径
    params: dict         # 模型推理参数 (如 text, conf, thresh 等)
    total_tasks: int     # 子任务总数 (图片数或图片对数)
    status: str = JobStatus.PENDING.value  # 初始状态为 pending
    created_at: str = ""  # 创建时间 ISO 格式字符串

    def __post_init__(self):
        """dataclass 初始化后自动设置创建时间"""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        """序列化为格式化 JSON 字符串"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, data: str | dict) -> "JobMeta":
        """从 JSON 字符串或字典反序列化"""
        if isinstance(data, str):
            data = json.loads(data)
        return cls(**data)


@dataclass
class TaskDescriptor:
    """
    单个任务描述符，写入 pending/ 目录供 Worker 读取
    描述了 Worker 需要处理的一张图片或一对图片的所有信息
    """
    task_id: str              # 子任务唯一标识 (UUID)
    job_id: str               # 所属批量任务 ID
    bucket_id: int            # 分桶编号，用于任务分配的空间局部性
    model_type: str           # 模型类型: "sam3" 或 "dreamsim"
    params: dict              # 推理参数字典
    image_path: Optional[str] = None   # SAM3 模式：单张图片路径
    image_pair: Optional[dict] = None  # DreamSim 模式：{"before": 路径, "after": 路径}

    def to_json(self) -> str:
        """序列化为格式化 JSON 字符串"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, data: str | dict) -> "TaskDescriptor":
        """从 JSON 字符串或字典反序列化"""
        if isinstance(data, str):
            data = json.loads(data)
        return cls(**data)

    @property
    def filename(self) -> str:
        """
        生成任务文件名，取 task_id 前 8 位作为标识
        用于 pending/ 目录中的任务描述符文件命名（如 task_a3f1c2d4.json）
        注意：实际调度器使用序号命名（task_000001.json），此属性用于备选命名方案
        """
        return f"task_{self.task_id[:8]}.json"


@dataclass
class TaskResult:
    """任务执行结果，由 Worker 在完成推理后生成"""
    task_id: str                       # 子任务 ID
    job_id: str                        # 所属 Job ID
    status: str                        # 结果状态: "done" 或 "failed"
    result_path: Optional[str] = None  # 结果文件路径 (成功时)
    error_msg: Optional[str] = None    # 错误信息 (失败时)
    duration_sec: float = 0.0          # 推理耗时 (秒)

    def to_json(self) -> str:
        """序列化为格式化 JSON 字符串"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# ============================================================
# 引擎抽象接口 - 所有推理引擎的统一规范
# ============================================================
class EngineInterface(ABC):
    """
    推理引擎抽象基类，定义了 load/infer/unload 三个生命周期方法
    新增模型只需实现此接口 + @register_engine 装饰器即可接入系统
    Worker (base_worker.py) 通过此接口多态调用，无需关心具体引擎实现
    """

    @abstractmethod
    def load(self, device: str, **kwargs) -> None:
        """
        加载模型到指定设备
        Args:
            device: 设备标识，如 "cuda:0", "cpu"
            **kwargs: 额外参数，如 model_path, dce_weights 等
        """

    @abstractmethod
    def infer(self, task: TaskDescriptor, result_dir: str) -> dict:
        """
        执行单次推理
        Args:
            task: 任务描述符，包含图片路径和推理参数
            result_dir: 结果文件写入目录
        Returns:
            {"result_path": str, "metadata": dict} 结果路径和元信息
        """

    @abstractmethod
    def unload(self) -> None:
        """释放模型资源和 GPU 显存"""


# ============================================================
# 引擎注册表 (工厂模式) - 按 model_type 字符串动态查找引擎类
# ============================================================
# 全局引擎注册表，key 是 model_type 字符串，value 是引擎类
# 注册表在模块导入时自动填充：当 engine_sam3.py 或 engine_dreamsim.py 被 import 时，
# @register_engine 装饰器会自动将引擎类注册到此字典中
# Worker 启动时通过 get_engine(model_type) 查找并实例化对应引擎
ENGINE_REGISTRY: dict[str, type[EngineInterface]] = {}


def register_engine(name: str):
    """
    装饰器：将推理引擎类注册到全局注册表
    使用方式: @register_engine("sam3")
    注册后可通过 get_engine("sam3") 获取引擎类
    """
    def decorator(cls):
        ENGINE_REGISTRY[name] = cls  # 将类注册到全局字典
        return cls
    return decorator


def get_engine(name: str) -> type[EngineInterface]:
    """
    工厂方法：按 model_type 字符串获取对应的引擎类
    Raises:
        ValueError: 未注册的引擎名称
    """
    if name not in ENGINE_REGISTRY:
        raise ValueError(f"Unknown engine: {name}. Available: {list(ENGINE_REGISTRY.keys())}")
    return ENGINE_REGISTRY[name]
