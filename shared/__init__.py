"""
shared 模块 - 跨层共享的配置和数据协议
本模块是 Gateway 和 Worker 之间的桥梁，包含：
  - config.py: 全局常量和连接配置（路径、数据库、伸缩参数等），通过环境变量可覆盖
  - protocol.py: 数据结构定义（JobMeta, TaskDescriptor, TaskResult）、
                 状态枚举（JobStatus, TaskStatus, ModelType）、
                 EngineInterface 抽象接口和引擎注册表
依赖规则：shared 层不依赖 gateway/ 或 worker/ 中的任何模块，处于依赖树的最底层
"""
