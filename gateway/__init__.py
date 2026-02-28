"""
gateway 模块 - API 网关层，运行在宿主机 conda 环境中
本模块包含以下子模块：
  - app.py:       FastAPI HTTP 入口，提供 RESTful API（任务管理、健康检查、结果导出）
  - db.py:        PostgreSQL asyncpg 连接池封装，提供 JobRepository 和 TaskRepository
  - indexer.py:   文件索引器，将用户输入（ZIP/目录）转化为图片路径列表
  - scheduler.py: 任务调度器，按 Bucket 策略将图片分配给 Worker
  - janitor.py:   后台守护线程，负责心跳巡检、超时回收、spillover 落盘
  - scaler.py:    弹性伸缩控制器，根据队列深度动态调整 Worker 进程数量
依赖规则：gateway 层依赖 shared/ 层，不依赖 worker/ 层
"""
