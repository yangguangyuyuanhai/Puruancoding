"""
worker 模块 - 推理工作进程层，运行在 Docker 容器中
本模块包含以下子模块：
  - launcher.py:        容器入口点，使用 multiprocessing 管理 N 个 Worker 子进程
  - base_worker.py:     Worker 主循环，负责任务抢占、推理调度、心跳维护
  - engine_sam3.py:     SAM3 分割检测引擎（Zero-DCE 增强 + SAM3 文本提示分割）
  - engine_dreamsim.py: DreamSim 变化检测引擎（滑窗扫描 + 感知距离计算 + 热力图生成）
依赖规则：worker 层依赖 shared/ 层（配置和协议），不依赖 gateway/ 层
每个容器绑定一张 GPU，容器内由 launcher.py 启动 1~4 个 Worker 子进程共享同一张 GPU
"""
