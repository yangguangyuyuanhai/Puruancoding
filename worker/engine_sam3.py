"""
SAM3 推理引擎 - 改编自 sam3.py + app.py
本模块封装了 SAM3 语义分割推理的完整流程：
  1. Zero-DCE 暗光增强：对输入图片进行亮度增强，改善低光照条件下的检测效果
  2. SAM3 分割推理：使用文本提示词进行语义分割，生成目标区域的掩码
  3. 后处理：形态学膨胀、孔洞填充、矩形修复等操作优化掩码质量
  4. 抠像输出：将掩码应用到原图上，生成黑底抠像结果
属于 worker 层，实现了 shared/protocol.py 中定义的 EngineInterface 接口
"""
from __future__ import annotations  # 支持类型注解的前向引用

import logging  # 日志记录
import os       # 文件路径操作
import time     # 计时（保留供未来使用）

import cv2           # OpenCV: 图片读写、形态学操作、轮廓检测
import numpy as np   # NumPy: 数组操作、掩码运算
import torch         # PyTorch: 深度学习推理框架
import torch.nn as nn         # PyTorch 神经网络模块
import torch.nn.functional as F  # PyTorch 函数式接口（tanh 等激活函数）
from PIL import Image  # Pillow: SAM3 模型接受 PIL 格式输入

# 导入引擎接口和注册装饰器
from shared.protocol import EngineInterface, TaskDescriptor, register_engine
# 导入 SAM3 相关配置（模型路径、DCE 权重路径、DCE 设备）
from shared.config import SAM3_MODEL_PATH, SAM3_DCE_WEIGHTS, SAM3_DCE_DEVICE

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# ============================================================
# Zero-DCE 暗光增强模块 (复用自 sam3.py)
# Zero-DCE (Zero-Reference Deep Curve Estimation) 是一种无监督的
# 低光照图像增强方法，通过学习像素级的曲线映射函数提升图像亮度
# ============================================================
class C_Net(nn.Module):
    """
    Zero-DCE 曲线估计网络
    网络结构：7 层卷积的 U-Net 变体
    - 编码器 (conv1-4): 逐层提取特征，通道数 3 -> 32
    - 解码器 (conv5-7): 通过跳跃连接 (skip connection) 融合特征
    - 输出 24 通道 = 8 组 x 3 通道 (RGB)，每组是一个增强曲线参数
    """
    def __init__(self):
        super(C_Net, self).__init__()
        number_f = 32  # 特征通道数
        # 编码器：4 层 3x3 卷积，padding=1 保持空间尺寸不变
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)        # 输入 RGB 3 通道 -> 32 通道
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)  # 32 -> 32
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)  # 32 -> 32
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)  # 32 -> 32
        # 解码器：通过 concat 跳跃连接，输入通道数 = 32*2 = 64
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)  # 64 -> 32 (x3+x4 拼接)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)  # 64 -> 32 (x2+x5 拼接)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)        # 64 -> 24 (x1+x6 拼接)
        # 24 = 8 组 × 3 通道(RGB)，每组是一个增强曲线的参数
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活函数，inplace=True 节省显存

    def forward(self, x):
        """
        前向传播
        输入 x: [B, 3, H, W] 归一化到 [0,1] 的 RGB 图像
        输出: [B, 3, H, W] 增强后的 RGB 图像
        """
        # 编码阶段：逐层提取特征
        x1 = self.relu(self.e_conv1(x))   # 第 1 层特征
        x2 = self.relu(self.e_conv2(x1))  # 第 2 层特征
        x3 = self.relu(self.e_conv3(x2))  # 第 3 层特征
        x4 = self.relu(self.e_conv4(x3))  # 第 4 层特征（最深层）
        # 解码阶段：通过 skip connection 拼接对称层的特征
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))  # 拼接 x3 和 x4
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))  # 拼接 x2 和 x5
        # 输出 24 通道的曲线参数，使用 tanh 将值域限制在 [-1, 1]
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        # 将 24 通道按 3 通道一组拆分为 8 组曲线参数 r1~r8
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        # 迭代式曲线增强：公式 x = x + r * (x^2 - x)
        # 这是 Zero-DCE 论文的核心公式，通过 8 次迭代逐步增强亮度
        # 每次迭代中 r * (x^2 - x) 是一个像素级的亮度调整量
        x = x + r1 * (torch.pow(x, 2) - x)             # 第 1 次增强
        x = x + r2 * (torch.pow(x, 2) - x)             # 第 2 次增强
        x = x + r3 * (torch.pow(x, 2) - x)             # 第 3 次增强
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)  # 第 4 次增强（中间结果）
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)  # 第 5 次
        x = x + r6 * (torch.pow(x, 2) - x)             # 第 6 次增强
        x = x + r7 * (torch.pow(x, 2) - x)             # 第 7 次增强
        enhance_image = x + r8 * (torch.pow(x, 2) - x)  # 第 8 次增强（最终结果）
        return enhance_image


class ZeroDCE_Enhancer:
    """
    Zero-DCE 暗光增强器封装
    将 C_Net 模型的加载、预处理、推理封装为简单的 enhance() 接口
    """
    def __init__(self, weights_path: str, device: str = "cuda"):
        """
        初始化增强器
        Args:
            weights_path: Zero-DCE 预训练权重文件路径
            device: 运行设备 "cuda" 或 "cpu"
        """
        self.device = device
        self.model = C_Net().to(device)  # 创建模型并移到指定设备
        if os.path.exists(weights_path):
            try:
                # 加载预训练权重
                self.model.load_state_dict(torch.load(weights_path, map_location=device))
                logger.info(f"[DCE] Enhancer loaded on {device}: {weights_path}")
            except Exception as e:
                logger.warning(f"[DCE] Failed to load weights: {e}")
        self.model.eval()  # 设置为推理模式（关闭 Dropout 和 BatchNorm 的训练行为）

    def enhance(self, cv2_img):
        """
        对 OpenCV 格式的图片进行暗光增强
        Args:
            cv2_img: OpenCV BGR 格式的图片数组 [H, W, 3]，值域 [0, 255]
        Returns:
            增强后的图片数组 [H, W, 3]，值域 [0, 255]，uint8 类型
        """
        # 归一化到 [0, 1] 区间
        img = (np.asarray(cv2_img) / 255.0)
        # 转换为 PyTorch 张量: [H,W,3] -> [3,H,W] -> [1,3,H,W] (增加 batch 维度)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():  # 推理模式，不计算梯度，节省显存
            enhanced_image = self.model(img)
        # 转回 NumPy 数组: [1,3,H,W] -> [3,H,W] -> [H,W,3] -> CPU
        enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # 反归一化到 [0, 255] 并裁剪越界值
        return np.clip(enhanced_image * 255, 0, 255).astype("uint8")


# ============================================================
# SAM3 推理引擎 - 实现 EngineInterface 接口
# ============================================================
@register_engine("sam3")  # 注册到全局引擎注册表，key 为 "sam3"
class SAM3Engine(EngineInterface):
    """
    SAM3 语义分割推理引擎
    实现了 EngineInterface 的 load/infer/unload 三个生命周期方法
    推理流程: 暗光增强 -> SAM3 文本提示分割 -> 后处理 -> 抠像输出
    """

    def __init__(self):
        self.device = "cpu"                        # 推理设备，load() 时设置
        self.sam3_model = None                     # SAM3 模型实例
        self.processor = None                      # SAM3 处理器（封装推理逻辑）
        self.dce: ZeroDCE_Enhancer | None = None   # Zero-DCE 增强器（可选）

    def load(self, device: str, **kwargs) -> None:
        """
        加载 SAM3 模型 + Zero-DCE 增强模块
        Args:
            device: 推理设备，如 "cuda:0" 或 "cpu"
            **kwargs: 额外参数
                model_path: SAM3 模型权重路径（默认从配置文件读取）
                dce_weights: Zero-DCE 权重路径
                dce_device: Zero-DCE 运行设备
        """
        self.device = device

        # === 加载 SAM3 分割模型 ===
        model_path = kwargs.get("model_path", SAM3_MODEL_PATH)
        logger.info(f"[SAM3] Loading model from {model_path} on {device}")

        # 延迟导入 sam3 库（避免在未安装 sam3 的环境中导入失败）
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # 构建 SAM3 模型实例
        self.sam3_model = build_sam3_image_model(
            checkpoint_path=model_path,  # 权重文件路径
            load_from_HF=False,          # 不从 HuggingFace 下载，使用本地权重
            device=device,               # 推理设备
        )
        # 创建处理器（封装了图片预处理和文本提示推理的逻辑）
        self.processor = Sam3Processor(self.sam3_model)
        logger.info("[SAM3] Model loaded")

        # === 加载 Zero-DCE 暗光增强模块（可选）===
        dce_path = kwargs.get("dce_weights", SAM3_DCE_WEIGHTS)
        dce_device = kwargs.get("dce_device", SAM3_DCE_DEVICE)
        if os.path.exists(dce_path):
            self.dce = ZeroDCE_Enhancer(dce_path, dce_device)
        else:
            # 权重文件不存在时跳过增强，不影响 SAM3 推理
            logger.warning(f"[DCE] Weights not found: {dce_path}, skipping enhancement")

    def infer(self, task: TaskDescriptor, result_dir: str) -> dict:
        """
        执行 SAM3 推理
        流程: 读图 -> (可选)暗光增强 -> SAM3 分割 -> 后处理 -> 保存结果
        Args:
            task: 任务描述符，包含图片路径和推理参数
            result_dir: 结果文件写入目录
        Returns:
            {"result_path": 结果图片路径, "metadata": {掩码数量, 图片尺寸}}
        """
        # 获取图片路径并校验
        img_path = task.image_path
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 从任务参数中提取推理配置
        params = task.params
        text = params.get("text", "building")      # SAM3 文本提示词（描述要检测的目标）
        conf = params.get("conf", 0.2)              # 置信度阈值，低于此值的掩码被过滤
        skip_dce = params.get("skip_dce", False)    # 是否跳过暗光增强
        dilate_size = params.get("dilate", 0)       # 形态学膨胀核大小（扩大掩码边缘）
        fill_holes = params.get("fill_holes", False)  # 是否填充掩码中的孔洞
        rect_fill = params.get("rect_fill", False)    # 是否在 bbox 区域做矩形修复
        bbox_pad = params.get("bbox_pad", 0)          # bbox 外扩像素数

        # ===== 第 1 步：读取原图 & 暗光增强 =====
        orig_cv2 = cv2.imread(img_path)  # 使用 OpenCV 读取图片（BGR 格式）
        if orig_cv2 is None:
            raise ValueError(f"Failed to read image: {img_path}")

        proc_cv2 = orig_cv2
        # 如果启用了 Zero-DCE 增强且未手动跳过，则进行暗光增强
        if self.dce and not skip_dce:
            proc_cv2 = self.dce.enhance(orig_cv2)
        h, w = proc_cv2.shape[:2]  # 获取图片高度和宽度

        # ===== 第 2 步：SAM3 文本提示分割推理 =====
        # 将 OpenCV BGR 图片转为 PIL RGB 格式（SAM3 模型要求 PIL 输入）
        proc_pil = Image.fromarray(cv2.cvtColor(proc_cv2, cv2.COLOR_BGR2RGB))
        # 设置输入图片，返回推理状态
        inference_state = self.processor.set_image(proc_pil)
        # 使用文本提示词进行分割推理
        results = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text,  # 文本提示词，如 "building"、"road" 等
        )

        # ===== 第 3 步：解析推理结果 =====
        masks = results.get("masks")     # 分割掩码 [N, H, W]，N 为检测到的目标数
        scores = results.get("scores")   # 置信度分数 [N]
        boxes = results.get("boxes")     # 边界框 [N, 4]，格式 (x1, y1, x2, y2)

        # 初始化最终合并掩码（全零）
        final_mask = np.zeros((h, w), dtype=np.uint8)

        if masks is not None:
            # 将 PyTorch 张量转为 NumPy 数组（如果是张量的话）
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()

            # 过滤低置信度的掩码：只保留置信度大于阈值的结果
            keep_idxs = np.where(scores > conf)[0]

            if len(keep_idxs) > 0:
                masks = masks[keep_idxs]    # 按索引筛选掩码
                boxes = boxes[keep_idxs]    # 按索引筛选边界框

                # 遍历每个掩码进行后处理
                for i, m in enumerate(masks):
                    # 如果掩码是 3D 的 [1, H, W]，取第一个通道
                    if m.ndim > 2:
                        m = m[0]
                    # 如果掩码尺寸与原图不匹配，用最近邻插值 resize
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), cv2.INTER_NEAREST)
                    m = m.astype(np.uint8)

                    # --- 形态学膨胀：扩大掩码边缘，弥补分割精度不足 ---
                    if dilate_size > 0:
                        # 确保核大小为奇数（OpenCV 要求）
                        k = dilate_size if dilate_size % 2 == 1 else dilate_size + 1
                        # 使用矩形结构元素进行膨胀操作
                        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=1)

                    # --- 矩形修复：在 bbox 区域内使用形态学闭运算填补间隙 ---
                    if rect_fill and boxes is not None:
                        x1, y1, x2, y2 = map(int, boxes[i])  # 获取当前目标的边界框坐标
                        pad = bbox_pad  # 外扩像素数
                        # 外扩 bbox 并限制在图片范围内
                        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                        # 取 bbox 区域内的掩码
                        roi = m[y1:y2, x1:x2]
                        # 闭运算（先膨胀后腐蚀）：填补掩码中的小间隙和裂缝
                        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE,
                                               cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))
                        m[y1:y2, x1:x2] = roi  # 写回修复后的区域

                    # --- 孔洞填充：找到外轮廓并填充内部孔洞 ---
                    if fill_holes:
                        # 查找外轮廓（RETR_EXTERNAL 只返回最外层轮廓）
                        cts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # 填充轮廓内部区域（cv2.FILLED = -1）
                        cv2.drawContours(m, cts, -1, 1, cv2.FILLED)

                    # 将当前掩码合并到最终掩码（按位或，多个目标区域叠加）
                    final_mask = cv2.bitwise_or(final_mask, m)

        # ===== 第 4 步：生成黑底抠像结果 =====
        # bitwise_and + mask 实现掩码抠像:
        # - 掩码值为 1 的区域: 保留原图像素（建筑/目标区域）
        # - 掩码值为 0 的区域: 变为纯黑色 (0, 0, 0)
        # - final_mask * 255 将二值掩码 (0/1) 转为 OpenCV 要求的 (0/255) 范围
        final = cv2.bitwise_and(proc_cv2, proc_cv2, mask=(final_mask * 255).astype(np.uint8))

        # ===== 第 5 步：保存结果图片 =====
        base_name = os.path.splitext(os.path.basename(img_path))[0]  # 提取不含扩展名的文件名
        result_filename = f"result_{base_name}.jpg"  # 结果文件名加 result_ 前缀
        result_path = os.path.join(result_dir, result_filename)
        cv2.imwrite(result_path, final)  # 保存为 JPEG 格式

        return {
            "result_path": result_path,
            "metadata": {
                "has_detection": int(np.sum(final_mask > 0) > 0),  # 是否检测到目标（0=无，1=有）
                "image_size": f"{w}x{h}",                          # 图片尺寸（宽x高）
            },
        }

    def unload(self) -> None:
        """
        释放模型资源
        将模型引用置为 None 让 Python GC 回收显存，然后手动清空 CUDA 缓存
        """
        self.sam3_model = None
        self.processor = None
        self.dce = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 释放 GPU 显存缓存
        logger.info("[SAM3] Engine unloaded")
