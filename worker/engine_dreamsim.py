"""
DreamSim 推理引擎 - 改编自 main_dreamsim.py
本模块封装了 DreamSim 变化检测的完整流程：
  1. 读取前后时相 (T1/T2) 图片对
  2. 滑动窗口裁剪 patch 对，使用 DreamSim 计算感知距离
  3. 将距离值累加到热力图上，可视化变化程度
  4. 阈值过滤 + 轮廓检测，标注变化区域的边界框
  5. 输出热力图叠加版和纯净标注版两种结果
属于 worker 层，实现了 shared/protocol.py 中定义的 EngineInterface 接口
DreamSim 是一种基于深度学习的感知相似度度量方法，
通过对比两个图像 patch 的深度特征计算相似度分数
"""
from __future__ import annotations  # 支持类型注解的前向引用

import logging  # 日志记录
import os       # 文件路径操作
import time     # 计时（保留供未来使用）

import cv2           # OpenCV: 图片读写、热力图可视化、轮廓检测
import numpy as np   # NumPy: 数组操作、热力图计算
import torch         # PyTorch: 深度学习推理框架
from PIL import Image  # Pillow: DreamSim 预处理要求 PIL 格式输入

# 导入引擎接口和注册装饰器
from shared.protocol import EngineInterface, TaskDescriptor, register_engine
# 导入 DreamSim 模型类型配置
from shared.config import DREAMSIM_MODEL_TYPE

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# ============================================================
# DreamSim 推理引擎 - 实现 EngineInterface 接口
# ============================================================
@register_engine("dreamsim")  # 注册到全局引擎注册表，key 为 "dreamsim"
class DreamSimEngine(EngineInterface):
    """
    DreamSim 变化检测推理引擎
    实现了 EngineInterface 的 load/infer/unload 三个生命周期方法
    推理流程: 读取图片对 -> 滑窗裁剪 -> 批量计算感知距离 -> 热力图 -> 标注输出
    适用于遥感图像变化检测场景（如违建检测、土地变化监测等）
    """

    def __init__(self):
        self.device = "cpu"       # 推理设备，load() 时设置
        self.model = None         # DreamSim 模型实例
        self.preprocess = None    # DreamSim 预处理函数（resize + normalize）

    def load(self, device: str, **kwargs) -> None:
        """
        加载 DreamSim 模型
        Args:
            device: 推理设备，如 "cuda:0" 或 "cpu"
            **kwargs: 额外参数
                model_type: DreamSim 模型类型（默认从配置文件读取，通常为 "ensemble"）
        """
        self.device = device
        model_type = kwargs.get("model_type", DREAMSIM_MODEL_TYPE)

        logger.info(f"[DreamSim] Loading model ({model_type}) on {device}")

        # 延迟导入 dreamsim 库（避免在未安装的环境中导入失败）
        from dreamsim import dreamsim

        # 加载预训练的 DreamSim 模型和对应的预处理函数
        # pretrained=True: 使用预训练权重
        # dreamsim_type: 模型变体，"ensemble" 效果最好但稍慢
        self.model, self.preprocess = dreamsim(
            pretrained=True,
            dreamsim_type=model_type,
            device=device,
        )
        self.model.to(device)  # 确保模型在正确的设备上
        self.model.eval()      # 设置为推理模式（关闭 Dropout 等训练行为）
        logger.info("[DreamSim] Model loaded")

    def infer(self, task: TaskDescriptor, result_dir: str) -> dict:
        """
        执行 DreamSim 变化检测
        Args:
            task: 任务描述符，包含图片对路径 (before/after) 和推理参数
            result_dir: 结果文件写入目录
        Returns:
            {"result_path": 热力图路径, "metadata": {纯净版路径, patch参数, 阈值}}
        """
        # 获取图片对路径并校验
        pair = task.image_pair
        if not pair:
            raise ValueError("DreamSim task requires image_pair")

        t1_path = pair["before"]  # 前时相（基准图）路径
        t2_path = pair["after"]   # 后时相（现状图）路径

        # 校验两张图片是否存在
        if not os.path.exists(t1_path):
            raise FileNotFoundError(f"Before image not found: {t1_path}")
        if not os.path.exists(t2_path):
            raise FileNotFoundError(f"After image not found: {t2_path}")

        # 从任务参数中提取推理配置
        params = task.params
        patch_size = params.get("crop", 224)     # 滑窗 patch 大小（默认 224，匹配 DreamSim 输入尺寸）
        step = params.get("step", 0)             # 滑窗步长
        stride = step if step > 0 else patch_size // 2  # 步长为 0 时自动设为 patch_size 的一半（50% 重叠）
        batch_size = params.get("batch", 16)     # GPU 批推理大小
        threshold = params.get("thresh", 0.3)    # 变化检测阈值，高于此值判定为变化区域

        # 生成结果文件路径
        base_name = os.path.splitext(os.path.basename(t2_path))[0]
        output_path = os.path.join(result_dir, f"heatmap_{base_name}.jpg")            # 热力图叠加版
        clean_output_path = os.path.join(result_dir, f"heatmap_{base_name}_clean.jpg")  # 纯净标注版

        # 执行滑窗扫描 + 热力图生成 + 标注框绘制
        self._scan_and_draw(
            t1_path, t2_path, output_path, clean_output_path,
            patch_size, stride, batch_size, threshold,
        )

        return {
            "result_path": output_path,
            "metadata": {
                "clean_result": clean_output_path,  # 纯净版结果路径
                "patch_size": patch_size,            # 使用的 patch 大小
                "stride": stride,                    # 使用的滑窗步长
                "threshold": threshold,              # 使用的检测阈值
            },
        }

    # ============================================================
    # 核心推理逻辑：滑窗扫描 + 热力图生成 + 标注框绘制
    # ============================================================
    def _scan_and_draw(
        self,
        t1_path: str,          # 前时相图片路径
        t2_path: str,          # 后时相图片路径
        output_path: str,      # 热力图叠加版输出路径
        clean_output_path: str,  # 纯净标注版输出路径
        patch_size: int,       # 滑窗 patch 大小
        stride: int,           # 滑窗步长
        batch_size: int,       # 批推理大小
        threshold: float,      # 变化检测阈值
    ):
        """
        滑窗扫描 + 热力图 + 标注图 (复用自 main_dreamsim.py)
        核心算法流程:
        1. 滑动窗口裁剪 patch 对
        2. 批量计算 DreamSim 感知距离
        3. 距离值累加到热力图
        4. 热力图可视化 + 阈值过滤
        5. 轮廓检测 + 画框标注
        """
        # ===== 第 1 步：读取图片 =====
        img1_cv = cv2.imread(t1_path)  # 读取前时相图片（基准图）
        img2_cv = cv2.imread(t2_path)  # 读取后时相图片（现状图）

        if img1_cv is None or img2_cv is None:
            raise ValueError(f"Failed to read images: {t1_path}, {t2_path}")

        # 强制对齐尺寸（以现状图 T2 为准，resize 基准图 T1）
        h, w = img2_cv.shape[:2]
        if img1_cv.shape[:2] != (h, w):
            img1_cv = cv2.resize(img1_cv, (w, h))

        # ===== 第 2 步：准备滑动窗口坐标 =====
        # 按 stride 步长在整张图上滑动，生成所有 patch 的左上角坐标 (x, y)
        coords = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                coords.append((x, y))

        if not coords:
            # 图片尺寸小于 patch_size，无法裁剪 patch，直接保存原图作为结果
            logger.warning(f"Image too small for patching: {w}x{h}, patch={patch_size}")
            cv2.imwrite(output_path, img2_cv)
            cv2.imwrite(clean_output_path, img2_cv)
            return

        total_patches = len(coords)  # 总 patch 数量
        all_distances = []           # 存储所有 patch 对的感知距离

        # ===== 第 3 步：批量推理 - 计算每个 patch 对的 DreamSim 感知距离 =====
        for i in range(0, total_patches, batch_size):
            batch_coords = coords[i:i + batch_size]  # 当前批次的坐标列表
            batch_p1 = []  # 当前批次的 T1 patch 张量列表
            batch_p2 = []  # 当前批次的 T2 patch 张量列表

            for (bx, by) in batch_coords:
                # 从 T1 和 T2 图片上裁剪同位置的 patch
                crop1 = img1_cv[by:by + patch_size, bx:bx + patch_size]
                crop2 = img2_cv[by:by + patch_size, bx:bx + patch_size]

                # 将 OpenCV BGR 格式转为 PIL RGB 格式，然后用 DreamSim 预处理
                # 预处理包括: resize 到模型要求尺寸 + normalize
                p1 = self.preprocess(Image.fromarray(cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB)))
                p2 = self.preprocess(Image.fromarray(cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB)))

                # 处理预处理结果的维度：如果已经是 4D [1,C,H,W]，去掉 batch 维变为 [C,H,W]
                if p1.ndim == 4:
                    p1 = p1.squeeze(0)
                if p2.ndim == 4:
                    p2 = p2.squeeze(0)

                batch_p1.append(p1)
                batch_p2.append(p2)

            # 将列表中的张量堆叠为批次张量 [B, C, H, W] 并移到 GPU
            t_p1 = torch.stack(batch_p1).to(self.device)
            t_p2 = torch.stack(batch_p2).to(self.device)

            with torch.no_grad():  # 推理模式，不计算梯度
                # DreamSim 模型接受两组图片，返回对应的感知距离 [B]
                # 距离越大表示两个 patch 的差异越大（即变化越明显）
                dist_batch = self.model(t_p1, t_p2)
                all_distances.append(dist_batch.cpu())  # 将结果移回 CPU 释放 GPU 显存

        # 拼接所有批次的距离结果为一个一维张量 [total_patches]
        distances = torch.cat(all_distances)

        # ===== 第 4 步：生成热力图数据 =====
        # 热力图累加原理：
        # - 由于滑窗有重叠（stride < patch_size），同一像素可能被多个 patch 覆盖
        # - heatmap 记录所有覆盖到该像素的 patch 距离值之和
        # - count_map 记录该像素被覆盖的次数
        # - heatmap / count_map = 该像素位置的平均变化程度
        heatmap = np.zeros((h, w), dtype=np.float32)      # 距离值累加图
        count_map = np.zeros((h, w), dtype=np.float32)     # 重叠计数图（用于求均值）

        # 获取距离值的范围（用于归一化可视化）
        min_v, max_v = distances.min().item(), distances.max().item()

        for idx, score in enumerate(distances):
            val = score.item()  # 提取标量值
            x, y = coords[idx]  # 获取该 patch 的左上角坐标
            # 将距离值累加到 patch 覆盖的区域（滑窗有重叠时同一像素会被多个 patch 覆盖）
            heatmap[y:y + patch_size, x:x + patch_size] += val
            count_map[y:y + patch_size, x:x + patch_size] += 1

        # 避免除零错误
        count_map[count_map == 0] = 1
        # 求均值：消除重叠区域的累加效应，得到每个像素的平均变化程度
        heatmap_avg = heatmap / count_map

        # ===== 第 5 步：可视化 - 生成热力图 + 标注框 =====
        # 归一化热力图到 [0, 255] 区间用于可视化
        # 使用 max_v（而非 heatmap_avg 的最大值）作为归一化因子，
        # 保持不同图片之间的可比性；最小值 0.1 防止全黑图片导致除零
        norm_factor = max(max_v, 0.1)  # 防止最大值为 0 导致除零
        heatmap_vis = (heatmap_avg / norm_factor * 255).clip(0, 255).astype(np.uint8)
        # 应用 JET 伪彩色映射：蓝色(低变化) -> 绿色 -> 红色(高变化)
        heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

        # 将热力图与原图 (T2 现状图) 加权融合
        # alpha 控制原图的透明度：0.4 表示 40% 原图 + 60% 热力图
        # 选择在 T2（现状图）上叠加热力图，便于直接看到变化区域在当前的位置
        alpha = 0.4  # 原图权重（较低值使热力图更醒目，较高值保留更多原图细节）
        blended_img = cv2.addWeighted(img2_cv, alpha, heatmap_color, 1.0 - alpha, 0)
        result_img = blended_img.copy()       # 热力图叠加版（用于绘制标注框）
        clean_result_img = img2_cv.copy()     # 纯净版（在原图上直接画框，不叠加热力图）

        # --- 阈值过滤：提取高变化区域的轮廓 ---
        # 对热力图进行二值化：高于阈值的区域标记为 255（白），低于的标记为 0（黑）
        _, thresh_img = cv2.threshold(heatmap_vis, int(255 * threshold), 255, cv2.THRESH_BINARY)
        # 查找二值图的外轮廓
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 绘制标注框 ---
        box_count = 0  # 有效变化区域计数
        for cnt in contours:
            area = cv2.contourArea(cnt)  # 轮廓面积（像素²）
            # 过滤过小的区域：面积小于 patch 面积的 3% 视为噪声
            # 例如 patch_size=224 时, min_area = 224*224*0.03 ≈ 1505 像素²
            # 这个阈值过滤了小于约 39×39 像素的碎片区域，避免误检
            min_area = (patch_size * patch_size) * 0.03
            if area > min_area:
                box_count += 1
                # 获取轮廓的外接矩形 (x, y, 宽, 高)
                x, y, bw, bh = cv2.boundingRect(cnt)
                # 计算该区域内的平均变化分数（标注在框上方）
                label = f"{heatmap_avg[y:y + bh, x:x + bw].mean():.2f}"

                # --- 热力图叠加版标注 ---
                # 白色外框（宽 4px）+ 红色内框（宽 2px）实现双线效果，提高辨识度
                cv2.rectangle(result_img, (x, y), (x + bw, y + bh), (255, 255, 255), 4)
                cv2.rectangle(result_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                # 在框上方绘制红色背景的分数标签
                cv2.rectangle(result_img, (x, y - 25), (x + 80, y), (0, 0, 255), -1)  # 红色填充矩形
                cv2.putText(result_img, label, (x + 5, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 白色文字

                # --- 纯净版标注（只在原图上画框，不叠加热力图）---
                cv2.rectangle(clean_result_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                cv2.rectangle(clean_result_img, (x, y - 25), (x + 80, y), (0, 0, 255), -1)
                cv2.putText(clean_result_img, label, (x + 5, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 保存结果图片
        cv2.imwrite(output_path, result_img)           # 热力图叠加版
        cv2.imwrite(clean_output_path, clean_result_img)  # 纯净标注版

        logger.info(f"[DreamSim] Scan done: {total_patches} patches, {box_count} regions, "
                     f"scores=[{min_v:.4f}, {max_v:.4f}]")

    def unload(self) -> None:
        """
        释放模型资源
        将模型引用置为 None 让 Python GC 回收显存，然后手动清空 CUDA 缓存
        """
        self.model = None
        self.preprocess = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 释放 GPU 显存缓存
        logger.info("[DreamSim] Engine unloaded")
