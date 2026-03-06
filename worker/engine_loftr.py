"""
LoFTR 配准模块 - 对 before/after 图片对进行几何对齐
改编自生产容器的 register_all.py，封装为可复用的 LoFTRRegistrator 类
调用位置：DINOv3/DreamSim 引擎的 infer() 方法内部，在变化检测前执行
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

from shared.config import LOFTR_WEIGHT_PATH, LOFTR_CONFIG_PATH, LOFTR_MIN_MATCHES

logger = logging.getLogger(__name__)


class RegistrationError(Exception):
    """配准失败异常，引擎捕获后标记任务为 failed"""
    pass


class LoFTRRegistrator:
    """
    LoFTR 特征匹配配准器
    1. 将 before/after 图片 resize 到 1024px 做特征匹配
    2. RANSAC 估计单应性矩阵
    3. 用 H 矩阵将 after 图 warp 到 before 视角
    4. 返回配准后的 numpy 数组（原始分辨率）
    """

    def __init__(self):
        self.matcher = None
        self.device = "cpu"

    def load(self, device: str) -> None:
        """
        加载 LoFTR 模型
        Args:
            device: 推理设备，如 "cuda:0"
        """
        self.device = device
        logger.info(f"[LoFTR] Loading model from {LOFTR_WEIGHT_PATH}")

        from src.loftr import LoFTR
        from src.config.default import get_cfg_defaults
        from src.loftr.utils.cvpr_ds_config import lower_config

        config = get_cfg_defaults()
        config.merge_from_file(LOFTR_CONFIG_PATH)
        lc = lower_config(config)

        self.matcher = LoFTR(config=lc['loftr'])
        state_dict = torch.load(LOFTR_WEIGHT_PATH, map_location='cpu')
        self.matcher.load_state_dict(state_dict['state_dict'])
        self.matcher.eval()
        self.matcher = self.matcher.to(device)

        logger.info(f"[LoFTR] Model loaded on {device}")

    def register(self, img1_path: str, img2_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        对一对图片执行配准
        Args:
            img1_path: before 图片路径（基准图）
            img2_path: after 图片路径（现状图，将被 warp 到 before 视角）
        Returns:
            (registered_img1, registered_img2): 配准后的 BGR numpy 数组，原始分辨率
        Raises:
            RegistrationError: 配准失败（特征点不足、RANSAC 失败等）
        """
        # 读取图片
        img1_color = cv2.imread(img1_path)
        img2_color = cv2.imread(img2_path)

        if img1_color is None:
            raise RegistrationError(f"Failed to read before image: {img1_path}")
        if img2_color is None:
            raise RegistrationError(f"Failed to read after image: {img2_path}")

        # 转灰度用于特征匹配
        img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

        # Resize 到最大 1024px 做配准（减少显存占用和计算量）
        img1_resized = self._resize_image(img1_gray, max_size=1024)
        img2_resized = self._resize_image(img2_gray, max_size=1024)

        # 转为 tensor
        img1_tensor = torch.from_numpy(img1_resized)[None][None].float() / 255.0
        img2_tensor = torch.from_numpy(img2_resized)[None][None].float() / 255.0
        img1_tensor = img1_tensor.to(self.device)
        img2_tensor = img2_tensor.to(self.device)

        # LoFTR 特征匹配
        with torch.no_grad():
            batch = {'image0': img1_tensor, 'image1': img2_tensor}
            self.matcher(batch)

        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()

        if len(mkpts0) < LOFTR_MIN_MATCHES:
            raise RegistrationError(
                f"Not enough feature matches: {len(mkpts0)} < {LOFTR_MIN_MATCHES}"
            )

        # RANSAC 估计单应性矩阵
        H, mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
        if H is None:
            raise RegistrationError("RANSAC homography estimation failed")

        # 将 H 矩阵缩放到原始分辨率
        orig_h1, orig_w1 = img1_color.shape[:2]
        orig_h2, orig_w2 = img2_color.shape[:2]
        resized_h1, resized_w1 = img1_resized.shape[:2]
        resized_h2, resized_w2 = img2_resized.shape[:2]

        # 构造缩放矩阵：从 resize 空间映射回原始空间
        S1 = np.array([
            [orig_w1 / resized_w1, 0, 0],
            [0, orig_h1 / resized_h1, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        S2_inv = np.array([
            [resized_w2 / orig_w2, 0, 0],
            [0, resized_h2 / orig_h2, 0],
            [0, 0, 1],
        ], dtype=np.float64)

        # 原始分辨率下的 H 矩阵
        H_orig = S1 @ H @ S2_inv

        # 用原始分辨率的 H warp after 图
        warped_img2 = cv2.warpPerspective(img2_color, H_orig, (orig_w1, orig_h1))

        logger.info(f"[LoFTR] Registration success: {len(mkpts0)} matches")
        return img1_color, warped_img2

    @staticmethod
    def _resize_image(img: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """将图片长边 resize 到 max_size，保持宽高比"""
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
