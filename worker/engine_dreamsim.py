"""
DreamSim 推理引擎 - 改编自 main_dreamsim.py
封装 DreamSim 滑窗比对 + 热力图 + 标注图生成
"""
from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from shared.protocol import EngineInterface, TaskDescriptor, register_engine
from shared.config import DREAMSIM_MODEL_TYPE

logger = logging.getLogger(__name__)


@register_engine("dreamsim")
class DreamSimEngine(EngineInterface):

    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.preprocess = None

    def load(self, device: str, **kwargs) -> None:
        """加载 DreamSim 模型"""
        self.device = device
        model_type = kwargs.get("model_type", DREAMSIM_MODEL_TYPE)

        logger.info(f"[DreamSim] Loading model ({model_type}) on {device}")

        from dreamsim import dreamsim

        self.model, self.preprocess = dreamsim(
            pretrained=True,
            dreamsim_type=model_type,
            device=device,
        )
        self.model.to(device)
        self.model.eval()
        logger.info("[DreamSim] Model loaded")

    def infer(self, task: TaskDescriptor, result_dir: str) -> dict:
        """执行 DreamSim 变化检测"""
        pair = task.image_pair
        if not pair:
            raise ValueError("DreamSim task requires image_pair")

        t1_path = pair["before"]
        t2_path = pair["after"]

        if not os.path.exists(t1_path):
            raise FileNotFoundError(f"Before image not found: {t1_path}")
        if not os.path.exists(t2_path):
            raise FileNotFoundError(f"After image not found: {t2_path}")

        params = task.params
        patch_size = params.get("crop", 224)
        step = params.get("step", 0)
        stride = step if step > 0 else patch_size // 2
        batch_size = params.get("batch", 16)
        threshold = params.get("thresh", 0.3)

        # 生成结果
        base_name = os.path.splitext(os.path.basename(t2_path))[0]
        output_path = os.path.join(result_dir, f"heatmap_{base_name}.jpg")
        clean_output_path = os.path.join(result_dir, f"heatmap_{base_name}_clean.jpg")

        self._scan_and_draw(
            t1_path, t2_path, output_path, clean_output_path,
            patch_size, stride, batch_size, threshold,
        )

        return {
            "result_path": output_path,
            "metadata": {
                "clean_result": clean_output_path,
                "patch_size": patch_size,
                "stride": stride,
                "threshold": threshold,
            },
        }

    def _scan_and_draw(
        self,
        t1_path: str,
        t2_path: str,
        output_path: str,
        clean_output_path: str,
        patch_size: int,
        stride: int,
        batch_size: int,
        threshold: float,
    ):
        """滑窗扫描 + 热力图 + 标注图 (复用自 main_dreamsim.py)"""
        # 1. 读取图片
        img1_cv = cv2.imread(t1_path)
        img2_cv = cv2.imread(t2_path)

        if img1_cv is None or img2_cv is None:
            raise ValueError(f"Failed to read images: {t1_path}, {t2_path}")

        # 强制对齐尺寸 (以现状图 T2 为准)
        h, w = img2_cv.shape[:2]
        if img1_cv.shape[:2] != (h, w):
            img1_cv = cv2.resize(img1_cv, (w, h))

        # 2. 准备滑动窗口坐标
        coords = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                coords.append((x, y))

        if not coords:
            logger.warning(f"Image too small for patching: {w}x{h}, patch={patch_size}")
            # 保存原图作为结果
            cv2.imwrite(output_path, img2_cv)
            cv2.imwrite(clean_output_path, img2_cv)
            return

        total_patches = len(coords)
        all_distances = []

        # 3. 批量推理
        for i in range(0, total_patches, batch_size):
            batch_coords = coords[i:i + batch_size]
            batch_p1 = []
            batch_p2 = []

            for (bx, by) in batch_coords:
                crop1 = img1_cv[by:by + patch_size, bx:bx + patch_size]
                crop2 = img2_cv[by:by + patch_size, bx:bx + patch_size]

                p1 = self.preprocess(Image.fromarray(cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB)))
                p2 = self.preprocess(Image.fromarray(cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB)))

                if p1.ndim == 4:
                    p1 = p1.squeeze(0)
                if p2.ndim == 4:
                    p2 = p2.squeeze(0)

                batch_p1.append(p1)
                batch_p2.append(p2)

            t_p1 = torch.stack(batch_p1).to(self.device)
            t_p2 = torch.stack(batch_p2).to(self.device)

            with torch.no_grad():
                dist_batch = self.model(t_p1, t_p2)
                all_distances.append(dist_batch.cpu())

        distances = torch.cat(all_distances)

        # 4. 生成热力图数据
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        min_v, max_v = distances.min().item(), distances.max().item()

        for idx, score in enumerate(distances):
            val = score.item()
            x, y = coords[idx]
            heatmap[y:y + patch_size, x:x + patch_size] += val
            count_map[y:y + patch_size, x:x + patch_size] += 1

        count_map[count_map == 0] = 1
        heatmap_avg = heatmap / count_map

        # 5. 可视化
        norm_factor = max(max_v, 0.1)
        heatmap_vis = (heatmap_avg / norm_factor * 255).clip(0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

        alpha = 0.4
        blended_img = cv2.addWeighted(img2_cv, alpha, heatmap_color, 1.0 - alpha, 0)
        result_img = blended_img.copy()
        clean_result_img = img2_cv.copy()

        # 阈值过滤与画框
        _, thresh_img = cv2.threshold(heatmap_vis, int(255 * threshold), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        box_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            min_area = (patch_size * patch_size) * 0.03
            if area > min_area:
                box_count += 1
                x, y, bw, bh = cv2.boundingRect(cnt)
                label = f"{heatmap_avg[y:y + bh, x:x + bw].mean():.2f}"

                # 热力图版
                cv2.rectangle(result_img, (x, y), (x + bw, y + bh), (255, 255, 255), 4)
                cv2.rectangle(result_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.rectangle(result_img, (x, y - 25), (x + 80, y), (0, 0, 255), -1)
                cv2.putText(result_img, label, (x + 5, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 纯净版
                cv2.rectangle(clean_result_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                cv2.rectangle(clean_result_img, (x, y - 25), (x + 80, y), (0, 0, 255), -1)
                cv2.putText(clean_result_img, label, (x + 5, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 保存结果
        cv2.imwrite(output_path, result_img)
        cv2.imwrite(clean_output_path, clean_result_img)

        logger.info(f"[DreamSim] Scan done: {total_patches} patches, {box_count} regions, "
                     f"scores=[{min_v:.4f}, {max_v:.4f}]")

    def unload(self) -> None:
        """释放模型资源"""
        self.model = None
        self.preprocess = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[DreamSim] Engine unloaded")
