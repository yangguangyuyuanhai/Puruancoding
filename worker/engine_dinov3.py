"""
DINOv3 推理引擎 - 改编自 diffsim/main_dreamsim.py (生产容器版本)
基于 DINOv3 Transformer 的变化检测引擎，核心优势：抗阴影干扰
流程：LoFTR 配准 → LAB 色彩对齐 → DINOv3 特征提取 → 自适应阈值 → 标注输出
实现了 shared/protocol.py 中定义的 EngineInterface 接口
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from shared.protocol import EngineInterface, TaskDescriptor, register_engine
from shared.config import DINOV3_PEAK_RATIO, DINOV3_RES, DINOV3_DINO_CONF
from worker.engine_loftr import LoFTRRegistrator, RegistrationError

logger = logging.getLogger(__name__)

DINO_PATCH_SIZE = 16


def color_transfer_lab(src: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    LAB 色彩空间光照迁移
    将 target 图片的色彩分布调整为与 src 一致，消除光照差异
    仅在 mask 有效区域内执行，避免黑底区域影响统计
    """
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    tar_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean, src_std = cv2.meanStdDev(src_lab, mask=mask)
    tar_mean, tar_std = cv2.meanStdDev(tar_lab, mask=mask)

    src_mean, src_std = src_mean.flatten(), src_std.flatten()
    tar_mean, tar_std = tar_mean.flatten(), tar_std.flatten()
    tar_std[tar_std == 0] = 1e-5

    res_lab = (tar_lab - tar_mean) * (src_std / tar_std) + src_mean
    res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
    res_bgr = cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

    target_matched = target.copy()
    target_matched[mask == 255] = res_bgr[mask == 255]
    return target_matched


@register_engine("dinov3")
class DINOv3Engine(EngineInterface):
    """
    DINOv3 变化检测推理引擎
    使用 DINOv3 Vision Transformer 提取 patch 级别的语义特征，
    通过余弦距离检测两张图片之间的结构性变化
    """

    def __init__(self):
        self.device = "cpu"
        self.dino_model = None
        self.dino_preprocess = None
        self.loftr = None

    def load(self, device: str, **kwargs) -> None:
        """加载 DINOv3 模型和 LoFTR 配准模型"""
        self.device = device
        model_path = kwargs.get("model_path", "/app")

        logger.info(f"[DINOv3] Loading model from {model_path} on {device}")

        from transformers import AutoModel, AutoImageProcessor

        self.dino_preprocess = AutoImageProcessor.from_pretrained(model_path)
        self.dino_model = AutoModel.from_pretrained(model_path).to(device).eval()

        logger.info("[DINOv3] DINOv3 model loaded")

        # 初始化 LoFTR 配准器
        self.loftr = LoFTRRegistrator()
        self.loftr.load(device)

        logger.info("[DINOv3] Engine fully loaded")

    def infer(self, task: TaskDescriptor, result_dir: str) -> dict:
        """
        执行 DINOv3 变化检测
        流程：配准 → 色彩对齐 → 特征提取 → 自适应阈值 → 标注输出
        """
        pair = task.image_pair
        if not pair:
            raise ValueError("DINOv3 task requires image_pair")

        t1_path = pair["before"]
        t2_path = pair["after"]

        if not os.path.exists(t1_path):
            raise FileNotFoundError(f"Before image not found: {t1_path}")
        if not os.path.exists(t2_path):
            raise FileNotFoundError(f"After image not found: {t2_path}")

        # 从 task.params 获取参数（重推时覆盖），fallback 到 job 级别 params
        params = task.params
        peak_ratio = params.get("peak_ratio", DINOV3_PEAK_RATIO)
        process_size = params.get("res", DINOV3_RES)
        dino_conf = params.get("dino_conf", DINOV3_DINO_CONF)

        # 生成结果文件路径
        base_name = os.path.splitext(os.path.basename(t2_path))[0]
        output_path = os.path.join(result_dir, f"heatmap_{base_name}.jpg")
        clean_output_path = os.path.join(result_dir, f"heatmap_{base_name}_clean.jpg")

        # Step 1: LoFTR 配准（失败则 raise，由 base_worker 标记 failed）
        img1_cv, img2_cv = self.loftr.register(t1_path, t2_path)

        # Step 2: 执行 DINOv3 推理
        self._scan_and_draw(
            img1_cv, img2_cv, output_path, clean_output_path,
            process_size, dino_conf, peak_ratio,
        )

        return {
            "result_path": output_path,
            "metadata": {
                "clean_result": clean_output_path,
                "peak_ratio": peak_ratio,
                "res": process_size,
                "dino_conf": dino_conf,
            },
        }

    def _scan_and_draw(
        self,
        img1_cv: np.ndarray,
        img2_cv: np.ndarray,
        output_path: str,
        clean_output_path: str,
        process_size: int,
        dino_conf: float,
        peak_ratio: float,
    ):
        """DINOv3 全图特征提取 + 自适应阈值 + 标注框绘制"""
        orig_h, orig_w = img2_cv.shape[:2]
        if img1_cv.shape[:2] != (orig_h, orig_w):
            img1_cv = cv2.resize(img1_cv, (orig_w, orig_h))

        # 输入域强同步：生成有效区域掩码
        gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
        _, mask1 = cv2.threshold(gray1, 5, 255, cv2.THRESH_BINARY)
        _, mask2 = cv2.threshold(gray2, 5, 255, cv2.THRESH_BINARY)
        valid_mask = cv2.bitwise_and(mask1, mask2)

        img1_cv[valid_mask == 0] = 0
        img2_cv[valid_mask == 0] = 0

        # LAB 色彩空间光照迁移
        img2_cv = color_transfer_lab(img1_cv, img2_cv, valid_mask)

        # DINOv3 全局特征提取
        proc_h = (process_size // DINO_PATCH_SIZE) * DINO_PATCH_SIZE
        proc_w = (process_size // DINO_PATCH_SIZE) * DINO_PATCH_SIZE

        img1_pil = Image.fromarray(cv2.cvtColor(img1_cv, cv2.COLOR_BGR2RGB))
        img2_pil = Image.fromarray(cv2.cvtColor(img2_cv, cv2.COLOR_BGR2RGB))

        inputs1 = self.dino_preprocess(
            images=img1_pil, return_tensors="pt",
            size={"height": proc_h, "width": proc_w},
        ).to(self.device)
        inputs2 = self.dino_preprocess(
            images=img2_pil, return_tensors="pt",
            size={"height": proc_h, "width": proc_w},
        ).to(self.device)

        with torch.no_grad():
            out1 = self.dino_model(**inputs1).last_hidden_state
            out2 = self.dino_model(**inputs2).last_hidden_state

        grid_h = proc_h // DINO_PATCH_SIZE
        grid_w = proc_w // DINO_PATCH_SIZE
        num_spatial_patches = grid_h * grid_w

        patch_feats1 = out1[:, -num_spatial_patches:, :]
        patch_feats2 = out2[:, -num_spatial_patches:, :]

        patch_feats1 = F.normalize(patch_feats1, p=2, dim=-1)
        patch_feats2 = F.normalize(patch_feats2, p=2, dim=-1)
        cos_sim = (patch_feats1 * patch_feats2).sum(dim=-1)
        distances = (1.0 - cos_sim).squeeze(0).cpu().numpy()

        # 空间复原与自适应局部阈值
        dist_map_2d = distances.reshape((grid_h, grid_w))
        heatmap_full = cv2.resize(dist_map_2d, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        blur_k = max(11, int(orig_w * 0.05) // 2 * 2 + 1)
        heatmap_smooth = cv2.GaussianBlur(heatmap_full, (blur_k, blur_k), 0)
        heatmap_smooth[valid_mask == 0] = 0

        # 自适应阈值计算
        valid_scores = heatmap_smooth[valid_mask > 0]

        if len(valid_scores) > 0:
            mu = valid_scores.mean()
            sigma = valid_scores.std()
            max_score = valid_scores.max()

            stat_thresh = mu + 3 * sigma
            peak_thresh = max_score * peak_ratio
            adaptive_thresh = max(stat_thresh, peak_thresh, dino_conf)
        else:
            adaptive_thresh = dino_conf
            max_score = 1.0

        if max_score < dino_conf:
            heatmap_smooth = np.zeros_like(heatmap_smooth)

        # 二值化
        thresh_img = np.zeros_like(heatmap_smooth, dtype=np.uint8)
        thresh_img[heatmap_smooth >= adaptive_thresh] = 255

        # 可视化
        norm_factor = max(max_score, 0.05)
        heatmap_vis = np.clip((heatmap_smooth / norm_factor) * 255, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

        draw_img_cv = img2_cv.copy()
        alpha = 0.4
        blended_img = cv2.addWeighted(draw_img_cv, alpha, heatmap_color, 1.0 - alpha, 0)
        blended_img[valid_mask == 0] = draw_img_cv[valid_mask == 0]

        result_img = blended_img.copy()
        clean_result_img = draw_img_cv.copy()

        # 轮廓检测与标注框
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        box_count = 0
        min_area_px = (orig_w * orig_h) * 0.003

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area_px:
                box_count += 1
                x, y, bw, bh = cv2.boundingRect(cnt)
                region_score = heatmap_smooth[y:y + bh, x:x + bw].max()
                label = f"Peak:{region_score:.2f}"

                # 热力图版标注
                cv2.rectangle(result_img, (x, y), (x + bw, y + bh), (255, 255, 255), 4)
                cv2.rectangle(result_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.rectangle(result_img, (x, y - 25), (x + 120, y), (0, 0, 255), -1)
                cv2.putText(result_img, label, (x + 5, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 纯净版标注
                cv2.rectangle(clean_result_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                cv2.rectangle(clean_result_img, (x, y - 25), (x + 120, y), (0, 0, 255), -1)
                cv2.putText(clean_result_img, label, (x + 5, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imwrite(output_path, result_img)
        cv2.imwrite(clean_output_path, clean_result_img)

        logger.info(f"[DINOv3] Scan done: {box_count} regions, "
                    f"adaptive_thresh={adaptive_thresh:.4f}, max_score={max_score:.4f}")

    def unload(self) -> None:
        """释放模型资源"""
        self.dino_model = None
        self.dino_preprocess = None
        self.loftr = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[DINOv3] Engine unloaded")
