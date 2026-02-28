"""
SAM3 推理引擎 - 改编自 sam3.py + app.py
封装 Zero-DCE 增强 + SAM3 分割推理
"""
from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from shared.protocol import EngineInterface, TaskDescriptor, register_engine
from shared.config import SAM3_MODEL_PATH, SAM3_DCE_WEIGHTS, SAM3_DCE_DEVICE

logger = logging.getLogger(__name__)


# ============================================================
# Zero-DCE 暗光增强模块 (复用自 sam3.py)
# ============================================================
class C_Net(nn.Module):
    def __init__(self):
        super(C_Net, self).__init__()
        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        return enhance_image


class ZeroDCE_Enhancer:
    def __init__(self, weights_path: str, device: str = "cuda"):
        self.device = device
        self.model = C_Net().to(device)
        if os.path.exists(weights_path):
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=device))
                logger.info(f"[DCE] Enhancer loaded on {device}: {weights_path}")
            except Exception as e:
                logger.warning(f"[DCE] Failed to load weights: {e}")
        self.model.eval()

    def enhance(self, cv2_img):
        img = (np.asarray(cv2_img) / 255.0)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            enhanced_image = self.model(img)
        enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(enhanced_image * 255, 0, 255).astype("uint8")


# ============================================================
# SAM3 推理引擎
# ============================================================
@register_engine("sam3")
class SAM3Engine(EngineInterface):

    def __init__(self):
        self.device = "cpu"
        self.sam3_model = None
        self.processor = None
        self.dce: ZeroDCE_Enhancer | None = None

    def load(self, device: str, **kwargs) -> None:
        """加载 SAM3 模型 + Zero-DCE 增强模块"""
        self.device = device

        # 加载 SAM3
        model_path = kwargs.get("model_path", SAM3_MODEL_PATH)
        logger.info(f"[SAM3] Loading model from {model_path} on {device}")

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.sam3_model = build_sam3_image_model(
            checkpoint_path=model_path,
            load_from_HF=False,
            device=device,
        )
        self.processor = Sam3Processor(self.sam3_model)
        logger.info("[SAM3] Model loaded")

        # 加载 Zero-DCE
        dce_path = kwargs.get("dce_weights", SAM3_DCE_WEIGHTS)
        dce_device = kwargs.get("dce_device", SAM3_DCE_DEVICE)
        if os.path.exists(dce_path):
            self.dce = ZeroDCE_Enhancer(dce_path, dce_device)
        else:
            logger.warning(f"[DCE] Weights not found: {dce_path}, skipping enhancement")

    def infer(self, task: TaskDescriptor, result_dir: str) -> dict:
        """执行 SAM3 推理"""
        img_path = task.image_path
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        params = task.params
        text = params.get("text", "building")
        conf = params.get("conf", 0.2)
        skip_dce = params.get("skip_dce", False)
        dilate_size = params.get("dilate", 0)
        fill_holes = params.get("fill_holes", False)
        rect_fill = params.get("rect_fill", False)
        bbox_pad = params.get("bbox_pad", 0)

        # 1. 读取 & 增强
        orig_cv2 = cv2.imread(img_path)
        if orig_cv2 is None:
            raise ValueError(f"Failed to read image: {img_path}")

        proc_cv2 = orig_cv2
        if self.dce and not skip_dce:
            proc_cv2 = self.dce.enhance(orig_cv2)
        h, w = proc_cv2.shape[:2]

        # 2. SAM3 推理
        proc_pil = Image.fromarray(cv2.cvtColor(proc_cv2, cv2.COLOR_BGR2RGB))
        inference_state = self.processor.set_image(proc_pil)
        results = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text,
        )

        # 3. 解析结果
        masks = results.get("masks")
        scores = results.get("scores")
        boxes = results.get("boxes")

        final_mask = np.zeros((h, w), dtype=np.uint8)

        if masks is not None:
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()

            # 过滤低置信度
            keep_idxs = np.where(scores > conf)[0]

            if len(keep_idxs) > 0:
                masks = masks[keep_idxs]
                boxes = boxes[keep_idxs]

                for i, m in enumerate(masks):
                    if m.ndim > 2:
                        m = m[0]
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), cv2.INTER_NEAREST)
                    m = m.astype(np.uint8)

                    # 形态学处理
                    if dilate_size > 0:
                        k = dilate_size if dilate_size % 2 == 1 else dilate_size + 1
                        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=1)

                    if rect_fill and boxes is not None:
                        x1, y1, x2, y2 = map(int, boxes[i])
                        pad = bbox_pad
                        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                        roi = m[y1:y2, x1:x2]
                        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE,
                                               cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))
                        m[y1:y2, x1:x2] = roi

                    if fill_holes:
                        cts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(m, cts, -1, 1, cv2.FILLED)

                    final_mask = cv2.bitwise_or(final_mask, m)

        # 4. 生成黑底抠像结果
        final = cv2.bitwise_and(proc_cv2, proc_cv2, mask=(final_mask * 255).astype(np.uint8))

        # 5. 保存结果
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        result_filename = f"result_{base_name}.jpg"
        result_path = os.path.join(result_dir, result_filename)
        cv2.imwrite(result_path, final)

        return {
            "result_path": result_path,
            "metadata": {
                "masks_count": int(np.sum(final_mask > 0) > 0),
                "image_size": f"{w}x{h}",
            },
        }

    def unload(self) -> None:
        """释放模型资源"""
        self.sam3_model = None
        self.processor = None
        self.dce = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[SAM3] Engine unloaded")
