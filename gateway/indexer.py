"""
文件索引器 - ZIP 解析 + 目录扫描 + DreamSim 配对
"""
from __future__ import annotations

import logging
import os
import zipfile
import tempfile
from pathlib import Path

from shared.config import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def scan_images_from_directory(directory: str) -> list[str]:
    """扫描目录下所有图片文件，递归，按路径排序"""
    images = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, f))
    images.sort()
    logger.info(f"Scanned {len(images)} images from {directory}")
    return images


def extract_zip(zip_path: str, extract_to: str) -> str:
    """解压 ZIP 文件，返回解压后的目录路径"""
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid ZIP file: {zip_path}")

    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    logger.info(f"Extracted ZIP {zip_path} to {extract_to}")
    return extract_to


def index_sam3_input(input_path: str, extract_base: str) -> list[str]:
    """
    索引 SAM3 输入：支持目录或 ZIP
    Returns: 排序后的图片路径列表
    """
    if input_path.lower().endswith(".zip"):
        extract_dir = os.path.join(extract_base, "extracted")
        extract_zip(input_path, extract_dir)
        return scan_images_from_directory(extract_dir)
    elif os.path.isdir(input_path):
        return scan_images_from_directory(input_path)
    else:
        raise ValueError(f"Input path must be a directory or ZIP file: {input_path}")


def index_dreamsim_input(input_path: str, extract_base: str) -> list[dict]:
    """
    索引 DreamSim 输入：
    - 目录下需包含 before/ 和 after/ 子目录
    - 同名文件自动配对
    Returns: [{"before": path, "after": path}, ...]
    """
    if input_path.lower().endswith(".zip"):
        extract_dir = os.path.join(extract_base, "extracted")
        extract_zip(input_path, extract_dir)
        base_dir = extract_dir
    elif os.path.isdir(input_path):
        base_dir = input_path
    else:
        raise ValueError(f"Input path must be a directory or ZIP file: {input_path}")

    before_dir = os.path.join(base_dir, "before")
    after_dir = os.path.join(base_dir, "after")

    if not os.path.isdir(before_dir):
        raise ValueError(f"Missing 'before/' subdirectory in {base_dir}")
    if not os.path.isdir(after_dir):
        raise ValueError(f"Missing 'after/' subdirectory in {base_dir}")

    before_files = {}
    for f in os.listdir(before_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            name = os.path.splitext(f)[0]
            before_files[name] = os.path.join(before_dir, f)

    pairs = []
    unmatched = []
    for f in sorted(os.listdir(after_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        name = os.path.splitext(f)[0]
        after_path = os.path.join(after_dir, f)
        if name in before_files:
            pairs.append({
                "before": before_files.pop(name),
                "after": after_path,
            })
        else:
            unmatched.append(f)

    if unmatched:
        logger.warning(f"Unmatched after/ files (no before counterpart): {unmatched}")
    if before_files:
        logger.warning(f"Unmatched before/ files (no after counterpart): {list(before_files.keys())}")

    if not pairs:
        raise ValueError("No matching image pairs found between before/ and after/ directories")

    pairs.sort(key=lambda p: p["after"])
    logger.info(f"Indexed {len(pairs)} image pairs from {base_dir}")
    return pairs
